import os
import argparse
import cv2
import easyocr
from pathlib import Path
from ultralytics import YOLO
import time
from collections import Counter
import numpy as np
import boto3
import imageio

from utils import get_logger

logger = get_logger()


class KVSStreamReader:
    def __init__(self, stream_name, region_name="us-east-1"):
        self.stream_name = stream_name
        self.region_name = region_name
        self.kvs_client = boto3.client("kinesisvideo", region_name=self.region_name)
        
        # Get the endpoint for the stream
        endpoint = self.kvs_client.get_data_endpoint(
            StreamName=self.stream_name,
            APIName="GET_MEDIA"
        )['DataEndpoint']
        
        self.kvs_media_client = boto3.client("kinesis-video-media", endpoint_url=endpoint, region_name=self.region_name)

    def get_frames(self):
        """
        Uses imageio and FFmpeg to robustly parse the Kinesis video stream.
        """
        # Start reading the media from the stream, starting from the latest fragment
        response = self.kvs_media_client.get_media(
            StreamName=self.stream_name,
            StartSelector={'StartSelectorType': 'NOW'}
        )
        
        # The payload is a streaming body, which imageio can read directly
        stream_payload = response['Payload']

        try:
            reader = imageio.get_reader(stream_payload, format='FFMPEG')
            
            # Iterate through the frames that the reader finds
            for frame in reader:
                # imageio reads frames in RGB format by default.
                # OpenCV expects BGR, so we need to convert the color space.
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                yield frame_bgr

        except Exception as e:
            logger.error(f"Error reading stream with imageio: {e}")

        logger.info("Kinesis video stream has ended.")


class VideoInferenceProcessor:
    """Processes race video footage to track racers, extract bib numbers using OCR, and determine finish times.
    This class uses YOLO models for object detection and tracking, and EasyOCR for reading bib numbers.
    It maintains a history of tracked racers, detects when they cross a virtual finish line, and produces live and final leaderboards.
    Args:
        model_path (str or Path): Path to the YOLO model weights.
        video_path (str or Path): Path to the input race video file.
        target_fps (int, optional): Target frames per second for processing. Defaults to 1.
        confidence_threshold (float, optional): Minimum confidence for detections. Defaults to 0.
        finish_line_fraction (float, optional): Fraction of frame width to set the finish line. Defaults to 0.85.
    Attributes:
        model_path (Path): Path to YOLO model weights.
        video_path (Path): Path to input video file.
        target_fps (int): Target FPS for processing.
        confidence_threshold (float): Detection confidence threshold.
        track_history (dict): History of tracked racers keyed by tracker ID.
        tracker_model (YOLO): YOLO model instance for tracking persons.
        predictor_model (YOLO): YOLO model instance for stateless predictions.
        ocr_reader (easyocr.Reader): EasyOCR reader for bib extraction.
        cap (cv2.VideoCapture): OpenCV video capture object.
        original_fps (float): Original FPS of the video.
        frame_width (int): Width of video frames.
        frame_height (int): Height of video frames.
        frame_skip (int): Number of frames to skip for target FPS.
        finish_line_x (int): X-coordinate of the virtual finish line.
        inference_interval (int): Interval for running inference.
        last_annotated_frame (np.ndarray): Last annotated frame for display.
    Methods:
        preprocess_for_easyocr(image_crop):
            Preprocesses an image crop for OCR (grayscale, denoise, threshold).
        extract_bib_with_easyocr(image_crop):
            Extracts bib number and confidence from an image crop using EasyOCR.
        crop_bib_from_prediction(image, bbox, padding=15):
            Crops the bib region from an image using bounding box coordinates.
        check_finish_line_crossings(tracked_persons):
            Checks if tracked racers have crossed the virtual finish line.
        update_history_hybrid(tracked_persons, all_detections):
            Updates racer history with OCR reads and finish line status.
        draw_hybrid_predictions(image, tracked_persons, all_detections, scale=1.0):
            Draws bounding boxes and labels for racers and bibs on the frame.
        determine_final_bibs():
            Aggregates OCR reads to determine the most likely bib for each racer.
        print_live_leaderboard():
            Prints the current leaderboard to the terminal, sorted by finish time.
        process_video(output_path=None, display=True):
            Processes the video, tracks racers, extracts bibs, and displays or saves annotated frames.
            Prints live and final leaderboards."""

    def __init__(
        self,
        model_path: str | Path,
        video_path: str | Path,
        stream_name: str = None,
        target_fps: int = 1,
        confidence_threshold: float = 0,
        finish_line_fraction: float = 0.85,
    ):
        """
        Initializes the VideoInferenceProcessor for live bib tracking.

        Args:
            model_path (str | Path): Path to YOLO model weights.
            video_path (str | Path): Path to input video file.
            target_fps (int, optional): Target FPS for processing. Defaults to 1.
            confidence_threshold (float, optional): YOLO detection confidence threshold. Defaults to 0.
            finish_line_fraction (float, optional): Fraction of frame width for finish line. Defaults to 0.85.
        """
        self.model_path = Path(model_path)
        self.video_path = Path(video_path)
        self.target_fps = target_fps
        self.confidence_threshold = confidence_threshold

        # This will store history keyed by the PERSON's tracker ID
        self.track_history = {}

        logger.info("Loading models...")
        # This model instance will be used ONLY for tracking and will become stateful
        self.tracker_model = YOLO(str(self.model_path))
        # This model instance will be used ONLY for prediction and will remain stateless
        self.predictor_model = YOLO(str(self.model_path))
        logger.info("Models loaded successfully!")

        # Initialize EasyOCR reader
        logger.info("Initializing EasyOCR reader...")
        self.ocr_reader = easyocr.Reader(["en"], gpu=True)
        logger.info("EasyOCR reader initialized!")

        # Video capture and properties
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_skip = max(1, int(self.original_fps / self.target_fps))
        self.finish_line_x = int(self.frame_width * finish_line_fraction)

        self.inference_interval = 1
        self.last_annotated_frame = None

    def preprocess_for_easyocr(self, image_crop: np.ndarray) -> np.ndarray:
        """
        Preprocesses an image crop for use with EasyOCR by converting it to grayscale,
        applying denoising, and adaptive thresholding.

        Args:
            image_crop (np.ndarray): The input image crop, either in color (BGR) or grayscale.

        Returns:
            np.ndarray: The processed binary image suitable for OCR.
        """
        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        return cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    def extract_bib_with_easyocr(
        self, image_crop: np.ndarray
    ) -> tuple[str | None, float | None]:
        """
        Extracts the bib number and its confidence score from an image crop using EasyOCR.

        Args:
            image_crop (np.ndarray): The preprocessed image crop containing the bib.

        Returns:
            tuple[str | None, float | None]: The detected bib number (as a string) and its confidence score.
                Returns (None, None) if no bib is detected or an error occurs.
        """

        try:
            result = self.ocr_reader.readtext(image_crop, allowlist="0123456789")
            if result:
                _, text, confidence = max(result, key=lambda x: x[2])
                return text, confidence
            return None, None
        except Exception:
            return None, None

    def crop_bib_from_prediction(
        self,
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
        padding: int = 15,
    ) -> np.ndarray:
        """
        Crops a region from the input image based on the bounding box coordinates,
        with optional padding applied.

        Args:
            image (numpy.ndarray): The input image from which the region will be cropped.
            bbox (tuple[float, float, float, float]): The bounding box coordinates
            (x1, y1, x2, y2) specifying the region to crop.
            padding (int, optional): The number of pixels to add as padding around
            the bounding box. Defaults to 15.

        Returns:
            numpy.ndarray: The cropped region of the image.
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(image.shape[1], x2 + padding), min(image.shape[0], y2 + padding)
        return image[y1:y2, x1:x2]

    def check_finish_line_crossings(self, tracked_persons: YOLO) -> None:
        """
        Checks if any tracked racers have crossed the virtual finish line.

        This method updates the track history for each racer by recording the finish time
        when a racer crosses the finish line from left to right. It also prints the live
        leaderboard whenever a racer finishes.

        Args:
            tracked_persons (YOLO): The YOLO tracking results containing bounding boxes
                and tracker IDs for detected persons.
        """

        if tracked_persons is None or tracked_persons.boxes.id is None:
            return

        for person_box in tracked_persons.boxes:
            person_id = int(person_box.id[0])
            # Use the center of the bounding box as the racer's position
            x1, _, x2, _ = person_box.xyxy[0]
            current_x_center = (x1 + x2) / 2

            history = self.track_history.get(person_id)
            # Proceed only if the racer is already being tracked and hasn't finished yet
            if history and history["finish_time_ms"] is None:
                last_x_center = history.get("last_x_center")

                # Check if the racer crossed the line from left to right
                if (
                    last_x_center is not None
                    and last_x_center < self.finish_line_x
                    and current_x_center >= self.finish_line_x
                ):
                    # Racer crossed the line! Record the video timestamp.
                    finish_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                    history["finish_time_ms"] = finish_time
                    logger.info(
                        f"Racer ID {person_id} finished at {finish_time / 1000:.2f}s"
                    )
                    self.print_live_leaderboard()
                # Update the last known position for the next frame
                history["last_x_center"] = current_x_center

    def update_history_hybrid(
        self, tracked_persons: YOLO, all_detections: YOLO
    ) -> None:
        """
        Updates the track history for racers by associating detected bibs with tracked persons.

        This method uses YOLO tracking results to identify persons and YOLO detection results
        to identify bibs. It associates bibs with persons based on bounding box overlap and
        updates the track history with OCR reads and finish line status.

        Args:
            tracked_persons (YOLO): YOLO tracking results containing bounding boxes and tracker IDs for detected persons.
            all_detections (YOLO): YOLO detection results containing bounding boxes and classes for detected objects.

        Returns:
            None
        """
        if (
            tracked_persons is None
            or tracked_persons.boxes.id is None
            or all_detections is None
        ):
            return

        bib_boxes = [box for box in all_detections.boxes if int(box.cls) == 1]
        if not bib_boxes:
            return

        for person_box in tracked_persons.boxes:
            person_id = int(person_box.id[0])
            px1, py1, px2, py2 = person_box.xyxy[0]

            for bib_box in bib_boxes:
                bx1, by1, bx2, by2 = bib_box.xyxy[0]
                if px1 < (bx1 + bx2) / 2 < px2 and py1 < (by1 + by2) / 2 < py2:
                    if person_id not in self.track_history:
                        # Initialize with new fields for timing
                        self.track_history[person_id] = {
                            "ocr_reads": [],
                            "last_x_center": None,
                            "finish_time_ms": None,
                        }
                    bib_crop = self.crop_bib_from_prediction(
                        all_detections.orig_img, bib_box.xyxy[0]
                    )
                    if bib_crop.size > 0:
                        bib_number, ocr_conf = self.extract_bib_with_easyocr(
                            self.preprocess_for_easyocr(bib_crop)
                        )
                        if bib_number and ocr_conf:
                            self.track_history[person_id]["ocr_reads"].append(
                                (bib_number, ocr_conf, float(bib_box.conf))
                            )
                    break

    def draw_hybrid_predictions(
        self,
        image: np.ndarray,
        tracked_persons: "Detections",
        all_detections: "Detections",
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Draws hybrid predictions on the input image, including finish line, detected boxes, and tracked persons with their IDs and best OCR bib reads.

        Args:
            image (np.ndarray): The input image on which to draw annotations.
            tracked_persons (Detections): Object containing tracked person detections, including IDs and bounding boxes.
            all_detections (Detections): Object containing all detection boxes and classes.
            scale (float, optional): Scale factor to adjust bounding box coordinates. Defaults to 1.0.

        Returns:
            np.ndarray: The annotated image with drawn predictions.
        """

        annotated_image = image.copy()
        cv2.line(
            annotated_image,
            (self.finish_line_x, 0),
            (self.finish_line_x, self.frame_height),
            (0, 255, 255),
            3,
        )

        if all_detections:
            for box in all_detections.boxes:
                if int(box.cls) == 1:
                    x1, y1, x2, y2 = [int(c / scale) for c in box.xyxy[0]]
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if tracked_persons and tracked_persons.boxes.id is not None:
            for box in tracked_persons.boxes:
                person_id = int(box.id[0])
                x1, y1, x2, y2 = [int(c / scale) for c in box.xyxy[0]]
                current_best_read = ""
                if (
                    person_id in self.track_history
                    and self.track_history[person_id]["ocr_reads"]
                ):
                    reads = [r[0] for r in self.track_history[person_id]["ocr_reads"]]
                    if reads:
                        current_best_read = Counter(reads).most_common(1)[0][0]
                label_text = f"Racer ID {person_id} | Bib: {current_best_read}"
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    annotated_image,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2,
                )
        return annotated_image

    def determine_final_bibs(self):
        """
        Determines the most likely bib number for each racer based on OCR and YOLO confidence scores.

        This method processes the aggregated `track_history` and filters OCR reads for reliability.
        It calculates a score for each bib number by multiplying OCR and YOLO confidences, then
        selects the bib number with the highest total score for each tracker.

        Returns:
            dict: A dictionary mapping tracker IDs to a dictionary containing:
                - "final_bib" (str): The most likely bib number.
                - "score" (float): The total confidence score for the selected bib number.
        """
        final_results = {}
        for tracker_id, data in self.track_history.items():
            ocr_reads = data.get("ocr_reads", [])
            if not ocr_reads:
                continue

            # Filter out unreliable reads (e.g., wrong length, low confidence)
            filtered_reads = [
                r for r in ocr_reads if 2 <= len(r[0]) <= 5 and r[1] > 0.4
            ]
            if not filtered_reads:
                continue
            scores = {}

            for bib_num, ocr_conf, yolo_conf in filtered_reads:
                score = ocr_conf * yolo_conf
                scores[bib_num] = scores.get(bib_num, 0) + score

            # Find the bib number with the highest total score
            if scores:
                most_likely_bib = max(scores, key=scores.get)
                final_results[tracker_id] = {
                    "final_bib": most_likely_bib,
                    "score": scores[most_likely_bib],
                }

        return final_results

    def print_live_leaderboard(self):
        """
        Clears the terminal and prints the current state of the leaderboard.
        """

        os.system("cls" if os.name == "nt" else "clear")

        current_bib_results = self.determine_final_bibs()

        leaderboard = []
        for tracker_id, history_data in self.track_history.items():
            # Check if this racer has a recorded finish time
            if history_data and history_data.get("finish_time_ms") is not None:
                bib_result = current_bib_results.get(tracker_id)
                # If the bib number is determined, use it. Otherwise, show "No Bib".
                bib_number = bib_result["final_bib"] if bib_result else "No Bib"

                leaderboard.append(
                    {
                        "id": tracker_id,
                        "bib": bib_number,
                        "time_ms": history_data["finish_time_ms"],
                    }
                )

        leaderboard.sort(key=lambda x: x["time_ms"])

        logger.info(
            f"--- 🏁 Live Race Leaderboard (Updated: {time.strftime('%I:%M:%S %p')}) 🏁 ---"
        )
        if leaderboard:
            for i, entry in enumerate(leaderboard):
                total_seconds = entry["time_ms"] / 1000
                minutes = int(total_seconds // 60)
                seconds = int(total_seconds % 60)
                milliseconds = int((total_seconds - int(total_seconds)) * 100)
                time_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:02d}"

                logger.info(
                    f"  {i + 1}. Racer ID: {entry['id']:<4} | Bib: {entry['bib']:<8} | Time: {time_str}"
                )
        else:
            logger.info("  Waiting for the first racer to finish...")

        logger.info("----------------------------------------------------------")
        logger.info(
            "\n(Processing video... Press Ctrl+C to stop and show final results)"
        )

    def process_video_live_stream(
        self, output_path: str | Path = None, display: bool = True
    ):
        """
        Processes a live video stream from Kinesis Video Streams to track racers,
        extract bib numbers using OCR, and determine finish times.
        Optionally displays the annotated video in real-time and/or saves it to an output file.
        Args:
            output_path (str | Path, optional): Path to save the annotated output video. Defaults to None.
            display (bool, optional): Whether to display the annotated video in real-time. Defaults to True.

        """
        #TODO: consolidate redundant logic with process_video
        kvs_reader = KVSStreamReader(self.stream_name)
        frame_generator = kvs_reader.get_frames()
        
        frame_count = 0
        start_time = time.time()
    
        try:
            for frame in frame_generator:
                if frame is None:
                    continue

                orig_h, orig_w = frame.shape[:2]
                proc_w = orig_w
                scale = proc_w / orig_w
                proc_h = int(orig_h * scale)

                processing_frame = frame #cv2.resize(frame, (proc_w, proc_h))
 
                if frame_count % self.inference_interval == 0:
                    # Track ONLY persons using the dedicated tracker_model
                    # person_results = None
                    person_results = self.tracker_model.track(
                        processing_frame,
                        persist=True,
                        tracker="config/custom_tracker.yaml",
                        classes=[0],
                        verbose=False,
                    )
                    tracked_persons = person_results[0] if person_results else None

                    self.check_finish_line_crossings(tracked_persons)

                    # Predict ALL objects using the separate, stateless predictor_model
                    # all_detections_results = None 
                    all_detections_results = self.predictor_model.predict(
                        processing_frame, conf=self.confidence_threshold, verbose=False
                    )
                    all_detections = (
                        all_detections_results[0] if all_detections_results else None
                    )

                    # Update history using the new hybrid function
                    self.update_history_hybrid(tracked_persons, all_detections)

                    # Draw predictions using the new hybrid function
                    annotated_frame = self.draw_hybrid_predictions(
                        frame, tracked_persons, all_detections, scale=scale
                    )

                    self.last_annotated_frame = annotated_frame
                    display_frame = annotated_frame

                    elapsed_seconds = time.time() - start_time                    
                    # --- TIMER BLOCK ---
                    minutes = int(elapsed_seconds // 60)
                    seconds = int(elapsed_seconds % 60)
                    timer_text = f"Live Timer: {minutes:02d}:{seconds:02d}"
                    
                    # Get text size to create a background rectangle for readability
                    (text_width, text_height), _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    text_x = self.frame_width - text_width - 20
                    text_y = text_height + 20
                    
                    # Draw black background rectangle
                    cv2.rectangle(display_frame, (text_x - 10, text_y - text_height - 10), (text_x + text_width + 10, text_y + 10), (0,0,0), -1)
                    # Draw timer text in white
                    cv2.putText(display_frame, timer_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    video_seconds = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    
                    if elapsed_seconds > 0:
                        lag_seconds = elapsed_seconds - video_seconds
                        processing_speed_factor = video_seconds / elapsed_seconds
                        # Create the text string to display
                        lag_text = f"Lag: {lag_seconds:.1f}s (Speed: {processing_speed_factor:.1f}x)"
                    else:
                        lag_text = "Lag: Calculating..."

                    # 3. Add the lag text below the live timer
                    (lag_text_width, _), _ = cv2.getTextSize(lag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    lag_text_x = self.frame_width - lag_text_width - 20
                    lag_text_y = text_y + text_height + 5 # Position it below the timer text
                    
                    # Draw black background rectangle and the lag text in yellow
                    cv2.rectangle(annotated_frame, (lag_text_x - 10, lag_text_y - text_height), (lag_text_x + lag_text_width + 10, lag_text_y + 10), (0,0,0), -1)
                    cv2.putText(annotated_frame, lag_text, (lag_text_x, lag_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                else:
                    if self.last_annotated_frame is not None:
                        display_frame = self.last_annotated_frame

                if display:
                    cv2.imshow("Live Bib Tracking", display_frame)
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_count += 1
        finally:
            # First, determine the final bib numbers for everyone
            final_bib_results = self.determine_final_bibs()

            leaderboard = []
            for tracker_id, result_data in final_bib_results.items():
                history_data = self.track_history.get(tracker_id)
                if history_data and history_data.get("finish_time_ms") is not None:
                    leaderboard.append(
                        {
                            "id": tracker_id,
                            "bib": result_data["final_bib"],
                            "time_ms": history_data["finish_time_ms"],
                        }
                    )

            # Sort the leaderboard by finish time (fastest first)
            leaderboard.sort(key=lambda x: x["time_ms"])

            logger.info("\n--- 🏁 Official Race Leaderboard 🏁 ---")
            if leaderboard:
                for i, entry in enumerate(leaderboard):
                    # Format time as MM:SS.ms
                    total_seconds = entry["time_ms"] / 1000
                    minutes = int(total_seconds // 60)
                    seconds = int(total_seconds % 60)
                    milliseconds = int((total_seconds - int(total_seconds)) * 100)
                    time_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:02d}"

                    logger.info(
                        f"  {i + 1}. Racer ID: {entry['id']:<4} | Bib: {entry['bib']:<6} | Time: {time_str}"
                    )
            else:
                logger.info("  No racers finished the race.")
            logger.info("----------------------------------------------------------")
            self.cap.release()

            cv2.destroyAllWindows()

    def process_video(
        self, output_path: str | Path = None, display: bool = True
    ) -> None:
        """
        Processes the race video to track racers, extract bib numbers using OCR, and determine finish times.
        Optionally displays the annotated video in real-time and/or saves it to an output file.

        Args:
            output_path (str | Path, optional): Path to save the annotated output video. Defaults to None.
            display (bool, optional): Whether to display the annotated video in real-time. Defaults to True.

        Returns:
            None
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None
        if output_path:
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                self.target_fps,
                (self.frame_width, self.frame_height),
            )

        frame_count = 0
        start_time = time.time()
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue

                orig_h, orig_w = frame.shape[:2]
                proc_w = orig_w
                scale = proc_w / orig_w
                proc_h = int(orig_h * scale)

                processing_frame = frame #cv2.resize(frame, (proc_w, proc_h))
 
                if frame_count % self.inference_interval == 0:
                    # Track ONLY persons using the dedicated tracker_model
                    # person_results = None
                    person_results = self.tracker_model.track(
                        processing_frame,
                        persist=True,
                        tracker="config/custom_tracker.yaml",
                        classes=[0],
                        verbose=False,
                    )
                    tracked_persons = person_results[0] if person_results else None

                    self.check_finish_line_crossings(tracked_persons)

                    # Predict ALL objects using the separate, stateless predictor_model
                    # all_detections_results = None 
                    all_detections_results = self.predictor_model.predict(
                        processing_frame, conf=self.confidence_threshold, verbose=False
                    )
                    all_detections = (
                        all_detections_results[0] if all_detections_results else None
                    )

                    # Update history using the new hybrid function
                    self.update_history_hybrid(tracked_persons, all_detections)

                    # Draw predictions using the new hybrid function
                    annotated_frame = self.draw_hybrid_predictions(
                        frame, tracked_persons, all_detections, scale=scale
                    )

                    self.last_annotated_frame = annotated_frame
                    display_frame = annotated_frame

                    elapsed_seconds = time.time() - start_time                    
                    # --- TIMER BLOCK ---
                    minutes = int(elapsed_seconds // 60)
                    seconds = int(elapsed_seconds % 60)
                    timer_text = f"Live Timer: {minutes:02d}:{seconds:02d}"
                    
                    # Get text size to create a background rectangle for readability
                    (text_width, text_height), _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    text_x = self.frame_width - text_width - 20
                    text_y = text_height + 20
                    
                    # Draw black background rectangle
                    cv2.rectangle(display_frame, (text_x - 10, text_y - text_height - 10), (text_x + text_width + 10, text_y + 10), (0,0,0), -1)
                    # Draw timer text in white
                    cv2.putText(display_frame, timer_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    video_seconds = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    
                    if elapsed_seconds > 0:
                        lag_seconds = elapsed_seconds - video_seconds
                        processing_speed_factor = video_seconds / elapsed_seconds
                        # Create the text string to display
                        lag_text = f"Lag: {lag_seconds:.1f}s (Speed: {processing_speed_factor:.1f}x)"
                    else:
                        lag_text = "Lag: Calculating..."

                    # 3. Add the lag text below the live timer
                    (lag_text_width, _), _ = cv2.getTextSize(lag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    lag_text_x = self.frame_width - lag_text_width - 20
                    lag_text_y = text_y + text_height + 5 # Position it below the timer text
                    
                    # Draw black background rectangle and the lag text in yellow
                    cv2.rectangle(annotated_frame, (lag_text_x - 10, lag_text_y - text_height), (lag_text_x + lag_text_width + 10, lag_text_y + 10), (0,0,0), -1)
                    cv2.putText(annotated_frame, lag_text, (lag_text_x, lag_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                else:
                    if self.last_annotated_frame is not None:
                        display_frame = self.last_annotated_frame

                if display:
                    cv2.imshow("Live Bib Tracking", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                # if out:
                #     out.write(display_frame)
                frame_count += 1

        finally:
            # First, determine the final bib numbers for everyone
            final_bib_results = self.determine_final_bibs()

            leaderboard = []
            for tracker_id, result_data in final_bib_results.items():
                history_data = self.track_history.get(tracker_id)
                if history_data and history_data.get("finish_time_ms") is not None:
                    leaderboard.append(
                        {
                            "id": tracker_id,
                            "bib": result_data["final_bib"],
                            "time_ms": history_data["finish_time_ms"],
                        }
                    )

            # Sort the leaderboard by finish time (fastest first)
            leaderboard.sort(key=lambda x: x["time_ms"])

            logger.info("\n--- 🏁 Official Race Leaderboard 🏁 ---")
            if leaderboard:
                for i, entry in enumerate(leaderboard):
                    # Format time as MM:SS.ms
                    total_seconds = entry["time_ms"] / 1000
                    minutes = int(total_seconds // 60)
                    seconds = int(total_seconds % 60)
                    milliseconds = int((total_seconds - int(total_seconds)) * 100)
                    time_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:02d}"

                    logger.info(
                        f"  {i + 1}. Racer ID: {entry['id']:<4} | Bib: {entry['bib']:<6} | Time: {time_str}"
                    )
            else:
                logger.info("  No racers finished the race.")
            logger.info("----------------------------------------------------------")
            self.cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Video Inference for Live Bib Tracking"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/raw/2024_race.MOV",
        help="Path to input video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/train2/weights/last.pt",
        help="Path to trained YOLO model",
    )
    parser.add_argument(
        "--fps", type=int, default=8, help="Target processing frame rate"
    )
    parser.add_argument(
        "--conf", type=float, default=0.3, help="YOLO confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save output video (optional)"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Disable real-time video display"
    )
    parser.add_argument(
        "--stream-name",
        type=str,
        default=None,
        help="Kinesis Video Stream name for live processing",
    )

    args = parser.parse_args()

    # Validate input files
    if not Path(args.video).exists():
        logger.info(f"Error: Video file not found: {args.video}")
        return

    if not Path(args.model).exists():
        logger.info(f"Error: Model file not found: {args.model}")
        return

    # Create processor and run inference
    try:
        processor = VideoInferenceProcessor(
            model_path=args.model,
            video_path=args.video,
            stream_name=args.stream_name,
            target_fps=args.fps,
            confidence_threshold=args.conf,
        )

        if args.stream_name:
            logger.info(f"Processing live stream: {args.stream_name}")
            processor.process_video_live_stream(
                output_path=args.output, display=not args.no_display
            )
        else:
            logger.info(f"Processing video file: {args.video}")
            processor.process_video(output_path=args.output, display=not args.no_display)

    except Exception as e:
        logger.info(f"Error during processing: {e}")
        return


if __name__ == "__main__":
    main()
