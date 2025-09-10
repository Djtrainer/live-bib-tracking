import os
import argparse
import cv2
import dotenv
import easyocr
from pathlib import Path
from ultralytics import YOLO
import time
from collections import Counter
import numpy as np
import requests
from typing import Generator
import asyncio
from fastapi import FastAPI, Response  # Add Request
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, Request # Add Request
# FastAPI imports
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn

from image_processor.utils import get_logger

logger = get_logger()
dotenv.load_dotenv()
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    logger.info("Application startup: FastAPI server ready")
    
    # Initialize processor using environment variables
    try:
        # Get configuration from environment variables
        video_path_str = os.getenv('VIDEO_PATH', 'data/raw/2024_race.MOV')
        model_path_str = os.getenv('MODEL_PATH', '/app/runs/detect/train2/weights/last.pt')
        target_fps = int(os.getenv('TARGET_FPS', '8'))
        confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.3'))
        
        # Check if video and model files exist
        from pathlib import Path
        video_path = Path(video_path_str)
        model_path = Path(model_path_str)
        
        if video_path.exists() and model_path.exists():
            logger.info(f"Initializing processor with video: {video_path_str}, model: {model_path_str}")
            logger.info(f"Configuration - FPS: {target_fps}, Confidence: {confidence_threshold}")
            
            processor = VideoInferenceProcessor(
                model_path=model_path_str,
                video_path=video_path_str,
                target_fps=target_fps,
                confidence_threshold=confidence_threshold,
            )
            app_state["processor"] = processor
            logger.info("✅ Video processor initialized successfully during startup!")
        else:
            logger.warning(f"Video or model file not found. Video: {video_path.exists()}, Model: {model_path.exists()}")
            logger.warning(f"Video path: {video_path_str}")
            logger.warning(f"Model path: {model_path_str}")
            app_state["processor"] = None
    except Exception as e:
        logger.error(f"Failed to initialize processor during startup: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        app_state["processor"] = None
    
    yield
    
    # Code to run on shutdown
    logger.info("Application shutdown: Cleaning up resources.")
    if app_state.get("processor"):
        try:
            app_state["processor"].cap.release()
        except Exception as e:
            logger.warning(f"Error releasing video capture: {e}")

# Attach the lifespan manager to the FastAPI app
app = FastAPI(lifespan=lifespan, title="Live Bib Tracking Video Stream")
processor = None


class VideoInferenceProcessor:
    """Processes race video footage to track racers, extract bib numbers using OCR, and determine finish times.
    This class uses YOLO models for object detection and tracking, and EasyOCR for reading bib numbers.
    It maintains a history of tracked racers, detects when they cross a virtual finish line, and produces live and final leaderboards."""

    def __init__(
        self,
        model_path: str | Path,
        video_path: str | Path,
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

        # Track when processing started for wall time calculations
        self.processing_start_time = None

    def preprocess_for_easyocr(self, image_crop: np.ndarray) -> np.ndarray:
        """
        Preprocesses an image crop for use with EasyOCR by converting it to grayscale,
        applying denoising, and adaptive thresholding.
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
        """
        try:
            # Validate input image crop
            if image_crop is None or image_crop.size == 0:
                logger.warning("Empty or None image crop provided to OCR")
                return None, None

            if len(image_crop.shape) < 2:
                logger.warning(f"Invalid image crop shape for OCR: {image_crop.shape}")
                return None, None

            # Perform OCR with error handling
            result = self.ocr_reader.readtext(image_crop, allowlist="0123456789")

            if result and len(result) > 0:
                # Find the result with highest confidence
                _, text, confidence = max(
                    result, key=lambda x: x[2] if len(x) >= 3 else 0
                )

                # Validate the extracted text
                if text and isinstance(text, str) and text.strip():
                    return text.strip(), float(confidence)
                else:
                    return None, None
            else:
                return None, None

        except Exception as e:
            logger.error(f"Error in OCR processing: {e}")
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
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(image.shape[1], x2 + padding), min(image.shape[0], y2 + padding)
        return image[y1:y2, x1:x2]

    def check_finish_line_crossings(self, tracked_persons: YOLO) -> None:
        """
        Checks if any tracked racers have crossed the virtual finish line.
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
                    # Racer crossed the line! Record both video timestamp and wall-clock time.
                    finish_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                    finish_wall_time = time.time()  # Capture current wall-clock time
                    history["finish_time_ms"] = finish_time
                    history["finish_wall_time"] = finish_wall_time
                    logger.info(
                        f"Racer ID {person_id} finished at {finish_time / 1000:.2f}s"
                    )

                    # Send result to local server
                    final_bib_results = self.determine_final_bibs()
                    bib_result = final_bib_results.get(person_id)

                    if bib_result:
                        payload = {
                            "bibNumber": bib_result["final_bib"],
                            "finishTime": finish_time,
                        }
                        try:
                            requests.post(
                                "http://localhost:8000/api/results",
                                json=payload,
                                timeout=2,
                            )
                            logger.info(
                                f"Sent finisher data to local UI: Bib #{payload['bibNumber']}"
                            )
                        except requests.exceptions.ConnectionError:
                            logger.warning(
                                "Could not connect to local UI server. Is it running?"
                            )

                    self.print_live_leaderboard()

                # Update the last known position for the next frame
                history["last_x_center"] = current_x_center

    def update_history_hybrid(
        self, tracked_persons: YOLO, all_detections: YOLO
    ) -> None:
        """
        Updates the track history for racers by associating detected bibs with tracked persons.
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
                            "finish_wall_time": None,
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
        tracked_persons,
        all_detections,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Draws hybrid predictions on the input image, including finish line, detected boxes, and tracked persons.
        """
        try:
            # Validate input image
            if image is None or image.size == 0:
                logger.warning("Invalid image provided to draw_hybrid_predictions")
                return np.zeros((480, 640, 3), dtype=np.uint8)

            annotated_image = image.copy()

            # Draw finish line
            try:
                cv2.line(
                    annotated_image,
                    (self.finish_line_x, 0),
                    (self.finish_line_x, self.frame_height),
                    (0, 255, 255),
                    3,
                )
            except Exception as e:
                logger.error(f"Error drawing finish line: {e}")

            # Draw bib detection boxes
            if all_detections and hasattr(all_detections, "boxes"):
                try:
                    for box in all_detections.boxes:
                        if (
                            hasattr(box, "cls")
                            and hasattr(box, "xyxy")
                            and int(box.cls) == 1
                        ):
                            try:
                                x1, y1, x2, y2 = [int(c / scale) for c in box.xyxy[0]]
                                # Validate coordinates
                                if (
                                    0 <= x1 < x2 <= annotated_image.shape[1]
                                    and 0 <= y1 < y2 <= annotated_image.shape[0]
                                ):
                                    cv2.rectangle(
                                        annotated_image,
                                        (x1, y1),
                                        (x2, y2),
                                        (0, 0, 255),
                                        2,
                                    )
                            except Exception as e:
                                logger.warning(f"Error drawing bib box: {e}")
                                continue
                except Exception as e:
                    logger.error(f"Error processing bib detections: {e}")

            # Draw tracked persons
            if (
                tracked_persons
                and hasattr(tracked_persons, "boxes")
                and tracked_persons.boxes.id is not None
            ):
                try:
                    for box in tracked_persons.boxes:
                        try:
                            if not hasattr(box, "id") or not hasattr(box, "xyxy"):
                                continue

                            person_id = int(box.id[0])
                            x1, y1, x2, y2 = [int(c / scale) for c in box.xyxy[0]]

                            # Validate coordinates
                            if not (
                                0 <= x1 < x2 <= annotated_image.shape[1]
                                and 0 <= y1 < y2 <= annotated_image.shape[0]
                            ):
                                continue

                            current_best_read = ""
                            try:
                                if (
                                    person_id in self.track_history
                                    and self.track_history[person_id]["ocr_reads"]
                                ):
                                    reads = [
                                        r[0]
                                        for r in self.track_history[person_id][
                                            "ocr_reads"
                                        ]
                                    ]
                                    if reads:
                                        current_best_read = Counter(reads).most_common(
                                            1
                                        )[0][0]
                            except Exception as e:
                                logger.warning(
                                    f"Error getting best OCR read for person {person_id}: {e}"
                                )

                            label_text = (
                                f"Racer ID {person_id} | Bib: {current_best_read}"
                            )

                            # Draw rectangle
                            cv2.rectangle(
                                annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2
                            )

                            # Draw text with background for better visibility
                            try:
                                text_size = cv2.getTextSize(
                                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
                                )[0]
                                text_x = max(0, x1)
                                text_y = max(text_size[1] + 10, y1 - 10)

                                # Draw text background
                                cv2.rectangle(
                                    annotated_image,
                                    (text_x - 5, text_y - text_size[1] - 5),
                                    (text_x + text_size[0] + 5, text_y + 5),
                                    (0, 0, 0),
                                    -1,
                                )

                                # Draw text
                                cv2.putText(
                                    annotated_image,
                                    label_text,
                                    (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9,
                                    (255, 0, 0),
                                    2,
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Error drawing text for person {person_id}: {e}"
                                )

                        except Exception as e:
                            logger.warning(f"Error processing tracked person box: {e}")
                            continue

                except Exception as e:
                    logger.error(f"Error processing tracked persons: {e}")

            return annotated_image

        except Exception as e:
            logger.error(f"Critical error in draw_hybrid_predictions: {e}")
            # Return original image as fallback
            return (
                image if image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            )

    def determine_final_bibs(self):
        """
        Determines the most likely bib number for each racer based on OCR and YOLO confidence scores.
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

                # Calculate elapsed time since script started using actual wall-clock time
                if (
                    self.processing_start_time is not None
                    and history_data.get("finish_wall_time") is not None
                ):
                    # Calculate actual elapsed time from script start to when racer finished
                    elapsed_seconds = (
                        history_data["finish_wall_time"] - self.processing_start_time
                    )

                    # Format as elapsed time (MM:SS)
                    minutes = int(elapsed_seconds // 60)
                    seconds = int(elapsed_seconds % 60)
                    wall_time_str = f"{minutes:02d}:{seconds:02d}"
                else:
                    wall_time_str = "N/A"

                leaderboard.append(
                    {
                        "id": tracker_id,
                        "bib": bib_number,
                        "time_ms": history_data["finish_time_ms"],
                        "wall_time": wall_time_str,
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
                video_time_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:02d}"

                logger.info(
                    f"  {i + 1}. Racer ID: {entry['id']:<4} | Bib: {entry['bib']:<8} | Video Time: {video_time_str} | Elapsed Time: {entry['wall_time']}"
                )
        else:
            logger.info("  Waiting for the first racer to finish...")

        logger.info("----------------------------------------------------------")
        logger.info(
            "\n(Processing video... Press Ctrl+C to stop and show final results)"
        )

    def draw_ui_overlays(
        self, frame: np.ndarray, start_time: float, cap: cv2.VideoCapture
    ) -> np.ndarray:
        """
        Draws all UI overlays on the frame including wall clock, live timer, and video timing information.
        """
        # --- WALL CLOCK (Top-left corner) ---
        current_time = time.strftime("%I:%M:%S %p")
        wall_clock_text = f"Wall Clock: {current_time}"

        # Get text size for wall clock
        (clock_width, clock_height), _ = cv2.getTextSize(
            wall_clock_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )

        # Position in top-left corner
        clock_x = 20
        clock_y = clock_height + 20

        # Draw semi-transparent black background for wall clock
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (clock_x - 10, clock_y - clock_height - 10),
            (clock_x + clock_width + 10, clock_y + 10),
            (0, 0, 0),
            -1,
        )
        # Blend with original frame for semi-transparency
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw wall clock text in white
        cv2.putText(
            frame,
            wall_clock_text,
            (clock_x, clock_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # --- LIVE TIMER (Top-right corner) ---
        elapsed_seconds = time.time() - start_time
        video_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        minutes = int(elapsed_seconds // 60)
        seconds = int(elapsed_seconds % 60)
        timer_text = f"Live Timer: {minutes:02d}:{seconds:02d}"

        # Get text size to create a background rectangle for readability
        (text_width, text_height), _ = cv2.getTextSize(
            timer_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
        )
        text_x = self.frame_width - text_width - 20
        text_y = text_height + 20

        # Draw black background rectangle
        cv2.rectangle(
            frame,
            (text_x - 10, text_y - text_height - 10),
            (text_x + text_width + 10, text_y + 10),
            (0, 0, 0),
            -1,
        )
        # Draw timer text in white
        cv2.putText(
            frame,
            timer_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # --- VIDEO TIME / LAG INFO (Below live timer) ---
        if elapsed_seconds > 0:
            lag_seconds = elapsed_seconds - video_seconds
            processing_speed_factor = video_seconds / elapsed_seconds
            lag_text = (
                f"Lag: {lag_seconds:.1f}s (Speed: {processing_speed_factor:.1f}x)"
            )
        else:
            lag_text = "Lag: Calculating..."

        # Add the lag text below the live timer
        (lag_text_width, _), _ = cv2.getTextSize(
            lag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        lag_text_x = self.frame_width - lag_text_width - 20
        lag_text_y = text_y + text_height + 5

        # Draw black background rectangle and the lag text in yellow
        cv2.rectangle(
            frame,
            (lag_text_x - 10, lag_text_y - text_height),
            (lag_text_x + lag_text_width + 10, lag_text_y + 10),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            lag_text,
            (lag_text_x, lag_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        return frame

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_count: int,
        start_time: float,
        cap: cv2.VideoCapture,
    ) -> np.ndarray:
        """
        Processes a single frame for object detection, tracking, and annotation.
        """
        try:
            # Validate input frame
            if frame is None or frame.size == 0:
                logger.warning(f"Invalid frame at count {frame_count}")
                return (
                    frame
                    if frame is not None
                    else np.zeros((480, 640, 3), dtype=np.uint8)
                )

            if len(frame.shape) != 3 or frame.shape[2] != 3:
                logger.warning(
                    f"Unexpected frame shape at count {frame_count}: {frame.shape}"
                )
                return frame

            orig_h, orig_w = frame.shape[:2]
            proc_w = orig_w
            scale = proc_w / orig_w
            proc_h = int(orig_h * scale)

            processing_frame = frame

            if frame_count % self.inference_interval == 0:
                tracked_persons = None
                all_detections = None

                try:
                    # Track ONLY persons using the dedicated tracker_model
                    person_results = self.tracker_model.track(
                        processing_frame,
                        persist=True,
                        tracker="config/custom_tracker.yaml",
                        classes=[0],
                        verbose=False,
                    )
                    tracked_persons = person_results[0] if person_results else None
                except Exception as e:
                    logger.error(
                        f"Error in person tracking at frame {frame_count}: {e}"
                    )
                    tracked_persons = None

                try:
                    self.check_finish_line_crossings(tracked_persons)
                except Exception as e:
                    logger.error(
                        f"Error checking finish line crossings at frame {frame_count}: {e}"
                    )

                try:
                    # Predict ALL objects using the separate, stateless predictor_model
                    all_detections_results = self.predictor_model.predict(
                        processing_frame, conf=self.confidence_threshold, verbose=False
                    )
                    all_detections = (
                        all_detections_results[0] if all_detections_results else None
                    )
                except Exception as e:
                    logger.error(
                        f"Error in object detection at frame {frame_count}: {e}"
                    )
                    all_detections = None

                try:
                    # Update history using the new hybrid function
                    self.update_history_hybrid(tracked_persons, all_detections)
                except Exception as e:
                    logger.error(f"Error updating history at frame {frame_count}: {e}")

                try:
                    # Draw predictions using the new hybrid function
                    annotated_frame = self.draw_hybrid_predictions(
                        frame, tracked_persons, all_detections, scale=scale
                    )
                    self.last_annotated_frame = annotated_frame
                    display_frame = annotated_frame
                except Exception as e:
                    logger.error(
                        f"Error drawing predictions at frame {frame_count}: {e}"
                    )
                    display_frame = frame
                    self.last_annotated_frame = frame

                try:
                    # Add all UI overlays to the frame
                    display_frame = self.draw_ui_overlays(
                        display_frame, start_time, cap
                    )
                except Exception as e:
                    logger.error(
                        f"Error drawing UI overlays at frame {frame_count}: {e}"
                    )
                    # Continue with frame without overlays
            else:
                if self.last_annotated_frame is not None:
                    display_frame = self.last_annotated_frame
                else:
                    display_frame = frame

            return display_frame

        except Exception as e:
            logger.error(
                f"Critical error in _process_frame at frame {frame_count}: {e}"
            )
            # Return original frame as fallback
            return (
                frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            )

    def process_video_generator(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields processed video frames for streaming.
        This replaces the original process_video method for web streaming.
        """
        frame_count = 0
        start_time = time.time()
        self.processing_start_time = start_time

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue

                # Process the frame
                display_frame = self._process_frame(
                    frame, frame_count, start_time, self.cap
                )

                # Yield the processed frame
                yield display_frame
                frame_count += 1

        except Exception as e:
            logger.error(f"Error in video generator: {e}")
        finally:
            self.cap.release()
            logger.info("Video processing completed")


# FastAPI endpoints
@app.get("/")
async def root():
    """Root endpoint that serves the viewer HTML page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Live Bib Tracking Video Stream</title>
        <style>
            body {
                background-color: #1a1a1a;
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
                font-family: Arial, sans-serif;
                color: white;
            }
            h1 {
                color: #ffffff;
                text-align: center;
                margin-bottom: 20px;
            }
            #video-container {
                border: 2px solid #333;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            }
            #video-stream {
                display: block;
                max-width: 100%;
                height: auto;
            }
            .info {
                margin-top: 20px;
                text-align: center;
                color: #ccc;
            }
        </style>
    </head>
    <body>
        <h1>🏃‍♂️ Live Bib Tracking Video Stream 🏁</h1>
        <div id="video-container">
            <img id="video-stream" src="/video_feed" alt="Live Video Stream">
        </div>
        <div class="info">
            <p>Live video processing with real-time bib number detection and race tracking</p>
            <p>Yellow line indicates the finish line | Blue boxes show detected racers | Red boxes show detected bibs</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/video_feed")
async def video_feed(request: Request):
    """Endpoint that streams the processed video as MJPEG."""
    try:
        # Get the processor from our state dictionary
        processor = app_state.get("processor")
        
        if processor is None:
            error_msg = "Video processor not initialized. Please ensure the server was started with proper command line arguments."
            logger.error(error_msg)
            return Response(error_msg, status_code=500)
        
        async def generate_frames():
            """Generator function that yields MJPEG frames asynchronously."""
            frame_count = 0
            error_count = 0
            max_errors = 10
            
            try:
                logger.info("Starting video frame generation...")
                
                # Use the processor's generator method
                for frame in processor.process_video_generator():
                    try:
                        # Validate frame
                        if frame is None or frame.size == 0:
                            logger.warning(f"Invalid frame at count {frame_count}")
                            continue
                            
                        # Encode frame as JPEG
                        ret, buffer = cv2.imencode(
                            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                        )
                        
                        if not ret or buffer is None:
                            logger.warning(f"Failed to encode frame {frame_count}")
                            error_count += 1
                            if error_count > max_errors:
                                logger.error("Too many encoding errors, stopping stream")
                                break
                            continue

                        frame_bytes = buffer.tobytes()
                        
                        if len(frame_bytes) == 0:
                            logger.warning(f"Empty frame bytes at count {frame_count}")
                            continue

                        # Yield the frame in MJPEG format
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                        )

                        frame_count += 1
                        
                        # Reset error count on successful frame
                        if error_count > 0:
                            error_count = 0

                        # Allow the server to handle other tasks
                        await asyncio.sleep(0.01)  # Small delay to prevent overwhelming

                    except Exception as frame_error:
                        error_count += 1
                        logger.error(f"Error processing frame {frame_count}: {frame_error}")
                        
                        if error_count > max_errors:
                            logger.error("Too many frame processing errors, stopping stream")
                            break
                        continue

                logger.info(f"Video stream ended. Processed {frame_count} frames with {error_count} errors.")

            except Exception as generator_error:
                logger.error(f"Critical error in video generator: {generator_error}")
                # Yield an error frame
                try:
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(error_frame, "Video Processing Error", (50, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(error_frame, str(generator_error)[:50], (50, 280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    ret, buffer = cv2.imencode(".jpg", error_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                        )
                except Exception as error_frame_error:
                    logger.error(f"Failed to generate error frame: {error_frame_error}")

        return StreamingResponse(
            generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
        
    except Exception as endpoint_error:
        error_msg = f"Error in video_feed endpoint: {str(endpoint_error)}"
        logger.error(error_msg)
        return Response(error_msg, status_code=500)


def main():
    """Main function that initializes the processor and starts the FastAPI server."""
    global processor

    parser = argparse.ArgumentParser(
        description="Live Bib Tracking Video Stream Server"
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
        default="/app/models/last.pt",
        help="Path to trained YOLO model",
    )
    parser.add_argument(
        "--fps", type=int, default=8, help="Target processing frame rate"
    )
    parser.add_argument(
        "--conf", type=float, default=0.3, help="YOLO confidence threshold"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="Port to bind the server to"
    )

    try:
        args = parser.parse_args()
    except SystemExit as e:
        logger.error(f"Argument parsing failed: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error parsing arguments: {e}")
        return

    # Validate input parameters
    try:
        if args.fps <= 0:
            logger.error(f"Invalid FPS value: {args.fps}. Must be greater than 0.")
            return
            
        if not (0.0 <= args.conf <= 1.0):
            logger.error(f"Invalid confidence threshold: {args.conf}. Must be between 0.0 and 1.0.")
            return
            
        if not (1 <= args.port <= 65535):
            logger.error(f"Invalid port number: {args.port}. Must be between 1 and 65535.")
            return
            
    except Exception as e:
        logger.error(f"Error validating parameters: {e}")
        return

    # Validate input files exist
    try:
        video_path = Path(args.video)
        model_path = Path(args.model)
        
        if not video_path.exists():
            logger.error(f"Video file not found: {args.video}")
            logger.info("Please check the path and ensure the file exists.")
            return

        if not model_path.exists():
            logger.error(f"Model file not found: {args.model}")
            logger.info("Please check the path and ensure the model file exists.")
            return
            
        # Check file permissions
        if not video_path.is_file():
            logger.error(f"Video path is not a file: {args.video}")
            return
            
        if not model_path.is_file():
            logger.error(f"Model path is not a file: {args.model}")
            return
            
        # Check file sizes (basic validation)
        video_size = video_path.stat().st_size
        model_size = model_path.stat().st_size
        
        if video_size == 0:
            logger.error(f"Video file is empty: {args.video}")
            return
            
        if model_size == 0:
            logger.error(f"Model file is empty: {args.model}")
            return
            
        logger.info(f"Video file size: {video_size / (1024*1024):.1f} MB")
        logger.info(f"Model file size: {model_size / (1024*1024):.1f} MB")
        
    except PermissionError as e:
        logger.error(f"Permission denied accessing files: {e}")
        return
    except Exception as e:
        logger.error(f"Error validating input files: {e}")
        return

    # Initialize the video processor with comprehensive error handling
    try:
        logger.info("Initializing video processor...")
        logger.info(f"Video: {args.video}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Target FPS: {args.fps}")
        logger.info(f"Confidence threshold: {args.conf}")
        
        processor = VideoInferenceProcessor(
            model_path=args.model,
            video_path=args.video,
            target_fps=args.fps,
            confidence_threshold=args.conf,
        )
        
        # Store processor in app state for the web endpoints
        app_state["processor"] = processor
        logger.info("Video processor initialized successfully!")

    except FileNotFoundError as e:
        logger.error(f"File not found during processor initialization: {e}")
        return
    except ValueError as e:
        logger.error(f"Invalid value during processor initialization: {e}")
        return
    except ImportError as e:
        logger.error(f"Missing dependency during processor initialization: {e}")
        logger.info("Please ensure all required packages are installed (ultralytics, easyocr, opencv-python)")
        return
    except Exception as e:
        logger.error(f"Unexpected error during processor initialization: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return

    # Start the FastAPI server with error handling
    try:
        logger.info(f"Starting video stream server on http://{args.host}:{args.port}")
        logger.info("Open your browser and navigate to the server URL to view the live stream")
        logger.info("Press Ctrl+C to stop the server")

        uvicorn.run(
            "__main__:app", 
            host=args.host,
            port=args.port,
            log_level="info",
            access_log=True
        )
        
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {args.port} is already in use. Please try a different port.")
        else:
            logger.error(f"Network error starting server: {e}")
        return
    except KeyboardInterrupt:
        logger.info("Server stopped by user (Ctrl+C)")
        return
    except Exception as e:
        logger.error(f"Unexpected error starting server: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return
    finally:
        # Cleanup
        if app_state.get("processor"):
            try:
                app_state["processor"].cap.release()
                logger.info("Video capture resources released")
            except Exception as e:
                logger.warning(f"Error releasing video capture: {e}")


if __name__ == "__main__":
    main()
