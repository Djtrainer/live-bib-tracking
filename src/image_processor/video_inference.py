"""
Video Inference Module

This module contains the VideoInferenceProcessor class and related utilities
for processing race video footage to track racers and extract bib numbers.

Note: This module has been refactored to be a reusable library.
The FastAPI server functionality has been moved to src/api_backend/local_server.py
"""

import csv
import os
import queue
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import dotenv
import easyocr
import numpy as np
from ultralytics import YOLO

from image_processor.utils import get_logger

logger = get_logger()
dotenv.load_dotenv()

COOL_DOWN_FRAMES = 10  # Number of frames for processing cooldown
FRAME_SKIP_FRAMES = 30  # Number of frames to skip when no racers detected
# Define the scaling factor to halve the pixel count (sqrt(0.5))
SCALE_FACTOR = 0.5
CROP_SCALE_FACTOR = 0.9  # Scale factor for cropping bib regions for OCR
class FrameReader:
    def __init__(self, cap, max_queue_size=1):
        self.cap = cap
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.thread = threading.Thread(target=self._read_loop, daemon=True)

    def _read_loop(self):
        """Continuously reads frames from the camera and puts them in the queue."""
        while self.running:
            if not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if not ret:
                break
            # If the queue is full, drop the old frame and put the new one
            if self.queue.full():
                self.queue.get_nowait()
            self.queue.put(frame)
        self.running = False

    def start(self):
        """Starts the frame reading thread."""
        self.running = True
        self.thread.start()
        logger.info("FrameReader thread started.")

    def get_latest_frame(self):
        """Gets the most recent frame from the queue without blocking."""
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Stops the frame reading thread."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        logger.info("FrameReader thread stopped.")


class VideoInferenceProcessor:
    """Processes race video footage to track racers, extract bib numbers using OCR, and determine finish times.
    This class uses YOLO models for object detection and tracking, and EasyOCR for reading bib numbers.
    It maintains a history of tracked racers, detects when they cross a virtual finish line, and produces live and final leaderboards."""

    def __init__(
        self,
        model_path: str | Path,
        video_path: str | Path | int,
        result_callback=None,
        camera_width: int | None = None,
        camera_height: int | None = None,
        leaderboard_csv_path: str | Path | None = None,
    ):
        """
        Initializes the VideoInferenceProcessor for live bib tracking.

        Args:
            model_path (str | Path): Path to YOLO model weights.
            video_path (str | Path | int): Path to input video file or camera index for live mode.
            result_callback (callable, optional): Function to call with results when a racer finishes.
            camera_width (int | None): Optional width to set for camera capture.
            camera_height (int | None): Optional height to set for camera capture.
        """
        # Optional camera resolution requested by caller (may be None)
        self.requested_width = camera_width
        self.requested_height = camera_height

        self.model_path = Path(model_path)

        # Handle both file paths and camera indices
        if isinstance(video_path, int):
            # Live camera mode - store the camera index directly
            self.video_path = video_path
            self.is_live_mode = True
        else:
            # Test video file mode - convert to Path
            self.video_path = Path(video_path)
            self.is_live_mode = False

        # This will store history keyed by the PERSON's tracker ID
        self.track_history = {}

        # Track the previous frame's tracked persons for efficiency
        self.previous_tracked_persons = set()

        logger.info("Loading models...")
        # This model instance will be used ONLY for tracking and will become stateful
        self.model = YOLO(str(self.model_path))
        logger.info("Models loaded successfully!")

        # Initialize EasyOCR reader
        logger.info("Initializing EasyOCR reader...")
        self.ocr_reader = easyocr.Reader(["en"])
        logger.info("EasyOCR reader initialized!")

        # Video capture and properties
        # Handle both file paths and camera indices using the stored video_path
        if self.is_live_mode:
            # Live camera mode - video_path is an integer camera index
            logger.info(f"Attempting to open camera with index: {self.video_path}")

            # Try different camera backends for better compatibility
            backends_to_try = [
                cv2.CAP_V4L2,  # Video4Linux2 (Linux)
                cv2.CAP_GSTREAMER,  # GStreamer
                cv2.CAP_ANY,  # Any available backend
            ]

            self.cap = None
            for backend in backends_to_try:
                try:
                    logger.info(f"Trying camera backend: {backend}")
                    cap_test = cv2.VideoCapture(self.video_path, backend)
                    if cap_test.isOpened():
                        # Test if we can actually read a frame
                        ret, frame = cap_test.read()
                        if ret and frame is not None:
                            logger.info(
                                f"✅ Successfully opened camera {self.video_path} with backend {backend}"
                            )
                            self.cap = cap_test

                            # If caller requested a specific resolution, apply it now.
                            try:
                                if self.requested_width is not None:
                                    logger.info(
                                        f"Setting requested camera width: {self.requested_width}"
                                    )
                                    self.cap.set(
                                        cv2.CAP_PROP_FRAME_WIDTH,
                                        int(self.requested_width),
                                    )
                                if self.requested_height is not None:
                                    logger.info(
                                        f"Setting requested camera height: {self.requested_height}"
                                    )
                                    self.cap.set(
                                        cv2.CAP_PROP_FRAME_HEIGHT,
                                        int(self.requested_height),
                                    )
                                # Read back actual frame size and log
                                actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                logger.info(
                                    f"Camera actual resolution after set: {actual_w}x{actual_h}"
                                )
                            except Exception as e:
                                logger.warning(f"Failed to set camera resolution: {e}")

                            break
                        else:
                            logger.warning(
                                f"Camera {self.video_path} opened but cannot read frames with backend {backend}"
                            )
                            cap_test.release()
                    else:
                        cap_test.release()
                except Exception as e:
                    logger.warning(
                        f"Failed to open camera {self.video_path} with backend {backend}: {e}"
                    )
                    continue

            if self.cap is None or not self.cap.isOpened():
                # Provide detailed error message with troubleshooting tips
                error_msg = f"""
❌ Could not open camera with index: {self.video_path}

🔧 Troubleshooting steps:
1. Check if camera is connected and not in use by another application
2. Try different camera indices (0, 1, 2, etc.)
3. Ensure Docker has camera permissions:
   - Linux: Add user to 'video' group, use --device=/dev/video0
   - macOS: Grant camera permissions to Docker/Terminal
   - Windows: Use --privileged flag

4. Available cameras can be checked with:
   ls /dev/video* (Linux)
   
5. Common camera indices:
   - 0: Built-in camera
   - 1: External USB camera (iPhone, webcam)
   - 2+: Additional cameras

6. If using Docker, ensure proper device mounting:
   --device=/dev/video0 --device=/dev/video1
"""
                logger.error(error_msg)
                raise ValueError(f"Could not open camera with index: {self.video_path}")
        else:
            # Test video file mode - video_path is a Path object
            logger.info(f"Opening video file: {self.video_path}")
            self.cap = cv2.VideoCapture(str(self.video_path))
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video file: {self.video_path}")
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._get_finish_line()

        # Track when processing started for wall time calculations
        self.processing_start_time = None

        # Initialize timing tracking
        self.timings = defaultdict(float)

        # Callback function for when racers finish
        self.result_callback = result_callback

        # Path to save leaderboard CSVs. If not provided, try env var or default to results/leaderboard_{timestamp}.csv
        env_path = os.getenv("LEADERBOARD_CSV_PATH")
        if leaderboard_csv_path is not None:
            self.leaderboard_csv_path = Path(leaderboard_csv_path)
        elif env_path:
            self.leaderboard_csv_path = Path(env_path)
        else:
            # Default path inside repository results folder with timestamp
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.leaderboard_csv_path = Path("data/results") / f"leaderboard_{ts}.csv"

        # Ensure parent directory exists
        try:
            self.leaderboard_csv_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If we can't create the directory, we'll log later when trying to write
            pass

        self.frames_to_skip = 0
        self.last_annotated_frame = None
        self.cool_down_frames = 0
        self.no_racers_frames_counter = 0
    # no testing_mode field (restored original behavior)

    def _get_finish_line(self):

        base_thickness = max(1, int(self.frame_width / 640))
        finish_line_thickness = max(2, int(self.frame_width / 240))
        
        self.guide_line_left = {
            "p1": (
                self.frame_width * 0.31,
                self.frame_height,
            ),  # Bottom-left (adjust X)
            "p2": (
                self.frame_width * 0.285,
                self.frame_height * 0.49,
            ),  # Mid-top (adjust both)
            "color": (1000, 500, 0),
            "thickness": base_thickness,
            "dash_length": 10,
        }

        self.guide_line_right = {
            "p1": (
                self.frame_width,
                self.frame_height * 0.78,
            ),  # Bottom-right (adjust X)
            "p2": (
                self.frame_width * 0.32,
                self.frame_height * 0.49,
            ),  # Mid-top (adjust both)
            "color": (1000, 500, 0),
            "thickness": base_thickness,
            "dash_length": 10,
        }

        self.guide_line_horizon = {
            "p1": (0, self.frame_height * 0.49),  # Bottom-right (adjust X)
            "p2": (
                self.frame_width * 0.35,
                self.frame_height * 0.49,
            ),  # Mid-top (adjust both)
            "color": (1000, 500, 0),
            "thickness": base_thickness,
            "dash_length": 10,
        }

        self.guide_finish_line = {
            "p1": (0, self.frame_height * 1.09),  # Bottom-right (adjust X)
            "p2": (self.frame_width, self.frame_height * 0.78),  # Mid-top (adjust both)
            "color": (0, 0, 500),
            "thickness": finish_line_thickness,
            "dash_length": 2,
        }

        x1, y1 = self.guide_finish_line["p1"]
        x2, y2 = self.guide_finish_line["p2"]

        # Calculate slope (m) and y-intercept (b)
        self.finish_line_m = (y2 - y1) / (x2 - x1)
        self.finish_line_b = y1 - self.finish_line_m * x1

        self.roi_points = np.array(
            [
                (self.guide_line_left['p2'][0], self.guide_line_left['p2'][1]*CROP_SCALE_FACTOR),      # Top-left
                (self.frame_width, self.guide_line_right['p2'][1]*CROP_SCALE_FACTOR), # Top-right
                (self.frame_width, self.frame_height), # Bottom-right
                (self.guide_line_left['p2'][0],self.frame_height),  
            ],
            dtype=np.int32,
        )
        self.crop_x1 = self.roi_points[:, 0].min()
        self.crop_y1 = self.roi_points[:, 1].min()
        self.crop_x2 = self.roi_points[:, 0].max()
        self.crop_y2 = self.roi_points[:, 1].max()

    def preprocess_for_easyocr(self, image_crop: np.ndarray) -> np.ndarray:
        """
        Faster preprocessing for EasyOCR:
        - convert to gray
        - resize to fixed height (maintain aspect)
        - apply CLAHE for contrast
        - apply Otsu threshold (fast)
        """
        if image_crop is None or image_crop.size == 0:
            return image_crop

        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop

        # scale crop to target height for OCR (improves recognition & is faster than heavy denoising)
        target_h = 120  # tune: 80-160 commonly good for bib text
        h, w = gray.shape[:2]
        if h == 0 or w == 0:
            return gray
        scale = max(1.0, target_h / float(h))
        if scale != 1.0:
            new_w = int(w * scale)
            gray = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Fast thresholding (Otsu)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

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

    def check_finish_line_crossings(
        self, person_id: int, person_data: dict, cap: cv2.VideoCapture
    ) -> None:
        """
        Checks if a tracked racer has entered the finish zone.
        Records their time and sets a flag the first moment they enter the zone.
        """
        history = self.track_history.get(person_id)
        if not history or history.get("has_finished", False):
            return

        # Get the four corner points of the bounding box from the translated dict
        x1, y1, x2, y2 = person_data.get("xyxy", (0, 0, 0, 0))
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

        crossed_line = False
        for px, py in corners:
            # Calculate the line's y-value at the point's x-position
            line_y_at_px = self.finish_line_m * px + self.finish_line_b

            # Check if the point is "below" the line (higher y-value in image coordinates)
            if py >= line_y_at_px:
                crossed_line = True
                break

        if crossed_line:
            # This is the first instance the box has crossed. Record the finish.
            history["has_finished"] = True
            history["finish_time_ms"] = cap.get(cv2.CAP_PROP_POS_MSEC)
            history["finish_wall_time"] = time.time()
            history["post_finish_counter"] = 30

            logger.info(
                f"Racer ID {person_id} CROSSED hardcoded line at {history['finish_time_ms'] / 1000:.2f}s"
            )

    def _handle_finisher(self, person_id: int, history: dict) -> None:
        """
        Determines the final bib for a finisher and sends the result via callback.
        """
        if not self.result_callback:
            return

        # Determine the final bib number using the voting system
        final_bib_results = self.determine_final_bibs()
        bib_result = final_bib_results.get(person_id)

        bib_number = bib_result["final_bib"] if bib_result else f"Unknown-{person_id}"

        # Construct the payload
        payload = {
            "bibNumber": bib_number,
            "wallClockTime": history["finish_wall_time"],
            "racerName": f"Racer {person_id}",  # Placeholder name
        }

        # Send the data to the backend
        try:
            logger.info(f"Sending finisher data to backend: {payload}")
            self.result_callback(payload)
            history["result_sent"] = True
            logger.info(f"Successfully sent finisher data for Bib #{bib_number}")
        except Exception as e:
            logger.error(
                f"Error calling result callback for Bib #{bib_number}: {e}",
                exc_info=True,
            )

    def _finalize_lost_trackers(self, tracked_person_ids: set) -> None:
        """
        Finds racers who have finished but are no longer tracked and finalizes their results.
        """
        # Create a list of IDs to avoid modifying the dictionary while iterating
        finished_but_lost_ids = []
        logger.debug(
            f"Checking for lost trackers among {len(self.track_history)} total tracked racers"
        )
        for person_id, history in self.track_history.items():
            logger.debug(f"Racer ID {person_id}: {history}")
            logger.debug(f"Currently tracked IDs: {tracked_person_ids}")
            logger.debug(
                f"Racer ID {person_id}: has_finished={history.get('has_finished')}, result_sent={history.get('result_sent')}, in_tracked={person_id in tracked_person_ids}"
            )
            if (
                history.get("has_finished")
                and not history.get("result_sent")
                and person_id not in tracked_person_ids
            ):
                finished_but_lost_ids.append(person_id)
                logger.debug(
                    f"Racer ID {person_id}: Found finished racer who left frame - will finalize"
                )

        for person_id in finished_but_lost_ids:
            logger.info(
                f"🏁 Racer ID {person_id} has left the frame. Finalizing result immediately."
            )
            self._handle_finisher(person_id, self.track_history[person_id])
            self.print_live_leaderboard()

        # Debug logging for tracking
        if tracked_person_ids:
            logger.debug(f"Currently tracked person IDs: {tracked_person_ids}")

        finished_not_sent = [
            pid
            for pid, hist in self.track_history.items()
            if hist.get("has_finished") and not hist.get("result_sent")
        ]

        if finished_not_sent:
            logger.debug(f"Finished racers not yet sent: {finished_not_sent}")

    def determine_final_bibs(self):
        """
        Determines the most likely bib number for each racer based on OCR and YOLO confidence scores.
        """
        final_results = {}
        for tracker_id, data in self.track_history.items():
            # If OCR was locked for this tracker, return that immediately
            if data.get("ocr_locked") and data.get("final_bib"):
                final_results[tracker_id] = {
                    "final_bib": data.get("final_bib"),
                    "score": data.get("final_bib_confidence", 1.0),
                }
                continue
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
                # The score for each vote is its confidence
                score = ocr_conf
                # Add the score to the total for that bib number
                scores[bib_num] = scores.get(bib_num, 0) + score

            # 3. Find the bib number with the highest total score
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

            # (no testing-mode multiple entries; original single finish entry above)

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

        # Save a local CSV copy of the live leaderboard each time it's updated
        try:
            # Use the frontend-format CSV so it matches the Download CSV from the Admin UI
            # Map the live leaderboard entries into the expected server format where possible
            results = []
            for entry in leaderboard:
                results.append(
                    {
                        "id": entry.get("id"),
                        "bibNumber": entry.get("bib"),
                        "racerName": "",  # live processor does not have roster names
                        "finishTime": entry.get("time_ms"),
                        "gender": "",
                        "team": "",
                    }
                )
            self.save_leaderboard_csv_frontend_format(results)
        except Exception as e:
            logger.warning(f"Failed to save live leaderboard CSV: {e}")

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

    def draw_predictions(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draws predictions on the input image, including finish line, detected boxes, and tracked persons.
        """
        try:
            # Validate input image
            if frame is None or frame.size == 0:
                logger.warning("Invalid image provided to draw_predictions")
                return np.zeros((480, 640, 3), dtype=np.uint8)

            annotated_image = frame.copy()

            def draw_dashed_line(
                img, p1, p2, color=(1000, 500, 0), thickness=3, dash_length=10
            ):
                dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                dashes = int(dist / dash_length)
                for i in range(dashes):
                    start = float(i) / dashes
                    end = float(i + 0.5) / dashes  # Draw for half the dash length

                    x_start = int(p1[0] + start * (p2[0] - p1[0]))
                    y_start = int(p1[1] + start * (p2[1] - p1[1]))
                    x_end = int(p1[0] + end * (p2[0] - p1[0]))
                    y_end = int(p1[1] + end * (p2[1] - p1[1]))

                    cv2.line(img, (x_start, y_start), (x_end, y_end), color, thickness)

            draw_dashed_line(
                annotated_image,
                self.guide_line_left["p1"],
                self.guide_line_left["p2"],
                self.guide_line_left["color"],
                self.guide_line_left["thickness"],
                self.guide_line_left["dash_length"],
            )
            draw_dashed_line(
                annotated_image,
                self.guide_line_right["p1"],
                self.guide_line_right["p2"],
                self.guide_line_right["color"],
                self.guide_line_right["thickness"],
                self.guide_line_right["dash_length"],
            )
            draw_dashed_line(
                annotated_image,
                self.guide_line_horizon["p1"],
                self.guide_line_horizon["p2"],
                self.guide_line_horizon["color"],
                self.guide_line_horizon["thickness"],
                self.guide_line_horizon["dash_length"],
            )
            draw_dashed_line(
                annotated_image,
                self.guide_finish_line["p1"],
                self.guide_finish_line["p2"],
                self.guide_finish_line["color"],
                self.guide_finish_line["thickness"],
                self.guide_finish_line["dash_length"],
            )

            cv2.polylines(
                annotated_image,
                [self.roi_points],
                isClosed=True,
                color=(0, 255, 0),
                thickness=3,
            )

            # Draw detection boxes and tracked persons from translated_boxes list
            try:
                if results:
                    for box_data in results:
                        try:
                            cls = int(box_data.get("cls", -1))
                            x1, y1, x2, y2 = [int(c) for c in box_data.get("xyxy", (0, 0, 0, 0))]

                            # Validate coordinates
                            if not (
                                0 <= x1 < x2 <= annotated_image.shape[1]
                                and 0 <= y1 < y2 <= annotated_image.shape[0]
                            ):
                                continue

                            # Draw bib detection boxes (class 1) in red
                            if cls == 1:
                                cv2.rectangle(
                                    annotated_image,
                                    (x1, y1),
                                    (x2, y2),
                                    (0, 0, 255),
                                    2,
                                )

                            # Draw tracked persons (class 0) with ID and bib info
                            elif cls == 0 and box_data.get("id") is not None:
                                person_id = int(box_data.get("id"))

                                current_best_read = ""
                                try:
                                    if (
                                        person_id in self.track_history
                                        and self.track_history[person_id]["ocr_reads"]
                                    ):
                                        reads = [r[0] for r in self.track_history[person_id]["ocr_reads"]]
                                        if reads:
                                            current_best_read = Counter(reads).most_common(1)[0][0]
                                except Exception as e:
                                    logger.warning(
                                        f"Error getting best OCR read for person {person_id}: {e}"
                                    )

                                # If a final bib was locked for this racer, prefer that label
                                final_bib = None
                                final_conf = None
                                try:
                                    if person_id in self.track_history:
                                        final_bib = self.track_history[person_id].get("final_bib")
                                        final_conf = self.track_history[person_id].get("final_bib_confidence")
                                except Exception:
                                    pass

                                displayed_bib = final_bib if final_bib else current_best_read
                                if final_bib and final_conf is not None:
                                    label_text = f"Racer ID {person_id} | Bib: {displayed_bib} ({final_conf:.2f})"
                                else:
                                    label_text = f"Racer ID {person_id} | Bib: {displayed_bib}"

                                # Draw rectangle
                                cv2.rectangle(
                                    annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2
                                )

                                # Draw text with background for better visibility
                                try:
                                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
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
                                    logger.warning(f"Error drawing text for person {person_id}: {e}")

                        except Exception as e:
                            logger.warning(f"Skipping a box in draw_predictions due to error: {e}")
            except Exception as e:
                logger.error(f"Error processing detection boxes: {e}")

            return annotated_image

        except Exception as e:
            logger.error(f"Critical error in draw_predictions: {e}")
            # Return original image as fallback
            return (
                frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            )
    
    def _process_frame(
        self,
        frame: np.ndarray,
        frame_count: int,
        start_time: float,
        cap: cv2.VideoCapture,
        timings: defaultdict,
    ) -> np.ndarray:
    
        # --- 0. Frame Skipping Cooldown Logic (No changes needed) ---
        if self.frames_to_skip > 0:
            logger.debug(f"Skipping frame, {self.frames_to_skip} frames left to skip.")
            if self.last_annotated_frame is not None:
                return self.draw_ui_overlays(
                    self.last_annotated_frame.copy(), start_time, cap
                )
            else:
                return self.draw_ui_overlays(frame, start_time, cap)

        try:
            # --- 1. CROP original frame and CREATE LOW-RES version ---
            # Crop the high-resolution ROI from the original frame
            high_res_crop = frame[self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2]

            # Get the target dimensions for the low-resolution version
            low_res_width = int(high_res_crop.shape[1] * SCALE_FACTOR)
            low_res_height = int(high_res_crop.shape[0] * SCALE_FACTOR)

            # Create the small, low-resolution image for the model
            low_res_crop = cv2.resize(
                high_res_crop, (low_res_width, low_res_height), interpolation=cv2.INTER_AREA
            )

            # --- 2. Run model on the SMALL, LOW-RESOLUTION image ---
            t0 = time.time()
            results = self.model.track(
                low_res_crop, # <-- Feed the small image to the model
                tracker="config/custom_tracker.yaml",
                classes=[0,1], # Recommend tracking only persons for speed
                verbose=False,
            )
            self.timings["YOLO_Unified_Track"] += time.time() - t0
            
            tracked_persons = {}
            detected_bibs = []
            # --- 3. EXTRACT and perform TWO-STEP TRANSLATION of coordinates ---
            translated_boxes = []
            if results and results[0].boxes.id is not None:
                for box in results[0].boxes:
                    # Extract coordinates from the low-res results
                    x1, y1, x2, y2 = box.xyxy[0].clone()
                    
                    # STEP A: Scale coordinates UP to match the high-res crop's dimensions
                    x1 /= SCALE_FACTOR
                    y1 /= SCALE_FACTOR
                    x2 /= SCALE_FACTOR
                    y2 /= SCALE_FACTOR
                    
                    # STEP B: Add the crop's top-left offset to match the full frame's dimensions
                    x1 += self.crop_x1
                    y1 += self.crop_y1
                    x2 += self.crop_x1
                    y2 += self.crop_y1
                    
                    # Create our own dictionary with fully translated coordinates
                    translated_boxes.append({
                        'xyxy': (x1, y1, x2, y2),
                        'id': int(box.id.item()),
                        'conf': float(box.conf.item()),
                        'cls': int(box.cls.item())
                    })
            
                # Helper: compute interpolated x on a guide line at a given y
                def _interpolate_x_at_y(line, y_val: float) -> float:
                    x1, y1 = line["p1"]
                    x2, y2 = line["p2"]
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    # If the guide line is (nearly) horizontal, just return x1
                    if abs(y2 - y1) < 1e-6:
                        return x1
                    t = (y_val - y1) / (y2 - y1)
                    return x1 + t * (x2 - x1)

                def _point_between_guides(px: float, py: float) -> bool:
                    xl = _interpolate_x_at_y(self.guide_line_left, py)
                    xr = _interpolate_x_at_y(self.guide_line_right, py)
                    left = min(xl, xr)
                    right = max(xl, xr)
                    return (px >= left) and (px <= right)

                # Now, loop over our clean, translated data
                for box_data in translated_boxes:
                    if box_data['cls'] == 0: # Person
                        # Check if the person is within the visual guide lines
                        px1, py1, px2, py2 = box_data['xyxy']
                        corners = [(px1, py1), (px2, py1), (px1, py2), (px2, py2)]
                        if any(_point_between_guides(cx, cy) for cx, cy in corners):
                            tracked_persons[box_data['id']] = box_data
                    
                    elif box_data['cls'] == 1: # Bib
                        detected_bibs.append(box_data)

            # --- NEW: Activate cooldown if no racers were found ---
            if tracked_persons:
                # Racers were detected, so reset our counter.
                self.no_racers_frames_counter = 0
            else:
                # No racers were detected, so increment our counter.
                self.no_racers_frames_counter += 1
                # logger.debug(
                #     f"No racers detected for {self.no_racers_frames_counter} consecutive frames."
                # )

                # Check if the counter has reached the cooldown threshold.
                if self.no_racers_frames_counter >= COOL_DOWN_FRAMES:
                    # logger.info(
                    #     f"No racers for {COOL_DOWN_FRAMES} frames. Activating skip for the next {FRAME_SKIP_FRAMES} frames."
                    # )
                    # Activate the skip
                    self.frames_to_skip = FRAME_SKIP_FRAMES
                    # Reset the counter so we don't immediately skip again
                    self.no_racers_frames_counter = 0
            # --- End of New Logic ---

            current_tracked_ids = set(tracked_persons.keys())
            if current_tracked_ids != self.previous_tracked_persons:
                self._finalize_lost_trackers(current_tracked_ids)
                self.previous_tracked_persons = current_tracked_ids

            # --- 3. Process each tracked person (OCR, finishing logic, etc.) ---
            if tracked_persons:
                for person_id, person_box in tracked_persons.items():
                    # (Your existing logic for processing each racer remains unchanged)
                    if person_id not in self.track_history:
                        self.track_history[person_id] = {
                            "ocr_reads": [],
                            "last_x_center": None,
                            "finish_time_ms": None,
                            "finish_wall_time": None,
                            "final_bib": None,
                                "final_bib_confidence": 0.0,
                                "ocr_locked": False,
                            "has_finished": False,
                            "result_sent": False,
                            "post_finish_counter": 0,
                        }

                    history = self.track_history[person_id]
                    if history["has_finished"] and not history.get(
                        "result_sent", False
                    ):
                        if history["post_finish_counter"] > 0:
                            history["post_finish_counter"] -= 1
                        if history["post_finish_counter"] == 0:
                            self._handle_finisher(person_id, history)
                            self.print_live_leaderboard()

                    self.check_finish_line_crossings(person_id, person_box, cap)

                    # If OCR has been locked (very high confidence) or we already
                    # have a high-confidence final bib, skip additional OCR work
                    if history.get("ocr_locked", False) or (
                        history["final_bib_confidence"] > 0.90
                        and history["post_finish_counter"] == 0
                    ):
                        continue

                    px1, py1, px2, py2 = person_box.get("xyxy", (0, 0, 0, 0))
                    for bib_box in detected_bibs:
                        bx1, by1, bx2, by2 = bib_box.get("xyxy", (0, 0, 0, 0))
                        if px1 < (bx1 + bx2) / 2 < px2 and py1 < (by1 + by2) / 2 < py2:
                            yolo_bib_conf = float(bib_box.get("conf", 0.0))
                            if yolo_bib_conf > 0.70:
                                t_ocr = time.time()
                                bib_crop = self.crop_bib_from_prediction(
                                    frame, bib_box.get("xyxy", (0, 0, 0, 0))
                                )
                                if bib_crop.size > 0:
                                    bib_number, ocr_conf = (
                                        self.extract_bib_with_easyocr(
                                            self.preprocess_for_easyocr(bib_crop)
                                        )
                                    )
                                    if bib_number and ocr_conf:
                                        logger.info(
                                            f"OCR Guess for Racer ID {person_id}: '{bib_number}' (Conf: {ocr_conf:.2f})"
                                        )
                                        # Record the OCR read
                                        history["ocr_reads"].append(
                                            (bib_number, ocr_conf, yolo_bib_conf)
                                        )

                                        # If the OCR is extremely confident, lock this bib to the racer
                                        if ocr_conf is not None and float(ocr_conf) > 0.99:
                                            history["final_bib"] = bib_number
                                            history["final_bib_confidence"] = float(ocr_conf)
                                            history["ocr_locked"] = True
                                            logger.info(
                                                f"Locked OCR bib for Racer ID {person_id}: '{bib_number}' (Conf: {ocr_conf:.3f})"
                                            )
                                timings["EasyOCR"] += time.time() - t_ocr
                            break

            # --- 4. Draw Predictions and UI Overlays ---
            t_draw = time.time()
            annotated_frame = self.draw_predictions(frame, translated_boxes)
            annotated_frame = self.draw_ui_overlays(annotated_frame, start_time, cap)
            timings["Drawing"] += time.time() - t_draw

            # --- 5. Store this frame for the skip logic and return ---
            self.last_annotated_frame = (
                annotated_frame.copy()
            )  # Use copy to prevent modification
            return annotated_frame

        except Exception as e:
            logger.error(
                f"Critical error in _process_frame at frame {frame_count}: {e}",
                exc_info=True,
            )
            # On error, don't skip, just return the raw frame and try again
            return frame

    def _print_timing_report(self):
        """
        Prints a detailed timing report of the video processing performance.
        """
        logger.info("\n" + "=" * 60)
        logger.info("📊 PERFORMANCE TIMING REPORT")
        logger.info("=" * 60)

        total_time = sum(self.timings.values())
        if total_time > 0:
            for operation, time_spent in sorted(
                self.timings.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (time_spent / total_time) * 100
                logger.info(
                    f"{operation:<20}: {time_spent:>8.2f}s ({percentage:>5.1f}%)"
                )

            logger.info("-" * 60)
            logger.info(f"{'Total Processing Time':<20}: {total_time:>8.2f}s")
        else:
            logger.info("No timing data available")

        logger.info("=" * 60)

    def _generate_final_leaderboard(self):
        """
        Generates and prints the final race leaderboard with all finishers.
        """
        logger.info("\n" + "🏁" * 30)
        logger.info("🏆 FINAL RACE LEADERBOARD 🏆")
        logger.info("🏁" * 30)

        current_bib_results = self.determine_final_bibs()

        leaderboard = []
        for tracker_id, history_data in self.track_history.items():
            # Normal mode: single finish_time
            if history_data and history_data.get("finish_time_ms") is not None:
                bib_result = current_bib_results.get(tracker_id)
                bib_number = bib_result["final_bib"] if bib_result else "No Bib"

                # Calculate elapsed time since script started using actual wall-clock time
                if (
                    self.processing_start_time is not None
                    and history_data.get("finish_wall_time") is not None
                ):
                    elapsed_seconds = (
                        history_data["finish_wall_time"] - self.processing_start_time
                    )
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

            # (original single finish entry handled above)

        leaderboard.sort(key=lambda x: x["time_ms"])

        if leaderboard:
            logger.info(
                f"{'Pos':<4} {'Racer ID':<10} {'Bib #':<8} {'Video Time':<12} {'Elapsed Time':<12}"
            )
            logger.info("-" * 60)
            for i, entry in enumerate(leaderboard):
                total_seconds = entry["time_ms"] / 1000
                minutes = int(total_seconds // 60)
                seconds = int(total_seconds % 60)
                milliseconds = int((total_seconds - int(total_seconds)) * 100)
                video_time_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:02d}"

                logger.info(
                    f"{i + 1:<4} {entry['id']:<10} {entry['bib']:<8} {video_time_str:<12} {entry['wall_time']:<12}"
                )
        else:
            logger.info("No finishers recorded.")

        logger.info("🏁" * 30)

        # Save final leaderboard CSV in frontend format
        try:
            # Map to server-style finisher dicts so the frontend-format writer can include names/teams if present
            results = []
            for entry in leaderboard:
                results.append(
                    {
                        "id": entry.get("id"),
                        "bibNumber": entry.get("bib"),
                        "racerName": "",
                        "finishTime": entry.get("time_ms"),
                        "gender": "",
                        "team": "",
                    }
                )
            self.save_leaderboard_csv_frontend_format(
                results, path=self.leaderboard_csv_path
            )
        except Exception as e:
            logger.error(f"Failed to save final leaderboard CSV: {e}")

    def save_leaderboard_csv_frontend_format(
        self, results: list, path: Path | str | None = None
    ) -> Path:
        """
        Save leaderboard in the exact CSV format produced by the front-end "Download CSV" button.

        Expected columns: Rank,Bib Number,Racer Name,Finish Time,Gender,Team

        Args:
            results: A list of finisher dicts as stored in the server `race_results` (with keys like id, bibNumber, racerName, finishTime, gender, team)
            path: Optional path to write to. If None, uses self.leaderboard_csv_path.

        Returns:
            Path to written CSV file.
        """
        csv_path = Path(path) if path is not None else self.leaderboard_csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        def format_time_ms(ms: float | None) -> str:
            if ms is None or ms == "" or ms is False:
                return ""
            try:
                ms_val = float(ms)
            except Exception:
                return ""
            total_seconds = int(ms_val // 1000)
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            centiseconds = int((ms_val % 1000) / 10)
            return f"{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

        # The front-end normalizes finishTime similarly: if it's a large epoch-like value (>1e12),
        # it converts to elapsed using race clock. The server caller should pass normalized elapsed ms
        # where possible; here we'll accept results as-is and format the provided finishTime.

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Rank", "Bib Number", "Racer Name", "Finish Time", "Gender", "Team"]
            )

            # Sort results by finishTime if present
            try:
                sorted_results = sorted(
                    results,
                    key=lambda r: (
                        r.get("finishTime") is None,
                        r.get("finishTime") or float("inf"),
                    ),
                )
            except Exception:
                sorted_results = list(results)

            for idx, r in enumerate(sorted_results, start=1):
                bib = r.get("bibNumber") or r.get("bib") or ""
                name = r.get("racerName") or r.get("racer_name") or ""
                finish_time_ms = (
                    r.get("finishTime") if "finishTime" in r else r.get("time_ms")
                )
                gender = r.get("gender") or ""
                team = r.get("team") or ""

                writer.writerow(
                    [
                        idx,
                        bib,
                        name,
                        format_time_ms(finish_time_ms),
                        gender,
                        team,
                    ]
                )

        logger.info(f"Saved frontend-format leaderboard CSV to: {csv_path}")
        return csv_path
