import os
import sys
import argparse
import cv2
import dotenv
import easyocr
from pathlib import Path
from ultralytics import YOLO
import time
import traceback
from collections import Counter
import numpy as np
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn
from collections import defaultdict
import asyncio

from image_processor.utils import get_logger

logger = get_logger()
dotenv.load_dotenv()
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    logger.info("Application startup: FastAPI server ready")

    # Initialize processor using environment variables (for Docker) or command-line args (for direct execution)
    try:
        # Check if processor is already initialized (from main() function)
        if app_state.get("processor") is not None:
            logger.info("Processor already initialized in main() - skipping lifespan initialization")
            yield
            return
            
        # Check if we're running via uvicorn (Docker) or direct execution
        if 'uvicorn' in sys.modules or any('uvicorn' in arg for arg in sys.argv):
            # Running via uvicorn (Docker) - use environment variables
            logger.info("Detected uvicorn execution - using environment variables")
            
            # Get configuration from environment variables
            video_path_str = os.getenv("VIDEO_PATH", "data/raw/race_1080p.mp4")
            model_path_str = os.getenv("MODEL_PATH", "/app/runs/detect/yolo11_reshuffled_data/weights/last.pt")
            # model_path_str = os.getenv("MODEL_PATH", "runs/detect/yolo11_reshuffled_data/weights/last.pt")
            target_fps = int(os.getenv("TARGET_FPS", "8"))
            confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))
            
            # Check for live mode environment variables
            inference_mode = os.getenv("INFERENCE_MODE", "test")
            camera_index = int(os.getenv("CAMERA_INDEX", "1"))
            
            logger.info(f"Environment INFERENCE_MODE: {inference_mode}")
            logger.info(f"Environment CAMERA_INDEX: {camera_index}")
            
            # Set video source based on inference mode
            if inference_mode == 'live':
                video_source = camera_index
                logger.info(f"Live Mode: Using camera index {video_source}")
            else:
                video_source = video_path_str
                logger.info(f"Test Mode: Using video file {video_source}")
            
            # Validate model file exists
            model_path = Path(model_path_str)
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path_str}")
                app_state["processor"] = None
                yield
                return
            
            # For test mode, validate video file exists
            if inference_mode == 'test':
                video_path = Path(video_path_str)
                if not video_path.exists():
                    logger.warning(f"Video file not found: {video_path_str}")
                    app_state["processor"] = None
                    yield
                    return
            
            logger.info(f"Initializing processor - Mode: {inference_mode}")
            logger.info(f"Model: {model_path_str}")
            logger.info(f"Video source: {video_source}")
            logger.info(f"Target FPS: {target_fps}, Confidence: {confidence_threshold}")

            processor = VideoInferenceProcessor(
                model_path=model_path_str,
                video_path=video_source,
                target_fps=target_fps,
                confidence_threshold=confidence_threshold,
            )
            app_state["processor"] = processor
            logger.info("✅ Video processor initialized successfully during startup!")
        else:
            # Running directly - processor will be initialized in main()
            logger.info("Direct execution detected - processor will be initialized in main()")
            app_state["processor"] = None
            
    except Exception as e:
        logger.error(f"Failed to initialize processor during startup: {e}")
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
        video_path: str | Path | int,
        target_fps: int = 1,
        confidence_threshold: float = 0,
        finish_line_fraction: float = 0.85,
    ):
        """
        Initializes the VideoInferenceProcessor for live bib tracking.

        Args:
            model_path (str | Path): Path to YOLO model weights.
            video_path (str | Path | int): Path to input video file or camera index for live mode.
            target_fps (int, optional): Target FPS for processing. Defaults to 1.
            confidence_threshold (float, optional): YOLO detection confidence threshold. Defaults to 0.
            finish_line_fraction (float, optional): Fraction of frame width for finish line. Defaults to 0.85.
        """
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
            
        self.target_fps = target_fps
        self.confidence_threshold = confidence_threshold

        # This will store history keyed by the PERSON's tracker ID
        self.track_history = {}

        logger.info("Loading models...")
        # This model instance will be used ONLY for tracking and will become stateful
        self.model = YOLO(str(self.model_path))
        logger.info("Models loaded successfully!")

        # Initialize EasyOCR reader
        logger.info("Initializing EasyOCR reader...")
        self.ocr_reader = easyocr.Reader(["en"], gpu=True)
        logger.info("EasyOCR reader initialized!")

        # Video capture and properties
        # Handle both file paths and camera indices using the stored video_path
        if self.is_live_mode:
            # Live camera mode - video_path is an integer camera index
            logger.info(f"Attempting to open camera with index: {self.video_path}")
            
            # Try different camera backends for better compatibility
            backends_to_try = [
                cv2.CAP_V4L2,    # Video4Linux2 (Linux)
                cv2.CAP_GSTREAMER,  # GStreamer
                cv2.CAP_ANY,     # Any available backend
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
                            logger.info(f"✅ Successfully opened camera {self.video_path} with backend {backend}")
                            self.cap = cap_test
                            break
                        else:
                            logger.warning(f"Camera {self.video_path} opened but cannot read frames with backend {backend}")
                            cap_test.release()
                    else:
                        cap_test.release()
                except Exception as e:
                    logger.warning(f"Failed to open camera {self.video_path} with backend {backend}: {e}")
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
        # self.frame_skip = max(1, int(self.original_fps / self.target_fps))
        self.finish_line_x = int(self.frame_width * finish_line_fraction)

        # Dynamic frame skip variables for conditional processing
        self.base_frame_skip = 1  # Scan mode - low frame rate when scene is empty
        self.focus_frame_skip = 1  # Focus mode - high frame rate when person detected
        self.current_frame_skip = self.base_frame_skip  # Start in scan mode
        self.detection_cooldown_frames = 30  # Frames to stay in focus mode after detection
        self.cooldown_counter = 0  # Timer to track cooldown

        self.inference_interval = 1
        self.last_annotated_frame = None

        # Track when processing started for wall time calculations
        self.processing_start_time = None

        # Initialize timing tracking
        self.timings = defaultdict(float)

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

    def check_finish_line_crossings(self, person_id: int, person_box, cap: cv2.VideoCapture) -> None:
        """
        Checks if a tracked racer has crossed the virtual finish line.
        """
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
                finish_time = cap.get(cv2.CAP_PROP_POS_MSEC)
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

            # Draw detection boxes and tracked persons
            if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
                try:
                    for box in results[0].boxes:
                        if not hasattr(box, "cls") or not hasattr(box, "xyxy"):
                            continue

                        cls = int(box.cls)
                        x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]

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

                        # Draw tracked persons (class 0) in blue with ID and bib info
                        elif cls == 0 and hasattr(box, "id") and box.id is not None:
                            person_id = int(box.id[0])
                            
                            current_best_read = ""
                            try:
                                if (
                                    person_id in self.track_history
                                    and self.track_history[person_id]["ocr_reads"]
                                ):
                                    reads = [
                                        r[0]
                                        for r in self.track_history[person_id]["ocr_reads"]
                                    ]
                                    if reads:
                                        current_best_read = Counter(reads).most_common(1)[0][0]
                            except Exception as e:
                                logger.warning(
                                    f"Error getting best OCR read for person {person_id}: {e}"
                                )

                            label_text = f"Racer ID {person_id} | Bib: {current_best_read}"

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
        """
        Processes a single frame for object detection, tracking, OCR, and annotation.
        This version uses a single, unified model call for efficiency.
        """
        try:
            # --- 1. Perform a single, unified tracking call for all classes ---
            t0 = time.time()
            results = self.model.track(
                frame,
                persist=True,
                tracker="config/custom_tracker.yaml",
                classes=[0, 1],  # Track both persons (0) and bibs (1)
                workers=4,
                verbose=False,
            )
            timings["YOLO_Unified_Track"] += time.time() - t0

            # --- 1.5. Implement conditional processing logic based on person detection ---
            person_detected = False
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                # Check if any persons (class 0) were detected
                for box in results[0].boxes:
                    if int(box.cls) == 0:  # Person class
                        person_detected = True
                        break

            # Update frame skip based on detection results
            if person_detected:
                # Person detected: switch to focus mode (high frame rate)
                self.current_frame_skip = self.focus_frame_skip
                self.cooldown_counter = self.detection_cooldown_frames
            else:
                # No person detected: check cooldown
                if self.cooldown_counter > 0:
                    # Still in cooldown period, stay in focus mode
                    self.cooldown_counter -= 1
                else:
                    # Cooldown expired, switch back to scan mode (low frame rate)
                    self.current_frame_skip = self.base_frame_skip

            # --- 2. Associate Bibs and Perform Smart OCR ---
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes

                # Separate the tracked objects into persons and bibs
                tracked_persons = {int(b.id): b for b in boxes if int(b.cls) == 0 and b.id is not None}
                detected_bibs = [b for b in boxes if int(b.cls) == 1]

                for person_id, person_box in tracked_persons.items():
                    # Initialize history for new racers
                    if person_id not in self.track_history:
                        self.track_history[person_id] = {
                            "ocr_reads": [],
                            "last_x_center": None,
                            "finish_time_ms": None,
                            "finish_wall_time": None,
                            "final_bib": None,
                            "final_bib_confidence": 0.0,
                        }

                    history = self.track_history[person_id]

                    # Check for finish line crossing
                    self.check_finish_line_crossings(person_id, person_box, cap)

                    # Smart OCR: Skip if we already have a high-confidence bib
                    if history["final_bib_confidence"] > 0.90:
                        continue

                    # Associate the closest bib if not yet found
                    px1, py1, px2, py2 = person_box.xyxy[0]
                    for bib_box in detected_bibs:
                        bx1, by1, bx2, by2 = bib_box.xyxy[0]
                        if px1 < (bx1 + bx2) / 2 < px2 and py1 < (by1 + by2) / 2 < py2:
                            # Only run OCR if YOLO confidence for the bib is high enough
                            yolo_bib_conf = float(bib_box.conf)
                            if yolo_bib_conf > 0.70:
                                t_ocr = time.time()
                                bib_crop = self.crop_bib_from_prediction(
                                    frame, bib_box.xyxy[0]
                                )
                                if bib_crop.size > 0:
                                    bib_number, ocr_conf = (
                                        self.extract_bib_with_easyocr(
                                            self.preprocess_for_easyocr(bib_crop)
                                        )
                                    )
                                    if bib_number and ocr_conf:
                                        history["ocr_reads"].append(
                                            (bib_number, ocr_conf, yolo_bib_conf)
                                        )
                                        # Lock in the bib if confidence is high
                                        if ocr_conf > history["final_bib_confidence"]:
                                            history["final_bib"] = bib_number
                                            history["final_bib_confidence"] = ocr_conf
                                timings["EasyOCR"] += time.time() - t_ocr
                            break  # Move to the next person

            # --- 3. Draw Predictions and UI Overlays ---
            t_draw = time.time()
            annotated_frame = self.draw_predictions(frame, results)
            annotated_frame = self.draw_ui_overlays(annotated_frame, start_time, cap)
            timings["Drawing"] += time.time() - t_draw

            return annotated_frame

        except Exception as e:
            logger.error(
                f"Critical error in _process_frame at frame {frame_count}: {e}",
                exc_info=True,
            )
            return frame

    def _print_timing_report(self):
        """
        Prints a detailed timing report of the video processing performance.
        """
        logger.info("\n" + "="*60)
        logger.info("📊 PERFORMANCE TIMING REPORT")
        logger.info("="*60)
        
        total_time = sum(self.timings.values())
        if total_time > 0:
            for operation, time_spent in sorted(self.timings.items(), key=lambda x: x[1], reverse=True):
                percentage = (time_spent / total_time) * 100
                logger.info(f"{operation:<20}: {time_spent:>8.2f}s ({percentage:>5.1f}%)")
            
            logger.info("-" * 60)
            logger.info(f"{'Total Processing Time':<20}: {total_time:>8.2f}s")
        else:
            logger.info("No timing data available")
        
        logger.info("="*60)

    def _generate_final_leaderboard(self):
        """
        Generates and prints the final race leaderboard with all finishers.
        """
        logger.info("\n" + "🏁"*30)
        logger.info("🏆 FINAL RACE LEADERBOARD 🏆")
        logger.info("🏁"*30)
        
        current_bib_results = self.determine_final_bibs()
        
        leaderboard = []
        for tracker_id, history_data in self.track_history.items():
            # Check if this racer has a recorded finish time
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

        leaderboard.sort(key=lambda x: x["time_ms"])
        
        if leaderboard:
            logger.info(f"{'Pos':<4} {'Racer ID':<10} {'Bib #':<8} {'Video Time':<12} {'Elapsed Time':<12}")
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
        
        logger.info("🏁"*30)


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
            start_time = time.time()
            
            # Set processing start time for timing calculations
            processor.processing_start_time = start_time

            try:
                logger.info("Starting video frame generation...")

                # Main video processing loop
                while True:
                    try:
                        # Read frame from video capture
                        ret, frame = processor.cap.read()
                        
                        if not ret:
                            logger.info("End of video reached")
                            break
                            
                        if frame is None or frame.size == 0:
                            logger.warning(f"Invalid frame at count {frame_count}")
                            continue

                        # For live mode, we need to check for detections to update frame skip
                        # before applying the skip logic
                        should_process_frame = (frame_count % processor.current_frame_skip == 0)
                        
                        if should_process_frame:
                            # Process the frame using the _process_frame method
                            processed_frame = processor._process_frame(
                                frame, frame_count, start_time, processor.cap, processor.timings
                            )
                        else:
                            # For skipped frames, we still need to run a quick detection check
                            # to update the dynamic frame skip logic for live mode
                            if processor.is_live_mode:
                                # Quick detection check without full processing
                                results = processor.model.track(
                                    frame,
                                    persist=True,
                                    tracker="config/custom_tracker.yaml",
                                    classes=[0],  # Only check for persons
                                    workers=1,
                                    verbose=False,
                                )
                                
                                # Update frame skip based on detection results
                                person_detected = False
                                if results and results[0].boxes is not None:
                                    for box in results[0].boxes:
                                        if int(box.cls) == 0:  # Person class
                                            person_detected = True
                                            break
                                
                                if person_detected:
                                    processor.current_frame_skip = processor.focus_frame_skip
                                    processor.cooldown_counter = processor.detection_cooldown_frames
                                else:
                                    if processor.cooldown_counter > 0:
                                        processor.cooldown_counter -= 1
                                    else:
                                        processor.current_frame_skip = processor.base_frame_skip
                            
                            frame_count += 1
                            continue

                        # Validate processed frame
                        if processed_frame is None or processed_frame.size == 0:
                            logger.warning(f"Invalid processed frame at count {frame_count}")
                            continue

                        # Encode frame as JPEG
                        ret, buffer = cv2.imencode(
                            ".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
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
                        logger.error(
                            f"Error processing frame {frame_count}: {frame_error}"
                        )

                        if error_count > max_errors:
                            logger.error(
                                "Too many frame processing errors, stopping stream"
                            )
                            break
                        continue

                logger.info(
                    f"Video stream ended. Processed {frame_count} frames with {error_count} errors."
                )
                
                # Generate final reports
                processor._print_timing_report()
                processor._generate_final_leaderboard()

            except Exception as generator_error:
                logger.error(f"Critical error in video generator: {generator_error}")
                # Yield an error frame
                try:
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        error_frame,
                        "Video Processing Error",
                        (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        error_frame,
                        str(generator_error)[:50],
                        (50, 280),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

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
            generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
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
        default="data/raw/race_1080p.mp4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/app/models/last.pt",
        help="Path to trained YOLO model",
    )
    parser.add_argument(
        "--fps", type=int, default=20, help="Target processing frame rate"
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
    parser.add_argument(
        "--inference_mode",
        choices=['test', 'live'],
        default='test',
        help='Set the inference mode to use a test video file or a live camera stream.'
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=1,
        help='The index of the camera to use for live mode (e.g., 0 for built-in, 1 for iPhone).'
    )

    try:
        args = parser.parse_args()
    except SystemExit as e:
        logger.error(f"Argument parsing failed: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error parsing arguments: {e}")
        return

    # Set video source based on inference mode
    if args.inference_mode == 'live':
        video_source = args.camera_index
        logger.info(f"Live Mode: Using camera index {video_source}")
    else:  # test mode
        video_source = args.video
        logger.info(f"Test Mode: Using video file {video_source}")

    # Validate input parameters
    try:
        if args.fps <= 0:
            logger.error(f"Invalid FPS value: {args.fps}. Must be greater than 0.")
            return

        if not (0.0 <= args.conf <= 1.0):
            logger.error(
                f"Invalid confidence threshold: {args.conf}. Must be between 0.0 and 1.0."
            )
            return

        if not (1 <= args.port <= 65535):
            logger.error(
                f"Invalid port number: {args.port}. Must be between 1 and 65535."
            )
            return

    except Exception as e:
        logger.error(f"Error validating parameters: {e}")
        return

    # Validate input files exist (skip video validation for live mode)
    try:
        model_path = Path(args.model)

        if not model_path.exists():
            logger.error(f"Model file not found: {args.model}")
            logger.info("Please check the path and ensure the model file exists.")
            return

        if not model_path.is_file():
            logger.error(f"Model path is not a file: {args.model}")
            return

        # Check model file size (basic validation)
        model_size = model_path.stat().st_size

        if model_size == 0:
            logger.error(f"Model file is empty: {args.model}")
            return

        logger.info(f"Model file size: {model_size / (1024 * 1024):.1f} MB")

        # Only validate video file for test mode
        if args.inference_mode == 'test':
            video_path = Path(args.video)

            if not video_path.exists():
                logger.error(f"Video file not found: {args.video}")
                logger.info("Please check the path and ensure the file exists.")
                return

            if not video_path.is_file():
                logger.error(f"Video path is not a file: {args.video}")
                return

            # Check video file size (basic validation)
            video_size = video_path.stat().st_size

            if video_size == 0:
                logger.error(f"Video file is empty: {args.video}")
                return

            logger.info(f"Video file size: {video_size / (1024 * 1024):.1f} MB")

    except PermissionError as e:
        logger.error(f"Permission denied accessing files: {e}")
        return
    except Exception as e:
        logger.error(f"Error validating input files: {e}")
        return

    # Initialize the video processor with comprehensive error handling
    try:
        logger.info("Initializing video processor...")
        if args.inference_mode == 'live':
            logger.info(f"Live Mode - Camera Index: {video_source}")
        else:
            logger.info(f"Test Mode - Video File: {video_source}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Target FPS: {args.fps}")
        logger.info(f"Confidence threshold: {args.conf}")

        processor = VideoInferenceProcessor(
            model_path=args.model,
            video_path=video_source,
            target_fps=args.fps,
            confidence_threshold=args.conf,
        )

        # Store processor in app state for the web endpoints
        app_state["processor"] = processor
        if args.inference_mode == 'live':
            logger.info("✅ Video processor initialized successfully in Live Mode!")
        else:
            logger.info("✅ Video processor initialized successfully in Test Mode!")

    except FileNotFoundError as e:
        logger.error(f"File not found during processor initialization: {e}")
        return
    except ValueError as e:
        logger.error(f"Invalid value during processor initialization: {e}")
        return
    except ImportError as e:
        logger.error(f"Missing dependency during processor initialization: {e}")
        logger.info(
            "Please ensure all required packages are installed (ultralytics, easyocr, opencv-python)"
        )
        return
    except Exception as e:
        logger.error(f"Unexpected error during processor initialization: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        
        logger.error(f"Traceback: {traceback.format_exc()}")
        return

    # Start the FastAPI server with error handling
    try:
        logger.info(f"Starting video stream server on http://{args.host}:{args.port}")
        logger.info(
            "Open your browser and navigate to the server URL to view the live stream"
        )
        logger.info("Press Ctrl+C to stop the server")

        uvicorn.run(
            "__main__:app",
            host=args.host,
            port=args.port,
            log_level="info",
            access_log=True,
        )

    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(
                f"Port {args.port} is already in use. Please try a different port."
            )
        else:
            logger.error(f"Network error starting server: {e}")
        return
    except KeyboardInterrupt:
        logger.info("Server stopped by user (Ctrl+C)")
        return
    except Exception as e:
        logger.error(f"Unexpected error starting server: {e}")
        logger.error(f"Error type: {type(e).__name__}")

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
