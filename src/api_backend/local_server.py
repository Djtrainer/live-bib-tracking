import os
import sys
import argparse
import dotenv
from pathlib import Path
import time
import traceback
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from typing import List, Dict, Any
import json
import numpy as np
import uvicorn

from image_processor.video_inference import VideoInferenceProcessor
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
            logger.info(
                "Processor already initialized in main() - skipping lifespan initialization"
            )
            yield
            return

        # Check if we're running via uvicorn (Docker) or direct execution
        if "uvicorn" in sys.modules or any("uvicorn" in arg for arg in sys.argv):
            # Running via uvicorn (Docker) - use environment variables
            logger.info("Detected uvicorn execution - using environment variables")

            # Get configuration from environment variables
            video_path_str = os.getenv("VIDEO_PATH", "data/raw/race_1080p.mp4")
            model_path_str = os.getenv(
                "MODEL_PATH", "/app/runs/detect/yolo11_new_data/weights/last.pt"
            )
            target_fps = int(os.getenv("TARGET_FPS", "8"))
            confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))

            # Check for live mode environment variables
            inference_mode = os.getenv("INFERENCE_MODE", "test")
            camera_index = int(os.getenv("CAMERA_INDEX", "1"))

            logger.info(f"Environment INFERENCE_MODE: {inference_mode}")
            logger.info(f"Environment CAMERA_INDEX: {camera_index}")

            # Set video source based on inference mode
            if inference_mode == "live":
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
            if inference_mode == "test":
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
            logger.info(
                "Direct execution detected - processor will be initialized in main()"
            )
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


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()

# --- In-Memory Database ---
# This will store the results while the server is running
race_results: List[Dict[str, Any]] = []

# --- FastAPI App ---
app = FastAPI(lifespan=lifespan, title="Live Bib Tracking - Unified Server")

# Add CORS middleware to allow requests from admin UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def time_string_to_milliseconds(time_str: str) -> float:
    """Converts a MM:SS.ms string to total milliseconds."""
    try:
        minutes, seconds_ms = time_str.split(":")
        seconds, centiseconds = seconds_ms.split(".")

        total_ms = (
            (int(minutes) * 60 * 1000)
            + (int(seconds) * 1000)
            + (int(centiseconds) * 10)
        )
        return float(total_ms)
    except (ValueError, IndexError):
        # Return an invalid value if the format is wrong
        return -1.0


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
                        should_process_frame = (
                            frame_count % processor.current_frame_skip == 0
                        )

                        if should_process_frame:
                            # Process the frame using the _process_frame method
                            processed_frame = processor._process_frame(
                                frame,
                                frame_count,
                                start_time,
                                processor.cap,
                                processor.timings,
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
                                    processor.current_frame_skip = (
                                        processor.focus_frame_skip
                                    )
                                    processor.cooldown_counter = (
                                        processor.detection_cooldown_frames
                                    )
                                else:
                                    if processor.cooldown_counter > 0:
                                        processor.cooldown_counter -= 1
                                    else:
                                        processor.current_frame_skip = (
                                            processor.base_frame_skip
                                        )

                            frame_count += 1
                            continue

                        # Validate processed frame
                        if processed_frame is None or processed_frame.size == 0:
                            logger.warning(
                                f"Invalid processed frame at count {frame_count}"
                            )
                            continue

                        # Encode frame as JPEG
                        import cv2

                        ret, buffer = cv2.imencode(
                            ".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                        )

                        if not ret or buffer is None:
                            logger.warning(f"Failed to encode frame {frame_count}")
                            error_count += 1
                            if error_count > max_errors:
                                logger.error(
                                    "Too many encoding errors, stopping stream"
                                )
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
                    import cv2

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


@app.get("/api/results")
async def get_results():
    """Endpoint to get the current list of all finishers."""
    # Sort by finish time before returning
    race_results.sort(key=lambda x: x["finishTime"])
    return {"success": True, "data": race_results}


@app.post("/api/results")
async def add_finisher(finisher_data: Dict[str, Any]):
    """Endpoint for your video_inference.py OR admin UI to add a finisher."""
    print(f"Received new finisher via POST: {finisher_data}")

    # Add a unique ID for admin UI purposes
    finisher_data["id"] = str(
        finisher_data["bibNumber"]
    )  # Use bib number as a simple ID
    if "finishTime" in finisher_data and isinstance(finisher_data["finishTime"], str):
        time_ms = time_string_to_milliseconds(finisher_data["finishTime"])
        if time_ms < 0:
            return {"success": False, "message": "Invalid time format. Use MM:SS.ms"}
        finisher_data["finishTime"] = time_ms

    race_results.append(finisher_data)

    # Broadcast the new finisher to all connected WebSocket clients (leaderboard and admin)
    await manager.broadcast(json.dumps({"type": "add", "data": finisher_data}))

    return {"success": True, "data": finisher_data}


@app.put("/api/results/{finisher_id}")
async def update_finisher(finisher_id: str, finisher_data: Dict[str, Any]):
    """Endpoint to update an existing finisher."""
    print(f"Updating finisher {finisher_id} with data: {finisher_data}")

    if "finishTime" in finisher_data and isinstance(finisher_data["finishTime"], str):
        time_ms = time_string_to_milliseconds(finisher_data["finishTime"])
        if time_ms < 0:
            return {"success": False, "message": "Invalid time format. Use MM:SS.ms"}
        finisher_data["finishTime"] = time_ms

    # Find the finisher by ID
    for i, finisher in enumerate(race_results):
        if finisher["id"] == finisher_id:
            # Update the finisher data
            finisher_data["id"] = finisher_id
            race_results[i] = finisher_data

            # Broadcast the update to all connected WebSocket clients
            await manager.broadcast(
                json.dumps({"type": "update", "data": finisher_data})
            )

            return {"success": True, "data": finisher_data}

    return {"success": False, "message": "Finisher not found"}


@app.delete("/api/results/{finisher_id}")
async def delete_finisher(finisher_id: str):
    """Endpoint to delete a finisher."""
    print(f"Deleting finisher {finisher_id}")

    # Find and remove the finisher by ID
    for i, finisher in enumerate(race_results):
        if finisher["id"] == finisher_id:
            deleted_finisher = race_results.pop(i)

            # Broadcast reload message to all connected WebSocket clients
            await manager.broadcast(json.dumps({"action": "reload"}))

            return {"success": True, "message": "Finisher deleted"}

    return {"success": False, "message": "Finisher not found"}


@app.post("/api/reorder")
async def reorder_finishers(order_data: Dict[str, Any]):
    """Endpoint to reorder finishers manually."""
    print(f"Reordering finishers: {order_data}")

    new_order = order_data.get("order", [])

    # Create a new ordered list based on the provided order
    reordered_results = []
    for order_item in new_order:
        finisher_id = order_item["id"]
        rank = order_item["rank"]

        # Find the finisher and update its rank
        for finisher in race_results:
            if finisher["id"] == finisher_id:
                finisher["rank"] = rank
                reordered_results.append(finisher)
                break

    # Update the global race_results
    race_results.clear()
    race_results.extend(reordered_results)

    # Broadcast reload message to all connected WebSocket clients
    await manager.broadcast(json.dumps({"action": "reload"}))

    return {"success": True, "message": "Finishers reordered successfully"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live leaderboard and admin sync."""
    await manager.connect(websocket)
    print(
        f"WebSocket client connected. Total clients: {len(manager.active_connections)}"
    )
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(
            f"WebSocket client disconnected. Total clients: {len(manager.active_connections)}"
        )


# Mount the frontend dist directory to serve static files (index.html, etc.)
# Use different paths for development vs production (Docker)
if os.path.exists("../frontend/dist"):
    # Development mode - running from src/api_backend
    static_dir = "../frontend/dist"
else:
    # Production mode - running from Docker container
    static_dir = "frontend/dist"

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


def main():
    """Main function that initializes the processor and starts the FastAPI server."""
    parser = argparse.ArgumentParser(description="Live Bib Tracking - Unified Server")
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
        "--port", type=int, default=8000, help="Port to bind the server to"
    )
    parser.add_argument(
        "--inference_mode",
        choices=["test", "live"],
        default="test",
        help="Set the inference mode to use a test video file or a live camera stream.",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=1,
        help="The index of the camera to use for live mode (e.g., 0 for built-in, 1 for iPhone).",
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
    if args.inference_mode == "live":
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
        if args.inference_mode == "test":
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
        if args.inference_mode == "live":
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
        if args.inference_mode == "live":
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
        logger.info(f"Starting unified server on http://{args.host}:{args.port}")
        logger.info("This server handles:")
        logger.info("  - REST API endpoints (/api/*)")
        logger.info("  - WebSocket connections (/ws)")
        logger.info("  - Live video stream (/video_feed)")
        logger.info("  - Admin frontend (static files)")
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
