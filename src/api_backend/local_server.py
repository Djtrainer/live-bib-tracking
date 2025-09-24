import argparse
import asyncio
import csv
import io
import json
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import dotenv
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from image_processor.utils import get_logger
from image_processor.video_inference import VideoInferenceProcessor

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
                "MODEL_PATH", "/app/runs/detect/yolo11_white_bibs/weights/last.pt"
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
        print(f"🔗 WebSocket client connected. Total clients: {len(self.active_connections)}")
        logger.info(f"🔗 WebSocket client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"❌ WebSocket client disconnected. Total clients: {len(self.active_connections)}")
        logger.info(f"❌ WebSocket client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        print(f"📡 BROADCAST DEBUG: Attempting to broadcast to {len(self.active_connections)} clients")
        logger.info(f"📡 BROADCAST DEBUG: Attempting to broadcast to {len(self.active_connections)} clients")
        
        if len(self.active_connections) == 0:
            print("⚠️ BROADCAST WARNING: No active WebSocket connections to broadcast to!")
            logger.warning("⚠️ BROADCAST WARNING: No active WebSocket connections to broadcast to!")
            return
        
        disconnected_clients = []
        for i, connection in enumerate(self.active_connections):
            try:
                await connection.send_text(message)
                print(f"✅ Successfully sent message to client {i+1}")
            except Exception as e:
                print(f"❌ Failed to send message to client {i+1}: {e}")
                logger.error(f"❌ Failed to send message to client {i+1}: {e}")
                disconnected_clients.append(connection)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            if client in self.active_connections:
                self.active_connections.remove(client)
        
        if disconnected_clients:
            print(f"🧹 Cleaned up {len(disconnected_clients)} disconnected clients. Active clients: {len(self.active_connections)}")


manager = ConnectionManager()

# --- In-Memory Database ---
# This will store the results while the server is running
race_results: List[Dict[str, Any]] = []

# --- Original Roster Data (Source of Truth) ---
# This stores the original, immutable roster data uploaded via CSV
# This should NEVER be modified after upload - it's our source of truth for lookups
original_roster: Dict[str, Dict[str, Any]] = {}  # Key: bibNumber, Value: racer data

# --- Race Clock State (Source of Truth) ---
# This stores the official race clock state
race_clock_state = {
    "raceStartTime": None,  # Unix timestamp when race officially started
    "status": "stopped",    # 'stopped', 'running', or 'paused'
    "offset": 0,           # Manual time adjustment in milliseconds
}

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

                        # Process every frame (no frame skipping)
                        processed_frame = processor._process_frame(
                            frame,
                            frame_count,
                            start_time,
                            processor.cap,
                            processor.timings,
                        )

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
    """Endpoint to get the current list of finishers (only racers who have completed the race)."""
    # Filter out racers who haven't finished (finishTime is None or null)
    finished_racers = [racer for racer in race_results if racer.get("finishTime") is not None]
    
    # Sort by finish time before returning
    finished_racers.sort(key=lambda x: x["finishTime"])
    
    return {"success": True, "data": finished_racers}


@app.post("/api/roster/upload")
async def upload_roster(file: UploadFile = File(...)):
    """Endpoint to upload a CSV roster file and merge with existing race data."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")
    
    try:
        # Read the uploaded file content
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Parse CSV content
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        
        # Validate required headers
        required_headers = {'bibNumber', 'racerName'}
        if not required_headers.issubset(set(csv_reader.fieldnames or [])):
            raise HTTPException(
                status_code=400, 
                detail=f"CSV must contain headers: {', '.join(required_headers)}"
            )
        
        # Fetch existing data and create a lookup dictionary keyed by bibNumber
        existing_data = {}
        for racer in race_results:
            existing_data[racer['bibNumber']] = racer.copy()
        
        uploaded_count = 0
        updated_count = 0
        errors = []
        csv_duplicates = set()  # Track duplicates within the CSV file
        
        for row_num, row in enumerate(csv_reader, start=2):  # Start at 2 because row 1 is headers
            try:
                # Validate required fields
                if not row.get('bibNumber') or not row.get('racerName'):
                    errors.append(f"Row {row_num}: Missing bibNumber or racerName")
                    continue
                
                bib_number = str(row['bibNumber'])
                
                # Check for duplicates within the CSV file
                if bib_number in csv_duplicates:
                    errors.append(f"Row {row_num}: Duplicate bib number {bib_number} in CSV file")
                    continue
                csv_duplicates.add(bib_number)
                
                # Create or update racer record
                if bib_number in existing_data:
                    # Update existing racer - preserve finishTime and rank if they exist
                    racer = existing_data[bib_number]
                    racer["racerName"] = str(row['racerName']).strip()
                    
                    # Update optional fields
                    if row.get('gender'):
                        gender = str(row['gender']).upper()
                        if gender in ["M", "MALE", "MAN"]:
                            racer["gender"] = "M"
                        elif gender in ["W", "F", "FEMALE", "WOMAN"]:
                            racer["gender"] = "W"
                        else:
                            racer["gender"] = gender
                    
                    if row.get('team'):
                        racer["team"] = str(row['team']).strip()
                    
                    updated_count += 1
                    logger.info(f"Updated existing racer: Bib #{bib_number} - {racer['racerName']}")
                    
                else:
                    # Create new racer record
                    racer = {
                        "id": bib_number,
                        "bibNumber": bib_number,
                        "racerName": str(row['racerName']).strip(),
                        "finishTime": None,  # Will be updated when racer finishes
                        "rank": None,  # Will be calculated when racer finishes
                    }
                    
                    # Add optional fields if present
                    if row.get('gender'):
                        gender = str(row['gender']).upper()
                        if gender in ["M", "MALE", "MAN"]:
                            racer["gender"] = "M"
                        elif gender in ["W", "F", "FEMALE", "WOMAN"]:
                            racer["gender"] = "W"
                        else:
                            racer["gender"] = gender
                    
                    if row.get('team'):
                        racer["team"] = str(row['team']).strip()
                    
                    existing_data[bib_number] = racer
                    uploaded_count += 1
                    logger.info(f"Added new racer: Bib #{bib_number} - {racer['racerName']}")
                
            except Exception as e:
                errors.append(f"Row {row_num}: {str(e)}")
        
        # Update the global race_results with the merged data
        race_results.clear()
        race_results.extend(existing_data.values())
        
        # CRITICAL: Update the original_roster dictionary (source of truth)
        # This preserves the original roster data for future lookups
        original_roster.clear()
        for racer in existing_data.values():
            # Only store racers who haven't finished yet as original roster entries
            # This preserves the original roster data for bib number lookups
            if racer.get("finishTime") is None:
                original_roster[racer["bibNumber"]] = {
                    "bibNumber": racer["bibNumber"],
                    "racerName": racer["racerName"],
                    "gender": racer.get("gender"),
                    "team": racer.get("team"),
                }
        
        logger.info(f"🔍 DEBUG: Updated original_roster with {len(original_roster)} entries")
        logger.info(f"🔍 DEBUG: Original roster bib numbers: {list(original_roster.keys())}")
        
        # Sort the results to maintain proper order (finished racers first, then by bib number)
        def sort_key(x):
            # First sort by whether they've finished (finished racers first)
            has_finished = x.get("finishTime") is None
            # Then by finish time (if they've finished)
            finish_time = x.get("finishTime") or float('inf')
            # Finally by bib number, handling non-numeric bibs like "Unknown-1"
            try:
                bib_sort_key = int(x["bibNumber"])
            except (ValueError, TypeError):
                # For non-numeric bibs (like "Unknown-1"), sort them after numeric bibs
                bib_sort_key = float('inf')
            
            return (has_finished, finish_time, bib_sort_key)
        
        race_results.sort(key=sort_key)
        
        # Broadcast roster update to all connected clients
        await manager.broadcast(json.dumps({"action": "reload"}))
        
        total_processed = uploaded_count + updated_count
        logger.info(f"Roster merge completed: {uploaded_count} new racers, {updated_count} updated racers")
        
        message_parts = []
        if uploaded_count > 0:
            message_parts.append(f"{uploaded_count} new racers added")
        if updated_count > 0:
            message_parts.append(f"{updated_count} existing racers updated")
        
        success_message = "Successfully processed roster: " + ", ".join(message_parts)
        
        return {
            "success": True,
            "message": success_message,
            "uploaded_count": uploaded_count,
            "updated_count": updated_count,
            "total_processed": total_processed,
            "errors": errors
        }
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")
    except Exception as e:
        logger.error(f"Error processing roster upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/api/results")
async def update_finish_time(finish_data: Dict[str, Any]):
    """Endpoint to update a pre-registered racer's finish time or add new finisher."""
    logger.info(f"🔍 DEBUG: === POST /api/results ENDPOINT CALLED ===")
    logger.info(f"🔍 DEBUG: Raw finish_data received: {finish_data}")
    logger.info(f"🔍 DEBUG: Type of finish_data: {type(finish_data)}")
    logger.info(f"🔍 DEBUG: Keys in finish_data: {list(finish_data.keys()) if isinstance(finish_data, dict) else 'Not a dict'}")

    # Validate required fields
    if "bibNumber" not in finish_data:
        logger.error(f"❌ DEBUG: Missing bibNumber field")
        return {"success": False, "message": "bibNumber is required"}

    bib_number = str(finish_data["bibNumber"])
    logger.info(f"🔍 DEBUG: Extracted bib_number: '{bib_number}' (type: {type(bib_number)})")
    
    # CRITICAL CHANGE: Handle both wall-clock time and legacy finish time formats
    finish_time = None
    
    if "wallClockTime" in finish_data:
        # NEW: Wall-clock time from video processor - calculate official finish time
        wall_clock_time = float(finish_data["wallClockTime"])
        logger.info(f"🔍 DEBUG: Received wall-clock time: {wall_clock_time}")
        
        if race_clock_state["raceStartTime"] is not None and race_clock_state["status"] == "running":
            # Calculate official finish time relative to race start
            official_finish_time_ms = (wall_clock_time - race_clock_state["raceStartTime"]) * 1000
            # Apply any manual offset
            official_finish_time_ms += race_clock_state["offset"]
            finish_time = official_finish_time_ms
            logger.info(f"🔍 DEBUG: Calculated official finish time: {finish_time}ms (race started at {race_clock_state['raceStartTime']}, offset: {race_clock_state['offset']}ms)")
        else:
            logger.warning(f"⚠️ DEBUG: Race clock not running - cannot calculate official finish time")
            return {"success": False, "message": "Race clock is not running. Please start the race clock first."}
    
    elif "finishTime" in finish_data:
        # LEGACY: Direct finish time (for manual entry or backward compatibility)
        raw_finish_time = finish_data["finishTime"]
        logger.info(f"🔍 DEBUG: Raw finish_time: {raw_finish_time} (type: {type(raw_finish_time)})")
        
        if isinstance(raw_finish_time, str):
            time_ms = time_string_to_milliseconds(raw_finish_time)
            if time_ms < 0:
                logger.error(f"❌ DEBUG: Invalid time format: {raw_finish_time}")
                return {"success": False, "message": "Invalid time format. Use MM:SS.ms"}
            finish_time = time_ms
            logger.info(f"🔍 DEBUG: Converted string time to milliseconds: {finish_time}")
        else:
            finish_time = float(raw_finish_time)
            logger.info(f"🔍 DEBUG: Finish time is already numeric: {finish_time}")
    
    else:
        logger.error(f"❌ DEBUG: Missing both wallClockTime and finishTime fields")
        return {"success": False, "message": "Either wallClockTime or finishTime is required"}

    # Debug: Show current race_results state
    logger.info(f"🔍 DEBUG: Current race_results count: {len(race_results)}")
    logger.info(f"🔍 DEBUG: Current race_results bib numbers: {[r.get('bibNumber', 'NO_BIB') for r in race_results]}")

    # Look up the pre-registered racer by bib number
    existing_racer = None
    racer_index = -1
    
    logger.info(f"🔍 DEBUG: Starting lookup for bib number: '{bib_number}'")
    for i, racer in enumerate(race_results):
        racer_bib = racer.get("bibNumber", "")
        logger.info(f"🔍 DEBUG: Comparing '{bib_number}' with racer[{i}] bib: '{racer_bib}' (match: {racer_bib == bib_number})")
        if racer["bibNumber"] == bib_number:
            existing_racer = racer
            racer_index = i
            logger.info(f"✅ DEBUG: MATCH FOUND! Racer index: {i}")
            break

    if existing_racer:
        # Merge with existing pre-registered racer data
        logger.info(f"✅ DEBUG: Found pre-registered racer at index {racer_index}")
        logger.info(f"🔍 DEBUG: Existing racer data: {existing_racer}")
        logger.info(f"🔍 DEBUG: Racer name: {existing_racer.get('racerName', 'Unknown')}")
        logger.info(f"🔍 DEBUG: Current finish time: {existing_racer.get('finishTime', 'None')}")
        
        # Update the existing racer with finish time, preserving all other data
        race_results[racer_index]["finishTime"] = finish_time
        logger.info(f"🔍 DEBUG: Updated finish time to: {finish_time}")
        
        # Calculate rank based on finish time among finished racers
        finished_racers = [r for r in race_results if r.get("finishTime") is not None]
        finished_racers.sort(key=lambda x: x["finishTime"])
        logger.info(f"🔍 DEBUG: Total finished racers: {len(finished_racers)}")
        
        # Update ranks for all finished racers
        for rank, finished_racer in enumerate(finished_racers, 1):
            for j, r in enumerate(race_results):
                if r["bibNumber"] == finished_racer["bibNumber"]:
                    race_results[j]["rank"] = rank
                    break

        # Get the complete updated racer object
        updated_racer = race_results[racer_index]
        logger.info(f"🔍 DEBUG: Complete updated racer object: {updated_racer}")
        
        # Broadcast the complete merged data to all connected WebSocket clients
        broadcast_data = {"type": "update", "data": updated_racer}
        broadcast_json = json.dumps(broadcast_data)
        logger.info(f"🔍 DEBUG: About to broadcast updated racer data")
        logger.info(f"📡 BROADCAST DEBUG: Broadcasting WebSocket message: {broadcast_json}")
        print(f"📡 Broadcasting update via WebSocket: {broadcast_json}")
        await manager.broadcast(broadcast_json)
        logger.info(f"✅ DEBUG: Successfully broadcasted updated racer data")
        
        logger.info(f"✅ DEBUG: Updated finish time for pre-registered racer: Bib #{bib_number} - {updated_racer['racerName']} - {finish_time}ms")
        return {"success": True, "data": updated_racer}

    else:
        # Racer not found in pre-registered list - create new entry (fallback)
        logger.warning(f"❌ DEBUG: Racer with bib #{bib_number} not found in roster. Creating new entry with default name.")
        
        # Calculate rank for new finisher
        current_finished_count = len([r for r in race_results if r.get("finishTime") is not None])
        new_rank = current_finished_count + 1
        logger.info(f"🔍 DEBUG: Calculated new rank: {new_rank}")
        
        # Create new finisher with merged data from finish_data and defaults
        new_finisher = {
            "id": bib_number,
            "bibNumber": bib_number,
            "racerName": finish_data.get("racerName", f"Racer #{bib_number}"),
            "finishTime": finish_time,
            "rank": new_rank,
        }
        logger.info(f"🔍 DEBUG: Created new finisher object: {new_finisher}")

        # Handle optional fields from finish_data if provided
        if "gender" in finish_data and finish_data["gender"]:
            gender = str(finish_data["gender"]).upper()
            if gender in ["M", "MALE", "MAN"]:
                new_finisher["gender"] = "M"
            elif gender in ["W", "F", "FEMALE", "WOMAN"]:
                new_finisher["gender"] = "W"
            else:
                new_finisher["gender"] = gender
            logger.info(f"🔍 DEBUG: Added gender: {new_finisher['gender']}")

        if "team" in finish_data and finish_data["team"]:
            new_finisher["team"] = str(finish_data["team"]).strip()
            logger.info(f"🔍 DEBUG: Added team: {new_finisher['team']}")

        # Add to race results
        race_results.append(new_finisher)
        logger.info(f"🔍 DEBUG: Added new finisher to race_results. Total count: {len(race_results)}")

        # Broadcast the new finisher to all connected WebSocket clients
        logger.info(f"🔍 DEBUG: About to broadcast new finisher data")
        await manager.broadcast(json.dumps({"type": "add", "data": new_finisher}))
        logger.info(f"✅ DEBUG: Successfully broadcasted new finisher data")

        logger.info(f"✅ DEBUG: Added new finisher: Bib #{bib_number} - {new_finisher['racerName']} - {finish_time}ms")
        return {"success": True, "data": new_finisher}


@app.put("/api/results/{finisher_id}")
async def update_finisher(finisher_id: str, finisher_data: Dict[str, Any]):
    """Endpoint to update an existing finisher with immutable roster lookup."""
    logger.info(f"🔍 DEBUG: === PUT /api/results/{finisher_id} ENDPOINT CALLED ===")
    logger.info(f"🔍 DEBUG: Updating finisher {finisher_id} with data: {finisher_data}")
    logger.info(f"🔍 DEBUG: Original roster has {len(original_roster)} entries: {list(original_roster.keys())}")

    if "finishTime" in finisher_data and isinstance(finisher_data["finishTime"], str):
        time_ms = time_string_to_milliseconds(finisher_data["finishTime"])
        if time_ms < 0:
            return {"success": False, "message": "Invalid time format. Use MM:SS.ms"}
        finisher_data["finishTime"] = time_ms

    # Handle optional gender and team fields
    if "gender" in finisher_data:
        # Normalize gender values
        gender = str(finisher_data["gender"]).upper()
        if gender in ["M", "MALE", "MAN"]:
            finisher_data["gender"] = "M"
        elif gender in ["W", "F", "FEMALE", "WOMAN"]:
            finisher_data["gender"] = "W"
        else:
            finisher_data["gender"] = gender  # Keep original if not standard

    if "team" in finisher_data and finisher_data["team"]:
        finisher_data["team"] = str(finisher_data["team"]).strip()

    # Find the finisher by ID
    for i, finisher in enumerate(race_results):
        if finisher["id"] == finisher_id:
            logger.info(f"🔍 DEBUG: Found finisher at index {i}: {finisher}")
            
            # Check if bibNumber has been changed
            original_bib = finisher.get("bibNumber", "")
            new_bib = finisher_data.get("bibNumber", original_bib)  # Use original if not provided
            bib_changed = new_bib != original_bib
            
            logger.info(f"🔍 DEBUG: Original bib: '{original_bib}', New bib: '{new_bib}', Changed: {bib_changed}")
            
            # CRITICAL FIX: Use immutable original_roster for lookup instead of race_results
            roster_racer = None
            if new_bib in original_roster:
                # Found in original roster - this is our source of truth!
                roster_racer = original_roster[new_bib].copy()  # Make a copy to avoid mutation
                logger.info(f"✅ DEBUG: Found in original_roster for bib #{new_bib}: {roster_racer}")
            else:
                logger.info(f"🔍 DEBUG: Bib #{new_bib} not found in original_roster")
            
            if roster_racer:
                # Create new finisher object by merging roster data with existing finish data
                merged_data = {
                    "id": finisher_id,  # Keep the original ID
                    "bibNumber": new_bib,
                    "racerName": roster_racer.get("racerName", f"Racer #{new_bib}"),
                    "finishTime": finisher.get("finishTime"),  # Preserve original finish time
                    "rank": finisher.get("rank"),  # Preserve original rank
                }
                
                # Add optional fields from roster if available
                if "gender" in roster_racer and roster_racer["gender"]:
                    merged_data["gender"] = roster_racer["gender"]
                if "team" in roster_racer and roster_racer["team"]:
                    merged_data["team"] = roster_racer["team"]
                
                # Override with any explicitly provided data from the update request
                # But prioritize roster data for name unless explicitly overridden
                for key, value in finisher_data.items():
                    if key not in ["id"] and value is not None:
                        if key == "racerName" and value.strip():
                            # Only override racerName if explicitly provided and not empty
                            merged_data[key] = value
                        elif key != "racerName":
                            # For other fields, use the provided value
                            merged_data[key] = value
                
                logger.info(f"🔍 DEBUG: Merged data from original roster: {merged_data}")
                
                # Update the finisher with merged data (IMMUTABLE - no roster mutation)
                race_results[i] = merged_data
                
                logger.info(f"✅ DEBUG: Successfully merged roster data for bib #{new_bib}")
                
                # If bib number changed, broadcast reload to ensure all clients get fresh data
                if bib_changed:
                    logger.info(f"🔍 DEBUG: Bib number changed - broadcasting reload signal")
                    await manager.broadcast(json.dumps({"action": "reload"}))
                else:
                    # Regular update broadcast
                    await manager.broadcast(json.dumps({"type": "update", "data": merged_data}))
                
                return {"success": True, "data": merged_data}
            else:
                logger.info(f"🔍 DEBUG: No roster entry found for bib #{new_bib} - proceeding with regular update")
                
                # Regular update (no roster match) - just update the provided fields
                updated_data = finisher.copy()  # Start with existing data
                updated_data.update(finisher_data)  # Update with provided data
                updated_data["id"] = finisher_id  # Ensure ID is preserved
                
                race_results[i] = updated_data
                
                logger.info(f"🔍 DEBUG: Updated finisher with regular data: {updated_data}")

                # Broadcast the update to all connected WebSocket clients
                await manager.broadcast(json.dumps({"type": "update", "data": updated_data}))

                return {"success": True, "data": updated_data}

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


# --- Race Clock API Endpoints ---

@app.get("/api/clock/status")
async def get_clock_status():
    """Get the current race clock status."""
    return {"success": True, "data": race_clock_state}


@app.post("/api/clock/start")
async def start_race_clock():
    """Start the race clock."""
    global race_clock_state
    
    current_time = time.time()
    race_clock_state["raceStartTime"] = current_time
    race_clock_state["status"] = "running"
    
    logger.info(f"🕐 Race clock started at {current_time}")
    
    # Broadcast clock update to all connected clients
    await manager.broadcast(json.dumps({
        "type": "clock_update", 
        "data": race_clock_state
    }))
    
    return {"success": True, "data": race_clock_state}


@app.post("/api/clock/stop")
async def stop_race_clock():
    """Stop the race clock."""
    global race_clock_state
    
    race_clock_state["status"] = "stopped"
    
    logger.info("🕐 Race clock stopped")
    
    # Broadcast clock update to all connected clients
    await manager.broadcast(json.dumps({
        "type": "clock_update", 
        "data": race_clock_state
    }))
    
    return {"success": True, "data": race_clock_state}


@app.post("/api/clock/edit")
async def edit_race_clock(edit_data: Dict[str, Any]):
    """Edit the race clock time."""
    global race_clock_state
    
    if "time" not in edit_data:
        return {"success": False, "message": "Time is required"}
    
    time_str = edit_data["time"]
    
    # Convert time string to milliseconds
    if isinstance(time_str, str):
        time_ms = time_string_to_milliseconds(time_str)
        if time_ms < 0:
            return {"success": False, "message": "Invalid time format. Use MM:SS.ms"}
    else:
        time_ms = float(time_str)
    
    # Calculate the offset needed to achieve the desired time
    if race_clock_state["raceStartTime"] is not None:
        current_time = time.time()
        current_race_time_ms = (current_time - race_clock_state["raceStartTime"]) * 1000
        race_clock_state["offset"] = time_ms - current_race_time_ms
    else:
        # If race hasn't started, set offset to the desired time
        race_clock_state["offset"] = time_ms
    
    logger.info(f"🕐 Race clock edited to {time_ms}ms (offset: {race_clock_state['offset']}ms)")
    
    # Broadcast clock update to all connected clients
    await manager.broadcast(json.dumps({
        "type": "clock_update", 
        "data": race_clock_state
    }))
    
    return {"success": True, "data": race_clock_state}


@app.post("/api/clock/reset")
async def reset_race_clock():
    """Reset the race clock."""
    global race_clock_state
    
    race_clock_state = {
        "raceStartTime": None,
        "status": "stopped",
        "offset": 0,
    }
    
    logger.info("🕐 Race clock reset")
    
    # Broadcast clock update to all connected clients
    await manager.broadcast(json.dumps({
        "type": "clock_update", 
        "data": race_clock_state
    }))
    
    return {"success": True, "data": race_clock_state}


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
static_dir = None
if os.path.exists("../frontend/dist"):
    # Development mode - running from src/api_backend
    static_dir = "../frontend/dist"
elif os.path.exists("frontend/dist"):
    # Production mode - running from Docker container
    static_dir = "frontend/dist"

# Only mount static files if the directory exists
if static_dir and os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    logger.info(f"Mounted static files from: {static_dir}")
else:
    logger.warning("Frontend dist directory not found - static files will not be served")
    logger.info("The server will still provide API endpoints and video streaming")


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
        default=0,
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

        # Define callback function to handle race results
        def result_callback(finisher_data):
            """Callback function called when a racer finishes - calls the API endpoint directly"""
            try:
                logger.info(f"🔍 DEBUG: === CALLBACK FUNCTION CALLED ===")
                logger.info(f"🔍 DEBUG: Callback received finisher_data: {finisher_data}")
                
                # CRITICAL FIX: Call the actual API endpoint instead of manipulating data directly
                # This ensures the proper WebSocket broadcast happens through the endpoint
                import asyncio
                
                async def call_api_endpoint():
                    """Call the POST /api/results endpoint with the finisher data"""
                    try:
                        logger.info(f"🔍 DEBUG: CALLBACK - Calling POST /api/results endpoint")
                        
                        # Call the endpoint function directly (since we're in the same process)
                        result = await update_finish_time(finisher_data)
                        
                        logger.info(f"✅ DEBUG: CALLBACK - API endpoint returned: {result}")
                        
                        if result.get("success"):
                            logger.info(f"✅ DEBUG: CALLBACK - Successfully processed finisher via API endpoint")
                        else:
                            logger.error(f"❌ DEBUG: CALLBACK - API endpoint failed: {result.get('message', 'Unknown error')}")
                            
                    except Exception as api_error:
                        logger.error(f"❌ DEBUG: CALLBACK - Error calling API endpoint: {api_error}")
                        logger.error(f"❌ DEBUG: CALLBACK - API error details: {type(api_error).__name__}: {str(api_error)}")
                
                # Schedule the API call in the event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Schedule the API call to run in the event loop
                        future = asyncio.run_coroutine_threadsafe(call_api_endpoint(), loop)
                        logger.info(f"✅ DEBUG: CALLBACK - Scheduled API call in event loop")
                    else:
                        logger.warning("Event loop is not running - cannot schedule API call")
                except RuntimeError as e:
                    logger.warning(f"No event loop available for API call: {e}")
                    # Fallback: Try to run the API call synchronously (not ideal but better than nothing)
                    try:
                        logger.info(f"🔍 DEBUG: CALLBACK - Attempting synchronous fallback")
                        # Create a new event loop for this thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        new_loop.run_until_complete(call_api_endpoint())
                        new_loop.close()
                        logger.info(f"✅ DEBUG: CALLBACK - Synchronous fallback completed")
                    except Exception as fallback_error:
                        logger.error(f"❌ DEBUG: CALLBACK - Synchronous fallback failed: {fallback_error}")
                
                logger.info(f"✅ DEBUG: CALLBACK - Callback processing completed")
                
            except Exception as e:
                logger.error(f"❌ DEBUG: CALLBACK - Error in result callback: {e}")
                logger.error(f"❌ DEBUG: CALLBACK - Exception details: {type(e).__name__}: {str(e)}")

        processor = VideoInferenceProcessor(
            model_path=args.model,
            video_path=video_source,
            target_fps=args.fps,
            confidence_threshold=args.conf,
            result_callback=result_callback,
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
