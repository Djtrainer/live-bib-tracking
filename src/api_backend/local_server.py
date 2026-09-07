import argparse
import asyncio
import csv
import io
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, AsyncGenerator, Tuple

import cv2
import logging
import numpy as np
import uvicorn
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

# Sibling module. The launcher runs this file as a script with PYTHONPATH=src;
# make the same import work when it is run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api_backend.journal import RaceJournal, _format_finish  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: F401  (kept for the legacy Docker image)

# The legacy in-process pipeline (image_processor.video_inference) is gone.
# It was imported here unconditionally, which made the results API load
# torch, EasyOCR and ultralytics at startup -- 1.8s and hundreds of MB on
# an 8 GB race machine -- for a class it never instantiated in --no-processor
# mode. race_cv owns the camera and the model; this process owns results.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("race_api")
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI application startup and shutdown.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: This is used as an async context manager for FastAPI lifespan.

    Raises:
        Exception: Any unexpected initialization error will be logged and handled.
    """
    # Code to run on startup
    logger.info("Application startup: FastAPI server ready")

    # Put the race back the way it was if this is a restart mid-race.
    # --fresh archives the old record instead. Either way, say what happened
    # at the one moment an operator is reading the log.
    if app_state.get("fresh"):
        archived = journal.archive()
        logger.info("Starting a fresh race%s", f"; previous record archived to {archived}" if archived else "")
    else:
        restored = journal.restore()
        if restored:
            saved_results, saved_clock = restored
            race_results[:] = saved_results
            race_clock_state.update(saved_clock)
            finished = sum(1 for r in race_results if r.get("finishTime") is not None)
            logger.warning(
                "RESTORED previous race state from %s: %d results (%d finished), clock %s. "
                "If this is a NEW race, stop and restart with --fresh.",
                journal.state_path, len(race_results), finished, race_clock_state.get("status"),
            )

    yield

    logger.info("Application shutdown")


class ConnectionManager:
    def __init__(self) -> None:
        """Manage active WebSocket connections.

        Initializes the connection list used to track currently connected WebSocket clients.

        Returns:
            None
        """
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection.

        Args:
            websocket: The incoming WebSocket connection to accept and track.

        Returns:
            None

        Raises:
            Exception: If accepting the WebSocket or appending to the internal list fails.
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"🔗 WebSocket client connected. Total clients: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a WebSocket connection.

        Args:
            websocket: The WebSocket connection to remove from active connections.

        Returns:
            None
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"❌ WebSocket client disconnected. Total clients: {len(self.active_connections)}"
        )

    async def broadcast(self, message: str) -> None:
        """Broadcast a text message to all active WebSocket clients.

        Args:
            message: The text message to send to all connected clients.

        Returns:
            None

        Raises:
            Exception: Errors encountered while sending to individual clients are logged
                and the failing connections are removed.
        """
        logger.info(
            f"📡 BROADCAST DEBUG: Attempting to broadcast to {len(self.active_connections)} clients"
        )

        if len(self.active_connections) == 0:
            logger.warning(
                "⚠️ BROADCAST WARNING: No active WebSocket connections to broadcast to!"
            )
            return

        disconnected_clients = []
        for i, connection in enumerate(self.active_connections):
            try:
                await connection.send_text(message)
                logger.debug(f"✅ Successfully sent message to client {i + 1}")
            except Exception as e:
                logger.error(f"❌ Failed to send message to client {i + 1}: {e}")
                disconnected_clients.append(connection)

        # Remove disconnected clients
        for client in disconnected_clients:
            if client in self.active_connections:
                self.active_connections.remove(client)

        if disconnected_clients:
            logger.debug(
                f"🧹 Cleaned up {len(disconnected_clients)} disconnected clients. Active clients: {len(self.active_connections)}"
            )


manager = ConnectionManager()

# --- In-Memory Database ---
# This will store the results while the server is running
race_results: List[Dict[str, Any]] = []

# --- Original Roster Data (Source of Truth) ---
# This stores the original, immutable roster data uploaded via CSV
# This should NEVER be modified after upload - it's our source of truth for lookups
original_roster: Dict[str, Dict[str, Any]] = {}  # Key: bibNumber, Value: racer data


def _save_server_leaderboard_snapshot() -> None:
    """Retired. Persistence is the journal (see ``journal.py``).

    This used to ask the legacy in-process pipeline to write a CSV, and did
    nothing at all when that pipeline was absent -- which is every race-day
    run. The mutation sites that call it now also call ``_journal`` directly,
    which writes the state, the human-readable leaderboard and the log. Kept
    as a no-op so those call sites need no change.
    """
    return None


# --- Race Clock State (Source of Truth) ---
# This stores the official race clock state
race_clock_state = {
    "raceStartTime": None,  # Unix timestamp when race officially started
    "status": "stopped",  # 'stopped', 'running', or 'paused'
    "offset": 0,  # Manual time adjustment in milliseconds
}

# Crash-recoverable record of everything above. Written on every change,
# restored at startup (unless --fresh). Same directory as race_cv's
# events.jsonl, so all of a race's records are in one place. See journal.py
# for why: with --no-processor nothing else persisted a manual add, a bib
# correction, or when the clock started.
journal = RaceJournal(Path(__file__).resolve().parents[2] / "data" / "results")


def _journal(action: str, detail: str = "") -> None:
    """Record the current results and clock after a change. Never raises."""
    journal.record(action, race_results, race_clock_state, detail)


def _describe(record: Dict[str, Any]) -> str:
    return (
        f"bib {record.get('bibNumber', '?')} "
        f"{record.get('racerName') or ''} "
        f"{_format_finish(record.get('finishTime'))} "
        f"[{record.get('source') or 'manual'}]"
    ).strip()

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
    """Converts a MM:SS.ms string to total milliseconds.

    Args:
        time_str: A string in the format MM:SS.ms (e.g. "01:23.45").

    Returns:
        The total time in milliseconds as a float. Returns -1.0 if the input format is invalid.

    Raises:
        ValueError: If the time components cannot be converted to integers. Handled internally
            and results in -1.0 being returned.
    """
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
async def root() -> HTMLResponse:
    """Serve the viewer HTML page at the root URL.

    Args:
        None

    Returns:
        HTMLResponse: The rendered HTML content for the viewer page.

    Raises:
        Exception: Unexpected errors while generating the HTML response will propagate as
            HTTP errors handled by FastAPI.
    """
    # When the built frontend is present, "/" is the leaderboard -- the page
    # the pavilion TV opens. The viewer page below is the fallback when it
    # is not built.
    if _FRONTEND_DIST is not None:
        return FileResponse(_FRONTEND_DIST / "index.html")
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


def _placeholder_frame(text: str) -> bytes:
    """A single JPEG shown while no external frame has arrived yet.

    Without this, an <img> pointed at /video_feed shows a broken-image icon
    until the first POST /api/frame lands, which looks identical to "this is
    broken" even when it's simply "race_cv hasn't started yet".
    """
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        image, text, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2
    )
    ok, buffer = cv2.imencode(".jpg", image)
    return buffer.tobytes() if ok else b""


async def _generate_relay_frames() -> AsyncGenerator[bytes, None]:
    """Stream whatever race_cv (or any other external CV service) posts to
    /api/frame, republishing it as MJPEG for the browser.

    Polls app_state rather than pushing, so any number of browser tabs can
    watch independently without race_cv needing to know how many viewers
    exist -- exactly the coupling that made closing a tab affect the pipeline
    in the legacy design.
    """
    last_sent_ts = None
    waiting_message = _placeholder_frame("Waiting for race_cv to publish a frame...")
    stale_after_seconds = 5.0

    while True:
        frame_bytes = app_state.get("latest_frame")
        frame_ts = app_state.get("latest_frame_ts")

        if frame_bytes is not None and frame_ts != last_sent_ts:
            last_sent_ts = frame_ts
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        elif frame_bytes is None or (time.time() - (frame_ts or 0)) > stale_after_seconds:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + waiting_message
                + b"\r\n"
            )

        await asyncio.sleep(0.05)


@app.post("/api/frame")
async def post_frame(request: Request) -> Dict[str, Any]:
    """Accept one annotated JPEG frame from an external CV service.

    Body is the raw JPEG bytes (Content-Type: image/jpeg), not multipart --
    this runs at several frames per second and the pipeline posting it must
    never block on anything heavier than a plain POST. Only the latest frame
    is kept; there is no history and no queue here on purpose, matching the
    "always show the newest thing, drop what's stale" policy used everywhere
    else frames are handled in this project.
    """
    body = await request.body()
    if not body or body[:2] != b"\xff\xd8":
        return {"success": False, "message": "expected raw JPEG bytes"}
    app_state["latest_frame"] = body
    app_state["latest_frame_ts"] = time.time()
    return {"success": True}


@app.get("/video_feed")
async def video_feed(request: Request) -> Response:
    """Stream processed video frames as an MJPEG multipart response.

    Args:
        request: The incoming FastAPI Request object (used to check client disconnects if needed).

    Returns:
        Response: A StreamingResponse that yields MJPEG frame bytes.

    Raises:
        HTTPException: If the processor is not initialized or a critical error occurs.
    """
    # race_cv owns the camera and the model; this process only republishes
    # the frames it posts to /api/frame. The in-process pipeline that used to
    # live inside this generator -- the architecture RACE_DAY_ANALYSIS.md
    # names as the root cause of the 2025 failures -- is gone.
    return StreamingResponse(
        _generate_relay_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/results")
async def get_results() -> Dict[str, Any]:
    """Return the current list of finished racers.

    Returns:
        A dictionary with success flag and data containing a list of finished racers sorted by finishTime.
    """
    # Filter out racers who haven't finished (finishTime is None or null)
    finished_racers = [
        racer for racer in race_results if racer.get("finishTime") is not None
    ]

    # Sort by finish time before returning
    finished_racers.sort(key=lambda x: x["finishTime"])

    return {"success": True, "data": finished_racers}


@app.post("/api/roster/upload")
async def upload_roster(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload and merge a roster CSV file into the in-memory race results.

    Args:
        file: The uploaded CSV file (must be UTF-8 encoded and contain 'bibNumber' and 'racerName' headers).

    Returns:
        A dictionary containing success status, counts of uploaded/updated racers, any errors, and a message.

    Raises:
        HTTPException: If the file is not a CSV, is not UTF-8, or required headers are missing.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV file")

    try:
        # Read the uploaded file content
        content = await file.read()
        csv_content = content.decode("utf-8")

        # Parse CSV content
        csv_reader = csv.DictReader(io.StringIO(csv_content))

        # Validate required headers
        required_headers = {"bibNumber", "racerName"}
        if not required_headers.issubset(set(csv_reader.fieldnames or [])):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain headers: {', '.join(required_headers)}",
            )

        # Fetch existing data and create a lookup dictionary keyed by bibNumber
        existing_data = {}
        for racer in race_results:
            existing_data[racer["bibNumber"]] = racer.copy()

        uploaded_count = 0
        updated_count = 0
        errors = []
        csv_duplicates = set()  # Track duplicates within the CSV file

        for row_num, row in enumerate(
            csv_reader, start=2
        ):  # Start at 2 because row 1 is headers
            try:
                # Validate required fields
                if not row.get("bibNumber") or not row.get("racerName"):
                    errors.append(f"Row {row_num}: Missing bibNumber or racerName")
                    continue

                bib_number = str(row["bibNumber"])

                # Check for duplicates within the CSV file
                if bib_number in csv_duplicates:
                    errors.append(
                        f"Row {row_num}: Duplicate bib number {bib_number} in CSV file"
                    )
                    continue
                csv_duplicates.add(bib_number)

                # Create or update racer record
                if bib_number in existing_data:
                    # Update existing racer - preserve finishTime and rank if they exist
                    racer = existing_data[bib_number]
                    racer["racerName"] = str(row["racerName"]).strip()

                    # Update optional fields
                    if row.get("gender"):
                        gender = str(row["gender"]).upper()
                        if gender in ["M", "MALE", "MAN"]:
                            racer["gender"] = "M"
                        elif gender in ["W", "F", "FEMALE", "WOMAN"]:
                            racer["gender"] = "W"
                        else:
                            racer["gender"] = gender

                    if row.get("team"):
                        racer["team"] = str(row["team"]).strip()

                    # Update optional age field if present
                    if row.get("age"):
                        racer["age"] = str(row["age"]).strip()

                    updated_count += 1
                    logger.info(
                        f"Updated existing racer: Bib #{bib_number} - {racer['racerName']}"
                    )

                else:
                    # Create new racer record
                    racer = {
                        "id": bib_number,
                        "bibNumber": bib_number,
                        "racerName": str(row["racerName"]).strip(),
                        "finishTime": None,  # Will be updated when racer finishes
                        "rank": None,  # Will be calculated when racer finishes
                    }

                    # Add optional fields if present
                    if row.get("gender"):
                        gender = str(row["gender"]).upper()
                        if gender in ["M", "MALE", "MAN"]:
                            racer["gender"] = "M"
                        elif gender in ["W", "F", "FEMALE", "WOMAN"]:
                            racer["gender"] = "W"
                        else:
                            racer["gender"] = gender

                    if row.get("team"):
                        racer["team"] = str(row["team"]).strip()

                    if row.get("age"):
                        racer["age"] = str(row["age"]).strip()

                    existing_data[bib_number] = racer
                    uploaded_count += 1
                    logger.info(
                        f"Added new racer: Bib #{bib_number} - {racer['racerName']}"
                    )

            except Exception as e:
                errors.append(f"Row {row_num}: {str(e)}")

        # Update the global race_results with the merged data
        race_results.clear()
        race_results.extend(existing_data.values())
        _journal("ROSTER", f"{len(existing_data)} racers loaded")

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
                    "age": racer.get("age"),
                }

        logger.info(
            f"🔍 DEBUG: Updated original_roster with {len(original_roster)} entries"
        )
        logger.info(
            f"🔍 DEBUG: Original roster bib numbers: {list(original_roster.keys())}"
        )

        # Sort the results to maintain proper order (finished racers first, then by bib number)
        def sort_key(x: Dict[str, Any]) -> Tuple[bool, float, Any]:
            """Sort key used to order race results after roster upload.

            Args:
                x: A racer dict from race_results.

            Returns:
                A tuple used for sorting: (has_finished_flag, finish_time_or_inf, bib_sort_key)
            """
            # First sort by whether they've finished (finished racers first)
            has_finished = x.get("finishTime") is None
            # Then by finish time (if they've finished)
            finish_time = x.get("finishTime") or float("inf")
            # Finally by bib number, handling non-numeric bibs like "Unknown-1"
            try:
                bib_sort_key = int(x["bibNumber"])
            except (ValueError, TypeError):
                # For non-numeric bibs (like "Unknown-1"), sort them after numeric bibs
                bib_sort_key = float("inf")

            return (has_finished, finish_time, bib_sort_key)

        race_results.sort(key=sort_key)

        # Broadcast roster update to all connected clients
        await manager.broadcast(json.dumps({"action": "reload"}))

        # Persist a server-side CSV snapshot of the leaderboard after roster changes
        try:
            _save_server_leaderboard_snapshot()
        except Exception as e:
            logger.warning(f"Failed to save leaderboard after roster upload: {e}")

        total_processed = uploaded_count + updated_count
        logger.info(
            f"Roster merge completed: {uploaded_count} new racers, {updated_count} updated racers"
        )

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
            "errors": errors,
        }

    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")
    except Exception as e:
        logger.error(f"Error processing roster upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/api/results")
async def update_finish_time(finish_data: Dict[str, Any]) -> Dict[str, Any]:
    """Add a new finisher entry to the in-memory leaderboard.

    This endpoint always creates a new record (unique ID generated) to allow duplicate bib numbers.

    Args:
        finish_data: A dictionary containing either 'wallClockTime' or 'finishTime' and 'bibNumber',
            and optionally 'racerName', 'gender', or 'team'.

    Returns:
        A dict with success flag and the created finisher data on success.

    Raises:
        HTTPException: For invalid inputs or when the race clock is not running and a wallClockTime is provided.
    """
    logger.debug("🔍 DEBUG: === POST /api/results ENDPOINT CALLED ===")
    logger.debug("🔍 DEBUG: Raw finish_data received: %s", finish_data)
    logger.debug("🔍 DEBUG: Type of finish_data: %s", type(finish_data))
    logger.info(
        "🔍 DEBUG: Keys in finish_data: %s",
        list(finish_data.keys()) if isinstance(finish_data, dict) else "Not a dict",
    )

    # Validate required fields
    if "bibNumber" not in finish_data:
        logger.error("❌ DEBUG: Missing bibNumber field")
        return {"success": False, "message": "bibNumber is required"}

    bib_number = str(finish_data["bibNumber"])
    logger.debug("🔍 DEBUG: Extracted bib_number: '%s' (type: %s)", bib_number, type(bib_number))

    # CRITICAL CHANGE: Handle both wall-clock time and legacy finish time formats
    finish_time = None

    if "wallClockTime" in finish_data:
        # NEW: Wall-clock time from video processor - calculate official finish time
        wall_clock_time = float(finish_data["wallClockTime"])
        logger.debug("🔍 DEBUG: Received wall-clock time: %s", wall_clock_time)

        if (
            race_clock_state["raceStartTime"] is not None
            and race_clock_state["status"] == "running"
        ):
            # Calculate official finish time relative to race start
            official_finish_time_ms = (
                wall_clock_time - race_clock_state["raceStartTime"]
            ) * 1000
            # Apply any manual offset
            official_finish_time_ms += race_clock_state["offset"]
            finish_time = official_finish_time_ms
            logger.info(
                "🔍 DEBUG: Calculated official finish time: %sms (race started at %s, offset: %sms)",
                finish_time,
                race_clock_state["raceStartTime"],
                race_clock_state["offset"],
            )
        else:
            logger.warning("⚠️ DEBUG: Race clock not running - cannot calculate official finish time")
            return {
                "success": False,
                "message": "Race clock is not running. Please start the race clock first.",
            }

    elif "finishTime" in finish_data:
        # LEGACY: Direct finish time (for manual entry or backward compatibility)
        raw_finish_time = finish_data["finishTime"]
        logger.debug("🔍 DEBUG: Raw finish_time: %s (type: %s)", raw_finish_time, type(raw_finish_time))

        if isinstance(raw_finish_time, str):
            time_ms = time_string_to_milliseconds(raw_finish_time)
            if time_ms < 0:
                logger.error("❌ DEBUG: Invalid time format: %s", raw_finish_time)
                return {
                    "success": False,
                    "message": "Invalid time format. Use MM:SS.ms",
                }
            finish_time = time_ms
            logger.debug("🔍 DEBUG: Converted string time to milliseconds: %s", finish_time)
        else:
            finish_time = float(raw_finish_time)
            logger.debug("🔍 DEBUG: Finish time is already numeric: %s", finish_time)

    else:
        logger.error("❌ DEBUG: Missing both wallClockTime and finishTime fields")
        return {
            "success": False,
            "message": "Either wallClockTime or finishTime is required",
        }

    # Debug: Show current race_results state
    logger.debug("🔍 DEBUG: Current race_results count: %d", len(race_results))
    logger.info(
        "🔍 DEBUG: Current race_results bib numbers: %s",
        [r.get("bibNumber", "NO_BIB") for r in race_results],
    )

    # CRITICAL FIX: Always create a new entry to allow duplicate bib numbers
    # Generate a unique ID using timestamp and bib number to ensure uniqueness
    import uuid

    # Idempotency on the client's eventId. race_cv sends one with every
    # finish and retries until this endpoint confirms; if the API died after
    # storing a result but before answering, the retry -- and any later
    # replay_events.py run -- would otherwise create a second finisher for
    # the same crossing. Manual adds from Live Management carry no eventId
    # and are always new, so duplicate bib numbers remain allowed.
    event_id = finish_data.get("eventId")
    if event_id:
        for existing in race_results:
            if existing.get("eventId") == event_id:
                logger.info("Duplicate eventId %s; returning the existing finisher", event_id)
                return {"success": True, "data": existing, "duplicate": True}

    unique_id = str(uuid.uuid4())
    logger.debug(f"🔍 DEBUG: Generated unique ID: {unique_id}")

    # Calculate rank for new finisher
    current_finished_count = len(
        [r for r in race_results if r.get("finishTime") is not None]
    )
    new_rank = current_finished_count + 1
    logger.debug(f"🔍 DEBUG: Calculated new rank: {new_rank}")

    # Look up roster data for this bib number (if available)
    roster_data = original_roster.get(bib_number, {})
    logger.debug(f"🔍 DEBUG: Roster data for bib #{bib_number}: {roster_data}")

    # Create new finisher with merged data from roster and finish_data
    new_finisher = {
        "id": unique_id,  # Always use unique ID
        "eventId": event_id,                                   # None for manual adds
        "source": finish_data.get("source") or "manual",       # race_cv / race_cv_replay / manual
        "bibNumber": bib_number,
        "racerName": roster_data.get(
            "racerName", finish_data.get("racerName", f"Racer #{bib_number}")
        ),
        "finishTime": finish_time,
        "rank": new_rank,
    }

    # Add optional fields from roster or finish_data
    if roster_data.get("gender") or finish_data.get("gender"):
        gender_source = roster_data.get("gender") or finish_data.get("gender")
        gender = str(gender_source).upper()
        if gender in ["M", "MALE", "MAN"]:
            new_finisher["gender"] = "M"
        elif gender in ["W", "F", "FEMALE", "WOMAN"]:
            new_finisher["gender"] = "W"
        else:
            new_finisher["gender"] = gender
        logger.debug(f"🔍 DEBUG: Added gender: {new_finisher['gender']}")

    if roster_data.get("team") or finish_data.get("team"):
        team_source = roster_data.get("team") or finish_data.get("team")
        new_finisher["team"] = str(team_source).strip()
        logger.debug(f"🔍 DEBUG: Added team: {new_finisher['team']}")

    # Add optional age from roster or finish_data
    if roster_data.get("age") or finish_data.get("age"):
        new_finisher["age"] = roster_data.get("age") or finish_data.get("age")
        logger.debug(f"🔍 DEBUG: Added age: {new_finisher['age']}")

    logger.debug(f"🔍 DEBUG: Created new finisher object: {new_finisher}")

    # Add to race results (always creates new entry)
    race_results.append(new_finisher)
    _journal("ADD", _describe(new_finisher))
    logger.info(
        f"🔍 DEBUG: Added new finisher to race_results. Total count: {len(race_results)}"
    )

    # Recalculate ranks for all finished racers
    finished_racers = [r for r in race_results if r.get("finishTime") is not None]
    finished_racers.sort(key=lambda x: x["finishTime"])
    logger.debug("🔍 DEBUG: Total finished racers: %d", len(finished_racers))

    # Update ranks for all finished racers
    for rank, finished_racer in enumerate(finished_racers, 1):
        for j, r in enumerate(race_results):
            if r["id"] == finished_racer["id"]:  # Match by unique ID, not bib number
                race_results[j]["rank"] = rank
                break

    # Broadcast the new finisher to all connected WebSocket clients
    logger.debug("🔍 DEBUG: About to broadcast new finisher data")
    await manager.broadcast(json.dumps({"type": "add", "data": new_finisher}))
    logger.debug("✅ DEBUG: Successfully broadcasted new finisher data")

    # Persist leaderboard snapshot after adding a new finisher
    try:
        _save_server_leaderboard_snapshot()
    except Exception as e:
        logger.warning(f"Failed to save leaderboard after adding finisher: {e}")

    logger.info(
        f"✅ DEBUG: Added new finisher: Bib #{bib_number} - {new_finisher['racerName']} - {finish_time}ms"
    )
    return {"success": True, "data": new_finisher}


@app.put("/api/results/{finisher_id}")
async def update_finisher(finisher_id: str, finisher_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing finisher by ID, merging immutable roster data when appropriate.

    Args:
        finisher_id: The unique ID of the finisher to update.
        finisher_data: Dictionary of fields to update. 'finishTime' may be a string (MM:SS.ms) or numeric.

    Returns:
        A dict with success flag and the updated finisher data on success.

    Raises:
        HTTPException: Not raised directly here, but invalid time formats return a failure response.
    """
    logger.debug(f"🔍 DEBUG: === PUT /api/results/{finisher_id} ENDPOINT CALLED ===")
    logger.debug(f"🔍 DEBUG: Updating finisher {finisher_id} with data: {finisher_data}")
    logger.info(
        f"🔍 DEBUG: Original roster has {len(original_roster)} entries: {list(original_roster.keys())}"
    )

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
            logger.debug(f"🔍 DEBUG: Found finisher at index {i}: {finisher}")

            # Check if bibNumber has been changed
            original_bib = finisher.get("bibNumber", "")
            new_bib = finisher_data.get(
                "bibNumber", original_bib
            )  # Use original if not provided
            bib_changed = new_bib != original_bib

            logger.info(
                f"🔍 DEBUG: Original bib: '{original_bib}', New bib: '{new_bib}', Changed: {bib_changed}"
            )

            # CRITICAL FIX: Only lookup roster data if bib number has changed
            roster_racer = None
            if bib_changed and new_bib in original_roster:
                # Found in original roster - this is our source of truth!
                roster_racer = original_roster[
                    new_bib
                ].copy()  # Make a copy to avoid mutation
                logger.info(
                    "✅ DEBUG: Bib changed - Found in original_roster for bib #%s: %s",
                    new_bib,
                    roster_racer,
                )
            elif bib_changed:
                logger.info(
                    "🔍 DEBUG: Bib changed but #%s not found in original_roster",
                    new_bib,
                )
            else:
                logger.debug("🔍 DEBUG: Bib not changed - skipping roster lookup")

            if roster_racer and bib_changed:
                # Create new finisher object by merging roster data with existing finish data
                merged_data = {
                    "id": finisher_id,  # Keep the original ID
                    "bibNumber": new_bib,
                    "racerName": roster_racer.get("racerName", f"Racer #{new_bib}"),
                    "finishTime": finisher.get(
                        "finishTime"
                    ),  # Preserve original finish time
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

                logger.info(
                    f"🔍 DEBUG: Merged data from original roster: {merged_data}"
                )

                # Update the finisher with merged data (IMMUTABLE - no roster mutation)
                race_results[i] = merged_data
                _journal("EDIT", _describe(merged_data))

                logger.info(
                    f"✅ DEBUG: Successfully merged roster data for bib #{new_bib}"
                )

                # If bib number changed, broadcast reload to ensure all clients get fresh data
                if bib_changed:
                    logger.info(
                        "🔍 DEBUG: Bib number changed - broadcasting reload signal"
                    )
                    await manager.broadcast(json.dumps({"action": "reload"}))
                else:
                    # Regular update broadcast
                    await manager.broadcast(
                        json.dumps({"type": "update", "data": merged_data})
                    )

                # Persist leaderboard snapshot after update
                try:
                    _save_server_leaderboard_snapshot()
                except Exception as e:
                    logger.warning(
                        f"Failed to save leaderboard after finisher update: {e}"
                    )

                return {"success": True, "data": merged_data}
            else:
                logger.info(
                    f"🔍 DEBUG: No roster entry found for bib #{new_bib} - proceeding with regular update"
                )

                # Regular update (no roster match) - just update the provided fields
                updated_data = finisher.copy()  # Start with existing data
                updated_data.update(finisher_data)  # Update with provided data
                updated_data["id"] = finisher_id  # Ensure ID is preserved

                # CRITICAL FIX: If bib number changed and not found in roster, update name to "Racer #<bib>"
                if bib_changed:
                    updated_data["racerName"] = f"Racer #{new_bib}"
                    logger.info(
                        f"🔍 DEBUG: Bib changed to unknown number - updated name to: {updated_data['racerName']}"
                    )

                race_results[i] = updated_data
                _journal("EDIT", _describe(updated_data))

                logger.info(
                    f"🔍 DEBUG: Updated finisher with regular data: {updated_data}"
                )

                # Broadcast the update to all connected WebSocket clients
                await manager.broadcast(
                    json.dumps({"type": "update", "data": updated_data})
                )

                # Persist leaderboard snapshot after update
                try:
                    _save_server_leaderboard_snapshot()
                except Exception as e:
                    logger.warning(
                        f"Failed to save leaderboard after finisher update: {e}"
                    )

                return {"success": True, "data": updated_data}

    return {"success": False, "message": "Finisher not found"}


@app.delete("/api/results/{finisher_id}")
async def delete_finisher(finisher_id: str) -> Dict[str, Any]:
    """Delete a finisher by its unique ID.

    Args:
        finisher_id: Unique ID of the finisher to remove.

    Returns:
        A dict indicating success or failure and a message.
    """
    logger.debug(f"Deleting finisher {finisher_id}")

    # Find and remove the finisher by ID
    for i, finisher in enumerate(race_results):
        if finisher["id"] == finisher_id:
            removed = race_results.pop(i)
            _journal("DELETE", _describe(removed))

            # Broadcast reload message to all connected WebSocket clients
            await manager.broadcast(json.dumps({"action": "reload"}))

            # Persist leaderboard snapshot after deletion
            try:
                _save_server_leaderboard_snapshot()
            except Exception as e:
                logger.warning(f"Failed to save leaderboard after deletion: {e}")

            return {"success": True, "message": "Finisher deleted"}

    return {"success": False, "message": "Finisher not found"}


@app.post("/api/reorder")
async def reorder_finishers(order_data: Dict[str, Any]) -> Dict[str, Any]:
    """Reorder finishers using client-provided order information.

    Args:
        order_data: A dictionary that must contain an 'order' list of objects with 'id' and 'rank'.

    Returns:
        A dict indicating success and a message.
    """
    logger.debug(f"Reordering finishers: {order_data}")

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
    _journal("REORDER", f"{len(reordered_results)} results")

    # Broadcast reload message to all connected WebSocket clients
    await manager.broadcast(json.dumps({"action": "reload"}))

    return {"success": True, "message": "Finishers reordered successfully"}


# --- Race Clock API Endpoints ---


@app.get("/api/clock/status")
async def get_clock_status() -> Dict[str, Any]:
    """Return the current race clock status.

    Returns:
        A dict with success flag and the race clock state data.
    """
    return {"success": True, "data": race_clock_state}


@app.post("/api/clock/start")
async def start_race_clock() -> Dict[str, Any]:
    """Start the official race clock and broadcast the update.

    Returns:
        A dict containing the updated race clock state.
    """
    global race_clock_state

    if race_clock_state.get("status") == "running":
        # A second press would silently re-base every time already recorded.
        # Found in review; the fix is to refuse, loudly, and leave the clock alone.
        logger.warning("Start Clock pressed while already running; ignored")
        return {
            "success": False,
            "message": "Race clock is already running. Use Reset (with force) to start over.",
            "data": race_clock_state,
        }
    current_time = time.time()
    race_clock_state["raceStartTime"] = current_time
    race_clock_state["status"] = "running"
    _journal("CLOCK", f"start at {time.strftime('%H:%M:%S', time.localtime(current_time))}")

    logger.info(f"🕐 Race clock started at {current_time}")

    # Broadcast clock update to all connected clients
    await manager.broadcast(
        json.dumps({"type": "clock_update", "data": race_clock_state})
    )

    return {"success": True, "data": race_clock_state}


@app.post("/api/clock/stop")
async def stop_race_clock() -> Dict[str, Any]:
    """Stop the official race clock and broadcast the update.

    Returns:
        A dict containing the updated race clock state.
    """
    global race_clock_state

    race_clock_state["status"] = "stopped"
    _journal("CLOCK", "stop")

    logger.info("🕐 Race clock stopped")

    # Broadcast clock update to all connected clients
    await manager.broadcast(
        json.dumps({"type": "clock_update", "data": race_clock_state})
    )

    return {"success": True, "data": race_clock_state}


@app.post("/api/clock/edit")
async def edit_race_clock(edit_data: Dict[str, Any]) -> Dict[str, Any]:
    """Edit the race clock to a specific time (MM:SS.ms or numeric milliseconds).

    Args:
        edit_data: Dict with a 'time' key containing either a time string or numeric milliseconds.

    Returns:
        A dict with success flag and the updated race clock state.
    """
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
    _journal("CLOCK", f"edit: set to {_format_finish(time_ms)} (offset {race_clock_state['offset']:.0f}ms)")

    logger.info(
        f"🕐 Race clock edited to {time_ms}ms (offset: {race_clock_state['offset']}ms)"
    )

    # Broadcast clock update to all connected clients
    await manager.broadcast(
        json.dumps({"type": "clock_update", "data": race_clock_state})
    )

    return {"success": True, "data": race_clock_state}


@app.post("/api/clock/reset")
async def reset_race_clock(force: bool = False) -> Dict[str, Any]:
    """Reset the race clock to its initial stopped state and broadcast the update.

    Returns:
        A dict with success flag and the reset race clock state.
    """
    global race_clock_state

    finished = sum(1 for r in race_results if r.get("finishTime") is not None)
    if finished and not force:
        # Every recorded time is relative to the clock. Resetting it with
        # results present makes them all meaningless; require intent.
        logger.warning("Clock reset refused: %d finishers recorded (pass force=true)", finished)
        return {
            "success": False,
            "message": f"{finished} finisher(s) are recorded against this clock. "
                       "Reset with force=true if you really mean to invalidate their times.",
            "data": race_clock_state,
        }

    race_clock_state = {
        "raceStartTime": None,
        "status": "stopped",
        "offset": 0,
    }

    _journal("CLOCK", "reset")
    logger.info("🕐 Race clock reset")

    # Broadcast clock update to all connected clients
    await manager.broadcast(
        json.dumps({"type": "clock_update", "data": race_clock_state})
    )

    return {"success": True, "data": race_clock_state}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle incoming WebSocket connections for real-time updates.

    Args:
        websocket: The WebSocket connection for this client.

    Returns:
        None

    Raises:
        WebSocketDisconnect: When the client disconnects.
    """
    await manager.connect(websocket)
    logger.debug(
        f"WebSocket client connected. Total clients: {len(manager.active_connections)}"
    )
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.debug(
            f"WebSocket client disconnected. Total clients: {len(manager.active_connections)}"
        )


# Serve the built frontend from this same server, so the leaderboard and the
# Live Management page are reachable from ANY machine on the network at
# http://<this-host>:<port>/ and /admin -- the pavilion TV, a tablet at the
# finish line -- with no screen sharing and no per-machine configuration.
#
# Why here rather than a separate static server: both pages build their API
# and WebSocket URLs from window.location.host, i.e. strictly same-origin.
# A site served from a different port only works if that server proxies
# /api and /ws back here. Serving from here makes that a non-problem.
#
# dist/ is resolved relative to THIS FILE, not the working directory. The
# previous lookup ("../frontend/dist" or "frontend/dist") depended on where
# the process was started from and found nothing when launched from the
# repo root, which is how start-race-cv.sh launches it.
_FRONTEND_DIST = None
for _candidate in (
    Path(__file__).resolve().parents[1] / "frontend" / "dist",
    Path("../frontend/dist"),
    Path("frontend/dist"),
):
    if (_candidate / "index.html").is_file():
        _FRONTEND_DIST = _candidate.resolve()
        break

if _FRONTEND_DIST is not None:
    logger.info(f"Serving the frontend from {_FRONTEND_DIST}")

    @app.get("/{path:path}", include_in_schema=False)
    async def _frontend(path: str):
        """Static files from dist/, and index.html for everything else.

        Registered last, so every API route above wins first. Unknown paths
        get index.html because the React app owns its own routes (/admin):
        a plain static mount 404s on them, which is exactly what happened.
        """
        if path.startswith(("api/", "ws", "video_feed")):
            raise HTTPException(status_code=404)
        candidate = (_FRONTEND_DIST / path).resolve() if path else None
        if (
            candidate is not None
            and str(candidate).startswith(str(_FRONTEND_DIST))
            and candidate.is_file()
        ):
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_DIST / "index.html")
else:
    logger.warning(
        "Frontend dist directory not found - static files will not be served. "
        "Build it with: (cd src/frontend && npm run build)"
    )
    logger.info("The server will still provide API endpoints and video streaming")


def main() -> None:
    """Run the results API.

    This process owns results, the clock, the journal, and the site. It does
    not own a camera or a model: race_cv does, as a separate process that
    posts finishes here. The flags the launcher passes are the only ones.
    """
    parser = argparse.ArgumentParser(description="Live bib tracking results API")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind address. 0.0.0.0 so the tablet and the pavilion can reach it.")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--fresh", action="store_true",
        help="Start a new race: archive any saved race state instead of restoring it. "
             "Without this, a restart puts the previous results and clock back.",
    )
    parser.add_argument(
        "--no-processor", action="store_true",
        help="Accepted for compatibility with start-race-cv.sh; this server never "
             "runs an in-process pipeline any more.",
    )
    args = parser.parse_args()

    if not (1 <= args.port <= 65535):
        logger.error("Invalid port %s; must be 1-65535", args.port)
        sys.exit(2)
    app_state["fresh"] = bool(args.fresh)

    logger.info("Starting results API on http://%s:%s", args.host, args.port)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except OSError as exc:
        if "address already in use" in str(exc).lower() or getattr(exc, "errno", None) == 48:
            logger.error("Port %s is already in use. Is a previous API still running? "
                         "./stop-race-cv.sh, or: lsof -Pi :%s -sTCP:LISTEN", args.port, args.port)
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
