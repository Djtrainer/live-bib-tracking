import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import json
import queue
from starlette.responses import StreamingResponse
import cv2


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
app = FastAPI()

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


async def frame_generator():
    """Generator function to yield frames for the browser."""
    while True:
        try:
            # Get the next frame from the queue (wait up to 1 second)
            frame = frame_queue.get(timeout=1)
            # Encode the frame as a JPEG and yield it
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
        except queue.Empty:
            # If the queue is empty, just continue waiting
            pass


@app.get("/video_feed")
async def video_feed():
    """Returns the MJPEG stream."""
    return StreamingResponse(
        frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


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
