#!/usr/bin/env python3
"""Interactively calibrate the finish line against real footage.

The 2025 pipeline hardcoded every line as a fraction of the frame dimensions,
tuned for one camera position, with no way to confirm it matched reality. Any
reposition silently invalidated it -- and because racers outside the geometry
were filtered out before the finish check, they simply never finished.

Run this against the camera in its race-day position, or against footage from
it, click the finish line, confirm the overlay, and save.

    python scripts/calibrate.py --source data/raw/race.mp4 --frame 900
    python scripts/calibrate.py --source 1 --config config/race_cv.yaml

Controls:
    left click x2   set the finish line endpoints
    f               flip which side counts as finished
    n / p           next / previous frame (video sources)
    r               reset
    s               save to the config file
    q / Esc         quit without saving
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from race_cv.config import Config  # noqa: E402

WINDOW = "calibrate finish line"
HELP = [
    "click 2 points = finish line   f = flip side",
    "n/p = frame   r = reset   s = save   q = quit",
]


def grab_frame(source: str, frame_index: int) -> tuple[np.ndarray, object]:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera {source}")
        for _ in range(10):  # let auto-exposure settle
            cap.read()
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("Camera opened but returned no frame")
        return frame, cap
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video {source}")
    if frame_index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read frame {frame_index}")
    return frame, cap


def render(frame, points, side, saved_msg) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    for point in points:
        cv2.circle(canvas, point, 8, (0, 255, 255), -1)
    if len(points) == 2:
        cv2.line(canvas, points[0], points[1], (0, 0, 255), max(2, w // 400))
        # Shade the finished side so the operator can see what "below" means.
        overlay = canvas.copy()
        (x1, y1), (x2, y2) = points
        far = h * 2 if side == "below" else -h
        polygon = np.array(
            [[x1, y1], [x2, y2], [x2, far], [x1, far]], dtype=np.int32
        )
        cv2.fillPoly(overlay, [polygon], (0, 0, 200))
        cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
        cv2.putText(
            canvas,
            f"finished side: {side}",
            (12, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.6, w / 1800),
            (0, 255, 255),
            2,
        )
    for i, text in enumerate(HELP + ([saved_msg] if saved_msg else [])):
        cv2.putText(
            canvas, text, (12, 30 + i * 32), cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, w / 2400), (255, 255, 255), 2,
        )
    return canvas


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the finish line")
    parser.add_argument("--source", required=True, help="Camera index or video path")
    parser.add_argument("--config", default="config/race_cv.yaml")
    parser.add_argument("--frame", type=int, default=0, help="Starting frame index")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = Config.load(config_path if config_path.exists() else None)

    frame_index = args.frame
    frame, cap = grab_frame(args.source, frame_index)
    height, width = frame.shape[:2]

    # Seed the display with whatever the config already holds.
    points: list[tuple[int, int]] = [
        (int(config.finish_line.p1[0] * width), int(config.finish_line.p1[1] * height)),
        (int(config.finish_line.p2[0] * width), int(config.finish_line.p2[1] * height)),
    ]
    side = config.finish_line.side
    saved_msg = ""

    def on_mouse(event, x, y, _flags, _param):
        nonlocal points, saved_msg
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) >= 2:
                points = []
            points.append((x, y))
            saved_msg = ""

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(1400, width), int(min(1400, width) * height / width))
    cv2.setMouseCallback(WINDOW, on_mouse)

    print(f"Frame {frame_index}: {width}x{height}")
    print("Click the two ends of the finish line, then press 's' to save.")

    while True:
        cv2.imshow(WINDOW, render(frame, points, side, saved_msg))
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            points, saved_msg = [], ""
        elif key == ord("f"):
            side = "above" if side == "below" else "below"
            saved_msg = ""
        elif key in (ord("n"), ord("p")) and not args.source.isdigit():
            frame_index = max(0, frame_index + (30 if key == ord("n") else -30))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, new_frame = cap.read()
            if ok:
                frame = new_frame
            print(f"frame {frame_index}")
        elif key == ord("s"):
            if len(points) != 2:
                saved_msg = "click two points first"
                continue
            config.finish_line.p1 = (points[0][0] / width, points[0][1] / height)
            config.finish_line.p2 = (points[1][0] / width, points[1][1] / height)
            config.finish_line.side = side
            config.save(config_path)
            saved_msg = f"saved to {config_path}"
            print(
                f"saved finish_line p1={config.finish_line.p1} "
                f"p2={config.finish_line.p2} side={side} -> {config_path}"
            )

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
