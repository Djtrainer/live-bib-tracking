#!/usr/bin/env python3
"""Live sharpness readout for focusing a lens by hand.

Prints a focus score twice a second while you turn the lens; stop where it
peaks. The score is the variance of the Laplacian of a centre crop (edges
get sharper, the number climbs); it is only comparable against itself on the
same scene, so aim at a fixed, detailed target at the distance you care
about -- for a finish line, the line itself from where the camera will stand
-- and do not move the camera while adjusting.

    python scripts/focus.py --source 0
    python scripts/focus.py --source 'v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=1920,height=1080,framerate=30/1 ! appsink drop=true max-buffers=1'
    python scripts/focus.py --source 0 --save focus_check.jpg   # writes the last frame on exit
    python scripts/focus.py --source http://localhost:8001/video_feed   # while the race stack holds the camera
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402

from race_cv.capture import open_source  # noqa: E402


def sharpness(image, fraction: float) -> float:
    h, w = image.shape[:2]
    ch, cw = int(h * fraction), int(w * fraction)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    gray = cv2.cvtColor(image[y0 : y0 + ch, x0 : x0 + cw], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mjpeg_frames(url: str):
    """Frames from an MJPEG stream, e.g. the API's /video_feed while race_cv runs."""
    import urllib.request

    import numpy as np

    from race_cv.capture import Frame

    stream = urllib.request.urlopen(url, timeout=10)
    buf = b""
    index = 0
    while True:
        chunk = stream.read(65536)
        if not chunk:
            return
        buf += chunk
        while True:
            start = buf.find(b"\xff\xd8")
            end = buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
            if start < 0 or end < 0:
                buf = buf[-2:] if start < 0 else buf[start:]
                break
            image = cv2.imdecode(np.frombuffer(buf[start : end + 2], np.uint8), cv2.IMREAD_COLOR)
            buf = buf[end + 2 :]
            if image is not None:
                yield Frame(image=image, capture_ts=time.time(), index=index)
                index += 1


class _Stream:
    def __init__(self, url):
        self.url = url
        self.frame_width = self.frame_height = 0

    def frames(self):
        for frame in mjpeg_frames(self.url):
            self.frame_height, self.frame_width = frame.image.shape[:2]
            yield frame

    def release(self):
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Live focus score for a camera")
    parser.add_argument("--source", required=True, help="camera index or GStreamer launch string")
    parser.add_argument("--region", type=float, default=0.5,
                        help="centre crop measured, as a fraction of each dimension")
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between readouts")
    parser.add_argument("--save", default=None, help="write the last frame here on exit")
    args = parser.parse_args(argv)

    source = _Stream(args.source) if args.source.startswith("http") else open_source(args.source)
    print(f"{args.source}; measuring the centre "
          f"{int(args.region * 100)}%; Ctrl-C to stop. Turn the lens slowly; stop at the peak.")
    best = 0.0
    last = 0.0
    frame = None
    try:
        for frame in source.frames():
            now = time.time()
            if now - last < args.interval:
                continue
            last = now
            score = sharpness(frame.image, args.region)
            best = max(best, score)
            bar = "#" * min(60, int(score / max(best, 1e-9) * 60))
            print(f"\r{score:8.0f}  (best so far {best:8.0f})  {bar:<60}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        if args.save and frame is not None:
            cv2.imwrite(args.save, frame.image)
            print(f"wrote {args.save}")
        source.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
