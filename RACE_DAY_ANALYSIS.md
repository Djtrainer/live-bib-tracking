# Why race day failed

Analysis of the 2025 system, with evidence from `backend.log` (run of 2025-10-24).
The reported symptom was **racers missed entirely**. That symptom has several
independent causes, and none of them reported an error to the operator.

## 1. Finishers were silently discarded

**This is proven, not inferred.**

`POST /api/results` refuses a finisher whenever the race clock is not running
(`src/api_backend/local_server.py:920-943`), returning HTTP 200 with
`{"success": false}`. The processor never looked at the answer:

```python
# src/image_processor/video_inference.py:501-503
self.result_callback(payload)          # fire-and-forget
history["result_sent"] = True          # marked sent before knowing the outcome
logger.info(f"Successfully sent finisher data for Bib #{bib_number}")
```

From `backend.log`:

```
562: Sending finisher data to backend: {'bibNumber': '10', ...}
567: Successfully sent finisher data for Bib #10          <-- not true
582: API endpoint returned: {'success': False,
       'message': 'Race clock is not running. Please start the race clock first.'}
647: Race clock started at 1761358257                     <-- 7 seconds too late
```

Bib #10 was detected correctly, reported in the log as a success, refused by the
API, and never retried. Three finishers were lost this way in one short session.
Because `result_sent` was already `True`, nothing would ever try again.

Any late clock start, stop, or reset silently voids every racer in that window,
and the log says everything worked.

## 2. Frame-skip death spiral

`COOL_DOWN_FRAMES = 10`, `FRAME_SKIP_FRAMES = 30`
(`src/image_processor/video_inference.py:32-33`). After 10 consecutive frames
with no *gated* person, the pipeline dropped 30 frames — a full second at 30 fps
— then immediately re-armed. In `backend.log` this fires continuously, every
~310 ms, for hundreds of consecutive lines.

A racer can cross the line entirely inside that blind second. It also wrecks
tracking: ByteTrack with `track_buffer: 120` fed frames a second apart produces
useless motion prediction, ID switches, and lost `has_finished` state.

## 3. Hardcoded geometry, with no way to check it

`_get_finish_line` (`video_inference.py:278-350`) hardcoded every line as a
fraction of the frame, tuned for one camera position. Two separate gates dropped
racers before they could ever finish:

- The ROI crop discarded the left ~28% of every frame (`crop_x1 = 0.285 * W`).
  Racers there were invisible to the detector.
- `_point_between_guides` (`video_inference.py:1088`) filtered tracked people to
  a wedge. Anyone outside it never reached the finish-line check.

There was no calibration step and no overlay confirming the geometry matched
reality, so a camera reposition invalidated it silently.

## 4. Configuration that did nothing

`--conf` and `--fps` were parsed, range-checked, and logged
(`local_server.py:1560-1641`) — then never passed to anything. Same for the
`TARGET_FPS` and `CONFIDENCE_THRESHOLD` environment variables
(`local_server.py:77-78`). `model.track()` (`video_inference.py:1034`) was called
with no `conf`, `imgsz`, `device` or `half`.

Every past tuning session was therefore uninterpretable: the knobs were not
connected to anything.

## 5. Timestamps measured the wrong thing

`finish_wall_time = time.time()` (`video_inference.py:463`) was stamped when a
frame finished *processing*, not when it was captured. `finish_time_ms` read
`cap.get(CAP_PROP_POS_MSEC)` from a capture object the reader thread had already
advanced well past, so it did not describe the frame being processed at all.
Under load, times drifted by however far behind the pipeline was running — and
the UI displayed that drift as "Lag: Xs" without connecting it to the results.

## 6. Single-digit bibs could not resolve

`determine_final_bibs` filtered to `2 <= len(bib) <= 5`
(`video_inference.py:574-576`), so bibs 1–9 were discarded by the voting path
entirely and survived only via the `>0.99` confidence lock.

## 7. The model's validation score was not measuring generalization

`yolo11n`, 200 epochs on CPU, `mAP50 = 0.941`, `mAP50-95 = 0.604`.

`src/yolo_utils/organize_labels.py:66-69` copies every labelled image into
`train/`, and the splits were made by hand from frames of the same sparse test
recordings. Sequential frames from one video appearing in both train and val
means the score measures memorization, not generalization.

Race day added packs of runners, occlusion, different light and a different
camera pose — none of it represented in training. This is why test runs looked
excellent and race day did not.

## 8. The architectural cause underneath all of it

The whole pipeline ran inside the MJPEG HTTP response generator
(`local_server.py:453`, `generate_frames`). A browser tab drove race timing:

- Heavy synchronous work (YOLO, EasyOCR, drawing, JPEG encode) ran on the
  asyncio event loop, starving WebSocket broadcasts and API handlers.
- Closing or refreshing the tab stopped or restarted processing, and each new
  connection reset `processing_start_time`.
- Multiple viewers meant multiple concurrent loops sharing one stateful
  `model.track(persist=True)`.

## Environment trap

`run_live_native.sh` resolves `python3` to the base conda environment, which has
`ultralytics 8.1.43`. That version **cannot load YOLO11 weights** at all
(`AttributeError: Can't get attribute 'C3k2'`). It can only load the CoreML
`.mlpackage`, so there was no working `.pt` fallback on race day. `pyproject.toml`
listed `xgboost`, `shap` and `torch-geometric` — copied from another project —
and none of `ultralytics`, `opencv-python`, `easyocr`, `fastapi` or `uvicorn`.

## What replaces it

`src/race_cv/` — a standalone CV service with no HTTP dependency. See
[src/race_cv/README.md](src/race_cv/README.md).

| Cause | Fix |
|---|---|
| Silent finisher loss | `sink.py`: persist before send, retry until confirmed, idempotency key |
| Frame-skip bursts | `pipeline.py`: uniform deadline-based pacing |
| Hardcoded geometry | `config.py` + `scripts/calibrate.py`, normalized coordinates |
| Tracking gate | Everyone is tracked; geometry decides only who finished |
| Dead config | Every parameter reaches the model, and typos raise |
| Processing-time stamps | `capture.py` stamps at grab; crossings are interpolated |
| Single-digit bibs | `ocr.py` length bounds are config, defaulting to 1 |
| Unmeasurable changes | `scripts/replay.py` + `scripts/score.py` |
