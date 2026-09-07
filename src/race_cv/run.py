"""Standalone CV service entrypoint.

Run this as its own process. It owns the camera, detects finishes, and posts
them to the results API through a durable sink. Nothing about race timing
depends on a browser being connected, which is the architectural fault that made
the previous system fragile: the whole pipeline lived inside the MJPEG response
generator, so closing a tab stopped the race.

    python -m race_cv.run --source 1 --config config/race_cv.yaml
    python -m race_cv.run --source data/raw/race.mp4 --no-api --preview
"""

from __future__ import annotations

import argparse
import csv
import logging
import signal
import sys
import time
from pathlib import Path

from .capture import CameraSource, VideoFileSource, open_source
from .config import Config
from .detect import Detector
from .geometry_check import check_roi_covers_course, describe_roi
from .ocr import BibReader
from .pipeline import Pipeline
from .preview import RateGate, downscale
from .sink import FinishEvent, ResultSink
from .stream import FrameStreamer

logger = logging.getLogger("race_cv")


def setup_logging(verbose: bool) -> None:
    """Configure the root logger once.

    ``image_processor.utils.get_logger`` added a fresh handler on every call, so
    a process that imported it twice printed every line twice -- visible all
    through backend.log.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        stream=sys.stderr,
    )


def load_roster(path: str | Path | None) -> set[str]:
    """Read the set of valid bib numbers from a start-list CSV.

    Used to break OCR ties toward numbers that actually exist in the race.
    Missing or unreadable files are not fatal: the pipeline just loses the tie
    breaker rather than refusing to start on race morning.
    """
    if not path:
        return set()
    csv_path = Path(path)
    if not csv_path.exists():
        logger.warning("Roster not found, continuing without it: %s", csv_path)
        return set()
    bibs: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            for key in ("Bib", "bib", "bibNumber", "Bib Number", "BIB"):
                if key in row and str(row[key]).strip().isdigit():
                    bibs.add(str(row[key]).strip())
                    break
    logger.info("Loaded %d bib numbers from roster %s", len(bibs), csv_path)
    return bibs


def build_pipeline(
    config: Config,
    source,
    run_id: str,
    roster: set[str] | None = None,
    emit=None,
) -> Pipeline:
    """Wire a pipeline against an already-open frame source."""
    detector = Detector(
        config.model, config.roi, source.frame_width, source.frame_height
    )
    reader = BibReader(config.ocr) if config.ocr.enabled else None
    return Pipeline(
        config=config,
        detector=detector,
        frame_width=source.frame_width,
        frame_height=source.frame_height,
        run_id=run_id,
        bib_reader=reader,
        roster=roster,
        emit=emit,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live bib tracking CV service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Camera index (e.g. 1) or path to a video file",
    )
    parser.add_argument("--config", default="config/race_cv.yaml", help="Config YAML")
    parser.add_argument("--api-url", default=None, help="Override sink.api_url")
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Write the event log only; do not post to the API",
    )
    parser.add_argument("--roster", default=None, help="Start-list CSV for bib snapping")
    parser.add_argument(
        "--model", default=None, help="Override model.path from the config file"
    )
    parser.add_argument(
        "--preview", action="store_true", help="Show an annotated preview window"
    )
    parser.add_argument(
        "--preview-fps", type=float, default=10.0,
        help="Redraw the preview window at most this often. imshow on a 1080p "
             "frame costs ~19ms and runs on the frame loop; at 30fps that was a "
             "third of the detector's throughput. 0 = every processed frame.",
    )
    parser.add_argument(
        "--preview-scale", type=float, default=0.5,
        help="Downscale the preview frame before showing it (cost is per "
             "pixel). Detection is unaffected. 1.0 = full resolution.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Do not publish annotated frames to the API server for browser viewing",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help=(
            "For a video file, play it at its real frame rate and drop frames "
            "the pipeline is too slow to collect, the way a camera does. "
            "Without this a file is processed as fast as the CPU allows "
            "(several times real time), which will not surface coverage gaps."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    config_path = Path(args.config)
    config = Config.load(config_path if config_path.exists() else None)
    if not config_path.exists():
        logger.warning("Config %s not found; using built-in defaults.", config_path)
    if args.api_url:
        config.sink.api_url = args.api_url
    if args.no_api:
        config.sink.api_url = ""
    if args.model:
        config.model.path = args.model

    run_id = time.strftime("%Y%m%d-%H%M%S")
    source = open_source(args.source, start_epoch=time.time(), realtime=args.realtime)
    logger.info(
        "Source %s: %dx%d @ %.1f fps (%s)",
        args.source,
        source.frame_width,
        source.frame_height,
        source.fps,
        "live"
        if source.is_live
        else ("file, real-time pacing" if args.realtime else "file, as fast as possible"),
    )

    sink = ResultSink(config.sink)
    sink.start()
    pipeline = build_pipeline(
        config, source, run_id, roster=load_roster(args.roster), emit=sink.submit
    )
    logger.info(
        "Finish line %s -> %s, reference point %s",
        pipeline.line.pixel_endpoints()[0],
        pipeline.line.pixel_endpoints()[1],
        config.finish_line.reference_point,
    )

    # Pay every model's first-inference cost now, while a stall is free.
    # Left lazy, the detector's cost lands on the first frame and the OCR
    # reader's lands on the first readable bib -- i.e. mid-race, right as a
    # racer reaches the line, with the capture thread dropping frames
    # throughout. Measured on this machine: ~3.4s and ~3.0s warm, worse cold.
    detector_warmup = pipeline.detector.warmup(source.frame_width, source.frame_height)
    logger.info("Detector warm-up: %.1fs", detector_warmup)
    # A config value the model will silently ignore is worse than a wrong one:
    # it reads as tuned. Say so loudly, at the one moment someone is watching.
    for warning in pipeline.detector.warnings:
        logger.warning("CONFIG MISMATCH: %s", warning)
    roi_summary = describe_roi(config, source.frame_width, source.frame_height)
    if roi_summary:
        logger.info("%s", roi_summary)
    # The crop is the one setting that can lose a racer with nothing to show
    # for it downstream, so it is checked against the geometry that decides
    # who finished rather than trusted on its own.
    for warning in check_roi_covers_course(
        config, source.frame_width, source.frame_height
    ):
        logger.warning("CONFIG MISMATCH: %s", warning)
    if pipeline.async_ocr is not None:
        logger.info("OCR runs off the frame loop (ocr.async_reads: true)")
    if pipeline.reader is not None:
        logger.info("OCR warm-up: %.1fs", pipeline.reader.warmup())

    stopping = {"flag": False}

    def handle_signal(signum, _frame):
        logger.info("Signal %s received; finishing current frame.", signum)
        stopping["flag"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Streaming publishes to the same API server results go to, so it's
    # automatically disabled whenever there's nowhere to send it (--no-api).
    streamer = None
    if config.stream.enabled and not args.no_stream and config.sink.api_url:
        streamer = FrameStreamer(config.sink.api_url, config.stream)
        streamer.start()
        logger.info(
            "Publishing preview frames to %s/api/frame at up to %.1f fps",
            config.sink.api_url,
            config.stream.target_fps,
        )

    on_result = None
    if args.preview or streamer is not None:
        import cv2

        from .overlay import annotate

        # Neither consumer of the annotated frame wants every frame: the
        # streamer publishes at stream.target_fps and the preview window is
        # for a human. Gate here so annotate() -- a 6MB copy plus drawing --
        # and imshow are skipped on frames nobody will see. Measured: imshow
        # on every 1080p frame cost 19ms/frame and a third of the throughput.
        preview_gate = RateGate(args.preview_fps)
        stream_gate = RateGate(config.stream.target_fps)

        def on_result(result):
            now = time.time()
            show = args.preview and preview_gate.due(now)
            publish = streamer is not None and stream_gate.due(now)
            if not (show or publish):
                return
            labels = {
                p.track_id: (pipeline.voter.resolve(p.track_id).text or "?")
                for p in result.people
            }
            annotated = annotate(
                result,
                pipeline.line,
                pipeline.detector.roi,
                pipeline.stats,
                labels,
                boundary=pipeline.boundary,
            )
            if publish:
                streamer.submit(annotated)
            if show:
                cv2.imshow("race_cv preview", downscale(annotated, args.preview_scale))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    # This stops race timing. It once ended a full-clip run at
                    # 260s of 311s with a clean "all events delivered" and no
                    # other trace, which looked exactly like a decoder fault
                    # until every other cause was ruled out. Say so.
                    logger.warning(
                        "'q' pressed in the preview window: stopping the race "
                        "service after this frame. If that was not deliberate, "
                        "restart immediately -- no frames are being processed."
                    )
                    stopping["flag"] = True

    last_report = time.time()
    try:
        for frame in source.frames():
            if stopping["flag"]:
                break
            pipeline.stats.frames_seen += 1
            if not pipeline.should_process(frame):
                pipeline.stats.frames_paced_out += 1
                continue
            result = pipeline.process(frame)
            if on_result is not None:
                on_result(result)
            if time.time() - last_report >= 10:
                _report(pipeline, sink, source, streamer)
                last_report = time.time()
    finally:
        pipeline.flush()
        pipeline.close()
        source.release()
        if streamer is not None:
            streamer.stop()
        if args.preview:
            import cv2

            cv2.destroyAllWindows()
        stats = sink.stop()
        _report(pipeline, sink, source, streamer)
        _report_undelivered(sink, stats)

    return 0


def _report(
    pipeline: Pipeline, sink: ResultSink, source, streamer: FrameStreamer | None = None
) -> None:
    stats = pipeline.stats
    sink_stats = sink.stats
    dropped = getattr(source, "dropped", 0)

    message = (
        f"health | processed {stats.frames_processed} ({stats.processed_fps:.1f} fps) | "
        f"paced out {stats.frames_paced_out} | source dropped {dropped} | "
        f"finishers {stats.events_emitted} (unknown bib {stats.unknown_bib_events}"
        f"{f', handed off {stats.handoffs}' if stats.handoffs else ''}) | "
        f"delivered {sink_stats.delivered} | pending {sink_stats.pending}"
    )
    if pipeline.async_ocr is not None:
        ocr = pipeline.async_ocr.stats
        message += f" | ocr read {ocr.completed}"
        backlog = ocr.dropped_backlog + ocr.skipped_inflight
        if backlog:
            message += f" skipped {backlog}"
        if ocr.wait_timeouts:
            message += f" late {ocr.wait_timeouts}"
    if stats.two_stage_errors:
        message += (
            f" | TWO-STAGE FAILING {stats.two_stage_errors}: "
            f"{getattr(pipeline.detector, 'two_stage_last_error', '')}"
        )
    if sink_stats.last_error:
        message += f" | last error: {sink_stats.last_error}"
    if streamer is not None:
        stream_stats = streamer.stats
        message += f" | preview sent {stream_stats.sent} dropped {stream_stats.dropped}"
        if stream_stats.last_error:
            message += f" | preview error: {stream_stats.last_error}"
    logger.info(message)


def _report_undelivered(sink: ResultSink, stats) -> None:
    """Never end a run without saying out loud what did not make it."""
    if stats.undelivered <= 0:
        logger.info("All %d finish events delivered.", stats.delivered)
        return
    logger.error(
        "%d finish event(s) were NOT delivered to the API. They are preserved in %s "
        "and must be entered manually or replayed.",
        stats.undelivered,
        sink.event_log,
    )
    for event in sink.undelivered_events():
        logger.error(
            "  undelivered: bib=%s track=%s capture_ts=%.3f",
            event.bib_number,
            event.track_id,
            event.capture_ts,
        )


if __name__ == "__main__":
    raise SystemExit(main())
