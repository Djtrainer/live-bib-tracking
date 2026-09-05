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
from .ocr import BibReader
from .pipeline import Pipeline
from .sink import FinishEvent, ResultSink

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
    source = open_source(args.source, start_epoch=time.time())
    logger.info(
        "Source %s: %dx%d @ %.1f fps (%s)",
        args.source,
        source.frame_width,
        source.frame_height,
        source.fps,
        "live" if source.is_live else "file",
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

    stopping = {"flag": False}

    def handle_signal(signum, _frame):
        logger.info("Signal %s received; finishing current frame.", signum)
        stopping["flag"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    on_result = None
    if args.preview:
        import cv2

        from .overlay import annotate

        def on_result(result):
            labels = {
                p.track_id: (pipeline.voter.resolve(p.track_id).text or "?")
                for p in result.people
            }
            frame = annotate(
                result, pipeline.line, pipeline.detector.roi, pipeline.stats, labels
            )
            cv2.imshow("race_cv preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
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
                _report(pipeline, sink, source)
                last_report = time.time()
    finally:
        pipeline.flush()
        source.release()
        if args.preview:
            import cv2

            cv2.destroyAllWindows()
        stats = sink.stop()
        _report(pipeline, sink, source)
        _report_undelivered(sink, stats)

    return 0


def _report(pipeline: Pipeline, sink: ResultSink, source) -> None:
    stats = pipeline.stats
    sink_stats = sink.stats
    dropped = getattr(source, "dropped", 0)
    logger.info(
        "health | processed %d (%.1f fps) | paced out %d | camera dropped %d | "
        "finishers %d (unknown bib %d) | delivered %d | pending %d%s",
        stats.frames_processed,
        stats.processed_fps,
        stats.frames_paced_out,
        dropped,
        stats.events_emitted,
        stats.unknown_bib_events,
        sink_stats.delivered,
        sink_stats.pending,
        f" | last error: {sink_stats.last_error}" if sink_stats.last_error else "",
    )


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
