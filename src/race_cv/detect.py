"""YOLO detection and tracking.

Every inference parameter is passed explicitly. The legacy call site invoked
``model.track()`` with no ``conf``, ``imgsz``, ``device`` or ``half`` and relied
on whatever defaults the installed Ultralytics version happened to use, while
the ``--conf`` and ``--fps`` command line flags were validated, logged, and then
never passed to anything. Tuning those knobs changed nothing, which made every
past tuning result uninterpretable.

Coordinate translation from ROI space back to full-frame space happens in
exactly one place, :meth:`Detector._to_full_frame`, so it can be tested.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import ModelConfig, RoiConfig

BBox = tuple[float, float, float, float]


_COMPUTE_UNITS_APPLIED: str | None = None


def _force_coreml_compute_units(name: str) -> str | None:
    """Make CoreML models load with the compute units we chose, not theirs.

    Ultralytics' CoreML backend picks ``CPU_AND_NE`` for detection models and
    offers no way to change it -- the choice is inline in the backend. On this
    hardware that is 3.6x slower than ``ALL`` for identical output, which is
    most of a frame budget thrown away.

    The only injection point is ``coremltools.models.MLModel`` itself, which
    the backend calls, so this subclasses it to pin ``compute_units``. The
    patch is process-wide and installed once: every CoreML model this process
    loads is one of ours, and the alternative -- patching around each load --
    breaks because ultralytics builds the backend lazily on first predict, not
    at construction.

    Import order matters and is not incidental: importing coremltools before
    torch segfaults the process (exit 139) on this machine, so the import here
    happens after ultralytics has already pulled torch in.

    Returns the unit name applied, or None if left alone.
    """
    global _COMPUTE_UNITS_APPLIED
    if not name or name.upper() == "DEFAULT":
        return None
    if _COMPUTE_UNITS_APPLIED is not None:
        return _COMPUTE_UNITS_APPLIED

    import coremltools as ct

    unit = getattr(ct.ComputeUnit, name.upper(), None)
    if unit is None:
        raise ValueError(
            f"Unknown coreml_compute_units {name!r}; expected one of "
            f"ALL, CPU_AND_NE, CPU_AND_GPU, CPU_ONLY, DEFAULT"
        )

    base = ct.models.MLModel

    class _PinnedComputeUnits(base):
        def __init__(self, *args, **kwargs):
            kwargs["compute_units"] = unit
            super().__init__(*args, **kwargs)

    ct.models.MLModel = _PinnedComputeUnits
    _COMPUTE_UNITS_APPLIED = name.upper()
    return _COMPUTE_UNITS_APPLIED


def normalize_imgsz(value) -> tuple[int, int]:
    """Return ``(width, height)`` from an int or an ultralytics-style pair.

    ``imgsz`` may be a single int for a square input, or ``[height, width]``
    for a rectangular one. That ordering is ultralytics' own -- ``imgsz=[736,
    1280]`` exports a 1280-wide, 736-tall model -- and it is height-first,
    which is the opposite of how every other size in this project is written.
    It is kept rather than "fixed" so the value here matches what you would
    pass to ultralytics directly, but everything downstream converts to
    (width, height) immediately so the ambiguity lives in exactly one place.

    Rectangular inputs matter because a 16:9 frame squeezed into a square
    model wastes 43.8% of every forward pass on grey letterbox padding.
    """
    if isinstance(value, bool):
        raise ValueError(f"imgsz must be an int or [height, width], got {value!r}")
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        height, width = value
        return int(width), int(height)
    raise ValueError(
        f"imgsz must be an int or a [height, width] pair, got {value!r}"
    )


def to_imgsz_arg(size: tuple[int, int]):
    """(width, height) -> what ultralytics wants: an int, or [height, width]."""
    width, height = size
    return width if width == height else [height, width]


def _engine_input_size(model_path: Path) -> tuple[int, int] | None:
    """The one input size a TensorRT engine accepts, as (width, height).

    An engine is built for a fixed input shape unless it was exported with
    dynamic axes (none of this repo's are). ultralytics prefixes its engines
    with a 4-byte length and a JSON metadata block that records the export
    ``imgsz`` as ``[height, width]``, so the size is read from there without
    deserializing the engine; an engine with no metadata is deserialized and
    its input tensor's shape read directly. Best effort: None on any failure.
    """
    try:
        data = model_path.read_bytes()
    except OSError:
        return None
    if len(data) < 4:
        return None
    meta_len = int.from_bytes(data[:4], "little")
    if 0 < meta_len < len(data) - 4:
        try:
            metadata = json.loads(data[4 : 4 + meta_len].decode("utf-8"))
            imgsz = metadata.get("imgsz")
            if isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
                height, width = imgsz
                return int(width), int(height)
            data = data[4 + meta_len :]
        except (UnicodeDecodeError, ValueError, AttributeError):
            pass
    try:
        import tensorrt as trt

        with trt.Runtime(trt.Logger(trt.Logger.ERROR)) as runtime:
            engine = runtime.deserialize_cuda_engine(data)
        if engine is None:
            return None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue
            shape = tuple(int(d) for d in engine.get_tensor_shape(name))
            if len(shape) == 4 and -1 not in shape:
                return shape[3], shape[2]
            return None
    except Exception:
        return None
    return None


def _fixed_input_size(model_path: Path) -> tuple[int, int] | None:
    """The one input size a fixed-size export accepts, as (width, height).

    A ``.mlpackage`` is exported at a single input resolution and, unless it
    was given size flexibility (none of this repo's exports were), CoreML
    refuses anything else outright:

        RuntimeError: Image size 640 x 640 not in allowed set of image sizes

    A TensorRT ``.engine`` is fixed the same way, and ultralytics asserts on
    every inference whose input does not match the engine's binding shape.

    Returns None when the model is flexible, is neither format, or cannot be
    inspected -- in all of which cases ``imgsz`` means what it says.
    """
    if model_path.suffix == ".engine":
        return _engine_input_size(model_path)
    if model_path.suffix != ".mlpackage":
        return None
    try:
        import coremltools as ct

        spec = ct.models.MLModel(
            str(model_path), compute_units=ct.ComputeUnit.CPU_ONLY
        ).get_spec()
    except Exception:
        return None  # best effort: never block startup on an inspection failure
    for descriptor in spec.description.input:
        image = descriptor.type.imageType
        width, height = int(image.width), int(image.height)
        if width and height and not image.WhichOneof("SizeFlexibility"):
            return width, height
    return None


def letterbox_geometry(
    height: int, width: int, target_h: int, target_w: int
) -> tuple[float, int, int, int, int]:
    """Where ultralytics' LetterBox puts an image: (ratio, new_w, new_h, left, top).

    Reproduces ``LetterBox(new_shape, auto=False, center=True, scaleup=True)``
    exactly -- the transform ultralytics applies before a fixed-size export
    -- so a GPU letterbox can place pixels where the CPU one would, and the
    weights see the geometry they were trained and validated with.
    """
    r = min(target_h / height, target_w / width)
    new_w, new_h = int(round(width * r)), int(round(height * r))
    dw, dh = (target_w - new_w) / 2, (target_h - new_h) / 2
    return r, new_w, new_h, int(round(dw - 0.1)), int(round(dh - 0.1))


def unletterbox_boxes(
    boxes: np.ndarray, height: int, width: int, target_h: int, target_w: int
) -> np.ndarray:
    """Map xyxy boxes from the letterboxed input back to image pixels, in place.

    The same arithmetic as ``ultralytics.utils.ops.scale_boxes`` (gain from
    the unrounded ratio, padding rounded with its -0.1 nudge, then clipped),
    so boxes from the direct path land where ultralytics would put them.
    """
    if len(boxes) == 0:
        return boxes
    gain = min(target_h / height, target_w / width)
    pad_x = round((target_w - width * gain) / 2 - 0.1)
    pad_y = round((target_h - height * gain) / 2 - 0.1)
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes[:, :4] /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes


def _engine_metadata(data: bytes) -> tuple[dict, bytes]:
    """Split ultralytics' JSON header off a serialized engine."""
    if len(data) < 4:
        return {}, data
    meta_len = int.from_bytes(data[:4], "little")
    if 0 < meta_len < len(data) - 4:
        try:
            return json.loads(data[4 : 4 + meta_len].decode("utf-8")), data[4 + meta_len :]
        except (UnicodeDecodeError, ValueError):
            pass
    return {}, data


class TrtRunner:
    """Run an ultralytics end2end TensorRT engine directly, preprocessing on the GPU.

    What ultralytics does per frame around a 5 ms TensorRT inference on an
    Orin Nano, measured: a single-threaded ``cv2.resize`` letterbox, a
    BGR→RGB / HWC→CHW copy, a pageable host→device copy, its own torch NMS
    with a GPU sync, ``get_cfg`` validation of every call's arguments, a
    fresh ``LoadPilAndNumpy`` dataset, and a ``Results`` object -- ~10 ms of
    CPU on a 1.7 GHz A78 core while the GPU idles. This class keeps only
    the parts that touch pixels: the crop goes to the GPU through a pinned
    buffer, the letterbox is a bilinear resize and a pad in torch, the
    engine returns final boxes because NMS is inside it, and the few
    surviving rows come back in one small copy.

    Requires an engine from ``scripts/export_tensorrt.py --nms``: the
    output is then ``(1, max_det, 6)`` rows of ``x1 y1 x2 y2 conf cls`` in
    input-pixel coordinates, zero-padded, with the export's conf/IoU
    already applied.
    """

    def __init__(self, engine_path: Path, device: str = "cuda:0"):
        data = engine_path.read_bytes()
        self.metadata, data = _engine_metadata(data)
        if not self.metadata:
            raise ValueError(
                f"{engine_path.name} carries no ultralytics metadata; model.backend "
                "'trt' needs an engine from scripts/export_tensorrt.py --nms."
            )
        if not self.metadata.get("args", {}).get("nms"):
            raise ValueError(
                f"{engine_path.name} was exported without NMS inside it. model.backend "
                "'trt' needs final boxes from the engine: rebuild with "
                "scripts/export_tensorrt.py --nms (and set model.path to it)."
            )
        import tensorrt as trt
        import torch

        self._torch = torch
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = self.runtime.deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError(
                f"TensorRT {trt.__version__} could not deserialize {engine_path}; "
                "engines are tied to the TensorRT version and GPU they were built on."
            )
        self.context = self.engine.create_execution_context()
        self.device = torch.device("cuda:0" if device in ("", "cuda", "gpu") else device)

        self.input_name = self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(int(d) for d in self.engine.get_tensor_shape(name))
            dtype = self._torch_dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name, self.input_shape, in_dtype = name, shape, dtype
            else:
                self.output_name, self.output_shape, out_dtype = name, shape, dtype
        if (
            self.input_name is None or self.output_name is None
            or len(self.input_shape) != 4 or -1 in self.input_shape
            or len(self.output_shape) != 3 or self.output_shape[-1] != 6
        ):
            raise ValueError(
                f"{engine_path.name}: expected one static (1,3,H,W) input and one "
                f"(1,max_det,6) output, got {self.input_shape} -> {self.output_shape}"
            )
        self.input_hw = (self.input_shape[2], self.input_shape[3])
        self.inp = torch.empty(self.input_shape, dtype=in_dtype, device=self.device)
        self.out = torch.empty(self.output_shape, dtype=out_dtype, device=self.device)
        self.context.set_tensor_address(self.input_name, self.inp.data_ptr())
        self.context.set_tensor_address(self.output_name, self.out.data_ptr())
        self._staging = None  # pinned host copy of the crop, reused across frames
        self._class_filters: dict[tuple[int, ...], "torch.Tensor"] = {}

    @staticmethod
    def _torch_dtype(np_dtype):
        import torch

        return {
            np.dtype("float32"): torch.float32,
            np.dtype("float16"): torch.float16,
            np.dtype("int32"): torch.int32,
            np.dtype("int64"): torch.int64,
            np.dtype("uint8"): torch.uint8,
        }[np.dtype(np_dtype)]

    @property
    def input_size(self) -> tuple[int, int]:
        """(width, height) the engine accepts."""
        return self.input_hw[1], self.input_hw[0]

    def preprocess(self, image: np.ndarray) -> None:
        """Letterbox ``image`` (BGR uint8) into the engine's input on the GPU."""
        torch = self._torch
        h, w = image.shape[:2]
        H, W = self.input_hw
        _, new_w, new_h, left, top = letterbox_geometry(h, w, H, W)
        if self._staging is None or tuple(self._staging.shape) != image.shape:
            self._staging = torch.empty(image.shape, dtype=torch.uint8, pin_memory=True)
        # copy_ handles the ROI view's strides; the pinned buffer makes the
        # device copy a DMA instead of a pageable staging copy.
        self._staging.copy_(torch.from_numpy(image))
        x = self._staging.to(self.device, non_blocking=True)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.inp.dtype)
        if (new_h, new_w) != (h, w):
            x = torch.nn.functional.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
        self.inp.fill_(114.0)  # ultralytics' grey border
        self.inp[:, :, top : top + new_h, left : left + new_w] = x.flip(1)  # BGR -> RGB
        self.inp.div_(255.0)

    def detect(
        self, image: np.ndarray, conf: float, classes: list[int] | None = None
    ) -> np.ndarray:
        """Final boxes for one BGR image: ``(n, 6)`` of x1 y1 x2 y2 conf cls, image pixels."""
        torch = self._torch
        self.preprocess(image)
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        out = self.out[0]
        keep = out[:, 4] > conf  # ultralytics keeps strictly above conf_thres
        if classes:
            key = tuple(sorted(classes))
            wanted = self._class_filters.get(key)
            if wanted is None:
                wanted = torch.tensor(key, device=self.device, dtype=out.dtype)
                self._class_filters[key] = wanted
            keep &= (out[:, 5:6] == wanted).any(1)
        dets = out[keep].float().cpu().numpy()  # the one sync per frame
        h, w = image.shape[:2]
        return unletterbox_boxes(dets, h, w, self.input_hw[0], self.input_hw[1])


class ByteTrackDirect:
    """ultralytics' own BYTETracker, driven the way its predictor drives it.

    Same tracker yaml, same ``frame_rate=30`` construction, updated on every
    frame including empty ones (so lost tracks age), GMC fed the same crop.
    """

    def __init__(self, tracker_path: Path):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import YAML, IterableSimpleNamespace

        cfg = IterableSimpleNamespace(**YAML.load(str(tracker_path)))
        if cfg.tracker_type != "bytetrack":
            raise ValueError(
                f"model.backend 'trt' supports tracker_type bytetrack, got {cfg.tracker_type!r}"
            )
        self._tracker = BYTETracker(args=cfg, frame_rate=30)

    def update(self, dets: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Rows of ``x1 y1 x2 y2 id conf cls idx`` for the tracks alive this frame."""
        from ultralytics.engine.results import Boxes

        boxes = Boxes(np.ascontiguousarray(dets, dtype=np.float32), image.shape[:2])
        return self._tracker.update(boxes, image)


@dataclass
class Detection:
    """One detected object, in full-frame pixel coordinates."""

    xyxy: BBox
    conf: float
    cls: int
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class Roi:
    """A rectangular crop of the frame, derived from a normalized polygon."""

    def __init__(self, config: RoiConfig, frame_width: int, frame_height: int):
        self.enabled = config.enabled and bool(config.polygon)
        self.scale = config.scale
        if not self.enabled:
            self.x1, self.y1 = 0, 0
            self.x2, self.y2 = frame_width, frame_height
        else:
            xs = [p[0] * frame_width for p in config.polygon]
            ys = [p[1] * frame_height for p in config.polygon]
            self.x1 = max(0, int(min(xs)))
            self.y1 = max(0, int(min(ys)))
            self.x2 = min(frame_width, int(max(xs)))
            self.y2 = min(frame_height, int(max(ys)))
            if self.x2 <= self.x1 or self.y2 <= self.y1:
                raise ValueError(
                    "ROI polygon collapses to an empty rectangle; recalibrate it."
                )

    def crop(self, image: np.ndarray) -> np.ndarray:
        region = image[self.y1 : self.y2, self.x1 : self.x2]
        if self.scale != 1.0:
            region = cv2.resize(
                region,
                (int(region.shape[1] * self.scale), int(region.shape[0] * self.scale)),
                interpolation=cv2.INTER_AREA,
            )
        return region

    def to_full_frame(self, box: BBox) -> BBox:
        """Map a box from cropped/scaled space back to full-frame pixels."""
        x1, y1, x2, y2 = box
        if self.scale != 1.0:
            x1, y1, x2, y2 = (c / self.scale for c in (x1, y1, x2, y2))
        return (x1 + self.x1, y1 + self.y1, x2 + self.x1, y2 + self.y1)


class Detector:
    """Stateful YOLO tracker over a fixed frame geometry."""

    # Class-level defaults so a Detector assembled without __init__ (the
    # tests build them around fake models) behaves as the ultralytics path.
    backend = "ultralytics"
    runner = None
    tracker = None
    second_runner = None

    def __init__(
        self,
        model_config: ModelConfig,
        roi_config: RoiConfig,
        frame_width: int,
        frame_height: int,
    ):
        from ultralytics import YOLO  # imported lazily: heavy and optional for tests

        # ultralytics' import just set cv2's thread count to 0. Put it back
        # if the config asks: the letterbox resize it runs on every frame
        # is the biggest CPU cost on the frame loop on a Jetson.
        if model_config.cv2_threads > 0:
            cv2.setNumThreads(int(model_config.cv2_threads))

        model_path = Path(model_config.path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        tracker_path = Path(model_config.tracker)
        if not tracker_path.exists():
            raise FileNotFoundError(f"Tracker config not found: {tracker_path}")

        self.config = model_config
        self.roi = Roi(roi_config, frame_width, frame_height)

        # Must happen before any model loads: the patch only affects models
        # constructed after it is installed.
        self.compute_units = None
        if model_path.suffix == ".mlpackage":
            self.compute_units = _force_coreml_compute_units(
                model_config.coreml_compute_units
            )

        self.backend = model_config.backend
        if self.backend not in ("ultralytics", "trt"):
            raise ValueError(
                f"Unknown model.backend {self.backend!r}; expected 'ultralytics' or 'trt'."
            )
        self.runner = self.tracker = self.second_runner = None
        if self.backend == "trt":
            if model_path.suffix != ".engine":
                raise ValueError("model.backend 'trt' needs a TensorRT .engine as model.path")
            self.runner = TrtRunner(model_path, model_config.device)
            self.tracker = ByteTrackDirect(tracker_path)
            self.model = None
        else:
            self.model = YOLO(str(model_path))

        # Second stage may run its own, smaller export. Sharing one YOLO object
        # across the two passes would also be a threading hazard if the second
        # stage ever moves off the frame loop, so keep them separate objects.
        self.second_stage = None
        self.crops_skipped = 0
        self.two_stage_errors = 0
        self.two_stage_last_error: str | None = None
        self.warnings: list[str] = []

        # Effective sizes, which are not necessarily the configured ones. A
        # CoreML export accepts exactly one input size and *raises* on any
        # other, so an imgsz the model cannot honour is not a slow path or a
        # quality trade -- it is a hard failure on every inference. Resolve it
        # here, once, and say so, rather than discovering it per frame.
        self.imgsz_wh = normalize_imgsz(model_config.imgsz)
        self.two_stage_imgsz_wh = normalize_imgsz(model_config.two_stage_imgsz)

        fixed = _fixed_input_size(model_path)
        if fixed is not None and fixed != self.imgsz_wh:
            fw, fh = fixed
            cw, ch = self.imgsz_wh
            self.warnings.append(
                f"imgsz={cw}x{ch} but {model_path.name} is exported at a fixed "
                f"{fw}x{fh} and rejects any other size. Running at {fw}x{fh}. "
                f"Set model.imgsz={to_imgsz_arg(fixed)} so the config describes "
                f"what happens."
            )
            self.imgsz_wh = fixed
        self.imgsz = to_imgsz_arg(self.imgsz_wh)

        if model_config.two_stage:
            second_path = Path(model_config.two_stage_model or model_config.path)
            if not second_path.exists():
                raise FileNotFoundError(f"Two-stage model not found: {second_path}")
            if self.backend == "trt":
                self.second_runner = (
                    self.runner if second_path == model_path
                    else TrtRunner(second_path, model_config.device)
                )
                self.second_stage = None
            else:
                self.second_stage = (
                    self.model if second_path == model_path else YOLO(str(second_path))
                )
            second_fixed = (
                fixed if second_path == model_path else _fixed_input_size(second_path)
            )
            if second_fixed is not None and second_fixed != self.two_stage_imgsz_wh:
                fw, fh = second_fixed
                cw, ch = self.two_stage_imgsz_wh
                self.warnings.append(
                    f"two_stage_imgsz={cw}x{ch} but {second_path.name} is exported "
                    f"at a fixed {fw}x{fh} and rejects any other size. Running crops "
                    f"at {fw}x{fh}, which is not the speedup two-stage is for -- "
                    f"export a {cw}x{ch} model and set model.two_stage_model to it."
                )
                self.two_stage_imgsz_wh = second_fixed
        self.two_stage_imgsz = to_imgsz_arg(self.two_stage_imgsz_wh)

    def warmup(self, frame_width: int, frame_height: int) -> float:
        """Run one throwaway inference so the first real frame isn't slow.

        Loading the CoreML model and running its first inference costs a few
        seconds; paying that on the first frame of a race means the pipeline
        is blind while it happens. Uses ``predict`` rather than ``track`` so
        the tracker's state isn't seeded with detections from a blank frame.
        """
        started = time.time()
        blank = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        if self.backend == "trt":
            try:
                self.runner.detect(self.roi.crop(blank), self.config.conf)
                if self.second_runner is not None and self.second_runner is not self.runner:
                    crop = np.zeros((frame_height // 3, frame_height // 6, 3), dtype=np.uint8)
                    self.second_runner.detect(crop, self.config.conf)
            except Exception:
                pass
            return time.time() - started
        try:
            self.model.predict(
                blank,
                conf=self.config.conf,
                imgsz=self.imgsz,
                device=self.config.device,
                half=self.config.half,
                verbose=False,
            )
        except Exception:
            pass
        if self.second_stage is not None and self.second_stage is not self.model:
            # Person-shaped, not frame-shaped: the second stage never sees a
            # full frame, and the first inference cost is paid per input shape.
            crop = np.zeros((frame_height // 3, frame_height // 6, 3), dtype=np.uint8)
            try:
                self.second_stage.predict(
                    crop,
                    conf=self.config.conf,
                    imgsz=self.two_stage_imgsz,
                    device=self.config.device,
                    half=self.config.half,
                    verbose=False,
                )
            except Exception:
                pass
        return time.time() - started

    def track(self, image: np.ndarray) -> list[Detection]:
        """Run detection + tracking on one frame.

        Returns detections in full-frame coordinates. Untracked detections keep
        ``track_id=None`` rather than being dropped, so bib boxes (which are not
        tracked) still reach the OCR stage.
        """
        region = self.roi.crop(image)
        if self.backend == "trt":
            return self._track_direct(region)
        results = self.model.track(
            region,
            persist=True,
            tracker=self.config.tracker,
            classes=[self.config.person_class, self.config.bib_class],
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=self.imgsz,
            device=self.config.device,
            half=self.config.half,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections: list[Detection] = []
        for box in boxes:
            raw = tuple(float(v) for v in box.xyxy[0].tolist())
            track_id = None if box.id is None else int(box.id.item())
            detections.append(
                Detection(
                    xyxy=self.roi.to_full_frame(raw),
                    conf=float(box.conf.item()),
                    cls=int(box.cls.item()),
                    track_id=track_id,
                )
            )
        return detections

    def _track_direct(self, region: np.ndarray) -> list[Detection]:
        """The ``track()`` contract on the direct backend.

        Mirrors ultralytics' tracker callback: when any track is alive the
        frame's detections *are* the tracks (Kalman-smoothed boxes, ids);
        when none is, the raw boxes are kept without ids, which is how bib
        boxes below the tracker's thresholds still reach OCR.
        """
        dets = self.runner.detect(
            region, self.config.conf,
            classes=[self.config.person_class, self.config.bib_class],
        )
        tracks = self.tracker.update(dets, region)
        detections: list[Detection] = []
        if len(tracks):
            for x1, y1, x2, y2, track_id, conf, cls, _ in tracks:
                detections.append(Detection(
                    xyxy=self.roi.to_full_frame((float(x1), float(y1), float(x2), float(y2))),
                    conf=float(conf), cls=int(cls), track_id=int(track_id),
                ))
        else:
            for x1, y1, x2, y2, conf, cls in dets:
                detections.append(Detection(
                    xyxy=self.roi.to_full_frame((float(x1), float(y1), float(x2), float(y2))),
                    conf=float(conf), cls=int(cls), track_id=None,
                ))
        return detections

    @staticmethod
    def merge(primary: list[Detection], extra: list[Detection], iou: float = 0.5):
        """Add `extra` detections, dropping ones that duplicate `primary`.

        The two-stage pass re-examines a region the full-frame pass already
        looked at, so any bib visible at both scales gets found twice. Left
        unmerged that produces stacked near-identical boxes: noise for a human
        reviewer, and duplicated supervision if the output is trained on.
        """
        def iou_of(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        merged = list(primary)
        for candidate in extra:
            if any(
                candidate.cls == kept.cls and iou_of(candidate.xyxy, kept.xyxy) >= iou
                for kept in merged
            ):
                continue
            merged.append(candidate)
        return merged

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect on a single image with no tracking state.

        ``track()`` runs ByteTrack with ``persist=True``, which is right for
        consecutive video frames and badly wrong for a folder of unrelated
        stills: the tracker tries to associate each image's detections with
        tracks from the previous, unrelated image and drops the ones that don't
        match. Anything iterating over independent images wants this instead.
        """
        region = self.roi.crop(image)
        if self.backend == "trt":
            return [
                Detection(
                    xyxy=self.roi.to_full_frame((float(x1), float(y1), float(x2), float(y2))),
                    conf=float(conf), cls=int(cls), track_id=None,
                )
                for x1, y1, x2, y2, conf, cls in self.runner.detect(region, self.config.conf)
            ]
        results = self.model.predict(
            region,
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=self.imgsz,
            device=self.config.device,
            half=self.config.half,
            verbose=False,
        )
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        return [
            Detection(
                xyxy=self.roi.to_full_frame(
                    tuple(float(v) for v in box.xyxy[0].tolist())
                ),
                conf=float(box.conf.item()),
                cls=int(box.cls.item()),
                track_id=None,
            )
            for box in boxes
        ]

    def bibs_in_people(
        self, image: np.ndarray, people: list[Detection]
    ) -> list[Detection]:
        """Second-stage pass: look for a bib inside each person's crop.

        A bib is tiny once a 1920px frame is squeezed into the model's 640px
        input -- median 15px wide in finish-line footage. Cropping to a single
        runner and feeding that to the same input gives the bib hundreds of
        pixels instead, which is a far larger effective upscale than raising
        the whole frame's resolution, at a fraction of the cost.

        Coordinates come back translated to full-frame space. Returns bib
        detections only; person boxes from the second pass are discarded
        because the first pass already owns tracking.

        Cost is one inference *per person*, so ``two_stage_max_crops`` bounds
        the worst case. When a pack exceeds it the largest boxes win -- they
        are the runners nearest the camera, hence nearest the line, hence the
        ones about to finish. Taking the list in detector order instead would
        make the choice arbitrary.
        """
        if not people:
            return []
        crops, origins = [], []
        height, width = image.shape[:2]
        ranked = sorted(
            people,
            key=lambda p: (p.xyxy[2] - p.xyxy[0]) * (p.xyxy[3] - p.xyxy[1]),
            reverse=True,
        )
        self.crops_skipped += max(0, len(ranked) - self.config.two_stage_max_crops)
        for person in ranked[: self.config.two_stage_max_crops]:
            x1, y1, x2, y2 = person.xyxy
            pad_x = (x2 - x1) * self.config.two_stage_padding
            pad_y = (y2 - y1) * self.config.two_stage_padding
            cx1 = max(0, int(x1 - pad_x))
            cy1 = max(0, int(y1 - pad_y))
            cx2 = min(width, int(x2 + pad_x))
            cy2 = min(height, int(y2 + pad_y))
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            crops.append(image[cy1:cy2, cx1:cx2])
            origins.append((cx1, cy1, cx2 - cx1, cy2 - cy1))
        if not crops:
            return []

        # One crop at a time rather than one batched call. The CoreML export is
        # fixed at batch=1, and handing it a list of differently-sized crops
        # makes ultralytics return fewer results than inputs and then index off
        # the end. Batching bought nothing here anyway: a batch-1 backend runs
        # them sequentially regardless.
        model = self.second_stage or self.model
        found: list[Detection] = []
        if self.backend == "trt":
            runner = self.second_runner or self.runner
            for crop, (ox, oy, cw, ch) in zip(crops, origins):
                try:
                    dets = runner.detect(crop, self.config.conf, classes=[self.config.bib_class])
                except Exception as exc:
                    self.two_stage_errors += 1
                    self.two_stage_last_error = f"{type(exc).__name__}: {exc}"
                    continue
                for bx1, by1, bx2, by2, conf, _ in dets:
                    found.append(Detection(
                        xyxy=(float(bx1) + ox, float(by1) + oy, float(bx2) + ox, float(by2) + oy),
                        conf=float(conf), cls=self.config.bib_class, track_id=None,
                    ))
            return found
        for crop, (ox, oy, cw, ch) in zip(crops, origins):
            try:
                results = model.predict(
                    crop,
                    conf=self.config.conf,
                    iou=self.config.iou,
                    imgsz=self.two_stage_imgsz,
                    device=self.config.device,
                    half=self.config.half,
                    verbose=False,
                )
            except Exception as exc:
                # Never silent. This swallowed a RuntimeError on *every* crop
                # when two_stage_imgsz did not match the CoreML export's fixed
                # input, so the second stage found nothing at all while the
                # config, the runbook and second_stage_bibs=0 all read as
                # "enabled, just not finding extra bibs". A feature that fails
                # closed and reports success is worse than one that is off.
                self.two_stage_errors += 1
                self.two_stage_last_error = f"{type(exc).__name__}: {exc}"
                continue
            if not results:
                continue
            boxes = getattr(results[0], "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                if int(box.cls.item()) != self.config.bib_class:
                    continue
                bx1, by1, bx2, by2 = (float(v) for v in box.xyxy[0].tolist())
                found.append(
                    Detection(
                        xyxy=(bx1 + ox, by1 + oy, bx2 + ox, by2 + oy),
                        conf=float(box.conf.item()),
                        cls=self.config.bib_class,
                        track_id=None,
                    )
                )
        return found

    def split(
        self, detections: list[Detection]
    ) -> tuple[list[Detection], list[Detection]]:
        """Partition detections into (tracked people, bibs).

        Unlike the legacy pipeline there is no geometric gate here. Every person
        in frame is tracked; the finish line alone decides who finished. The old
        code filtered ``tracked_persons`` to a hardcoded wedge, so a racer
        outside it could never finish no matter how clearly they were detected.
        """
        people = [
            d
            for d in detections
            if d.cls == self.config.person_class and d.track_id is not None
        ]
        bibs = [d for d in detections if d.cls == self.config.bib_class]
        return people, bibs
