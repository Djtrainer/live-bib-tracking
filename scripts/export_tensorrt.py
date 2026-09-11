#!/usr/bin/env python3
"""Export the trained weights to a TensorRT engine at the size the pipeline runs.

The Jetson counterpart of ``export_coreml.py``. Everything that script says
about *why* it is a script applies here, plus two things TensorRT adds:

* An ``.engine`` is built for **this GPU, this TensorRT version and one
  input size**. It is not portable: build it on the board that will race,
  and rebuild it after a JetPack upgrade. ``models/`` is gitignored, so the
  engine never travels through git anyway.

* Building takes minutes and a lot of memory on an 8 GB Orin Nano (the ONNX
  export runs through torch first, then TensorRT tries kernel tactics
  against a workspace). Build one engine at a time, keep ``--workspace``
  modest, and if the process dies without a traceback it was the OOM
  killer: check ``dmesg`` and free memory before retrying.

Like the CoreML script, this never exports in place -- ultralytics writes
``<name>.onnx`` and ``<name>.engine`` next to the source ``.pt`` -- so the
weights are copied to a scratch directory first and only the finished
engine is moved to its destination.

    .venv/bin/python scripts/export_tensorrt.py --size 512 928
    .venv/bin/python scripts/export_tensorrt.py --size 640 640 --out models/exports/trt_640x640_fp16.engine
    .venv/bin/python scripts/export_tensorrt.py --size 512 928 --fp32

``--size`` is HEIGHT then WIDTH, ultralytics' ordering, the same one
``model.imgsz`` uses in the config. Nothing reads an engine's size back
into the config for you: set ``model.imgsz`` by hand to what you exported
(race_cv checks the engine's fixed size at startup and prints a CONFIG
MISMATCH if the two disagree, but the pipeline runs at the engine's size
either way).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path


def engine_input(path: Path) -> tuple[tuple[int, ...], str, dict] | None:
    """(input shape, input dtype, ultralytics metadata) of a built engine."""
    import tensorrt as trt

    data = path.read_bytes()
    metadata: dict = {}
    meta_len = int.from_bytes(data[:4], "little")
    if 0 < meta_len < len(data):
        try:
            metadata = json.loads(data[4 : 4 + meta_len].decode("utf-8"))
            data = data[4 + meta_len :]
        except (UnicodeDecodeError, ValueError):
            metadata = {}
    logger = trt.Logger(trt.Logger.ERROR)
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(data)
    if engine is None:
        return None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            return (
                tuple(int(d) for d in engine.get_tensor_shape(name)),
                str(engine.get_tensor_dtype(name)),
                metadata,
            )
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export weights to a TensorRT engine")
    parser.add_argument(
        "--weights", default="models/gpu_runs/yolo11n_1280/weights/best.pt")
    parser.add_argument(
        "--size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=[512, 928],
        help="Input size, HEIGHT then WIDTH (ultralytics' ordering). "
             "Each must be a multiple of 32.")
    parser.add_argument(
        "--out", default=None,
        help="Destination .engine (default: models/exports/trt_<W>x<H>_<fp16|fp32>.engine)")
    parser.add_argument(
        "--fp32", action="store_true",
        help="Build without FP16. The default is FP16, which is what the Orin's "
             "tensor cores are for; FP32 exists to measure the difference.")
    parser.add_argument(
        "--nms", action="store_true",
        help="Bake NMS into the engine (ultralytics nms=True, an 'end2end' export). "
             "The engine then returns final boxes and ultralytics skips its own "
             "torch NMS, which costs ~4 ms per frame on the Orin's CPU-side "
             "launch overhead. conf/iou are frozen at export from --conf/--iou.")
    parser.add_argument("--conf", type=float, default=0.25, help="NMS confidence baked in with --nms")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU baked in with --nms")
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset for the intermediate export. Pinned because ultralytics "
             "8.3.176's 'latest opset' probe reads torch.onnx attributes that torch "
             "2.9 no longer exposes and falls back to 10, which cannot express "
             "torchvision::nms (opset 11+). 17 is what TensorRT 10 expects.")
    parser.add_argument(
        "--workspace", type=float, default=2.0,
        help="TensorRT workspace in GiB. Bounded on purpose: the default "
             "(auto) lets the builder take everything on an 8 GB board.")
    parser.add_argument("--device", default="0", help="CUDA device for the export")
    parser.add_argument("--force", action="store_true", help="Overwrite the destination")
    args = parser.parse_args(argv)

    height, width = args.size
    for v in (height, width):
        if v % 32:
            print(f"size {v} is not a multiple of 32", file=sys.stderr)
            return 2
    weights = Path(args.weights)
    if not weights.exists():
        print(f"weights not found: {weights}", file=sys.stderr)
        return 1
    precision = "fp32" if args.fp32 else "fp16"
    suffix = "_nms" if args.nms else ""
    out = Path(args.out or f"models/exports/trt_{width}x{height}_{precision}{suffix}.engine")
    if out.exists() and not args.force:
        print(f"{out} exists; pass --force to replace it", file=sys.stderr)
        return 1

    import torch
    from ultralytics import YOLO

    # torch >= 2.9 defaults torch.onnx.export(dynamo=True), the new exporter
    # that needs onnxscript and traces the graph differently. ultralytics
    # 8.3.176 -- the race-validated version, pinned here on purpose --
    # predates that default and calls torch.onnx.export without saying
    # which exporter it wants. Pin the legacy TorchScript exporter, the one
    # every export of these weights has gone through so far.
    _torch_onnx_export = torch.onnx.export

    def _legacy_exporter(*args, **kwargs):
        kwargs["dynamo"] = False
        return _torch_onnx_export(*args, **kwargs)

    torch.onnx.export = _legacy_exporter

    # Never export in place: see the module docstring.
    with tempfile.TemporaryDirectory(prefix="trt_export_") as scratch:
        staged = Path(scratch) / "m.pt"
        shutil.copy2(weights, staged)
        print(f"exporting {weights} at {width}x{height} (h={height}, w={width}) "
              f"{precision}, workspace {args.workspace:g} GiB ...", flush=True)
        produced = YOLO(str(staged)).export(
            format="engine",
            imgsz=[height, width],
            half=not args.fp32,
            device=args.device,
            batch=1,
            dynamic=False,
            simplify=True,
            workspace=args.workspace,
            opset=args.opset,
            nms=args.nms,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), str(out))

    # Confirm the artifact says what we asked for; a silent h/w swap here
    # would fail on the first frame on race day.
    try:
        found = engine_input(out)
    except ImportError:
        print(f"wrote {out}  (tensorrt not importable here to verify input size)")
        found = None
    if found is not None:
        shape, dtype, metadata = found
        got = (shape[-1], shape[-2])
        if got != (width, height):
            print(f"engine input is {got[0]}x{got[1]}, expected {width}x{height}",
                  file=sys.stderr)
            return 1
        print(f"wrote {out}  (input {shape} {dtype}, "
              f"{out.stat().st_size / 1e6:.1f} MB, metadata imgsz {metadata.get('imgsz')}, "
              f"nms {metadata.get('args', {}).get('nms')})")
    print(f"\nconfig/race_cv.jetson.yaml:\n  model:\n    path: {out}\n"
          f"    imgsz: [{height}, {width}]\n    device: cuda:0\n    half: {'false' if args.fp32 else 'true'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
