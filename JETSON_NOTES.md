# Jetson Orin Nano evaluation notes

Running log of every measurement taken on the Jetson, with the power mode
and the exact command that produced it. Branch `jetson-eval`.

## Board

Verified 2026-09-11:

| item | value | how |
|---|---|---|
| module | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super | `/sys/devices/soc0/machine` |
| L4T | R39.2.1 (JetPack 7.2.1-b49) | `cat /etc/nv_tegra_release`, `dpkg -l nvidia-jetpack` |
| CUDA | 13.2 (nvcc V13.2.86; driver 595.78, CUDA 13.2 per nvidia-smi) | `nvcc --version`, `nvidia-smi` |
| cuDNN / TensorRT | 9.20.0 / 10.16.2.10 (`python3-libnvinfer`) | `ls /usr/lib/aarch64-linux-gnu` |
| Python | 3.12.3 system | `python3 --version` |
| OpenCV | 4.8.0 system (`nvidia-opencv` 7.2.1), GStreamer YES, FFMPEG YES | `cv2.getBuildInformation()` |
| RAM | 7485 MB, **no swap/zram**; ~3.1 GB in use at idle by the desktop session (gnome-shell, cursor-server, update-manager, dockerd) | `free -m`, `tegrastats` |
| power mode | `nvpmodel -q` -> **MAXN_SUPER (mode 2)**, the highest mode in `/etc/nvpmodel.conf` (15W, 25W, MAXN_SUPER) | `nvpmodel -q` |
| clocks | GPU 1020 MHz (its `max_freq`) and all six CPUs at 1728 MHz read at idle, consistent with `jetson_clocks` already applied at boot. **`sudo` needs a password in this session, so `jetson_clocks` / `jetson_clocks --show` could not be run or confirmed.** Clocks during long runs are read off `tegrastats` instead. | `/sys/class/devfreq/17000000.gpu/cur_freq`, `scaling_cur_freq` |
| camera | **none attached** (no `/dev/video*`; USB bus holds a hub, keyboard, Bluetooth, flash drive) | `ls /dev/video*`, `lsusb` |
| hardware decode | GStreamer `nvv4l2decoder` (H.264/HEVC), `nvjpegdec` (MJPEG), `nvvidconv` present | `gst-inspect-1.0` |
| clips | 16 x 1920x1080 HEVC `.mov` in `data/raw/` (27.0 min); 15 at 30 fps, one 2026-09-05 clip at 60 fps; 14 are listed in `smoke_test.yaml` | OpenCV `CAP_PROP_*` |

### Contradictions with the brief

1. **No `jp7` channel on pypi.jetson-ai-lab.io.** The index root lists
   `jp6/cu126`, `jp6/cu128`, `jp6/cu129`, `sbsa/cu130`, `sbsa/dev`;
   `jp7/cu130` and `jp7/cu132` return empty package lists. The only CUDA 13
   channel is `sbsa/cu130` (torch 2.9.0/2.9.1/2.10.0/2.11.0 and torchvision
   0.24.1/0.25.0, all cp312 `linux_aarch64`). Whether those SBSA builds
   carry sm_87 kernels for Orin is tested below.
2. **`sudo` needs a password**, so `jetson_clocks` and `nvpmodel -m` are
   not available to this session. All figures are at MAXN_SUPER under DVFS.
3. **No camera attached**, so `v4l2-ctl --list-formats-ext` has nothing to
   list. The capture-side question is answered from the decode elements
   present and a prototype pipeline string, not a measurement.

## Environment

```bash
python3 -m venv --system-site-packages .venv        # sees JetPack's cv2, tensorrt, numpy 1.26.4
```

pip gotcha: with `--index-url <jetson> --extra-index-url pypi.org`, pip
picked PyPI's `manylinux_2_28_aarch64` **CPU** torch 2.9.1 over the Jetson
channel's `linux_aarch64` wheel of the same version (manylinux tags rank
higher). `torch.__version__` read `2.9.1+cpu`, `cuda available False`.
Install torch/torchvision from the Jetson index alone with `--no-deps`:

```bash
.venv/bin/pip install --no-deps --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130 torch==2.9.1 torchvision==0.24.1
.venv/bin/pip install jinja2 typing-extensions filelock sympy networkx fsspec
```

ultralytics and easyocr both declare a pip OpenCV dependency
(`opencv-python` / `opencv-python-headless`) that would shadow JetPack's
GStreamer-enabled cv2, so they are installed `--no-deps` and their other
dependencies explicitly. numpy stays at the system 1.26.4: JetPack's cv2
bindings were built against numpy 1.x.

```bash
.venv/bin/pip install --no-deps ultralytics==8.3.176 easyocr==1.7.2
.venv/bin/pip install pillow tqdm psutil py-cpuinfo ultralytics-thop matplotlib seaborn onnx onnxslim \
    scikit-image python-bidi shapely pyclipper ninja \
    "fastapi>=0.115,<1" "starlette<1" "uvicorn>=0.30" "python-multipart>=0.0.9" "httpx>=0.27" "pytest>=8.0" \
    python-dotenv websockets "numpy<2"
```

### The torch wheel problem (measured, not assumed)

`sbsa/cu130` torch 2.9.1 imports (after linking the NVPL and cuDSS libs
it needs from the `nvpl-blas`, `nvpl-lapack`, `nvidia-cudss-cu13` PyPI
wheels into `torch/lib`), and `torch.cuda.is_available()` is True, but:

```
arch list ['sm_110', 'sm_121']
device Orin capability (8, 7)
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

The channel is built for Thor (sm_110) and GB10 (sm_121) only. There is no
prebuilt CUDA-13 torch for Orin on pypi.jetson-ai-lab.io as of 2026-09-11.

**Resolution: PyTorch's official `cu130` aarch64 wheels.** Their build sets
`TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0;11.0;12.0+PTX"` (`.ci/aarch64_linux/aarch64_ci_build.sh`
at v2.9.1), and sm_80 SASS executes on sm_87 by CUDA binary compatibility.
JetPack 7.2's CUDA is itself laid out as `targets/sbsa-linux`, so the SBSA
cuBLAS/cuDNN these wheels pull in are the same builds the board ships.

```bash
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple "torch==2.9.1+cu130"
.venv/bin/pip install --no-deps --index-url https://download.pytorch.org/whl/cu130 "torchvision==0.24.1"
```

Verified (`scratchpad/bench_torch.py`, MAXN_SUPER, warm, 50 iterations):

| op | torch's pip cuBLAS 13.0 / cuDNN 9.13 | JetPack cuBLAS 13.4 / cuDNN 9.20 via `LD_PRELOAD` |
|---|---|---|
| fp16 matmul 2048² | 1.95 ms (8.8 TFLOPS) | 1.98 ms (8.7 TFLOPS) |
| fp32 matmul 2048² | 12.79 ms (1.3 TFLOPS) | 12.78 ms |
| fp16 conv 3→64 @512x928 | 5.64 ms | 5.63 ms |
| resnet18 fp16 / fp32 @224 | 7.22 / 6.79 ms | 7.33 / 6.95 ms |
| BiLSTM 2x256, seq 64 | 7.00 ms | 6.68 ms |

Identical within noise, so the plain pip install is used with no library
overrides. `torch.cuda.get_arch_list()` reports `sm_80 … sm_120, compute_120`;
`torch.cuda.get_device_capability()` is (8, 7). (The `sbsa/cu130` Jetson-channel
wheel is uninstalled; nothing from pypi.jetson-ai-lab.io is in the venv.)

## 1. Environment sanity

`.venv/bin/python -m pytest tests/ -q` -> **240 passed, 2 skipped** (the two
CoreML skips). The brief's 238/2 assumed a built frontend: three
`tests/test_api.py` site-serving tests skip as "frontend not built" until
`src/frontend/dist/` exists. Node is not installed on the board, so a
user-local Node 22 tarball (in the session scratchpad, not on the system)
ran `npm ci && npm run build` once. The extra two passes are the new
`TestEngineInputDetection` tests for reading a `.engine`'s fixed input size.

## Engines

Built on this board with `scripts/export_tensorrt.py` (ultralytics 8.3.176
ONNX via the legacy TorchScript exporter, then TensorRT 10.16.2, batch 1,
static shape, 2 GiB workspace). Engines live in `models/exports/`
(gitignored) and are named `trt_<W>x<H>_<fp16|fp32>.engine`.

| engine | command | build time | peak RAM during build (tegrastats) |
|---|---|---|---|
| `trt_928x512_fp16.engine` (8.6 MB) | `YOLO_AUTOINSTALL=false .venv/bin/python scripts/export_tensorrt.py --size 512 928` | 334 s engine generation, 348 s total | 6168 MB of 7485 (desktop session idle ~3.1-3.9 GB of that), GR3D 99%, tj 58 C |
| `trt_1280x736_fp16.engine` (8.6 MB) | `... --size 736 1280` | 359 s | 6390 MB, tj 59 C |
| `trt_1376x768_fp16.engine` (7.9 MB) | `... --size 768 1376` (the ROI crop is 1383x756; nearest multiples of 32) | 355 s | 6390 MB |
| `trt_1920x1088_fp16.engine` (8.2 MB) | `... --size 1088 1920` | 399 s | **6802 MB of 7485**, tj 61 C -- the closest call; the builder itself sat at ~2.4 GB RSS on top of the desktop session |
| `trt_640x640_fp16.engine` (8.6 MB) | `... --size 640 640` (second stage) | 331 s | 6802 MB window |
| `trt_928x512_fp32.engine` (12.8 MB) | `... --size 512 928 --fp32` | 115 s | -- |

All built with the desktop session (gnome-shell, Cursor server, update
manager) resident: ~3.5-4.9 GB in use before the builder started, no swap.
Nothing was OOM-killed, but 1920x1088 left under 700 MB free. On a board
that runs headless there would be 3 GB more headroom; on this one, build
one engine at a time (the chain script did) and do not run the race
stack while building.

Notes on the engine: the input binding stays `FLOAT` (fp32) with FP16
layers inside, which is how ultralytics builds `half=True` engines;
AutoBackend therefore reports `fp16=False` and feeds fp32 input, and
`model.half: true` in the config is harmless. `YOLO_AUTOINSTALL=false`
stops ultralytics from pip-installing `onnxruntime-gpu` and downgrading
`onnx` inside the venv during export (its `check_requirements` pins
`onnx<1.18`; onnx 1.22 exports fine).

### One clip end to end on CUDA (step 1)

Clip `2025-10-04 10-25-15` (50.6 s, 1517 frames, bib 121 expected at 29 s),
928x512 FP16 engine on the crop, roster loaded, MAXN_SUPER:

```bash
rm -f data/results/events.jsonl
PYTHONPATH=src YOLO_AUTOINSTALL=false .venv/bin/python -m race_cv.run \
  --source "data/raw/Camo Recording 2025-10-04 10-25-15.mov" \
  --config config/race_cv.jetson.yaml --no-api --roster data/raw/roster_example.csv --realtime
```

```
Loading models/exports/trt_928x512_fp16.engine for TensorRT inference...
Detector warm-up: 4.8s
OCR warm-up: 5.7s
health | processed 1505 (29.9 fps) | paced out 0 | source dropped 12 | finishers 1 (unknown bib 0) | delivered 1 | pending 0 | ocr read 5
All 1 finish events delivered.
```

The event in `events.jsonl`: bib **121**, 5 votes, score 0.998, in roster.
`tegrastats` over the run: CPU 24% of 6 cores at 1728 MHz, GR3D 16% mean
(36% max), 8.8 W mean, tj 55 C. EasyOCR reports `device cuda`.

The same clip flat out (`scripts/replay.py`, target_fps 0): **39.6 fps end
to end** (decode included) -- and the offline OCR starvation the runbook
warns about is *worse* here: 2 reads, 22 dropped, 1 late, bib unresolved.
Bib accuracy is scored in real time only, as the brief says.

`PYTHONPATH=src` is needed for `python -m race_cv.run` outside the launcher
(the launcher sets it itself); `YOLO_AUTOINSTALL=false` keeps ultralytics
from pip-installing into the venv mid-run (it wanted `lap`, now pinned in
`requirements-jetson.txt`).

### Where a frame goes at 928x512 (the surprise)

`scripts/bench_detector.py` on the reference clip `14-48-12`, frames
930-1529 (the 36 s crossing), race config, OCR worker running:

```
decode    median  5.8 ms   (HEVC 1080p, OpenCV/FFmpeg, CPU)
detector  median 19.6 ms  p90 21.2   -> 51 fps
loop      median 19.7 ms  p90 21.3   -> 51 fps
end to end (decode + loop): 38.2 fps
```

Inside `model.track()` (`scratchpad/profile_track.py`, ultralytics'
`results.speed` plus wall timers, 300 frames):

| stage | ms/frame | notes |
|---|---|---|
| ROI crop | 0.02 | numpy view |
| **preprocess** | **8.08** | ultralytics `LetterBox` resize + BGR→RGB transpose + `ascontiguousarray` + host→device copy, all on one CPU core |
| inference | 5.31 | the TensorRT engine |
| postprocess / NMS | 4.34 | includes the GPU sync the async inference defers |
| ByteTrack + glue | 1.88 | `gmc_method: sgbof` vs `none`: 19.61 vs 19.43 ms -- GMC is free here |
| unpack boxes | 0.34 | per-box `.item()` syncs |
| **total** | **19.6** | |

So the Mac's ~20 ms loop is reproduced almost exactly -- but for the
opposite reason. On the Mac the model was the cost; here the model is 5 ms
and **the CPU-side glue around it is 14 ms**. During the flat-out replay
the GPU averaged 17% busy and the CPU 33% of six cores, i.e. two cores'
worth, one of them pegged. Preprocess scales with input pixels, which is
what the cost matrix below shows.

## 3. OCR headroom

`scripts/bench_ocr.py`: 166 real bib crops taken from the `10-25-15` replay
trace exactly as the pipeline cuts them (padding 15, `min_bib_yolo_conf`
0.25; widths 49-251 px, median 92; heights 60-228 px), preprocessed to
120 px height and width buckets, read with `allowlist` digits on CUDA:

```
easyocr.Reader construction 7.8s (device cuda)
warm-up sweep, first call at each bucket width:
  width  128: first  333.6 ms, second  27.7 ms
  width  160: first   65.6 ms, second  28.5 ms
  width  192: first   64.6 ms, second  29.3 ms
  width  224: first   66.8 ms, second  30.9 ms
  width  256: first   72.2 ms, second  37.0 ms
  width  320: first   64.9 ms, second  43.0 ms
  width  384: first   65.7 ms, second  48.3 ms
  width  448: first   69.0 ms, second  41.8 ms
preprocess (CPU, worker thread)   median  0.3 ms
read, bucketed width              median 26.9 ms  p90 42.1  max 67.7  mean 28.0  -> 36/s sustained
137/166 crops produced digits; most common [('121', 111), ('124', 16), ('12', 5), ('21', 3), ('421', 1)]
read, unbucketed width            median 26.9 ms  p90 35.4  max 64.7  mean 27.4  -> 37/s
  first call at each of 55 new widths: median 27.9 ms  max 64.7
```

| | Mac (MPS) | Jetson (CUDA) |
|---|---|---|
| real-crop read, median | ~48 ms | **26.9 ms** |
| worker capacity | ~20 reads/s | **~36 reads/s** |
| first call at a new width | 155-1084 ms | 65-72 ms (333 ms once, the very first call) |

So a read is 1.8x cheaper and the kernel-compile stall that
`width_buckets_px` exists for is mostly absent on CUDA (a new width costs
one extra ~40 ms, not a second). Buckets stay on: they cost nothing and
the sweep still pays the one-off 333 ms before the race.

What the capacity buys: at `async_min_submit_interval_s: 0.12` one runner
offers ~8 reads/s, so the worker is idle 75% of the time. At 0.05 (20/s)
one runner uses ~55% of it and two runners at the line together
(~40/s) just exceed it; at 0.03 (33/s) a single runner saturates it and a
second one is dropped or late. **0.05 is the floor with margin; 0.03 is
not.** Both are tested on the 14-clip real-time smoke set below.

### The preprocess cost, and OpenCV's thread count

`scratchpad/profile_pre.py` (approximate: a TensorRT build was running on
another core) splits ultralytics' preprocess for a 1383x756 crop:

| input | LetterBox (`cv2.resize` + border) | BGR→RGB / BHWC→BCHW / contiguous | host→device | `/255` on GPU |
|---|---|---|---|---|
| 928x512 | 5.51 ms | 0.98 | 0.25 | 0.28 |
| 1280x736 | 8.55 | 2.03 | 0.40 | 0.55 |
| 1376x768 | 9.38 | 2.30 | 0.44 | 0.61 |
| 1920x1088 (full frame, pad only) | 0.71 | 4.58 | 0.82 | 1.20 |

The resize is the cost, and it is single-threaded: `ultralytics/utils`
calls `cv2.setNumThreads(0)` at import (to keep OpenCV out of DataLoader
workers), and that setting stays in force for the race process. Measured
LetterBox ms against `cv2.setNumThreads(n)` after the ultralytics import:

| cv2 threads | 928x512 | 1280x736 | 1376x768 |
|---|---|---|---|
| 0 (ultralytics' setting) | 5.52 | 8.55 | 9.39 |
| 1 | 5.50 | 8.45 | 9.36 |
| 2 | 3.26 | 4.49 | 5.00 |
| 3 | 2.50 | 3.29 | 3.85 |
| **4** | **1.81** | **2.75** | **2.79** |
| 6 | 1.86 | 2.75 | 2.75 |

Output is bit-identical (same function, more threads). Hence the new
`model.cv2_threads` knob (default 0 = unchanged; the Jetson config sets 4,
leaving cores for the OCR worker, the capture thread and decode). A GPU-side
letterbox in torch would be ~1-2 ms more (upload 0.9 ms + 1.1-1.9 ms) but
needs ultralytics' tensor-input path and its own box un-letterboxing; not
worth the intrusion when four threads recover most of it.

### Capture side: what a 1080p60 USB camera would cost

No camera is attached, so this is decode cost on real 1080p content, not
a device measurement. 120 frames of the reference clip were JPEG-encoded
at quality 85 (733 KB mean, 44 MB/s at 60 fps -- a typical 1080p MJPEG
webcam is 300-800 KB/frame) and decoded three ways. **Preliminary: taken
while a TensorRT build occupied a core and the GPU; re-measured on a
quiet board further down.**

| decode path | ms/frame | fps | where it runs |
|---|---|---|---|
| `cv2.imdecode` (what OpenCV's V4L2 backend does for MJPEG today) | 21.0 | 48 | CPU, one core (libjpeg-turbo, no threading gain) |
| GStreamer `jpegdec` → `videoconvert` | 26.1 | 38 | CPU |
| GStreamer `nvjpegdec` → `nvvidconv` → BGRx → `videoconvert` BGR | 22.7 | 44 | NVJPG + VIC, but the element copies through system memory |
| GStreamer **`nvv4l2decoder mjpeg=1`** → `nvvidconv` → BGRx → `videoconvert` BGR | **11.0** | **91** | NVJPG via V4L2 |

So on the CPU a 1080p60 MJPEG camera cannot even be decoded at 60 fps on
this board (21 ms/frame is 126% of one core), while `nvv4l2decoder mjpeg=1`
does it on the hardware JPEG engine with the CPU paying only the final
BGRx→BGR conversion. The HEVC clips decode at 5.8-10 ms/frame on the CPU
through OpenCV/FFmpeg and at 10 ms/frame through NVDEC (the conversion
back to BGR dominates), so file replays are unaffected.

What changed in code, all off by default: `open_source` accepts a
GStreamer launch string ending in `appsink` and opens it with
`cv2.CAP_GSTREAMER`; `CameraSource` takes an optional `fps` request
(`--camera-fps` on `race_cv.run`) sent as `CAP_PROP_FPS`, since the
capture path never asked the driver for a rate and V4L2 will not pick
1080p60 on its own. A device index with no `--camera-fps` sends exactly
what it always sent. Verified with `videotestsrc` at 1080p60 through
`appsink` (60.0 fps delivered) and in `tests/test_capture.py`.

The pipeline string a USB MJPEG camera would use here (untested for want of
a camera; `v4l2-ctl --list-formats-ext -d /dev/video0` first to confirm the
camera offers `MJPG 1920x1080 @ 60`):

```bash
PYTHONPATH=src .venv/bin/python -m race_cv.run --config config/race_cv.jetson.yaml -r roster.csv --source \
  'v4l2src device=/dev/video0 io-mode=2 ! image/jpeg,width=1920,height=1080,framerate=60/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1'
```

## 2. Detector cost matrix

`scratchpad/run_matrix.sh` -> `scripts/bench_detector.py`, reference clip
`14-48-12` frames 930-1529 (the 36 s crossing: one runner approaching,
spectators in shot), race config with the OCR worker running, roster
loaded, **MAXN_SUPER, DVFS (no `jetson_clocks`)**, `model.cv2_threads: 4`
unless marked t0. "loop" is `Pipeline.process()` -- what a frame costs the
frame loop; "e2e" adds the CPU HEVC decode (5.7 ms), which a camera
source would not pay in this form.

| configuration | detector median / p90 | **loop median / p90 / mean** | fps at loop median | e2e fps flat out | people / bib boxes in 600 frames |
|---|---|---|---|---|---|
| 928x512 FP16 crop, cv2 threads 0 (the Mac config, as shipped) | 19.4 / 21.0 ms | 19.5 / 21.1 / 19.8 | 51 | 38.4 | 79 / 53 |
| **928x512 FP16 crop**, threads 4 | 15.8 / 17.4 | **15.9 / 17.5 / 16.1** | **63** | 45.0 | 79 / 53 |
| 928x512 FP32 crop, threads 4 | 19.6 / 21.2 | 19.7 / 21.3 / 19.9 | 51 | 38.8 | 80 / 53 |
| 1280x736 FP16 crop | 23.5 / 24.0 | 23.6 / 24.2 / 23.7 | 42 | 34.0 | 544 / 84 |
| **1376x768 FP16 crop (native)** | 20.9 / 25.2 | **21.0 / 25.4 / 22.6** (max 136) | 48 | 35.3 | 161 / 94 |
| 1920x1088 FP16 full frame, ROI off | 30.3 / 34.7 | 30.4 / 34.8 / 31.8 (max 153) | 33 | 26.6 | 147 / 92 |
| 928x512 FP16 + two-stage 640x640 on each runner | 16.0 / 17.5 | 16.1 / **30.2** / 17.9 | 62 | 41.6 | 79 / 82 (+29 from the second stage) |

Readings:

- **FP16 vs FP32 at 928x512: 3.8 ms.** The FP32 engine's inference is ~9 ms
  against ~5 ms; everything else is identical. FP16 everywhere.
- **What clears 30 fps (33 ms) with margin:** 928x512, 1376x768 and
  1280x736 all do, with p90s of 17.5, 25.4 and 24.2 ms. The full
  1920x1088 frame does not: 30.4 ms median, 34.8 ms p90 -- it would drop
  frames at every crossing.
- **What clears 60 fps (16.7 ms):** only 928x512, and only at the median
  (15.9 ms; p90 17.5). That is not margin. Nothing else is close.
- **1376x768 is cheaper than 1280x736** despite more pixels: the 1280
  input scales the 1383-wide crop by 0.925, which makes objects ~1.4x
  their training scale and the detector fires on far more small people
  (544 person boxes vs 161), and NMS/ByteTrack cost grows with candidates.
  At 1376x768 the crop is fed at ~1:1 (scale 0.995). Neither number says
  which is *right* -- the smoke test does.
- **Two-stage is nearly free at the median and expensive at the tail:**
  +0.2 ms median, but p90 30 ms because a frame with a runner in it pays a
  640x640 pass per crop (~14 ms each). Mean 17.9 ms. Affordable at 30 fps.
- Inference is 5 ms of every one of these; the rest is CPU glue. Preprocess
  (resize, transpose, upload) and NMS/sync are the two remaining blocks,
  and the next experiment (an engine with NMS inside TensorRT) targets the
  second.

### 60 fps ingestion on real 60 fps footage

The one 60 fps recording (`2026-09-05 15-11-20`, 15.3 s, 920 frames, no
expectations file entry) played in real time with `target_fps: 60`,
`confirm_frames: 16`, `min_observations: 10` and a 240-frame
`track_buffer` (the 30 fps values doubled), `scratchpad/run_60fps.sh`:

| config | health line at end of clip |
|---|---|
| 928x512 FP16, threads 4 | `processed 698 (45.9 fps) \| source dropped 222 \| finishers 1` |
| 1280x736 FP16 | `processed 483 (31.8 fps) \| source dropped 437 \| finishers 1` |

So a 60 fps *file* is consumed at 46 fps at 928x512, with a quarter of
the frames dropped. Two caveats that go opposite ways: (1) a file source
decodes HEVC on the frame-loop thread (5.7 ms/frame), which a camera does
not -- the capture thread decodes on its own core -- so the loop alone
(15.9 ms median) sits right at the 16.7 ms a 60 fps camera allows;
(2) "right at" means p90 17.5 ms, which is over. Verdict from throughput:
**60 fps is not affordable with margin at any size today.** It becomes
affordable at 928x512 only if the loop loses ~4 ms, which is what the
end2end-NMS engine below is for.

### MJPEG decode, quiet board (final numbers)

Same 120 q85 frames, nothing else running:

| decode path | ms/frame | fps |
|---|---|---|
| `cv2.imdecode` (CPU; OpenCV's V4L2 backend does this for an MJPEG camera) | 23.5 | 43 |
| GStreamer `jpegdec` (CPU) → `videoconvert` | 26.2 | 38 |
| GStreamer `nvjpegdec` → `nvvidconv` → `videoconvert` | 22.3 | 45 |
| **GStreamer `nvv4l2decoder mjpeg=1` → `nvvidconv` BGRx → `videoconvert` BGR** | **10.8** | **93** |

A 1080p60 MJPEG camera opened by index (`cv2.VideoCapture(0)`) would be
decoded on the CPU at ~43 fps -- it cannot even reach 60, and it would
take a whole core from the frame loop while trying. Through
`nvv4l2decoder mjpeg=1` the JPEG engine does it at 93 fps and the CPU pays
only the BGRx→BGR conversion. That is the pipeline string quoted above,
and it is the reason `open_source` now accepts one.

### ONNX opset under torch 2.9 (found while building the NMS engine)

ultralytics 8.3.176 picks the ONNX opset with `get_latest_opset()`, which
counts `torch.onnx.symbolic_opset*` attributes; torch 2.9 exposes only
`symbolic_opset9/10`, so the probe returns **9** and the export runs at
opset 10. The plain engines above were built from such a graph: they
parse, build and detect (the smoke and replay results are on them), but
`torchvision::nms` needs opset 11+, so the `--nms` export failed with
"opset version 10 is not supported". `scripts/export_tensorrt.py` now
pins `--opset 17`. The engines the recommendation names are rebuilt at
opset 17 below and re-benchmarked; earlier opset-10 rows are labelled.
