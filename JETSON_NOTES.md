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

### Opset-17 rebuilds and the end2end NMS engine

Same window, same conditions, after rebuilding the two candidate engines
at opset 17 (`ONLY=1 scratchpad/run_matrix.sh ...`):

| configuration | detector median / p90 | **loop median / p90 / mean** | fps at loop median | people / bib boxes |
|---|---|---|---|---|
| 928x512 FP16 (opset 17) | 15.2 / 16.8 | **15.3 / 16.9 / 15.6** | 65 | 79 / 53 (identical to opset 10) |
| **928x512 FP16 + NMS in the engine** (`--nms`, opset 17) | 13.3 / 14.8 | **13.4 / 14.9 / 13.7** | **75** | 79 / 53 (identical) |
| 1376x768 FP16 (opset 17) | 19.2 / 23.6 | **19.3 / 23.8 / 20.9** (max 133) | 52 | 161 / 94 (identical) |
| 928x512 FP16 + two-stage (opset 17 first stage) | 15.2 / 16.8 | 15.3 / 29.4 / 17.1 | 65 | 79 / 82 |

The opset-17 graphs are the same network and produce the same boxes;
they run 0.6-1.7 ms faster. **The NMS-in-engine build is the one that
clears 60 fps with margin at 928x512: p90 14.9 ms against a 16.7 ms
frame.** It returns the same 79 person and 53 bib boxes on this window,
so ultralytics' torch NMS was pure overhead here. Its conf 0.25 / IoU
0.7 are frozen at export (`--conf`, `--iou`), matching `model.conf` and
`model.iou` in the config; changing those in the config would no longer
take effect for the first stage, which is the price of this engine and
is written next to it in the Jetson config.

Loop-time summary at 928x512 FP16, cv2 threads 4: **19.5 ms (Mac config
as shipped) → 15.3 ms (threads) → 13.4 ms (NMS in engine)**, of which
~5 ms is the model.

## 5. Real-time 14-clip smoke runs

`scratchpad/run_smoke.sh` -> `scripts/smoke_test.py --expected smoke_test.yaml
--roster data/raw/roster_example.csv --realtime --config <cfg>`, one run at
a time with nothing else on the board, MAXN_SUPER, DVFS. Each run is
~26 min of wall clock (27 min of footage, paced). Results in
`runs/jetson/smoke_<label>/{log.txt,results.json}`.

### S1 -- baseline: `config/race_cv.jetson.yaml` (928x512 FP16, cv2 threads 4, OCR interval 0.12)

```
RECALL (found / expected)           96.2%   (25/26)
bib exactly right                 23/25
  of racers wearing a bib         16/18
  of racers with no bib           7/7
missed (never detected)                  1     No bib @ 22s  2025-10-03 17-45-02 (the expectation on the clip's final frame)
ghosts (detected, not expected)          2     225 @ 20.0s and 131.5s in 14-42-58 (the two real crossings the file omits)
median |time drift|                   0.3s
worst |time drift|                    2.3s   (the No-bib @122s in 14-48-12, -2.3 s, as on the Mac)
WRONG BIB: 531 -> No bib, 76 -> No bib   (2026-09-07 07-57-12: both off the three-bib example roster)
```

| clip | frames processed | source dropped | ocr reads / skipped / late |
|---|---|---|---|
| 2025-10-03 17-45-02 | 648 | 5 | 0 / 0 / 0 |
| 2025-10-04 10-25-15 | 1514 | 3 | 3 / 0 / 0 |
| 2025-10-04 14-42-58 | 6419 | 4 | 5 / 0 / 0 |
| 2025-10-04 14-48-12 | 9335 | 3 | 9 / 0 / 0 |
| 2025-10-04 15-07-11 | 4917 | 4 | 2 / 0 / 0 |
| 2025-10-04 15-12-22 | 5288 | 4 | 6 / 0 / 0 |
| 2025-10-04 15-16-23 | 3108 | 4 | 5 / 0 / 0 |
| 2025-10-04 15-32-31 | 2995 | 4 | 1 / 0 / 0 |
| 2025-10-04 15-35-06 | 2632 | 3 | 1 / 0 / 0 |
| 2025-10-04 15-39-37 | 2660 | 4 | 4 / 0 / 0 |
| 2025-10-19 11-03-25 | 890 | 4 | 0 / 0 / 0 |
| 2025-10-19 11-04-32 | 905 | 3 | 0 / 0 / 0 |
| 2025-10-19 11-05-07 | 1926 | 4 | 0 / 0 / 0 |
| 2026-09-07 07-57-12 | 2433 | 2 | 18 / 0 / 0 |
| **total** | **45670** | **51 (0.11%)** | **54 / 0 / 0**, 2 hand-offs |

Against the Mac baseline (25/26, 0 genuine ghosts, 0.4 s, 22/25 bibs, 36
dropped of 9338 on the longest clip = 0.4%): same recall, same two
flagged crossings, the same miss, drift 0.3 s, one more bib right
(23/25: the clipped 120 resolves through the roster snap here too), and
a quarter of the drop rate. (Where in each clip the 3-5 drops fall is not
recorded by the smoke test; the single-clip `race_cv.run` health lines
earlier showed 5 by 10 s and 12 by 40 s on `10-25-15`, so some land near
the crossing and some at the start.)

### S2 -- OCR headroom used: `async_min_submit_interval_s: 0.05`, `async_max_inflight_per_track: 4` (else as S1)

```
RECALL 96.2% (25/26) | bib exactly right 23/25 (wearing 16/18, no-bib 7/7) | ghosts 2 | drift median 0.3 s, worst 2.3 s
source dropped 54 / 45667 | ocr read 77, skipped 0, late 0        (S1: 54 reads at 0.12)
```

Per clip the reads rose everywhere a bib was legible (10-25-15: 3→4,
14-42-58: 5→8, 14-48-12: 9→11, 15-39-37: 4→7, 07-57-12: 18→29) with no
crop ever skipped and no finish resolved before its reads landed, so at
0.05 the CUDA worker is nowhere near its ~36 reads/s. **Bib accuracy did
not move**: the same 23/25, the same two wrong (531 and 76, which are not
on the three-bib example roster and need two agreeing off-roster reads to
win). The reads that decide a bib are bounded by the frames on which the
detector finds the bib at ≥0.25, not by the OCR rate; on this footage
that was already saturated at 0.12. 0.05 is still the right setting on
the Jetson -- it costs nothing measurable here and buys a wider vote when
a bib is only legible for a few frames -- but it is not an accuracy lever
on this set. 0.03 with 6 in flight is run last in the chain to show where
the worker gives out.

### S3 -- two-stage: `two_stage: true`, `two_stage_model: trt_640x640_fp16.engine`, `two_stage_imgsz: 640` (else as S1)

```
RECALL 96.2% (25/26) | bib exactly right 23/25 (wearing 16/18, no-bib 7/7) | ghosts 2 | drift median 0.3 s, worst 2.3 s
source dropped 426 / 45295 (0.9%)   <- S1: 51 (0.11%)
ocr read 99, skipped 0, late 0       <- S1: 54
```

Bib boxes rose on every clip (14-48-12: 119→163, 07-57-12: 83→137,
15-32-31: 22→49) and OCR read almost twice as often, and **not one bib
changed**. What did change is the drop count: 8x, and it lands on the
clips with several people in shot (11-05-07 with four finishers: 125
drops in 64 s; 15-32-31: 52), because through ultralytics each crop is a
full `predict()` -- ~14 ms of the same glue the first stage pays -- and a
frame with three runners costs 40+ ms. Through the direct backend a crop
should be ~5 ms; that is measured below before deciding. On the
ultralytics path, two-stage is not worth its drops on this footage.

### S4 -- native crop resolution: 1376x768 FP16 (else as S1)

```
RECALL 92.3% (24/26)   <- one racer LOST vs S1
bib exactly right 22/24 (wearing 15/17, no-bib 7/7) | ghosts 2 | drift median 0.4 s, worst 2.3 s
source dropped 67 / 45654 | ocr read 97, skipped 0, late 0
MISSED: No bib @ 22s (17-45-02, as always)  and  120 @ 172s (14-48-12)
```

The new miss is the hand-off case: in S1 that crossing is recovered
across a track break (`handoffs 1`); at 1376x768 the track fragments at
the line and the newborn fails `min_observations`
(`finishes_below_min_observations` 0→1 on that clip, and 0→1 on
15-07-11 too, where the finisher was still found). The larger input finds
more of everything -- person boxes 514→773 on 14-48-12, bib boxes
119→175, OCR reads 54→97 across the set -- and every extra bib box bought
nothing (the same 2 wrong: the off-roster pair), while the busier boxes
near the camera cost a finisher. `RACE_DAY_ANALYSIS.md` predicted this
("raising detector resolution ... fragments tracks more readily").
**928x512 stays.** The tracker-side levers (`min_observations`,
`handoff_window_s`) could be retuned for 1376x768, but there is no
accuracy on the table to pay for it.

### S5 -- NMS inside the engine: `trt_928x512_fp16_nms.engine` (else as S1)

```
RECALL 96.2% (25/26) | bib exactly right 23/25 (wearing 16/18, no-bib 7/7) | ghosts 2 | drift median 0.3 s, worst 2.3 s
source dropped 55 / 45666 | ocr read 52, skipped 0, late 0
```

Per clip the person and bib box counts are identical to S1 on 13 of 14
clips (514→513 and 234→236 on the other two), and **all 28 finisher rows
are identical to S1 in bib and in time (within 0.2 s)**. The engine with
NMS baked in is the same detector; the 2 ms it saves per frame is pure
ultralytics overhead. It is the first stage of the recommended config.

### S6 -- OCR at the floor: `async_min_submit_interval_s: 0.03`, `async_max_inflight_per_track: 6` (else as S1)

```
RECALL 96.2% (25/26) | bib exactly right 23/25 (wearing 16/18, no-bib 7/7) | ghosts 2 | drift median 0.3 s, worst 2.3 s
source dropped 60 / 45661 | ocr read 116, skipped 0, late 0        (S1: 54 at 0.12, S2: 77 at 0.05)
```

Reads doubled again (14-48-12: 11→20, 07-57-12: 29→41) and the worker
*still* never skipped a crop or resolved a finish late: on this footage
there is rarely more than one runner at the line, so 33 offered reads/s
sit under the ~36/s the CUDA worker sustains. Accuracy unchanged, drops
unchanged within noise. The recommendation stays at **0.05 / 4**: it
already collects every read that mattered and leaves capacity for two
runners abreast (2 x 20/s), which 0.03 would not (2 x 33/s > 36/s) -- and
that is the pack situation a race has and this footage does not.

| OCR spacing | reads (14 clips) | skipped / late | bibs right |
|---|---|---|---|
| 0.12 s, 3 in flight (Mac value) | 54 | 0 / 0 | 23/25 |
| 0.05 s, 4 in flight | 77 | 0 / 0 | 23/25 |
| 0.03 s, 6 in flight | 116 | 0 / 0 | 23/25 |

## The direct TensorRT backend (`model.backend: trt`)

Motivation: at 928x512 the model is 5 ms of a 13-15 ms frame; the rest is
ultralytics' CPU glue. `detect.TrtRunner` uploads the crop through a
pinned buffer, letterboxes in torch on the GPU (geometry checked against
ultralytics' `LetterBox`/`scale_boxes` in `tests/test_trt_backend.py`),
runs the `--nms` engine with `execute_async_v3`, and hands the surviving
rows to ultralytics' own `BYTETracker` built from the same yaml, updated
on every frame like the predictor does. No `get_cfg`, no dataset object,
no `Results`.

**Equivalence** (`scratchpad/equivalence.py`, same NMS engine both ways,
300 frames of the 14-48-12 crossing):

```
[detect] matched 395 boxes (IoU>=0.5), only-ultralytics 2, only-direct 6 | IoU median 0.999 p10 0.996 | |dconf| median 0.0010 max 0.015 | ms/frame ultralytics 12.2 direct 7.9
[track]  matched 331 boxes,             only-ultralytics 2, only-direct 3 | IoU median 0.998 p10 0.996 | |dconf| median 0.0010 max 0.015 | ms/frame ultralytics 13.6 direct 8.7 | track-id mapping consistent on 132/132 tracked pairs
```

The residual differences are the GPU bilinear resize versus cv2's (sub-pixel
sampling, conf within 0.015); the handful of unmatched boxes are at the 0.25
conf edge.

**Cost** (`bench_detector`, same window, cv2 threads 4):

| configuration | loop median / p90 / mean | fps at median | e2e fps (with CPU decode) |
|---|---|---|---|
| NMS engine via ultralytics | 13.4 / 14.9 / 13.7 | 75 | 49.5 |
| **NMS engine, direct backend** | **8.9 / 9.8 / 9.1** | **113** | 60.9 |
| direct backend + two-stage (640 `--nms` engine, direct) | 8.8 / 16.5 / 10.0 | 114 | 58.8 |

So the loop went **19.5 → 8.9 ms** across the session (2.2x), and the 60
fps budget of 16.7 ms is now cleared with margin at 928x512 -- even the
file replay with its 5.7 ms CPU decode on the loop thread runs at 60.9
fps. Two-stage on this path costs +1 ms mean and +6.7 ms p90 (its 29
extra bib boxes on the window are the same 29 as before).

`trt_640x640_fp16_nms.engine` built in 303 s (peak RAM in the same band as
the other builds).

### Rejected: OCR without CRAFT

EasyOCR's `readtext` runs CRAFT text localisation before recognition; the
crop already is a bib, so `recognize` alone looked like a free 2x
(12.3 ms vs 27.2 ms per read). It is not: on the pipeline's preprocessed
crops it agreed with `readtext` on **0/166** (top answers '41', '4',
'0'), and trimming the crop to its dark pixels first did no better
(0/166). CRAFT is finding the digit line inside a crop that also holds
shirt and background; the recogniser cannot do that itself. Left as is.

## 6. Report

Every row: Jetson Orin Nano Developer Kit Super, JetPack 7.2.1, TensorRT
10.16, **MAXN_SUPER under DVFS** (`jetson_clocks` not confirmable without
sudo; GPU 1020 MHz and CPUs 1728 MHz read at idle and under load), desktop
session resident (~3-4 GB), no swap. ms/frame and flat-out fps are the
frame loop (`Pipeline.process`) on the 14-48-12 crossing window, opset-17
engines, cv2 threads 4 unless noted. Recall / bibs / drift / dropped are
the 14-clip real-time smoke run (26 finishers, three-bib example roster).
Peak temperature is `tj` from tegrastats over that run.

| configuration | power mode | ms/frame (median / p90) | fps flat out (loop) | recall | bibs right | drift (median) | source dropped | peak tj |
|---|---|---|---|---|---|---|---|---|
| Mac M2 baseline (CoreML 928x512, from the brief) | -- | ~20-24 | ~42-50 | 25/26 | 22/25 | 0.4 s | ~0.4% | -- |
| 928x512 FP16, Mac config as shipped (cv2 threads 0) | MAXN_SUPER | 19.5 / 21.1 | 51 | -- | -- | -- | -- | -- |
| S1 928x512 FP16, cv2 threads 4, OCR 0.12 | MAXN_SUPER | 15.3 / 16.9 | 65 | 25/26 | 23/25 | 0.3 s | 51 / 45670 (0.11%) | 56.5 C |
| S2 + OCR 0.05 / 4 in flight | MAXN_SUPER | 15.3 / 16.9 | 65 | 25/26 | 23/25 | 0.3 s | 54 (0.12%) | 56.5 C |
| S6 + OCR 0.03 / 6 in flight | MAXN_SUPER | 15.3 / 16.9 | 65 | 25/26 | 23/25 | 0.3 s | 60 (0.13%) | 54.2 C |
| S3 + two-stage 640 (ultralytics path) | MAXN_SUPER | 15.3 / 29.4 (mean 17.1) | 65 | 25/26 | 23/25 | 0.3 s | **426 (0.9%)** | 55.3 C |
| S4 1376x768 FP16 (native crop) | MAXN_SUPER | 19.3 / 23.8 | 52 | **24/26** | 22/24 | 0.4 s | 67 (0.15%) | 58.0 C |
| 1280x736 FP16 (bench only) | MAXN_SUPER | 23.6 / 24.2 | 42 | -- | -- | -- | -- | -- |
| 1920x1088 FP16 full frame, ROI off (bench only) | MAXN_SUPER | 30.4 / 34.8 | 33 | -- | -- | -- | -- | -- |
| 928x512 FP32 (bench only) | MAXN_SUPER | 19.7 / 21.3 | 51 | -- | -- | -- | -- | -- |
| S5 928x512 FP16 with NMS in the engine (ultralytics path) | MAXN_SUPER | 13.4 / 14.9 | 75 | 25/26 | 23/25 | 0.3 s | 55 (0.12%) | 55.2 C |
| **RECOMMENDED: 928x512 FP16 NMS engine, direct backend, cv2 threads 4, OCR 0.05 / 4** | MAXN_SUPER | **8.9 / 9.8** | **113** | **25/26** | **23/25** | **0.3 s** | **48 / 45673 (0.11%)** | **55.2 C** |
| recommended + two-stage 640 NMS engine, direct | MAXN_SUPER | 8.8 / 16.5 (mean 10.0) | 114 | 25/26 | 23/25 | 0.3 s | 45 (0.10%) | 55.5 C |

### S7 -- confirmation of the recommended config (`config/race_cv.jetson.yaml` as committed)

```
RECALL 96.2% (25/26) | bib exactly right 23/25 (wearing 16/18, no-bib 7/7) | ghosts 2 | drift median 0.3 s, worst 2.3 s
source dropped 48 / 45673 (0.11%) | ocr read 73, skipped 0, late 0
tegrastats over the run: CPU 19% of 6 cores (S1: 24%), GR3D 17%, 8.6 W, tj max 55.2 C
```

All 28 finisher rows are identical to S1 in bib and time (within 0.2 s);
per-clip person and bib box counts match S1 within a few boxes (the GPU
resize's sub-pixel differences at the 0.25 conf edge). The direct backend
changes the cost of the race, not its result.

### S8 -- recommended config with `two_stage: true` (direct backend, 640 `--nms` engine)

```
RECALL 96.2% (25/26) | bib exactly right 23/25 (wearing 16/18, no-bib 7/7) | ghosts 2 | drift median 0.3 s, worst 2.3 s
source dropped 45 / 45676 (0.10%)   <- S3 through ultralytics: 426 | ocr read 147, skipped 0, late 0   <- S7: 73
```

All 28 finisher rows identical to S7. Bib boxes +50-100% per clip
(14-48-12: 119→174, 07-57-12: 84→153), OCR reads doubled, and the drop
penalty that made S3 unusable is gone: on the direct backend a runner's
640 crop is ~5 ms. Two-stage is now affordable; it is left off because on
this footage it changes no verdict, and it is one line to enable.

### 60 fps ingestion with the recommended config

The 60 fps clip (`2026-09-05 15-11-20`, 920 frames) in real time with
`target_fps: 60`, `confirm_frames: 16`, `min_observations: 10`,
`track_buffer: 240`:

```
health | processed 894 (58.8 fps) | source dropped 26 | finishers 1
```

versus 45.9 fps / 222 dropped at 928x512 through ultralytics at the start
of the session. The 26 drops are the file path's CPU HEVC decode (5.7 ms)
sitting on the loop thread next to the 8.9 ms loop; a camera decodes on
its own thread, so live the loop alone has to fit 16.7 ms and it does
with p90 9.8 ms. What this footage cannot show is 60 fps *accuracy*
(no expectations exist for it).

### What improved over the Mac baseline

- **Frame cost: 19.5 → 8.9 ms** (median, 928x512). The Mac's ~20 ms
  reproduced exactly on the Jetson as shipped, then fell in three measured
  steps: OpenCV threads back on (-4 ms), NMS inside the engine (-2 ms),
  and the direct TensorRT backend (-4.5 ms). The 33 ms budget at 30 fps is
  now used 27%; the 16.7 ms budget at 60 fps is cleared with margin
  (p90 9.8 ms), which the Mac never could.
- **Dropped frames: ~0.4% → ~0.1%** live, with the loop idle most of the time.
- **OCR: 48 → 26.9 ms per real-crop read**, ~36 reads/s, and the 150-1000 ms
  new-width stalls that forced `width_buckets_px` are 65 ms here. The
  per-track spacing is 0.05 s instead of 0.12 s with zero skipped or late.
- **Two-stage is affordable** (+1 ms mean, +6.7 ms p90 on the direct
  backend) where the Mac could not run it at all.
- **A 1080p60 USB camera is decodable**, on the JPEG engine
  (`nvv4l2decoder mjpeg=1`, 93 fps) through the new GStreamer source; the
  CPU path tops out at 43 fps.
- Thermals and power are a non-issue: tj peaks 54-58 C, 8.5-10.5 W, over
  26-minute real-time runs.

### What did not improve

- **Accuracy.** 25/26 found, 23/25 bibs, 0.3 s drift is where the Mac
  already was (its 22/25 is 23/25 with the roster snap it now has). Every
  extra read, bib box and second-stage crop this board affords changed no
  verdict: the two wrong bibs are racers not on the example roster, the one
  miss is an expectation on a clip's last frame. Bib accuracy is roster-
  and footage-bound, not compute-bound.
- **Resolution.** More detector pixels found more boxes and lost a
  finisher (1376x768: 24/26 -- the track fragments at the line and the
  hand-off fails). The full frame does not clear 30 fps. 928x512 stays.
- **The board's memory.** 7.5 GB with no swap and a desktop holding 3-4 GB
  of it; engine builds peaked at 6.8 GB. Build before race day, headless.

### The next limit

The loop is 8.9 ms of which the model is ~5; ByteTrack + Kalman + GMC
are ~1.9 ms on the CPU and the GPU letterbox + one small D2H copy make up
the rest. The next things that would matter, in order: (1) with a live
camera the capture thread's BGRx→BGR `videoconvert` and the frame-loop's
pinned copy could be replaced by uploading BGRx straight to the GPU; (2)
the OCR worker shares the GIL with the loop and CRAFT is CPU-heavy, so at
two runners abreast and 0.05 s spacing the worker (~36/s) is the first
thing to saturate; (3) beyond that it is the tracker's Python, which is
the same code the Mac runs. None of these is needed for 30 fps. For a
60 fps camera the frame-counted levers in the config
(`finish_line.confirm_frames`, `min_observations`, `track_buffer`) must
be doubled, and the one 60 fps clip shows the loop keeps up; what is not
yet shown is 60 fps *accuracy*, because there is no 60 fps footage with
expectations.

### Recommended `config/race_cv.jetson.yaml`

Committed on `jetson-eval`. The deltas from the Mac config, all measured
above:

| key | Mac | Jetson | why |
|---|---|---|---|
| `model.path` | `rect_928x512.mlpackage` | `trt_928x512_fp16_nms.engine` | S5: same boxes, -2 ms; built by `export_tensorrt.py --size 512 928 --nms` |
| `model.backend` | ultralytics | **trt** | direct path: 13.4 → 8.9 ms, same boxes (IoU 0.999) |
| `model.device` / `half` | cpu / false | cuda:0 / true | |
| `model.cv2_threads` | 0 | 4 | letterbox 5.5 → 1.8 ms |
| `model.two_stage_model` | null | `trt_640x640_fp16_nms.engine` | ready; `two_stage` stays false (no accuracy change, S3) |
| `ocr.async_min_submit_interval_s` / `async_max_inflight_per_track` | 0.12 / 3 | 0.05 / 4 | S2: reads 54 → 77, 0 skipped, 0 late |
| everything else (geometry, thresholds, tracker yaml) | -- | unchanged | |

Run it with `--config config/race_cv.jetson.yaml` (the launcher's `-v`/`-c`
flags pass through), `PYTHONPATH=src` when calling `race_cv.run` directly,
and `YOLO_AUTOINSTALL=false` in the environment so ultralytics never pip-
installs into the venv mid-race.

## Launcher on the Jetson

`./start-race-cv.sh` defaulted to `config/race_cv.yaml`, whose model is
the Mac's `rect_928x512.mlpackage`, and failed the model preflight with a
misleading "install coremltools" (CoreML cannot run on Linux at all). The
launcher now picks `config/race_cv.jetson.yaml` by itself when
`/etc/nv_tegra_release` exists (`--config` / `RACE_CV_CONFIG` still win),
explains a `.mlpackage` on Linux as a wrong config rather than a missing
package, reads memory from `/proc/meminfo` instead of `vm_stat`, prints
the LAN address from `hostname -I`, and exports `YOLO_AUTOINSTALL=false`
so ultralytics never pip-installs mid-race. Verified 2026-09-12 with

```bash
./start-race-cv.sh -v "data/raw/Camo Recording 2026-09-07 07-57-12.mov" -r data/results/test_starter_list_real.csv --preview --native-frontend
```

which ran the whole clip at 29.9 fps (14 of 2421 frames dropped), opened
the GTK preview window, delivered 3 finishers to the API, and stopped the
API on exit as preview mode does. The `Gtk-Message: Failed to load module
"canberra-gtk-module"` line is harmless (a sound theme module).

### `--preview` over SSH

With no display, `cv2.imshow` raised "GTK backend is not available" on
every frame the preview gate fired on, counted as frame errors (100+ per
clip, tracebacks in the log) while timing carried on unaffected.
`race_cv.run` now disables the window with one warning when
`DISPLAY`/`WAYLAND_DISPLAY` is empty on Linux, and once at the first
failed draw otherwise, pointing at the browser stream (`/video_feed`).
Verified on the 60 fps clip: one warning, 0 frame errors, 59.7 fps.

## Boxes popping in and out of the preview (2026-09-12)

Reported on `2026-09-07 07-57-12` with the recommended config. Measured
from replay traces (`scratchpad/continuity.py`: per track, the fraction
of frames within its span on which it was drawn; a bib box "in" a
runner means its centre lies inside the person box):

| detector / conf fed to ByteTrack | runner tracks drawn (frames in span) | gaps | bib in runner (3 finishers) |
|---|---|---|---|
| 928x512, conf 0.25 (as recommended) | 66% / 61% / 61% | 33 | 38% / 2% / 2% |
| **928x512, conf 0.1** (ultralytics path, plain engine) | 73% / 72% / 73% | 18 | 41% / 1% / 1% |
| 928x512, conf 0.1, direct path (`--nms --conf 0.1` engine) | 73% / 72% / 73% | 18 | identical |
| 1376x768, conf 0.25 | 63% / 58% / 57% | 35 | 54% / 13% / 2% |
| 1376x768, conf 0.1 | 76% / 63% / 65% | 22 | 57% / 15% / 1% |

Where the gaps are: the far half of the approach (runner 230-480 px tall,
box bottom y 590-800); from ~600 px tall to the line presence is ~100%.
The longest gap (16 frames) opens on a box at conf 0.11 and closes on one
at 0.56 -- the detector's confidence on this 2026 footage swings across
the threshold frame to frame. So: **not rendering** (the overlay draws
exactly the tracker's output), and two causes:

1. ByteTrack is starved. The config runs the detector at conf 0.25;
   ByteTrack's second association exists to keep tracks alive on
   0.1-0.3 boxes (ultralytics' own `track()` defaults conf to 0.1 for
   this), and `new_track_thresh: 0.5` still gates new tracks, so the low
   band never creates ghosts. Conf 0.1 halves the gaps. On the direct
   backend the threshold is frozen in the engine, so a
   `trt_928x512_fp16_nms_c010.engine` (`--nms --conf 0.1`) was built.
2. The model is less sure on this footage than on the 2025 clips it was
   trained on; ~27% of frames are still missing at conf 0.1. More input
   resolution does not fix that (1376x768: 30% missing) though it sees
   bibs earlier. The fix for this part is 2026 frames in the training
   set (`scripts/autolabel.py`, `scripts/mine_errors.py`).

The tracker keeps the same id across these gaps (track_buffer 4 s), so
the finish logic is not what flickers; the finishers on this clip were
found in every variant above.

### S9 -- conf 0.1 fed to ByteTrack (`trt_928x512_fp16_nms_c010.engine`, direct backend, else as S7)

```
RECALL 96.2% (25/26) | bib exactly right 23/25 (wearing 16/18, no-bib 7/7) | ghosts 2 | drift median 0.3 s, worst 2.3 s
source dropped 48 / 45673 | ocr read 76, skipped 0, late 0 | loop 9.5 / 10.0 ms (bench window)
```

All 28 finisher rows identical to S7. Person-box counts rise sharply on
the busy 2025 clips (14-48-12: 513 → 7342, 14-42-58: 496 → 3882), and
`people_outside_boundary` stays at 0: these are people **on the course**
-- marshals, walkers, far racers -- who once scored ≥0.5 and now keep a
track alive on 0.1-0.3 boxes instead of flickering in and out, exactly
the mechanism that keeps the runner's track alive. No new ghosts, no
change in hand-offs or min-observation rejections: such tracks never
approach and cross the line, and `require_approach` /
`min_observations` are what stand between them and the leaderboard (the
boundary is not involved). Adopted in the recommended config: the runner
tracks on the 2026 clip go from 64% to 73% drawn with half the gaps, at
+0.6 ms per frame; watch the preview for more blue boxes on bystanders,
which is the price.
