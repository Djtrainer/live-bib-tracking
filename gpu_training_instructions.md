# Training the bib detector on a GCP GPU

Local training is not viable for this. On the M-series MPS backend one batch of
8 at 960px took **264 seconds**, which puts 150 epochs at roughly 18 days. The
same run on an L4 is minutes. The dataset is small (376 labelled images), so
this is a short, cheap session rather than a standing cost.

Project already configured: `stunning-vertex-437612-f6`
Quota confirmed available: `NVIDIA_L4_GPUS = 1` in `us-central1`, `GPUS_ALL_REGIONS = 4`.

## 0. Build the dataset first (local, seconds)

```bash
python scripts/build_dataset.py --out data/dataset_v2
git checkout -- config/yolo_dataset.yaml      # the builder rewrites its path; keep the tracked one pointing at data/dataset
sed "s|^path: .*|path: $PWD/data/dataset_v2|" config/yolo_dataset.yaml > config/yolo_dataset_v2.yaml
tar -czf bib_dataset_v2.tar.gz -C data dataset_v2
```

Build into a *new* directory each time labels grow (`data/dataset` is the
376-image set the deployed model trained on; `data/dataset_v2` is the
638-image set from September 2026). Both are gitignored.

`build_dataset.py` is what makes the val score meaningful: it rebuilds one
canonical dataset from every labelled image that exists and holds out
finish-line footage only, split by time segment. See its docstring for why the
previous split could not measure anything.

## 1. Create the VM

L4 is the pick: ~$0.71/hr on demand in `us-central1`, 24GB, and comfortably
fast enough that the whole experiment matrix fits in about an hour.

```bash
gcloud compute instances create bib-train \
  --project=stunning-vertex-437612-f6 \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"
```

Add `--provisioning-model=SPOT` to cut the price roughly in half. Spot can be
reclaimed mid-run, which for a 10-minute training job is usually an acceptable
trade — but the driver install alone takes a few minutes, so a reclaim early on
costs more than it saves.

Image families get retired; if this one 404s, list current ones with
`gcloud compute images list --project=deeplearning-platform-release --filter="family~cu1"`.

The first boot installs the NVIDIA driver. Wait for it:

```bash
gcloud compute ssh bib-train --zone=us-central1-a --command="nvidia-smi" 
```

Retry until it prints a GPU table rather than a "driver not loaded" error.

## 2. Upload the dataset and scripts

```bash
gcloud compute scp bib_dataset_v2.tar.gz bib-train:~ --zone=us-central1-a
gcloud compute scp scripts/train.py bib-train:~ --zone=us-central1-a
gcloud compute scp config/yolo_dataset_v2.yaml bib-train:~ --zone=us-central1-a
```

## 3. Train

```bash
gcloud compute ssh bib-train --zone=us-central1-a
```

Then on the VM:

The deep-learning image is headless and has no `python` on PATH, so two things
need handling before ultralytics will import:

```bash
# 1. No `python`, only `python3`.
# 2. ultralytics pulls in opencv-python, which needs libGL that a server image
#    doesn't ship. Install the system libs and use the headless build -- and
#    pin below 5.x, because opencv-python-headless 5.0.0.x currently installs
#    a wheel whose `import cv2` fails on this image.
sudo apt-get update -qq && sudo apt-get install -y libgl1 libglib2.0-0
python3 -m pip install -q ultralytics coremltools
python3 -m pip uninstall -y -q opencv-python opencv-contrib-python
python3 -m pip install -q 'opencv-python-headless<5'
python3 -c "import cv2, ultralytics, torch; print(cv2.__version__, ultralytics.__version__, torch.cuda.is_available())"

tar -xzf bib_dataset_v2.tar.gz
mkdir -p config && mv yolo_dataset_v2.yaml config/
# The generated config has an absolute path from the laptop; point it at the VM copy.
sed -i "s|^path: .*|path: $HOME/dataset_v2|" config/yolo_dataset_v2.yaml

# Run under nohup so a dropped SSH session cannot kill a multi-hour matrix.
nohup python3 train.py \
  --data config/yolo_dataset_v2.yaml \
  --models yolo11n,yolo11s,yolo11m,yolo11l \
  --imgsz 960,1280 \
  --epochs 150 \
  --batch -1 \
  --device 0 \
  --export coreml \
  > train.log 2>&1 &
```

`--batch -1` lets ultralytics pick the batch per run (it chose 16 for
every model at both sizes on the L4). Its AutoBatch step deliberately raises
the batch until the card refuses, so **"CUDA out of memory" lines near the
start of each run are the probe, not a failure** -- the line after them says
which batch was chosen. `train.py` catches a failed run and moves on, and
rewrites `training_results.json` after every run, so the file existing does
not mean the matrix is finished; `grep -c "epochs completed in" train.log`
does. The September 2026 matrix took 7.5 hours: n 13+20 min, s 31+56,
m 51+113, l 80+165.

`train.py` validates every run against the same held-out finish-line split and
prints a ranked table, so the six runs are directly comparable to each other
and to the deployed model's honest baseline:

```
mAP50 0.917   bib mAP50 0.870   bib recall 0.833
```

Exports happen at the trained size. That matters: the current CoreML model has
a **fixed 640x640 input**, and exporting a 960-trained model at 640 would throw
away what the larger training bought.

## 4. Bring the results back

Ultralytics prefixes a relative `--project models` with `runs/detect/`, so
the runs are under `~/runs/detect/models/<model>_<imgsz>/`, not `~/models`.
Pack just the weights, exports and curves (the batch/label previews are
hundreds of MB of JPEGs) and copy one file:

```bash
gcloud compute ssh bib-train --zone=us-central1-a --command \
  "cd ~/runs/detect/models && tar -czf ~/gpu_runs_v2.tar.gz */weights/best.pt */weights/best.mlpackage */results.csv */args.yaml"
mkdir -p models/gpu_runs_v2
gcloud compute scp bib-train:~/gpu_runs_v2.tar.gz bib-train:~/training_results.json bib-train:~/train.log models/gpu_runs_v2/ --zone=us-central1-a
tar -xzf models/gpu_runs_v2/gpu_runs_v2.tar.gz -C models/gpu_runs_v2
```

## 5. Stop, then delete the VM

An idle L4 bills at the same rate as a busy one. Stop it the moment the copy
is verified; a stopped VM bills only its disk (about $10 a month for 100 GB)
and keeps the dataset and runs for a re-check. Delete it when they are no
longer needed.

```bash
gcloud compute instances stop bib-train --zone=us-central1-a --quiet
gcloud compute instances delete bib-train --zone=us-central1-a --quiet
```

## What to look for in the results

The deployed model scores `bib mAP50 = 0.870` at 640 on this val set, so that
is the bar. Two specific things to check beyond the headline number:

* **Does a larger `imgsz` actually help once trained at that size?** Running
  the *640-trained* model at 1280 measured slightly worse on val (0.848) while
  finding 13x more bibs in real video — the val set is dominated by large,
  clearly-visible bibs because those are the frames a human chose to annotate.
  If the resolution hypothesis is right, a model *trained* at 960/1280 should
  improve both.
* **`bib recall`, not just mAP.** A missed bib is a racer entered as "No bib";
  a slightly loose box is not. Recall is the number that maps to race day.

Then re-measure end to end, which is what actually decides it:

```bash
python scripts/smoke_test.py --expected smoke_test.yaml --config config/race_cv.yaml
```

## September 2026 matrix: larger models on the 638-image set

Val is the 60 held-out finish-line frames (65 bibs), so one bib is 1.5
points; rows within that of each other are ties.

| model | imgsz | mAP50 | mAP50-95 | bib mAP50 | bib recall | bib precision | minutes |
|---|---|---|---|---|---|---|---|
| yolo11s | 1280 | 0.953 | 0.683 | **0.913** | 0.834 | 0.931 | 56 |
| yolo11m | 1280 | 0.951 | 0.681 | 0.909 | 0.846 | 0.974 | 114 |
| yolo11n | 1280 | 0.949 | 0.673 | 0.906 | 0.722 | 1.000 | 20 |
| yolo11n | 960 | 0.939 | 0.644 | 0.884 | 0.846 | 0.948 | 13 |
| yolo11m | 960 | 0.924 | 0.664 | 0.855 | 0.815 | 0.908 | 51 |
| yolo11s | 960 | 0.915 | 0.660 | 0.839 | 0.814 | 0.914 | 31 |
| yolo11l | 1280 | 0.908 | 0.663 | 0.833 | 0.769 | 0.908 | 166 |
| yolo11l | 960 | 0.907 | 0.645 | 0.822 | 0.723 | 0.917 | 80 |

Three findings:

* **Resolution beats capacity.** Every 1280 run beats every 960 run of the
  same model, and the 262 extra finish-line frames lifted yolo11n@1280 from
  0.881 to 0.906 bib mAP50 (different val, same architecture).
* **s, m and n at 1280 are a three-way tie** on this val. Bigger is not
  automatically better here; the recall/precision columns are at
  ultralytics' own threshold, not the pipeline's 0.25, so the end-to-end
  smoke test decides between them, and the Jetson's frame budget decides
  what is deployable.
* **yolo11l loses**, at both sizes, and early-stopped: with 578 training
  images the largest model overfits before it learns small bibs.

Weights and 1280 CoreML exports for all eight are in `models/gpu_runs_v2/`.

### End to end, which is what decides it

Each 1280-trained candidate exported to the deployed 928x512 rectangular
CoreML geometry and run through the 14-clip **real-time** smoke test on the
M2 Air, same example roster as the baseline. Frame cost is flat-out on one
clip under `bib_env`.

| model | ms/frame (Mac) | found | bibs right | notes |
|---|---|---|---|---|
| yolo11n v1 (deployed) | 19 | 25/26 | 22/25 | the baseline |
| yolo11n v2 | 19 | 25/26 | 22/25 | same headline, different errors: fixed the clipped 120, lost 801 |
| yolo11s v2 | 22 | 25/26 | 22/24 | lost the clipped-120 crossing outright; found the 17-45-02 crossing n never has |
| yolo11m v2 | 30 | **26/26** | **23/25** | reads 801; one extra no-bib detection of a bystander at the crop's left edge (a boundary-gate matter) |

"found" counts use the corrected 17-45-02 expectation (the runner crosses at
~16.3 s, not on the final frame at 22 s; s and m detect it, n does not). The
two 225 detections on 14-42-58 are real crossings the expectations omit and
are excluded above. 531 and 76 on the backyard clip are unread by every
model: fingers over the digits.

What this means for deployment: the val table's three-way tie resolves in
yolo11m's favour end to end, by one crossing and one bib. On the Mac, m's
30 ms/frame leaves no headroom at 30 fps (n runs at 19 ms and still drops
~3% of live frames), so the Mac stays on yolo11n unless it runs at 15 fps.
On the Jetson, where TensorRT FP16 should put m well inside the budget, m is
the model to benchmark first; the weights are `models/gpu_runs_v2/yolo11m_1280/weights/best.pt`.
