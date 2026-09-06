# Training the bib detector on a GCP GPU

Local training is not viable for this. On the M-series MPS backend one batch of
8 at 960px took **264 seconds**, which puts 150 epochs at roughly 18 days. The
same run on an L4 is minutes. The dataset is small (376 labelled images), so
this is a short, cheap session rather than a standing cost.

Project already configured: `stunning-vertex-437612-f6`
Quota confirmed available: `NVIDIA_L4_GPUS = 1` in `us-central1`, `GPUS_ALL_REGIONS = 4`.

## 0. Build the dataset first (local, seconds)

```bash
python scripts/build_dataset.py
tar -czf /tmp/bib_dataset.tar.gz -C data dataset
```

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
gcloud compute scp /tmp/bib_dataset.tar.gz bib-train:~ --zone=us-central1-a
gcloud compute scp scripts/train.py bib-train:~ --zone=us-central1-a
gcloud compute scp config/yolo_dataset.yaml bib-train:~ --zone=us-central1-a
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
python3 -m pip install -q ultralytics
python3 -m pip uninstall -y -q opencv-python opencv-contrib-python
python3 -m pip install -q 'opencv-python-headless<5'
python3 -c "import cv2, ultralytics, torch; print(cv2.__version__, ultralytics.__version__, torch.cuda.is_available())"

tar -xzf bib_dataset.tar.gz
mkdir -p config && mv yolo_dataset.yaml config/
# The generated config has an absolute path from the laptop; point it at the VM copy.
sed -i "s|^path: .*|path: $HOME/dataset|" config/yolo_dataset.yaml

# The matrix worth running: resolution is the open question, capacity is second.
python3 train.py \
  --models yolo11n,yolo11s \
  --imgsz 640,960,1280 \
  --epochs 150 \
  --batch 16 \
  --device 0 \
  --export coreml
```

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

```bash
gcloud compute scp --recurse bib-train:~/models ./models/gpu_runs --zone=us-central1-a
gcloud compute scp bib-train:~/training_results.json . --zone=us-central1-a
```

## 5. Delete the VM

An idle L4 bills at the same rate as a busy one.

```bash
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
