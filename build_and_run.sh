docker build -f docker/Dockerfile -t live-bib-tracking .

 docker run --rm \
  -v /Users/dantrainer/projects/live-bib-tracking/data:/app/data \
  -v /Users/dantrainer/projects/live-bib-tracking/runs/detect/train2/weights:/app/models \
  live-bib-tracking \
  python src/image_processor/video_inference.py \
  --video /app/data/raw/2024_race.MOV \
  --model /app/models/last.pt