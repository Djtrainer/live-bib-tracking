docker build -f docker/Dockerfile -t live-bib-tracking-fullstack .

 docker run --rm \
    --env-file .env \
    -v /Users/dantrainer/projects/live-bib-tracking/data:/app/data \
    -v /Users/dantrainer/projects/live-bib-tracking/runs/detect/yolo11_white_bibs/weights:/app/models \
    live-bib-tracking-fullstack \
    python src/image_processor/video_inference.py \
    --video /app/data/raw/race_1080p.mp4 \
    --model /app/models/last.pt 
