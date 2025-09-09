docker build -f docker/Dockerfile -t live-bib-tracking-fullstack .

 docker run --rm \
   --gpus all \
   --env-file .env \
   -v /home/ec2-user/live-bib-tracking/data:/app/data \
   -v /home/ec2-user/live-bib-tracking/runs/detect/train2/weights:/app/models \
   live-bib-tracking-fullstack \
   python src/image_processor/video_inference.py \
   --video /app/data/raw/2024_race.MOV \
   --model /app/models/last.pt \
   --no-display
