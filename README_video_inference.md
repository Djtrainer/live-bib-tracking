# Video Inference Script for Live Bib Tracking

This script processes video files to detect people and race bibs using a trained YOLO model, and performs OCR on detected bibs to read the numbers.

## Features

- **YOLO Object Detection**: Detects people (class 0) and bibs (class 1)
- **OCR Integration**: Uses EasyOCR to read bib numbers from detected bib regions
- **Configurable Frame Rate**: Process video at custom frame rates to balance speed vs accuracy
- **Real-time Display**: Shows processed video with bounding boxes and labels
- **Video Output**: Optionally save the processed video to file
- **Interactive Controls**: Pause/resume with 'p', quit with 'q'

## Installation

Make sure you have the required dependencies:

```bash
pip install ultralytics opencv-python easyocr numpy
```

## Usage

### Basic Usage
```bash
python src/video_inference.py
```

### Custom Parameters
```bash
python src/video_inference.py \
    --video data/raw/2024_race.MOV \
    --model runs/detect/train2/weights/last.pt \
    --fps 5 \
    --conf 0.25 \
    --output output_video.mp4
```

### Parameters

- `--video`: Path to input video file (default: `data/raw/2024_race.MOV`)
- `--model`: Path to trained YOLO model (default: `runs/detect/train2/weights/last.pt`)
- `--fps`: Target processing frame rate (default: 5)
- `--conf`: YOLO confidence threshold (default: 0.25)
- `--output`: Path to save output video (optional)
- `--no-display`: Disable real-time video display

### Interactive Controls

While the video is playing:
- Press `q` to quit
- Press `p` to pause/resume

## Output

The script displays:
- **Blue boxes**: People detections with confidence scores
- **Red boxes**: Bib detections with:
  - YOLO confidence score
  - OCR-detected bib number (if successful)
  - OCR confidence score
- **Frame information**: Current frame, processed frames, and processing FPS

## Performance Tips

1. **Frame Rate**: Lower FPS (3-5) for better accuracy, higher FPS (10-15) for faster processing
2. **Confidence Threshold**: Lower values (0.1-0.2) detect more objects but may include false positives
3. **Video Resolution**: Smaller videos process faster but may reduce detection accuracy

## Example Output

```
Loading YOLO model from runs/detect/train2/weights/last.pt
Model loaded successfully!
Model classes: {0: 'person', 1: 'bib'}
Initializing EasyOCR reader...
EasyOCR reader initialized!
Video properties:
  Original FPS: 29.97
  Total frames: 14985
  Resolution: 1920x1080
  Target processing FPS: 5
  Processing every 6 frames

Starting video processing...
Press 'q' to quit, 'p' to pause/resume
Progress: 10.0% | Processing FPS: 4.2 | Processed frames: 100
...
```

## Troubleshooting

1. **Video not found**: Check the video file path
2. **Model not found**: Ensure the YOLO model exists at the specified path
3. **Slow processing**: Reduce target FPS or video resolution
4. **OCR errors**: EasyOCR may take time to initialize on first run
5. **Memory issues**: Process shorter video segments or reduce frame rate

## Technical Details

- **Object Detection**: Uses YOLOv8 for real-time object detection
- **OCR**: EasyOCR with number-only allowlist for bib number extraction
- **Image Processing**: Adaptive thresholding and bilateral filtering for better OCR accuracy
- **Video Processing**: Configurable frame skipping to achieve target processing rates
