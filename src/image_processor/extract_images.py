import cv2
import os
from typing import Optional
import logging
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_images_from_video(
    video_path: str,
    output_dir: str,
    frame_interval: int = 1,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    image_format: str = 'jpg',
    image_quality: int = 95,
    yolo_confidence_threshold: float = 0.5
) -> int:
    """
    Extract images from a video file, saving only frames where a person
    is detected in the rightmost 1/4 of the screen.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save extracted images.
        frame_interval (int): Process every nth frame for detection (default: 1).
        start_time (float, optional): Start time in seconds.
        end_time (float, optional): End time in seconds.
        image_format (str): Output image format ('jpg', 'png').
        image_quality (int): JPEG quality (0-100).
        yolo_confidence_threshold (float): Confidence threshold for person detection.
    
    Returns:
        int: Number of images extracted.
    
    Raises:
        FileNotFoundError: If video file doesn't exist.
        ValueError: If video cannot be opened or parameters are invalid.
    """
    
    # --- Input Validation ---
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if frame_interval < 1:
        raise ValueError("Frame interval must be >= 1")
    if not 0 <= image_quality <= 100:
        raise ValueError("Image quality must be between 0 and 100")
    if not 0 < yolo_confidence_threshold < 1:
        raise ValueError("YOLO confidence threshold must be between 0 and 1")

    # --- Initialization ---
    os.makedirs(output_dir, exist_ok=True)
    
    # Load a pre-trained YOLO model (yolov8n.pt is small and fast)
    logger.info("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    try:
        # --- Video Properties and Frame Calculation ---
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info("Video properties:")
        logger.info(f"  - FPS: {fps:.2f}, Total frames: {total_frames}, Duration: {duration:.2f}s")
        
        start_frame = int(start_time * fps) if start_time is not None else 0
        end_frame = int(end_time * fps) if end_time is not None else total_frames
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)
        
        if start_frame >= end_frame:
            raise ValueError("Start time must be less than end time")
            
        logger.info(f"Processing frames from {start_frame} to {end_frame}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Define the region of interest (rightmost 1/4 of the frame)
        roi_start_x = frame_width * 0.5
        
        # --- Frame Processing Loop ---
        frame_count = 0
        extracted_count = 0
        current_frame_idx = start_frame
        
        # Set up encoding parameters
        extension = f'.{image_format.lower()}'
        if image_format.lower() in ['jpg', 'jpeg']:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, image_quality]
        elif image_format.lower() == 'png':
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        else:
            encode_params = []

        while current_frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {current_frame_idx}")
                break
            
            # Only process the frame for detection at the specified interval
            if frame_count % frame_interval == 0:
                # Run YOLO detection
                results = model(frame, conf=yolo_confidence_threshold, classes=[0], verbose=False) # class 0 is 'person'
                
                should_save = False
                # Check results for any person in the ROI
                for result in results:
                    for box in result.boxes:
                        # Check if the detected object is a person (class 0)
                        if int(box.cls[0]) == 0:
                            # Get bounding box coordinates
                            x1, _, x2, _ = box.xyxy[0]
                            # Check if the center of the box is in the rightmost 1/4
                            box_center_x = (x1 + x2) / 2
                            if box_center_x >= roi_start_x:
                                should_save = True
                                break  # Found a person in the ROI, no need to check others
                    if should_save:
                        break
                
                # Save the frame if the condition was met
                if should_save:
                    timestamp = current_frame_idx / fps
                    filename = f"frame_{current_frame_idx:06d}_t{timestamp:.3f}s{extension}"
                    output_path = os.path.join(output_dir, filename)
                    
                    if cv2.imwrite(output_path, frame, encode_params):
                        extracted_count += 1
                        if extracted_count % 50 == 0:
                            logger.info(f"Extracted {extracted_count} images so far...")
                    else:
                        logger.warning(f"Failed to save image: {output_path}")

            frame_count += 1
            current_frame_idx += 1
            
        logger.info(f"Successfully extracted {extracted_count} images to {output_dir}")
        return extracted_count
        
    finally:
        cap.release()
        logger.info("Video capture released.")


if __name__ == "__main__":
    # --- Configuration ---
    video_file = "data/raw/race_1080p.mp4"       # CHANGE THIS to your video file
    output_directory = "data/processed/person_in_roi_frames" # CHANGE THIS to your desired output folder
    
    try:
        num_extracted = extract_images_from_video(
            video_path=video_file,
            output_dir=output_directory,
            frame_interval=15,          # Check every 15th frame
            image_format='jpg',
            image_quality=90,
            yolo_confidence_threshold=0.4 # Lower confidence for potentially less clear shots
        )
        print(f"\nExtraction complete! {num_extracted} images saved.")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)