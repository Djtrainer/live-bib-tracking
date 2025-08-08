import cv2
import os
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_images_from_video(
    video_path: str,
    output_dir: str,
    frame_interval: int = 1,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    image_format: str = 'jpg',
    image_quality: int = 95
) -> int:
    """
    Extract images from a video file (.MOV or other formats supported by OpenCV).
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted images
        frame_interval (int): Extract every nth frame (default: 1 = every frame)
        start_time (float, optional): Start time in seconds (default: from beginning)
        end_time (float, optional): End time in seconds (default: until end)
        image_format (str): Output image format ('jpg', 'png', etc.)
        image_quality (int): JPEG quality (0-100, only for jpg format)
    
    Returns:
        int: Number of images extracted
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be opened or parameters are invalid
    """
    
    # Validate inputs
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if frame_interval < 1:
        raise ValueError("Frame interval must be >= 1")
    
    if image_quality < 0 or image_quality > 100:
        raise ValueError("Image quality must be between 0 and 100")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    try:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info("Video properties:")
        logger.info(f"  - FPS: {fps}")
        logger.info(f"  - Total frames: {total_frames}")
        logger.info(f"  - Duration: {duration:.2f} seconds")
        
        # Calculate start and end frames
        start_frame = int(start_time * fps) if start_time is not None else 0
        end_frame = int(end_time * fps) if end_time is not None else total_frames
        
        # Ensure frame bounds are valid
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)
        
        if start_frame >= end_frame:
            raise ValueError("Start time must be less than end time")
        
        logger.info(f"Extracting frames {start_frame} to {end_frame}")
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = 0
        extracted_count = 0
        current_frame = start_frame
        
        # Set up encoding parameters
        if image_format.lower() == 'jpg' or image_format.lower() == 'jpeg':
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, image_quality]
            extension = '.jpg'
        elif image_format.lower() == 'png':
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
            extension = '.png'
        else:
            encode_params = []
            extension = f'.{image_format.lower()}'
        
        while current_frame < end_frame:
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame {current_frame}")
                break
            
            # Extract frame at specified intervals
            if frame_count % frame_interval == 0:
                # Generate filename
                timestamp = current_frame / fps
                filename = f"frame_{current_frame:06d}_t{timestamp:.3f}s{extension}"
                output_path = os.path.join(output_dir, filename)
                
                # Save image
                success = cv2.imwrite(output_path, frame, encode_params)
                
                if success:
                    extracted_count += 1
                    if extracted_count % 100 == 0:  # Log progress every 100 images
                        logger.info(f"Extracted {extracted_count} images...")
                else:
                    logger.warning(f"Failed to save image: {output_path}")
            
            frame_count += 1
            current_frame += 1
        
        logger.info(f"Successfully extracted {extracted_count} images to {output_dir}")
        return extracted_count
        
    finally:
        cap.release()


def extract_images_from_mov(
    mov_path: str,
    output_dir: str,
    **kwargs
) -> int:
    """
    Convenience function specifically for .MOV files.
    
    Args:
        mov_path (str): Path to the .MOV file
        output_dir (str): Directory to save extracted images
        **kwargs: Additional arguments passed to extract_images_from_video
    
    Returns:
        int: Number of images extracted
    """
    return extract_images_from_video(mov_path, output_dir, **kwargs)


if __name__ == "__main__":
    # Example usage
    video_file = "../data/raw/IMG_0053.MOV"
    output_directory = "../data/processed/extracted_frames"
    
    try:
        # Extract every 30th frame (roughly 1 frame per second for 30fps video)
        num_extracted = extract_images_from_mov(
            mov_path=video_file,
            output_dir=output_directory,
            frame_interval=30,
            image_format='jpg',
            image_quality=90
        )
        print(f"Extraction complete! {num_extracted} images saved.")
        
    except Exception as e:
        print(f"Error: {e}")