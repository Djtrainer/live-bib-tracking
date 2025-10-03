import cv2
import argparse
import os

def convert_video_fps(input_path, output_path, target_fps):
    """
    Converts a video file to a new target frame rate by dropping or duplicating frames.

    Args: 
        input_path (str): Path to the input video file.
        output_path (str): Path to save the converted output video file.
        target_fps (int): The new target frame rate for the video.
    """
    # --- 1. Input Validation ---
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    # --- 2. Open Video Capture ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'")
        return

    # --- 3. Get Original Video Properties ---
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_fps == 0:
        print("Error: Could not determine original FPS. Aborting.")
        cap.release()
        return

    print(f"Original video: {original_width}x{original_height} @ {original_fps:.2f} FPS")
    print(f"Target frame rate: {target_fps} FPS")

    # --- 4. Define Video Writer ---
    # Define the codec for the output file (MP4V is a good choice for .mp4 files)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, target_fps, (original_width, original_height))
    if not writer.isOpened():
        print(f"Error: Could not create video writer for '{output_path}'")
        cap.release()
        return

    # --- 5. Loop and Write Frames ---
    # The VideoWriter handles the timing. By reading every frame from the source
    # and writing it, the writer will automatically drop the necessary frames
    # to match the lower target FPS.
    processed_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video
            break

        writer.write(frame)

        processed_frames += 1
        # Print progress update
        print(f"\rProcessing frame {processed_frames}/{frame_count}...", end="")

    print(f"\nSuccessfully converted video and saved to '{output_path}'")

    # --- 6. Release Resources ---
    cap.release()
    writer.release()
    print("Resources released.")

if __name__ == "__main__":
    # --- Setup Command-Line Argument Parser ---
    parser = argparse.ArgumentParser(description="Convert the frame rate of a video.")
    parser.add_argument("input", help="Path to the input video file.")
    parser.add_argument("output", help="Path to save the output converted video file.")
    parser.add_argument("--fps", type=int, default=30, help="The target frame rate (default: 30).")

    args = parser.parse_args()

    # --- Call the main function ---
    convert_video_fps(args.input, args.output, args.fps)
