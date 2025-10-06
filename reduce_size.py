import cv2
import argparse
import os

def resize_video(input_path, output_path, width, height):
    """
    Resizes a video file to a new width and height.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the resized output video file.
        width (int): The new width for the video.
        height (int): The new height for the video.
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

    print(f"Original video: {original_width}x{original_height} @ {original_fps:.2f} FPS")
    print(f"Target resolution: {width}x{height}")

    # --- 4. Define Video Writer ---
    # Define the codec for the output file (MP4V is a good choice for .mp4 files)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))
    if not writer.isOpened():
        print(f"Error: Could not create video writer for '{output_path}'")
        cap.release()
        return

    # --- 5. Loop, Resize, and Write Frames ---
    processed_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video
            break

        # Resize the frame
        resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        # Write the resized frame to the output file
        writer.write(resized_frame)

        processed_frames += 1
        # Print progress update
        print(f"\rProcessing frame {processed_frames}/{frame_count}...", end="")

    print(f"\nSuccessfully resized video and saved to '{output_path}'")

    # --- 6. Release Resources ---
    cap.release()
    writer.release()
    print("Resources released.")
 
if __name__ == "__main__":
    # --- Setup Command-Line Argument Parser ---
    parser = argparse.ArgumentParser(description="Resize a video to a specified resolution.")
    parser.add_argument("input", help="Path to the input video file.")
    parser.add_argument("output", help="Path to save the output resized video file.")
    parser.add_argument("--width", type=int, default=1280, help="New width of the video (default: 1280).")
    parser.add_argument("--height", type=int, default=720, help="New height of the video (default: 720).")

    args = parser.parse_args()

    # --- Call the main function ---
    resize_video(args.input, args.output, args.width, args.height)
