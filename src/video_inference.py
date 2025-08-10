#!/usr/bin/env python3
"""
Video Inference Script for Live Bib Tracking

This script processes a video file, runs YOLO inference to detect people and bibs,
performs OCR on detected bibs, and displays the results with bounding boxes and labels.

Usage:
    python src/video_inference.py --video data/raw/2024_race.MOV --fps 5 --conf 0.25
"""

import argparse
import cv2
import easyocr
from pathlib import Path
from ultralytics import YOLO
import time


class VideoInferenceProcessor:
    
    def __init__(self, model_path, video_path, target_fps=1, confidence_threshold=0.25):
        """
        Initialize the video inference processor.

        Args:
            model_path (str): Path to the trained YOLO model
            video_path (str): Path to the input video file
            target_fps (int): Target processing frame rate
            confidence_threshold (float): YOLO confidence threshold
        """
        self.model_path = Path(model_path)
        self.video_path = Path(video_path)
        self.target_fps = target_fps
        self.confidence_threshold = confidence_threshold
        self.track_history = {}

        # Load YOLO model
        print(f"Loading YOLO model from {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print("Model loaded successfully!")
        print(f"Model classes: {self.model.names}")

        # Initialize EasyOCR reader
        print("Initializing EasyOCR reader...")
        self.ocr_reader = easyocr.Reader(["en"])
        print("EasyOCR reader initialized!")

        # Video capture
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        # Get video properties
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print("Video properties:")
        print(f"  Original FPS: {self.original_fps:.2f}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  Target processing FPS: {self.target_fps}")

        # Calculate frame skip interval
        self.frame_skip = max(1, int(self.original_fps / self.target_fps))
        print(f"  Processing every {self.frame_skip} frames")

    def preprocess_for_easyocr(self, image_crop):
        """
        Preprocess cropped bib image for better OCR results.

        Args:
            image_crop (np.ndarray): Cropped bib image

        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop

        # Apply bilateral filter for noise reduction while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Apply adaptive thresholding for better contrast
        binary_image = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return binary_image

    def extract_bib_with_easyocr(self, image_crop):
        """
        Extract bib number using EasyOCR.

        Args:
            image_crop (np.ndarray): Cropped bib image

        Returns:
            tuple: (bib_number, confidence) or (None, None) if no text found
        """
        try:
            # Use EasyOCR to detect text (numbers only)
            result = self.ocr_reader.readtext(image_crop, allowlist="0123456789")

            if result:
                # Get the result with highest confidence
                best_result = max(result, key=lambda x: x[2])
                bbox, text, confidence = best_result
                return text, confidence
            else:
                return None, None

        except Exception as e:
            print(f"EasyOCR error: {e}")
            return None, None

    def crop_bib_from_prediction(self, image, bbox, padding=15):
        """
        Crop bib region from image with padding.

        Args:
            image (np.ndarray): Input image
            bbox (list): Bounding box coordinates [x1, y1, x2, y2]
            padding (int): Padding around the bounding box

        Returns:
            np.ndarray: Cropped image region
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)

        return image[y1:y2, x1:x2]

    def draw_predictions(self, image, predictions):
        """
        Draw bounding boxes and labels on the image.

        Args:
            image (np.ndarray): Input image
            predictions: YOLO prediction results

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        if predictions is None or len(predictions.boxes) == 0:
            return annotated_image

        boxes = predictions.boxes.xyxy.cpu().numpy()
        classes = predictions.boxes.cls.cpu().numpy()
        confidences = predictions.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            class_id = int(cls)
            class_name = self.model.names.get(class_id, f"Class {class_id}")

            # Set colors: blue for person (class 0), red for bib (class 1)
            if class_id == 0:  # Person
                color = (255, 0, 0)  # Blue in BGR
                label = f"Person: {conf:.2f}"
            elif class_id == 1:  # Bib
                color = (0, 0, 255)  # Red in BGR

                # Crop bib region and perform OCR
                bib_crop = self.crop_bib_from_prediction(image, box)
                if bib_crop.size > 0:
                    bib_number, ocr_confidence = self.extract_bib_with_easyocr(bib_crop)
                    if bib_number:
                        label = (
                            f"Bib {bib_number}: {conf:.2f} (OCR: {ocr_confidence:.2f})"
                        )
                    else:
                        label = f"Bib: {conf:.2f} (OCR: Failed)"
                else:
                    label = f"Bib: {conf:.2f}"
            else:
                color = (0, 255, 0)  # Green for other classes
                label = f"{class_name}: {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                annotated_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        return annotated_image

    def process_video(self, output_path=None, display=True):
        """
        Process the video file with YOLO inference and OCR.

        Args:
            output_path (str, optional): Path to save the output video
            display (bool): Whether to display the video in real-time
        """
        # Setup video writer if output path is provided
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None
        if output_path:
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                self.target_fps,
                (self.frame_width, self.frame_height),
            )

        frame_count = 0
        processed_frames = 0
        start_time = time.time()

        print("\nStarting video processing...")
        print("Press 'q' to quit, 'p' to pause/resume")

        paused = False

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Skip frames to achieve target FPS
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue

                # Run YOLO inference
                results = self.model(
                    frame, conf=self.confidence_threshold, verbose=False
                )
                predictions = results[0] if results else None

                # Draw predictions and OCR results
                annotated_frame = self.draw_predictions(frame, predictions)

                # Add frame info
                info_text = (
                    f"Frame: {frame_count}/{self.total_frames} | "
                    f"Processed: {processed_frames} | "
                    f"FPS: {processed_frames / (time.time() - start_time):.1f}"
                )
                cv2.putText(
                    annotated_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Save frame if output video is specified
                if out:
                    out.write(annotated_frame)

                # Display frame
                if display:
                    cv2.imshow("Live Bib Tracking", annotated_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("p"):
                        paused = not paused
                        if paused:
                            print("Video paused. Press 'p' to resume.")
                        else:
                            print("Video resumed.")

                    while paused:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("p"):
                            paused = False
                            print("Video resumed.")
                            break
                        elif key == ord("q"):
                            return

                frame_count += 1
                processed_frames += 1

                # Print progress every 100 processed frames
                if processed_frames % 100 == 0:
                    elapsed_time = time.time() - start_time
                    fps = processed_frames / elapsed_time
                    progress = (frame_count / self.total_frames) * 100
                    print(
                        f"Progress: {progress:.1f}% | "
                        f"Processing FPS: {fps:.1f} | "
                        f"Processed frames: {processed_frames}"
                    )

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")

        finally:
            # Cleanup
            self.cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()

            # Print final statistics
            total_time = time.time() - start_time
            avg_fps = processed_frames / total_time if total_time > 0 else 0
            print("\nProcessing completed!")
            print(f"Total frames processed: {processed_frames}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average processing FPS: {avg_fps:.2f}")
            if output_path:
                print(f"Output video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Video Inference for Live Bib Tracking"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/raw/2024_race.MOV",
        help="Path to input video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/train2/weights/last.pt",
        help="Path to trained YOLO model",
    )
    parser.add_argument(
        "--fps", type=int, default=5, help="Target processing frame rate"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="YOLO confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save output video (optional)"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Disable real-time video display"
    )

    args = parser.parse_args()

    # Validate input files
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return

    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return

    # Create processor and run inference
    try:
        processor = VideoInferenceProcessor(
            model_path=args.model,
            video_path=args.video,
            target_fps=args.fps,
            confidence_threshold=args.conf,
        )

        processor.process_video(output_path=args.output, display=not args.no_display)

    except Exception as e:
        print(f"Error during processing: {e}")
        return


if __name__ == "__main__":
    main()
