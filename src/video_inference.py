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
from collections import Counter
from pathlib import Path
from ultralytics import YOLO
import cv2
import easyocr
import time

class VideoInferenceProcessor:
    
    def __init__(self, model_path, video_path, target_fps=1, confidence_threshold=0.25):
        self.model_path = Path(model_path)
        self.video_path = Path(video_path)
        self.target_fps = target_fps
        self.confidence_threshold = confidence_threshold
        # This will store history keyed by the PERSON's tracker ID
        self.track_history = {}

        # --- KEY CHANGE: LOAD TWO SEPARATE MODEL INSTANCES ---
        print("Loading models...")
        # This model instance will be used ONLY for tracking and will become stateful
        self.tracker_model = YOLO(str(self.model_path))
        # This model instance will be used ONLY for prediction and will remain stateless
        self.predictor_model = YOLO(str(self.model_path))
        print("Models loaded successfully!")
        
        # Initialize EasyOCR reader
        print("Initializing EasyOCR reader...")
        self.ocr_reader = easyocr.Reader(["en"])
        print("EasyOCR reader initialized!")

        # Video capture and properties
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_skip = max(1, int(self.original_fps / self.target_fps))

    # --- NO CHANGES to helper functions below ---
    def preprocess_for_easyocr(self, image_crop):
        if len(image_crop.shape) == 3: gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else: gray = image_crop
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        return cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def extract_bib_with_easyocr(self, image_crop):
        try:
            result = self.ocr_reader.readtext(image_crop, allowlist="0123456789")
            if result:
                _, text, confidence = max(result, key=lambda x: x[2])
                return text, confidence
            return None, None
        except Exception: return None, None

    def crop_bib_from_prediction(self, image, bbox, padding=15):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(image.shape[1], x2 + padding), min(image.shape[0], y2 + padding)
        return image[y1:y2, x1:x2]

    # --- HYBRID LOGIC (NO CHANGES) ---
    def update_history_hybrid(self, tracked_persons, all_detections):
        if tracked_persons is None or tracked_persons.boxes.id is None or all_detections is None:
            return

        bib_boxes = [box for box in all_detections.boxes if int(box.cls) == 1]
        if not bib_boxes: return

        for person_box in tracked_persons.boxes:
            person_id = int(person_box.id[0])
            px1, py1, px2, py2 = person_box.xyxy[0]

            for bib_box in bib_boxes:
                bx1, by1, bx2, by2 = bib_box.xyxy[0]
                if px1 < (bx1 + bx2) / 2 < px2 and py1 < (by1 + by2) / 2 < py2:
                    if person_id not in self.track_history:
                        self.track_history[person_id] = {'ocr_reads': []}
                    bib_crop = self.crop_bib_from_prediction(all_detections.orig_img, bib_box.xyxy[0])
                    if bib_crop.size > 0:
                        bib_number, ocr_conf = self.extract_bib_with_easyocr(self.preprocess_for_easyocr(bib_crop))
                        if bib_number and ocr_conf:
                            self.track_history[person_id]['ocr_reads'].append((bib_number, ocr_conf, float(bib_box.conf)))
                    break

    def draw_hybrid_predictions(self, image, tracked_persons, all_detections):
        annotated_image = image.copy()
        if all_detections:
            for box in all_detections.boxes:
                if int(box.cls) == 1:
                    x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        if tracked_persons and tracked_persons.boxes.id is not None:
            for box in tracked_persons.boxes:
                person_id = int(box.id[0])
                x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
                current_best_read = ""
                if person_id in self.track_history and self.track_history[person_id]['ocr_reads']:
                    reads = [r[0] for r in self.track_history[person_id]['ocr_reads']]
                    if reads: current_best_read = Counter(reads).most_common(1)[0][0]
                label_text = f"Racer ID {person_id} | Bib: {current_best_read}"
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return annotated_image

    def determine_final_bibs(self):
        """
        Processes the aggregated track_history to determine the final bib for each racer.
        This version contains the fix for the 'scores' variable error.
        """
        final_results = {}
        for tracker_id, data in self.track_history.items():
            ocr_reads = data.get('ocr_reads', [])
            if not ocr_reads:
                continue

            # Filter out unreliable reads (e.g., wrong length, low confidence)
            filtered_reads = [r for r in ocr_reads if 2 <= len(r[0]) <= 5 and r[1] > 0.4]
            if not filtered_reads:
                continue

            # --- CORRECTED LOGIC FOR CALCULATING SCORES ---
            # 1. First, initialize an empty dictionary.
            scores = {}
            # 2. Then, loop through the filtered reads to populate it.
            for bib_num, ocr_conf, yolo_conf in filtered_reads:
                # The score for this single reading combines OCR and YOLO confidence
                score = ocr_conf * yolo_conf
                # Add this score to the cumulative total for that bib number
                scores[bib_num] = scores.get(bib_num, 0) + score
            # --- END OF CORRECTION ---

            # Find the bib number with the highest total score
            if scores:
                most_likely_bib = max(scores, key=scores.get)
                final_results[tracker_id] = {
                    'final_bib': most_likely_bib,
                    'score': scores[most_likely_bib]
                }
                
        return final_results

    def process_video(self, output_path=None, display=True):
        # ... video writer setup ...
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None
        if output_path:
            out = cv2.VideoWriter(output_path, fourcc, self.target_fps, (self.frame_width, self.frame_height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret: break
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue

                # --- HYBRID MODEL CALLS USING TWO SEPARATE INSTANCES ---
                # 1. Track ONLY persons using the dedicated tracker_model
                person_results = self.tracker_model.track(frame, persist=True, classes=[0], verbose=False)
                tracked_persons = person_results[0] if person_results else None

                # 2. Predict ALL objects using the separate, stateless predictor_model
                all_detections_results = self.predictor_model.predict(frame, conf=self.confidence_threshold, verbose=False)
                all_detections = all_detections_results[0] if all_detections_results else None

                # 3. Update history using the new hybrid function
                self.update_history_hybrid(tracked_persons, all_detections)

                # 4. Draw predictions using the new hybrid function
                annotated_frame = self.draw_hybrid_predictions(frame, tracked_persons, all_detections)
                
                if display:
                    cv2.imshow("Live Bib Tracking", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"): break
                if out: out.write(annotated_frame)
                frame_count += 1

        finally:
            final_results = self.determine_final_bibs()
            print("\n--- Final Bib Number Results ---")
            if final_results:
                for tracker_id, data in final_results.items():
                    print(f"  Racer ID {tracker_id}: Final Bib = {data['final_bib']} (Score: {data['score']:.2f})")
            else:
                print("  No reliable bib numbers were finalized.")
            self.cap.release()
            if out: out.release()
            cv2.destroyAllWindows()
            

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
