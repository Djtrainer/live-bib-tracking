import os
import argparse
import cv2
import easyocr
from pathlib import Path
from ultralytics import YOLO
import time
from collections import Counter


class VideoInferenceProcessor:
    def __init__(
        self,
        model_path,
        video_path,
        target_fps=1,
        confidence_threshold=0,
        finish_line_fraction=0.85,
    ):
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
        self.finish_line_x = int(self.frame_width * finish_line_fraction)

        self.inference_interval = 1
        self.last_annotated_frame = None

    def preprocess_for_easyocr(self, image_crop):
        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        return cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    def extract_bib_with_easyocr(self, image_crop):
        try:
            result = self.ocr_reader.readtext(image_crop, allowlist="0123456789")
            if result:
                _, text, confidence = max(result, key=lambda x: x[2])
                return text, confidence
            return None, None
        except Exception:
            return None, None

    def crop_bib_from_prediction(self, image, bbox, padding=15):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(image.shape[1], x2 + padding), min(image.shape[0], y2 + padding)
        return image[y1:y2, x1:x2]

    def check_finish_line_crossings(self, tracked_persons):
        """Checks if any tracked racers have crossed the virtual finish line."""
        if tracked_persons is None or tracked_persons.boxes.id is None:
            return

        for person_box in tracked_persons.boxes:
            person_id = int(person_box.id[0])
            # Use the center of the bounding box as the racer's position
            x1, _, x2, _ = person_box.xyxy[0]
            current_x_center = (x1 + x2) / 2

            history = self.track_history.get(person_id)
            # Proceed only if the racer is already being tracked and hasn't finished yet
            if history and history["finish_time_ms"] is None:
                last_x_center = history.get("last_x_center")

                # Check if the racer crossed the line from left to right
                if (
                    last_x_center is not None
                    and last_x_center < self.finish_line_x
                    and current_x_center >= self.finish_line_x
                ):
                    # Racer crossed the line! Record the video timestamp.
                    finish_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                    history["finish_time_ms"] = finish_time
                    print(f"Racer ID {person_id} finished at {finish_time / 1000:.2f}s")
                    self.print_live_leaderboard()
                # Update the last known position for the next frame
                history["last_x_center"] = current_x_center

    def update_history_hybrid(self, tracked_persons, all_detections):
        if (
            tracked_persons is None
            or tracked_persons.boxes.id is None
            or all_detections is None
        ):
            return

        bib_boxes = [box for box in all_detections.boxes if int(box.cls) == 1]
        if not bib_boxes:
            return

        for person_box in tracked_persons.boxes:
            person_id = int(person_box.id[0])
            px1, py1, px2, py2 = person_box.xyxy[0]

            for bib_box in bib_boxes:
                bx1, by1, bx2, by2 = bib_box.xyxy[0]
                if px1 < (bx1 + bx2) / 2 < px2 and py1 < (by1 + by2) / 2 < py2:
                    if person_id not in self.track_history:
                        # Initialize with new fields for timing
                        self.track_history[person_id] = {
                            "ocr_reads": [],
                            "last_x_center": None,
                            "finish_time_ms": None,
                        }
                    bib_crop = self.crop_bib_from_prediction(
                        all_detections.orig_img, bib_box.xyxy[0]
                    )
                    if bib_crop.size > 0:
                        bib_number, ocr_conf = self.extract_bib_with_easyocr(
                            self.preprocess_for_easyocr(bib_crop)
                        )
                        if bib_number and ocr_conf:
                            self.track_history[person_id]["ocr_reads"].append(
                                (bib_number, ocr_conf, float(bib_box.conf))
                            )
                    break

    def draw_hybrid_predictions(
        self, image, tracked_persons, all_detections, scale=1.0
    ):
        annotated_image = image.copy()
        cv2.line(
            annotated_image,
            (self.finish_line_x, 0),
            (self.finish_line_x, self.frame_height),
            (0, 255, 255),
            3,
        )

        if all_detections:
            for box in all_detections.boxes:
                if int(box.cls) == 1:
                    x1, y1, x2, y2 = [int(c / scale) for c in box.xyxy[0]]
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if tracked_persons and tracked_persons.boxes.id is not None:
            for box in tracked_persons.boxes:
                person_id = int(box.id[0])
                x1, y1, x2, y2 = [int(c / scale) for c in box.xyxy[0]]
                current_best_read = ""
                if (
                    person_id in self.track_history
                    and self.track_history[person_id]["ocr_reads"]
                ):
                    reads = [r[0] for r in self.track_history[person_id]["ocr_reads"]]
                    if reads:
                        current_best_read = Counter(reads).most_common(1)[0][0]
                label_text = f"Racer ID {person_id} | Bib: {current_best_read}"
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    annotated_image,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2,
                )
        return annotated_image

    def determine_final_bibs(self):
        """
        Processes the aggregated track_history to determine the final bib for each racer.
        This version contains the fix for the 'scores' variable error.
        """
        final_results = {}
        for tracker_id, data in self.track_history.items():
            ocr_reads = data.get("ocr_reads", [])
            if not ocr_reads:
                continue

            # Filter out unreliable reads (e.g., wrong length, low confidence)
            filtered_reads = [
                r for r in ocr_reads if 2 <= len(r[0]) <= 5 and r[1] > 0.4
            ]
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
                    "final_bib": most_likely_bib,
                    "score": scores[most_likely_bib],
                }

        return final_results

    def print_live_leaderboard(self):
        """
        Clears the terminal and prints the current state of the leaderboard.
        """
        # 1. Clear the terminal screen (works on Windows, macOS, and Linux)
        os.system("cls" if os.name == "nt" else "clear")

        # 2. Get the most up-to-date bib number guesses
        current_bib_results = self.determine_final_bibs()

        # 3. Assemble the list of finished racers
        leaderboard = []
        for tracker_id, history_data in self.track_history.items():
            # Check if this racer has a recorded finish time
            if history_data and history_data.get("finish_time_ms") is not None:
                bib_result = current_bib_results.get(tracker_id)
                # If the bib number is determined, use it. Otherwise, show "Pending".
                bib_number = bib_result["final_bib"] if bib_result else "Pending"

                leaderboard.append(
                    {
                        "id": tracker_id,
                        "bib": bib_number,
                        "time_ms": history_data["finish_time_ms"],
                    }
                )

        # 4. Sort the leaderboard by finish time
        leaderboard.sort(key=lambda x: x["time_ms"])

        # 5. Print the formatted leaderboard
        print(
            f"--- 🏁 Live Race Leaderboard (Updated: {time.strftime('%I:%M:%S %p')}) 🏁 ---"
        )
        if leaderboard:
            for i, entry in enumerate(leaderboard):
                total_seconds = entry["time_ms"] / 1000
                minutes = int(total_seconds // 60)
                seconds = int(total_seconds % 60)
                milliseconds = int((total_seconds - int(total_seconds)) * 100)
                time_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:02d}"

                print(
                    f"  {i + 1}. Racer ID: {entry['id']:<4} | Bib: {entry['bib']:<8} | Time: {time_str}"
                )
        else:
            print("  Waiting for the first racer to finish...")

        print("----------------------------------------------------------")
        print("\n(Processing video... Press Ctrl+C to stop and show final results)")

    def process_video(self, output_path=None, display=True):
        # ... video writer setup ...
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
        start_time = time.time()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue

                orig_h, orig_w = frame.shape[:2]
                proc_w = orig_w
                scale = proc_w / orig_w
                proc_h = int(orig_h * scale)

                processing_frame = cv2.resize(frame, (proc_w, proc_h))
                if frame_count % self.inference_interval == 0:
                    # --- HYBRID MODEL CALLS USING TWO SEPARATE INSTANCES ---
                    # 1. Track ONLY persons using the dedicated tracker_model
                    person_results = self.tracker_model.track(
                        processing_frame,
                        persist=True,
                        tracker="config/custom_tracker.yaml",
                        classes=[0],
                        verbose=False,
                    )
                    tracked_persons = person_results[0] if person_results else None

                    self.check_finish_line_crossings(tracked_persons)

                    # 2. Predict ALL objects using the separate, stateless predictor_model
                    all_detections_results = self.predictor_model.predict(
                        processing_frame, conf=self.confidence_threshold, verbose=False
                    )
                    all_detections = (
                        all_detections_results[0] if all_detections_results else None
                    )

                    # 3. Update history using the new hybrid function
                    self.update_history_hybrid(tracked_persons, all_detections)

                    # 4. Draw predictions using the new hybrid function
                    annotated_frame = self.draw_hybrid_predictions(
                        frame, tracked_persons, all_detections, scale=scale
                    )

                    self.last_annotated_frame = annotated_frame
                    display_frame = annotated_frame
                else:
                    if self.last_annotated_frame is not None:
                        display_frame = self.last_annotated_frame

                if display:
                    cv2.imshow("Live Bib Tracking", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if out:
                    out.write(display_frame)
                frame_count += 1

        finally:
            # First, determine the final bib numbers for everyone
            final_bib_results = self.determine_final_bibs()

            # --- NEW: ASSEMBLE AND PRINT THE LEADERBOARD ---
            leaderboard = []
            for tracker_id, result_data in final_bib_results.items():
                history_data = self.track_history.get(tracker_id)
                if history_data and history_data.get("finish_time_ms") is not None:
                    leaderboard.append(
                        {
                            "id": tracker_id,
                            "bib": result_data["final_bib"],
                            "time_ms": history_data["finish_time_ms"],
                        }
                    )

            # Sort the leaderboard by finish time (fastest first)
            leaderboard.sort(key=lambda x: x["time_ms"])

            print("\n--- 🏁 Official Race Leaderboard 🏁 ---")
            if leaderboard:
                for i, entry in enumerate(leaderboard):
                    # Format time as MM:SS.ms
                    total_seconds = entry["time_ms"] / 1000
                    minutes = int(total_seconds // 60)
                    seconds = int(total_seconds % 60)
                    milliseconds = int((total_seconds - int(total_seconds)) * 100)
                    time_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:02d}"

                    print(
                        f"  {i + 1}. Racer ID: {entry['id']:<4} | Bib: {entry['bib']:<6} | Time: {time_str}"
                    )
            else:
                print("  No racers finished the race.")
            # print("----------------------------------------")
            # final_results = self.determine_final_bibs()
            # print("\n--- Final Bib Number Results ---")
            # if final_results:
            #     for tracker_id, data in final_results.items():
            #         print(f"  Racer ID {tracker_id}: Final Bib = {data['final_bib']} (Score: {data['score']:.2f})")
            # else:
            #     print("  No reliable bib numbers were finalized.")
            self.cap.release()
            if out:
                out.release()
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
        "--fps", type=int, default=10, help="Target processing frame rate"
    )
    parser.add_argument(
        "--conf", type=float, default=0.3, help="YOLO confidence threshold"
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
