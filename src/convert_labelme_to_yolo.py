import json
import os
from pathlib import Path
import argparse
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_labelme_to_yolo(json_dir: str, output_dir: str):
    """
    Converts LabelMe JSON annotations to YOLOv5 format text files.

    Args:
        json_dir (str): Path to the directory containing LabelMe JSON files.
        output_dir (str): Path to the directory where YOLO .txt files will be saved.
    """
    # Ensure the output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = list(Path(json_dir).glob("*.json"))
    if not json_files:
        logging.warning(f"No JSON files found in {json_dir}")
        return

    logging.info(f"Found {len(json_files)} JSON files. Starting conversion...")

    converted_count = 0
    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Get image dimensions
            image_width = float(data['imageWidth'])
            image_height = float(data['imageHeight'])
            
            if image_width <= 0 or image_height <= 0:
                logging.warning(f"Skipping {json_file.name} due to invalid image dimensions.")
                continue

            yolo_labels = []
            for shape in data['shapes']:
                if shape['shape_type'] != 'rectangle':
                    continue

                class_id = int(shape['label'])
                
                # Extract points and ensure correct order (top-left, bottom-right)
                points = shape['points']
                x1 = min(points[0][0], points[1][0])
                y1 = min(points[0][1], points[1][1])
                x2 = max(points[0][0], points[1][0])
                y2 = max(points[0][1], points[1][1])

                # Calculate bounding box center, width, and height in pixels
                box_width = x2 - x1
                box_height = y2 - y1
                center_x = x1 + box_width / 2
                center_y = y1 + box_height / 2

                # Normalize coordinates for YOLO
                norm_center_x = center_x / image_width
                norm_center_y = center_y / image_height
                norm_width = box_width / image_width
                norm_height = box_height / image_height
                
                # Format for YOLO .txt file
                yolo_labels.append(
                    f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} "
                    f"{norm_width:.6f} {norm_height:.6f}"
                )

            # Write the YOLO labels to a .txt file
            output_txt_file = output_path / f"{json_file.stem}.txt"
            with open(output_txt_file, "w") as f:
                f.write("\n".join(yolo_labels))
            
            converted_count += 1

        except Exception as e:
            logging.error(f"Failed to process {json_file.name}: {e}")

    logging.info(f"Conversion complete. Successfully converted {converted_count} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LabelMe JSON annotations to YOLO .txt format."
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        default="data/processed/labels_json/train",
        required=False,
        help="Path to the directory containing LabelMe JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/labels/train",
        required=False,
        help="Path to the directory to save the output YOLO .txt files.",
    )
    args = parser.parse_args()

    convert_labelme_to_yolo(args.json_dir, args.output_dir)