import cv2
import numpy as np
import os
import shutil
from pathlib import Path


def transform_bib_colors(image_path: str, bbox: list):
    """
    Loads an image and, within a given bounding box, transforms the
    chartreuse/yellow-green background to white and reddish-orange text to black,
    using wider HSV ranges for better coverage.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image at {image_path}")
        return None

    x_center, y_center, width, height = bbox
    img_h, img_w, _ = image.shape

    # Convert YOLO format (center x, center y, width, height) to pixel coordinates
    w = int(width * img_w)
    h = int(height * img_h)
    x1 = int((x_center * img_w) - w / 2)
    y1 = int((y_center * img_h) - h / 2)
    x2 = x1 + w
    y2 = y1 + h

    # Ensure bounding box is within image bounds
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

    bib_roi = image[y1:y2, x1:x2]
    if bib_roi.size == 0:
        print(
            f"Warning: Bounding box is invalid or outside the image for {image_path}, skipping."
        )
        return image  # Return original image if ROI is invalid

    # Convert the ROI to HSV color space
    hsv_roi = cv2.cvtColor(bib_roi, cv2.COLOR_BGR2HSV)

    # --- WIDER TUNED COLOR RANGES ---

    # Yellow-green background (expanded Hue, lower Saturation and Value to catch muted areas)
    # Hue: broader range from yellowish to greenish
    # Saturation: lowered to catch less vibrant yellow-greens
    # Value: lowered to catch darker/shadowed yellow-greens
    lower_yellow_green = np.array(
        [25, 50, 50]
    )  # Starting more towards yellow, less saturated, darker
    upper_yellow_green = np.array(
        [85, 255, 255]
    )  # Extending to more green, full saturation, full brightness

    # Reddish-orange text (expanded Hue, lower Saturation and Value to catch muted text)
    # Red hue wraps around 0 and 180 in HSV, so sometimes two ranges are needed.
    # However, for the specific reddish-orange in the bib, one range around 0-15 should be enough
    # if we broaden the saturation and value.
    lower_red_orange = np.array(
        [0, 100, 100]
    )  # Starting at true red hue, less saturated, darker
    upper_red_orange = np.array(
        [25, 255, 255]
    )  # Extending slightly into orange, full saturation, full brightness

    # If the text is truly more red and could wrap around, uncomment these two blocks:
    # lower_red_orange_wrap1 = np.array([0, 100, 100])
    # upper_red_orange_wrap1 = np.array([15, 255, 255])
    # lower_red_orange_wrap2 = np.array([165, 100, 100]) # Example for if red crosses 180
    # upper_red_orange_wrap2 = np.array([180, 255, 255])
    # text_mask = cv2.bitwise_or(cv2.inRange(hsv_roi, lower_red_orange_wrap1, upper_red_orange_wrap1),
    #                            cv2.inRange(hsv_roi, lower_red_orange_wrap2, upper_red_orange_wrap2))

    # Create masks
    background_mask = cv2.inRange(hsv_roi, lower_yellow_green, upper_yellow_green)
    text_mask = cv2.inRange(hsv_roi, lower_red_orange, upper_red_orange)

    # Apply transformations
    bib_roi[background_mask > 0] = [255, 255, 255]  # White BGR
    bib_roi[text_mask > 0] = [0, 0, 0]  # Black BGR

    image[y1:y2, x1:x2] = bib_roi

    return image


# The process_dataset function remains the same.
# Ensure you use the updated transform_bib_colors within it.


def process_dataset(image_dir, label_dir, output_dir):
    """
    Processes all images in a directory, applies color transformation to the bib,
    saves the new image, and copies the corresponding label file.
    """
    # Create output directories if they don't exist
    output_image_dir = Path(output_dir) / "images"
    output_label_dir = Path(output_dir) / "labels"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"Found {len(image_files)} images to process...")

    for image_filename in image_files:
        image_path = os.path.join(image_dir, image_filename)
        label_filename = Path(image_filename).stem + ".txt"
        label_path = os.path.join(label_dir, label_filename)

        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for {image_filename}, skipping.")
            continue

        # Read the label file to find the bib's bounding box (assuming class '1' is the bib)
        with open(label_path, "r") as f:
            lines = f.readlines()

        bib_bbox = None
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            if class_id == 1:  # Assuming '1' is your bib class ID
                bib_bbox = [float(p) for p in parts[1:]]
                break

        if bib_bbox is None:
            print(
                f"Warning: No bib found in label for {image_filename}, skipping transformation."
            )
            continue

        # Apply the color transformation
        transformed_image = transform_bib_colors(image_path, bib_bbox)

        if transformed_image is not None:
            # Save the new augmented image
            new_image_filename = f"aug_{image_filename}"
            new_image_path = os.path.join(output_image_dir, new_image_filename)
            cv2.imwrite(new_image_path, transformed_image)

            # Copy the original label file with a matching new name
            new_label_filename = f"aug_{label_filename}"
            new_label_path = os.path.join(output_label_dir, new_label_filename)
            shutil.copyfile(label_path, new_label_path)

            print(f"Processed and saved: {new_image_filename}")

    print("\nAugmentation complete!")
    print(f"New images saved to: {output_image_dir}")
    print(f"New labels saved to: {output_label_dir}")


# --- HOW TO USE ---
if __name__ == "__main__":
    # 1. Define your dataset paths
    source_image_dir = "/Users/dantrainer/projects/live-bib-tracking/data/test/images"
    # This is the directory of your original training labels
    source_label_dir = "/Users/dantrainer/projects/live-bib-tracking/data/test/labels"
    # This is where the new augmented images and labels will be saved
    augmented_output_dir = (
        "/Users/dantrainer/projects/live-bib-tracking/data/augmented_train"
    )

    # 2. Run the processing function
    process_dataset(source_image_dir, source_label_dir, augmented_output_dir)
