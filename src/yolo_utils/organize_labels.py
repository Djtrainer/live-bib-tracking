import os
import shutil
from pathlib import Path

def organize_dataset(source_dir, json_dest_dir, image_dest_dir):
    """
    Copies JSON and corresponding JPG files from a source directory
    to separate destination directories.

    Args:
        source_dir (str): The directory containing the mixed JPG and JSON files.
        json_dest_dir (str): The destination directory for JSON label files.
        image_dest_dir (str): The destination directory for JPG image files.
    """
    # Create destination directories if they don't exist
    os.makedirs(json_dest_dir, exist_ok=True)
    os.makedirs(image_dest_dir, exist_ok=True)
    
    print(f"Scanning source directory: {source_dir}")
    
    copied_count = 0
    
    # Get a list of all JSON files in the source directory
    json_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.json')]
    
    if not json_files:
        print("No JSON files found in the source directory.")
        return

    print(f"Found {len(json_files)} JSON files. Starting copy process...")

    # Iterate through each JSON file
    for json_filename in json_files:
        # Get the base name of the file without the extension
        base_name = Path(json_filename).stem
        
        # Define the expected corresponding JPG filename
        jpg_filename = f"{base_name}.jpg"
        
        # Define the full paths for the source files
        source_json_path = os.path.join(source_dir, json_filename)
        source_jpg_path = os.path.join(source_dir, jpg_filename)
        
        # Check if the corresponding JPG file actually exists
        if os.path.exists(source_jpg_path):
            # Define the full paths for the destination files
            dest_json_path = os.path.join(json_dest_dir, json_filename)
            dest_jpg_path = os.path.join(image_dest_dir, jpg_filename)
            
            # 1. Copy the JSON file
            shutil.copyfile(source_json_path, dest_json_path)
            
            # 2. Copy the corresponding JPG file
            shutil.copyfile(source_jpg_path, dest_jpg_path)
            
            print(f"  - Copied {json_filename} and {jpg_filename}")
            copied_count += 1
        else:
            print(f"  - Skipping {json_filename} (corresponding JPG not found).")

    print(f"\nProcess complete. Copied {copied_count} image/label pairs.")

# --- HOW TO USE ---
if __name__ == "__main__":
    # 1. Define your directory paths
    source_directory = "data/processed/race_2023/"
    json_destination = "data/processed/labels_json/train"
    image_destination = "data/processed/images/train"
    
    # 2. Run the function
    organize_dataset(source_directory, json_destination, image_destination)