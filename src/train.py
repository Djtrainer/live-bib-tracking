from ultralytics import YOLO

def train_model():
    """
    Trains a small YOLOv8 model on the custom bib dataset.
    """
    # 1. Load a pre-trained 'small' model
    model = YOLO('yolo11n.pt') # Or 'yolov8n.pt' for the fastest version

    # 2. Train the model on your dataset
    # This will automatically download the dataset if it's not present
    # and start the training process.
    print("Starting model training...")
    model.train(
        data='config/yolo_config.yaml',  # Path to your dataset yaml file
        epochs=200,                      # Number of training epochs (e.g., 50-100)
        # imgsz=640,                      # Image size for training
        name='yolo11_white_bibs'     # Name for the output folder
    )
    print("Training complete!")
    print("Model and results saved in the 'runs/detect/' directory.")

if __name__ == '__main__':
    train_model()