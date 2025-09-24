from ultralytics import YOLO
import os
import shutil

def train_yolov8_model(epochs=10, batch_size=16, img_size=640):
    """
    Fresh training of YOLOv8 model on the prepared face dataset.
    Old training checkpoints will be removed before starting.
    """
    train_dir = "runs/train/face_detection_model"

    if os.path.exists(train_dir):
        print(f"🗑️ Removing old training folder: {train_dir}")
        shutil.rmtree(train_dir)

    print("✨ Starting fresh training with yolov8n.pt")
    model = YOLO("yolov8n.pt")  

    # Train (CPU only here)
    results = model.train(
        data="faces_data.yaml",
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        name="face_detection_model",
        project="runs/train",
        workers=4,
        device="cpu",
        resume=False
    )

    print("🎉 Training complete! Check results in 'runs/train/face_detection_model'")
    return results


if __name__ == "__main__":
    train_yolov8_model(epochs=10, batch_size=32, img_size=640)
