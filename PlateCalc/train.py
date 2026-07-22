from ultralytics import YOLO

def main():
    # Load pretrained YOLOv8 Nano Segmentation model
    model = YOLO("yolov8n-seg.pt")

    # Train on your converted dataset
    model.train(
        data="food_config.yaml",
        epochs=15,          # 15 epochs for fast completion
        imgsz=640,          # Standard resolution
        batch=16,           # Reduce to 8 if VRAM/RAM is tight
        device=0,           # Set to 'cpu' if no CUDA GPU is available
        project="runs/segment",
        name="foodseg103_yolov8n",
        exist_ok=True
    )

if __name__ == "__main__":
    main()