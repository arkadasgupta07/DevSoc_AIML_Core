import os
import shutil
import cv2
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "FoodSeg103" / "Images"
YOLO_DIR = BASE_DIR / "data" / "yolo_dataset"

SPLITS = {
    "train": ("img_dir/train", "ann_dir/train"),
    "val": ("img_dir/test", "ann_dir/test")
}

def convert_mask_to_yolo_seg(mask_path, label_out_path):
    """
    Reads an indexed PNG mask image from FoodSeg103 and converts 
    it into normalized YOLO segmentation polygons.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False

    img_h, img_w = mask.shape
    unique_classes = np.unique(mask)
    lines = []

    for class_id in unique_classes:
        if class_id == 0:  # 0 is background
            continue

        # Create binary mask for specific class ID
        binary_mask = np.uint8(mask == class_id)

        # Extract contours using OpenCV
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # Ignore noise (tiny regions under 10 pixels area)
            if cv2.contourArea(contour) < 10:
                continue

            polygon = []
            for point in contour:
                x, y = point[0]
                # Normalize coordinates to [0.0, 1.0]
                polygon.extend([f"{x / img_w:.6f}", f"{y / img_h:.6f}"])

            if len(polygon) >= 6:  # At least 3 points required
                # Map to 0-indexed YOLO class
                yolo_class = class_id - 1
                lines.append(f"{yolo_class} " + " ".join(polygon))

    if lines:
        with open(label_out_path, "w") as f:
            f.write("\n".join(lines))
        return True
    return False


def process_dataset():
    print("Starting FoodSeg103 to YOLOv8-Seg conversion...")

    for split_name, (img_sub, ann_sub) in SPLITS.items():
        img_src_folder = RAW_DIR / img_sub
        ann_src_folder = RAW_DIR / ann_sub

        img_dest_folder = YOLO_DIR / "images" / split_name
        lbl_dest_folder = YOLO_DIR / "labels" / split_name

        img_files = list(img_src_folder.glob("*.jpg")) + list(img_src_folder.glob("*.png"))
        print(f"Processing '{split_name}' split ({len(img_files)} images)...")

        converted_count = 0
        for img_path in img_files:
            # Corresponding mask file name (same basename, .png extension)
            mask_path = ann_src_folder / f"{img_path.stem}.png"
            lbl_out_path = lbl_dest_folder / f"{img_path.stem}.txt"

            if mask_path.exists():
                # Convert mask to polygon text label
                has_polygons = convert_mask_to_yolo_seg(mask_path, lbl_out_path)
                
                # Copy image to YOLO directory
                shutil.copy(img_path, img_dest_folder / img_path.name)
                converted_count += 1

        print(f"Finished '{split_name}': Copied {converted_count} images & generated labels.")

    print("\nDataset Conversion Complete!")

if __name__ == "__main__":
    process_dataset()