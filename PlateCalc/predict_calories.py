import os
import cv2
import numpy as np
import glob
from ultralytics import YOLO

# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================
MODEL_PATH = os.path.join("runs", "segment", "runs", "segment", "foodseg103_yolov8n", "weights", "best.pt")

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()

print(f"Loading trained model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"Model Class Names: {model.names}")

# ==========================================
# 2. NUTRITION DATABASE & CONSTANTS
# ==========================================
NUTRITION_DB = {
    # Existing Items
    "corn": {"cal_100g": 86, "density": 0.72},
    "rice": {"cal_100g": 130, "density": 0.85},
    "pizza": {"cal_100g": 266, "density": 0.60},
    "potato": {"cal_100g": 77, "density": 0.65},
    "broccoli": {"cal_100g": 34, "density": 0.35},
    "carrot": {"cal_100g": 41, "density": 0.60},
    "tomato": {"cal_100g": 18, "density": 0.50},
    "cabbage": {"cal_100g": 25, "density": 0.35},
    "cucumber": {"cal_100g": 15, "density": 0.45},
    "french beans": {"cal_100g": 31, "density": 0.50},

    # --- Add New/Common Dataset Items Here ---
    "egg": {"cal_100g": 155, "density": 0.60},
    "chicken": {"cal_100g": 165, "density": 0.70},
    "shrimp": {"cal_100g": 99, "density": 0.65},
    "pork": {"cal_100g": 242, "density": 0.75},
    "beef": {"cal_100g": 250, "density": 0.75},
    "pasta": {"cal_100g": 131, "density": 0.80},
    "bread": {"cal_100g": 265, "density": 0.30},
    # --- Added from terminal warnings ---
    "ice cream": {"cal_100g": 207, "density": 0.55},
    "chicken duck": {"cal_100g": 165, "density": 0.70},
    "cilantro mint": {"cal_100g": 23, "density": 0.20},
    "sauce": {"cal_100g": 100, "density": 0.90},
    "cake": {"cal_100g": 371, "density": 0.45},
    "biscuit": {"cal_100g": 353, "density": 0.40},
    "blueberry": {"cal_100g": 57, "density": 0.60},
    "pie": {"cal_100g": 265, "density": 0.55},
    "green beans": {"cal_100g": 31, "density": 0.45},
}

DEFAULT_NUTRITION = {"cal_100g": 120, "density": 0.50}
SCALE_FACTOR = 0.012 

# ==========================================
# 3. CALORIE ESTIMATION & VISUALIZATION
# ==========================================
def process_and_estimate(image_path, output_save_path="test_prediction.jpg"):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    results = model(image_path)[0]
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]
    
    mask_overlay = img.copy()
    total_meal_calories = 0.0
    detected_items = 0

    print("\n" + "="*60)
    print(f" PROCESSING IMAGE: {image_path}")
    print("="*60)

    if results.masks is not None and len(results.masks) > 0:
        colors = [
            (50, 205, 50),   # Lime Green
            (0, 165, 255),   # Orange
            (255, 191, 0),   # Deep Sky Blue
            (147, 112, 219), # Medium Purple
            (0, 215, 255)    # Gold
        ]

        label_data = []

        for i, mask in enumerate(results.masks.data):
            class_id = int(results.boxes.cls[i].item())
            class_name = model.names[class_id]
            confidence = float(results.boxes.conf[i].item())

            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (img_w, img_h))
            binary_mask = (mask_resized > 0.5).astype(np.uint8)

            pixel_area = int(np.sum(binary_mask))
            info = NUTRITION_DB.get(class_name, DEFAULT_NUTRITION)

            # small warning check so you always know when a default fallback was used
            if class_name not in NUTRITION_DB:
                print(f"  ⚠️ Warning: '{class_name}' missing from NUTRITION_DB! Using default fallbacks.")

            weight_grams = pixel_area * SCALE_FACTOR * info["density"]
            calories = (weight_grams / 100.0) * info["cal_100g"]
            
            total_meal_calories += calories
            detected_items += 1

            color = colors[i % len(colors)]

            # A. Draw colored translucent segmentation mask on overlay
            mask_overlay[binary_mask == 1] = color

            # B. Draw thin bounding box contour
            box = results.boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # C. Calculate mask centroid (Center of Mass) for text placement
            M = cv2.moments(binary_mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            label_text = f"{class_name}: ~{int(calories)} kcal ({int(weight_grams)}g)"
            label_data.append({
                "cx": cx, "cy": cy,
                "text": label_text,
                "color": color
            })

            print(f"[{detected_items}] {class_name:<12} | Conf: {confidence:.2f} | Area: {pixel_area:>6} px | Weight: {weight_grams:>6.1f}g | Cal: {calories:>6.1f} kcal")

        # Blend semi-transparent segmentation masks
        cv2.addWeighted(mask_overlay, 0.35, img, 0.65, 0, img)

        # D. Draw Label Badges at Mask Centroids
        badge_overlay = img.copy()

        for item in label_data:
            cx, cy = item["cx"], item["cy"]
            text = item["text"]
            color = item["color"]

            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

            bx1 = max(5, cx - tw // 2 - 8)
            by1 = max(50, cy - th // 2 - 8)
            bx2 = min(img_w - 5, bx1 + tw + 16)
            by2 = min(img_h - 5, by1 + th + 16)

            cv2.rectangle(badge_overlay, (bx1, by1), (bx2, by2), (15, 15, 15), -1)
            cv2.rectangle(badge_overlay, (bx1, by1), (bx2, by2), color, 1)

            tx = bx1 + 8
            ty = by1 + th + 4

        # 1. Blend the dark badge backgrounds onto the image first
        cv2.addWeighted(badge_overlay, 0.70, img, 0.30, 0, img)

        # 2. Draw crisp pure white text directly on top so it doesn't get darkened
        for item in label_data:
            cx, cy, text = item["cx"], item["cy"], item["text"]
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            bx1 = max(5, cx - tw // 2 - 8)
            by1 = max(50, cy - th // 2 - 8)
            cv2.putText(img, text, (bx1 + 8, by1 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # E. Top-Left Header: TOTAL MEAL ENERGY
        header_text = f"TOTAL MEAL ENERGY: {int(total_meal_calories)} kcal"
        (hw, hh), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        
        cv2.rectangle(img, (10, 10), (20 + hw, 20 + hh + 10), (0, 0, 0), -1)
        cv2.rectangle(img, (10, 10), (20 + hw, 20 + hh + 10), (0, 255, 255), 2)
        cv2.putText(img, header_text, (18, 15 + hh + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        print("="*60)
        print(f" TOTAL ESTIMATED MEAL CALORIES: {total_meal_calories:.1f} kcal")
        print("="*60 + "\n")

    # Save finalized image
    cv2.imwrite(output_save_path, img)
    print(f" Clean, non-overlapping result saved to: {output_save_path}")

'''
# ==========================================
# 4. RUN INFERENCE
# ==========================================
if __name__ == "__main__":
    val_folder = os.path.join("data", "yolo_dataset", "images", "val")
    image_files = (
        glob.glob(os.path.join(val_folder, "*.jpg")) +
        glob.glob(os.path.join(val_folder, "*.jpeg")) +
        glob.glob(os.path.join(val_folder, "*.png"))
    )

    if image_files:
        test_image_path = image_files[0]
        process_and_estimate(test_image_path, output_save_path="test_prediction.jpg")
    else:
        print(f"Error: No image files found in '{val_folder}'.")
'''

# ==========================================
# 4. RUN INFERENCE (BATCH TESTING)
# ==========================================
if __name__ == "__main__":
    val_folder = os.path.join("data", "yolo_dataset", "images", "val")
    output_dir = "output_predictions"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Gather all images in validation folder
    image_files = (
        glob.glob(os.path.join(val_folder, "*.jpg")) +
        glob.glob(os.path.join(val_folder, "*.jpeg")) +
        glob.glob(os.path.join(val_folder, "*.png"))
    )

    if image_files:
        # Limit to first 10 images for testing
        test_batch = image_files[:10]
        print(f"Found {len(image_files)} total images. Processing batch of {len(test_batch)}...")

        for idx, img_path in enumerate(test_batch, start=1):
            file_name = os.path.basename(img_path)
            output_path = os.path.join(output_dir, f"prediction_{idx}_{file_name}")
            process_and_estimate(img_path, output_save_path=output_path)

        print("\n" + "="*60)
        print(f" SUCCESS: All {len(test_batch)} test predictions saved to '{output_dir}/'")
        print("="*60)
    else:
        print(f"Error: No image files found in '{val_folder}'.")