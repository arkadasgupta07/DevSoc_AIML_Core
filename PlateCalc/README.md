<img width="512" height="512" alt="devsoc_logo_clear" src="https://github.com/user-attachments/assets/7cc58b7b-49b0-4e14-95ed-111ccc08418c" />

# 🥗 AI-Based Food Segmentation & Calorie Estimation System

An end-to-end Computer Vision pipeline built with **YOLOv8 Instance Segmentation** and **OpenCV** to automatically detect food items from images, estimate their physical weight via 2D spatial area analysis, and calculate total meal caloric intake.

---

## 📌 Features

* **Instance Segmentation:** Uses `YOLOv8n-seg` trained on the FoodSeg103 dataset to generate pixel-precise masks for multi-class food items.
* **Mass & Calorie Math Engine:** Integrates volume density constants and spatial calibration factors to estimate item mass in grams and energy in kilocalories ($\text{kcal}$).
* **Non-Overlapping Centroid Labeling:** Uses Image Moments ($M_{00}, M_{10}, M_{01}$) to compute mask centers of mass and draw crisp badge overlays without label clutter.
* **Fallback Safety System:** Gracefully handles unknown food classes using default fallback parameters to avoid runtime crashes.

---

## 🏗️ System Architecture & Workflow

1. **Input Image:** Loaded via OpenCV matrix (`BGR` format).
2. **YOLO Forward Pass:** Produces binary mask tensors $(H \times W)$ and confidence scores.
3. **Area Extraction:** Sums active foreground mask pixels:
   $$\text{Pixel Area} = \sum (\text{Binary Mask} == 1)$$
4. **Weight & Calorie Estimation:**
   $$\text{Weight (g)} = \text{Pixel Area} \times \text{Scale Factor} \times \text{Density}$$
   $$\text{Calories (kcal)} = \left(\frac{\text{Weight (g)}}{100}\right) \times \text{Cal/100g}$$
5. **Multi-Pass Visualization:** Renders $35\%$ translucent colored masks, centroid badge rectangles, and pure white text labels onto `test_prediction.jpg`.

---

## 📊 Model Training & Performance Metrics

The model was trained on the **FoodSeg103** dataset using `YOLOv8n-seg`.

### Key Assets (Located in `runs/segment/runs/segment/foodseg103_yolov8n/`):
* **`results.png`**: Displays loss reduction curves (`box_loss`, `seg_loss`, `cls_loss`) and precision-recall gains over training epochs.
--> Segmentation Loss (seg_loss): Accuracy of the polygon outline around the food.
--> Class Loss (cls_loss): Accuracy of identifying the food name.
--> mAP_{50}: Overall precision-recall score at a 50% overlap threshold.
* **`confusion_matrix.png`**: Per-class error breakdown of correct vs misclassified labels across visually similar food items, highlighting feature overlaps.
* **`val_batch0_pred.jpg` vs `val_batch0_labels.jpg`**: Visual comparison between ground-truth annotations and model inference outputs.
* **`results.csv`** logs epoch-by-epoch training and validation performance, tracking loss reduction alongside key accuracy metrics like Precision, Recall and mean Average Precision (mAP).

---

## 🛠️ Project File Structure

```text
AIML CORE ASSIGNMENT/
├── data/
│   ├── raw/
│   └── yolo_dataset/          # Processed YOLO format images & labels
├── output_predictions/        # Batch testing output results
├── runs/                      # YOLO model weights & training metric graphs
│   └── segment/.../weights/
│       ├── best.pt            # Highest accuracy PyTorch checkpoint
│       └── last.pt            # Final epoch checkpoint
├── food_config.yaml           # Dataset configuration file
├── predict_calories.py        # Core inference & calorie estimation script
├── train.py                   # Model training script
└── README.md                  # Project documentation

```

---

## 💻 How to Run Inference
1. Requirements
Ensure required Python packages are installed:
```text
pip install ultralytics opencv-python numpy
```

2. Run Batch Predictions
Execute the main script to process validation images and generate annotated outputs in output_predictions/:
```text
python predict_calories.py
```

---

## ⚠️ Limitations & Error Analysis

--> 2D Surface Approximation:
    Estimating 3D volume from 2D pixel area assumes standard thickness; highly volumetric or stacked foods can be underestimated.

--> Fine-Grained Texture Confusion:
    Visually similar items (e.g., cooked shrimp vs. chicken/duck) can occasionally yield class confusion on lightweight Nano architectures.

--> Lighting Sensitivity:
    Low-contrast glazes or sauces can blend overlapping food boundaries into unified segmentation shapes.
