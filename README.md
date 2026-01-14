
# Smart Eye Project
A multi-model computer vision pipeline for detecting, classifying, and measuring the depth of dents in images using YOLOv8, and custom ResNet-based models.

## Overview
-   **Detection:** Locates potential dent regions in the input image using an object detection model.
    
-   **Classification:** Validates true dents versus false positives using a ResNet-based classifier.
    
-   **Depth Estimation:** Predicts dent depth in millimeters using a ResNet-based regression model.


| **Stage** | **Model Used** | **Purpose** | **Input** | **Output** |
|---------|---------------|------------|----------|-----------|
| Dent Detection | YOLOv8 (Ultralytics) | Detects and localizes potential dent regions | Full car image (RGB) | Bounding boxes (ROIs) of candidate dents |
| Auxiliary Feature Extraction | Classical CV (OpenCV) | Extracts physics-guided surface cues | Cropped ROI image | Brightness, gradient, specular, shading features |
| Dent Validation | ResNet-18 + Auxiliary Feature Fusion | Classifies real dents vs reflections | Cropped ROI image + auxiliary features | Dent probability / Binary decision (Dent or Not Dent) |
| Dent Depth Estimation | ResNet-18 Regression Network | Predicts dent depth | Validated dent ROI image | Dent depth (mm) ||

## Datasets

This project uses **three separate datasets**, each dedicated to a specific stage of the Smart Eye pipeline.  
All datasets are curated to match the task requirements of detection, classification, and depth estimation.

---

### 1. Dent Detection Dataset (YOLOv8)

- **Purpose:** Train the YOLOv8 model to detect and localize dent regions in car images.
- **Annotations:** Bounding boxes around visible dents.
- **Platform:** Roboflow
- **Link:**  
  https://app.roboflow.com/nishant-gosavi-vnvdz/my-first-project-en9if/11

**Usage:**  
Used exclusively by the **dent detection stage** to generate high-recall Regions of Interest (ROIs).

---

### 2. Dent Validation Dataset (0 / 1 Classification)

- **Purpose:** Train the ResNet-based classifier to distinguish **real dents (1)** from **false positives (0)** such as reflections and lighting artifacts.
- **Data Format:** Cropped dent-like regions with binary labels.
- **Link:**  
  https://drive.google.com/drive/folders/1LsdV9Ma3HaZ_TtlLmad65DeWtKLofMyW?usp=sharing

**Usage:**  
Used by the **dent validation stage** to filter invalid detections before depth estimation.

---

### 3. Dent Depth Prediction Dataset

- **Purpose:** Train the regression model to predict **dent depth in millimeters**.
- **Data Format:** Cropped dent images with corresponding numerical depth values.
- **Link:**  
  https://drive.google.com/drive/folders/1470SzHxH2PjpUt1VoUHjDFwMWvePnGyF?usp=sharing

**Usage:**  
Used by the **depth estimation stage** to quantify dent severity.

---

### Dataset Design Insight
> *Each dataset is task-specific, enabling modular training and preventing error propagation across stages.*


## Setup Instructions

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```
### 2. Install Dependencies

```bash
# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Computer vision and ML libraries
pip install ultralytics opencv-python albumentations joblib scipy

# Visualization
pip install matplotlib
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### 3. Required Model Files

Place these pre-trained model files in your project root directory:
```
project/
├── (Model_1)Smart_Eye_dent_detection_weights.pt         # YOLOv8 dent detection weights
├── (Model_2)Smart_Eye_dent_0_1_weights.pth         # Dent classification model weights
├── (Model_3)Smart_Eye_dent_depth_weights.pth       # Depth estimation model weights
└── (Model_3)_scaling_weights.pkl       # Scaler for depth predictions
```

