
# Smart Eye Project

A multi-stage computer vision pipeline for detecting surface damage (dents and scratches), estimating dent depth, extracting geometric features, and generating structured repair recommendations using deep learning and LLM-based reasoning.

----------

## 🚀 Overview

Smart Eye is designed as an intelligent surface inspection assistant that goes beyond simple detection. The system performs:

-   **Damage Detection** – Detects dents and scratches using YOLO.
    
-   **Geometric Feature Extraction** – Computes measurable characteristics from bounding boxes.
    
-   **Dent Depth Estimation** – Predicts dent depth in millimeters using a ResNet-based regression model.
    
-   **Multi-Damage Aggregation** – Structures all detected damage into a unified geometry report.
    
-   **LLM-Based Repair Recommendation (Optional)** – Generates structured, conservative repair suggestions based strictly on geometric data.


## 🏗 Pipeline Stages

| Stage | Model / Method Used | Purpose | Input | Output |
|--------|---------------------|----------|--------|--------|
| **Damage Detection** | YOLO (Ultralytics) | Detects and localizes dents and scratches | Full RGB image | Bounding boxes, class labels (dent/scratch), confidence scores |
| **Geometric Feature Extraction** | Bounding box computation | Computes measurable geometric features | Detected bounding boxes | Width, height, area (px), aspect ratio |
| **Dent Depth Estimation** | ResNet-based Regression Model (`predict_depth`) | Estimates dent depth in millimeters | Cropped dent ROI image | Estimated dent depth (mm) |
| **Damage Aggregation** | Structured data assembly (Python dict) | Organizes all damage information | Per-damage geometry + depth | Structured `damage_list` |
| **Repair Recommendation** | LLM with structured output schema | Generates geometry-based conservative repair strategy | Structured damage report | Detailed repair recommendations + prioritization |

---

## 📂 Datasets

This project uses separate datasets dedicated to specific stages of the Smart Eye pipeline.  
Each dataset is curated to match the requirements of detection and depth estimation tasks.

---

### 1️⃣ Dent & Scratch Detection Dataset (YOLO)

**Purpose:**  
Train the YOLO model to detect and localize dent and scratch regions in images.

**Annotations:**  
Bounding boxes around visible dents and scratches

**Platform:**  
Roboflow

**Link:**  
https://universe.roboflow.com/ali-tl4zm/dent-and-scratch/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true

**Usage:**  
Used exclusively by the damage detection stage to generate high-recall Regions of Interest (ROIs).

---

### 2️⃣ Dent Depth Prediction Dataset

**Purpose:**  
Train the regression model to predict dent depth in millimeters.

**Data Format:**  
Cropped dent images paired with corresponding numerical depth values.

**Link:**  
https://drive.google.com/drive/folders/1470SzHxH2PjpUt1VoUHjDFwMWvePnGyF?usp=sharing

**Usage:**  
Used by the depth estimation stage to quantify dent severity.



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
├── (Model_1)_dent_&_scratch_detection_yoloV11.pt         # YOLOv8 dent detection weights
├── (Model_2)Smart_Eye_dent_depth_weights.pth       # Depth estimation model weights
└── (Model_2)_scaling_weights.pkl       # Scaler for depth predictions
```

