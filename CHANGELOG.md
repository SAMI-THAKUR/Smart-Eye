# Smart Eye - Change Log

## 📋 Recent Updates & New Features

### Version 2.0 - Video Processing & LLM Integration

#### 🎯 Major Changes

##### 1. **Video Processing Pipeline** (`SmartEye_video.py`)
- **Frame-by-Frame Analysis**: Automated video file processing with intelligent frame extraction
- **Real-time Detection**: Process video files and extract damage detections from each frame
- **Batch Frame Analysis**: Efficient handling of multiple frames with GPU acceleration

##### 2. **Dual YOLO Model Architecture**
- **Separate Specialized Models**:
  - `best_recall.pt` - High-recall dent detection model
  - `(Model_1_1)_Scratch_Detection_Model.pt` - Specialized scratch detection model
- **Improved Accuracy**: Each model optimized for its specific damage type
- **Independent Detection Streams**: Parallel detection for dents and scratches

##### 3. **Enhanced Depth Estimation**
- **Improved Scaler Integration**: Using `depth_scaler.pkl` for consistent depth predictions
- **ResNet18-based Regressor**: Fine-tuned backbone with frozen lower layers
- **Two-stage Regression**: Feature extraction + regression head for better accuracy
- **Calibrated Predictions**: Depth values in millimeters with proper scaling

##### 4. **LLM-Based Repair Recommendations**
- **Groq API Integration**: Fast, efficient LLM reasoning for repair strategies
- **Structured Output Schema**: Pydantic models for consistent JSON responses
- **Conservative Recommendations**: Evidence-based repair suggestions using:
  - Geometric measurements (width, height, aspect ratio)
  - Estimated depth information
  - Severity classification (Low/Medium/High)
  - Downtime estimates (6-12 hours typical)

##### 5. **Comprehensive Damage Reports**
- **Structured JSON Output**: Machine-readable damage analysis
- **Per-Damage Details**:
  - Damage ID, type, and description
  - Pixel dimensions and aspect ratio
  - Estimated depth in millimeters
  - Depth classification (surface / bare metal / through-thickness)
  - Severity levels and confidence scores
  - Repair downtime estimates
  - LLM-generated repair recommendations

##### 6. **Bird Strike Detection** (Optional)
- **Framework for Future Enhancement**: Placeholder for bird strike analysis
- **Extensible Architecture**: Easy to add additional damage types

---

## 📂 New & Updated Files

### Core Pipeline Files
| File | Purpose | Status |
|------|---------|--------|
| `SmartEye_video.py` | **NEW** - Video-to-report end-to-end pipeline | ✅ Production Ready |
| `new_pipeline.py` | **NEW** - Refactored modular pipeline | ✅ Production Ready |
| `pipeline1.py` | Original pipeline implementation | 📦 Legacy |
| `Smart_Eye_End_to_End_PipeLine.ipynb` | Jupyter notebook version | 📚 Reference |

### Model Files
| File | Model Type | Purpose |
|------|-----------|---------|
| `best_recall.pt` | YOLO (Dent Detection) | High-recall dent detection |
| `(Model_1_1)_Scratch_Detection_Model.pt` | YOLO (Scratch Detection) | Specialized scratch detection |
| `best_model.pth` | ResNet18 Regressor | Depth estimation |
| `depth_scaler.pkl` | StandardScaler | Depth value normalization |

### Output Files
| File | Format | Contains |
|------|--------|----------|
| `report.json` | JSON | Structured damage analysis & repair recommendations |

---

## 🚀 Usage Guide

### Running Video Analysis

```bash
# Analyze a single video file
python SmartEye_video.py --video path/to/video.mp4

# Example with provided video
python SmartEye_video.py --video video.47.14.mp4
```

### Output
- **JSON Report**: `report_<session_id>_<timestamp>.json`
- **Contains**:
  - Session metadata (ID, date, aircraft info)
  - Per-frame damage detections
  - Geometric measurements
  - Depth estimates
  - LLM-based repair recommendations
  - Downtime estimates

### Example Output Structure
```json
{
  "session_id": "SE-2026-0420-FCD5",
  "report_date": "2026-04-20T09:06:13.620385+00:00",
  "aircraft": {
    "registration": "UNKNOWN",
    "type": "UNKNOWN"
  },
  "damages": [
    {
      "damage_id": "D0",
      "damage_type": "scratch",
      "description": "Scratch, width 171 px, height 114 px, area 19,494 px²...",
      "dimensions": {
        "width": "171",
        "height": "114",
        "depth": "0",
        "unit": "px/mm"
      },
      "depth_estimate": {
        "level": "surface",
        "detail": "Surface-level deformation only"
      },
      "downtime_estimate": {
        "min_hours": 6,
        "max_hours": 12
      },
      "severity": "high",
      "repair_recommendations": [...]
    }
  ]
}
```

---

## 🔧 Technical Improvements

### Detection Pipeline
```
Video Input
    ↓
Frame Extraction
    ↓
┌───────────────────────┐
│   YOLO Dent Model     │  best_recall.pt
│   YOLO Scratch Model  │  Scratch Detection
└───────────────────────┘
    ↓
ROI Extraction (Cropped damage regions)
    ↓
┌───────────────────────┐
│ ResNet18 Regressor    │  + Scaler
│ (Depth Estimation)    │
└───────────────────────┘
    ↓
Geometric Features Computation
(width, height, area, aspect ratio)
    ↓
┌───────────────────────┐
│ LLM Reasoning Engine  │  Groq API
│ (Repair Strategy)     │
└───────────────────────┘
    ↓
Structured JSON Report
```

### Model Improvements
- **Dual Detection Architecture**: Specialized models for different damage types
- **Frozen Backbone**: ResNet18 frozen except layer4 for efficient fine-tuning
- **Two-Stage Regression**: Feature extraction + dense layers for better predictions
- **Proper Scaling**: JobLib scaler ensures consistent depth normalization

### Performance Features
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Efficient Frame Processing**: Skip redundant frames, batch processing where possible
- **Lazy Loading**: Models loaded only when needed
- **Memory Optimization**: Careful tensor management for long videos

---

## 📊 Report Fields Explained

### Severity Levels
| Level | Criteria | Action |
|-------|----------|--------|
| **Low** | Surface scratches, no depth | Monitor, defer repair |
| **Medium** | Shallow dents (< 2mm), minor surface damage | Schedule repair |
| **High** | Deeper dents (≥ 2mm), multiple damages | Urgent repair required |

### Depth Classification
- **Surface**: Paint/primer only, no substrate penetration
- **Bare Metal**: Paint/primer removed, substrate exposed
- **Through-Thickness**: Structural skin penetration (rare, high priority)

### Downtime Estimates
- **6-12 hours**: Typical repair window for most damage types
- Based on damage complexity, severity, and repair method
- Includes inspection, preparation, and finishing time

---

## 🔐 API Keys & Configuration

### Required Configuration (in `SmartEye_video.py`)
```python
GROQ_API_KEY = "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"
DENT_MODEL = "best_recall.pt"
SCRATCH_MODEL = "(Model_1_1)_Scratch_Detection_Model.pt"
DEPTH_MODEL = "best_model.pth"
DEPTH_SCALER = "depth_scaler.pkl"
```

### Aircraft Metadata (Customizable)
```python
DEFAULT_META = {
    "session_id": "Auto-generated",
    "report_date": "Auto-generated",
    "location": "UNKNOWN",  # Set as needed
    "aircraft": {
        "registration": "UNKNOWN",  # Set as needed
        "type": "UNKNOWN",           # Set as needed
        "airline": "UNKNOWN"         # Set as needed
    }
}
```

---

## ✅ Dependencies

### Core Libraries
```
torch torchvision          # Deep learning framework
ultralytics               # YOLO models
langchain-groq            # LLM integration
langchain-core            # LLM core utilities
albumentations            # Image augmentation
opencv-python (cv2)       # Video processing
numpy                     # Numerical computation
joblib                    # Model serialization
matplotlib                # Visualization
pydantic                  # Data validation
```

### Installation
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics langchain-groq langchain-core albumentations
pip install opencv-python matplotlib pydantic joblib numpy
```

---

## 🐛 Known Issues & Limitations

1. **Bird Strike Detection**: Currently disabled (placeholder in report)
2. **Aircraft Metadata**: Defaults to "UNKNOWN" - must be set manually or via CLI
3. **Single Video Processing**: Processes one video per execution (easily parallelizable)
4. **Frame Rate**: Extracts all frames (consider frame skipping for efficiency on long videos)
5. **GPU Memory**: Large videos may require frame batching on limited VRAM systems

---

## 🔮 Future Enhancements

- [ ] Batch video processing
- [ ] Frame skipping algorithm for efficiency
- [ ] Bird strike detection module
- [ ] Multi-aircraft damage aggregation
- [ ] Web dashboard for report visualization
- [ ] Real-time stream processing
- [ ] Repair cost estimation
- [ ] Comparison with historical damage databases

---

## 📞 Support

For issues or questions:
1. Check the `report.json` output for detailed error information
2. Verify all model files are present and accessible
3. Ensure GPU drivers are up to date (for CUDA acceleration)
4. Validate API keys are correctly configured

---

## 📝 Version History

| Version | Date | Key Changes |
|---------|------|------------|
| 2.0 | 2026-04-20 | Video processing, dual YOLO models, LLM integration, structured reports |
| 1.0 | 2026-03-01 | Initial pipeline with YOLO detection and depth estimation |

