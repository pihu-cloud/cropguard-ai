# CropGuard AI — Setup Guide

## Model Info

Your `resnet.h5` is a **ResNet50V2** trained on a **10-class PlantVillage subset**:

| # | Class Label                  | Crop   | Disease              |
|---|------------------------------|--------|----------------------|
| 0 | Corn___Common_Rust           | Corn   | Common Rust          |
| 1 | Corn___Gray_Leaf_Spot        | Corn   | Gray Leaf Spot       |
| 2 | Corn___Healthy               | Corn   | Healthy              |
| 3 | Corn___Northern_Leaf_Blight  | Corn   | Northern Leaf Blight |
| 4 | Potato___Early_Blight        | Potato | Early Blight         |
| 5 | Potato___Healthy             | Potato | Healthy              |
| 6 | Potato___Late_Blight         | Potato | Late Blight          |
| 7 | Tomato___Bacterial_Spot      | Tomato | Bacterial Spot       |
| 8 | Tomato___Healthy             | Tomato | Healthy              |
| 9 | Tomato___Late_Blight         | Tomato | Late Blight          |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your model
Put `resnet.h5` in the **same folder as app.py** (or in a `models/` subfolder):
```
cropguard/
├── app.py
├── predict_pipeline.py
├── class_names.json        ← auto-generated, matches your 10-class model
├── disease_info.json       ← auto-generated, full metadata for all 10 classes
├── requirements.txt
├── resnet.h5               ← place your model here  ✅
└── static/
    └── uploads/
```

### 3. Run
```bash
python app.py
```
Visit http://localhost:5000

---

## What was fixed

| Bug | Fix |
|-----|-----|
| `class_names.json` missing | Generated for your exact 10 classes |
| `disease_info.json` missing | Generated with full disease metadata |
| Wrong model path — looked in `models/` only | Now checks root folder first, then `models/` |
| `import tensorflow` at top of app.py — crashed if TF not installed | Made lazy (only imported when loading model) |
| Both models referenced everywhere even after removing MobileNet | Cleaned to ResNet-only throughout |
| `load_models()` fallback used MobileNetV2 architecture for ResNet | Removed wrong fallback |
| Demo mode message still said mobilenet/resnet | Updated to ResNet-only message |

---

## Expanding to more classes later

If you retrain your model with more PlantVillage classes (e.g. all 38):
1. Update `class_names.json` — add new class label strings in the exact training order
2. Update `disease_info.json` — add metadata entries for the new classes
3. The rest of the code adjusts automatically via `NUM_CLASSES = len(CLASS_NAMES)`
