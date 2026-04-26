# 🌿 CropGuard AI — Crop Disease Detection System

CropGuard AI is a deep learning–powered web application that detects plant diseases from leaf images and provides actionable insights such as disease name, confidence score, severity, and treatment suggestions.

---

## 🚀 Live Demo

👉 **Try it here:**
https://cropguard-ai-80cp.onrender.com

---

## 🚀 Features

* 🔍 **AI-based Disease Detection** using ResNet50V2
* 🖼️ **Image Upload Interface** (drag & drop + preview)
* 📊 **Confidence Score Visualization**
* 🌱 **Disease Metadata Display** (symptoms, severity, treatment)
* ⚡ **Real-time Predictions via Flask API**
* 🎯 Designed for **agriculture support & early disease diagnosis**

---

## 🧠 Model Details

* **Architecture:** ResNet50V2
* **Dataset:** PlantVillage (10-class subset)
* **Input Size:** 224 × 224 × 3
* **Output:** Softmax (10 classes)

### 📋 Classes

| # | Class Label                 | Crop   | Disease              |
| - | --------------------------- | ------ | -------------------- |
| 0 | Corn___Common_Rust          | Corn   | Common Rust          |
| 1 | Corn___Gray_Leaf_Spot       | Corn   | Gray Leaf Spot       |
| 2 | Corn___Healthy              | Corn   | Healthy              |
| 3 | Corn___Northern_Leaf_Blight | Corn   | Northern Leaf Blight |
| 4 | Potato___Early_Blight       | Potato | Early Blight         |
| 5 | Potato___Healthy            | Potato | Healthy              |
| 6 | Potato___Late_Blight        | Potato | Late Blight          |
| 7 | Tomato___Bacterial_Spot     | Tomato | Bacterial Spot       |
| 8 | Tomato___Healthy            | Tomato | Healthy              |
| 9 | Tomato___Late_Blight        | Tomato | Late Blight          |

---

## 🛠️ Tech Stack

* **Backend:** Flask (Python)
* **ML Framework:** TensorFlow / Keras
* **Image Processing:** OpenCV, Pillow
* **Frontend:** HTML, CSS, JavaScript
* **Model:** ResNet50V2

---

## 📁 Project Structure

```
cropguard/
├── app.py
├── predict_pipeline.py
├── class_names.json
├── disease_info.json
├── requirements.txt
├── resnet.h5
├── static/
│   └── uploads/
└── templates/
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/cropguard-ai.git
cd cropguard-ai
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Add Model File

Place your trained model:

```
resnet.h5
```

Inside project root (or `/models/` folder).

---

### 4. Run Application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 🧪 How It Works

1. User uploads leaf image
2. Image is preprocessed (resize, normalize)
3. Model performs inference
4. Top prediction is selected
5. Metadata is fetched (symptoms, treatment, severity)
6. Results are displayed in UI

---

## 📊 Example Output

* **Prediction:** Tomato – Late Blight
* **Confidence:** 92.4%
* **Severity:** High
* **Treatment:** Apply fungicide and remove infected leaves

---

## ⚠️ Notes

* Model is trained on **controlled PlantVillage dataset**
* Real-world accuracy may vary due to:

  * Lighting conditions
  * Background noise
  * Image quality

---

## 🔮 Future Improvements

* Expand to full **38-class PlantVillage dataset**
* Add **real-world dataset fine-tuning**
* Deploy model via cloud storage (Drive/S3)
* Mobile app integration
* Ensemble learning for better accuracy

---

## 👩‍💻 Author

**Priyanshi Agarwal**

---

## 📌 License

This project is for educational and research purposes.
