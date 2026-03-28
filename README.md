# 🫁 AI-Powered Lung Cancer Detection System

**NEXATHON 2.0 — Problem Statement C02 — Healthcare Innovation**

A multi-task deep learning system that classifies CT scan slices as Normal / Benign / Malignant, detects and localises lung nodules with bounding boxes, and generates explainable Grad-CAM heatmaps — all delivered through a browser-based Streamlit web application.

---

## Architecture Overview

```
CT Scan Slice (PNG/JPG)
        │
        ▼
  Preprocessing
  (HU windowing + lung mask)
        │
   ┌────┴────────────┐
   ▼                 ▼
EfficientNet-B4    YOLOv8-m
  Classifier        Detector
  (Normal /        (Bounding boxes
  Benign /          per nodule)
  Malignant)
   │
   ▼
Grad-CAM Heatmap
        │
        ▼
  Streamlit Web App
  + PDF Report
```

---

## Project Structure

```
lung_cancer_detection/
├── configs/
│   └── config.yaml              ← All hyperparameters in one place
├── data/
│   ├── iq_oth_nccd/             ← Classification dataset (3 class folders)
│   ├── ct_lung_finder/          ← Segmentation dataset (2D images + masks)
│   ├── nodule_malignancy/       ← Detection dataset (images + XML annotations)
│   └── yolo_dataset/            ← Auto-generated YOLO format dataset
├── models/
│   ├── classifier/
│   │   ├── dataset.py           ← PyTorch Dataset + DataLoader
│   │   └── train_classifier.py  ← EfficientNet-B4 training script
│   └── detector/
│       ├── convert_to_yolo.py   ← VOC XML → YOLO format converter
│       └── train_detector.py    ← YOLOv8-m training script
├── gradcam/
│   └── gradcam_utils.py         ← Grad-CAM generation utilities
├── app/
│   ├── app.py                   ← Main Streamlit application
│   ├── inference.py             ← Model loading + prediction pipeline
│   └── report_generator.py      ← PDF report generation
├── utils/
│   ├── preprocess.py            ← CT image preprocessing utilities
│   └── visualize.py             ← Visualisation helpers
├── download_data.py             ← One-click Kaggle dataset downloader
├── requirements.txt
└── README.md
```

---

## Quick Start (Step by Step)

### Step 1 — Set Up Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Verify your GPU:
```python
import torch
print(torch.cuda.get_device_name(0))   # Should show RTX 4050
```

### Step 2 — Configure Kaggle API

1. Go to https://www.kaggle.com/settings → API → Create New Token
2. Place the downloaded `kaggle.json` at `C:\Users\YourName\.kaggle\kaggle.json`

### Step 3 — Download Datasets

```bash
python download_data.py
```

This downloads all three Kaggle datasets (~1.1 GB total).

### Step 4 — Convert Detection Dataset to YOLO Format

```bash
python models/detector/convert_to_yolo.py
```

This converts PASCAL VOC XML annotations to YOLO .txt format and creates the `data/yolo_dataset/` directory.

### Step 5 — Train the Classifier

```bash
python models/classifier/train_classifier.py
```

Expected time on RTX 4050 6GB: **~1 hour**
Saves: `models/classifier/best_model.pth`

### Step 6 — Train the Detector

```bash
python models/detector/train_detector.py
```

Expected time on RTX 4050 6GB: **~1.5 hours**
Saves: `models/detector/runs/weights/best.pt`

### Step 7 — Launch the Web App

```bash
streamlit run app/app.py
```

Open your browser at http://localhost:8501

---

## Datasets

| Dataset | Source | Task | Size |
|---------|--------|------|------|
| IQ-OTH/NCCD | Kaggle (hamdallak) | Classification | ~600 MB |
| Finding Lungs in CT Data | Kaggle (kmader) | Lung Segmentation | ~662 MB |
| Lung Nodule Malignancy | Kaggle (andrewmvd) | Nodule Detection | ~400 MB |

---

## Model Details

**EfficientNet-B4 Classifier**
- Input: 512×512 RGB CT slice
- Backbone: EfficientNet-B4 pretrained on ImageNet
- Head: AdaptiveAvgPool → Dropout(0.4) → FC(1792→512→3)
- Output: Softmax probabilities for [Normal, Benign, Malignant]
- Training: Transfer learning, freeze early layers, fine-tune last 3 blocks
- Mixed precision: Yes (RTX 4050 float16)
- Target accuracy: 94%+

**YOLOv8-m Detector**
- Input: 640×640 CT slice
- Architecture: CSPDarknet + PANet + Decoupled anchor-free head
- Output: Bounding boxes [x,y,w,h] + confidence per nodule
- Target mAP@0.5: 88%+

---

## Hardware Requirements

- GPU: NVIDIA RTX 4050 6GB VRAM (or equivalent)
- RAM: 16GB recommended
- Storage: ~5 GB free (datasets + model weights)
- Python: 3.10+
- CUDA: 12.1+

---

## Disclaimer

This system is a research prototype built for NEXATHON 2.0. It is not a certified medical device and must not be used for actual clinical diagnosis without review by a qualified radiologist.


