# 🚦 Real-Time Traffic Sign Detection with YOLOv8 & NVIDIA RTX 3060

> A high-performance, real-time traffic sign detection system — developed as a pilot study for an AI-based industrial **PCB Defect Detection** project.

---

## 🚀 Project Highlights

| | |
|---|---|
| **Architecture** | YOLOv8s (Small) by Ultralytics |
| **Hardware** | NVIDIA GeForce RTX 3060 (12GB VRAM) |
| **Environment** | Python 3.10 · PyTorch · CUDA 12.1 |
| **Dataset** | ~2,093 annotated images (Train / Val / Test) |
| **Inference Speed** | Stable **30+ FPS** on live webcam |

---

## 📊 Model Performance (Validation — 100 Epochs)

| Metric | Value |
|:---|:---|
| **mAP@50** | 0.964 |
| **Precision** | 0.950 |
| **Recall** | 0.932 |

> High precision minimizes false positives — critical for safety-sensitive applications like traffic sign recognition and PCB inspection.

---

## 📦 Dataset

**[road signs Dataset — Roboflow 100](https://universe.roboflow.com/roboflow-100/road-signs-6ih4y)**

This dataset is part of **RF100**, an Intel-sponsored benchmark initiative for evaluating object detection model generalizability.

| | |
|---|---|
| **Source** | Roboflow Universe (`roboflow-100/road-signs-6ih4y`) |
| **Total Images** | ~2,093 (annotated) |
| **Splits** | Train / Validation / Test |
| **License** | Open Source |

```bibtex
@misc{ road-signs-6ih4y_dataset,
  title        = { road signs Dataset },
  type         = { Open Source Dataset },
  author       = { Roboflow 100 },
  howpublished = { \url{ https://universe.roboflow.com/roboflow-100/road-signs-6ih4y } },
  url          = { https://universe.roboflow.com/roboflow-100/road-signs-6ih4y },
  journal      = { Roboflow Universe },
  publisher    = { Roboflow },
  year         = { 2023 },
  month        = { may }
}
```

---

## 🛠️ Local Setup & Installation

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- NVIDIA GPU with up-to-date drivers

### 1 — Create the Environment

```bash
conda create -n yolo_env python=3.10 -y
conda activate yolo_env
```

### 2 — Install PyTorch with CUDA Support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3 — Install Dependencies

```bash
pip install ultralytics opencv-python
```

---

## ▶️ Running Inference

Start real-time detection via webcam:

```bash
python test_canli.py
```

---

## 📂 Repository Structure

```
├── egitim.py          # Model training script
├── test_canli.py      # Real-time webcam inference & video recording
├── best.pt            # Trained model weights (100 epochs)
├── data.yaml          # Dataset class definitions and paths
└── proje_test.mp4     # Demo video
```

---

## 🔮 Future Work — PCB Defect Detection

This project is a **proof-of-concept** for an upcoming AI-based PCB Quality Control System.  
The same pipeline (YOLOv8 + CUDA + Custom Training) will be adapted to detect manufacturing defects including:

- Missing solder joints
- Short circuits
- Component misalignments

---

## 👤 Author

**Esra Evrim** — Computer Engineering Student
