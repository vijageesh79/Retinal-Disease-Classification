# Concept-Aware Retinal Disease Classification using Explainable AI

A deep learning framework for **retinal disease classification** using fundus images with integrated **Explainable AI visualizations**.
The project focuses on building an interpretable medical imaging system that can assist in identifying ocular diseases while showing **where the model is focusing inside the retina**.

---

# Project Overview

Retinal diseases are a major cause of vision impairment worldwide. Automated detection systems using **deep learning** can assist ophthalmologists by providing faster preliminary diagnosis.

However, most deep learning models behave like **black boxes**.
To address this problem, this project integrates **Explainable AI techniques** to visualize which regions of the retinal image influence the prediction.

This project implements:

* Retinal image classification
* Model interpretability using Grad-CAM
* Visual overlays for clinical insight
* A pipeline suitable for research experimentation

---

# Dataset

This project uses the **ODIR-5K (Ocular Disease Intelligent Recognition)** dataset.

Dataset contains:

* ~5000 retinal fundus images
* Left and right eye images
* Multiple ocular disease labels

### Disease Categories

| Label | Disease                          |
| ----- | -------------------------------- |
| N     | Normal                           |
| D     | Diabetic Retinopathy             |
| G     | Glaucoma                         |
| C     | Cataract                         |
| A     | Age-related Macular Degeneration |
| H     | Hypertension                     |
| M     | Myopia                           |
| O     | Other abnormalities              |

Dataset link:

https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

---

# Project Architecture

```
Retinal Image
      │
      ▼
Image Preprocessing
(resize, normalize)
      │
      ▼
Deep Learning Model
(CNN classifier)
      │
      ▼
Prediction
(retinal disease class)
      │
      ▼
Explainability Module
(Grad-CAM)
      │
      ▼
Visual Interpretation
(heatmap + overlay)
```

---

# Explainable AI

Medical AI systems require transparency for clinical adoption.
This project integrates **Grad-CAM** to visualize model attention.

Grad-CAM highlights regions that contribute the most to a prediction.

Example pipeline:

```
Input Retina Image
      ↓
Forward Pass
      ↓
Gradient Computation
      ↓
Feature Map Weighting
      ↓
Heatmap Generation
      ↓
Overlay on Original Image
```

Output visualization:

* Original retinal image
* Grad-CAM heatmap
* Overlay highlighting model focus

---

# Current Results

| Metric              | Value |
| ------------------- | ----- |
| Training Accuracy   | ~95%  |
| Validation Accuracy | ~50%  |

The model currently shows **overfitting**, which is common in medical imaging tasks due to limited dataset size. Future improvements will address this.

---

# Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy, Pandas
* Matplotlib
* scikit-learn
* SHAP (explainability)
* Kaggle Notebooks (optional)

---

# Project Structure

```
retinal-disease-classification/
├── dataset/              # ODIR-5K (or ODIR-2) images + CSV
├── docs/
│   └── SDP_ROADMAP.md   # Senior Design Project roadmap
├── configs/
├── models/               # Saved weights (e.g. concept_lnn_optimal.h5)
├── notebooks/
│   └── training_pipeline.ipynb
├── outputs/
│   ├── evaluation/      # Metrics, ROC/PR curves, confusion matrices, feature importance
│   ├── gradcam/
│   ├── shap/
│   └── cross_dataset/   # Cross-dataset evaluation results
├── scripts/
│   ├── evaluate_cross_dataset.py
│   └── generate_explanation.py  # LLM-style text explanation for an image
├── src/
│   ├── dataset.py
│   ├── data_pipeline.py  # tf.data + augmentation
│   ├── model.py          # CNN + Liquid Neural Network
│   ├── explainability.py # Grad-CAM + SHAP
│   ├── evaluation.py     # Metrics + visualizations
│   └── llm_explanation.py
├── main.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```
git clone [https://github.com/yourusername/retinal-disease-classification](https://github.com/saagarnkashyap/DrishtiNet).git
```

Install dependencies:

```
pip install tensorflow
pip install opencv-python
pip install numpy pandas matplotlib
```

---

# Running the Project

1. Download the dataset from Kaggle; place it in `dataset/` (see `dataset/README.md`).
2. Install: `pip install -r requirements.txt`
3. **Full pipeline (train + evaluate + explainability):**
   ```bash
   python main.py
   ```
4. **Or run the notebook:** `notebooks/training_pipeline.ipynb` (same pipeline).
5. **Optional:** Cross-dataset evaluation:
   ```bash
   python scripts/evaluate_cross_dataset.py [--csv path/to.csv] [--img_dir path/to/images]
   ```
6. **Optional:** Generate natural-language explanation for one image:
   ```bash
   python scripts/generate_explanation.py --image path/to/fundus.jpg --model models/concept_lnn_optimal.h5
   ```
   Use `--use_api` and set `OPENAI_API_KEY` for API-generated explanations.

---

# Grad-CAM Visualization Example

The system generates interpretability visualizations showing the retinal regions responsible for predictions.

Output includes:

* Original image
* Heatmap
* Overlay visualization

These visualizations help verify that the model focuses on clinically relevant retinal structures.

---

# Research Contribution

This project explores **interpretable deep learning for medical imaging** by combining:

* Image classification
* Explainable AI
* Retinal disease detection

The system can serve as a research baseline for:

* medical AI interpretability
* cross-dataset generalization
* clinical decision support systems

---

# Future Work (Implementation Status)

| Item | Status | Notes |
|------|--------|------|
| **Transfer learning** | Done | ResNet50V2 backbone in `src/model.py` |
| **Data augmentation** | Done | In-model augmenter + `src/data_pipeline.py` (tf.data) |
| **Regularization** | Done | Dropout, L2, label smoothing; optional `WEIGHT_DECAY` env for AdamW |
| **SHAP explanations** | Done | `src/explainability.py` |
| **Feature importance visualization** | Done | Per-class mean prediction + prevalence in `outputs/evaluation/feature_importance_by_class.png` |
| **Liquid Neural Networks** | Done | `LiquidCell` and LNN head in `src/model.py` |
| **LLM-based medical explanation** | Done | Template + optional API in `src/llm_explanation.py`; `scripts/generate_explanation.py` |
| **Cross-dataset evaluation** | Done | `scripts/evaluate_cross_dataset.py` (ODIR-format CSV + image dir) |

---

# Senior Design Project (SDP)

This repository is structured to support a **Senior Design Project**. See **[docs/SDP_ROADMAP.md](docs/SDP_ROADMAP.md)** for:

* SDP objectives, deliverables, and timeline
* Problem statement and report outline
* How to run the full pipeline and extensions

---

# Authors

Saagar N Kashyap,
B.Tech Computer Science Engineering

Rishi Jha,
B.Tech Computer Science Engineering

Vijageesh,
B.Tech Computer Science Engineering

---

# License

This project is intended for **academic research and educational purposes**.
