# Senior Design Project (SDP) Roadmap

## Concept-Aware Retinal Disease Classification with Explainable AI

This document outlines how to shape the current project into a **Senior Design Project (SDP)** and what to implement to meet typical SDP requirements.

---

## 1. What Makes This an SDP?

A strong SDP typically has:

| SDP Requirement | How This Project Addresses It |
|----------------|-------------------------------|
| **Clear problem statement** | Automated, interpretable detection of retinal diseases from fundus images to support (not replace) clinical diagnosis. |
| **Engineering scope** | Full pipeline: data → preprocessing → CNN+LNN model → evaluation → explainability (Grad-CAM, SHAP) → optional LLM explanations and cross-dataset evaluation. |
| **Novelty / research angle** | Liquid Neural Networks for adaptive reasoning; multi-method explainability; (optional) LLM-generated clinical-style explanations; cross-dataset generalization. |
| **Deliverables** | Trained model, metrics, visualizations, documentation, (optional) demo app or API. |
| **Evaluation & reproducibility** | Comprehensive metrics (accuracy, F1, AUC, PR, per-class), plots, fixed splits, and clear run instructions. |
| **Documentation** | README, SDP report/thesis, architecture diagrams, user guide. |

---

## 2. SDP Structure (Suggested)

### 2.1 Title and Team

- **Title:** *Concept-Aware Retinal Disease Classification Using CNN, Liquid Neural Networks, and Explainable AI*
- **Team:** Saagar N Kashyap, Rishi Jha, Vijageesh (B.Tech CSE)
- **Advisor:** [Assign faculty advisor]

### 2.2 Problem Statement

> Retinal diseases (e.g., diabetic retinopathy, glaucoma, cataract) are a leading cause of vision loss. Early detection via fundus imaging can slow progression, but expert review is scarce. We aim to build an **interpretable** deep learning system that (1) classifies multiple retinal conditions from fundus images, (2) explains *where* and *why* the model focuses, and (3) can be extended to natural-language explanations and cross-dataset use.

### 2.3 Objectives

1. **Primary:** Multi-label classification of 8 retinal disease categories (ODIR-5K) using a CNN for feature extraction and Liquid Neural Networks for the classifier.
2. **Explainability:** Integrate Grad-CAM and SHAP, plus feature importance visualizations, so predictions are interpretable.
3. **Robustness:** Use transfer learning, data augmentation, and regularization to reduce overfitting and improve generalization.
4. **Extensions (Future Work):** LLM-based medical explanation generation; cross-dataset evaluation and domain adaptation.

### 2.4 Deliverables Checklist

- [ ] **Codebase:** Preprocessing, training, evaluation, explainability (Grad-CAM, SHAP), and extension modules.
- [ ] **Model & artifacts:** Saved best model, configs, and evaluation metrics (e.g. `outputs/evaluation/metrics.json`).
- [ ] **Visualizations:** Training curves, confusion matrices, ROC/PR curves, per-class metrics, Grad-CAM and SHAP samples, feature importance plots.
- [ ] **Documentation:** README, setup/run instructions, SDP report (problem, methods, results, future work).
- [ ] **Reproducibility:** `requirements.txt`, fixed random seeds, and clear data placement (e.g. `dataset/README.md`).
- [ ] **(Optional) Demo:** Simple web or CLI demo: upload image → prediction + explanation (heatmap + optional text).

### 2.5 Timeline (Example – One Semester)

| Phase | Weeks | Activities |
|-------|--------|------------|
| **Phase 1** | 1–2 | Finalize dataset, preprocessing, and baseline (CNN+LNN) training. |
| **Phase 2** | 3–4 | Add augmentation pipeline, regularization; tune and record metrics. |
| **Phase 3** | 5–6 | Implement and compare Grad-CAM and SHAP; add feature importance visualization. |
| **Phase 4** | 7–8 | LLM-based explanation module; cross-dataset evaluation / domain adaptation. |
| **Phase 5** | 9–10 | Integration, documentation, SDP report, and (optional) demo. |

---

## 3. Future Work (from README) – Implementation Status

| Item | Status | Location / Notes |
|------|--------|-------------------|
| **Transfer learning** | ✅ Done | ResNet50V2 backbone in `src/model.py` |
| **Data augmentation** | ✅ Enhanced | Augmenter in model; optional `tf.data` pipeline in `src/data_pipeline.py` |
| **Regularization** | ✅ Enhanced | Dropout, L2 in LNN; optional weight decay in `main.py` / config |
| **SHAP explanations** | ✅ Done | `src/explainability.py` – GradientExplainer + visualizations |
| **Feature importance visualization** | ✅ Done | Per-class SHAP bar and summary in `src/explainability.py` / `src/evaluation.py` |
| **Liquid Neural Networks** | ✅ Done | `LiquidCell` and LNN head in `src/model.py` |
| **LLM-based medical explanation** | ✅ Added | `src/llm_explanation.py` – template + optional API for natural language explanations |
| **Cross-dataset domain adaptation** | ✅ Added | `scripts/evaluate_cross_dataset.py` and optional fine-tune script |

---

## 4. What to Write in Your SDP Report

1. **Introduction:** Motivation, problem, scope, and contributions.
2. **Related work:** Retinal disease classification, explainable AI in medical imaging, LNNs.
3. **Methodology:** Dataset, preprocessing, architecture (CNN + LNN), training (warm-up + fine-tuning), explainability (Grad-CAM, SHAP), and extensions (LLM, cross-dataset).
4. **Experiments:** Setup, metrics, main results (tables/figures), ablation (e.g. with/without augmentation, with/without LNN).
5. **Discussion:** Strengths, limitations, clinical relevance, ethical considerations.
6. **Conclusion and future work:** Summary and next steps (e.g. external validation, clinician study).

---

## 5. Running the Full SDP Pipeline

```bash
# 1. Setup
pip install -r requirements.txt
# Place ODIR-5K in dataset/ (see dataset/README.md)

# 2. Train + evaluate + explainability
python main.py

# 3. (Optional) Cross-dataset evaluation
python scripts/evaluate_cross_dataset.py --config configs/cross_dataset.yaml

# 4. (Optional) Generate LLM-style explanation for an image
python scripts/generate_explanation.py --image path/to/image.jpg --model models/concept_lnn_optimal.h5
```

---

## 6. References (for SDP Report)

- ODIR-5K dataset; Grad-CAM (Selvaraju et al.); SHAP (Lundberg & Lee); Liquid Time-Constant Networks (Hasani et al.); relevant retinal screening / explainability papers.

Use this roadmap to align your implementation, report, and deliverables with typical SDP expectations.
