# Review Presentation (10 Slides)

## Slide 1: Title
- Concept-Aware Multi-Label Retinal Disease Classification
- Team names, guide name, department, date
- One-line goal: "Build accurate and explainable retinal screening AI"

## Slide 2: Clinical Problem
- Retinal diseases cause preventable vision loss
- Manual screening is specialist-intensive and slow
- Need: scalable and interpretable AI-assisted triage

## Slide 3: Problem Statement and Objectives
- Multi-label disease prediction from fundus images (8 ODIR classes)
- Objectives:
  - improve classification performance
  - provide explanation with Grad-CAM and SHAP

## Slide 4: Dataset and Labels
- ODIR-5K style dataset
- 8 labels: N, D, G, C, A, H, M, O
- Data split and preprocessing pipeline summary

## Slide 5: Proposed Method
- Backbone: ResNet50V2 transfer learning
- Head: Liquid Neural Network (LNN)
- Two-phase training: warm-up + fine-tuning

## Slide 6: Explainability
- Grad-CAM: where the model looked
- SHAP: which features contributed
- Why this matters: clinician trust + error analysis

## Slide 7: Current Implementation Status (Evidence)
- Core modules implemented:
  - `main.py`, `src/model.py`, `src/dataset.py`, `src/evaluation.py`, `src/explainability.py`
- Report draft completed with 13 required sections
- Preliminary metrics available in `outputs/evaluation/metrics.json`

## Slide 8: Current Preliminary Results (Verifiable)
- Use table from `outputs/review/review_metrics_table.md`
- Mention clearly: "Preliminary baseline snapshot; tuning in progress"
- Example key numbers:
  - Macro F1: 0.1095
  - Micro F1: 0.2647
  - ROC-AUC macro: 0.4090

## Slide 9: Gaps, Risks, and Next 36 Hours
- Gap: environment mismatch (TensorFlow with Python 3.14)
- Risk mitigation:
  - move runs to compatible env (Python 3.10/3.11 or Kaggle/Colab)
  - produce final plots and explainability artifacts
- Timeline: run training, finalize tables/graphs, lock paper draft

## Slide 10: Conclusion and Ask
- Work done: full pipeline + report structure ready
- In progress: final reproducible training/evaluation artifacts
- Ask from panel/guide:
  - feedback on model direction
  - approval for final hyperparameter sweep and external validation

---

## Speaker Notes (short)
- Do not over-claim; separate "implemented" vs "final validated."
- Always show file evidence (`outputs/review/*`, `docs/*`).
- If asked about weak metrics: state root cause and concrete fix plan.

