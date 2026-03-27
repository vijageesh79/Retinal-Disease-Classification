# Retinal Disease Classification Using CNN, Liquid Neural Networks, and Explainable AI

## 1. Title
**Concept-Aware Multi-Label Retinal Disease Classification from Fundus Images Using ResNet50V2, Liquid Neural Networks, Grad-CAM, and SHAP**

---

## 2. Abstract (2-3 points)
- Retinal diseases such as diabetic retinopathy, glaucoma, cataract, AMD, and myopia are major causes of preventable blindness. Manual screening of fundus images is time-consuming and limited by specialist availability.
- This work proposes an interpretable multi-label classification pipeline using a pretrained CNN (`ResNet50V2`) for feature extraction and a `Liquid Neural Network` classifier for adaptive decision modeling.
- To improve clinical trust, the model provides explainability via `Grad-CAM` (region-level attention) and `SHAP` (feature attribution), with evaluation using accuracy, F1-score, AUC-ROC/AUC-PR, Hamming loss, and cross-dataset testing.

---

## 3. Objectives (1-2 points)
- Build an accurate multi-label retinal disease classifier for fundus images covering Normal, DR, Glaucoma, Cataract, AMD, Hypertension, Myopia, and Other abnormalities.
- Develop an interpretable and clinically meaningful AI system by integrating visual and attribution-based explanations for every prediction.

---

## 4. Introduction
Retinal diseases are among the leading causes of visual impairment worldwide. Many of these conditions can be managed effectively if detected early, but delayed diagnosis remains common, especially in regions with limited specialist availability. Fundus photography is a cost-effective and non-invasive modality used in population-scale eye screening programs.

Recent progress in deep learning has enabled automatic retinal disease detection from fundus images with high accuracy. However, most models are designed as black boxes and offer limited insight into how decisions are made. In healthcare, this limits physician trust and slows adoption in real clinical workflows.

This project addresses both predictive performance and interpretability. We combine transfer learning with explainable AI to build a robust and transparent retinal disease classification framework suitable for screening support.

---

## 5. Problem Statement
The problem is to design an AI system that can classify multiple retinal conditions from a single fundus image with high accuracy while also providing interpretable evidence for each prediction. Existing high-performing models often fail to explain *why* they predicted a disease class. Our aim is to develop a multi-label model that is:
- Accurate across major retinal disease categories,
- Interpretable through saliency and attribution maps,
- Generalizable across data splits and external datasets,
- Practical for clinical decision support and screening triage.

---

## 6. Literature Survey and Inference (25 reference papers)

### 6.1 Survey Summary
Prior work in retinal AI can be grouped into:
1. **Disease-specific models** (mostly diabetic retinopathy grading),
2. **General ophthalmic deep learning frameworks**,
3. **Explainable AI methods in medical imaging**,
4. **Generalization and clinical deployment studies**.

Most studies report strong performance on benchmark datasets but have gaps in interpretability, cross-dataset robustness, or multi-label clinical realism.

### 6.2 Reference Papers
1. Gulshan V, et al. Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs. *JAMA*. 2016.  
2. Ting DSW, et al. Development and Validation of a Deep Learning System for Diabetic Retinopathy and Related Eye Diseases. *JAMA*. 2017.  
3. Gargeya R, Leng T. Automated Identification of Diabetic Retinopathy Using Deep Learning. *Ophthalmology*. 2017.  
4. Pratt H, et al. Convolutional Neural Networks for Diabetic Retinopathy. *Procedia Computer Science*. 2016.  
5. Abràmoff MD, et al. Pivotal Trial of an Autonomous AI-Based Diagnostic System for Diabetic Retinopathy. *npj Digital Medicine*. 2018.  
6. Li Z, et al. Efficacy of a Deep Learning System for Detecting Glaucomatous Optic Neuropathy. *Nature Biomedical Engineering*. 2018.  
7. Christopher M, et al. Performance of Deep Learning Architectures for Glaucoma Detection in Fundus Photographs. *Scientific Reports*. 2018.  
8. Burlina PM, et al. Automated Detection of Age-Related Macular Degeneration from Color Fundus Images. *JAMA Ophthalmology*. 2017.  
9. Grassmann F, et al. Deep Learning for AMD Classification and Beyond. *Nature Communications*. 2018.  
10. Kermany DS, et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*. 2018.  
11. Poplin R, et al. Prediction of Cardiovascular Risk Factors from Retinal Fundus Photographs via Deep Learning. *Nature Biomedical Engineering*. 2018.  
12. Quellec G, et al. Deep Image Mining for Diabetic Retinopathy Screening. *Medical Image Analysis*. 2017.  
13. Voets M, et al. Replication Study: Development and Validation of a Deep Learning Algorithm for DR Detection. *PLOS ONE*. 2019.  
14. Beede E, et al. Human-Centered Evaluation of a Deep Learning System in Clinics. *CHI*. 2020.  
15. Tjoa E, Guan C. A Survey on Explainable Artificial Intelligence (XAI) for Healthcare. *IEEE TNNLS*. 2021.  
16. Selvaraju RR, et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *ICCV*. 2017.  
17. Lundberg SM, Lee SI. A Unified Approach to Interpreting Model Predictions (SHAP). *NeurIPS*. 2017.  
18. Zeiler MD, Fergus R. Visualizing and Understanding Convolutional Networks. *ECCV*. 2014.  
19. Ribeiro MT, et al. “Why Should I Trust You?” Explaining the Predictions of Any Classifier (LIME). *KDD*. 2016.  
20. Dosovitskiy A, et al. An Image is Worth 16x16 Words: Vision Transformer. *ICLR*. 2021.  
21. Tan M, Le Q. EfficientNet: Rethinking Model Scaling for CNNs. *ICML*. 2019.  
22. Howard A, et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*. 2018.  
23. Huang G, et al. Densely Connected Convolutional Networks (DenseNet). *CVPR*. 2017.  
24. Lechner M, et al. Liquid Time-Constant Networks. *AAAI*. 2020.  
25. ODIR Team. ODIR-2019/ODIR-5K: Ocular Disease Intelligent Recognition Dataset and Challenge Reports. 2019.

### 6.3 Inference from Literature
- High benchmark performance is achievable, but many models are disease-specific and not fully multi-label.
- Explainability is frequently added as an afterthought; integrated explainable pipelines are limited.
- Cross-dataset validation is often weak, which affects real-world reliability.
- There is strong research opportunity in combining adaptive models (LNN) with clinically interpretable outputs (Grad-CAM + SHAP).

---

## 7. Motivation
- Increasing diabetic and age-related eye disease burden requires scalable screening tools.
- Manual reading of large fundus volumes causes delays in referral and treatment.
- Clinicians need transparent AI outputs to trust model-assisted decisions.
- A practical AI solution should be accurate, interpretable, and deployable on moderate hardware.

---

## 8. Existing or Related Work (if any)
- Most existing work focuses on **single-task diagnosis** (e.g., only diabetic retinopathy).
- Some multi-disease systems exist, but many do not provide robust explanation mechanisms.
- Common backbones include `ResNet`, `EfficientNet`, and `DenseNet`; explainability methods include Grad-CAM and SHAP, but integration with temporal/adaptive classifiers like LNN is relatively less explored.
- Clinical deployment efforts show promising accuracy, but generalization and workflow acceptance remain open challenges.

---

## 9. Methodology
1. **Dataset and Preprocessing**
   - Use ODIR-5K fundus image dataset.
   - Resize, normalize, and clean label noise where possible.
   - Apply augmentation: rotation, flip, brightness/contrast jitter, zoom.

2. **Model Architecture**
   - Backbone: `ResNet50V2` (ImageNet pretrained) for deep feature extraction.
   - Classification head: `Liquid Neural Network` layers for adaptive multi-label decision making.
   - Output: sigmoid activations for multi-label probabilities.

3. **Training Strategy**
   - Phase 1: freeze backbone; train head.
   - Phase 2: unfreeze top backbone blocks and fine-tune end-to-end.
   - Regularization: dropout, L2 weight decay, early stopping, label smoothing.

4. **Explainability**
   - `Grad-CAM` to localize class-relevant retinal regions.
   - `SHAP` to quantify contribution of input features/patterns.
   - Optional text explanation templates for report generation.

5. **Evaluation**
   - Metrics: accuracy, precision, recall, F1 (macro/micro), AUC-ROC, AUC-PR, Hamming loss, subset accuracy.
   - Per-class confusion analysis and cross-dataset test (if secondary dataset available).

---

## 10. Timeline Plan for Implementation
| Week | Activity | Deliverable |
|---|---|---|
| 1-2 | Problem finalization, literature survey | Survey matrix, finalized scope |
| 3-4 | Dataset preparation and preprocessing pipeline | Cleaned dataset + scripts |
| 5-6 | Baseline CNN model training | Baseline metrics |
| 7-8 | LNN integration and tuning | Improved model checkpoints |
| 9 | Explainability modules (Grad-CAM, SHAP) | Visual explanation outputs |
| 10 | Evaluation and ablation studies | Metric tables, graphs |
| 11 | Cross-dataset testing and error analysis | Generalization report |
| 12 | Final documentation and presentation | Final report + PPT |

---

## 11. Hardware and Software Requirements

### Hardware
- CPU: Intel i5/Ryzen 5 or better
- RAM: Minimum 16 GB (recommended 32 GB)
- GPU: NVIDIA GPU with >=6 GB VRAM (recommended RTX 3060/4060 or better)
- Storage: >=100 GB free space (dataset + checkpoints + logs)

### Software
- OS: Windows/Linux/macOS
- Python: 3.9+
- Libraries: TensorFlow/PyTorch, NumPy, Pandas, scikit-learn, OpenCV, Matplotlib, Seaborn, SHAP
- Tools: Jupyter Notebook / VS Code / Cursor
- Version control: Git (recommended)

---

## 12. Results (Graphs / Tables / Test Cases)

### 12.1 Quantitative Results Table (template)
| Model | Accuracy | Macro F1 | Micro F1 | AUC-ROC | Hamming Loss |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |
| CNN + LNN (Proposed) | XX.XX | XX.XX | XX.XX | XX.XX | XX.XX |

### 12.1.1 Current Verified Preliminary Results (from repository)
The following values are directly taken from `outputs/evaluation/metrics.json` and should be treated as preliminary:

| Metric | Value |
|---|---:|
| Accuracy | 0.0625 |
| Macro F1 | 0.1095 |
| Micro F1 | 0.2647 |
| ROC-AUC (Macro) | 0.4090 |
| ROC-AUC (Micro) | 0.4798 |
| Average Precision (Macro) | 0.3207 |
| Hamming Loss | 0.3906 |
| Subset Accuracy | 0.0625 |

Note: These numbers indicate the model/pipeline needs further tuning and proper environment setup before final submission metrics are reported.

### 12.2 Class-wise Performance (template)
| Class | Precision | Recall | F1-score | AUC |
|---|---:|---:|---:|---:|
| Normal | XX.XX | XX.XX | XX.XX | XX.XX |
| DR | XX.XX | XX.XX | XX.XX | XX.XX |
| Glaucoma | XX.XX | XX.XX | XX.XX | XX.XX |
| Cataract | XX.XX | XX.XX | XX.XX | XX.XX |
| AMD | XX.XX | XX.XX | XX.XX | XX.XX |
| Hypertension | XX.XX | XX.XX | XX.XX | XX.XX |
| Myopia | XX.XX | XX.XX | XX.XX | XX.XX |
| Other | XX.XX | XX.XX | XX.XX | XX.XX |

### 12.3 Graphs to Include
- Training vs validation loss curve
- Training vs validation accuracy/F1 curve
- ROC and Precision-Recall curves (macro + per class)
- Confusion matrix heatmaps (one-vs-rest or class-wise)
- Grad-CAM visual examples (correct and incorrect predictions)
- SHAP summary plots for representative samples

### 12.4 Test Cases (template)
| Test Case | Input | Expected Output | Observed Output | Status |
|---|---|---|---|---|
| TC-01 | Normal fundus image | Normal high probability | Normal | Pass |
| TC-02 | DR image with lesions | DR positive | DR positive | Pass |
| TC-03 | Multi-disease sample | Multiple labels | Partial labels | Review |
| TC-04 | Low-quality image | Low confidence / warning | Unstable prediction | Needs preprocessing |

---

## 13. Conclusion
This project proposes an interpretable AI framework for multi-label retinal disease classification using transfer learning (`ResNet50V2`), adaptive classification (`Liquid Neural Networks`), and explainability (`Grad-CAM` + `SHAP`). The approach targets both strong predictive performance and clinical transparency, addressing a major gap in current medical AI workflows. With proper validation and cross-dataset testing, the framework can support early screening, triage, and physician-assisted diagnosis in ophthalmology.

Future work includes larger external validation, uncertainty estimation, handling domain shift from different camera devices, and lightweight deployment for edge or mobile screening systems.

