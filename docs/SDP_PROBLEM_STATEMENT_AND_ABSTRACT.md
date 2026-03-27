# Senior Design Project — Problem Statement & Abstract

**Title:** Concept-Aware Retinal Disease Classification Using CNN, Liquid Neural Networks, and Explainable AI  

**Team:** Saagar N Kashyap, Rishi Jha, Vijageesh — B.Tech Computer Science Engineering  

---

## Abstract

Retinal diseases such as diabetic retinopathy, glaucoma, cataract, and age-related macular degeneration are among the leading causes of preventable vision loss worldwide. Fundus imaging enables early screening, but the shortage of specialists and the volume of images make manual review a bottleneck. Deep learning models can automate preliminary detection, yet most operate as black boxes and lack the interpretability needed for clinical trust and adoption. This project designs and implements an **interpretable** multi-label retinal disease classification system that (1) uses a **CNN (ResNet50V2)** for feature extraction and **Liquid Neural Networks (LNN)** as the main classifier on the ODIR-5K dataset, (2) integrates **Explainable AI** through **Grad-CAM** and **SHAP** to visualize where and why the model focuses in the fundus image, and (3) extends the pipeline with **template- and optional LLM-based natural-language explanations** and **cross-dataset evaluation** for generalization assessment. The system is trained with transfer learning, data augmentation, and regularization to improve robustness, and is evaluated using accuracy, F1, ROC-AUC, precision–recall curves, and per-disease metrics. The resulting framework provides both accurate multi-disease predictions and interpretable visual and textual explanations, suitable as a screening aid and a research baseline for interpretable medical AI.

**Keywords:** Retinal disease classification, fundus imaging, explainable AI, Grad-CAM, SHAP, Liquid Neural Networks, deep learning, medical imaging, ODIR-5K.

---

## Problem Statement

Retinal diseases—including diabetic retinopathy (DR), glaucoma, cataract, age-related macular degeneration (AMD), hypertension retinopathy, and myopia—are major causes of vision impairment and blindness. Early detection through fundus imaging can slow progression and preserve sight, but timely diagnosis is limited by the scarcity of ophthalmologists and the growing number of screenings. Automated systems based on deep learning have shown strong performance on retinal image analysis; however, they are often opaque to clinicians, who need to understand **where** in the image the model bases its decision and **why** a given finding is suggested. Without such interpretability, adoption in clinical workflows and regulatory acceptance remain difficult.

The core **problem** addressed in this project is therefore twofold: (1) to build an accurate **multi-label** classifier for multiple retinal conditions from fundus images (ODIR-5K: Normal, Diabetic Retinopathy, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other), and (2) to make the system **interpretable** by providing visual explanations (e.g., attention heatmaps and feature importance) and optional natural-language summaries, so that it can serve as a transparent screening aid rather than an opaque black box. Additionally, the system should be **robust** (via transfer learning, augmentation, and regularization) and **generalizable** (assessed through cross-dataset evaluation), aligning with real-world deployment and research needs.

By combining a CNN for feature extraction, Liquid Neural Networks for adaptive reasoning, and Grad-CAM plus SHAP for explainability, this project delivers a single pipeline that meets both accuracy and interpretability requirements for a Senior Design Project in the domain of interpretable medical AI for retinal disease screening.

---

## Short Version (for forms or space limits)

### Abstract (short)

We build an interpretable deep learning system for multi-label retinal disease classification from fundus images (ODIR-5K). The architecture uses a CNN (ResNet50V2) for features and Liquid Neural Networks for classification, with Grad-CAM and SHAP for explainability and optional LLM-based text explanations. The pipeline is evaluated with standard metrics and cross-dataset evaluation to support its use as a screening aid and research baseline.

### Problem Statement (short)

Retinal diseases cause significant vision loss; early detection via fundus imaging is limited by specialist availability. Deep learning can automate screening but often lacks interpretability. This project develops an accurate, multi-label retinal disease classifier that is interpretable (Grad-CAM, SHAP, optional text explanations) and robust (transfer learning, augmentation, regularization, cross-dataset evaluation), to serve as a transparent screening aid and research baseline.
