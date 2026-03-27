# IEEE Conference Paper — Extended Abstract & Problem Statement

**Title:** Concept-Aware Retinal Disease Classification Using CNN, Liquid Neural Networks, and Explainable AI

**Authors:** Saagar N Kashyap, Rishi Jha, and Vijageesh  
**Affiliation:** B.Tech Computer Science Engineering

---

## I. ABSTRACT (Extended — IEEE Style)

Retinal diseases, including diabetic retinopathy, glaucoma, cataract, age-related macular degeneration (AMD), hypertension retinopathy, and myopia, constitute a major cause of preventable vision loss and blindness globally. Fundus photography provides a non-invasive, cost-effective means for large-scale screening; however, the growing volume of examinations and the limited number of specialist ophthalmologists create a critical bottleneck in timely diagnosis and referral. Deep learning-based computer-aided diagnosis (CAD) systems have demonstrated high accuracy in detecting various retinal conditions from fundus images, yet the majority of such systems remain opaque: they do not reveal which regions of the image drive the prediction or how different findings contribute to the output. This lack of interpretability hinders clinical adoption, regulatory approval, and user trust, particularly in safety-critical medical applications.

In this work, we present an end-to-end interpretable framework for multi-label retinal disease classification from fundus images. Our system employs a convolutional neural network (CNN), specifically ResNet50V2 with transfer learning, for hierarchical feature extraction, and Liquid Neural Networks (LNNs) as the primary classification head to model adaptive, continuous dynamics over the extracted feature space. To ensure transparency and clinical usability, we integrate two complementary explainability mechanisms: Gradient-weighted Class Activation Mapping (Grad-CAM) for spatial attention visualization, and SHAP (SHapley Additive exPlanations) for input-level feature attribution. We further extend the pipeline with template-based and optional large language model (LLM)–driven natural-language explanations, and we evaluate generalization through cross-dataset assessment. The model is trained on the ODIR-5K (Ocular Disease Intelligent Recognition) dataset using a two-phase strategy—warm-up with a frozen backbone followed by fine-tuning—along with data augmentation and regularization (label smoothing, dropout, L2, and optional weight decay) to mitigate overfitting. Performance is measured using accuracy, macro and micro precision, recall, F1-score, area under the ROC curve (AUC-ROC), average precision (AUC-PR), Hamming loss, subset accuracy, and per-disease confusion analysis. The resulting framework delivers both discriminative performance and interpretable visual and textual outputs, positioning it as a viable screening aid and a reproducible baseline for interpretable deep learning in retinal imaging.

**Index Terms—** Retinal disease classification, fundus imaging, explainable artificial intelligence (XAI), Grad-CAM, SHAP, Liquid Neural Networks, convolutional neural networks (CNN), transfer learning, multi-label classification, medical image analysis, ODIR-5K, computer-aided diagnosis (CAD).

---

## II. INTRODUCTION AND PROBLEM STATEMENT (Extended — for IEEE Paper)

### A. Background and Motivation

Retinal and ocular diseases represent a significant public health burden worldwide. Diabetic retinopathy (DR), a microvascular complication of diabetes, is among the leading causes of blindness in working-age adults; glaucoma, characterized by progressive optic neuropathy, and cataract, the clouding of the lens, similarly contribute to substantial visual impairment and disability-adjusted life years lost [1]–[3]. Age-related macular degeneration (AMD), hypertension-related retinopathy, and pathological myopia further compound the diagnostic and screening load on healthcare systems. Early detection and timely intervention can slow or prevent disease progression in many of these conditions, making large-scale screening programs a priority in both developed and resource-limited settings.

Fundus photography—imaging of the posterior segment of the eye—provides a non-invasive, widely deployable modality for screening. The acquisition of fundus images has been scaled through portable and smartphone-based devices, leading to an increase in the volume of images that require expert interpretation. However, the number of trained ophthalmologists and retinal specialists remains limited, creating a gap between screening capacity and diagnostic throughput. Computer-aided diagnosis (CAD) systems based on deep learning have shown considerable promise in automating the detection and grading of multiple retinal diseases from fundus images, with performance often approaching or matching expert-level accuracy on selected tasks [4]–[6]. Such systems have the potential to triage cases, prioritize referrals, and support clinicians in resource-constrained environments.

### B. The Interpretability Gap

Despite their accuracy, many deep learning models for medical imaging operate as black boxes: given an input image, they produce a prediction (e.g., a disease label or severity grade) without exposing the reasoning process or the image regions that influenced the decision. In clinical practice, however, interpretability is essential. Clinicians need to understand *where* in the fundus the model has focused (e.g., optic disc, macula, vessels, lesions) and *why* a particular finding or combination of findings led to the output. Regulatory frameworks and medical device standards increasingly emphasize the need for explainability and transparency in AI-driven diagnostic support [7], [8]. Moreover, user acceptance and trust by physicians and patients depend on the ability to scrutinize and validate model behavior. Without interpretability, deployment of such systems in real-world clinical workflows remains limited, and the full potential of AI-assisted screening cannot be realized.

### C. Problem Formulation

Against this background, we formulate the following problem. We consider the task of **multi-label** retinal disease classification from a single fundus image: for each image, the system must predict the presence or absence of each of several disease categories (e.g., Normal, Diabetic Retinopathy, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other abnormalities), as in the ODIR-5K benchmark. The problem is thus not only to achieve high discriminative performance (accuracy, F1, AUC) but also to provide **interpretable** outputs that satisfy the following requirements: (1) **Spatial interpretability:** visualization of the image regions that the model attends to for a given prediction (e.g., via attention or activation maps); (2) **Attribution interpretability:** quantification of the contribution of input features or intermediate representations to the output (e.g., via gradient-based or Shapley-based methods); and (3) **Linguistic interpretability (optional):** generation of natural-language summaries that describe the model’s findings in a form accessible to clinicians. Furthermore, the system should be **robust** to limited data and domain shifts—achieved through transfer learning, data augmentation, and regularization—and **generalizable**, in the sense that its performance and explanations can be assessed on datasets beyond the primary training distribution (cross-dataset evaluation).

### D. Objectives and Contributions

The primary objectives of this work are: (1) to design and implement a multi-label retinal disease classification pipeline that combines a pretrained CNN for feature extraction with a Liquid Neural Network (LNN) classifier, leveraging the adaptive, continuous-time dynamics of LNNs for improved representation learning; (2) to integrate Grad-CAM and SHAP into the pipeline so that every prediction is accompanied by visual and attribution-based explanations; (3) to extend the system with template-based and optional LLM-based natural-language explanations; (4) to train and evaluate the system on the ODIR-5K dataset using a two-phase training strategy and comprehensive metrics; and (5) to assess generalization through cross-dataset evaluation and to release a reproducible codebase and evaluation protocol.

The main **contributions** of this paper are summarized as follows:

- **C1.** An end-to-end interpretable architecture for multi-label retinal disease classification that couples a ResNet50V2 backbone (transfer learning) with a Liquid Neural Network head, trained via warm-up and fine-tuning with augmentation and regularization.
- **C2.** Integration of two complementary explainability methods—Grad-CAM for spatial attention maps and SHAP (GradientExplainer) for feature attribution—with visualizations and feature importance summaries suitable for clinical review.
- **C3.** Extension of the pipeline with template-based and optional LLM-based natural-language explanation generation, enabling both offline and API-driven textual summaries of model predictions.
- **C4.** A comprehensive evaluation protocol including accuracy, precision, recall, F1, AUC-ROC, AUC-PR, Hamming loss, subset accuracy, per-class confusion matrices, ROC and precision–recall curves, and feature importance plots, along with cross-dataset evaluation to assess generalization.
- **C5.** Release of a modular, documented implementation (preprocessing, training, evaluation, explainability, and cross-dataset scripts) to support reproducibility and future research in interpretable retinal CAD.

### E. Scope and Organization

The remainder of the paper is organized as follows. Section II (or Related Work) reviews prior art in retinal disease classification and explainable AI in medical imaging. Section III describes the dataset, the proposed architecture (CNN + LNN), the training procedure, and the explainability modules (Grad-CAM, SHAP, LLM explanation). Section IV presents the experimental setup, metrics, and results, including ablation and cross-dataset analysis. Section V discusses the findings, limitations, and clinical relevance. Section VI concludes the paper and outlines future directions.

---

## III. STANDALONE PARAGRAPHS (for copy-paste into manuscript)

### Abstract (single paragraph, ~280 words)

Retinal diseases such as diabetic retinopathy, glaucoma, cataract, and age-related macular degeneration are among the leading causes of preventable vision loss worldwide. Fundus photography enables non-invasive, large-scale screening, but the scarcity of specialists and the volume of images create a bottleneck for timely diagnosis. Deep learning–based computer-aided diagnosis systems have achieved high accuracy in detecting retinal conditions from fundus images; however, most operate as black boxes and do not reveal which image regions or features drive their predictions, limiting clinical adoption and regulatory acceptance. In this work, we present an interpretable framework for multi-label retinal disease classification that combines a convolutional neural network (ResNet50V2) for feature extraction with Liquid Neural Networks (LNNs) as the classification head, trained on the ODIR-5K dataset using transfer learning, data augmentation, and regularization. We integrate Gradient-weighted Class Activation Mapping (Grad-CAM) and SHAP for spatial and attribution-based explainability, and we extend the pipeline with template-based and optional LLM-driven natural-language explanations. Generalization is assessed via cross-dataset evaluation. The system is evaluated using accuracy, macro and micro precision, recall, F1, AUC-ROC, average precision, Hamming loss, subset accuracy, and per-disease metrics. The framework delivers both discriminative performance and interpretable visual and textual outputs, suitable as a screening aid and a reproducible baseline for interpretable deep learning in retinal imaging.

### Problem Statement (single paragraph, for Introduction)

The core problem we address is twofold: first, to build an accurate multi-label classifier for multiple retinal conditions (Normal, Diabetic Retinopathy, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other) from fundus images using the ODIR-5K benchmark; and second, to make the system interpretable by providing (i) spatial visualizations of where the model attends in the image (e.g., via Grad-CAM), (ii) input-level or representation-level attributions (e.g., via SHAP), and (iii) optional natural-language summaries of the model’s findings. The system must further be robust to limited data and overfitting—achieved through transfer learning, augmentation, and regularization—and generalizable, as assessed by cross-dataset evaluation. By combining a CNN for feature extraction, Liquid Neural Networks for adaptive reasoning, and Grad-CAM plus SHAP for explainability, we deliver a single pipeline that meets both accuracy and interpretability requirements for clinically relevant, explainable retinal disease screening.

---

## IV. REFERENCES (placeholder — replace with actual citations)

[1] WHO World report on vision, 2019.  
[2] Yau et al., “Global prevalence and major risk factors of diabetic retinopathy,” Diabetes Care, 2012.  
[3] Tham et al., “Global prevalence of glaucoma and projections,” Ophthalmology, 2014.  
[4] Gulshan et al., “Development and validation of a deep learning algorithm for detection of diabetic retinopathy,” JAMA Ophthalmol., 2016.  
[5] Ting et al., “Deep learning in ophthalmology,” Prog. Retin. Eye Res., 2019.  
[6] ODIR-5K / Ocular disease recognition benchmarks.  
[7] FDA guidance on AI/ML-based SaMD.  
[8] EU AI Act / regulatory frameworks for explainability.

---

*Document prepared for IEEE conference submission. Replace placeholders (section numbers, references) as per the target conference template.*
