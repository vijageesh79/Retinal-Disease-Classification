# Review Q&A Prep (Likely Questions)

## Q1. Why this topic?
Retinal diseases are high-impact and screening volume is increasing. AI can support triage, but explainability is necessary for clinical trust.

## Q2. Why multi-label instead of single-label?
Real fundus images can contain multiple co-existing findings. Multi-label setup is closer to clinical reality.

## Q3. Why ResNet50V2?
Strong transfer-learning baseline in medical imaging with stable feature extraction and good ecosystem support.

## Q4. Why Liquid Neural Networks (LNN)?
To explore adaptive dynamics on top of CNN embeddings and compare with standard dense heads.

## Q5. Why Grad-CAM and SHAP both?
They provide complementary explainability:
- Grad-CAM: spatial focus regions
- SHAP: contribution-based attribution

## Q6. Current metrics are weak. Why?
Current metrics are preliminary snapshots. Final training/tuning is blocked by local runtime compatibility issue (TensorFlow with Python 3.14).

## Q7. How will you improve performance quickly?
- Use compatible Python/TensorFlow environment
- Perform controlled hyperparameter tuning
- Add class imbalance handling and better threshold tuning
- Validate on held-out and cross-dataset splits

## Q8. What have you completed for review?
- 13-point report draft
- modular code pipeline
- preliminary evaluation outputs
- review-ready status, slides outline, and artifact builder

## Q9. How do you prevent over-claiming?
Only report numbers present in versioned output files (`outputs/evaluation/*.json` and generated review tables).

## Q10. What is your next immediate milestone?
Produce final run artifacts: trained checkpoint, improved metrics, curves, confusion matrices, Grad-CAM/SHAP visuals, and updated paper results section.

