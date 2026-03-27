"""
LLM-based medical explanation generation (SDP Future Work).
Produces natural-language explanations for model predictions using:
1) Template-based explanations (no API, offline).
2) Optional OpenAI-compatible API for richer text (set OPENAI_API_KEY if desired).
"""
import os
import re

# ODIR disease names for readable explanations
DISEASE_NAMES = {
    "N": "No abnormality",
    "D": "Diabetic Retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "Age-related Macular Degeneration",
    "H": "Hypertension",
    "M": "Myopia",
    "O": "Other abnormalities",
}

LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]


def _template_explanation(pred_probs, top_k=3, threshold=0.3):
    """
    Build a short clinical-style explanation from prediction probabilities.
    pred_probs: (8,) or list of 8 floats (sigmoid outputs).
    """
    pred_probs = list(pred_probs)[:8]
    indexed = [(LABELS[i], pred_probs[i], DISEASE_NAMES.get(LABELS[i], LABELS[i])) for i in range(len(LABELS))]
    indexed.sort(key=lambda x: -x[1])
    positive = [(lbl, p, name) for lbl, p, name in indexed if p >= threshold][:top_k]
    if not positive:
        positive = [indexed[0]]
    lines = [
        "The model indicates the following findings (probabilities in parentheses):"
    ]
    for lbl, p, name in positive:
        pct = round(100 * p, 1)
        lines.append(f"  • {name} ({pct}%)")
    lines.append("\nThis is a screening aid only. Please consult an ophthalmologist for diagnosis.")
    return "\n".join(lines)


def generate_explanation(pred_probs, use_api=False, image_path=None, gradcam_summary=None):
    """
    Generate natural-language explanation for a single prediction.
    pred_probs: array/list of length 8 (sigmoid outputs).
    use_api: if True and OPENAI_API_KEY is set, call OpenAI (or compatible) API for a longer explanation.
    image_path, gradcam_summary: optional context for API call.
    Returns: string explanation.
    """
    explanation = _template_explanation(pred_probs)
    if not use_api:
        return explanation
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return explanation
    try:
        import openai
        if not hasattr(openai, "OpenAI"):
            return explanation
        client = openai.OpenAI(api_key=api_key)
        prompt = (
            "You are a medical AI assistant. Based on the following retinal screening model output, "
            "write 2-3 short, clear sentences for a clinician. Do not diagnose; state only what the model suggests.\n\n"
            f"Model output (probabilities per condition): {dict(zip(LABELS, [round(float(p), 3) for p in pred_probs]))}\n\n"
            + (f"Grad-CAM summary: {gradcam_summary}\n" if gradcam_summary else "")
            + "Reply in plain text only."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        if getattr(resp, "choices", None) and len(resp.choices) > 0:
            text = getattr(resp.choices[0].message, "content", "") or ""
            if text.strip():
                return text.strip() + "\n\n[Template fallback]\n" + explanation
    except Exception:
        pass
    return explanation


def generate_explanation_for_image(model, image_batch, class_names=LABELS):
    """
    Convenience: run model on one image batch, return template explanation for the first sample.
    image_batch: (1, H, W, 3) numpy array, normalized [0,1].
    """
    preds = model.predict(image_batch, verbose=0)
    probs = preds[0]
    return generate_explanation(probs, use_api=False), probs
