"""
Generate natural-language explanation for a single fundus image (SDP Future Work – LLM).
Usage:
  python scripts/generate_explanation.py --image path/to/image.jpg --model models/concept_lnn_optimal.h5
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import cv2
import tensorflow as tf

from src.model import build_concept_aware_lnn
from src.llm_explanation import generate_explanation_for_image, generate_explanation


def load_image(path, size=224):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = (img / 255.0).astype(np.float32)
    return np.expand_dims(img, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to fundus image")
    parser.add_argument("--model", type=str, default=os.path.join(ROOT, "models", "concept_lnn_optimal.h5"))
    parser.add_argument("--use_api", action="store_true", help="Use OpenAI API for richer explanation (set OPENAI_API_KEY)")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Image not found: {args.image}")
        return 1
    if not os.path.isfile(args.model):
        print(f"Model not found: {args.model}. Train first with python main.py")
        return 1

    print("Loading model...")
    model, _, _ = build_concept_aware_lnn(input_shape=(224, 224, 3), num_classes=8)
    model.load_weights(args.model)

    print("Loading image...")
    batch = load_image(args.image)
    explanation, probs = generate_explanation_for_image(model, batch)
    text = generate_explanation(probs, use_api=args.use_api)
    print("\n--- Explanation ---\n")
    print(text)
    return 0


if __name__ == "__main__":
    exit(main())
