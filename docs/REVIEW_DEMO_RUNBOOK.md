# Review Demo Runbook (No-Hallucination)

## 1) Open these files first
- `docs/PROJECT_REPORT_FULL_DRAFT.md`
- `docs/REVIEW_36H_STATUS_AND_PLAN.md`
- `outputs/review/review_metrics_table.md`
- `outputs/review/review_classwise_table.md`

## 2) Demo flow (2-4 minutes)
1. Show project scope:
   - "8-class multi-label retinal classification with explainability."
2. Show implementation evidence:
   - `main.py` and `src/*` modules.
3. Show current outputs:
   - `outputs/evaluation/metrics.json`
   - `outputs/review/review_metrics_table.md`
4. Show report readiness:
   - 13 sections in `docs/PROJECT_REPORT_FULL_DRAFT.md`.
5. End with next-36-hours action:
   - environment fix, final training, final graphs, final paper edits.

## 3) If panel asks "why results are low?"
- Say:
  - "Current numbers are preliminary and verifiable from the repository."
  - "Main blocker was local TensorFlow environment mismatch."
  - "Immediate plan: run full training in compatible environment and regenerate metrics and visual artifacts."

## 4) If panel asks "what is complete today?"
- Complete:
  - problem framing + literature + methodology
  - modular implementation (dataset/model/eval/explainability)
  - preliminary metrics generation and review packaging
- In progress:
  - final tuned run and stronger result artifacts

## 5) Commands for review artifact refresh
```bash
python3 scripts/build_review_artifacts.py
```

