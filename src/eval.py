from __future__ import annotations

"""evaluate.py – minimal scorer for DBGorilla

Usage
-----
    python evaluate.py predictions.json

*The ground‑truth file path is fixed inside the script*
(`eval/ground_truth.json`).  You only pass the
**predictions JSON** you want to score.

* `predictions.json` – JSON list (or {"predictions": [...]}) of 315 dicts, in
  the same order as the ground truth.

The script prints overall **exact‑match accuracy** and **mean AST score**.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
GT_PATH = Path(__file__).resolve().parent / "eval" / "ground_truth.json"

WEIGHTS = {
    "collection": 0.40,
    "search_query": 0.15,
    "filters": 0.15,
    "aggregation": 0.15,
    "group_by": 0.15,
}

FILTER_KEYS = {
    "integer_property_filter",
    "text_property_filter",
    "boolean_property_filter",
}

AGG_KEYS = {
    "integer_property_aggregation",
    "text_property_aggregation",
    "boolean_property_aggregation",
}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _eq(a, b):
    if a in (None, {}, []) and b in (None, {}, []):
        return True
    return a == b


def _filters_equal(gt: Dict[str, Any], pr: Dict[str, Any]) -> bool:
    return all(_eq(gt.get(k), pr.get(k)) for k in FILTER_KEYS)


def _aggs_equal(gt: Dict[str, Any], pr: Dict[str, Any]) -> bool:
    return all(_eq(gt.get(k), pr.get(k)) for k in AGG_KEYS)


def _score_record(gt: Dict[str, Any], pr: Dict[str, Any]):
    detail = {
        "collection": gt.get("target_collection") == pr.get("target_collection"),
        "search_query": _eq(gt.get("search_query"), pr.get("search_query")),
        "filters": _filters_equal(gt, pr),
        "aggregation": _aggs_equal(gt, pr),
        "group_by": _eq(gt.get("groupby_property"), pr.get("groupby_property")),
    }
    ast = sum(w for k, w in WEIGHTS.items() if detail[k])
    return ast, all(detail.values())


def _load_gt() -> List[Dict[str, Any]]:
    if not GT_PATH.exists():
        print("Ground‑truth file not found at", GT_PATH, file=sys.stderr)
        sys.exit(1)
    with GT_PATH.open("r", encoding="utf-8") as f:
        return [rec["query"] for rec in json.load(f)]


def _load_preds(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "predictions" in data:
        data = data["predictions"]
    if not isinstance(data, list):
        print('Predictions file must be a list or {"predictions": [...] }', file=sys.stderr)
        sys.exit(1)
    return data


# ----------------------------------------------------------------------------
# Entry‑point
# ----------------------------------------------------------------------------


def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py predictions.json", file=sys.stderr)
        sys.exit(1)

    pred_file = Path(sys.argv[1])
    if not pred_file.exists():
        print("Predictions file not found:", pred_file, file=sys.stderr)
        sys.exit(1)

    gt = _load_gt()
    preds = _load_preds(pred_file)

    if len(gt) != len(preds):
        print(
            "Length mismatch:",
            len(gt),
            "ground truth vs",
            len(preds),
            "predictions",
            file=sys.stderr,
        )
        sys.exit(1)

    exact_hits = 0
    ast_total = 0.0
    for g, p in zip(gt, preds):
        ast, exact = _score_record(g, p)
        ast_total += ast
        exact_hits += exact

    n = len(gt)
    print("Exact‑match accuracy:", f"{exact_hits / n:.2%}")
    print("Mean AST score     :", f"{ast_total / n:.3f}")


if __name__ == "__main__":
    main()
