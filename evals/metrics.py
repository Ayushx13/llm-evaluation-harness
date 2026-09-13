"""Accuracy metrics. Plain Python, no sklearn, so every number is auditable."""

from __future__ import annotations

from collections import defaultdict

from app.schema import CATEGORIES


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def accuracy(results: list[dict], field: str = "category_correct") -> float:
    scored = [r for r in results if r.get("predicted")]
    return _rate(sum(1 for r in scored if r[field]), len(scored))


def accuracy_by_input_type(results: list[dict]) -> dict:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if r.get("predicted"):
            buckets[r["input_type"]].append(r)

    out = {}
    for t, rows in sorted(buckets.items()):
        out[t] = {
            "n": len(rows),
            "category_accuracy": _rate(sum(r["category_correct"] for r in rows), len(rows)),
            "priority_accuracy": _rate(sum(r["priority_correct"] for r in rows), len(rows)),
            "exact_accuracy": _rate(sum(r["exact_correct"] for r in rows), len(rows)),
            "mean_confidence": round(
                sum(r["confidence"] for r in rows) / len(rows), 4
            ),
        }
    return out


def per_class(results: list[dict]) -> dict:
    """Precision, recall and F1 per category, computed by hand."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for r in results:
        if not r.get("predicted"):
            continue
        gold = r["expected"]["category"]
        pred = r["predicted"]["category"]
        if gold == pred:
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1

    out = {}
    for c in CATEGORIES:
        precision = _rate(tp[c], tp[c] + fp[c])
        recall = _rate(tp[c], tp[c] + fn[c])
        f1 = _rate(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        out[c] = {
            "support": tp[c] + fn[c],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return out


def confusion(results: list[dict]) -> dict:
    """gold -> predicted -> count, for the categories only."""
    matrix: dict[str, dict[str, int]] = {g: {p: 0 for p in CATEGORIES} for g in CATEGORIES}
    for r in results:
        if not r.get("predicted"):
            continue
        gold = r["expected"]["category"]
        pred = r["predicted"]["category"]
        if gold in matrix and pred in matrix[gold]:
            matrix[gold][pred] += 1
    return matrix


def summarise(results: list[dict]) -> dict:
    scored = [r for r in results if r.get("predicted")]
    errors = [r for r in results if not r.get("predicted")]
    normal = [r for r in scored if r["input_type"] == "normal"]
    adversarial = [r for r in scored if r["input_type"] != "normal"]

    return {
        "n_total": len(results),
        "n_scored": len(scored),
        "n_errors": len(errors),
        "overall_category_accuracy": accuracy(scored, "category_correct"),
        "overall_priority_accuracy": accuracy(scored, "priority_correct"),
        "overall_exact_accuracy": accuracy(scored, "exact_correct"),
        "normal_accuracy": accuracy(normal, "category_correct"),
        "adversarial_accuracy": accuracy(adversarial, "category_correct"),
        "by_input_type": accuracy_by_input_type(scored),
        "per_class": per_class(scored),
        "confusion": confusion(scored),
    }
