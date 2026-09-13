"""Hallucination detection.

Definition used in this system
------------------------------
A hallucination is an output element that is not grounded in the input
ticket. In a classifier with no free text, there are exactly two things that
can be ungrounded:

  1. A label outside the allowed taxonomy (schema violation).
  2. An evidence_span the model presents as quoted from the ticket that does
     not actually occur in the ticket.

Case 2 is the interesting one and it is why evidence_span exists in the
schema. The model is told to copy text verbatim. If it returns something it
did not copy, it has fabricated its own justification.

Why this is not plain string matching
-------------------------------------
An exact substring test alone would be wrong in both directions. A model
that rewrites "four hundred rupees" as "400" is being sloppy, not
fabricating. So the check is three-way:

  SUPPORTED    exact substring after normalising whitespace and case
  PARAPHRASED  not exact, but every content word appears in the input and
               fuzzy similarity to the best-matching window is high
  FABRICATED   neither; the span contains material absent from the input

Only FABRICATED counts towards the hallucination rate. PARAPHRASED is
tracked separately and reported as a grounding-discipline failure. The
report documents where this detector breaks: it cannot catch a span that is
copied correctly but is irrelevant to the label, and it treats number/word
equivalence as paraphrase rather than resolving it.
"""

from __future__ import annotations

import difflib
import re

from app.schema import CATEGORIES, PRIORITIES

SUPPORTED = "supported"
PARAPHRASED = "paraphrased_unverified"
FABRICATED = "fabricated"
EMPTY = "empty_span"

STOPWORDS = {
    "a", "an", "the", "is", "was", "are", "were", "be", "been", "to", "of",
    "and", "or", "but", "in", "on", "at", "for", "it", "this", "that", "my",
    "i", "me", "not", "no", "has", "have", "had", "do", "does", "did", "so",
}

FUZZY_THRESHOLD = 0.82


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def content_words(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def best_window_ratio(span: str, source: str) -> float:
    """Highest similarity between the span and any same-length window of the
    source. Catches near-copies with a dropped or altered word."""
    span_n, src_n = normalise(span), normalise(source)
    if not span_n or not src_n:
        return 0.0
    n = len(span_n)
    if n >= len(src_n):
        return difflib.SequenceMatcher(None, span_n, src_n).ratio()
    best = 0.0
    step = max(1, n // 6)
    for start in range(0, len(src_n) - n + 1, step):
        window = src_n[start:start + n]
        best = max(best, difflib.SequenceMatcher(None, span_n, window).ratio())
        if best >= 0.99:
            break
    return best


def check_span(span: str, source: str) -> dict:
    if not span or not span.strip():
        return {"verdict": EMPTY, "ratio": 0.0, "missing_words": []}

    if normalise(span) in normalise(source):
        return {"verdict": SUPPORTED, "ratio": 1.0, "missing_words": []}

    src_words = set(content_words(source))
    missing = [w for w in content_words(span) if w not in src_words]
    ratio = best_window_ratio(span, source)

    if not missing and ratio >= FUZZY_THRESHOLD:
        return {"verdict": PARAPHRASED, "ratio": round(ratio, 3), "missing_words": []}

    return {"verdict": FABRICATED, "ratio": round(ratio, 3), "missing_words": missing[:8]}


def check_output(source: str, predicted: dict) -> dict:
    """Full grounding check for one model output."""
    problems = []

    if predicted.get("category") not in CATEGORIES:
        problems.append(f"invalid category: {predicted.get('category')!r}")
    if predicted.get("priority") not in PRIORITIES:
        problems.append(f"invalid priority: {predicted.get('priority')!r}")

    conf = predicted.get("confidence")
    if not isinstance(conf, (int, float)) or not 0.0 <= float(conf) <= 1.0:
        problems.append(f"confidence out of range: {conf!r}")

    span_result = check_span(predicted.get("evidence_span", ""), source)

    is_hallucination = bool(problems) or span_result["verdict"] == FABRICATED

    return {
        "is_hallucination": is_hallucination,
        "span_verdict": span_result["verdict"],
        "span_ratio": span_result["ratio"],
        "span_missing_words": span_result["missing_words"],
        "schema_problems": problems,
    }


def summarise(results: list[dict]) -> dict:
    """Aggregate hallucination stats over a list of raw result records."""
    scored = [r for r in results if r.get("predicted")]
    if not scored:
        return {}

    counts: dict[str, int] = {}
    for r in scored:
        v = r["hallucination"]["span_verdict"]
        counts[v] = counts.get(v, 0) + 1

    n_hall = sum(1 for r in scored if r["hallucination"]["is_hallucination"])

    by_type: dict[str, dict] = {}
    for r in scored:
        t = r["input_type"]
        b = by_type.setdefault(t, {"n": 0, "hallucinations": 0})
        b["n"] += 1
        b["hallucinations"] += int(r["hallucination"]["is_hallucination"])
    for t, b in by_type.items():
        b["rate"] = round(b["hallucinations"] / b["n"], 4)

    return {
        "n_scored": len(scored),
        "hallucination_rate": round(n_hall / len(scored), 4),
        "span_verdicts": counts,
        "by_input_type": by_type,
    }
