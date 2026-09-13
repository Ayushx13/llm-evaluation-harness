"""Failure mode clustering.

Grouping failures by input_type alone would be a surface-level grouping: it
says where the failure happened, not why. So every failure is first tagged
with observable signals (did the predicted label match a word that only
appears in the distractor? was the decoy keyword negated? was confidence
high?), and clusters are defined by those signals plus a stated hypothesis
about model behaviour.

Each cluster carries:
  hypothesis  what the model is doing wrong, at the behaviour level
  fix         one concrete change another engineer could implement

The clusters are assigned in priority order so each failure lands in exactly
one bucket. Cross-cutting tags (overconfidence, fabricated evidence) are
counted separately because they co-occur with all of the above.
"""

from __future__ import annotations

import re
from collections import defaultdict

from evals.adversarial import CONTRADICTIONS, DISTRACTORS

CATEGORY_VOCAB = {
    "payment": {"charge", "charged", "payment", "paid", "card", "upi", "deducted", "money", "emi", "invoice"},
    "delivery": {"delivery", "delivered", "courier", "parcel", "package", "shipment", "shipped", "tracking", "address"},
    "account": {"login", "log", "password", "otp", "account", "profile", "email", "unsubscribe", "points"},
    "product_defect": {"broken", "cracked", "defect", "damaged", "working", "missing", "size", "leaked", "quality"},
    "refund": {"refund", "refunded", "return", "returned", "cancelled", "credit", "money back", "pickup"},
    "other": set(),
}

HIGH_CONF = 0.85

CLUSTER_DEFS = {
    "negation_blindness": {
        "label": "Negation blindness",
        "hypothesis": (
            "The model routes on keyword presence rather than keyword scope. When a "
            "category word appears inside a negated clause ('I am not asking for a "
            "refund'), attention still fires on the token and the label follows the "
            "word instead of the meaning. The negation cue is a short function word "
            "carrying little weight against a strong domain noun."
        ),
        "fix": (
            "Add a scope-resolution step before classification: ask the model to first "
            "return the single clause that states the actual request, then classify only "
            "that clause. Add 8 to 10 negation examples to the prompt as few-shot pairs, "
            "each pairing a negated decoy with the correct label."
        ),
    },
    "distractor_capture": {
        "label": "Irrelevant-context capture",
        "hypothesis": (
            "Unrelated chatter that happens to contain domain vocabulary pulls the label "
            "toward the chatter's category. The model treats the whole message as one "
            "bag of topic evidence and has no notion of which sentence is the request. "
            "Longer inputs dilute the real complaint's share of the signal."
        ),
        "fix": (
            "Two-stage pipeline: a cheap extraction call returns only the sentences that "
            "state a problem or request, then the classifier runs on that reduced text. "
            "Log both texts so the reduction step itself can be evaluated."
        ),
    },
    "contradiction_collapse": {
        "label": "Contradiction collapse",
        "hypothesis": (
            "Given two conflicting facts the model silently picks the more recent or "
            "more concrete one instead of flagging the conflict, and does not lower "
            "confidence when it does. It has no way to express 'the ticket disagrees "
            "with itself', so an unrepresentable state gets forced into a confident label."
        ),
        "fix": (
            "Add a boolean field conflicting_claims to the output schema with a required "
            "explanation, and a routing rule that any ticket with conflicting_claims=true "
            "goes to a human queue regardless of category."
        ),
    },
    "ood_overreach": {
        "label": "Out-of-distribution overreach",
        "hypothesis": (
            "Given an input from outside the support domain the model still produces a "
            "confident in-domain label. It was given six categories and treats them as "
            "exhaustive; 'other' is read as a weak catch-all for odd tickets rather than "
            "as a refusal. Confidence is generated as fluent text, not from evidence, so "
            "it stays high even with nothing to ground it."
        ),
        "fix": (
            "Make refusal explicit: rename the escape hatch to not_a_support_ticket, "
            "describe it in the prompt as the correct answer for anything not about an "
            "order or account, and enforce a hard rule that an empty evidence_span forces "
            "that label with confidence capped at 0.3."
        ),
    },
    "surface_brittleness": {
        "label": "Surface-form brittleness",
        "hypothesis": (
            "The label moves when only the wording changes. Character noise and rewording "
            "break the token patterns the model relies on, which shows that the decision "
            "rests on lexical cues rather than the situation described. Typos also break "
            "verbatim span copying, so grounding degrades at the same time."
        ),
        "fix": (
            "Add a normalisation pass in front of the classifier (spell correction limited "
            "to a domain dictionary) and make evidence_span a character offset pair into "
            "the original input rather than free text, so the model cannot silently repair "
            "the quote."
        ),
    },
    "taxonomy_boundary": {
        "label": "Taxonomy boundary confusion",
        "hypothesis": (
            "Failures concentrate on a small number of category pairs where the taxonomy "
            "itself is under-specified, for example a charged-but-undelivered order that "
            "is both payment and delivery. The model is not wrong so much as forced to "
            "choose, and it chooses by surface salience rather than by root cause."
        ),
        "fix": (
            "Write explicit tie-break rules into the prompt for the three most confused "
            "pairs found in the confusion matrix, stated as root-cause rules, and allow a "
            "secondary_category field so a genuinely dual ticket is not scored as a miss."
        ),
    },
    "unclustered": {
        "label": "Unclustered",
        "hypothesis": "Failures that did not match any signal rule. Inspect by hand.",
        "fix": "Read these individually before the report is final.",
    },
}


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def vocab_pull(text: str, category: str) -> int:
    return len(words(text) & CATEGORY_VOCAB.get(category, set()))


def tag(failure: dict) -> list[str]:
    """Observable signals for one failing result."""
    tags = []
    itype = failure["input_type"]
    pred = failure["predicted"]["category"]
    src = failure["input"]

    if itype == "negation":
        tags.append("negation_input")
    if itype == "irrelevant_context":
        distractor_text = " ".join(d for d in DISTRACTORS if d in src)
        if distractor_text and vocab_pull(distractor_text, pred) > 0:
            tags.append("predicted_from_distractor")
        else:
            tags.append("irrelevant_context_input")
    if itype == "contradiction":
        if any(c in src for c in CONTRADICTIONS):
            tags.append("contradiction_input")
    if itype == "ood":
        tags.append("ood_input")
    if itype in {"typos", "paraphrase"}:
        tags.append("surface_change_input")
    if itype == "normal":
        tags.append("clean_input")

    if failure["confidence"] >= HIGH_CONF:
        tags.append("high_confidence")
    if failure.get("hallucination", {}).get("is_hallucination"):
        tags.append("fabricated_evidence")

    # Did a negated decoy word drive the label?
    lowered = src.lower()
    for neg in ("not ", "n't ", "no "):
        if neg in lowered:
            idx = lowered.find(neg)
            window = lowered[idx:idx + 60]
            if vocab_pull(window, pred) > 0:
                tags.append("label_from_negated_clause")
                break

    return tags


def assign(failure: dict, tags: list[str]) -> str:
    """Priority-ordered cluster assignment. One failure, one cluster."""
    if "label_from_negated_clause" in tags or "negation_input" in tags:
        return "negation_blindness"
    if "predicted_from_distractor" in tags or "irrelevant_context_input" in tags:
        return "distractor_capture"
    if "ood_input" in tags:
        return "ood_overreach"
    if "contradiction_input" in tags:
        return "contradiction_collapse"
    if "surface_change_input" in tags:
        return "surface_brittleness"
    if "clean_input" in tags:
        return "taxonomy_boundary"
    return "unclustered"


def cluster(results: list[dict], max_examples: int = 4) -> dict:
    failures = [
        r for r in results
        if r.get("predicted") and not r["category_correct"]
    ]

    grouped: dict[str, list[dict]] = defaultdict(list)
    tag_counts: dict[str, int] = defaultdict(int)

    for f in failures:
        tags = tag(f)
        for t in tags:
            tag_counts[t] += 1
        grouped[assign(f, tags)].append({**f, "_tags": tags})

    clusters = []
    for name, rows in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        meta = CLUSTER_DEFS[name]
        pairs: dict[str, int] = defaultdict(int)
        for r in rows:
            pairs[f"{r['expected']['category']} -> {r['predicted']['category']}"] += 1

        clusters.append({
            "cluster": name,
            "label": meta["label"],
            "n_failures": len(rows),
            "share_of_failures": round(len(rows) / len(failures), 4) if failures else 0.0,
            "mean_confidence": round(sum(r["confidence"] for r in rows) / len(rows), 4),
            "top_confusions": dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:5]),
            "hypothesis": meta["hypothesis"],
            "proposed_fix": meta["fix"],
            "examples": [
                {
                    "test_id": r["test_id"],
                    "input": r["input"][:220],
                    "expected": r["expected"]["category"],
                    "predicted": r["predicted"]["category"],
                    "confidence": r["confidence"],
                    "evidence_span": r["predicted"].get("evidence_span", "")[:120],
                }
                for r in rows[:max_examples]
            ],
        })

    return {
        "n_failures": len(failures),
        "n_clusters": len(clusters),
        "cross_cutting_tags": dict(sorted(tag_counts.items(), key=lambda kv: -kv[1])),
        "clusters": clusters,
    }
