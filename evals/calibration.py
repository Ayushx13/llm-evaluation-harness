"""Confidence calibration.

The question: when the system says 0.9, is it right about 90 percent of the
time? Bucket every prediction by its stated confidence, compare the mean
confidence in a bucket against the observed accuracy in that bucket, and
summarise the gap.

  ECE  expected calibration error, the bucket-size weighted mean gap
  MCE  maximum calibration error, the worst single bucket
  overconfidence  signed mean of (confidence - correct), positive means the
                  system claims more certainty than it earns
"""

from __future__ import annotations

import pathlib

BINS = [(i / 10, (i + 1) / 10) for i in range(10)]


def bucket(results: list[dict]) -> list[dict]:
    scored = [r for r in results if r.get("predicted")]
    out = []
    for lo, hi in BINS:
        rows = [
            r for r in scored
            if (lo <= r["confidence"] < hi) or (hi == 1.0 and r["confidence"] == 1.0)
        ]
        if not rows:
            out.append({"lo": lo, "hi": hi, "n": 0, "mean_confidence": None,
                        "accuracy": None, "gap": None})
            continue
        mean_conf = sum(r["confidence"] for r in rows) / len(rows)
        acc = sum(r["category_correct"] for r in rows) / len(rows)
        out.append({
            "lo": lo,
            "hi": hi,
            "n": len(rows),
            "mean_confidence": round(mean_conf, 4),
            "accuracy": round(acc, 4),
            "gap": round(mean_conf - acc, 4),
        })
    return out


def ece(buckets: list[dict], total: int) -> float:
    if not total:
        return 0.0
    return round(
        sum(b["n"] / total * abs(b["gap"]) for b in buckets if b["n"]), 4
    )


def mce(buckets: list[dict]) -> float:
    gaps = [abs(b["gap"]) for b in buckets if b["n"]]
    return round(max(gaps), 4) if gaps else 0.0


def summarise(results: list[dict]) -> dict:
    scored = [r for r in results if r.get("predicted")]
    buckets = bucket(scored)
    total = len(scored)

    signed = (
        sum(r["confidence"] - int(r["category_correct"]) for r in scored) / total
        if total else 0.0
    )

    high = [r for r in scored if r["confidence"] >= 0.9]
    ood = [r for r in scored if r["input_type"] == "ood"]

    return {
        "n_scored": total,
        "ece": ece(buckets, total),
        "mce": mce(buckets),
        "mean_overconfidence": round(signed, 4),
        "buckets": buckets,
        "high_confidence_slice": {
            "n": len(high),
            "threshold": 0.9,
            "accuracy": round(sum(r["category_correct"] for r in high) / len(high), 4)
            if high else None,
        },
        "ood_slice": {
            "n": len(ood),
            "mean_confidence": round(sum(r["confidence"] for r in ood) / len(ood), 4)
            if ood else None,
            "accuracy": round(sum(r["category_correct"] for r in ood) / len(ood), 4)
            if ood else None,
        },
    }


def plot(results: list[dict], path: str = "results/calibration_curve.png") -> str | None:
    """Reliability diagram. Returns the path, or None if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return None

    buckets = [b for b in bucket(results) if b["n"]]
    if not buckets:
        return None

    xs = [b["mean_confidence"] for b in buckets]
    ys = [b["accuracy"] for b in buckets]
    ns = [b["n"] for b in buckets]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}
    )

    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.plot(xs, ys, "o-", color="#c0392b", label="observed")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                    xytext=(6, -10), fontsize=8)
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("actual accuracy")
    ax.set_title("Reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    ax2.bar([b["lo"] + 0.05 for b in buckets], ns, width=0.08, color="#2980b9")
    ax2.set_xlabel("confidence bucket")
    ax2.set_ylabel("count")
    ax2.set_xlim(0, 1)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
