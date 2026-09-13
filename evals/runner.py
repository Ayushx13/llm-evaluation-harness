"""Main evaluation runner.

    python -m evals.runner --prompt v1

Calls the model once per test case, writes every raw result to disk, then
computes metrics from that file. Nothing downstream ever calls the API
again, so a bug in the metrics code costs nothing to fix.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime, timezone

from app.classifier import MODEL, SEED, classify, stats
from evals import calibration, clustering, hallucination, metrics

GT_PATH = pathlib.Path("data/ground_truth.json")
ADV_PATH = pathlib.Path("data/adversarial.json")
RESULTS_DIR = pathlib.Path("results")


def load_suite(include_adversarial: bool = True) -> list[dict]:
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    suite = [
        {
            "test_id": c["id"],
            "base_id": c["id"],
            "input_type": "normal",
            "input": c["input"],
            "expected": c["expected"],
            "note": c.get("label_reason", ""),
        }
        for c in gt
    ]

    if include_adversarial:
        if not ADV_PATH.exists():
            print(f"{ADV_PATH} not found. Run: python -m evals.adversarial")
            sys.exit(1)
        for c in json.loads(ADV_PATH.read_text(encoding="utf-8")):
            suite.append({
                "test_id": c["id"],
                "base_id": c.get("base_id"),
                "input_type": c["input_type"],
                "input": c["input"],
                "expected": c["expected"],
                "note": c.get("note", ""),
            })
    return suite


def run_one(case: dict, prompt_version: str) -> dict:
    record = {**case, "prompt_version": prompt_version, "predicted": None, "error": None}
    try:
        pred = classify(case["input"], prompt_version=prompt_version)
    except Exception as err:
        record["error"] = f"{type(err).__name__}: {err}"
        record["confidence"] = 0.0
        record["category_correct"] = False
        record["priority_correct"] = False
        record["exact_correct"] = False
        return record

    record["predicted"] = pred
    record["confidence"] = float(pred.get("confidence", 0.0))
    record["category_correct"] = pred["category"] == case["expected"]["category"]
    record["priority_correct"] = pred["priority"] == case["expected"]["priority"]
    record["exact_correct"] = record["category_correct"] and record["priority_correct"]
    record["hallucination"] = hallucination.check_output(case["input"], pred)
    return record


def main():
    ap = argparse.ArgumentParser(description="Run the full evaluation suite.")
    ap.add_argument("--prompt", default="v1", help="prompt version (v1 or v2)")
    ap.add_argument("--normal-only", action="store_true", help="skip adversarial cases")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    suite = load_suite(include_adversarial=not args.normal_only)
    if args.limit:
        suite = suite[: args.limit]

    print(f"model {MODEL} | prompt {args.prompt} | seed {SEED} | {len(suite)} cases\n")

    results = []
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = time.time()
    for i, case in enumerate(suite, 1):
        results.append(run_one(case, args.prompt))
        if i % 10 == 0 or i == len(suite):
            ok = sum(r["category_correct"] for r in results)
            print(f"  {i}/{len(suite)}  running accuracy {ok / i:.1%}")

    elapsed = round(time.time() - started, 1)

    raw_path = RESULTS_DIR / f"raw_results_{args.prompt}.json"
    raw_path.write_text(json.dumps(
        {
            "meta": {
                "model": MODEL,
                "seed": SEED,
                "prompt_version": args.prompt,
                "n_cases": len(results),
                "seconds": elapsed,
                "run_started_utc": started_at,
                # live API calls vs cache hits, so the report can say how much was replayed
                "classifier_stats": stats(),
            },
            "results": results,
        },
        indent=2, ensure_ascii=False,
    ), encoding="utf-8")

    acc = metrics.summarise(results)
    hal = hallucination.summarise(results)
    cal = calibration.summarise(results)
    clusters = clustering.cluster(results)
    plot_path = calibration.plot(results, str(RESULTS_DIR / f"calibration_{args.prompt}.png"))

    (RESULTS_DIR / f"metrics_{args.prompt}.json").write_text(json.dumps(
        {"accuracy": acc, "hallucination": hal, "calibration": cal},
        indent=2, ensure_ascii=False,
    ), encoding="utf-8")
    (RESULTS_DIR / f"failure_clusters_{args.prompt}.json").write_text(
        json.dumps(clusters, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 58)
    print(f"category accuracy      {acc['overall_category_accuracy']:.1%}")
    print(f"  normal inputs        {acc['normal_accuracy']:.1%}")
    print(f"  adversarial inputs   {acc['adversarial_accuracy']:.1%}")
    print(f"priority accuracy      {acc['overall_priority_accuracy']:.1%}")
    print(f"hallucination rate     {hal.get('hallucination_rate', 0):.1%}")
    print(f"ECE                    {cal['ece']}")
    print(f"mean overconfidence    {cal['mean_overconfidence']:+}")
    print(f"failure clusters       {clusters['n_clusters']} "
          f"over {clusters['n_failures']} failures")
    print("=" * 58)
    print("\naccuracy by input type")
    for t, row in acc["by_input_type"].items():
        print(f"  {t:<20} n={row['n']:<4} acc={row['category_accuracy']:.1%} "
              f"conf={row['mean_confidence']:.2f}")
    print("\nfailure clusters")
    for c in clusters["clusters"]:
        print(f"  {c['label']:<32} {c['n_failures']:>3} failures  "
              f"(mean conf {c['mean_confidence']:.2f})")

    print(f"\nwrote {raw_path}")
    if plot_path:
        print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
