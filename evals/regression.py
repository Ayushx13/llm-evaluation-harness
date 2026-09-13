"""Regression comparison between two prompt versions.

    python -m evals.regression --baseline v1 --candidate v2

Reads the raw result files written by evals.runner and compares them test by
test. Aggregate accuracy can rise while individual cases break, so every case
is placed in one of four buckets:

  fixed       failed on the baseline, passes on the candidate
  regressed   passed on the baseline, fails on the candidate
  still_pass  passed on both
  still_fail  failed on both

Correctness here is category correctness, the same field the headline accuracy
uses. Priority flips are counted separately so they are not hidden. A case that
errored on either run is reported as unscored rather than as a pass or a fail.

The script never calls the API.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

RESULTS_DIR = pathlib.Path("results")


def load(version: str) -> tuple[dict, dict[str, dict]]:
    path = RESULTS_DIR / f"raw_results_{version}.json"
    if not path.exists():
        print(f"{path} not found. Run: python -m evals.runner --prompt {version}")
        sys.exit(1)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw["meta"], {r["test_id"]: r for r in raw["results"]}


def _row(before: dict, after: dict) -> dict:
    return {
        "test_id": after["test_id"],
        "input_type": after["input_type"],
        "input": after["input"][:220],
        "expected": after["expected"]["category"],
        "before": before["predicted"]["category"],
        "after": after["predicted"]["category"],
        "confidence_before": before["confidence"],
        "confidence_after": after["confidence"],
    }


def compare(base: dict[str, dict], cand: dict[str, dict]) -> dict:
    # Different suites would make every count meaningless, so refuse instead of
    # silently comparing the overlap.
    if set(base) != set(cand):
        only_base = sorted(set(base) - set(cand))[:5]
        only_cand = sorted(set(cand) - set(base))[:5]
        raise SystemExit(
            "baseline and candidate ran different test suites "
            f"(only in baseline: {only_base}, only in candidate: {only_cand}). "
            "Rerun both on the same data before comparing."
        )

    fixed, regressed, unscored = [], [], []
    still_pass = still_fail = 0
    priority_fixed = priority_regressed = 0

    for test_id in sorted(base):
        b, c = base[test_id], cand[test_id]
        if not b.get("predicted") or not c.get("predicted"):
            unscored.append(test_id)
            continue

        if b["category_correct"] and c["category_correct"]:
            still_pass += 1
        elif b["category_correct"]:
            regressed.append(_row(b, c))
        elif c["category_correct"]:
            fixed.append(_row(b, c))
        else:
            still_fail += 1

        if b["priority_correct"] and not c["priority_correct"]:
            priority_regressed += 1
        elif c["priority_correct"] and not b["priority_correct"]:
            priority_fixed += 1

    n_scored = len(base) - len(unscored)
    base_acc = sum(base[t]["category_correct"] for t in base if t not in unscored)
    cand_acc = sum(cand[t]["category_correct"] for t in cand if t not in unscored)

    return {
        "n_cases": len(base),
        "n_scored_both": n_scored,
        "n_unscored": len(unscored),
        "baseline_category_accuracy": round(base_acc / n_scored, 4) if n_scored else 0.0,
        "candidate_category_accuracy": round(cand_acc / n_scored, 4) if n_scored else 0.0,
        "fixed": len(fixed),
        "regressed": len(regressed),
        "still_pass": still_pass,
        "still_fail": still_fail,
        "net_change": len(fixed) - len(regressed),
        "priority_fixed": priority_fixed,
        "priority_regressed": priority_regressed,
        "regressions_by_input_type": dict(Counter(r["input_type"] for r in regressed).most_common()),
        "fixes_by_input_type": dict(Counter(r["input_type"] for r in fixed).most_common()),
        "regressions": regressed,
        "fixes": fixed,
        "unscored_test_ids": unscored,
    }


def main():
    ap = argparse.ArgumentParser(description="Compare two prompt versions test by test.")
    ap.add_argument("--baseline", default="v1")
    ap.add_argument("--candidate", default="v2")
    args = ap.parse_args()

    base_meta, base = load(args.baseline)
    cand_meta, cand = load(args.candidate)

    if base_meta.get("model") != cand_meta.get("model"):
        print(f"warning: model differs ({base_meta.get('model')} vs {cand_meta.get('model')}), "
              "so changes cannot be attributed to the prompt alone")

    report = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "model": cand_meta.get("model"),
        **compare(base, cand),
    }

    out = RESULTS_DIR / "regression_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 58)
    print(f"{args.baseline} -> {args.candidate}   {report['n_scored_both']} cases scored on both")
    print(f"category accuracy      {report['baseline_category_accuracy']:.1%} -> "
          f"{report['candidate_category_accuracy']:.1%}")
    print(f"fixed                  {report['fixed']}")
    print(f"regressed              {report['regressed']}")
    print(f"net change             {report['net_change']:+}")
    print(f"priority fixed/broken  {report['priority_fixed']} / {report['priority_regressed']}")
    print("=" * 58)
    if report["regressions"]:
        print("\nregressions")
        for r in report["regressions"]:
            print(f"  [{r['test_id']}] {r['input_type']:<20} {r['expected']} -> {r['after']} "
                  f"(conf {r['confidence_after']})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()