"""
merge_for_eval.py — Merge expected_standards into inference output for local evaluation.

Usage:
    python scripts/merge_for_eval.py \
        --predictions data/public_results.json \
        --ground-truth public_test_set.json \
        --output data/public_results_eval.json

Then evaluate:
    python eval_script.py --results data/public_results_eval.json
"""

import json
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge expected_standards into predictions for eval")
    parser.add_argument("--predictions", required=True, help="Path to inference output JSON")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth JSON with expected_standards")
    parser.add_argument("--output", required=True, help="Path to write merged output JSON")
    args = parser.parse_args()

    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    with open(args.ground_truth, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Build lookup: id -> expected_standards
    gt_map = {item["id"]: item.get("expected_standards", []) for item in ground_truth}

    merged = []
    for pred in predictions:
        item = dict(pred)
        item["expected_standards"] = gt_map.get(pred["id"], [])
        merged.append(item)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Merged {len(merged)} records → {args.output}")


if __name__ == "__main__":
    main()
