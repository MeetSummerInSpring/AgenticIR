from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def _candidate_from_statistics(
    execution_path: str,
    statistics: dict[str, Any],
) -> dict[str, Any]:
    sample_count = int(statistics["total"])
    failure_rates = {
        degradation: float(value)
        for degradation, value in statistics["fail rate"].items()
        if degradation != "total"
    }
    failure_counts = {
        degradation: int(statistics[degradation])
        for degradation in failure_rates
    }
    return {
        "order": execution_path.split("+"),
        "sample_count": sample_count,
        "failure_counts": failure_counts,
        "failure_rates": failure_rates,
        "total_failure_rate": float(statistics["fail rate"]["total"]),
    }


def _ordering_confidence(
    best: dict[str, Any],
    runner_up: dict[str, Any],
) -> float:
    """Bound-inspired confidence from aggregate bounded failure observations."""

    margin = (
        runner_up["total_failure_rate"]
        - best["total_failure_rate"]
    )
    if margin <= 0:
        return 0.0

    n_best = best["sample_count"]
    n_runner_up = runner_up["sample_count"]
    denominator = (1.0 / n_best) + (1.0 / n_runner_up)
    confidence = 1.0 - math.exp(-2.0 * margin * margin / denominator)
    return min(max(confidence, 0.0), 1.0)


def build_rules(
    fail_rate_hub: dict[str, Any],
    *,
    source_path: str,
) -> dict[str, Any]:
    rules = []
    for degradation_key, plan_statistics in sorted(fail_rate_hub.items()):
        candidates = [
            _candidate_from_statistics(execution_path, statistics)
            for execution_path, statistics in plan_statistics.items()
        ]
        candidates.sort(
            key=lambda candidate: (
                candidate["total_failure_rate"],
                candidate["order"],
            )
        )
        if not candidates:
            continue

        best = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else candidates[0]
        margin = max(
            0.0,
            runner_up["total_failure_rate"] - best["total_failure_rate"],
        )
        rules.append(
            {
                "rule_id": f"order:{degradation_key}",
                "condition": {
                    "degradations": degradation_key.split("+"),
                },
                "recommendation": {
                    "order": best["order"],
                    "total_failure_rate": best["total_failure_rate"],
                    "runner_up_order": runner_up["order"],
                    "runner_up_total_failure_rate": runner_up["total_failure_rate"],
                    "failure_rate_margin": margin,
                    "confidence": _ordering_confidence(best, runner_up),
                    "confidence_method": "bounded_mean_margin_v1",
                    "supporting_samples": best["sample_count"],
                    "comparison_samples": runner_up["sample_count"],
                },
                "candidates": candidates,
                "provenance": {
                    "source_path": source_path,
                    "source_key": degradation_key,
                    "aggregation": (
                        "Mean of per-degradation binary failure rates; "
                        "lower is better."
                    ),
                },
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source": source_path,
        "rules": rules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build structured scheduling rules from exploration statistics."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("memory/fail_rate.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("memory/schedule_rules.json"),
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as file:
        fail_rate_hub = json.load(file)
    rules = build_rules(fail_rate_hub, source_path=str(args.input))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(rules, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Wrote {len(rules['rules'])} structured rules to {args.output}.")


if __name__ == "__main__":
    main()
