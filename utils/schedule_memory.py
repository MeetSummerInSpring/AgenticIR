from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence


class ScheduleMemory:
    """Structured, query-specific scheduling evidence."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        self.legacy_text: str | None = None
        if payload.get("schema_version") == 1 and isinstance(payload.get("rules"), list):
            self.rules = payload["rules"]
            self._validate_rules()
        elif isinstance(payload.get("distilled"), str):
            # Backward compatibility for explicitly supplied old memory files.
            self.rules = []
            self.legacy_text = payload["distilled"]
        else:
            raise ValueError(f"Unsupported schedule memory format: {self.path}")

    def query(
        self,
        degradations: Iterable[str],
        agenda: Sequence[str],
        *,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        degradation_set = set(degradations)
        agenda_set = set(agenda)
        matches: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for rule in self.rules:
            condition_set = set(rule["condition"]["degradations"])
            if not condition_set.issubset(degradation_set):
                continue

            recommendation = rule["recommendation"]
            order = recommendation["order"]
            if not set(order).issubset(agenda_set):
                continue

            exact = condition_set == degradation_set
            rank = (
                0 if exact else 1,
                -len(condition_set),
                -float(recommendation["confidence"]),
                -float(recommendation["failure_rate_margin"]),
                rule["rule_id"],
            )
            matches.append((rank, rule))

        matches.sort(key=lambda item: item[0])
        return [rule for _, rule in matches[:top_k]]

    def render_for_prompt(
        self,
        degradations: Iterable[str],
        agenda: Sequence[str],
        *,
        top_k: int = 3,
    ) -> tuple[str, list[str]]:
        if self.legacy_text is not None:
            return self.legacy_text, ["legacy-distilled"]

        degradation_list = list(degradations)
        agenda_list = list(agenda)
        matches = self.query(degradation_list, agenda_list, top_k=top_k)
        compact_rules = []
        for rule in matches:
            compact_rules.append(
                {
                    "rule_id": rule["rule_id"],
                    "condition": rule["condition"],
                    "recommendation": rule["recommendation"],
                    "candidates": rule["candidates"],
                    "provenance": rule["provenance"],
                }
            )

        payload = {
            "schema_version": 1,
            "instruction": (
                "Treat numeric fields as immutable evidence. Do not copy, "
                "recalculate, or invent statistics. Rules are conditional "
                "evidence, not hard constraints."
            ),
            "query": {
                "degradations": sorted(set(degradation_list)),
                "agenda": agenda_list,
            },
            "matched_rules": compact_rules,
        }
        return (
            json.dumps(payload, ensure_ascii=False, indent=2),
            [rule["rule_id"] for rule in matches],
        )

    def _validate_rules(self) -> None:
        seen_ids: set[str] = set()
        for rule in self.rules:
            required = {
                "rule_id",
                "condition",
                "recommendation",
                "candidates",
                "provenance",
            }
            if set(rule) != required:
                raise ValueError(
                    f"Invalid schedule rule fields for {rule.get('rule_id')}: "
                    f"{set(rule)}"
                )
            rule_id = rule["rule_id"]
            if not isinstance(rule_id, str) or not rule_id:
                raise ValueError("Each schedule rule requires a non-empty rule_id.")
            if rule_id in seen_ids:
                raise ValueError(f"Duplicate schedule rule id: {rule_id}")
            seen_ids.add(rule_id)

            degradations = rule["condition"].get("degradations")
            if (
                not isinstance(degradations, list)
                or not degradations
                or not all(isinstance(value, str) for value in degradations)
            ):
                raise ValueError(f"Invalid degradation condition in {rule_id}.")

            recommendation = rule["recommendation"]
            order = recommendation.get("order")
            if not isinstance(order, list) or not all(
                isinstance(value, str) for value in order
            ):
                raise ValueError(f"Invalid recommended order in {rule_id}.")
            confidence = recommendation.get("confidence")
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise ValueError(f"Invalid confidence in {rule_id}.")

            candidates = rule["candidates"]
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"Rule {rule_id} has no candidate evidence.")
