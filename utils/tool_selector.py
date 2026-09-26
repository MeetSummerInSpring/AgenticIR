from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Optional, Sequence

from .episode_store import EpisodeStore, ToolStatistics


@dataclass(frozen=True)
class RankedTool:
    tool_name: str
    score: float
    posterior_failure_rate: float
    mean_duration_seconds: Optional[float]
    peak_vram_mb: Optional[float]
    quality_observations: int
    execution_failures: int
    evidence_observations: int
    priority: int
    selected: bool = False
    exploration_pick: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToolSelector:
    """Deterministic, evidence-gated and cost-aware tool ranking.

    During calibration, least-observed tools are tried first so early stopping
    cannot permanently starve tools later in the registry.
    """

    def __init__(
        self,
        *,
        profile_path: Path | str,
        episode_store: EpisodeStore,
        top_k_override: Optional[int] = None,
        exploration_rate_override: Optional[float] = None,
        seed: int = 0,
    ) -> None:
        self.profile_path = Path(profile_path)
        with self.profile_path.open("r", encoding="utf-8") as file:
            self.profile = json.load(file)

        if self.profile.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported tool profile schema: {self.profile.get('schema_version')}"
            )

        defaults = self.profile.get("defaults", {})
        self.default_top_k = self._positive_int(
            top_k_override
            if top_k_override is not None
            else defaults.get("max_tools", 3),
            "max_tools",
        )
        self.minimum_observations = self._nonnegative_int(
            defaults.get("minimum_observations_for_pruning", 3),
            "minimum_observations_for_pruning",
        )
        self.quality_weight = self._nonnegative_float(
            defaults.get("quality_weight", 0.75), "quality_weight"
        )
        self.latency_weight = self._nonnegative_float(
            defaults.get("latency_weight", 0.20), "latency_weight"
        )
        self.vram_weight = self._nonnegative_float(
            defaults.get("vram_weight", 0.05), "vram_weight"
        )
        self.exploration_rate = self._probability(
            exploration_rate_override
            if exploration_rate_override is not None
            else defaults.get("exploration_rate", 0.05),
            "exploration_rate",
        )
        self.episode_store = episode_store
        self.random = random.Random(seed)

    def select(
        self,
        subtask: str,
        tools: Sequence[Any],
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        if not tools:
            raise ValueError(f"No tools registered for subtask {subtask!r}.")

        tool_config = (
            self.profile.get("subtasks", {})
            .get(subtask, {})
            .get("tools", {})
        )
        statistics = self.episode_store.get_tool_statistics(subtask)
        observed_latencies = [
            stat.mean_duration_seconds
            for stat in statistics.values()
            if stat.mean_duration_seconds is not None
        ]
        neutral_latency = median(observed_latencies) if observed_latencies else None

        raw_rows: list[dict[str, Any]] = []
        for original_index, tool in enumerate(tools):
            name = tool.tool_name
            config = tool_config.get(name, {})
            stat = statistics.get(name)
            failure_rate = (
                stat.posterior_failure_rate if stat is not None else 0.5
            )
            quality_observations = (
                stat.quality_observations if stat is not None else 0
            )
            execution_failures = (
                stat.execution_failures if stat is not None else 0
            )
            evidence_observations = (
                stat.evidence_observations if stat is not None else 0
            )
            duration = (
                stat.mean_duration_seconds
                if stat is not None and stat.mean_duration_seconds is not None
                else config.get("estimated_latency_seconds", neutral_latency)
            )
            peak_vram = config.get("peak_vram_mb")
            priority = int(config.get("priority", original_index))
            raw_rows.append(
                {
                    "tool": tool,
                    "tool_name": name,
                    "failure_rate": float(failure_rate),
                    "quality_observations": quality_observations,
                    "execution_failures": execution_failures,
                    "evidence_observations": evidence_observations,
                    "duration": self._optional_nonnegative_float(duration),
                    "peak_vram": self._optional_nonnegative_float(peak_vram),
                    "priority": priority,
                }
            )

        latency_values = [
            row["duration"] for row in raw_rows if row["duration"] is not None
        ]
        vram_values = [
            row["peak_vram"] for row in raw_rows if row["peak_vram"] is not None
        ]
        max_log_latency = max(
            (math.log1p(value) for value in latency_values),
            default=0.0,
        )
        max_vram = max(vram_values, default=0.0)

        for row in raw_rows:
            latency_cost = (
                math.log1p(row["duration"]) / max_log_latency
                if row["duration"] is not None and max_log_latency > 0
                else 0.0
            )
            vram_cost = (
                row["peak_vram"] / max_vram
                if row["peak_vram"] is not None and max_vram > 0
                else 0.0
            )
            row["score"] = (
                self.quality_weight * row["failure_rate"]
                + self.latency_weight * latency_cost
                + self.vram_weight * vram_cost
            )

        enough_evidence = all(
            row["evidence_observations"] >= self.minimum_observations
            for row in raw_rows
        )
        if enough_evidence:
            raw_rows.sort(
                key=lambda row: (
                    row["score"],
                    row["priority"],
                    row["tool_name"],
                )
            )
        else:
            raw_rows.sort(
                key=lambda row: (
                    row["evidence_observations"],
                    row["score"],
                    row["priority"],
                    row["tool_name"],
                )
            )

        subtask_config = self.profile.get("subtasks", {}).get(subtask, {})
        top_k = self._positive_int(
            subtask_config.get("max_tools", self.default_top_k),
            f"{subtask}.max_tools",
        )
        effective_top_k = min(top_k, len(raw_rows)) if enough_evidence else len(raw_rows)

        selected_rows = raw_rows[:effective_top_k]
        exploration_name: Optional[str] = None
        excluded_rows = raw_rows[effective_top_k:]
        if (
            excluded_rows
            and self.exploration_rate > 0
            and self.random.random() < self.exploration_rate
        ):
            exploration_row = self.random.choice(excluded_rows)
            selected_rows[-1] = exploration_row
            exploration_name = exploration_row["tool_name"]

        selected_names = {row["tool_name"] for row in selected_rows}
        selected_tools = [row["tool"] for row in selected_rows]
        diagnostics = [
            RankedTool(
                tool_name=row["tool_name"],
                score=round(row["score"], 6),
                posterior_failure_rate=round(row["failure_rate"], 6),
                mean_duration_seconds=row["duration"],
                peak_vram_mb=row["peak_vram"],
                quality_observations=row["quality_observations"],
                execution_failures=row["execution_failures"],
                evidence_observations=row["evidence_observations"],
                priority=row["priority"],
                selected=row["tool_name"] in selected_names,
                exploration_pick=row["tool_name"] == exploration_name,
            ).as_dict()
            for row in raw_rows
        ]
        return selected_tools, diagnostics

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{name} must be positive.")
        return parsed

    @staticmethod
    def _nonnegative_int(value: Any, name: str) -> int:
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"{name} must be non-negative.")
        return parsed

    @staticmethod
    def _nonnegative_float(value: Any, name: str) -> float:
        parsed = float(value)
        if parsed < 0:
            raise ValueError(f"{name} must be non-negative.")
        return parsed

    @classmethod
    def _probability(cls, value: Any, name: str) -> float:
        parsed = cls._nonnegative_float(value, name)
        if parsed > 1:
            raise ValueError(f"{name} must be between 0 and 1.")
        return parsed

    @staticmethod
    def _optional_nonnegative_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        parsed = float(value)
        if parsed < 0:
            raise ValueError("Cost hints must be non-negative.")
        return parsed
