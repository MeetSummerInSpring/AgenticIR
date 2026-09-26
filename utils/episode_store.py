from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional


EPISODE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ToolStatistics:
    tool_name: str
    attempts: int
    execution_failures: int
    quality_observations: int
    quality_successes: int
    mean_duration_seconds: Optional[float]

    @property
    def evidence_observations(self) -> int:
        """Outcomes that provide evidence about whether the tool is usable."""

        return self.quality_observations + self.execution_failures

    @property
    def posterior_failure_rate(self) -> float:
        """Beta(1, 1) posterior mean for execution or quality failure."""

        quality_failures = self.quality_observations - self.quality_successes
        observed_failures = quality_failures + self.execution_failures
        return (observed_failures + 1.0) / (self.evidence_observations + 2.0)


class EpisodeStore:
    """Append-only SQLite storage for restoration experience events."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS episode_events (
                    sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    action_json TEXT NOT NULL,
                    transition_json TEXT NOT NULL,
                    outcome_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_episode_events_run
                ON episode_events(run_id, sequence_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_episode_events_type
                ON episode_events(event_type, sequence_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_episode_events_tool_attempt
                ON episode_events(
                    json_extract(action_json, '$.subtask'),
                    json_extract(action_json, '$.tool')
                ) WHERE event_type = 'tool_attempt'
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS episode_events_no_update
                BEFORE UPDATE ON episode_events
                BEGIN
                    SELECT RAISE(ABORT, 'episode_events is append-only');
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS episode_events_no_delete
                BEFORE DELETE ON episode_events
                BEGIN
                    SELECT RAISE(ABORT, 'episode_events is append-only');
                END
                """
            )
            connection.execute(f"PRAGMA user_version = {EPISODE_SCHEMA_VERSION}")

    def append_event(
        self,
        *,
        run_id: str,
        event_type: str,
        state: Optional[Mapping[str, Any]] = None,
        action: Optional[Mapping[str, Any]] = None,
        transition: Optional[Mapping[str, Any]] = None,
        outcome: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> str:
        if not run_id:
            raise ValueError("run_id must not be empty.")
        if not event_type:
            raise ValueError("event_type must not be empty.")

        event_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        values = (
            event_id,
            run_id,
            created_at,
            event_type,
            self._dumps(state),
            self._dumps(action),
            self._dumps(transition),
            self._dumps(outcome),
            self._dumps(provenance),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO episode_events (
                    event_id, run_id, created_at, event_type,
                    state_json, action_json, transition_json,
                    outcome_json, provenance_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        return event_id

    def iter_events(
        self,
        *,
        run_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> Iterator[dict[str, Any]]:
        clauses: list[str] = []
        params: list[str] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            "SELECT sequence_id, event_id, run_id, created_at, event_type, "
            "state_json, action_json, transition_json, outcome_json, "
            f"provenance_json FROM episode_events{where} ORDER BY sequence_id"
        )
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        for row in rows:
            yield {
                "sequence_id": row["sequence_id"],
                "event_id": row["event_id"],
                "run_id": row["run_id"],
                "created_at": row["created_at"],
                "event_type": row["event_type"],
                "state": json.loads(row["state_json"]),
                "action": json.loads(row["action_json"]),
                "transition": json.loads(row["transition_json"]),
                "outcome": json.loads(row["outcome_json"]),
                "provenance": json.loads(row["provenance_json"]),
            }

    def get_tool_statistics(self, subtask: str) -> dict[str, ToolStatistics]:
        query = """
            SELECT
                json_extract(action_json, '$.tool') AS tool_name,
                COUNT(*) AS attempts,
                SUM(CASE WHEN json_extract(outcome_json, '$.status') = 'execution_error' THEN 1 ELSE 0 END) AS execution_failures,
                SUM(CASE WHEN json_type(outcome_json, '$.quality_success') IN ('true', 'false') THEN 1 ELSE 0 END) AS quality_observations,
                SUM(CASE WHEN json_type(outcome_json, '$.quality_success') = 'true' THEN 1 ELSE 0 END) AS quality_successes,
                AVG(CASE
                    WHEN json_type(transition_json, '$.duration_seconds') IN ('integer', 'real')
                         AND json_extract(transition_json, '$.duration_seconds') >= 0
                    THEN json_extract(transition_json, '$.duration_seconds')
                END) AS mean_duration_seconds
            FROM episode_events
            WHERE event_type = 'tool_attempt'
              AND json_extract(action_json, '$.subtask') = ?
              AND json_type(action_json, '$.tool') = 'text'
            GROUP BY tool_name
        """
        with self._connect() as connection:
            rows = connection.execute(query, (subtask,)).fetchall()

        return {
            row["tool_name"]: ToolStatistics(
                tool_name=row["tool_name"],
                attempts=row["attempts"],
                execution_failures=row["execution_failures"],
                quality_observations=row["quality_observations"],
                quality_successes=row["quality_successes"],
                mean_duration_seconds=row["mean_duration_seconds"],
            )
            for row in rows
        }

    @staticmethod
    def _dumps(value: Optional[Mapping[str, Any]]) -> str:
        return json.dumps(
            dict(value or {}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
