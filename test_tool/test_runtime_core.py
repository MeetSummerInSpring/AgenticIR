from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from executor.brightening.darkir import DarkIRLarge, DarkIRMultiTask
from executor.tool import Tool, ToolExecutionError
from utils.episode_store import EpisodeStore
from utils.schedule_memory import ScheduleMemory
from utils.tool_selector import ToolSelector


class _CopyTool(Tool):
    def _invoke(self) -> None:
        image = cv2.imread(str(next(self.input_dir.iterdir())))
        if not cv2.imwrite(str(self.output_dir / "result.jpg"), image):
            raise RuntimeError("Could not write test output.")


class _FailingProcessTool(Tool):
    def _invoke(self) -> None:
        command = [
            sys.executable,
            "-c",
            "import sys; print('child stdout'); "
            "print('child stderr', file=sys.stderr); raise SystemExit(7)",
        ]
        self._last_command = tuple(command)
        self.work_dir = self.output_dir.parent
        self._run_command(command)


class _NamedTool:
    def __init__(self, name: str) -> None:
        self.tool_name = name


class ToolRunnerTests(unittest.TestCase):
    def test_validates_and_normalizes_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            image = np.full((12, 16, 3), 127, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(input_dir / "input.png"), image))

            result = _CopyTool("copy", "test")(input_dir, output_dir, silent=True)

            self.assertEqual(result.output_path, output_dir / "output.png")
            self.assertTrue(result.output_path.is_file())
            self.assertGreaterEqual(result.duration_seconds, 0.0)

    def test_subprocess_failure_preserves_exit_code_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "tool" / "output"
            input_dir.mkdir()
            output_dir.mkdir(parents=True)
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(input_dir / "input.png"), image))

            with self.assertRaises(ToolExecutionError) as raised:
                _FailingProcessTool("failure", "test")(
                    input_dir, output_dir, silent=False
                )

            error = raised.exception
            self.assertEqual(error.returncode, 7)
            self.assertIsNotNone(error.duration_seconds)
            self.assertIn("child stdout", error.stdout_log.read_text())
            self.assertIn("child stderr", error.stderr_log.read_text())

    def test_conda_resolution_returns_absolute_executable(self) -> None:
        conda_executable = Tool._resolve_conda_executable()
        self.assertTrue(conda_executable.is_absolute())
        self.assertTrue(conda_executable.is_file())


class DarkIRWrapperTests(unittest.TestCase):
    def test_variants_use_darkir_environment_and_matching_assets(self) -> None:
        expected = {
            "darkir_mt": ("real_lsrw.yml", "DarkIR_1k_cr_mt.pt"),
            "darkir_l": ("LOLBlur.yml", "DarkIR_64width.pt"),
        }
        for tool in (DarkIRMultiTask(), DarkIRLarge()):
            tool.input_dir = Path("/tmp/darkir-test-input")
            tool.output_dir = Path("/tmp/darkir-test-output")
            command = tool._get_cmd()

            self.assertEqual(command[3:5], ["-n", "darkir"])
            self.assertEqual(command[5], "python")
            self.assertTrue(Path(command[6]).is_file())

            config_name, weight_name = expected[tool.tool_name]
            self.assertTrue(any(item.endswith(config_name) for item in command))
            self.assertTrue(any(item.endswith(weight_name) for item in command))


class MemoryAndSelectionTests(unittest.TestCase):
    @staticmethod
    def _append_attempt(
        store: EpisodeStore,
        *,
        tool: str,
        success: bool,
        duration: float,
    ) -> None:
        store.append_event(
            run_id="selection-test",
            event_type="tool_attempt",
            action={"subtask": "demo", "tool": tool},
            transition={"duration_seconds": duration},
            outcome={"status": "ok", "quality_success": success},
        )

    def test_episode_store_is_append_only_and_aggregates_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "episodes.sqlite3"
            store = EpisodeStore(database_path)
            self._append_attempt(store, tool="a", success=True, duration=2.0)
            self._append_attempt(store, tool="a", success=False, duration=4.0)
            store.append_event(
                run_id="selection-test",
                event_type="tool_attempt",
                action={"subtask": "demo", "tool": "a"},
                transition={"duration_seconds": 1.0},
                outcome={"status": "execution_error", "quality_success": None},
            )
            store.append_event(
                run_id="selection-test",
                event_type="tool_attempt",
                action={"subtask": "demo", "tool": "cached"},
                transition={"duration_seconds": None},
                outcome={"status": "ok", "quality_success": True},
            )
            store.append_event(
                run_id="selection-test",
                event_type="tool_attempt",
                action={"subtask": "other", "tool": "a"},
                transition={"duration_seconds": 100.0},
                outcome={"status": "ok", "quality_success": False},
            )

            statistics_by_tool = store.get_tool_statistics("demo")
            statistics = statistics_by_tool["a"]
            self.assertEqual(statistics.quality_observations, 2)
            self.assertEqual(statistics.quality_successes, 1)
            self.assertEqual(statistics.execution_failures, 1)
            self.assertEqual(statistics.evidence_observations, 3)
            self.assertEqual(statistics.mean_duration_seconds, 7 / 3)
            self.assertEqual(statistics.posterior_failure_rate, 0.6)
            cached_statistics = statistics_by_tool["cached"]
            self.assertIsNone(cached_statistics.mean_duration_seconds)

            with sqlite3.connect(database_path) as connection:
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        "DELETE FROM episode_events WHERE sequence_id = 1"
                    )

    def test_selector_waits_for_evidence_then_applies_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            profile_path = root / "profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "defaults": {
                            "max_tools": 2,
                            "minimum_observations_for_pruning": 2,
                            "quality_weight": 0.8,
                            "latency_weight": 0.2,
                            "vram_weight": 0.0,
                            "exploration_rate": 0.0,
                        },
                        "subtasks": {
                            "demo": {
                                "tools": {
                                    "a": {"priority": 0},
                                    "b": {"priority": 1},
                                    "c": {"priority": 2},
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            store = EpisodeStore(root / "episodes.sqlite3")
            selector = ToolSelector(
                profile_path=profile_path,
                episode_store=store,
                seed=11,
            )
            tools = [_NamedTool("c"), _NamedTool("a"), _NamedTool("b")]

            cold_start, cold_diagnostics = selector.select("demo", tools)
            self.assertEqual([tool.tool_name for tool in cold_start], ["a", "b", "c"])
            self.assertTrue(all(row["selected"] for row in cold_diagnostics))

            self._append_attempt(store, tool="a", success=True, duration=1.0)
            calibrating, _ = selector.select("demo", tools)
            self.assertEqual(
                [tool.tool_name for tool in calibrating],
                ["b", "c", "a"],
            )

            for _ in range(2):
                self._append_attempt(store, tool="a", success=True, duration=1.0)
                self._append_attempt(store, tool="b", success=True, duration=8.0)
                self._append_attempt(store, tool="c", success=False, duration=1.0)

            selected, diagnostics = selector.select("demo", tools)
            self.assertEqual([tool.tool_name for tool in selected], ["a", "b"])
            self.assertEqual(sum(row["selected"] for row in diagnostics), 2)

    def test_schedule_memory_retrieves_exact_evidence_without_rewriting_it(self) -> None:
        memory = ScheduleMemory(Path("memory/schedule_rules.json"))
        rules = memory.query(
            ["dark", "noise"],
            ["brightening", "denoising"],
        )

        self.assertEqual(rules[0]["rule_id"], "order:dark+noise")
        candidates = rules[0]["candidates"]
        brightening_first = next(
            candidate
            for candidate in candidates
            if candidate["order"] == ["brightening", "denoising"]
        )
        self.assertAlmostEqual(
            brightening_first["failure_rates"]["noise"],
            150 / 360,
        )
        rendered, rule_ids = memory.render_for_prompt(
            ["dark", "noise"],
            ["brightening", "denoising"],
        )
        self.assertEqual(rule_ids[0], "order:dark+noise")
        self.assertIn('"matched_rules"', rendered)
        self.assertNotIn("42%", rendered)

        generated_render, _ = memory.render_for_prompt(
            iter(["dark", "noise"]),
            ["brightening", "denoising"],
        )
        self.assertEqual(
            json.loads(generated_render)["query"]["degradations"],
            ["dark", "noise"],
        )


if __name__ == "__main__":
    unittest.main()
