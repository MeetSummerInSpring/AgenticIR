from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2


DEFAULT_TOOL_TIMEOUT_SECONDS = 1800.0


@dataclass(frozen=True)
class ToolRunResult:
    """Metadata produced by one successful tool invocation."""

    tool_name: str
    subtask: str
    duration_seconds: Optional[float]
    output_path: Path
    command: Optional[tuple[str, ...]] = None


class ToolExecutionError(RuntimeError):
    """Raised when a restoration tool cannot be executed or validated."""

    def __init__(
        self,
        tool_name: str,
        subtask: str,
        stage: str,
        message: str,
        *,
        command: Optional[Sequence[str]] = None,
        returncode: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        stdout_log: Optional[Path] = None,
        stderr_log: Optional[Path] = None,
    ) -> None:
        self.tool_name = tool_name
        self.subtask = subtask
        self.stage = stage
        self.command = tuple(command) if command is not None else None
        self.returncode = returncode
        self.duration_seconds = duration_seconds
        self.stdout_log = stdout_log
        self.stderr_log = stderr_log

        details = [f"Tool {tool_name!r} failed during {stage}: {message}"]
        if returncode is not None:
            details.append(f"return code: {returncode}")
        if self.command is not None:
            details.append(f"command: {subprocess.list2cmdline(list(self.command))}")
        if stderr_log is not None:
            details.append(f"stderr log: {stderr_log}")
        super().__init__("; ".join(details))


class Tool:
    """Base wrapper for a restoration tool.

    Third-party tools run through conda run using an absolute Conda
    executable. This avoids depending on shell activation inside the child
    process. Child stdout/stderr are written to per-tool log files beside
    the output directory so model output does not flood the agent terminal. Set AGENTICIR_CONDA_EXE
    when Conda is installed in a non-standard location and
    AGENTICIR_TOOL_TIMEOUT_SECONDS to override the default timeout.
    """

    def __init__(
        self,
        tool_name: str,
        subtask: str,
        work_dir: Optional[Path] = None,
        script_rel_path: Optional[Path | str] = None,
    ) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.tool_name = tool_name
        self.subtask = subtask
        self.work_dir: Optional[Path] = None
        self.script_path: Optional[Path] = None
        if work_dir is not None:
            if script_rel_path is None:
                raise ValueError(
                    "script_rel_path is required when work_dir is provided."
                )
            self.work_dir = self.repo_root / "executor" / subtask / "tools" / work_dir
            self.script_path = self.work_dir / script_rel_path

        self.input_dir: Path
        self.output_dir: Path
        self._last_command: Optional[tuple[str, ...]] = None

    def __call__(
        self,
        input_dir: Path,
        output_dir: Path,
        silent: bool = False,
        *args,
    ) -> ToolRunResult:
        """Execute the tool and return validated invocation metadata."""

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self._last_command = None

        if not silent:
            print("-" * 100)
            print(f"Subtask\t: {self.subtask}")
            print(f"Tool\t: {self.tool_name}")

        start_time = time.monotonic()
        stage = "precheck"
        try:
            input_path = self._precheck()
            if not silent:
                print(f"Input\t: {input_path}")

            stage = "execution"
            self._invoke(*args)

            stage = "postcheck"
            output_path = self._postcheck()
        except ToolExecutionError as exc:
            if exc.duration_seconds is None:
                exc.duration_seconds = time.monotonic() - start_time
            raise
        except Exception as exc:
            raise ToolExecutionError(
                self.tool_name,
                self.subtask,
                stage,
                f"{type(exc).__name__}: {exc}",
                command=self._last_command,
                duration_seconds=time.monotonic() - start_time,
            ) from exc

        duration = time.monotonic() - start_time
        if not silent:
            print(f"Output\t: {output_path}")
            print(f"Time\t: {duration:.3f}s")

        return ToolRunResult(
            tool_name=self.tool_name,
            subtask=self.subtask,
            duration_seconds=duration,
            output_path=output_path,
            command=self._last_command,
        )

    def _precheck(self) -> Path:
        """Validate input/output directories and return the input image path."""

        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
        if not self.output_dir.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {self.output_dir}")

        inputs = list(self.input_dir.iterdir())
        if len(inputs) != 1 or not inputs[0].is_file():
            raise ValueError(
                f"Input directory must contain exactly one file: {self.input_dir}"
            )
        if any(self.output_dir.iterdir()):
            raise ValueError(f"Output directory must be empty: {self.output_dir}")

        self._validate_image(inputs[0], role="input")
        return inputs[0]

    def _postcheck(self) -> Path:
        """Normalize the output name and validate that it is a decodable image."""

        outputs = list(self.output_dir.iterdir())
        if len(outputs) != 1 or not outputs[0].is_file():
            raise ValueError(
                "Tool output directory must contain exactly one file, "
                f"found {len(outputs)} in {self.output_dir}."
            )

        output_path = outputs[0]
        normalized_path = self.output_dir / "output.png"
        if output_path.name != normalized_path.name:
            output_path.replace(normalized_path)
            output_path = normalized_path

        self._validate_image(output_path, role="output")
        return output_path

    @staticmethod
    def _validate_image(path: Path, role: str) -> None:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise ValueError(f"{role.capitalize()} is not a decodable image: {path}")
        if image.ndim not in {2, 3}:
            raise ValueError(
                f"Unexpected {role} image rank {image.ndim} for {path}."
            )
        if image.ndim == 3 and image.shape[2] not in {1, 3, 4}:
            raise ValueError(
                f"Unexpected {role} channel count {image.shape[2]} for {path}."
            )

    def _invoke(self) -> None:
        self._preprocess()
        command = self._get_cmd()
        self._last_command = tuple(command)
        self._run_command(command)
        self._postprocess()

    def _get_cmd(self) -> list[str]:
        if self.work_dir is None or self.script_path is None:
            raise RuntimeError(
                f"Tool {self.tool_name!r} does not define an external script."
            )
        if not self.work_dir.is_dir():
            raise FileNotFoundError(f"Tool directory does not exist: {self.work_dir}")
        if not self.script_path.is_file():
            raise FileNotFoundError(f"Tool script does not exist: {self.script_path}")

        conda_exe = self._resolve_conda_executable()
        env_name = self.tool_name.split("_")[0]
        opts = [str(opt) for opt in self._get_cmd_opts()]
        return [
            str(conda_exe),
            "run",
            "--no-capture-output",
            "-n",
            env_name,
            "python",
            str(self.script_path),
            *opts,
        ]

    def _run_command(self, command: Sequence[str]) -> None:
        timeout = self._tool_timeout_seconds()

        stdout_log = self.output_dir.parent / f"{self.tool_name}.stdout.log"
        stderr_log = self.output_dir.parent / f"{self.tool_name}.stderr.log"
        try:
            with stdout_log.open("wb") as stdout, stderr_log.open("wb") as stderr:
                subprocess.run(
                    list(command),
                    cwd=self.work_dir,
                    check=True,
                    timeout=timeout,
                    stdout=stdout,
                    stderr=stderr,
                )
        except subprocess.CalledProcessError as exc:
            raise ToolExecutionError(
                self.tool_name,
                self.subtask,
                "execution",
                "external process returned a non-zero status",
                command=command,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                returncode=exc.returncode,
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ToolExecutionError(
                self.tool_name,
                self.subtask,
                "execution",
                f"timed out after {timeout:.1f} seconds",
                command=command,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(
                self.tool_name,
                self.subtask,
                "execution",
                f"could not start process: {exc}",
                command=command,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
            ) from exc

    @staticmethod
    def _resolve_conda_executable() -> Path:
        candidates: list[Path] = []
        for env_name in ("AGENTICIR_CONDA_EXE", "CONDA_EXE"):
            if value := os.environ.get(env_name):
                candidates.append(Path(value).expanduser())

        if discovered := shutil.which("conda"):
            candidates.append(Path(discovered))

        python_path = Path(sys.executable).resolve()
        candidates.extend(
            [
                python_path.parent / "conda",
                Path("/root/miniconda3/bin/conda"),
                Path("/opt/conda/bin/conda"),
            ]
        )
        for parent in python_path.parents:
            candidates.append(parent / "bin" / "conda")

        checked: list[str] = []
        for candidate in candidates:
            candidate = candidate.expanduser().resolve()
            candidate_str = str(candidate)
            if candidate_str in checked:
                continue
            checked.append(candidate_str)
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate

        raise FileNotFoundError(
            "Could not find a Conda executable. Set AGENTICIR_CONDA_EXE to "
            f"an absolute path. Checked: {checked}"
        )

    @staticmethod
    def _tool_timeout_seconds() -> float:
        raw_value = os.environ.get("AGENTICIR_TOOL_TIMEOUT_SECONDS")
        if raw_value is None:
            return DEFAULT_TOOL_TIMEOUT_SECONDS
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                "AGENTICIR_TOOL_TIMEOUT_SECONDS must be a positive number."
            ) from exc
        if value <= 0:
            raise ValueError(
                "AGENTICIR_TOOL_TIMEOUT_SECONDS must be a positive number."
            )
        return value

    def _get_cmd_opts(self, *args) -> list[str]:
        raise NotImplementedError

    def _preprocess(self) -> None:
        """Hook for a tool-specific input transformation."""

    def _postprocess(self) -> None:
        """Hook for a tool-specific output normalization."""
