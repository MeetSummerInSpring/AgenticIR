from pathlib import Path

from ..tool import Tool


class DarkIRTool(Tool):
    """AgenticIR wrapper for a DarkIR checkpoint/configuration pair."""

    def __init__(
        self,
        *,
        tool_name: str,
        config_name: str,
        weight_name: str,
    ) -> None:
        super().__init__(
            tool_name=tool_name,
            subtask="brightening",
            work_dir="DarkIR",
            script_rel_path=Path("../../darkir_inference.py"),
        )
        self.config_path = self.work_dir / "options" / "inference" / config_name
        self.weight_path = self.work_dir / "models" / weight_name

    def _get_cmd_opts(self) -> list[Path | str]:
        return [
            "--input_dir",
            self.input_dir,
            "--output_dir",
            self.output_dir,
            "--config",
            self.config_path,
            "--weights",
            self.weight_path,
        ]


class DarkIRMultiTask(DarkIRTool):
    """Width-32 DarkIR trained on multiple low-light restoration datasets."""

    def __init__(self) -> None:
        super().__init__(
            tool_name="darkir_mt",
            config_name="real_lsrw.yml",
            weight_name="DarkIR_1k_cr_mt.pt",
        )


class DarkIRLarge(DarkIRTool):
    """Width-64 DarkIR trained for low-light deblurring."""

    def __init__(self) -> None:
        super().__init__(
            tool_name="darkir_l",
            config_name="LOLBlur.yml",
            weight_name="DarkIR_64width.pt",
        )
