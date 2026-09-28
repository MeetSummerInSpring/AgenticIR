"""Histoformer real-image checkpoint, using the standard directory-based Tool contract."""
import os
from pathlib import Path
from ..tool import Tool


class HistoformerReal(Tool):
    def __init__(self):
        super().__init__(tool_name='histoformer_real', subtask='deraining',
                         work_dir='Histoformer', script_rel_path=Path('../../histoformer_inference.py'))
        self.weight_path = self.work_dir / 'Allweather/pretrained_models/net_g_real.pth'
        self.config_path = self.repo_root / 'executor/deraining/configs/histoformer.yml'

    def _get_cmd(self):
        command = super()._get_cmd()
        # Reuse a verified environment without upgrading the main agent runtime.
        command[command.index('-n') + 1] = os.environ.get('AGENTICIR_HISTOFORMER_ENV', 'ridcp')
        return command

    def _get_cmd_opts(self):
        return ['--input_dir', self.input_dir, '--output_dir', self.output_dir,
                '--weights', self.weight_path, '--config', self.config_path]
