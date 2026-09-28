"""Author UDR-S2Former configurations under the standard directory Tool contract."""
import os
from pathlib import Path
from ..tool import Tool


class UDRS2Former(Tool):
    def __init__(self, checkpoint='raindrop_real'):
        if checkpoint not in ('raindrop_real','agan'):
            raise ValueError('unsupported UDR checkpoint')
        super().__init__(tool_name='udr_'+checkpoint,subtask='deraining',
                         work_dir='UDR-S2Former',script_rel_path=Path('../../udr_inference.py'))
        self.weight_path=self.work_dir/'pretrained'/('udrs2former_'+checkpoint+'.pth')

    def _get_cmd(self):
        cmd=super()._get_cmd()
        cmd[cmd.index('-n')+1]=os.environ.get('AGENTICIR_UDR_ENV','ridcp')
        return cmd

    def _get_cmd_opts(self):
        return ['--input_dir',self.input_dir,'--output_dir',self.output_dir,'--weights',self.weight_path]
