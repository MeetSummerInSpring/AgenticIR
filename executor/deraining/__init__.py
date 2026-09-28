import os

from ..tool import Tool
from ..multitask_tools import *
from .histoformer import HistoformerReal
from .udr import UDRS2Former


__all__ = ['deraining_toolbox']


subtask = 'deraining'
deraining_toolbox = [
    MAXIM(subtask=subtask),
    XRestormer(subtask=subtask),
    Restormer(subtask=subtask),
    MPRNet(subtask=subtask),
    HistoformerReal(),
    UDRS2Former('raindrop_real'),
    UDRS2Former('agan'),
]
