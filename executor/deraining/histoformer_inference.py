"""Local inference adapter for the pinned author Histoformer network.

Same RGB/[0,1], float32, reflection padding to 8, strict weights and unpadding
as the author test script. No resizing, tiling, augmentation or online downloads.
Import only the architecture to avoid unrelated BasicSR training dependencies.
"""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml


def restore(model, array):
    height, width = array.shape[:2]
    tensor = torch.from_numpy(array.astype(np.float32) / 255).permute(2, 0, 1)[None].cuda()
    padding = (0, (-width) % 8, 0, (-height) % 8)
    if any(padding):
        tensor = F.pad(tensor, padding, 'reflect' if min(height, width) > 7 else 'replicate')
    with torch.inference_mode():
        result = model(tensor)[..., :height, :width].clamp(0, 1)
    if not torch.isfinite(result).all():
        raise ValueError('nonfinite restoration output')
    return np.rint(result[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    for name in ['input_dir', 'output_dir', 'weights', 'config']:
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    vendor = Path(__file__).resolve().parent / 'tools/Histoformer'
    source = vendor / 'basicsr/models/archs/histoformer_arch.py'
    spec = importlib.util.spec_from_file_location('author_histoformer_arch', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(928); torch.cuda.manual_seed_all(928)
    torch.backends.cudnn.benchmark = False
    model = module.Histoformer(**yaml.safe_load(args.config.read_text()))
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['params'], strict=True)
    model = model.eval().cuda()
    paths = list(args.input_dir.iterdir())
    if len(paths) != 1 or any(args.output_dir.iterdir()):
        raise ValueError('expected one input and an empty output directory')
    array = np.asarray(Image.open(paths[0]).convert('RGB'))
    Image.fromarray(restore(model, array)).save(args.output_dir / 'output.png')


if __name__ == '__main__':
    main()
