from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as functional
import yaml
from PIL import Image
from torchvision.transforms import functional as vision_functional

sys.path.insert(0, str(Path.cwd()))

from archs.DarkIR import DarkIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgenticIR DarkIR inference")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    return parser.parse_args()


def build_model(network_options: dict, weights_path: Path) -> DarkIR:
    model = DarkIR(
        img_channel=network_options["img_channels"],
        width=network_options["width"],
        middle_blk_num_enc=network_options["middle_blk_num_enc"],
        middle_blk_num_dec=network_options["middle_blk_num_dec"],
        enc_blk_nums=network_options["enc_blk_nums"],
        dec_blk_nums=network_options["dec_blk_nums"],
        dilations=network_options["dilations"],
        extra_depth_wise=network_options["extra_depth_wise"],
    )
    checkpoint = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["params"], strict=True)
    return model


def pad_to_multiple(image: torch.Tensor, multiple: int = 8) -> torch.Tensor:
    height, width = image.shape[-2:]
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    return functional.pad(image, (0, pad_width, 0, pad_height), value=0)


def main() -> None:
    args = parse_args()
    input_paths = [path for path in args.input_dir.iterdir() if path.is_file()]
    if len(input_paths) != 1:
        raise ValueError(
            f"Expected exactly one input image in {args.input_dir}, "
            f"found {len(input_paths)}."
        )

    with args.config.open("r", encoding="utf-8") as config_file:
        options = yaml.safe_load(config_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(options["network"], args.weights).to(device).eval()

    image = Image.open(input_paths[0]).convert("RGB")
    input_tensor = vision_functional.to_tensor(image).unsqueeze(0).to(device)
    height, width = input_tensor.shape[-2:]
    padded_input = pad_to_multiple(input_tensor)

    with torch.inference_mode():
        output = model(padded_input)
        output = output[:, :, :height, :width].clamp(0.0, 1.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_image = vision_functional.to_pil_image(output.squeeze(0).cpu())
    output_image.save(args.output_dir / "output.png")


if __name__ == "__main__":
    main()
