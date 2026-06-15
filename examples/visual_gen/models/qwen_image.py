# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen-Image text-to-image and Qwen-Image-Layered generation.

Usage:
    python qwen_image.py
    python qwen_image.py --visual_gen_args ../configs/qwen-image-fp8-1gpu.yaml
    python qwen_image.py \
        --model Qwen/Qwen-Image-Layered \
        --visual_gen_args ../configs/qwen-image-layered-fp8-blockscale-edge-bf16-sage-fp8-1gpu.yaml \
        --image input.png --prompt ""
"""

import argparse
from pathlib import Path

from tensorrt_llm import VisualGen, VisualGenArgs


def _output_paths(output_path: str, num_images: int) -> str | list[str]:
    if num_images == 1:
        return output_path

    path = Path(output_path)
    return [str(path.with_name(f"{path.stem}_{index}{path.suffix}")) for index in range(num_images)]


def _layer_path(output_path: str, batch_idx: int, layer_idx: int) -> Path:
    path = Path(output_path)
    return path.with_name(f"{path.stem}_batch{batch_idx + 1}_layer{layer_idx + 1}.png")


def _composite_path(output_path: str, batch_idx: int) -> Path:
    path = Path(output_path)
    return path.with_name(f"{path.stem}_batch{batch_idx + 1}_composite.png")


def _to_uint8(tensor):
    import torch

    if tensor.dtype == torch.uint8:
        return tensor
    return tensor.clamp(0, 255).round().to(torch.uint8)


def _save_layered_output(output, output_path: str) -> list[Path]:
    from PIL import Image

    if output.error is not None:
        raise RuntimeError(f"Generation failed: {output.error}")
    if output.video is None:
        raise ValueError("Qwen-Image-Layered output did not contain a layer stack.")

    layers = output.video
    if layers.dim() == 4:
        layers = layers.unsqueeze(0)
    if layers.dim() != 5:
        raise ValueError(
            f"Expected layered output shape (B, layers, H, W, C), got {tuple(layers.shape)}"
        )

    saved: list[Path] = []
    for batch_idx in range(layers.shape[0]):
        batch = _to_uint8(layers[batch_idx].detach().cpu())
        composite = None
        if batch.shape[-1] == 4:
            height, width = int(batch.shape[1]), int(batch.shape[2])
            composite = Image.new("RGBA", (width, height), (255, 255, 255, 255))

        for layer_idx in range(batch.shape[0]):
            layer = batch[layer_idx].numpy()
            mode = "RGBA" if layer.shape[-1] == 4 else "RGB"
            image = Image.fromarray(layer, mode=mode)
            path = _layer_path(output_path, batch_idx, layer_idx)
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            saved.append(path)
            if composite is not None:
                composite = Image.alpha_composite(composite, image.convert("RGBA"))

        if composite is not None:
            path = _composite_path(output_path, batch_idx)
            composite.convert("RGB").save(path)
            saved.append(path)

    return saved


def _set_if_not_none(obj: object, field: str, value: object | None) -> None:
    if value is not None:
        setattr(obj, field, value)


def _set_dict_if_not_none(values: dict[str, object], key: str, value: object | None) -> None:
    if value is not None:
        values[key] = value


def _set_layered_extra_params(params: object, args: argparse.Namespace) -> None:
    has_extra = (
        args.layers is not None
        or args.resolution is not None
        or args.cfg_normalize
        or args.use_en_prompt
    )
    if not has_extra:
        return
    if params.extra_params is None:
        params.extra_params = {}

    _set_dict_if_not_none(params.extra_params, "layers", args.layers)
    _set_dict_if_not_none(params.extra_params, "resolution", args.resolution)
    if args.cfg_normalize:
        params.extra_params["cfg_normalize"] = True
    if args.use_en_prompt:
        params.extra_params["use_en_prompt"] = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image",
        help="Hugging Face model id or local checkpoint path.",
    )
    parser.add_argument(
        "--visual_gen_args",
        "--extra_visual_gen_options",
        dest="visual_gen_args",
        help="Optional VisualGenArgs YAML file.",
    )
    parser.add_argument(
        "--prompt",
        default="A serene mountain lake at sunrise, watercolor style, highly detailed",
        help="Text prompt for image generation.",
    )
    parser.add_argument(
        "--negative_prompt",
        help="Negative prompt for classifier-free guidance.",
    )
    parser.add_argument(
        "--image",
        help="Input image path. Required for Qwen-Image-Layered.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for the prompt.",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Output height in pixels.",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Output width in pixels.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        help="Max tokens for text encoding.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        help="Qwen-Image-Layered: number of latent output layers.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=(640, 1024),
        help="Qwen-Image-Layered: resolution bucket.",
    )
    parser.add_argument(
        "--cfg_normalize",
        action="store_true",
        help="Qwen-Image-Layered: normalize CFG prediction by conditional norm.",
    )
    parser.add_argument(
        "--use_en_prompt",
        action="store_true",
        help="Qwen-Image-Layered: use English auto-caption prompt when prompt is empty.",
    )
    parser.add_argument(
        "--output_path",
        default="qwen_image_output.png",
        help="Image output path. Multiple images append an index before the suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_images_per_prompt < 1:
        raise ValueError("--num_images_per_prompt must be >= 1")

    if "layered" in args.model.lower() and args.image is None:
        raise ValueError("--image is required for Qwen-Image-Layered")

    extra_args = VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else None
    visual_gen = VisualGen(model=args.model, args=extra_args)
    params = visual_gen.default_params
    params.num_images_per_prompt = args.num_images_per_prompt
    _set_if_not_none(params, "negative_prompt", args.negative_prompt)
    _set_if_not_none(params, "image", args.image)
    _set_if_not_none(params, "height", args.height)
    _set_if_not_none(params, "width", args.width)
    _set_if_not_none(params, "num_inference_steps", args.num_inference_steps)
    _set_if_not_none(params, "guidance_scale", args.guidance_scale)
    _set_if_not_none(params, "seed", args.seed)
    _set_if_not_none(params, "max_sequence_length", args.max_sequence_length)
    _set_layered_extra_params(params, args)

    output = visual_gen.generate(inputs=args.prompt, params=params)
    if output.video is not None and output.image is None:
        saved = _save_layered_output(output, args.output_path)
    else:
        saved = output.save(_output_paths(args.output_path, args.num_images_per_prompt))
    print(f"Saved image(s) to {saved}")


if __name__ == "__main__":
    main()
