#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate a FLUX image and report LPIPS against a golden image.

Example config:

  model_path: /path/to/FLUX.1-dev
  visual_gen_args:
    device: cuda
    dtype: bfloat16
    pipeline: {}
  params:
    prompt: a tiny astronaut hatching from an egg on the moon
    height: 256
    width: 256
    num_inference_steps: 4
    guidance_scale: 3.5
    seed: 42
  golden_image: /path/to/golden.png
  lpips_threshold: 0.05

Run:

  python scripts/flux_lpips_score.py --config flux_lpips.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import time
from typing import Any

import lpips
import numpy as np
import torch
import yaml
from PIL import Image

PARAM_KEYS = {
    "prompt",
    "height",
    "width",
    "num_inference_steps",
    "guidance_scale",
    "max_sequence_length",
    "seed",
    "negative_prompt",
    "num_images_per_prompt",
    "extra_params",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a FLUX image and compute LPIPS against a golden image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Config keys:\n"
            "  model_path or checkpoint_path: FLUX checkpoint path / HF model id\n"
            "  visual_gen_args: VisualGenArgs fields such as dtype, pipeline, attention\n"
            "  params: FLUX forward() fields such as prompt, height, width, seed\n"
            "  golden_image: golden PNG/JPEG path\n"
            "  lpips_threshold: optional pass/fail threshold\n"
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        type=pathlib.Path,
        help="YAML or JSON config describing FLUX model args and generation params.",
    )
    parser.add_argument(
        "--golden-image",
        type=pathlib.Path,
        help="Override config golden_image path.",
    )
    parser.add_argument(
        "--generated-image",
        type=pathlib.Path,
        help=(
            "Use an existing generated image instead of running FLUX generation. "
            "Useful for rescoring saved outputs."
        ),
    )
    parser.add_argument(
        "--output-image",
        type=pathlib.Path,
        help="Optional path to save the generated image.",
    )
    parser.add_argument(
        "--output-json",
        type=pathlib.Path,
        help="Optional path to write the LPIPS result as JSON.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Override config lpips_threshold. Exit code is 2 if score is not below threshold.",
    )
    parser.add_argument(
        "--lpips-net",
        default=None,
        choices=("alex", "vgg", "squeeze"),
        help="LPIPS backbone. Defaults to config lpips_net or alex.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="LPIPS device override. Defaults to VisualGenArgs device or cuda.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Override params.prompt from the config.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override config model_path/checkpoint_path.",
    )
    parser.add_argument(
        "--no-skip-warmup",
        action="store_true",
        help="Run pipeline warmup during load. By default this script skips warmup.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print only the LPIPS result JSON to stdout.",
    )
    return parser.parse_args()


def _load_config(path: pathlib.Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            if path.suffix.lower() == ".json":
                loaded = json.load(file)
            else:
                loaded = yaml.safe_load(file)
    except FileNotFoundError as exc:
        raise ValueError(f"Config file does not exist: {path}") from exc
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Failed to parse config file {path}: {exc}") from exc

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected top-level mapping in {path}, got {type(loaded).__name__}."
        )
    return loaded


def _as_mapping(value: Any, key: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping, got {type(value).__name__}.")
    return dict(value)


def _resolve_path(path_value: str | pathlib.Path, base_dir: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _resolve_cli_path(path_value: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(path_value).expanduser()


def _get_first_present(config: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return None


def _prepare_run_config(
    raw_config: dict[str, Any],
    cli_args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pathlib.Path,
    pathlib.Path | None,
    float | None,
    str,
    str,
]:
    base_dir = cli_args.config.resolve().parent
    visual_gen_args = _as_mapping(
        raw_config.get("visual_gen_args", raw_config.get("args")),
        "visual_gen_args",
    )
    params = _as_mapping(
        raw_config.get("params", raw_config.get("generation_params")),
        "params",
    )
    for key in PARAM_KEYS:
        if key in raw_config and key not in params:
            params[key] = raw_config[key]
    if "steps" in raw_config and "num_inference_steps" not in params:
        params["num_inference_steps"] = raw_config["steps"]
    if "steps" in params:
        if "num_inference_steps" not in params:
            params["num_inference_steps"] = params["steps"]
        params.pop("steps")

    model_path = cli_args.model_path or _get_first_present(
        raw_config,
        ("model_path", "model", "checkpoint_path"),
    )
    if model_path is not None:
        visual_gen_args["checkpoint_path"] = str(model_path)
    if "checkpoint_path" not in visual_gen_args or not visual_gen_args["checkpoint_path"]:
        raise ValueError(
            "Missing FLUX checkpoint. Set model_path/checkpoint_path in config "
            "or pass --model-path."
        )

    prompt = cli_args.prompt or raw_config.get("prompt")
    if prompt is not None:
        params["prompt"] = prompt
    if "prompt" not in params or params["prompt"] is None:
        raise ValueError("Missing generation prompt. Set params.prompt or pass --prompt.")

    golden_image = cli_args.golden_image or raw_config.get("golden_image", raw_config.get("image"))
    if golden_image is None:
        raise ValueError("Missing golden image. Set golden_image in config or pass --golden-image.")
    golden_image_path = (
        _resolve_cli_path(cli_args.golden_image)
        if cli_args.golden_image is not None
        else _resolve_path(golden_image, base_dir)
    )
    if not golden_image_path.exists():
        raise ValueError(f"Golden image does not exist: {golden_image_path}")

    generated_image = cli_args.generated_image or raw_config.get("generated_image")
    if cli_args.generated_image is not None:
        generated_image_path = _resolve_cli_path(cli_args.generated_image)
    else:
        generated_image_path = (
            _resolve_path(generated_image, base_dir) if generated_image is not None else None
        )
    if generated_image_path is not None and not generated_image_path.exists():
        raise ValueError(f"Generated image does not exist: {generated_image_path}")

    threshold = cli_args.threshold
    if threshold is None:
        threshold = raw_config.get("lpips_threshold", raw_config.get("threshold"))
    if threshold is not None:
        threshold = float(threshold)

    lpips_net = cli_args.lpips_net or raw_config.get("lpips_net", "alex")
    lpips_device = cli_args.device or raw_config.get("lpips_device") or visual_gen_args.get(
        "device", "cuda"
    )
    return (
        visual_gen_args,
        params,
        golden_image_path,
        generated_image_path,
        threshold,
        str(lpips_net),
        str(lpips_device),
    )


def _to_lpips_tensor(image: Any, device: str) -> torch.Tensor:
    """Convert an RGB image-like object to LPIPS input: NCHW float in [-1, 1]."""
    if isinstance(image, Image.Image):
        tensor = torch.from_numpy(np.array(image.convert("RGB")))
    elif isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image)
    elif isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
    else:
        raise TypeError(f"Unsupported image type for LPIPS: {type(image)}")

    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D image tensor, got shape {tuple(tensor.shape)}")

    if tensor.shape[0] == 3 and tensor.shape[-1] != 3:
        tensor = tensor.permute(1, 2, 0)
    if tensor.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {tuple(tensor.shape)}")

    tensor = tensor.to(device=device, dtype=torch.float32)
    if tensor.max() > 2.0:
        tensor = tensor / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor * 2.0 - 1.0


def _extract_first_image(result: Any) -> Any:
    image = getattr(result, "image", result)
    if isinstance(image, (list, tuple)):
        if not image:
            raise ValueError("Generation returned an empty image list.")
        image = image[0]
    if isinstance(image, torch.Tensor) and image.dim() == 4:
        image = image[0]
    if isinstance(image, torch.Tensor):
        return image.detach().cpu()
    return image


def _generate_flux_image(
    visual_gen_args: dict[str, Any],
    params: dict[str, Any],
    *,
    skip_warmup: bool,
) -> Any:
    from tensorrt_llm._torch.visual_gen.config import VisualGenArgs
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader

    args = VisualGenArgs(**visual_gen_args)
    pipeline = PipelineLoader(args).load(skip_warmup=skip_warmup)
    try:
        with torch.no_grad():
            result = pipeline.forward(**params)
        return _extract_first_image(result)
    finally:
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_image(path: pathlib.Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _save_image(image: Any, path: pathlib.Path) -> None:
    from tensorrt_llm.serve.media_storage import MediaStorage

    path.parent.mkdir(parents=True, exist_ok=True)
    MediaStorage.save_image(image, str(path))


def _compute_lpips_score(
    generated_image: Any,
    golden_image: Image.Image,
    *,
    device: str,
    net: str,
) -> float:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"LPIPS device is {device}, but CUDA is not available.")

    lpips_model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    try:
        generated_tensor = _to_lpips_tensor(generated_image, device)
        golden_tensor = _to_lpips_tensor(golden_image, device)
        if generated_tensor.shape != golden_tensor.shape:
            raise ValueError(
                "Generated image and golden image must have the same LPIPS tensor shape: "
                f"{tuple(generated_tensor.shape)} vs {tuple(golden_tensor.shape)}."
            )

        with torch.no_grad():
            return float(lpips_model(generated_tensor, golden_tensor).item())
    finally:
        del lpips_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    cli_args = parse_args()
    raw_config = _load_config(cli_args.config)
    (
        visual_gen_args,
        params,
        golden_image_path,
        generated_image_path,
        threshold,
        lpips_net,
        lpips_device,
    ) = _prepare_run_config(raw_config, cli_args)

    if generated_image_path is None:
        start_time = time.time()
        generated_image = _generate_flux_image(
            visual_gen_args,
            params,
            skip_warmup=not cli_args.no_skip_warmup,
        )
        generation_time_sec = time.time() - start_time
    else:
        generated_image = _load_image(generated_image_path)
        generation_time_sec = None

    output_image = cli_args.output_image or raw_config.get(
        "output_image", raw_config.get("output_path")
    )
    output_image_path = None
    if output_image is not None and generated_image_path is None:
        output_image_path = (
            _resolve_cli_path(cli_args.output_image)
            if cli_args.output_image is not None
            else _resolve_path(output_image, cli_args.config.resolve().parent)
        )
        _save_image(generated_image, output_image_path)
    elif generated_image_path is not None:
        output_image_path = generated_image_path

    golden_image = _load_image(golden_image_path)
    score = _compute_lpips_score(
        generated_image,
        golden_image,
        device=lpips_device,
        net=lpips_net,
    )
    passed = None if threshold is None else score < threshold

    result = {
        "lpips_score": score,
        "lpips_net": lpips_net,
        "lpips_device": lpips_device,
        "threshold": threshold,
        "passed": passed,
        "golden_image": str(golden_image_path),
        "generated_image": str(output_image_path) if output_image_path is not None else None,
        "generation_time_sec": generation_time_sec,
        "checkpoint_path": visual_gen_args["checkpoint_path"],
        "prompt": params["prompt"],
    }

    if cli_args.output_json is not None:
        output_json_path = _resolve_cli_path(cli_args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if cli_args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"LPIPS score: {score:.6f}")
        print(f"Golden image: {golden_image_path}")
        if output_image_path is not None:
            print(f"Generated image: {output_image_path}")
        if generation_time_sec is not None:
            print(f"Generation time: {generation_time_sec:.2f}s")
        if threshold is not None:
            status = "PASS" if passed else "FAIL"
            print(f"Threshold: {threshold:.6f} ({status})")

    if passed is False:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
