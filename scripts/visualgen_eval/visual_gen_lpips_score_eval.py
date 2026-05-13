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

r"""Evaluate VisualGen image/video quality with LPIPS against a JSON dataset.

The YAML config follows the same VisualGenArgs format used by trtllm-serve
--extra_visual_gen_options and examples/visual_gen/configs.

Example:
  python scripts/visualgen_eval/visual_gen_lpips_score_eval.py \\
      --model flux2-dev \\
      --config examples/visual_gen/serve/configs/flux2.yml \\
      --dataset /path/to/flux_lpips_dataset.json \\
      --output-dir /tmp/flux_lpips_outputs \\
      --output-json /tmp/flux_lpips_outputs/results.json

Example dataset:

  [
    {
      "id": "astronaut",
      "prompt": "a tiny astronaut hatching from an egg on the moon",
      "reference_image_path": "golden/flux2_astronaut.png",
      "params": {
        "height": 256,
        "width": 256,
        "num_inference_steps": 4,
        "guidance_scale": 3.5,
        "seed": 42
      }
    }
  ]

The dataset may also be a top-level object with "samples" plus optional
"params" defaults shared by all samples.

For video-only scoring, provide "reference_video_path" and
"generated_video_path"; model/config are only required when generation is
needed.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import pathlib
import tempfile
import time
from typing import Any

import yaml

try:
    import lpips
except ModuleNotFoundError:
    lpips = None
try:
    import numpy as np
except ModuleNotFoundError:
    np = None
try:
    import torch
except ModuleNotFoundError:
    torch = None
try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

MODEL_ALIASES: dict[str, tuple[str, str]] = {
    "flux1": ("FLUX.1-dev", "black-forest-labs/FLUX.1-dev"),
    "flux1-dev": ("FLUX.1-dev", "black-forest-labs/FLUX.1-dev"),
    "flux.1-dev": ("FLUX.1-dev", "black-forest-labs/FLUX.1-dev"),
    "flux2": ("FLUX.2-dev", "black-forest-labs/FLUX.2-dev"),
    "flux2-dev": ("FLUX.2-dev", "black-forest-labs/FLUX.2-dev"),
    "flux.2-dev": ("FLUX.2-dev", "black-forest-labs/FLUX.2-dev"),
}

REFERENCE_IMAGE_KEYS = (
    "reference_image_path",
    "reference_image",
    "golden_image_path",
    "golden_image",
)
GENERATED_IMAGE_KEYS = ("generated_image_path", "generated_image")
OUTPUT_IMAGE_KEYS = ("output_image_path", "output_image", "output_path")
REFERENCE_VIDEO_KEYS = (
    "reference_video_path",
    "reference_video",
    "golden_video_path",
    "golden_video",
)
GENERATED_VIDEO_KEYS = ("generated_video_path", "generated_video")
OUTPUT_VIDEO_KEYS = ("output_video_path", "output_video", "output_path")
GENERATION_PARAM_KEYS = {
    "height",
    "width",
    "num_inference_steps",
    "steps",
    "guidance_scale",
    "max_sequence_length",
    "seed",
    "num_frames",
    "frame_rate",
    "negative_prompt",
    "image",
    "mask",
    "image_cond_strength",
    "num_images_per_prompt",
    "audio_sample_rate",
    "extra_params",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or score VisualGen media and compute average LPIPS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Dataset formats:\n"
            '  [{"prompt": "...", "reference_image_path": "golden.png"}]\n'
            '  [{"reference_video_path": "golden.mp4", "generated_video_path": "out.mp4"}]\n'
            '  {"params": {...}, "samples": [{"prompt": "...", '
            '"reference_image_path": "golden.png"}]}\n'
            "\n"
            "Samples can include params/generation_params, generated_image_path or "
            "generated_video_path to rescore existing outputs, and output_*_path "
            "to save generated media."
        ),
    )
    parser.add_argument(
        "--model",
        help=(
            "Model path, Hugging Face ID, or known alias such as flux1-dev/flux2-dev. "
            "Known aliases prefer local $LLM_MODELS_ROOT checkpoints when available. "
            "Required only when generation is needed."
        ),
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        help=(
            "VisualGenArgs YAML config. This should match trtllm-serve "
            "--extra_visual_gen_options / examples/visual_gen/configs format. "
            "Required only when generation is needed."
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=pathlib.Path,
        help="JSON dataset with prompt/reference path samples or existing generated media paths.",
    )
    parser.add_argument(
        "--model-root",
        type=pathlib.Path,
        help="Optional root used to resolve known model aliases before falling back to HF IDs.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="Optional directory for generated images and default result JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=pathlib.Path,
        help="Optional path to write per-sample LPIPS results as JSON.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help=(
            "Average LPIPS pass/fail threshold. Overrides dataset lpips_threshold/threshold "
            "when present."
        ),
    )
    parser.add_argument(
        "--lpips-net",
        default=None,
        choices=("alex", "vgg", "squeeze"),
        help="LPIPS backbone. Defaults to dataset lpips_net or alex.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="LPIPS device override. Defaults to CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--video-image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help=(
            "Resize decoded video frames to WIDTH HEIGHT before LPIPS. "
            "Defaults to dataset lpips_image_size/video_image_size or 256 256."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print only the result JSON to stdout.",
    )
    return parser.parse_args()


def _load_json(path: pathlib.Path) -> Any:
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise ValueError(f"Dataset file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse dataset JSON {path}: {exc}") from exc


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except FileNotFoundError as exc:
        raise ValueError(f"Config file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse config YAML {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected top-level mapping in config {path}.")
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


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _candidate_model_roots(model_root: pathlib.Path | None) -> list[pathlib.Path]:
    roots: list[pathlib.Path] = []
    if model_root is not None:
        roots.append(model_root.expanduser())
    if os.environ.get("LLM_MODELS_ROOT"):
        roots.append(pathlib.Path(os.environ["LLM_MODELS_ROOT"]).expanduser())
    roots.extend(
        [
            pathlib.Path("/home/scratch.trt_llm_data_ci/llm-models"),
            pathlib.Path("/scratch.trt_llm_data/llm-models"),
        ]
    )

    unique_roots = []
    seen = set()
    for root in roots:
        root_str = str(root)
        if root_str not in seen:
            unique_roots.append(root)
            seen.add(root_str)
    return unique_roots


def _resolve_model(model: str, model_root: pathlib.Path | None) -> str:
    model_path = pathlib.Path(model).expanduser()
    if model_path.exists():
        return str(model_path)

    alias = MODEL_ALIASES.get(model.lower())
    if alias is None:
        return model

    local_name, hf_id = alias
    for root in _candidate_model_roots(model_root):
        candidate = root / local_name
        if candidate.exists():
            return str(candidate)
    return hf_id


def _normalize_steps_alias(params: dict[str, Any]) -> dict[str, Any]:
    params = dict(params)
    if "steps" in params:
        if "num_inference_steps" not in params:
            params["num_inference_steps"] = params["steps"]
        params.pop("steps")
    return params


def _extract_generation_params(mapping: dict[str, Any]) -> dict[str, Any]:
    params = _as_mapping(
        mapping.get("params", mapping.get("generation_params")),
        "params",
    )
    for key in GENERATION_PARAM_KEYS:
        if key in mapping and key not in params:
            params[key] = mapping[key]
    return _normalize_steps_alias(params)


def _normalize_dataset(
    loaded: Any,
    dataset_path: pathlib.Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], float | None, str]:
    if isinstance(loaded, list):
        raw_samples = loaded
        defaults: dict[str, Any] = {}
        threshold = None
        lpips_net = "alex"
    elif isinstance(loaded, dict):
        if "samples" in loaded:
            raw_samples = loaded["samples"]
        elif "data" in loaded:
            raw_samples = loaded["data"]
        elif "items" in loaded:
            raw_samples = loaded["items"]
        elif "prompts" in loaded and "reference_image_paths" in loaded:
            prompts = loaded["prompts"]
            references = loaded["reference_image_paths"]
            if not isinstance(prompts, list) or not isinstance(references, list):
                raise ValueError("'prompts' and 'reference_image_paths' must be lists.")
            if len(prompts) != len(references):
                raise ValueError("'prompts' and 'reference_image_paths' must have the same length.")
            raw_samples = [
                {"prompt": prompt, "reference_image_path": reference}
                for prompt, reference in zip(prompts, references)
            ]
        elif "prompts" in loaded and "reference_video_paths" in loaded:
            prompts = loaded["prompts"]
            references = loaded["reference_video_paths"]
            if not isinstance(prompts, list) or not isinstance(references, list):
                raise ValueError("'prompts' and 'reference_video_paths' must be lists.")
            if len(prompts) != len(references):
                raise ValueError("'prompts' and 'reference_video_paths' must have the same length.")
            raw_samples = [
                {"prompt": prompt, "reference_video_path": reference}
                for prompt, reference in zip(prompts, references)
            ]
        elif (
            _first_present(loaded, REFERENCE_IMAGE_KEYS) is not None
            or _first_present(loaded, REFERENCE_VIDEO_KEYS) is not None
        ):
            raw_samples = [loaded]
        else:
            raise ValueError(
                "Dataset object must contain samples/data/items, prompts plus "
                "reference_image_paths/reference_video_paths, or a single "
                "reference_*_path sample."
            )
        defaults = _extract_generation_params(loaded)
        threshold_value = loaded.get("lpips_threshold", loaded.get("threshold"))
        threshold = float(threshold_value) if threshold_value is not None else None
        lpips_net = str(loaded.get("lpips_net", "alex"))
    else:
        raise ValueError("Dataset must be a JSON list or object.")

    if not isinstance(raw_samples, list):
        raise ValueError("Dataset samples must be a list.")

    samples = []
    dataset_dir = dataset_path.resolve().parent
    for index, raw_sample in enumerate(raw_samples):
        if not isinstance(raw_sample, dict):
            raise ValueError(f"Dataset sample {index} must be an object.")
        sample = dict(raw_sample)
        sample_id = str(sample.get("id", sample.get("name", f"sample_{index:04d}")))

        media_type = sample.get("media_type")
        reference_image = _first_present(sample, REFERENCE_IMAGE_KEYS)
        reference_video = _first_present(sample, REFERENCE_VIDEO_KEYS)
        if media_type is not None:
            media_type = str(media_type).lower()
            if media_type not in ("image", "video"):
                raise ValueError(
                    f"Dataset sample {sample_id} has unsupported media_type: {media_type}"
                )
        elif reference_video is not None:
            media_type = "video"
        elif reference_image is not None:
            media_type = "image"
        else:
            media_type = "image"

        if media_type == "video":
            reference = reference_video
            generated = _first_present(sample, GENERATED_VIDEO_KEYS)
            output_media = _first_present(sample, OUTPUT_VIDEO_KEYS)
            media_label = "video"
        else:
            reference = reference_image
            generated = _first_present(sample, GENERATED_IMAGE_KEYS)
            output_media = _first_present(sample, OUTPUT_IMAGE_KEYS)
            media_label = "image"

        if reference is None:
            raise ValueError(
                f"Dataset sample {sample_id} is missing reference_{media_label}_path/"
                f"golden_{media_label}_path."
            )
        reference_path = _resolve_path(reference, dataset_dir)
        if not reference_path.exists():
            raise ValueError(
                f"Reference {media_label} for sample {sample_id} does not exist: {reference_path}"
            )

        generated_path = _resolve_path(generated, dataset_dir) if generated is not None else None
        if generated_path is not None and not generated_path.exists():
            raise ValueError(
                f"Generated {media_label} for sample {sample_id} does not exist: {generated_path}"
            )

        prompt = sample.get("prompt")
        if generated_path is None and not isinstance(prompt, str):
            raise ValueError(f"Dataset sample {sample_id} must provide a string prompt.")

        sample_params = defaults | _extract_generation_params(sample)
        samples.append(
            {
                "id": sample_id,
                "prompt": prompt,
                "media_type": media_type,
                "reference_path": reference_path,
                "generated_path": generated_path,
                "output_media": output_media,
                "params": sample_params,
            }
        )

    if not samples:
        raise ValueError("Dataset must contain at least one sample.")
    return samples, defaults, threshold, lpips_net


def _visual_gen_params_from_dict(default_params: Any, params: dict[str, Any]) -> Any:
    from tensorrt_llm import VisualGenParams

    known_fields = set(VisualGenParams.model_fields)
    params = _normalize_steps_alias(params)
    updates: dict[str, Any] = {}
    extra_params = {}

    existing_extra_params = getattr(default_params, "extra_params", None)
    if isinstance(existing_extra_params, dict):
        extra_params.update(existing_extra_params)

    sample_extra_params = params.pop("extra_params", None)
    if sample_extra_params is not None:
        extra_params.update(_as_mapping(sample_extra_params, "extra_params"))

    for key, value in params.items():
        if key in known_fields:
            updates[key] = value
        else:
            extra_params[key] = value

    if extra_params:
        updates["extra_params"] = extra_params
    return default_params.model_copy(update=updates, deep=True)


def _load_image(path: pathlib.Path) -> Image.Image:
    if Image is None:
        raise RuntimeError("The pillow package is required to load reference images.")
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _to_lpips_tensor(image: Any, device: str) -> torch.Tensor:
    """Convert an RGB image-like object to LPIPS input: NCHW float in [-1, 1]."""
    if torch is None:
        raise RuntimeError("The torch package is required to run LPIPS scoring.")
    if np is None:
        raise RuntimeError("The numpy package is required to prepare LPIPS image tensors.")
    if Image is not None and isinstance(image, Image.Image):
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


def _extract_image(output: Any) -> Any:
    if torch is None:
        raise RuntimeError("The torch package is required to process VisualGen image outputs.")
    image = getattr(output, "image", None)
    if image is None:
        raise ValueError(
            "VisualGen output does not contain an image. "
            "This LPIPS evaluator supports image outputs."
        )
    if isinstance(image, (list, tuple)):
        if not image:
            raise ValueError("VisualGen returned an empty image list.")
        image = image[0]
    if isinstance(image, torch.Tensor) and image.dim() == 4:
        image = image[0]
    if isinstance(image, torch.Tensor):
        return image.detach().cpu()
    return image


def _extract_video(output: Any) -> Any:
    if torch is None:
        raise RuntimeError("The torch package is required to process VisualGen video outputs.")
    video = getattr(output, "video", None)
    if video is None:
        raise ValueError(
            "VisualGen output does not contain a video. "
            "This sample is marked as video media."
        )
    if isinstance(video, (list, tuple)):
        if not video:
            raise ValueError("VisualGen returned an empty video list.")
        video = video[0]
    if isinstance(video, torch.Tensor):
        return video.detach().cpu()
    return video


def _score_images(
    lpips_model: Any,
    generated_image: Any,
    reference_image: Image.Image,
    device: str,
) -> float:
    generated_tensor = _to_lpips_tensor(generated_image, device)
    reference_tensor = _to_lpips_tensor(reference_image, device)
    if generated_tensor.shape != reference_tensor.shape:
        raise ValueError(
            "Generated image and reference image must have the same LPIPS tensor shape: "
            f"{tuple(generated_tensor.shape)} vs {tuple(reference_tensor.shape)}."
        )
    with torch.no_grad():
        return float(lpips_model(generated_tensor, reference_tensor).reshape(-1).mean().item())


def _score_video_files(
    lpips_model: Any,
    generated_video_path: pathlib.Path,
    reference_video_path: pathlib.Path,
    device: str,
    image_size: tuple[int, int],
) -> float:
    helper_path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "tests"
        / "unittest"
        / "_torch"
        / "visual_gen"
        / "lpips_video_utils.py"
    )
    if not helper_path.exists():
        raise RuntimeError(f"Video LPIPS helper does not exist: {helper_path}")

    spec = importlib.util.spec_from_file_location("lpips_video_utils", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load video LPIPS helper: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return float(
        module.average_video_file_lpips_score(
            generated_video_path,
            reference_video_path,
            lpips_model,
            device,
            image_size=image_size,
        )
    )


def _resolve_output_path(
    output_media: Any,
    output_dir: pathlib.Path | None,
    dataset_dir: pathlib.Path,
    sample_id: str,
    default_suffix: str,
) -> pathlib.Path | None:
    if output_media is None and output_dir is None:
        return None

    if output_media is None:
        output_path = pathlib.Path(f"{sample_id}{default_suffix}")
    else:
        output_path = pathlib.Path(output_media).expanduser()

    if output_path.is_absolute():
        return output_path
    if output_dir is not None:
        return output_dir / output_path
    return dataset_dir / output_path


def _save_image(image: Any, output_path: pathlib.Path) -> None:
    from tensorrt_llm.media.encoding import save_image

    if torch is None:
        raise RuntimeError("The torch package is required to save generated images.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if Image is not None and isinstance(image, Image.Image):
        if np is None:
            raise RuntimeError("The numpy package is required to save PIL images.")
        image = torch.from_numpy(np.array(image.convert("RGB")))
    elif np is not None and isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    elif not isinstance(image, torch.Tensor):
        raise TypeError(f"Unsupported image type for saving: {type(image)}")

    save_image(image.detach().cpu(), output_path)


def _save_video(
    output: Any,
    video: Any,
    output_path: pathlib.Path,
    params: dict[str, Any],
) -> pathlib.Path:
    from tensorrt_llm.media.encoding import save_video

    if torch is None:
        raise RuntimeError("The torch package is required to save generated videos.")
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"Unsupported video type for saving: {type(video)}")

    audio = getattr(output, "audio", None)
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu()
    frame_rate = getattr(output, "frame_rate", None) or params.get("frame_rate") or 24.0
    audio_sample_rate = (
        getattr(output, "audio_sample_rate", None) or params.get("audio_sample_rate") or 24000
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = save_video(
        video.detach().cpu(),
        output_path,
        audio=audio,
        frame_rate=float(frame_rate),
        audio_sample_rate=int(audio_sample_rate),
    )
    return pathlib.Path(saved_path)


def _video_lpips_options(
    loaded_dataset: Any,
    args: argparse.Namespace,
) -> tuple[int, int]:
    if isinstance(loaded_dataset, dict):
        image_size_value = loaded_dataset.get(
            "lpips_image_size", loaded_dataset.get("video_image_size")
        )
    else:
        image_size_value = None

    image_size = args.video_image_size or image_size_value or (256, 256)
    if len(image_size) != 2:
        raise ValueError("Video LPIPS image size must contain exactly [width, height].")
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("Video LPIPS image size values must be positive.")

    return (width, height)


def _make_lpips_model(net: str, device: str) -> Any:
    if lpips is None:
        raise RuntimeError("The lpips package is required to run LPIPS scoring.")
    if torch is None:
        raise RuntimeError("The torch package is required to run LPIPS scoring.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"LPIPS device is {device}, but CUDA is not available.")
    return lpips.LPIPS(net=net, verbose=False).to(device).eval()


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.config is not None:
        _load_yaml(args.config)
    loaded_dataset = _load_json(args.dataset)
    samples, _, dataset_threshold, dataset_lpips_net = _normalize_dataset(
        loaded_dataset,
        args.dataset,
    )
    video_image_size = _video_lpips_options(loaded_dataset, args)

    threshold = args.threshold if args.threshold is not None else dataset_threshold
    lpips_net = args.lpips_net or dataset_lpips_net
    lpips_device = args.device or (
        "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    )
    resolved_model = _resolve_model(args.model, args.model_root) if args.model else None
    output_dir = args.output_dir.expanduser() if args.output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    needs_generation = any(sample["generated_path"] is None for sample in samples)
    visual_gen = None
    default_params = None
    if needs_generation:
        if args.model is None:
            raise ValueError("--model is required when any sample needs generation.")
        if args.config is None:
            raise ValueError("--config is required when any sample needs generation.")
        from tensorrt_llm import VisualGen, VisualGenArgs

        extra_args = VisualGenArgs.from_yaml(args.config)
        visual_gen = VisualGen(model=resolved_model, args=extra_args)
        default_params = visual_gen.default_params

    lpips_model = _make_lpips_model(lpips_net, lpips_device)
    dataset_dir = args.dataset.resolve().parent
    sample_results = []
    temp_video_dir = tempfile.TemporaryDirectory(prefix="visual_gen_lpips_video_")
    try:
        for index, sample in enumerate(samples):
            start_time = time.time()
            media_type = sample["media_type"]
            generated_path = sample["generated_path"]
            output_path = None

            if generated_path is not None:
                generation_time_sec = None
                output_path = generated_path
                if media_type == "image":
                    generated_media = _load_image(generated_path)
                else:
                    generated_media = generated_path
            else:
                if visual_gen is None or default_params is None:
                    raise RuntimeError("Internal error: VisualGen was not initialized.")
                params = _visual_gen_params_from_dict(default_params, sample["params"])
                output = visual_gen.generate(inputs=sample["prompt"], params=params)
                generation_time_sec = time.time() - start_time
                if media_type == "image":
                    generated_media = _extract_image(output)
                    output_path = _resolve_output_path(
                        sample["output_media"],
                        output_dir,
                        dataset_dir,
                        sample["id"],
                        ".png",
                    )
                    if output_path is not None:
                        _save_image(generated_media, output_path)
                else:
                    generated_media = _extract_video(output)
                    output_path = _resolve_output_path(
                        sample["output_media"],
                        output_dir,
                        dataset_dir,
                        sample["id"],
                        ".mp4",
                    )
                    score_video_path = output_path or (
                        pathlib.Path(temp_video_dir.name) / f"{sample['id']}.mp4"
                    )
                    score_video_path = _save_video(
                        output,
                        generated_media,
                        score_video_path,
                        sample["params"],
                    )
                    if output_path is not None:
                        output_path = score_video_path
                    generated_media = score_video_path

            if media_type == "image":
                reference_media = _load_image(sample["reference_path"])
                lpips_score = _score_images(
                    lpips_model,
                    generated_media,
                    reference_media,
                    lpips_device,
                )
            else:
                lpips_score = _score_video_files(
                    lpips_model,
                    generated_media,
                    sample["reference_path"],
                    lpips_device,
                    video_image_size,
                )

            sample_results.append(
                {
                    "index": index,
                    "id": sample["id"],
                    "media_type": media_type,
                    "prompt": sample["prompt"],
                    "lpips_score": lpips_score,
                    "reference_path": str(sample["reference_path"]),
                    "generated_path": str(output_path) if output_path is not None else None,
                    "generation_time_sec": generation_time_sec,
                    "params": sample["params"],
                }
            )
    finally:
        del lpips_model
        if visual_gen is not None:
            visual_gen.shutdown()
        temp_video_dir.cleanup()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores = [sample["lpips_score"] for sample in sample_results]
    mean_score = float(sum(scores) / len(scores))
    passed = None if threshold is None else mean_score < threshold
    return {
        "model": args.model,
        "resolved_model": resolved_model,
        "config": str(args.config) if args.config is not None else None,
        "dataset": str(args.dataset),
        "lpips_net": lpips_net,
        "lpips_device": lpips_device,
        "video_image_size": list(video_image_size),
        "video_frame_selection": "all_paired_frames",
        "threshold": threshold,
        "passed": passed,
        "num_samples": len(sample_results),
        "mean_lpips_score": mean_score,
        "samples": sample_results,
    }


def main() -> None:
    args = parse_args()
    result = _evaluate(args)

    output_json = args.output_json
    if output_json is None and args.output_dir is not None:
        output_json = args.output_dir / "lpips_results.json"
    if output_json is not None:
        output_json = output_json.expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"Mean LPIPS score: {result['mean_lpips_score']:.6f}")
        print(f"Samples: {result['num_samples']}")
        print(f"Model: {result['resolved_model']}")
        print(f"Config: {result['config']}")
        print(f"Dataset: {result['dataset']}")
        if output_json is not None:
            print(f"Results JSON: {output_json}")
        if result["threshold"] is not None:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"Threshold: {result['threshold']:.6f} ({status})")

    if result["passed"] is False:
        raise RuntimeError(
            f"Mean LPIPS score {result['mean_lpips_score']:.6f} "
            f"exceeds threshold {result['threshold']:.6f}."
        )


if __name__ == "__main__":
    main()
