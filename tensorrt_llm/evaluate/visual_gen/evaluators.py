# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Protocol

import click

from .config import load_yaml_mapping
from .qwen_image_bench import QwenImageBenchEvaluator, QwenImageBenchEvaluatorArgs

_FORBIDDEN_EVALUATOR_CONFIG_KEYS = ("model", "type", "backend")


def _is_qwen_image_bench_checkpoint(evaluator_model: str) -> bool:
    model_name = Path(evaluator_model).name.replace("_", "-").lower()
    return "qwen-image-bench" in model_name


def _resolve_evaluator(evaluator_model: str):
    if _is_qwen_image_bench_checkpoint(evaluator_model):
        return QwenImageBenchEvaluator, QwenImageBenchEvaluatorArgs
    raise click.BadParameter(
        "Unsupported image evaluator checkpoint. Currently supported: "
        "Qwen Image Bench via --evaluator <qwen-image-bench path>.",
        param_hint="--evaluator",
    )


def build_image_evaluator(evaluator_model: str, evaluator_options: str | None):
    config = load_yaml_mapping(evaluator_options, param_hint="--evaluator-options")
    forbidden = sorted(set(config) & set(_FORBIDDEN_EVALUATOR_CONFIG_KEYS))
    if forbidden:
        raise click.BadParameter(
            "Do not set model, type, or backend in --evaluator-options. "
            f"Use --evaluator for the checkpoint path; unsupported keys: {forbidden}",
            param_hint="--evaluator-options",
        )

    evaluator_cls, args_cls = _resolve_evaluator(evaluator_model)
    arg_names = {field.name for field in fields(args_cls)}
    unknown = sorted(set(config) - arg_names)
    if unknown:
        raise click.BadParameter(
            f"Unknown Qwen Image Bench evaluator options: {unknown}",
            param_hint="--evaluator-options",
        )

    evaluator_args = replace(args_cls(), **{key: config[key] for key in config})
    return evaluator_cls(evaluator_model, evaluator_args)


class VideoEvaluator(Protocol):
    def evaluate(self, prompt_id: str, prompt: str, video: Path) -> dict[str, float]: ...

    def close(self) -> None: ...


class LPIPSVideoEvaluator:
    """Mean framewise LPIPS with bounded device memory and strict frame pairing."""

    def __init__(self, references: dict[str, Path], net: str, device: str) -> None:
        from .lpips import make_lpips_model

        self.references = references
        self.device = device
        self.model = make_lpips_model(net, device)

    def evaluate(self, prompt_id: str, prompt: str, video: Path) -> dict[str, float]:
        from .lpips import score_video_files

        return {
            "lpips": score_video_files(
                self.model, video, self.references[prompt_id], self.device, strict_pairing=True
            )
        }

    def close(self) -> None:
        del self.model


def resolve_video_directory(value: str, base_dir: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise click.BadParameter(f"{name} must be a non-empty directory path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_dir():
        raise click.BadParameter(f"{name} is not an existing directory: {path}")
    return path.resolve()


def ordered_video_paths(directory: Path, count: int) -> list[Path]:
    """Resolve zero-padded MP4 filenames in prompt order, starting at 0000.mp4."""
    paths = [directory / f"{index:04d}.mp4" for index in range(count)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise click.BadParameter(f"Missing videos in {directory}: {', '.join(missing)}")
    return paths


def build_video_evaluator(
    name: str, config: dict, base_dir: Path, prompt_ids: list[str]
) -> VideoEvaluator:
    if name != "lpips":
        raise click.BadParameter("Supported video evaluator: lpips", param_hint="--evaluator")
    unknown = set(config) - {"reference_dir", "net", "device"}
    if unknown:
        raise click.BadParameter(f"Unknown LPIPS options: {sorted(unknown)}")
    if len(set(prompt_ids)) != len(prompt_ids):
        raise click.BadParameter("Prompt IDs must be unique", param_hint="--prompts")
    directory = resolve_video_directory(config.get("reference_dir"), base_dir, "reference_dir")
    paths = ordered_video_paths(directory, len(prompt_ids))
    net = config.get("net", "alex")
    device = config.get("device", "cpu")
    if net not in ("alex", "vgg", "squeeze"):
        raise click.BadParameter("net must be alex, vgg, or squeeze")
    if not isinstance(device, str) or not device:
        raise click.BadParameter("device must be a non-empty torch device string")
    return LPIPSVideoEvaluator(dict(zip(prompt_ids, paths, strict=True)), net, device)
