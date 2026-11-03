# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import click


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
