# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import click

from .config import load_yaml_mapping


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


def build_video_evaluator(name: str, options: str, prompt_ids: list[str]) -> VideoEvaluator:
    if name != "lpips":
        raise click.BadParameter("Supported video evaluator: lpips", param_hint="--evaluator")
    config = load_yaml_mapping(options, param_hint="--evaluator-options")
    unknown = set(config) - {"references", "net", "device"}
    if unknown:
        raise click.BadParameter(f"Unknown LPIPS options: {sorted(unknown)}")
    references = config.get("references")
    if not isinstance(references, dict) or not references:
        raise click.BadParameter("references must map prompt IDs to baseline video paths")
    if len(set(prompt_ids)) != len(prompt_ids):
        raise click.BadParameter("Prompt IDs must be unique", param_hint="--prompts")
    resolved = {}
    for prompt_id in prompt_ids:
        value = references.get(prompt_id)
        if not isinstance(value, str) or not value:
            raise click.BadParameter(f"Missing reference video for prompt ID {prompt_id!r}")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(options).resolve().parent / path
        if not path.is_file():
            raise click.BadParameter(f"Reference video does not exist: {path}")
        resolved[prompt_id] = path
    net = config.get("net", "alex")
    device = config.get("device", "cpu")
    if net not in ("alex", "vgg", "squeeze"):
        raise click.BadParameter("net must be alex, vgg, or squeeze")
    if not isinstance(device, str) or not device:
        raise click.BadParameter("device must be a non-empty torch device string")
    return LPIPSVideoEvaluator(resolved, net, device)
