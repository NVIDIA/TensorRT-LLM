# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import yaml

GENERATION_PARAMS_CONFIG_KEYS = (
    "generation_params",
    "visual_gen_params",
    "extra_visual_gen_options",
)


def load_yaml_mapping(path: str | None, *, param_hint: str) -> dict[str, Any]:
    if path is None:
        return {}

    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise click.BadParameter(
            f"{param_hint} must contain a YAML mapping at the document root",
            param_hint=param_hint,
        )
    return dict(config)


def load_prompt_records(
    prompts_path: str,
    num_samples: int | None = None,
) -> tuple[list[str], list[str]]:
    path = Path(prompts_path)
    if not path.exists():
        raise click.BadParameter(f"Prompt file does not exist: {path}", param_hint="--prompts")
    if num_samples is not None and num_samples < 1:
        raise click.BadParameter("--num-samples must be >= 1", param_hint="--num-samples")

    if path.suffix == ".json":
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, list):
            raise click.BadParameter(
                "JSON prompt files must contain a list", param_hint="--prompts"
            )
        items = parsed
    else:
        items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                items.append(json.loads(line))
            else:
                items.append(line)

    prompt_ids: list[str] = []
    prompts: list[str] = []
    for index, item in enumerate(items):
        if isinstance(item, dict):
            prompt = item.get("prompt")
            prompt_id = item.get("id", str(index))
        else:
            prompt = item
            prompt_id = str(index)
        if not isinstance(prompt, str) or not prompt:
            raise click.BadParameter(
                "Every prompt entry must contain a non-empty string prompt",
                param_hint="--prompts",
            )
        prompt_ids.append(str(prompt_id))
        prompts.append(prompt)
        if num_samples is not None and len(prompts) >= num_samples:
            break

    if not prompts:
        raise click.BadParameter("Prompt file did not contain any prompts", param_hint="--prompts")
    return prompt_ids, prompts


def split_generator_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    visual_gen_args_config = dict(config)
    generation_params_config: dict[str, Any] = {}

    for key in GENERATION_PARAMS_CONFIG_KEYS:
        value = visual_gen_args_config.pop(key, None)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise click.BadParameter(f"{key} must be a mapping", param_hint="--visual_gen_args")
        generation_params_config.update(value)

    if "backend" in visual_gen_args_config:
        raise click.BadParameter(
            "Do not set a generator backend in --visual_gen_args. For "
            "image_generation_eval, root --model and --visual_gen_args configure "
            "VisualGen directly.",
            param_hint="--visual_gen_args",
        )

    return visual_gen_args_config, generation_params_config


_load_yaml_mapping = load_yaml_mapping
_load_prompt_records = load_prompt_records
_split_generator_config = split_generator_config
