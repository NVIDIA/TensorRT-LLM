# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

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
