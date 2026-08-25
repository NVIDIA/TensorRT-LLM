# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Sequence

import click

from .config import load_yaml_mapping, split_generator_config


class VisualGenGeneratorBackend:
    def __init__(self, model: str, args: Any | None = None) -> None:
        from tensorrt_llm.visual_gen import VisualGen

        self._visual_gen = VisualGen(model=model, args=args)

    @property
    def default_params(self) -> Any:
        return self._visual_gen.default_params

    def generate(self, inputs: Sequence[str], params: Any | None = None) -> list[Any]:
        return self._visual_gen.generate(inputs=inputs, params=params)

    def close(self) -> None:
        self._visual_gen.shutdown()

    def shutdown(self) -> None:
        self.close()


def build_visual_generator(model: str, config_path: str | None):
    from tensorrt_llm.visual_gen import VisualGenArgs

    config = load_yaml_mapping(config_path, param_hint="--visual_gen_args")
    visual_gen_args_config, generation_params_config = split_generator_config(config)
    visual_gen_args = (
        VisualGenArgs.from_dict(visual_gen_args_config) if visual_gen_args_config else None
    )
    generator = VisualGenGeneratorBackend(model=model, args=visual_gen_args)
    params = generator.default_params

    for key, value in generation_params_config.items():
        if key not in type(params).model_fields:
            raise click.BadParameter(
                f"Unknown VisualGenParams field in --visual_gen_args: {key}",
                param_hint="--visual_gen_args",
            )
        setattr(params, key, value)

    return generator, params
