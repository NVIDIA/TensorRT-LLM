# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed Transformer execution paths selected outside compiled model forwards."""

from dataclasses import dataclass
from typing import Iterator

from torch import nn

from tensorrt_llm.models.modeling_utils import QuantConfig

from ..config import DiffusionModelConfig, create_attention_metadata_state


@dataclass(frozen=True)
class TransformerQuantSchedule:
    """Component-local Linear and attention exclusions, independent of model state."""

    linear: frozenset[int] = frozenset()
    attention: frozenset[int] = frozenset()

    @classmethod
    def from_config(cls, config: DiffusionModelConfig) -> "TransformerQuantSchedule":
        recipe = config.attention.quant_attention_config
        return cls(
            frozenset(getattr(config.quant_config, "ignore_steps", ())),
            frozenset(recipe.ignore_steps if recipe is not None else ()),
        )

    @property
    def warmup_keys(self) -> tuple[tuple[bool, bool], ...]:
        states = {self.key(step) for step in self.linear | self.attention}
        return ((False, False), *sorted(states - {(False, False)}))

    def key(self, step: int | None) -> tuple[bool, bool]:
        return step in self.linear, step in self.attention

    def for_stage(self, step_offset: int) -> "TransformerQuantSchedule":
        """Translate configured exclusions into a stage's local step indices."""
        if not step_offset:
            return self
        return TransformerQuantSchedule(
            frozenset(step - step_offset for step in self.linear if step >= step_offset),
            frozenset(step - step_offset for step in self.attention if step >= step_offset),
        )


def fixed_quant_config(
    config: DiffusionModelConfig, linear_quant_ignored: bool
) -> DiffusionModelConfig:
    """Fix Linear precision while retaining attention's per-step schedule.

    The quantized configuration keeps its Linear schedule for checkpoint
    validation in the loader. Selection happens outside the model forward.
    """
    quant = QuantConfig() if linear_quant_ignored else config.quant_config.model_copy(deep=True)
    return config.model_copy(
        update={
            "quant_config": quant,
            "quant_config_dict": None if linear_quant_ignored else config.quant_config_dict,
            "dynamic_weight_quant": False if linear_quant_ignored else config.dynamic_weight_quant,
            "force_dynamic_quantization": False
            if linear_quant_ignored
            else config.force_dynamic_quantization,
            "attention": config.attention.model_copy(deep=True),
            "attention_metadata_state": create_attention_metadata_state(),
        }
    )


class TransformerQuantDispatcher(nn.Module):
    """Own fixed model instances and select an instance before model execution.

    Pipeline component attributes refer to the selected real model. Lifecycle
    operations explicitly visit these registered models instead of forwarding
    arbitrary attributes to a default model.
    """

    def __init__(self, schedule: TransformerQuantSchedule) -> None:
        super().__init__()
        self.schedule = schedule
        self.variants = nn.ModuleDict()

    @staticmethod
    def variant_name(linear_quant_ignored: bool) -> str:
        return "bf16" if linear_quant_ignored else "quantized"

    def add(self, linear_quant_ignored: bool, model: nn.Module) -> None:
        name = self.variant_name(linear_quant_ignored)
        if name in self.variants:
            raise ValueError(f"Transformer path {name!r} is already registered")
        self.variants[name] = model

    def models(self) -> Iterator[tuple[str, nn.Module]]:
        yield from self.variants.items()

    def select(self, step: int | None, *, force_ignore: bool | None = None) -> nn.Module:
        ignored = step in self.schedule.linear if force_ignore is None else force_ignore
        return self.variants[self.variant_name(ignored)]
