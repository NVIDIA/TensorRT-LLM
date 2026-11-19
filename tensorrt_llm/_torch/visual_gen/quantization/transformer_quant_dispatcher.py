# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed Transformer execution paths selected outside compiled model forwards."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass
from typing import Iterator

from torch import nn

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..config import DiffusionModelConfig, create_attention_metadata_state

QuantIgnoreKey = tuple[bool, bool]


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
    def keys(self) -> tuple[QuantIgnoreKey, ...]:
        states = {self.key(step) for step in self.linear | self.attention}
        return ((False, False), *sorted(states - {(False, False)}))

    def key(self, step: int | None) -> QuantIgnoreKey:
        return step in self.linear, step in self.attention

    def for_stage(self, step_offset: int) -> "TransformerQuantSchedule":
        """Translate configured exclusions into a stage's local step indices."""
        if not step_offset:
            return self
        return TransformerQuantSchedule(
            frozenset(step - step_offset for step in self.linear if step >= step_offset),
            frozenset(step - step_offset for step in self.attention if step >= step_offset),
        )


def fixed_quant_config(config: DiffusionModelConfig, key: QuantIgnoreKey) -> DiffusionModelConfig:
    """Derive a fixed path from an already resolved component configuration."""
    linear_ignored, attention_ignored = key
    quant = config.quant_config.model_copy(deep=True)
    if hasattr(quant, "ignore_steps"):
        quant.ignore_steps = []
    if linear_ignored:
        quant = QuantConfig()
    attention = config.attention.model_copy(deep=True)
    if attention_ignored:
        attention.quant_attention_config = None
    elif attention.quant_attention_config is not None:
        attention.quant_attention_config.ignore_steps = []
    return config.model_copy(
        update={
            "quant_config": quant,
            "quant_config_dict": None if linear_ignored else config.quant_config_dict,
            "dynamic_weight_quant": False if linear_ignored else config.dynamic_weight_quant,
            "force_dynamic_quantization": False
            if linear_ignored
            else config.force_dynamic_quantization,
            "attention": attention,
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
        self.stage = "base"
        self.step: int | None = None
        self.forced_key: QuantIgnoreKey | None = None

    @staticmethod
    def variant_name(stage: str, key: QuantIgnoreKey) -> str:
        return f"{stage}_linear{int(key[0])}_attention{int(key[1])}"

    def add(self, key: QuantIgnoreKey, model: nn.Module, stage: str = "base") -> None:
        name = self.variant_name(stage, key)
        if name in self.variants:
            raise ValueError(f"Transformer path {name!r} is already registered")
        self.variants[name] = model

    def models(self, stage: str | None = None) -> Iterator[tuple[str, nn.Module]]:
        for name, model in self.variants.items():
            if stage is None or name.startswith(stage + "_"):
                yield name, model

    def select(self, step: int | None = None) -> nn.Module:
        self.step = step
        key = self.forced_key if self.forced_key is not None else self.schedule.key(step)
        return self.variants[self.variant_name(self.stage, key)]

    def forward(self, *args, **kwargs):
        return self.select(self.step)(*args, **kwargs)


@dataclass
class TransformerWeightReuse:
    """Loading-only context; no model forward depends on this context."""

    source: nn.Module | None
    target: nn.Module
    require_dynamic_weights: bool
    reused_modules: frozenset[int] = frozenset()


_weight_reuse: ContextVar[dict[str | None, TransformerWeightReuse]] = ContextVar(
    "visual_gen_transformer_weight_reuse", default={}
)


def current_weight_reuse(component: str | None) -> TransformerWeightReuse | None:
    return _weight_reuse.get().get(component)


def _bind_linear(source: Linear, target: Linear) -> Linear:
    """Reuse final weights and load-derived state with independent module children."""
    if type(source) is not type(target) or (
        source.in_features,
        source.out_features,
        source.quant_config,
        source.tp_mode,
    ) != (
        target.in_features,
        target.out_features,
        target.quant_config,
        target.tp_mode,
    ):
        raise ValueError("Cannot share Linear weights across incompatible execution paths")
    bound = copy(source)
    bound._parameters = source._parameters.copy()
    bound._buffers = source._buffers.copy()
    bound._non_persistent_buffers_set = source._non_persistent_buffers_set.copy()
    bound._modules = target._modules.copy()
    bound.quant_config = target.quant_config
    bound.quant_method = copy(source.quant_method)
    bound.weights_loading_config = target.weights_loading_config
    bound.rebuild_tensor_metadata = {
        name: {**metadata, "param": getattr(bound, name)}
        for name, metadata in source.rebuild_tensor_metadata.items()
    }
    bound.nvfp4_allowed_backends = source.nvfp4_allowed_backends.copy()
    bound._locality_domain_runtime = copy(source._locality_domain_runtime)
    bound._locality_domain_weight_shards = (
        [shard.copy() for shard in source._locality_domain_weight_shards]
        if source._locality_domain_weight_shards is not None
        else None
    )
    if hasattr(source, "fp8_low_m_gemm"):
        bound.fp8_low_m_gemm = copy(source.fp8_low_m_gemm)
    bound._output_slice_cache = None
    for name in target.__dict__:
        if "hook" in name:
            bound.__dict__[name] = target.__dict__[name]
    return bound


@contextmanager
def reuse_transformer_weights(
    source: nn.Module | None, target: nn.Module, *, require_dynamic_weights: bool
) -> Iterator[TransformerWeightReuse]:
    """Bind completed Linear weights before running the target's native loader."""
    reused: set[int] = set()
    if source is target:
        reused.update(id(module) for module in target.modules() if isinstance(module, Linear))
    elif source is not None:
        source_modules = dict(source.named_modules())
        for name, module in list(target.named_modules()):
            if isinstance(module, Linear):
                original = source_modules.get(name)
                if not isinstance(original, Linear):
                    raise ValueError(f"Transformer path has no matching Linear {name!r}")
                parent_name, _, child_name = name.rpartition(".")
                parent = target.get_submodule(parent_name) if parent_name else target
                bound = _bind_linear(original, module)
                parent.add_module(child_name, bound)
                reused.add(id(bound))
    reuse = TransformerWeightReuse(source, target, require_dynamic_weights, frozenset(reused))
    token = _weight_reuse.set({**_weight_reuse.get(), target.model_config.component_name: reuse})
    try:
        yield reuse
    finally:
        _weight_reuse.reset(token)


def share_non_linear_parameters(source: nn.Module, target: nn.Module) -> None:
    """Share unchanged trained parameters after each model's native loading hooks."""
    originals = dict(source.named_modules())
    for name, module in target.named_modules():
        if isinstance(module, Linear):
            continue
        original = originals.get(name)
        if original is None or type(original) is not type(module):
            continue
        for parameter_name, parameter in module._parameters.items():
            shared = original._parameters.get(parameter_name)
            if (
                parameter is not None
                and shared is not None
                and parameter.shape == shared.shape
                and parameter.dtype == shared.dtype
            ):
                module.register_parameter(parameter_name, shared)


def bind_model_weights(
    source: nn.Module, target: nn.Module, *, share_parameters: bool = True
) -> None:
    """Reuse loaded tensors, optionally with independent Parameter bindings."""
    with reuse_transformer_weights(source, target, require_dynamic_weights=False):
        originals = dict(source.named_modules())
        for name, module in target.named_modules():
            if isinstance(module, Linear):
                continue
            original = originals.get(name)
            if original is None or type(original) is not type(module):
                raise ValueError(f"Stage model has no matching module {name!r}")
            for parameter_name, parameter in original._parameters.items():
                module.register_parameter(parameter_name, parameter)
            for buffer_name, buffer in original._buffers.items():
                module.register_buffer(
                    buffer_name,
                    buffer,
                    persistent=buffer_name not in original._non_persistent_buffers_set,
                )
    if not share_parameters:
        parameters = {}
        for module in target.modules():
            for name, parameter in list(module._parameters.items()):
                if parameter is None:
                    continue
                if id(parameter) not in parameters:
                    parameters[id(parameter)] = nn.Parameter(
                        parameter.detach(), requires_grad=parameter.requires_grad
                    )
                module.register_parameter(name, parameters[id(parameter)])
                metadata = getattr(module, "rebuild_tensor_metadata", {})
                if name in metadata:
                    metadata[name]["param"] = getattr(module, name)
