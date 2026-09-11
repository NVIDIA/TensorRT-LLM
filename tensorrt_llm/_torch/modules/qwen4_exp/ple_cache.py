# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle-coupled PLE cache state for Qwen4-Exp."""

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterable

import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    MambaAuxCacheExtension,
    MambaCacheBuildContext,
    register_mamba_aux_cache_estimator,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import get_pp_layers
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, DataRole

if TYPE_CHECKING:
    import transformers


PLE_CACHE_EXTENSION_KEY = "qwen4_exp_ple"
PLE_NGRAM_CONTEXT = DataRole("ple_ngram_context")
PLE_CONV_STATE = DataRole("ple_conv_state")


@dataclass(frozen=True, kw_only=True)
class Qwen4ExpPLECacheParams:
    """Shapes and dtypes for PLE recurrent-state pools."""

    ple_layer_mask: tuple[bool, ...]
    num_ple_layers: int
    short_conv_channels: int
    short_conv_state_len: int
    ngram_context_len: int
    eos_token_id: int
    conv_state_dtype: torch.dtype


def get_qwen4_exp_ple_layer_mask(config: "transformers.PretrainedConfig") -> list[bool]:
    """Return the decoder-layer mask for the one-based PLE layer IDs."""
    ple_layer_ids = list(getattr(config, "ple_layer_ids", None) or [])
    invalid_ids = [
        layer_id
        for layer_id in ple_layer_ids
        if not isinstance(layer_id, int)
        or isinstance(layer_id, bool)
        or not 1 <= layer_id <= config.num_hidden_layers
    ]
    if invalid_ids:
        raise ValueError(
            "ple_layer_ids must contain one-based decoder-layer IDs in "
            f"[1, {config.num_hidden_layers}], got {invalid_ids}"
        )
    if len(ple_layer_ids) != len(set(ple_layer_ids)):
        raise ValueError("ple_layer_ids must not contain duplicate layer IDs")
    if len(ple_layer_ids) > 1:
        raise ValueError(
            "Qwen4-Exp currently supports at most one PLE decoder layer, "
            f"got ple_layer_ids={ple_layer_ids}"
        )
    ple_layer_id_set = set(ple_layer_ids)
    return [(layer_id + 1) in ple_layer_id_set for layer_id in range(config.num_hidden_layers)]


def extract_qwen4_exp_ple_cache_params(
    config: "transformers.PretrainedConfig",
) -> Qwen4ExpPLECacheParams:
    """Derive PLE recurrent-state pool dimensions from the model config."""
    from tensorrt_llm._torch.pyexecutor.config_utils import resolve_hf_torch_dtype

    ple_layer_mask = get_qwen4_exp_ple_layer_mask(config)
    hc_count = getattr(config, "hc_count", 1) or 1
    return Qwen4ExpPLECacheParams(
        ple_layer_mask=tuple(ple_layer_mask),
        num_ple_layers=sum(ple_layer_mask),
        short_conv_channels=hc_count * config.hidden_size,
        short_conv_state_len=(config.ple_conv_kernel_size - 1) * config.ngram_size,
        ngram_context_len=config.ngram_size - 1,
        eos_token_id=int(getattr(config, "eos_token_id", 0) or 0),
        conv_state_dtype=resolve_hf_torch_dtype(config) or torch.bfloat16,
    )


def build_qwen4_exp_ple_cache_extension(
    config: "transformers.PretrainedConfig",
    *,
    total_layers: int,
    is_draft: bool,
) -> "Qwen4ExpPLECacheExtension | None":
    """Build an extension aligned with a target-plus-draft cache layout."""
    if is_draft:
        return None
    params = extract_qwen4_exp_ple_cache_params(config)
    num_target_layers = len(params.ple_layer_mask)
    if num_target_layers > total_layers:
        raise ValueError(
            "PLE layer mask cannot exceed the hybrid cache layout: "
            f"got {num_target_layers}, expected at most {total_layers}"
        )
    if num_target_layers < total_layers:
        params = replace(
            params,
            ple_layer_mask=params.ple_layer_mask + (False,) * (total_layers - num_target_layers),
        )
    return Qwen4ExpPLECacheExtension(params)


class Qwen4ExpPLECacheExtension(MambaAuxCacheExtension):
    """PLE conv/ngram state sharing each GDN slot lifecycle."""

    def __init__(self, params: Qwen4ExpPLECacheParams) -> None:
        self.params = params
        self._layer_ids = tuple(
            layer_id for layer_id, active in enumerate(params.ple_layer_mask) if active
        )
        self._conv_state_shape = (
            params.short_conv_channels,
            params.short_conv_state_len,
        )
        self._ngram_context_shape = (params.ngram_context_len,)
        self._conv_states: dict[int, torch.Tensor] = {}
        self._ngram_contexts: dict[int, torch.Tensor] = {}

    @property
    def key(self) -> str:
        return PLE_CACHE_EXTENSION_KEY

    @property
    def data_roles(self) -> tuple[DataRole, ...]:
        return PLE_CONV_STATE, PLE_NGRAM_CONTEXT

    def validate(self, context: MambaCacheBuildContext) -> None:
        if len(self.params.ple_layer_mask) != len(context.layer_mask):
            raise ValueError(
                "PLE layer mask length must match the hybrid layer mask: "
                f"got {len(self.params.ple_layer_mask)}, expected {len(context.layer_mask)}"
            )
        if len(self._layer_ids) != self.params.num_ple_layers:
            raise ValueError(
                "PLE layer mask count does not match num_ple_layers: "
                f"got {len(self._layer_ids)}, expected {self.params.num_ple_layers}"
            )
        if any(not context.layer_mask[layer_id] for layer_id in self._layer_ids):
            raise ValueError("PLE lifecycle state must belong to recurrent layers")
        if (
            self.params.short_conv_channels <= 0
            or self.params.short_conv_state_len <= 0
            or self.params.ngram_context_len <= 0
        ):
            raise ValueError("PLE recurrent-state dimensions must be positive")

    def buffer_configs(
        self, context: MambaCacheBuildContext, layer_id: int
    ) -> tuple[BufferConfig, ...]:
        del context
        if layer_id not in self._layer_ids:
            return ()
        return (
            BufferConfig(
                role=PLE_CONV_STATE,
                size=math.prod(self._conv_state_shape) * self.params.conv_state_dtype.itemsize,
            ),
            BufferConfig(
                role=PLE_NGRAM_CONTEXT,
                size=math.prod(self._ngram_context_shape) * torch.int64.itemsize,
            ),
        )

    def bind(self, context: MambaCacheBuildContext, manager: object) -> None:
        if context.slot_capacity is None:
            raise RuntimeError("PLE cache binding requires the resolved slot capacity")
        get_state_buffer = getattr(manager, "_get_state_buffer")
        layer_offsets = getattr(manager, "layer_offsets")
        for layer_id in self._layer_ids:
            local_layer_idx = layer_offsets.get(layer_id)
            if local_layer_idx is None:
                continue
            conv_state = get_state_buffer(
                local_layer_idx,
                PLE_CONV_STATE,
                self.params.conv_state_dtype,
                list(self._conv_state_shape),
            )
            ngram_context = get_state_buffer(
                local_layer_idx,
                PLE_NGRAM_CONTEXT,
                torch.long,
                list(self._ngram_context_shape),
            )
            if (
                conv_state.shape[0] != context.slot_capacity
                or ngram_context.shape[0] != context.slot_capacity
            ):
                raise RuntimeError(
                    "PLE and recurrent lifecycle buffers must have the same number of slots: "
                    f"layer={layer_id}, recurrent={context.slot_capacity}, "
                    f"conv={conv_state.shape[0]}, ngram={ngram_context.shape[0]}"
                )
            self._conv_states[layer_id] = conv_state
            self._ngram_contexts[layer_id] = ngram_context

    def get_layer_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        conv = self._conv_states.get(layer_idx)
        ngram = self._ngram_contexts.get(layer_idx)
        if conv is None or ngram is None:
            return None
        return conv, ngram

    def reset_slots(self, slots: torch.Tensor, host_slots: list[int]) -> None:
        del host_slots
        for conv_state in self._conv_states.values():
            conv_state[slots] = 0
        for ngram_context in self._ngram_contexts.values():
            ngram_context[slots] = self.params.eos_token_id

    def iter_buffers(self) -> Iterable[torch.Tensor]:
        yield from self._ngram_contexts.values()
        yield from self._conv_states.values()

    def shutdown(self) -> None:
        self._ngram_contexts.clear()
        self._conv_states.clear()


def estimate_qwen4_exp_ple_state_bytes_per_rank(
    model_config: object,
    params: object,
    mapping: object,
    *,
    spec_config: object | None,
    use_separate_draft_kv_cache: bool,
) -> int:
    """Return this rank's PLE lifecycle bytes per recurrent slot."""
    pretrained_config = getattr(model_config, "pretrained_config", None)
    if pretrained_config is None:
        return 0
    from tensorrt_llm._torch.pyexecutor.config_utils import is_qwen4_exp

    if not is_qwen4_exp(pretrained_config):
        return 0
    ple_params = extract_qwen4_exp_ple_cache_params(pretrained_config)
    if not any(ple_params.ple_layer_mask):
        return 0
    mamba_layer_mask, full_attention_layer_mask = params.get_layer_masks(
        is_draft=False,
        use_separate_draft_kv_cache=use_separate_draft_kv_cache,
    )
    combined_layer_mask = [
        is_mamba or is_attention
        for is_mamba, is_attention in zip(mamba_layer_mask, full_attention_layer_mask)
    ]
    local_layer_indices, _ = get_pp_layers(
        sum(combined_layer_mask),
        mapping,
        spec_config=spec_config,
        layer_mask=combined_layer_mask,
    )
    local_ple_layers = sum(
        layer_id < len(ple_params.ple_layer_mask) and ple_params.ple_layer_mask[layer_id]
        for layer_id in local_layer_indices
    )
    bytes_per_layer = (
        ple_params.short_conv_channels
        * ple_params.short_conv_state_len
        * ple_params.conv_state_dtype.itemsize
        + ple_params.ngram_context_len * torch.int64.itemsize
    )
    return local_ple_layers * bytes_per_layer


register_mamba_aux_cache_estimator(
    PLE_CACHE_EXTENSION_KEY,
    estimate_qwen4_exp_ple_state_bytes_per_rank,
)


__all__ = [
    "PLE_CACHE_EXTENSION_KEY",
    "PLE_CONV_STATE",
    "PLE_NGRAM_CONTEXT",
    "Qwen4ExpPLECacheExtension",
    "Qwen4ExpPLECacheParams",
    "build_qwen4_exp_ple_cache_extension",
    "estimate_qwen4_exp_ple_state_bytes_per_rank",
    "extract_qwen4_exp_ple_cache_params",
    "get_qwen4_exp_ple_layer_mask",
]
