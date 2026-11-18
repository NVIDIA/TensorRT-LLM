# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep request-dependent M3 attention outside captured prefill segments."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from weakref import ReferenceType, ref

import torch
from torch import nn

from tensorrt_llm._torch.utils import get_model_extra_attrs


@dataclass
class _PrefillCore:
    core: ReferenceType[nn.Module]
    producer: Callable[..., tuple[torch.Tensor, torch.Tensor] | None]
    attention: Callable[..., None]


_CORES: dict[str, _PrefillCore] = {}


def _forget_core(key: str, reference: ReferenceType[nn.Module]) -> None:
    _CORES.pop(key, None)


def register_prefill_core(
    core: nn.Module,
    producer: Callable[..., tuple[torch.Tensor, torch.Tensor] | None],
    attention: Callable[..., None],
) -> str:
    key = f"m3_prefill_{id(core)}"
    _CORES[key] = _PrefillCore(ref(core, partial(_forget_core, key)), producer, attention)
    return key


@torch.library.custom_op("trtllm::modeling_v2_m3_prefill_attention", mutates_args=("output",))
def prefill_attention(
    packed: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    q: torch.Tensor | None,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    idx_q: torch.Tensor | None,
    output: torch.Tensor,
    core_key: str,
    layer: int,
) -> None:
    entry = _CORES[core_key]
    core = entry.core()
    if core is None:
        raise RuntimeError("MiniMax-M3 prefill core is no longer available")
    metadata = get_model_extra_attrs()["attention_metadata"]()
    live = int(metadata.num_tokens)
    if packed is not None:
        horizontal = entry.producer(
            core,
            layer,
            packed[:live],
            position_ids[..., :live] if position_ids is not None else None,
            metadata.kv_cache_manager,
            metadata.msa_idx_k_cache,
            metadata.msa_out_cache_loc,
        )
        if horizontal is None:
            raise RuntimeError(
                f"MiniMax-M3 sparse layer {layer}: no compatible horizontal producer"
            )
        q, idx_q = horizontal
        k = v = None
    entry.attention(core, layer, q, k, v, idx_q, metadata, output)


@prefill_attention.register_fake
def _prefill_attention_fake(
    packed: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    q: torch.Tensor | None,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    idx_q: torch.Tensor | None,
    output: torch.Tensor,
    core_key: str,
    layer: int,
) -> None:
    pass
