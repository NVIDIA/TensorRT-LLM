# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom ops that keep request- and chunk-dependent M3 prefill work out of the compiled graph."""

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
    indexer: Callable[..., torch.Tensor]
    routed_experts: Callable[..., torch.Tensor]


_CORES: dict[str, _PrefillCore] = {}


def _forget_core(key: str, reference: ReferenceType[nn.Module]) -> None:
    _CORES.pop(key, None)


def register_prefill_core(
    core: nn.Module,
    producer: Callable[..., tuple[torch.Tensor, torch.Tensor] | None],
    attention: Callable[..., None],
    indexer: Callable[..., torch.Tensor],
    routed_experts: Callable[..., torch.Tensor],
) -> str:
    """Register ``core``'s prefill callbacks and return the key the ops take.

    The registry holds ``core`` weakly, so releasing the model drops its entry.
    """
    key = f"m3_prefill_{id(core)}"
    _CORES[key] = _PrefillCore(
        ref(core, partial(_forget_core, key)), producer, attention, indexer, routed_experts
    )
    return key


def _live_core(core_key: str) -> tuple[_PrefillCore, nn.Module]:
    entry = _CORES[core_key]
    core = entry.core()
    if core is None:
        raise RuntimeError("MiniMax-M3 prefill core is no longer available")
    return entry, core


def run_prefill_attention(
    packed: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    q: torch.Tensor | None,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    idx_qk: torch.Tensor | None,
    output: torch.Tensor,
    core_key: str,
    layer: int,
) -> None:
    """Run one prefill layer's attention over the live rows of the current attention metadata.

    A sparse layer with the five-way projection passes ``packed``: the horizontal producer writes K, V and index-K
    to the paged caches and returns Q and index-Q. An unfused sparse layer passes ``idx_qk``, whose index-K is
    written before attention. A dense layer passes only ``q``, ``k`` and ``v``. Pad rows of ``output`` are left as
    allocated.

    Breakable prefill graphs call this function directly at every replay. Compiled prefill calls it through
    ``prefill_attention``, a piecewise boundary op.
    """
    entry, core = _live_core(core_key)
    metadata = get_model_extra_attrs()["attention_metadata"]()
    live = int(metadata.num_tokens)
    if position_ids is not None:
        position_ids = position_ids[..., :live]
    idx_q = None
    if packed is not None:
        horizontal = entry.producer(
            core,
            layer,
            packed[:live],
            position_ids,
            metadata.kv_cache_manager,
            metadata.msa_idx_k_cache,
            metadata.msa_out_cache_loc,
        )
        if horizontal is None:
            raise RuntimeError(
                f"MiniMax-M3 sparse layer {layer}: the horizontal producer requires an FP8 or "
                "NVFP4 HND pool with an FP8 index-K cache"
            )
        q, idx_q = horizontal
        k = v = None
    elif idx_qk is not None:
        # Insert index-K where attention reads it, using this step's live cache slots.
        idx_q = entry.indexer(
            core,
            layer,
            idx_qk[:live],
            position_ids,
            metadata.msa_idx_k_cache,
            metadata.msa_out_cache_loc,
        )
    entry.attention(core, layer, q, k, v, idx_q, metadata, output)


@torch.library.custom_op("trtllm::modeling_v2_m3_prefill_attention", mutates_args=("output",))
def prefill_attention(
    packed: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    q: torch.Tensor | None,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    idx_qk: torch.Tensor | None,
    output: torch.Tensor,
    core_key: str,
    layer: int,
) -> None:
    """``run_prefill_attention`` as an op the piecewise optimizer splits on (it is not captured)."""
    run_prefill_attention(packed, position_ids, q, k, v, idx_qk, output, core_key, layer)


@prefill_attention.register_fake
def _prefill_attention_fake(
    packed: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    q: torch.Tensor | None,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    idx_qk: torch.Tensor | None,
    output: torch.Tensor,
    core_key: str,
    layer: int,
) -> None:
    pass


@torch.library.custom_op("trtllm::modeling_v2_m3_prefill_routed_experts", mutates_args=())
def prefill_routed_experts(
    x: torch.Tensor, router_logits: torch.Tensor, core_key: str, layer: int
) -> torch.Tensor:
    """The finalized routed experts of MoE layer ``layer`` over the bf16 rows ``x``, as the eager prefill runs them:
    in ``moe_max_num_tokens``-row runner calls.

    One op for any number of chunks, so the compiled graph neither unrolls the chunks nor guards on their count. It is
    not a piecewise boundary: each bucket's segment captures the calls its rows make.
    """
    entry, core = _live_core(core_key)
    return entry.routed_experts(core, layer, x, router_logits)


@prefill_routed_experts.register_fake
def _prefill_routed_experts_fake(
    x: torch.Tensor, router_logits: torch.Tensor, core_key: str, layer: int
) -> torch.Tensor:
    return x.new_empty(x.shape)
