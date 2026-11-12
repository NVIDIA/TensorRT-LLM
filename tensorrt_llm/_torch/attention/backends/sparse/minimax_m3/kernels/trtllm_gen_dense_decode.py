# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm-gen decode attention for MiniMax-M3's dense layers (0-2).

Those layers attend the whole page table, so nothing about them needs MSA;
they only run there because MsaPrefillFmha claims every M3 layer. MSA's kernel
uses the context schedule, spending a 128-row Q tile on one decode token, while
trtllm-gen has a generation tile scheduler for exactly this shape.

The layer's own native attention call cannot be reused as-is. It reaches the
pool through the layer's pool pointer and pool mapping, which assume each layer
contributes exactly K+V to a pool slot. M3 packs K+V for every layer of a group
into one slot, and sparse layers add an index-K sub-page on top, so there is no
uniform per-layer stride and _kv_pool_mapping_offset is only a ranking, not an
addressable offset. This module goes around that: it builds the flat sub-page
pool and [batch, 2, max_blocks] block table the kernel expects directly out of
M3's own slot geometry, then runs the native trtllm-gen generation
(TrtllmGenFmha) as a prepared-query call over that table, on the step's
attention metadata with its pool table rooted at this layer's K sub-page.
"""

from __future__ import annotations

import copy
import functools
import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.paged_cache import (
    fp8_e4m3_saturate,
)
from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers

from .msa_utils import check_decode_span_shape

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.fmha.trtllm_gen import TrtllmGenFmha
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

# The pool mapping of the one-pool view below: local layer 0 is layer 0 of pool 0.
_POOL_MAPPING = torch.zeros((1, 2), dtype=torch.int32)


class _LayerKvPool:
    """A KV cache manager as the native op reads it for one dense decode call.

    The op addresses a layer's pages through the manager's pool table and the
    layer's row of its pool mapping, which cannot address M3's pool (see the
    module docstring). This view holds one pool whose primary pointer is this
    layer's K sub-page, so the sub-page indexing of the block table built here
    lands on it, and the call names local layer 0. Every other attribute is the
    manager's.
    """

    kv_cache_pool_mapping = _POOL_MAPPING

    def __init__(self, manager, kv_pool: torch.Tensor) -> None:
        self._manager = manager
        # [pool, (primary, secondary)]
        self.kv_cache_pool_pointers = torch.tensor([[kv_pool.data_ptr(), 0]], dtype=torch.int64)

    def __getattr__(self, name):
        manager = self.__dict__.get("_manager")
        if manager is None:
            raise AttributeError(name)
        return getattr(manager, name)


def _dense_call_metadata(
    metadata: "TrtllmAttentionMetadata", kv_pool: torch.Tensor, subpage_table: torch.Tensor
) -> "TrtllmAttentionMetadata":
    """The step's metadata as one dense decode call reads it.

    A shallow copy, so the step's own metadata and buffers are untouched. Its
    manager is the layer's one-pool view and its block offsets are the sub-page
    table, which kv_block_table also names. The verify rows of a step form a
    causal chain over the trailing KV that the generation kernel's causal mask
    covers, so the speculative-decoding state of the step, which describes the
    layer's own call, is cleared.
    """
    call_metadata = copy.copy(metadata)
    call_metadata.kv_cache_manager = _LayerKvPool(metadata.kv_cache_manager, kv_pool)
    # Present so the op sees a paged KV cache; kv_block_table replaces its rows.
    call_metadata.kv_cache_block_offsets = subpage_table
    call_metadata.is_spec_decoding_enabled = False
    call_metadata.is_spec_dec_tree = False
    call_metadata.force_prepare_spec_dec_tree_mask = False
    call_metadata.max_total_draft_tokens = None
    return call_metadata


def _device_index(device: torch.device) -> int:
    """Device ordinal of device, resolving the current one where it has none."""
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.lru_cache(maxsize=None)
def _dense_attention(
    device_index: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    fp8_kv: bool,
    q_scaling: float,
) -> Tuple["TrtllmAttention", "TrtllmGenFmha"]:
    """The native attention layer of a dense decode call's geometry, and its trtllm-gen library.

    It describes the call alone rather than reusing the M3 layer's own
    attention, whose sparse configuration and KV quantization are those of the
    model's sparse layers, not of this layer's pool. Built once per geometry: a
    layer allocates its constant buffers on construction.
    """
    # Function-local: the attention backends reach this package through the
    # minimax_m3 package init, which a module-scope import here would cycle with.
    from tensorrt_llm._torch.attention.backends.fmha.trtllm_gen import TrtllmGenFmha
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8.value) if fp8_kv else None
    with torch.cuda.device(device_index):
        attn = TrtllmAttention(
            layer_idx=0,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
        )
    attn.local_layer_idx = 0
    return attn, TrtllmGenFmha(attn)


@functools.lru_cache(maxsize=None)
def _fp8_scales(sm_scale: float, device_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Device BMM1 [softmax scale, softmax scale * log2(e)] and BMM2 [1] scales of an FP8 call.

    M3 stores unscaled E4M3, so only the softmax scale is folded in. Filled on
    the device rather than copied from the host so that a first call under CUDA
    graph capture records the fill.
    """
    device = torch.device("cuda", device_index)
    bmm1 = torch.full((2,), sm_scale, dtype=torch.float32, device=device)
    bmm1[1:].mul_(math.log2(math.e))
    return bmm1, torch.ones(1, dtype=torch.float32, device=device)


def subpage_block_table(
    block_table: torch.Tensor, subpages_per_slot: int, reserve: bool = False
) -> torch.Tensor:
    """Expand a slot table into trtllm-gen's separate K and V page rows.

    uses_shared_paged_kv_idx is False for TensorRT-LLM, so the kernel takes
    [batch, 2, max_blocks] and indexes K and V independently. Rooting the pool
    at this layer's K (see get_kv_subpage_pool) puts slot s's K at
    s * subpages_per_slot and its V one sub-page later.

    The result is a function of the slot table and that factor alone, so every
    dense layer of a step would compute the same one. prepare() therefore
    stages it once into a graph-stable buffer, and this runs only where it
    could not: a manager whose layers disagree on the factor, or a caller that
    skipped prepare().
    """
    batch, max_blocks = block_table.shape
    out = get_memory_buffers().get_buffer(
        [batch, 2, max_blocks],
        torch.int32,
        buffer_name="m3_trtllm_gen_subpage_block_table",
        reserve_buffer=reserve,
    )
    write_subpage_block_table(block_table, subpages_per_slot, out)
    return out


def write_subpage_block_table(
    block_table: torch.Tensor, subpages_per_slot: int, out: torch.Tensor
) -> None:
    """Write the K and V sub-page rows of block_table into out."""
    torch.mul(block_table, subpages_per_slot, out=out[:, 0])
    torch.add(out[:, 0], 1, out=out[:, 1])


def uniform_subpages_per_slot(kv_cache_manager) -> int:
    """Sub-pages per slot when every layer of the pool agrees, else 0.

    The factor is a property of a layer group, so a single-group model has one
    for the whole pool and prepare() can expand the block table without naming
    a layer (see subpage_block_table). A manager with no sub-page pool, or one
    whose groups disagree, reports 0 rather than a guess.
    """
    get_pool = getattr(kv_cache_manager, "get_kv_subpage_pool", None)
    layer_offsets = getattr(kv_cache_manager, "layer_offsets", None)
    if get_pool is None or not layer_offsets:
        return 0
    is_nvfp4 = getattr(kv_cache_manager, "is_nvfp4_layer", lambda _: False)
    is_hybrid_draft = getattr(kv_cache_manager, "is_hybrid_draft_layer", lambda _: False)
    # NVFP4 sparse layers and the hybrid draft layer (read through its own view)
    # never read the staged table and may sit in pools with another factor.
    factors = {
        int(get_pool(layer_idx, "HND")[1])
        for layer_idx in layer_offsets
        if not is_nvfp4(layer_idx) and not is_hybrid_draft(layer_idx)
    }
    return factors.pop() if len(factors) == 1 else 0


def minimax_m3_trtllm_gen_dense_decode(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    metadata: "TrtllmAttentionMetadata",
    layer_idx: int,
    block_table: torch.Tensor,  # [batch, max_blocks] slot ids
    seq_lens: torch.Tensor,  # [batch] int32
    *,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    decode_query_len: int,
    max_seq_len: int,
    workspace: torch.Tensor,
    staged_subpage_table: Optional[torch.Tensor] = None,
    staged_subpages_per_slot: int = 0,
) -> None:
    """Full-context decode attention through the native trtllm-gen library, in place into output.

    metadata is the step's attention metadata, whose KV cache manager owns the
    layer's pool. block_table and seq_lens are the decode rows' slot table and
    KV lengths, and max_seq_len bounds those lengths for the multi-CTA KV split.

    staged_subpage_table is the expansion of block_table prepare() already
    staged, used when staged_subpages_per_slot matches this layer's factor and
    expanded here otherwise; see subpage_block_table.

    workspace is the attention workspace the native op works in and grows
    outside CUDA graph capture. The multi-CTA KV counters are the op's own,
    sized for the metadata's max_num_requests.
    """
    # The multi-CTA KV counters are sized against max_num_requests, so a batch
    # read out of a longer q would undersize them.
    check_decode_span_shape(
        "MiniMax-M3 trtllm-gen dense decode",
        int(q.shape[0]),
        int(seq_lens.shape[0]),
        decode_query_len,
    )

    kv_pool, subpages_per_slot = metadata.kv_cache_manager.get_kv_subpage_pool(layer_idx, "HND")
    num_tokens, num_heads, head_dim = (int(x) for x in q.shape)
    num_kv_heads, tokens_per_block = int(kv_pool.shape[1]), int(kv_pool.shape[2])
    batch = int(seq_lens.shape[0])
    device_index = _device_index(q.device)

    # The kernels share one dtype across q/k/v, so an FP8 pool needs FP8 Q. M3
    # stores unscaled E4M3, so this is a plain cast and a no-op when the fused
    # producer already emitted FP8.
    fp8_kv = kv_pool.dtype == torch.float8_e4m3fn
    if fp8_kv:
        prepared_query = fp8_e4m3_saturate(q)
        bmm1_scale, bmm2_scale = _fp8_scales(float(sm_scale), device_index)
    elif kv_pool.dtype == q.dtype:
        prepared_query, bmm1_scale, bmm2_scale = q, None, None
    else:
        raise ValueError(
            f"MiniMax-M3 dense decode got a {q.dtype} query for a {kv_pool.dtype} KV pool; "
            "a non-FP8 pool is read with a query of its own dtype."
        )
    if staged_subpage_table is None or staged_subpages_per_slot != subpages_per_slot:
        staged_subpage_table = subpage_block_table(
            block_table, subpages_per_slot, torch.cuda.is_current_stream_capturing()
        )

    from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )

    # BF16/FP16 runs take the softmax scale from q_scaling; FP8 runs read the device scales.
    attn, fmha = _dense_attention(
        device_index,
        num_heads,
        num_kv_heads,
        head_dim,
        fp8_kv,
        1.0 / (float(sm_scale) * math.sqrt(head_dim)),
    )
    # The op names the activation dtype by its fused-QKV input; the kernel reads
    # only the prepared query, so a query already in FP8 is named by the output.
    activation = q if q.dtype in (torch.bfloat16, torch.float16) else output
    # M3 dense layers are fully causal: a window covering both the page table
    # and the length bound never slides.
    window = max(int(staged_subpage_table.shape[-1]) * tokens_per_block, int(max_seq_len))
    fmha.run_generation(
        FmhaParams(
            attn=attn,
            meta=_dense_call_metadata(metadata, kv_pool, staged_subpage_table),
            fwd=AttentionForwardArgs(
                output=output,
                attention_input_type=AttentionInputType.generation_only,
                is_fused_qkv=True,
            ),
            workspace=workspace,
            qkv_input=activation.reshape(num_tokens, num_heads * head_dim),
            output=output.view(num_tokens, num_heads, head_dim),
            sequence_lengths=seq_lens,
            context_lengths=seq_lens,
            input_seq_length=decode_query_len,
            max_past_kv_length=int(max_seq_len),
            max_attention_window_size=window,
            cyclic_attention_window_size=window,
            num_tokens=num_tokens,
            tokens_per_block=tokens_per_block,
            kv_factor=2,
            batch_size=batch,
            num_requests=batch,
            num_gen_tokens=num_tokens,
            local_layer_idx=0,
            prepared_query=prepared_query.contiguous(),
            prepared_bmm1_scale=bmm1_scale,
            prepared_bmm2_scale=bmm2_scale,
            kv_block_table=staged_subpage_table,
        )
    )


def dense_decode_unsupported_reason(kv_cache_manager, head_dim: int) -> Optional[str]:
    """Return None when the geometry is supported, else why it is not.

    Takes head_dim rather than a query tensor so the metadata's up-front
    validation can reach the same verdict as the call site without one, and so
    the two cannot drift.
    """
    if not hasattr(kv_cache_manager, "get_kv_subpage_pool"):
        return "the KV cache manager does not expose a flat sub-page pool."
    if int(head_dim) != 128:
        return f"head_dim {int(head_dim)}; only 128 is validated on trtllm-gen here."
    return None


__all__ = [
    "dense_decode_unsupported_reason",
    "minimax_m3_trtllm_gen_dense_decode",
    "subpage_block_table",
    "uniform_subpages_per_slot",
    "write_subpage_block_table",
]
