# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm-gen decode attention for MiniMax-M3's dense layers (0-2), for callers without attention metadata.

The in-tree MsaDecodeFmha runs these layers through
``sparse.minimax_m3.kernels.trtllm_gen_dense_decode`` on the step's
TrtllmAttentionMetadata. The modeling_v2 ops hold only the cache manager, the
slot table and the lengths, so this module stands in the metadata the native op
reads.

The layer's own native attention call cannot be reused as-is. It reaches the
pool through the layer's pool pointer and pool mapping, which assume each layer
contributes exactly K+V to a pool slot. M3 packs K+V for every layer of a group
into one slot, and sparse layers add an index-K sub-page on top, so there is no
uniform per-layer stride and _kv_pool_mapping_offset is only a ranking, not an
addressable offset. This module goes around that: it builds the flat sub-page
pool and [batch, 2, max_blocks] block table the kernel expects directly out of
M3's own slot geometry, then runs the native trtllm-gen generation
(TrtllmGenFmha) as a prepared-query call over that table, with the pool rooted
at this layer's K sub-page.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    check_decode_span_shape,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.paged_cache import (
    fp8_e4m3_saturate,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.trtllm_gen_dense_decode import (  # noqa: E501
    subpage_block_table,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.fmha.trtllm_gen import TrtllmGenFmha
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention

# The pool mapping of the one-pool table below: local layer 0 is layer 0 of pool 0.
_POOL_MAPPING = torch.zeros((1, 2), dtype=torch.int32)
# The [context, generation] KV totals of a call. Only context attention reads
# them; given, they keep the op from reducing the host KV lengths, which these
# callers do not have.
_KV_TOTALS = torch.zeros(2, dtype=torch.int32)


@dataclass
class _DenseDecodeMeta:
    """The attention metadata the native op reads, for one dense decode call.

    A prepared-query generation call reads only the batch bound, the page table
    and the pool, so this stands in for the step's TrtllmAttentionMetadata,
    which cannot address M3's pool (see the module docstring). Its pool table
    holds one pool whose primary pointer is this layer's K sub-page, so the
    native op's sub-page indexing matches the block table built here. Every
    other field is the value of a plain one-beam decode step without
    speculative decoding, cross attention, Helix or sparse attention.
    """

    max_num_requests: int
    max_seq_len: int
    max_context_length: int
    # Present so the op sees a paged KV cache; kv_block_table replaces its rows.
    kv_cache_block_offsets: torch.Tensor
    host_kv_cache_pool_pointers: torch.Tensor
    host_kv_cache_pool_mapping: torch.Tensor = _POOL_MAPPING
    max_num_sequences: Optional[int] = None
    cache_indirection: Optional[torch.Tensor] = None
    is_cross: bool = False
    effective_beam_width: int = 1
    max_context_q_len_override: Optional[int] = None
    is_spec_decoding_enabled: bool = False
    is_spec_dec_tree: bool = False
    force_prepare_spec_dec_tree_mask: bool = False
    max_total_draft_tokens: Optional[int] = None
    trtllm_gen_jit_warmup: bool = False
    helix_position_offsets: Optional[torch.Tensor] = None
    helix_is_inactive_rank: Optional[torch.Tensor] = None
    block_ids_per_seq: Optional[torch.Tensor] = None
    flash_mla_tile_scheduler_metadata: Optional[torch.Tensor] = None
    flash_mla_num_splits: Optional[torch.Tensor] = None
    use_paged_context_fmha: bool = False
    num_sparse_topk: int = 0
    helix_spec_tokens_valid: bool = False
    # Host lengths; a generation call reads them only for the KV totals, which it passes.
    kv_lens_runtime: Optional[torch.Tensor] = None
    prompt_lens_cpu_runtime: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.max_num_sequences is None:
            self.max_num_sequences = self.max_num_requests


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


@functools.lru_cache(maxsize=None)
def _standalone_workspace(device_index: int) -> torch.Tensor:
    """The per-device workspace of these calls; the native op grows it on first use."""
    return torch.empty((0,), dtype=torch.uint8, device=torch.device("cuda", device_index))


def minimax_m3_trtllm_gen_dense_decode(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache_manager,
    layer_idx: int,
    block_table: torch.Tensor,  # [batch, max_blocks] slot ids
    seq_lens: torch.Tensor,  # [batch] int32
    *,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    decode_query_len: int,
    max_seq_len: int,
    max_num_requests: int,
) -> None:
    """Full-context decode attention through the native trtllm-gen library, in place into output.

    The native op works in a per-device workspace it grows outside CUDA graph
    capture, and owns its multi-CTA KV counters, sized for max_num_requests.
    """
    # The multi-CTA KV counters are sized against max_num_requests, so a batch
    # read out of a longer q would undersize them.
    check_decode_span_shape(
        "MiniMax-M3 trtllm-gen dense decode",
        int(q.shape[0]),
        int(seq_lens.shape[0]),
        decode_query_len,
    )

    kv_pool, subpages_per_slot = kv_cache_manager.get_kv_subpage_pool(layer_idx, "HND")
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
    subpage_table = subpage_block_table(
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
    meta = _DenseDecodeMeta(
        max_num_requests=int(max_num_requests),
        max_seq_len=int(max_seq_len),
        max_context_length=int(decode_query_len),
        kv_cache_block_offsets=subpage_table,
        # [pool, (primary, secondary)]: the primary pool starts at this layer's K sub-page.
        host_kv_cache_pool_pointers=torch.tensor([[kv_pool.data_ptr(), 0]], dtype=torch.int64),
    )
    # The op names the activation dtype by its fused-QKV input; the kernel reads
    # only the prepared query, so a query already in FP8 is named by the output.
    activation = q if q.dtype in (torch.bfloat16, torch.float16) else output
    # M3 dense layers are fully causal: a window covering both the page table
    # and the length bound never slides.
    window = max(int(subpage_table.shape[-1]) * tokens_per_block, int(max_seq_len))
    fmha.run_generation(
        FmhaParams(
            attn=attn,
            meta=meta,
            fwd=AttentionForwardArgs(
                output=output,
                attention_input_type=AttentionInputType.generation_only,
                is_fused_qkv=True,
            ),
            workspace=_standalone_workspace(device_index),
            qkv_input=activation.reshape(num_tokens, num_heads * head_dim),
            output=output.view(num_tokens, num_heads, head_dim),
            sequence_lengths=seq_lens,
            context_lengths=seq_lens,
            host_total_kv_lens=_KV_TOTALS,
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
            kv_block_table=subpage_table,
        )
    )


__all__ = ["minimax_m3_trtllm_gen_dense_decode"]
