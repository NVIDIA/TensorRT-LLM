# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latent MQA decode over the paged bf16 MLA cache on FlashInfer's TRTLLM-Gen MLA kernel: what the built-in MLA runs the
generation rows of Kimi K3's mixed steps (context and generation requests in one step) on (the stock FlashInfer
TRTLLM-Gen FMHA library)."""

from __future__ import annotations

import math
from typing import Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl

#: The fused nanobind ops the FlashInfer TRTLLM-Gen library needs from this build.
_REQUIRED_THOP = (
    "get_trtllm_gen_context_workspace_layout",
    "get_trtllm_gen_generation_workspace_layout",
    "build_trtllm_gen_kv_cache_metadata",
)
# The trtllm-gen kernels index the multi-CTA KV counters in groups of eight.
_COUNTER_ALIGNMENT = 8


def _thop():
    from tensorrt_llm.bindings.internal import thop

    return thop


def available() -> bool:
    """FlashInfer is installed and this build has the fused ops its TRTLLM-Gen library needs."""
    if not IS_FLASHINFER_AVAILABLE:
        return False
    thop = _thop()
    return all(hasattr(thop, name) for name in _REQUIRED_THOP)


def enable_pdl() -> bool:
    """Whether the launch uses programmatic dependent launch (``TRTLLM_ENABLE_PDL``, on by default)."""
    return get_env_enable_pdl()


def workspace_bytes(
    dtype: torch.dtype,
    num_tokens: int,
    max_context_length: int,
    max_num_sequences: int,
    num_heads: int,
    head_size: int,
    rope_dim: int,
) -> int:
    """Bytes of the attention workspace the stock library requires for a generation-only call of ``num_tokens``
    rows: the larger of the TRTLLM-Gen context and generation layouts at the call's extents."""
    thop = _thop()
    max_tokens = max(num_tokens, max_context_length, max_num_sequences)
    max_gen_tokens = max(num_tokens, max_num_sequences)
    context = thop.get_trtllm_gen_context_workspace_layout(
        dtype, max_num_sequences, max_tokens, num_heads, head_size, rope_dim, True, False
    )
    generation = thop.get_trtllm_gen_generation_workspace_layout(
        dtype, max_num_sequences, max_gen_tokens, num_heads, head_size, rope_dim, 1
    )
    return max(int(context["total_size"]), int(generation["total_size"]))


def counter_bytes(num_heads: int, max_num_sequences: int, multi_processor_count: int) -> int:
    """Bytes of the multi-CTA KV counter: one int32 per head per sequence, at least the SM count, in groups of 8."""
    n = max(num_heads * max_num_sequences, multi_processor_count)
    return (n + _COUNTER_ALIGNMENT - 1) // _COUNTER_ALIGNMENT * _COUNTER_ALIGNMENT * 4


def paged_cache(
    host_kv_cache_pool_pointers: torch.Tensor,
    host_kv_cache_pool_mapping: torch.Tensor,
    kv_cache_block_offsets: torch.Tensor,
    layer_idx: int,
    tokens_per_block: int,
    head_size: int,
    total_num_blocks: int,
    batch_start: int,
    batch_size: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(kv_cache, block_tables)`` as the kernel reads them: the layer's latent pool (one head, kv factor 1, bf16)
    and the block-offset rows ``[batch_start, batch_start + batch_size)`` of its pool."""
    kv_cache, block_tables, _ = _thop().build_trtllm_gen_kv_cache_metadata(
        host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping,
        kv_cache_block_offsets,
        layer_idx,
        1,
        tokens_per_block,
        head_size,
        1,
        total_num_blocks,
        0,
        batch_start,
        batch_size,
        dtype,
    )
    return kv_cache, block_tables


class FlashinferMlaDecode(OpWrapper):
    """Write ``out`` ``[R, S, num_heads, kv_lora_rank]``: for request ``r``'s query token ``s`` and each head,
    ``softmax(q . K^T * bmm1_scale) V`` over its first ``seq_lens[r] - S + 1 + s`` cached latent rows
    ``[c_latent | c_rope]`` (``K`` the whole row, ``V`` the latent part). ``query`` is ``[R, S, num_heads, 576]``;
    ``kv_cache`` and ``block_tables`` come from ``paged_cache``."""

    ARCHS = frozenset({Arch.SM_100})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="Kimi K3 mixed step: 6 heads, page 64, the 10 generation requests of 1 token after a context request",
            spec=dict(
                num_heads=6,
                seq_len_q=1,
                num_contexts=1,
                kv_lens=(152, 69, 106, 3, 67, 139, 9, 92, 35, 66),
            ),
        ),
        Cell(
            why="a mixed step whose generation requests verify 8 tokens (DSpark)",
            spec=dict(num_heads=6, seq_len_q=8, num_contexts=1, kv_lens=(160, 77, 114)),
        ),
    )

    note = """
    bf16 only, at Kimi K3's latent geometry (512 + 64, v 512) and 64-token pages. Linear-chain causal verification
    only (`is_var_seq`); a tree mask is not expressible. `workspace` is the attention workspace sized by
    `workspace_bytes`, viewed as `[-1, 4]`; `counter` the layer's multi-CTA KV counter of `counter_bytes` zeroed bytes,
    which the kernel leaves zeroed. `max_seq_len` must cover the longest `seq_lens` entry.
    """

    def __call__(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        workspace: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        out: torch.Tensor,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        bmm1_scale: float,
        enable_pdl: bool,
        counter: torch.Tensor,
    ) -> torch.Tensor:
        import flashinfer

        flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
            query,
            kv_cache,
            workspace.view(-1, 4),
            qk_nope_head_dim,
            kv_lora_rank,
            qk_rope_head_dim,
            block_tables,
            seq_lens,
            max_seq_len,
            0,
            out,
            bmm1_scale,
            1.0,
            None,
            None,
            enable_pdl,
            backend="trtllm-gen",
            is_var_seq=True,
            uses_shared_paged_kv_idx=False,
            multi_ctas_kv_counter_buffer=counter,
        )
        return out

    def reference(
        self,
        query: torch.Tensor,
        rows: list,
        out: torch.Tensor,
        bmm1_scale: float,
        kv_lora_rank: int,
    ) -> torch.Tensor:
        """fp32 latent MQA over each request's cached rows ``rows[r]`` (``[len, 576]``, as the test reads them back
        through the KV cache manager's block ids). The pool addressing arguments are the test's business."""
        r_count, s_count, heads, _ = query.shape
        ref = torch.empty(
            r_count, s_count, heads, kv_lora_rank, dtype=torch.float32, device=query.device
        )
        for r in range(r_count):
            kv = rows[r].float()
            for s in range(s_count):
                visible = kv.shape[0] - s_count + 1 + s
                scores = (query[r, s].float() @ kv[:visible].t()) * bmm1_scale
                ref[r, s] = scores.softmax(-1) @ kv[:visible, :kv_lora_rank]
        return ref.to(out.dtype)

    def is_valid(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        workspace: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        out: torch.Tensor,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        bmm1_scale: float,
        enable_pdl: bool,
        counter: torch.Tensor,
    ) -> None:
        assert query.dtype == out.dtype == torch.bfloat16, "bf16 query and output only"
        assert int(seq_lens.max()) <= max_seq_len, (
            "max_seq_len shorter than a request's KV length; the kernel reads past the rows it plans for"
        )
        assert not 64 < query.shape[2] < 128, (
            "TRTLLM-Gen MLA decode selects unsupported kernels at 64 < H < 128"
        )


def bmm1_scale(qk_nope_head_dim: int, qk_rope_head_dim: int, q_scaling: float) -> float:
    return 1.0 / (q_scaling * math.sqrt(qk_nope_head_dim + qk_rope_head_dim))


flashinfer_mla_decode = FlashinferMlaDecode()


__all__ = [
    "available",
    "bmm1_scale",
    "counter_bytes",
    "enable_pdl",
    "flashinfer_mla_decode",
    "paged_cache",
    "workspace_bytes",
]
