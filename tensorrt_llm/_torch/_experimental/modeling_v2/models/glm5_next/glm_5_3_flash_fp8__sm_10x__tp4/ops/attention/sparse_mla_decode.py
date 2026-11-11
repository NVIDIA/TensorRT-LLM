# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA over the FP8 latent cache: FlashInfer's TRTLLM-GEN native NoPE MLA decode kernel.

Every query row ``r`` (a decode, verify or context row alike) is its own sequence of one token:
it attends, with all its heads, to the latent rows ``packed[r, :lengths[r]]`` of the cache,
which serve as both keys and values. The scores are ``q . k * bmm1_scale`` over the 512-wide
latent space (the queries are absorbed; there is no rotary part), the softmax runs in FP32, and
the output ``softmax @ v * bmm2_scale`` is BF16. For an FP8 cache with dequantization scale
``s`` and the model's softmax scale ``qk_nope_head_dim ** -0.5``, ``bmm1_scale = s *
qk_nope_head_dim ** -0.5`` and ``bmm2_scale = s``.

Buffers the caller owns and passes on every call:

- ``workspace``: uint8, at least ``WORKSPACE_BYTES``, zeroed before its first use. The kernel
  keeps the partial results of a row split across CTAs there.
- ``counter``: uint8, contiguous, 16-byte aligned, zeroed once, at least
  ``counter_bytes(rows, heads, sm_count)`` for the call's rows and heads. The kernel resets
  its semaphores after every launch, so ordered launches on one stream (and replays of one
  CUDA graph) share it; launches that can run concurrently each need their own.

``buffer_bytes`` gives both sizes for a largest call; a CUDA-graph capture needs both buffers
allocated before it starts.
"""

from __future__ import annotations

import torch
from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

#: Workspace bytes the kernel needs: the TRTLLM-GEN share of the generation attention
#: workspace (``CUBLAS_WORKSPACE_SIZE``), which is what the module path reserves for it.
WORKSPACE_BYTES = 32 * 1024 * 1024

#: The native NoPE MLA geometry: absorbed 512-wide queries and latents, no rotary part, and
#: the latent cache read as 32-row pages.
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 256
PAGE_ROWS = 32


def counter_bytes(rows: int, num_heads: int, sm_count: int) -> int:
    """Bytes of multi-CTA semaphores a call of ``rows`` query rows of ``num_heads`` heads needs:
    one 4-byte counter per (row, head), at least one per SM, rounded up to 8 counters."""
    return get_trtllm_gen_multi_ctas_kv_counter_bytes(rows, num_heads, sm_count)


def buffer_bytes(max_rows: int, num_heads: int, sm_count: int) -> tuple[int, int]:
    """``(workspace, counter)`` bytes covering every call of up to ``max_rows`` rows (for
    example the engine's ``max_num_tokens``) of ``num_heads`` heads on a GPU of ``sm_count``
    SMs (``torch.cuda.get_device_properties(device).multi_processor_count``)."""
    return WORKSPACE_BYTES, counter_bytes(max_rows, num_heads, sm_count)


class SparseMlaDecode(OpWrapper):
    """``query`` E4M3 ``[R, H, 512]`` over ``kv_rows`` E4M3 ``[N, 1, 512]`` -> BF16 ``[R, H, 512]``.

    ``packed`` int32 ``[R, W]`` holds row ``r``'s latent row ids (rows of ``kv_rows``) in its
    first ``lengths[r]`` columns and ``-1`` after them (``compact_sparse_rows``); ``lengths``
    int32 ``[R]`` is at least 1. A row whose selection was empty carries one dummy id and
    length 1, and gets that row's value: the caller discards it. ``bmm1_scale`` and
    ``bmm2_scale`` are Python floats (fixed in a captured graph) or FP32 device tensors of one
    element.
    """

    ARCHS = frozenset({Arch.SM_100})
    # TP4 decode and verify rows attend with the rank's 16 heads; the sequence-parallel
    # context rows attend with all 64 for the rank's quarter of a prompt, and a prompt below
    # the sequence-parallel threshold (2,048 tokens) with 16 heads for every row. Every row
    # selects 512 pools of 4 plus a tail of up to 3: lengths up to 2,051 in packed rows of
    # 2,112. `kv` is the requests' KV length after the step (generation cells).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MTP-3 verify of one 13K-token request, 16 heads",
            spec=dict(heads=16, requests=1, tokens=4, kv=13312),
        ),
        Cell(
            why="MTP-3 verify of eight 13K-token requests, 16 heads",
            spec=dict(heads=16, requests=8, tokens=4, kv=13312),
        ),
        Cell(
            why="MTP-3 verify of 64 requests, the largest CUDA-graph batch; one row empty",
            spec=dict(heads=16, requests=64, tokens=4, kv=13312, empty_rows=1),
        ),
        Cell(
            why="single-token decode of eight 13K-token requests, 16 heads",
            spec=dict(heads=16, requests=8, tokens=1, kv=13312),
        ),
        Cell(
            why="one rank's sequence-parallel quarter of a 13,312-token prompt, 64 heads",
            spec=dict(heads=64, context=((0, 13312),), rows=(0, 3328)),
        ),
        Cell(
            why="a 2,047-token prompt below the sequence-parallel threshold, 16 heads",
            spec=dict(heads=16, context=((0, 2047),), rows=(0, 2047)),
        ),
    )

    note = """
    The kernel rounds the softmax probabilities to E4M3 for the value product (the queries
    and the cache are E4M3 already), so an output moves by a few percent of its own scale
    against FP32 attention; the entry's test derives its band from that rounding.

    The query is E4M3: the caller applies the saturating conversion (`clamp(+-448)`) the
    module path applies to the absorbed BF16 query. `kv_rows` is the uniformly strided row
    view of the latent pool (row id = base_row + slot * rows_per_slot + offset); the kernel
    reads it as 32-row pages, so its row count is a multiple of 32.

    Heads: 16, 32 and 64 run this kernel in the module path; 64 < H < 128 has no TRTLLM-GEN
    kernel and raises. A zero length is not supported by the kernel; `is_valid` refuses it.
    """

    def __call__(self, query, kv_rows, packed, lengths, workspace, counter, bmm1_scale, bmm2_scale):
        return trtllm_batch_decode_with_kv_cache_mla(
            query=query.unsqueeze(1),
            kv_cache=kv_rows.view(-1, 1, PAGE_ROWS, KV_LORA_RANK),
            workspace_buffer=workspace,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=0,
            block_tables=packed.unsqueeze(1),
            seq_lens=lengths,
            max_seq_len=packed.shape[1],
            sparse_mla_top_k=packed.shape[1],
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            backend="trtllm-gen",
            sparse_mla_top_k_lens=lengths,
            multi_ctas_kv_counter_buffer=counter,
            use_fp16_softmax=False,
        ).squeeze(1)

    def reference(
        self, query, kv_rows, packed, lengths, workspace, counter, bmm1_scale, bmm2_scale
    ):
        """FP32 attention of the dequantized (exact) E4M3 operands over each row's selected
        latent rows, one BF16 rounding. ``workspace`` and ``counter`` are not read."""
        rows, heads, dim = query.shape
        # Gathered as bytes: advanced indexing does not take every FP8 dtype.
        latent = kv_rows.reshape(-1, dim).view(torch.uint8)
        out = torch.empty(rows, heads, dim, dtype=torch.bfloat16, device=query.device)
        col = torch.arange(packed.shape[1], device=query.device)
        for r0 in range(0, rows, 16):
            part = slice(r0, min(r0 + 16, rows))
            selected = col[None, :] < lengths[part, None].long()
            ids = packed[part].long().clamp(min=0)
            values = latent[ids].view(kv_rows.dtype).float()  # [c, W, dim]
            # The padding past each length is never read: no value of it may reach the sums.
            values = values.masked_fill(~selected[:, :, None], 0.0)
            with true_fp32_matmul():
                logits = torch.einsum("rhd,rwd->rhw", query[part].float(), values) * bmm1_scale
            probs = logits.masked_fill(~selected[:, None, :], float("-inf")).softmax(dim=-1)
            with true_fp32_matmul():
                attended = torch.einsum("rhw,rwd->rhd", probs, values) * bmm2_scale
            out[part] = attended.to(torch.bfloat16)
        return out

    def is_valid(self, query, kv_rows, packed, lengths, workspace, counter, bmm1_scale, bmm2_scale):
        assert query.dtype == kv_rows.dtype == torch.float8_e4m3fn, (
            f"the kernel is the E4M3 x E4M3 one; got query {query.dtype}, cache {kv_rows.dtype}"
        )
        assert query.dim() == 3 and query.shape[2] == KV_LORA_RANK and query.is_contiguous()
        rows, heads, _ = query.shape
        assert heads in (16, 32, 64), f"{heads} heads; the module path runs 16, 32 or 64"
        assert kv_rows.shape[-1] == KV_LORA_RANK and kv_rows.shape[0] % PAGE_ROWS == 0
        assert packed.dtype == torch.int32 and packed.dim() == 2 and packed.is_contiguous()
        assert lengths.dtype == torch.int32 and lengths.shape == (rows,)
        assert workspace.dtype == torch.uint8 and workspace.is_contiguous()
        # The kernel's split-KV scratch is "the rest of the workspace": a short one is
        # overrun, not refused.
        assert workspace.numel() >= WORKSPACE_BYTES, (
            f"workspace of {workspace.numel()} bytes < {WORKSPACE_BYTES}"
        )
        # Semaphores left nonzero by anything but this kernel desynchronize the split rows.
        assert not bool(counter.any()), "the multi-CTA counter must be zero between launches"
        if rows == 0:
            return
        # A zero length is unsupported by the kernel; a length past the row reads -1 padding.
        assert int(lengths.min()) >= 1 and int(lengths.max()) <= packed.shape[1]
        col = torch.arange(packed.shape[1], device=packed.device)
        selected = packed[col[None, :] < lengths[:, None].long()]
        # An id outside the cache is read, not refused.
        assert int(selected.min()) >= 0 and int(selected.max()) < kv_rows.shape[0], (
            "every selected id must be a row of kv_rows"
        )


sparse_mla_decode = SparseMlaDecode()
