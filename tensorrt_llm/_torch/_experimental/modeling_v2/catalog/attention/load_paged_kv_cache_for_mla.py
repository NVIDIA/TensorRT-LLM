# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather context rows out of the paged MLA latent cache, via the trtllm op."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: quant_mode bit for an fp8 KV cache (tensorrt_llm.quantization.mode.QuantMode).
QM_FP8_KV_CACHE = 128


class _LoadPagedKvCacheForMla(OpWrapper):
    """Copy positions `[0, L_s)` of every context sequence out of the paged MLA
    latent cache into two new contiguous tensors, `(compressed_kv, k_pe)`.

    A pure gather: the pool is read and never written, and the slot one past
    each sequence's range is not touched. Nothing is recomputed -- when the pool
    holds e4m3 the op dequantizes, and that is the whole of its arithmetic.
    """

    ARCHS = frozenset({Arch.SM_103})

    # deepseek-r1's MLA latent geometry, which is the only caller: a 512-wide
    # compressed_kv and a 64-wide k_pe sharing one 576-wide pool row. The page
    # sizes are the engine default and the value a tuned MLA target opts into;
    # both are covered because the gather addresses pages, not tokens.
    _R1 = dict(kv_lora_rank=512, qk_rope_head_dim=64, quant_mode=QM_FP8_KV_CACHE)

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="production context call: fp8 pool, scale omitted, bf16 out, page 32",
            spec=dict(
                cached=[0, 32, 100, 511],
                new=[37, 64, 300, 1],
                tokens_per_block=32,
                out_dtype=torch.bfloat16,
                read_scale=None,
                **_R1,
            ),
        ),
        Cell(
            why="page 64, the size a tuned MLA target opts into",
            spec=dict(
                cached=[0, 32, 100, 511],
                new=[37, 64, 300, 1],
                tokens_per_block=64,
                out_dtype=torch.bfloat16,
                read_scale=None,
                **_R1,
            ),
        ),
        Cell(
            why="an explicit read scale, which the omitted form has to equal at 1.0",
            spec=dict(
                cached=[0, 32],
                new=[37, 64],
                tokens_per_block=64,
                out_dtype=torch.bfloat16,
                read_scale=1.0,
                **_R1,
            ),
        ),
        Cell(
            why="a non-unit read scale: the dequantization is the op's only arithmetic",
            spec=dict(
                cached=[0, 32],
                new=[37, 64],
                tokens_per_block=64,
                out_dtype=torch.bfloat16,
                read_scale=0.375,
                **_R1,
            ),
        ),
    )

    note = """
    No `is_valid`. Everything this op could get wrong is either rejected by the
    op itself or is a property of what some earlier call *wrote* -- a read scale
    that does not match the write scale returns plausible numbers, but nothing
    visible at this call site can tell. The test drives that case directly
    instead, which is the only place it can be seen.

    A bf16 pool (`quant_mode=0`) works and is a straight copy, but every shipped
    target runs the fp8 pool, so it is not a cell.
    """

    def __call__(
        self,
        out_dtype: torch.dtype,
        num_contexts: int,
        num_ctx_kv_tokens: int,
        max_ctx_kv_len: int,
        cu_ctx_kv_lens: torch.Tensor,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        host_kv_cache_pool_mapping: torch.Tensor,
        kv_scale_quant_orig: Optional[torch.Tensor],
        layer_idx: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        tokens_per_block: int,
        attention_window_size: int,
        beam_width: int,
        quant_mode: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.load_paged_kv_cache_for_mla(
            out_dtype,
            num_contexts,
            num_ctx_kv_tokens,
            max_ctx_kv_len,
            cu_ctx_kv_lens,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            kv_scale_quant_orig,
            layer_idx,
            kv_lora_rank,
            qk_rope_head_dim,
            tokens_per_block,
            attention_window_size,
            beam_width,
            quant_mode,
        )

    def reference(
        self,
        stored: torch.Tensor,
        kv_scale_quant_orig: Optional[torch.Tensor],
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """`float(e4m3 byte) * scale`, in fp32, rounded once to `out_dtype`.

        The second documented exception to mirroring `__call__`, after the
        collectives: this op's output is a function of what the paged cache
        holds, and its arguments only say where to look. `stored` is the rows
        the gather was pointed at; the test reads them out of the pool, which is
        the only place they exist.

        An omitted scale behaves exactly as 1.0 -- stated here rather than in
        prose because a cell drives both forms against this one line.
        """
        wide = stored.float()
        if kv_scale_quant_orig is not None:
            wide = wide * kv_scale_quant_orig[0]
        return wide.to(out_dtype)

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """A gate of exactly zero: one multiply and one rounding, on both sides.

        The op dequantizes in fp32 and rounds once; so does `reference`. There
        is no accumulation to disagree about, so any difference is a wrong
        gather, a wrong scale, or a wrong row -- never drift.
        """
        assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)


load_paged_kv_cache_for_mla = _LoadPagedKvCacheForMla()
