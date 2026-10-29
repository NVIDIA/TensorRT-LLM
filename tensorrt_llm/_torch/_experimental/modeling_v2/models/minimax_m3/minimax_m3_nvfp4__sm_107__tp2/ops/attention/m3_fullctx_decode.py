# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dense layers' full-context decode attention at TP2 on Rubin (``ops/m3_fullctx_decode``): PrimTS split-KV
decode at the recipe modeling's ``FULLCTX_DECODE_RECIPES`` pins per (query rows per request, requests)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.paged_cache import (
    fp8_e4m3_saturate,
)

from .. import m3_fullctx_decode as kernel
from .m3_dense_decode import dense_attention_reference, gather_paged_kv

__all__ = ["m3_fullctx_decode"]

HEADS = kernel.HEADS
KV_HEADS = kernel.KV_HEADS
HEAD_DIM = kernel.HEAD_DIM
#: The plan capacity (the per-request K/V length bound PrimTS sizes its policy for): dispatch's
#: FULLCTX_DECODE_MAX_KV_LEN, which the forward's prebuilt plans use.
MAX_KV_LEN = 1 << 21


class M3FullCtxDecode(OpWrapper):
    """Decode attention of every query row over its request's full causal context in the paged FP8 pool, written
    into ``output`` (BF16 [T, heads, head_dim]) in place.

    ``recipe`` is ``modeling's FULLCTX_DECODE_RECIPES[(qlen, requests)]``: the PrimTS Q tile and the split-KV fanout.
    ``own_reducer`` merges the splits on the BF16 op (modeling's ``FULLCTX_DECODE_OWN_REDUCER``) rather than PrimTS's
    FP16 reducer and a cast.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one request verifying 1 + 3 rows over an AgentX-length (165k) context",
            spec=dict(kv_lens=(165525,), qlen=4),
        ),
        Cell(
            why="one request, one draft-pass row, over a 165k context",
            spec=dict(kv_lens=(165522,), qlen=1),
        ),
        Cell(
            why="two requests verifying 1 + 3 rows at unequal lengths (165k, 120k)",
            spec=dict(kv_lens=(165525, 120011), qlen=4),
        ),
        Cell(
            why="two requests, one row each, at unequal lengths (165k, 120k)",
            spec=dict(kv_lens=(165522, 120008), qlen=1),
        ),
        Cell(
            why="one request verifying 1 + 3 rows over a short (4.1k) context: most splits are empty",
            spec=dict(kv_lens=(4104,), qlen=4),
        ),
    )

    note = """
    Same operands and arithmetic as the stock trtllm-gen call (E4M3 q, E4M3 K/V, P quantized to E4M3 per split,
    fp32 accumulation): the split differs, so the result differs from stock as a stock split change does (the P
    quantization is per split). Deterministic, and the same bits eager and in a CUDA graph. Held to the fp32
    reference within the FP8-KV band of the in-tree dense decode test.
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv_cache_manager,
        layer_idx: int,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        sm_scale: float,
        output: torch.Tensor,
        decode_query_len: int,
        recipe,
        max_kv_len: int = MAX_KV_LEN,
        own_reducer: bool = True,
    ) -> torch.Tensor:
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        shape = (decode_query_len, int(seq_lens.shape[0]))
        kernel.prebuild(q.device, {shape: recipe}, max_kv_len, own_reducer, get_env_enable_pdl())
        plan = kernel.plan_for(q.device, *shape, own_reducer)
        pool, subpages = kv_cache_manager.get_kv_subpage_pool(layer_idx, "HND")
        k_pages, v_pages = kernel.kv_page_views(pool, subpages)
        plan.run(fp8_e4m3_saturate(q), k_pages, v_pages, block_table, seq_lens, sm_scale, output)
        return output

    def reference(
        self,
        q: torch.Tensor,
        kv_cache_manager,
        layer_idx: int,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        sm_scale: float,
        output: torch.Tensor,
        decode_query_len: int,
        recipe,
        max_kv_len: int = MAX_KV_LEN,
        own_reducer: bool = True,
    ) -> torch.Tensor:
        """fp32 attention over the pool's K / V, q rounded to E4M3 first as the kernel runs it."""
        keys, values = gather_paged_kv(kv_cache_manager, layer_idx, block_table, seq_lens)
        return dense_attention_reference(
            fp8_e4m3_saturate(q), keys, values, seq_lens, decode_query_len, sm_scale
        )

    def is_valid(
        self,
        q: torch.Tensor,
        kv_cache_manager,
        layer_idx: int,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        sm_scale: float,
        output: torch.Tensor,
        decode_query_len: int,
        recipe,
        max_kv_len: int = MAX_KV_LEN,
        own_reducer: bool = True,
    ) -> None:
        pool, _ = kv_cache_manager.get_kv_subpage_pool(layer_idx, "HND")
        assert pool.dtype == torch.float8_e4m3fn, "an FP8 E4M3 pool"
        assert tuple(pool.shape[1:]) == (KV_HEADS, kernel.PAGE, HEAD_DIM), (
            "the TP2 dense-layer pool geometry"
        )
        assert q.shape[1:] == (HEADS, HEAD_DIM), "32 query heads of 128 per rank"
        assert int(q.shape[0]) == int(seq_lens.shape[0]) * decode_query_len, (
            "q covers exactly the batch"
        )
        if torch.cuda.is_current_stream_capturing():
            return  # the length checks read the device: eager calls only
        assert bool((seq_lens >= decode_query_len).all()), "a request shorter than its query rows"
        assert bool((seq_lens <= max_kv_len).all()), "a request longer than the plan capacity"


m3_fullctx_decode = M3FullCtxDecode()
