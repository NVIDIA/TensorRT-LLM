# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer decode / verify logits over the paged FP8 index-key cache (DeepGEMM)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.deep_gemm import fp8_paged_mqa_logits as _deep_gemm_fp8_paged_mqa_logits


def _spec(batch, next_n, context_lens, **overrides) -> dict:
    """``context_lens``: (low, high) of the per-request key counts."""
    spec = dict(
        batch=batch,
        next_n=next_n,
        num_heads=32,
        head_dim=128,
        tokens_per_block=64,
        context_lens=tuple(context_lens),
        page_stride_pad=0,
        max_context_len=32768,
        data="int",
    )
    spec.update(overrides)
    return spec


class Fp8PagedMqaLogits(OpWrapper):
    """``logits[b * n + t, j] = k_scale[j] * sum_h weights[b * n + t, h] * relu(q[b, t, h] . k[j])``
    for the keys ``j <= context_lens[b, t] - n + t`` of request ``b``.

    Same cache layout as ``cute_dsl_fp8_paged_mqa_logits``; ``context_lens`` is int32 [B, n] (the
    request's key count repeated over its n rows), ``schedule_meta`` the DeepGEMM schedule built
    from those 2-D lengths with block_kv 64. Returns a new fp32 [B * n, max_context_len] tensor;
    columns past a row's causal end hold unspecified values.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="plain decode, one request at the top-k edge",
            spec=_spec(1, 1, (2049, 2049)),
        ),
        Cell(
            why="plain decode, 8 requests of 1.5k-10k keys",
            spec=_spec(8, 1, (1500, 10000)),
        ),
        Cell(
            why="plain decode, 32-token pages",
            spec=_spec(4, 1, (3000, 9000), tokens_per_block=32),
        ),
        Cell(
            why="plain decode, 128-token pages",
            spec=_spec(4, 1, (3000, 9000), tokens_per_block=128),
        ),
        Cell(
            why="verify rows n=2 (a native DeepGEMM next_n, off the default path)",
            spec=_spec(4, 2, (3000, 9000)),
        ),
        Cell(
            why="verify rows n=4 (DeepGEMM's widest native next_n on sm_100f)",
            spec=_spec(4, 4, (3000, 9000)),
        ),
        Cell(
            why="padded page stride: the V2 manager's per-layer view of a wider page",
            spec=_spec(4, 1, (3000, 5000), page_stride_pad=512),
        ),
        Cell(
            why="real-valued UE8M0-scaled keys and weights: accumulation order visible",
            spec=_spec(4, 1, (3000, 6000), data="randn"),
        ),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300) and sm_107 (VR200).

    The K3.1 general target calls this from the DSA indexer for every generation row when
    use_cute_dsl_paged_mqa_logits is off (or CuTe DSL is unavailable), with q [B, n, 32, 128] and
    context_lens [B, n] (the request's key count repeated); otherwise the indexer runs
    cute_dsl_fp8_paged_mqa_logits. use_cute_dsl_paged_mqa_logits defaults to True, so this entry
    runs only when a deployment turns the CuTe DSL kernel off. Pages of 32, 64 and 128 tokens.
    The output is a fresh allocation per call. Deterministic for equal inputs.
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
    ) -> torch.Tensor:
        return _deep_gemm_fp8_paged_mqa_logits(
            q, kv_cache, weights, context_lens, block_table, schedule_meta, int(max_context_len)
        )

    def reference(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
    ) -> torch.Tensor:
        """fp64 evaluation over each row's causal keys, -inf elsewhere (``schedule_meta`` is
        ignored)."""
        batch, next_n, num_heads, head_dim = q.shape
        page = kv_cache.shape[1]
        logits = torch.full(
            (batch * next_n, max_context_len), float("-inf"), dtype=torch.float32, device=q.device
        )
        for b in range(batch):
            ctx = int(context_lens[b, -1])
            pages = block_table[b, : -(-ctx // page)].long()
            raw = kv_cache[pages].reshape(pages.shape[0], -1)
            keys = raw[:, : page * head_dim].contiguous().view(torch.float8_e4m3fn)
            scales = raw[:, page * head_dim : page * (head_dim + 4)].contiguous()
            scales = scales.view(torch.float32)
            k = keys.reshape(-1, head_dim)[:ctx].double() * scales.reshape(-1)[:ctx, None].double()
            for t in range(next_n):
                row = b * next_n + t
                end = ctx - next_n + t + 1
                if end <= 0:
                    continue
                score = (q[b, t].double() @ k[:end].t()).relu()
                logits[row, :end] = (score * weights[row].double()[:, None]).sum(0).float()
        return logits


fp8_paged_mqa_logits = Fp8PagedMqaLogits()
