# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 Rubin FP8 split-K context FMHA: attend one KV partition and merge it into the output."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ...kernels.k3r_mla_ctx import k3_context_fmha_op as _fmha

# Heads per reference block: bounds the FP32 score tensor of an 8k x 8k causal chunk.
_REF_HEADS = 4


def _merge(old_out, old_max, old_sum, cur_out, cur_max, cur_sum):
    """Merge two softmax-normalized partial outputs by their (row max, row sum) statistics."""
    merged_max = torch.maximum(old_max, cur_max)
    old_weight = old_sum * (old_max - merged_max).exp()
    cur_weight = cur_sum * (cur_max - merged_max).exp()
    merged_sum = old_weight + cur_weight
    merged = (old_out * old_weight[..., None] + cur_out * cur_weight[..., None]) / merged_sum[
        ..., None
    ]
    return merged, merged_max, merged_sum


class K3ContextFmha(OpWrapper):
    """``run_k3_context_fmha``: FP8 Q ``[T_q, H, 192]`` against one FP8 KV partition (K-nope
    ``[T_kv, H, 128]``, the shared K-pe ``[T_kv, 1, 64]``, V ``[T_kv, H, 128]``), merged into the
    BF16 ``output`` ``[T_q, H, 128]`` and FP32 ``merge_stats`` ``[T_q, H, 2]`` (row max, row sum,
    natural log) per request by ``merge_op``: 2 writes, 1 merges, 0 leaves both unchanged."""

    ARCHS = frozenset({Arch.SM_107})
    # Spec: per-request `q_lens` / `kv_lens` of one partition, `causal`, the per-request
    # `merge_ops`, and the `cu_kv` dtype (int64: a cached-prefix partition's cu_chunked_seq_len;
    # int32: the current chunk, whose cu_kv is cu_q). 96 heads: the tp1 context server.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="tp1 cached prefix, first partition: two requests write output and stats (op 2)",
            spec=dict(
                q_lens=[257, 17], kv_lens=[384, 128], causal=False, merge_ops=[2, 2], cu_kv="int64"
            ),
        ),  # fmt: skip
        Cell(
            why="tp1 cached prefix, later partition: merge (op 1) beside a request without one (0)",
            spec=dict(
                q_lens=[257, 17], kv_lens=[384, 0], causal=False, merge_ops=[1, 0], cu_kv="int64"
            ),
        ),  # fmt: skip
        Cell(
            why="tp1 current chunk after a cached prefix: causal, merged into the prefix (op 1)",
            spec=dict(
                q_lens=[257, 17], kv_lens=[257, 17], causal=True, merge_ops=[1, 1], cu_kv="int32"
            ),
        ),  # fmt: skip
        Cell(
            why="tp1 first chunk of an 8k prompt: causal over itself, writes (op 2)",
            spec=dict(q_lens=[8192], kv_lens=[8192], causal=True, merge_ops=[2], cu_kv="int32"),
        ),
        Cell(
            why="tp1 8k chunk against an 8k cached-prefix partition (op 1)",
            spec=dict(q_lens=[8192], kv_lens=[8192], causal=False, merge_ops=[1], cu_kv="int64"),
        ),
    )

    note = """
    The chunked prefill of an NVFP4-KV MLA layer runs this once per cached-prefix partition
    (non-causal, the partition's `merge_op_tensor` row) and once for the current chunk (causal,
    top-left aligned mask: query i of a request sees keys 0..i of its partition). Writes `output`
    and `merge_stats` in place and returns None.

    Inputs are unit-scale E4M3; the kernel rounds the softmax probabilities to E4M3 (prescale 256)
    before the PV product, so a causal row over few keys differs from the FP32 reference by up to
    half an E4M3 step of P times |V|. The stats are (row max, row sum) of exp(score - max) in
    natural-log units; only max + log(sum) is determined (the kernel may keep another max).

    A process cache holds one compiled kernel per (Q dtype, heads, causal, cu_kv_seqlens dtype);
    the first call of a key compiles (not capturable). `cu_q_seqlens` int32, `cu_kv_seqlens` int32
    or int64, both `[B + 1]`; `merge_op` int64 `[B]`. K-nope / V need unit head stride and a
    128-element head pitch (views of the kv_b output are fine). SM107 only; the launcher raises
    ValueError on any other device or layout.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        merge_stats: torch.Tensor,
        merge_op: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        softmax_scale: float,
        is_causal: bool,
    ) -> None:
        _fmha.run_k3_context_fmha(
            q,
            k_nope,
            k_pe,
            v,
            output,
            cu_q_seqlens,
            cu_kv_seqlens,
            merge_stats,
            merge_op,
            max_q_len,
            max_kv_len,
            softmax_scale,
            is_causal,
        )

    def reference(
        self,
        q: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        merge_stats: torch.Tensor,
        merge_op: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        softmax_scale: float,
        is_causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The post-state ``(output, merge_stats)``: FP32 attention of every request over its
        partition, merged into the pre-call state per ``merge_op``. Reads ``output`` and
        ``merge_stats`` (the running state) and writes neither; ``max_q_len`` / ``max_kv_len``
        are launch bounds only."""
        heads = q.shape[1]
        out = output.float().clone()
        stats = merge_stats.clone()
        cu_q = cu_q_seqlens.tolist()
        cu_kv = cu_kv_seqlens.tolist()
        ops = merge_op.tolist()
        with true_fp32_matmul():
            for b, op in enumerate(ops):
                if op == 0:
                    continue
                q0, q1, k0, k1 = cu_q[b], cu_q[b + 1], cu_kv[b], cu_kv[b + 1]
                if q1 == q0:
                    continue
                for h0 in range(0, heads, _REF_HEADS):
                    hs = slice(h0, min(h0 + _REF_HEADS, heads))
                    qb = q[q0:q1, hs].float()
                    scores = torch.einsum(
                        "qhd,khd->hqk", qb[..., :128], k_nope[k0:k1, hs].float()
                    ) + torch.einsum("qhd,kd->hqk", qb[..., 128:], k_pe[k0:k1, 0].float())
                    scores *= softmax_scale
                    if is_causal:
                        rows = torch.arange(q1 - q0, device=q.device)[:, None]
                        cols = torch.arange(k1 - k0, device=q.device)[None, :]
                        scores.masked_fill_((cols > rows)[None], -torch.inf)
                    cur_max = scores.amax(-1)
                    weights = (scores - cur_max[..., None]).exp()
                    cur_sum = weights.sum(-1)
                    cur_out = torch.einsum("hqk,khd->hqd", weights, v[k0:k1, hs].float())
                    cur_out = (cur_out / cur_sum[..., None]).transpose(0, 1)
                    cur_max, cur_sum = cur_max.transpose(0, 1), cur_sum.transpose(0, 1)
                    if op == 1:
                        cur_out, cur_max, cur_sum = _merge(
                            out[q0:q1, hs],
                            stats[q0:q1, hs, 0],
                            stats[q0:q1, hs, 1],
                            cur_out,
                            cur_max,
                            cur_sum,
                        )
                    out[q0:q1, hs] = cur_out
                    stats[q0:q1, hs, 0] = cur_max
                    stats[q0:q1, hs, 1] = cur_sum
        return out.to(output.dtype), stats

    def is_valid(
        self,
        q: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        merge_stats: torch.Tensor,
        merge_op: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        softmax_scale: float,
        is_causal: bool,
    ) -> None:
        # The kernel tiles queries and keys up to these bounds: a request longer than them
        # is attended only partly, with no error.
        q_lens = cu_q_seqlens.diff()
        kv_lens = cu_kv_seqlens.diff()
        assert int(q_lens.max()) <= max_q_len, "max_q_len below a request's query count"
        assert int(kv_lens.max()) <= max_kv_len, "max_kv_len below a request's key count"
        # A merge code outside {0, 1, 2} selects no defined epilogue.
        assert bool(((merge_op >= 0) & (merge_op <= 2)).all()), "merge_op codes are 0, 1, 2"


k3_context_fmha = K3ContextFmha()
