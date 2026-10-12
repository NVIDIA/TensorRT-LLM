# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-family sparse MQA over [sliding window | compressed] KV rows with attention sinks.

The kernel is the FlashInfer Python callable
`flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4`, not a `torch.ops.trtllm`
op. On sm_103 one call launches one trtllm-gen FMHA kernel of the H512
`DynamicTokenSparse` family (at 64 heads on the split-KV path, followed by
FlashInfer's `fmhaReductionKernel`); the entry pins `backend="trtllm-gen"` and
the `HND` pool layout, which is the path its cells certify.
"""

from __future__ import annotations

from typing import Optional

import torch
from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_HEAD_DIM = 512
_WINDOW = 128
#: The gate's relative band: torch's bf16 default.
_RTOL = 1.6e-2
#: The gate's absolute band per unit of max|V|: one bf16 ulp at the largest V element.
_ATOL_PER_VMAX = 2.0**-8
#: Query rows per chunk of the reference (bounds its [rows, width, 512] fp32 gather).
_REF_CHUNK = 256

#: Per-rank query heads of the target's members: 64 / attention TP -- 16 at tep4 / tp4,
#: 64 under attention DP and on the DSpark drafter, 8 at tep8, 32 at pp2 x tep2 / tp2.
_ALL_HEADS = (8, 16, 32, 64)
#: Decode positions: window counts 1..128, compressed counts 0..512 at ratio 2.
_DECODE_POSITIONS = (0, 3, 60, 126, 127, 128, 500, 1500, 5000)
_DECODE_REQUESTS = tuple((p, 1) for p in _DECODE_POSITIONS)
#: DSpark verify: the last accepted token plus 5 drafts per request.
_VERIFY_REQUESTS = ((0, 6), (70, 6), (125, 6), (2000, 6))
#: Varlen prefill steps, (requests as (cached, q_len), index width, what the step covers).
_PREFILL_STEPS = (
    (((0, 4096),), 640, "1 x 4096 fresh, a prefill-sized prompt"),
    (((0, 8192),), 640, "1 x 8192 fresh, the largest prefill chunk"),
    (((0, 1500), (3000, 700), (90, 1)), 640, "1500 fresh + 700 at 3000 + 1 row: chunked"),
    (((0, 3000), (200, 5)), 128, "3000 fresh + 5 at 200, window-only layer"),
)


class TrtllmSparseMlaDsv4(OpWrapper):
    """Sparse MQA (K == V, head_dim 512) of each query row over the pool rows its index row names.

    For query row `r` (request `b`, query `j` of that request's `q_len_b`), head `h`::

        window_count[r] = min(128, seq_lens[b] - q_len_b + j + 1)
        attended[r]     = {c : c < sparse_topk_lens[r] and (c >= 128 or c < window_count[r])}
        kv(c)           = swa rows[idx[r, c]] for c < 128, compressed rows[idx[r, c]] for c >= 128,
                          an all-zero row where idx[r, c] == -1
        s_c             = softmax_scale * <q[r, h], kv(c)>              (fp32, natural-log units)
        out[r, h]       = output_scale * sum_c exp(s_c - m) kv(c) / (sum_c exp(s_c - m) + exp(sinks[h] - m))

    Pool rows are in each pool's own flattened row numbering. The sink joins
    the denominator only; duplicates are attended as many times as they are
    named. Probabilities enter the PV product as bf16; scores, max and sum are
    fp32.

    Fusion boundary: one call does the gather, QK, the softmax with sink and
    PV. The caller owns RoPE of q and of the KV rows, the writes into both
    pools, the index rows and lengths, and the inverse RoPE / output projection
    afterwards. Returns the bf16 output, shaped like `query` (`out` when given).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"decode H={h} topk={topk}: q_len 1, window counts 1..128",
                spec=dict(kind="decode", H=h, topk=topk, requests=_DECODE_REQUESTS),
            )
            for h in _ALL_HEADS
            for topk in (128, 640)
        )
        + tuple(
            Cell(
                why=f"verify H={h}: dense q_len 6 (DSpark verify), 4 requests, topk 640",
                spec=dict(kind="verify", H=h, topk=640, requests=_VERIFY_REQUESTS),
            )
            for h in _ALL_HEADS
        )
        + tuple(
            Cell(
                why=f"prefill H={h} topk={topk}: {label}",
                spec=dict(kind="prefill", H=h, topk=topk, requests=requests),
            )
            for h in (8, 16, 32)
            for requests, topk, label in _PREFILL_STEPS
        )
        + (
            Cell(
                why="prefill H=64 topk=640: 1100 fresh + 300 at 2500",
                spec=dict(kind="prefill", H=64, topk=640, requests=((0, 1100), (2500, 300))),
            ),
            Cell(
                why="window-only H=16: NaN compressed placeholder never read",
                spec=dict(kind="window_only", H=16, topk=128, requests=_DECODE_REQUESTS),
            ),
            Cell(
                why="draft block H=16: 5 non-causal queries, virtual seq_lens",
                spec=dict(kind="draft_block", H=16, ring_rows=(3, 77, 128), block=5, width=136),
            ),
        )
    )

    note = """
    Metadata consumed: none. Everything the kernel reads is an argument; the
    caller builds the index rows (window positions and compressed top-k mapped
    to pool rows) and the lengths.

    Preconditions, none of them checked by the call:
    - Compact each segment and bound it with the lengths: attended rows occupy
      the first `window_count[r]` window columns and the first
      `sparse_topk_lens[r] - 128` compressed columns; `-1` is the safe filler
      after them. A `-1` inside the counts is an all-zero row (score 0, joins
      the denominator), not a mask.
    - `window_count` comes from `seq_lens` alone: a caller that needs another
      count per row passes virtual `seq_lens` (each query its own q_len-1
      request). `seq_lens = 0` is not an empty window: it attends all 128
      window columns, silently. A floored window passes `kv_len - floor`.
    - `128 <= sparse_topk_lens[r] <= sparse_indices.shape[1]`; past the width
      the call ends in an illegal memory access that poisons the CUDA context.
      Lengths below 128 are outside FlashInfer's contract.
    - Every attended index is a real row of its pool (an index past the end
      returns garbage).
    - Masked columns can still be loaded: the whole 128-column window and the
      rest of the last group of 4 that `sparse_topk_lens` cuts into. A NaN or
      Inf row named there poisons the output. A loaded `-1` reads one row in
      96..127 of its pool (then discards it), so every pool view a `-1` is
      loaded from must have at least 128 rows of mapped memory from its first
      row (measured with guard pages; a shorter allocation can fault). The
      DSpark block buffer sits at the head of an allocation of >= 128 rows, or
      repeats column 132's row in its masked tail instead of `-1`.
    - Window-only layers still pass a compressed pool; at index width 128 it
      is never read.
    - Index width a multiple of 4 and >= 128, int32; bf16 query and pools;
      fp32 sinks of exactly [H], natural-log units; `softmax_scale` the plain
      scale (no log2(e) folded in). FlashInfer raises ValueError on these.
    - `workspace_buffer`: the target's one 128 MiB buffer, zeroed once and
      shared by every call. The call checks no size and uses at most
      SMs x 64 x 1032 bytes of it; its content does not matter.

    Measured on sm_103: deterministic, and a CUDA-graph replay equals the eager
    call bitwise; no 32-bit addressing wrap up to pools of 275,251,200 rows.
    First use fetches the selected cubins from FlashInfer's artifactory unless
    FLASHINFER_CUBIN_DIR points at a prefilled cache.

    The cells are the small-pool calls. The test also drives, outside the
    cells, the target's engine-size pools, the engine envelope (context
    calls up to 16384 rows and 256 requests, decoder replay passes up to 32768
    rows, kv lengths up to 1,048,595), the native-storage compact read forms
    (`t * 128 + c`, `b * L_r + e`, `t * 512 + k`), the DSpark drafter block at
    every G 1..256, CUDA-graph capture at the 46-batch capture set, and the
    workspace high-water mark, at H 8, 16, 32 and 64.

    Not certified: `output_scale != 1`; H outside {8, 16, 32, 64};
    `sparse_topk_lens < 128`; explicit `enable_pdl`; context calls beyond the
    envelope above; decode / verify batches above 256 or `next_n` other than 1
    and 6; capture at a batch outside the capture set; the draft block at H
    other than 64 (beyond the one H=16 cell) or width other than 136; a
    workspace other than the 128 MiB buffer; E4M3 operands (that is
    `trtllm_sparse_mla_dsv4_fp8`).
    """

    def raw_call(
        self,
        query: torch.Tensor,
        swa_kv_cache: torch.Tensor,
        compressed_kv_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        sparse_indices: torch.Tensor,
        sparse_topk_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        sinks: Optional[torch.Tensor],
        softmax_scale: float,
        output_scale: float = 1.0,
        cum_seq_lens_q: Optional[torch.Tensor] = None,
        max_q_len: Optional[int] = None,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor:
        return trtllm_batch_decode_sparse_mla_dsv4(
            query=query,
            swa_kv_cache=swa_kv_cache,
            workspace_buffer=workspace_buffer,
            sparse_indices=sparse_indices,
            compressed_kv_cache=compressed_kv_cache,
            sparse_topk_lens=sparse_topk_lens,
            seq_lens=seq_lens,
            out=out,
            bmm1_scale=softmax_scale,
            bmm2_scale=output_scale,
            sinks=sinks,
            kv_layout="HND",
            cum_seq_lens_q=cum_seq_lens_q,
            max_q_len=max_q_len,
            enable_pdl=enable_pdl,
            backend="trtllm-gen",
        )

    def reference(
        self,
        query: torch.Tensor,
        swa_kv_cache: torch.Tensor,
        compressed_kv_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        sparse_indices: torch.Tensor,
        sparse_topk_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        sinks: Optional[torch.Tensor],
        softmax_scale: float,
        output_scale: float = 1.0,
        cum_seq_lens_q: Optional[torch.Tensor] = None,
        max_q_len: Optional[int] = None,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The attention read through the index rows, and the V scale the gate needs.

        A function of the arguments: the pools and the index rows are both
        passed. Gathers the rows each index row names, fp32 scores scaled by
        `softmax_scale`, fp32 max and sum, probabilities cast to bf16 for the
        PV product (the precision of DeepSeek-V4.1's reference `sparse_attn`),
        the per-head sink added to the denominator only, in natural-log units.
        `workspace_buffer`, `max_q_len`, `out` and `enable_pdl` are ignored.

        Returns `(expected, v_max)`: `expected` fp32, shaped like `query` and
        not rounded to bf16 (the gate was measured against the unrounded
        value); `v_max` a 0-d fp32 tensor, `|output_scale|` times the largest
        |element| of the K/V rows any query row attends.
        """
        dev = query.device
        h = query.shape[-2]
        q = query.reshape(-1, h, _HEAD_DIM)
        n = q.shape[0]
        window_count = _window_counts(query, seq_lens, cum_seq_lens_q)
        swa_f = swa_kv_cache.reshape(-1, _HEAD_DIM)
        comp_f = compressed_kv_cache.reshape(-1, _HEAD_DIM)
        width = sparse_indices.shape[1]
        col = torch.arange(width, device=dev)
        if sinks is None:  # no sink: nothing joins the denominator
            sk = torch.full((1, h), float("-inf"), dtype=torch.float32, device=dev)
        else:
            sk = sinks[None, :].float()
        expected = torch.empty(n, h, _HEAD_DIM, dtype=torch.float32, device=dev)
        v_max = torch.zeros((), dtype=torch.float32, device=dev)
        for s0 in range(0, n, _REF_CHUNK):
            s1 = min(n, s0 + _REF_CHUNK)
            ib = sparse_indices[s0:s1].long()
            kv = torch.cat(
                [swa_f[ib[:, :_WINDOW].clamp_min(0)], comp_f[ib[:, _WINDOW:].clamp_min(0)]], 1
            ).float()
            attended = (col[None] < sparse_topk_lens[s0:s1, None]) & (
                (col[None] >= _WINDOW) | (col[None] < window_count[s0:s1, None])
            )
            kv = kv * (ib >= 0)[..., None]
            lo, hi = torch.aminmax(kv, dim=-1)  # no |kv| temporary: the gather is the big tensor
            row_max = torch.where(attended, torch.maximum(hi, -lo), torch.zeros((), device=dev))
            v_max = torch.maximum(v_max, row_max.amax())
            s = torch.einsum("nhd,ntd->nht", q[s0:s1].float(), kv) * softmax_scale
            s = s.masked_fill(~attended[:, None, :], float("-inf"))
            m = s.amax(-1)
            m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
            p = torch.exp(s - m[..., None])
            denom = p.sum(-1) + torch.exp(sk - m)
            num = torch.einsum("nht,ntd->nhd", p.to(torch.bfloat16).float(), kv)
            o = num / denom[..., None]
            expected[s0:s1] = torch.where(attended.any(-1)[:, None, None], o, torch.zeros_like(o))
        expected = expected * output_scale
        return expected.reshape(query.shape), v_max * abs(output_scale)

    def compare(self, out: torch.Tensor, ref: tuple[torch.Tensor, torch.Tensor]) -> None:
        """`rtol` 1.6e-2 plus `atol` 2^-8 * v_max, against the fp32 reference.

        `rtol` is torch's bf16 default. The kernel casts its unnormalized
        probabilities to bf16 for PV against its own running max, which differs
        from the reference's, so each probability carries up to 2^-9 relative
        rounding the reference cannot reproduce; summed into a convex
        combination of the attended V rows that is at most 2^-8 * max|V|
        absolute -- one bf16 ulp at the largest V element. Measured on sm_103:
        the correct reference sits at 0.11-0.39 of this gate across the decode,
        verify and drafter cases, and the wrong readings at 19-61 times it
        (sink ignored, sink also in the numerator, sink times log2(e) or
        softmax_scale, duplicates de-duplicated, an attended `-1` read as a
        mask).
        """
        expected, v_max = ref
        assert out.dtype == torch.bfloat16, out.dtype
        assert out.shape == expected.shape, (out.shape, expected.shape)
        torch.testing.assert_close(
            out.float(), expected, rtol=_RTOL, atol=_ATOL_PER_VMAX * float(v_max)
        )


def _window_counts(
    query: torch.Tensor, seq_lens: torch.Tensor, cum_seq_lens_q: Optional[torch.Tensor]
) -> torch.Tensor:
    """Per query row: min(128, seq_lens[b] - q_len_b + j + 1); dense when `cum_seq_lens_q` is None."""
    dev = query.device
    if cum_seq_lens_q is None:
        q_len = query.shape[1] if query.dim() == 4 else 1
        cu = torch.arange(seq_lens.numel() + 1, device=dev) * q_len
    else:
        cu = cum_seq_lens_q.long()
    q_lens = cu[1:] - cu[:-1]
    req = torch.repeat_interleave(torch.arange(seq_lens.numel(), device=dev), q_lens)
    j = torch.arange(req.numel(), device=dev) - cu[:-1][req]
    return (seq_lens.long()[req] - q_lens[req] + j + 1).clamp(max=_WINDOW)


trtllm_sparse_mla_dsv4 = TrtllmSparseMlaDsv4()
