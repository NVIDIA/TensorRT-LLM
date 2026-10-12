# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-family sparse MQA over [sliding window | compressed] KV rows with attention sinks, FP8
compute: E4M3 query and E4M3 pools, each with one per-tensor scale, bf16 output.

The kernel is the FlashInfer Python callable
`flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4`, not a `torch.ops.trtllm`
op. On sm_103 one call launches one trtllm-gen FMHA kernel of the H512
`DynamicTokenSparse` family with `QkvE4m3OBfloat16` operands (FP8 tensor-core
MMA for QK and PV); the entry pins `backend="trtllm-gen"` and the `HND` pool
layout.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_HEAD_DIM = 512
_WINDOW = 128
#: The gate, per query row and head: ||o - ref|| <= _REL * ||ref|| + _ABS * sqrt(512) * v_max.
_REL = 2.0**-4
_ABS = 2.0**-7
#: Query rows per chunk of the reference (bounds its [rows, width, 512] fp32 gather).
_REF_CHUNK = 256

#: The target's (q_scale, kv_scale) in every cell but the scale cells.
_SCALES = (2.0**-3, 2.0**-6)
_GS = (1, 2, 3, 8, 16, 24, 32)
_HEADS = (16, 64)
#: Layer kinds: 0 window-only (index width 128), 2 and 1 compressed at that ratio (width 640).
_RATIOS = (0, 2, 1)
#: Varlen prefill steps, (label, requests as (cached, q_len)).
_PREFILL_STEPS = (
    ("1", ((0, 1),)),
    ("37", ((0, 37),)),
    ("2048", ((0, 2048),)),
    ("8192", ((0, 8192),)),
    ("cached8192+4096", ((8192, 4096),)),
    ("mixed4", ((0, 300), (5000, 240), (0, 1), (16000, 64))),
    ("32x256", ((0, 256),) * 32),
    ("mixed5", ((4000, 100), (0, 3000), (12000, 700), (0, 5), (60, 129))),
)
_SCALE_PAIRS = ((2.0**-3, 2.0**-6), (2.0**-1, 2.0**-2), (1.0, 1.0), (2.0**-5, 2.0**-8))
#: The per-query compressed read buffer, (label, max_num_tokens, requests, dense decode call).
_PER_QUERY_STEPS = (
    ("d32-2048", 2048, tuple((9000 + 37 * b, 1) for b in range(32)), True),
    ("v6x32-2048", 2048, tuple((20000 + 91 * b, 6) for b in range(32)), True),
    ("v6x32-8192", 8192, tuple((130000 + b, 6) for b in range(32)), True),
    ("p2048", 2048, ((0, 2048),), False),
    ("mixed-8192", 8192, ((1 << 20, 300), (0, 4000), (50000, 3000), (7, 1)), False),
)


class TrtllmSparseMlaDsv4Fp8(OpWrapper):
    """The sparse attention of `trtllm_sparse_mla_dsv4`, on E4M3 query and pools.

    Every query row attends the first `window_count[r] = min(128, seq_lens[b]
    - q_len_b + j + 1)` columns of its index row in the SWA pool and the
    columns `[128, sparse_topk_lens[r])` in the compressed pool; K == V
    (head_dim 512); an attended `-1` reads as an all-zero row; one fp32 sink
    per head (natural-log units) joins the softmax denominator only. With
    `q8`, `kv8` the E4M3 codes and `q_scale` / `kv_scale` their per-tensor
    dequantization scales::

        bmm1_scale = softmax_scale * q_scale * kv_scale    (float, or an fp32 [1] device tensor)
        bmm2_scale = kv_scale
        s_c        = bmm1_scale * <q8[r, h], kv8(c)>       (fp32)
        out[r, h]  = bmm2_scale * sum_c e4m3(p_c) kv8(c) / (sum_c p_c + exp(sink_h - m)),
                     p_c = exp(s_c - m)

    Both MMAs are E4M3: PV takes the probabilities rounded to E4M3 (scaled by
    448 first, the factor folded back). One `bmm1`/`bmm2` pair per call, so
    both pools carry the same `kv_scale`; no block scale is read.

    Fusion boundary: as the bf16 entry -- the gather, QK, the softmax with
    sink and PV are inside; the caller owns RoPE, the pool writes and their
    E4M3 quantization, the index rows and lengths, and what follows the
    output. Returns the bf16 output, shaped like `query` (`out` when given).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=(
                    f"{'decode' if nn == 1 else 'verify q_len 6'} G={g} H={h} ratio={r}: "
                    "dense, positions up to 16397"
                ),
                spec=dict(
                    kind="decode",
                    G=g,
                    H=h,
                    ratio=r,
                    next_n=nn,
                    q_scale=_SCALES[0],
                    kv_scale=_SCALES[1],
                ),
            )
            for g in _GS
            for h in _HEADS
            for r in _RATIOS
            for nn in (1, 6)
        )
        + tuple(
            Cell(
                why=f"prefill {label} H={h} ratio={r}: varlen",
                spec=dict(
                    kind="prefill",
                    requests=requests,
                    H=h,
                    ratio=r,
                    q_scale=_SCALES[0],
                    kv_scale=_SCALES[1],
                ),
            )
            for label, requests in _PREFILL_STEPS
            for h in _HEADS
            for r in _RATIOS
        )
        + tuple(
            Cell(
                why=f"scales q_scale={sq!r} kv_scale={skv!r}: decode H=16 ratio=1",
                spec=dict(
                    kind="scales",
                    requests=((700, 1), (3, 1), (16000, 1)),
                    H=16,
                    ratio=1,
                    q_scale=sq,
                    kv_scale=skv,
                ),
            )
            for sq, skv in _SCALE_PAIRS
        )
        + tuple(
            Cell(
                why=f"per-query {label} H={h} ratio={r}: compressed row t*512+k, decoys",
                spec=dict(
                    kind="per_query",
                    max_t=max_t,
                    requests=requests,
                    decode=decode,
                    H=h,
                    ratio=r,
                    q_scale=_SCALES[0],
                    kv_scale=_SCALES[1],
                ),
            )
            for label, max_t, requests, decode in _PER_QUERY_STEPS
            for h in _HEADS
            for r in (2, 1)
        )
    )

    note = """
    Metadata consumed: none. Everything is an argument.

    Preconditions: every precondition of `trtllm_sparse_mla_dsv4` holds here
    with bf16 read as E4M3 -- compact segments bounded by the lengths, the
    window count from `seq_lens` (0 attends the whole window), lengths within
    [128, index width], attended indices in range, NaN/Inf rows never named
    where a column is loaded, and at least 128 rows of mapped memory from the
    first row of any pool a `-1` is loaded from. In addition: query and both
    pools float8_e4m3fn (FlashInfer raises otherwise); both pools quantized
    with the same per-tensor `kv_scale`, with no value the caller needs above
    `448 * kv_scale`. An MXFP8 or NVFP4 pool is re-quantized to plain E4M3
    first: no block scale is read.

    `bmm1_scale` may be a float or an fp32 [1] device tensor (FlashInfer
    multiplies a tensor by log2(e) in a small torch op inside the call); the
    two forms give bitwise-identical output, and the tensor form is what a
    CUDA graph captures. `bmm2_scale` is a float.

    Each probability carries up to 2^-4 relative rounding (its E4M3 cast) that
    the reference does not reproduce; `compare` states the band.

    Beyond the cells the test certifies: a device `bmm1_scale` bitwise equal
    to the float, and decode / verify CUDA-graph capture at G 1, 2, 4, 8, 16,
    24, 32 x q_len 1 / 6 x H 16 / 64, each replay (new q, index rows, lengths
    and a changed `bmm1_scale`) bitwise equal to a fresh eager call.

    Not certified: the bf16 entry's pool-size, far-end, guard-page and
    kernel-band sweeps (the FP8 cubins are a different variant set); H other
    than 16 / 64; `bmm2_scale` as a tensor; an output other than bf16.
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
        bmm1_scale: Union[float, torch.Tensor],
        bmm2_scale: float,
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
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
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
        bmm1_scale: Union[float, torch.Tensor],
        bmm2_scale: float,
        cum_seq_lens_q: Optional[torch.Tensor] = None,
        max_q_len: Optional[int] = None,
        out: Optional[torch.Tensor] = None,
        enable_pdl: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 attention on the same E4M3 values, and the V scale the gate needs.

        A function of the arguments: the pools and the index rows are both
        passed. The E4M3 codes are exact in fp32; scores are `bmm1_scale`
        times the code dot product, fp32 max and sum, fp32 probabilities (no
        E4M3 rounding: that is the kernel's own arithmetic the gate bounds),
        the per-head sink in the denominator only, and the output
        `bmm2_scale` times the probability-weighted code rows. A `None` sink
        adds nothing. `workspace_buffer`, `max_q_len`, `out` and `enable_pdl`
        are ignored.

        Returns `(expected, v_max)`: `expected` fp32, shaped like `query`;
        `v_max` a 0-d fp32 tensor, `|bmm2_scale|` times the largest |code| of
        the K/V rows any query row attends (max|V| in dequantized units).
        """
        dev = query.device
        h = query.shape[-2]
        q = query.reshape(-1, h, _HEAD_DIM).float()
        n = q.shape[0]
        bmm1 = float(bmm1_scale)
        window_count = _window_counts(query, seq_lens, cum_seq_lens_q)
        swa_f = swa_kv_cache.reshape(-1, _HEAD_DIM)
        comp_f = compressed_kv_cache.reshape(-1, _HEAD_DIM)
        col = torch.arange(sparse_indices.shape[1], device=dev)
        if sinks is None:  # no sink: nothing joins the denominator
            sk = torch.full((1, h, 1), float("-inf"), dtype=torch.float32, device=dev)
        else:
            sk = sinks.float()[None, :, None]
        expected = torch.empty(n, h, _HEAD_DIM, dtype=torch.float32, device=dev)
        v_max = torch.zeros((), dtype=torch.float32, device=dev)
        for r0 in range(0, n, _REF_CHUNK):
            r1 = min(n, r0 + _REF_CHUNK)
            idx = sparse_indices[r0:r1].long()
            att = (col[None] < sparse_topk_lens[r0:r1, None]) & (
                (col[None] >= _WINDOW) | (col[None] < window_count[r0:r1, None])
            )
            rows = torch.cat(
                [swa_f[idx[:, :_WINDOW].clamp_min(0)], comp_f[idx[:, _WINDOW:].clamp_min(0)]], 1
            ).float()
            rows = torch.where((idx >= 0)[..., None], rows, 0.0)
            lo, hi = torch.aminmax(rows, dim=-1)
            row_max = torch.where(att, torch.maximum(hi, -lo), torch.zeros((), device=dev))
            v_max = torch.maximum(v_max, row_max.amax())
            s = torch.einsum("rhd,rcd->rhc", q[r0:r1], rows) * bmm1
            s = torch.where(att[:, None, :], s, -torch.inf)
            m = torch.maximum(s.amax(-1, keepdim=True), sk)
            m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
            p = torch.exp(s - m)
            den = p.sum(-1, keepdim=True) + torch.exp(sk - m)
            o = torch.einsum("rhc,rcd->rhd", p, rows) / den
            o = torch.where(att.any(-1)[:, None, None], o, torch.zeros_like(o))
            expected[r0:r1] = o * bmm2_scale
        return expected.reshape(query.shape), v_max * abs(bmm2_scale)

    def compare(self, out: torch.Tensor, ref: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Per query row and head: ||o - ref||_2 <= 2^-4 ||ref||_2 + 2^-7 sqrt(512) v_max.

        2^-4 is the largest relative rounding of one E4M3 probability, which is
        the whole error of a row whose softmax sits on one key; the absolute
        term covers rows whose output nearly cancels. Measured on sm_103: the
        worst correct cell sits at 0.59 of this gate (0.44 on the mixed control
        step), and the wrong readings at 13.6-113 times it (sink ignored,
        `q_scale` dropped, every window count 128, sink times log2(e),
        `kv_scale` applied twice, compressed segment dropped).
        """
        expected, v_max = ref
        assert out.dtype == torch.bfloat16, out.dtype
        assert out.shape == expected.shape, (out.shape, expected.shape)
        err = (out.float() - expected).norm(dim=-1)
        tol = _REL * expected.norm(dim=-1) + _ABS * math.sqrt(_HEAD_DIM) * float(v_max)
        within = err <= tol
        worst = (err / tol.clamp_min(1e-30)).max().item()
        assert bool(within.all()), (
            f"{int((~within).sum())} (row, head) outputs outside the gate, worst {worst:.3f}"
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


trtllm_sparse_mla_dsv4_fp8 = TrtllmSparseMlaDsv4Fp8()
