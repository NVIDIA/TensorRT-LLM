# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Indexer logits over packed MXFP4 q/k with contiguous (non-paged) keys: the prefill form.

The op is the vendored DeepGEMM Python API `tensorrt_llm.deep_gemm.fp8_fp4_mqa_logits`
(built from the `deepgemm` entry of `3rdparty/fetch_content.json`), a pybind-backed callable
rather than a `torch.ops.trtllm` op.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm import deep_gemm
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

#: (T, S, H, ratio, what else it covers) of one request's causal prefill from empty: S = T // ratio.
_CAUSAL = (
    (1, 1, 32, 1, ""),
    (5, 2, 32, 2, ", row 0 sees no key"),
    (64, 64, 32, 1, ""),
    (333, 166, 32, 2, ", T not a multiple of the 4-row query block"),
    (4096, 2048, 32, 2, ""),
    (4096, 4096, 32, 1, ""),
    (8192, 4096, 32, 2, ""),
    (8192, 8192, 32, 1, ""),
    (64, 64, 8, 1, ", a TP4 rank's quarter of the heads"),
    (4096, 4096, 8, 1, ", a TP4 rank's quarter of the heads"),
)

#: (cached, new, ratio) of one request's context chunk over its cached keys.
_CHUNKED = ((8000, 6, 1), (8190, 1, 2), (3000, 1000, 2), (12288, 4096, 1))

#: Long-context prefill shapes, as per-request (cached, new) lists and the ratio: the 24K passkey
#: prompt's last chunk (T 7674 x S 24058, a runtime dump), S 24058 at ratio 2, S 32768 at 4096 and
#: 8192 rows, and three packed requests with their own key ranges (S 32576 / 16288).
_LONG = (
    ([(16384, 7674)], 1),
    ([(44020, 4096)], 2),
    ([(28672, 4096)], 1),
    ([(24576, 8192)], 1),
    ([(61440, 4096)], 2),
    ([(16384, 4096), (0, 3000), (8000, 1096)], 1),
    ([(16384, 4096), (0, 3000), (8000, 1096)], 2),
)

#: (T, S, ratio) of the recipe-regime causal chunks whose last row sees key S - 1.
_RECIPE = (
    (1, 4096, 1),
    (512, 1024, 1),
    (4096, 2048, 2),
    (4096, 4096, 1),
    (7674, 24058, 1),
    (4096, 32768, 1),
)


def _extent(reqs, ratio: int) -> tuple[int, int]:
    """(query rows, packed key rows) of a list of (cached, new) requests at a ratio."""
    return sum(n for _, n in reqs), sum((c + n) // ratio for c, n in reqs)


def _decode(packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """int8 `[..., 64]` packed E2M1 + int32 `[...]` packed UE8M0 -> float64 `[..., 128]`, exactly.

    Element `2j` is the low nibble of byte `j`; byte `b` of the little-endian
    word is the exponent of elements `32b .. 32b+31`.
    """
    lut = torch.tensor(_E2M1, dtype=torch.float64, device=packed.device)
    p = packed.view(torch.uint8).long()
    v = torch.stack([lut[p & 0xF], lut[p >> 4]], dim=-1).flatten(-2)
    e = sf.contiguous().view(torch.uint8).view(*sf.shape, 4).long() - 127
    scale = ((e + 1023) << 52).view(torch.float64)  # 2^e from the exponent bits
    return (v.unflatten(-1, (4, 32)) * scale.unsqueeze(-1)).flatten(-2)


class DeepgemmFp8Fp4MqaLogits(OpWrapper):
    """Weighted-ReLU MQA logits of every query row against its range of contiguous key rows.

    The callable is `tensorrt_llm.deep_gemm.fp8_fp4_mqa_logits` (vendored DeepGEMM, pybind),
    one call. With `ks = cu_seq_len_k_start`, `ke = cu_seq_len_k_end`:

        logits[t, s] = sum_h relu(dec(q[t, h]) . dec(kv[s])) * weights[t, h]   if ks[t] <= s < ke[t]
                     = -inf                      otherwise, when clean_logits=True
                     = not specified             otherwise, when clean_logits=False

    `q` is int8 `[T, H, 64]` packed E2M1 with int32 `q_sf [T, H]` (four UE8M0
    bytes, one per 32 elements), `kv` int8 `[S, 64]` with int32 `kv_sf [S]`,
    `weights` fp32 `[T, H]`. The relu is per head, before the weight; the head
    sum is inside the call. Products come from the block-scaled FP4 MMA with
    fp32 accumulation; relu, weighting and the head sum are fp32.

    `max_seqlen_k > 0` selects the compressed form (requires
    `clean_logits=False`): column `j` of row `t` is key `ks[t] + j` for
    `j < ke[t] - ks[t]`, the rest not specified.

    Fusion boundary: decoding both operands, the per-head dot products, relu,
    weighting, the head sum and the `-inf` fill. The caller owns quantizing q
    and k, RoPE, the weights projection and its scale fold, gathering k out of
    the paged pool, candidate masking, the top-k and any cross-rank reduction.

    Returns a fresh `[T, S]` (or `[T, max_seqlen_k]`) tensor of
    `logits_dtype`: a strided view whose row stride is padded (see `note`).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"exact T={t} S={s} H={h} r={r}: causal prefill{extra}",
                spec=dict(regime="exact", H=h, r=r, reqs=[(0, t)]),
            )
            for t, s, h, r, extra in _CAUSAL
        )
        + tuple(
            Cell(
                why=f"exact T={n} S={(c + n) // r} r={r}: chunk after {c} cached",
                spec=dict(regime="exact", H=32, r=r, reqs=[(c, n)]),
            )
            for c, n, r in _CHUNKED
        )
        + tuple(
            Cell(
                why=f"exact packed r={r}: 4 requests at own key offsets",
                spec=dict(regime="exact", H=32, r=r, reqs=[(0, 700), (0, 1), (512, 2500), (0, 37)]),
            )
            for r in (1, 2)
        )
        + tuple(
            Cell(
                why=(
                    f"exact long T={_extent(reqs, r)[0]} S={_extent(reqs, r)[1]} r={r}"
                    f" x{len(reqs)}: allocation = the logits buffer"
                ),
                spec=dict(regime="exact", H=32, r=r, reqs=reqs, alloc_check=True),
            )
            for reqs, r in _LONG
        )
        + tuple(
            Cell(
                why=f"recipe T={t} S={s} r={r}: bf16 randn via per_token_cast_to_fp4",
                spec=dict(regime="recipe", H=32, r=r, reqs=[(s * r - t, t)]),
            )
            for t, s, r in _RECIPE
        )
        + (
            Cell(
                why="wide UE8M0 bytes 90..160, per token and per key",
                spec=dict(regime="wide_scale", H=32, r=1, reqs=[(256, 256)]),
            ),
            Cell(
                why="clean_logits=False: in-range values, shifted ranges",
                spec=dict(regime="exact", H=32, T=300, S=600, ranges="shifted"),
            ),
            Cell(
                why="compressed form: column j is key ks + j, empty rows",
                spec=dict(regime="exact", H=32, T=200, S=1000, ranges="compressed"),
            ),
        )
    )

    note = """
    Cell regimes: "exact" packs random E2M1 bytes with UE8M0 bytes 126..128 and
    weights `n * 2^-10` (|n| <= 8) times the 2^-6 fold, so every head term and
    partial sum is exact in fp32 (the test asserts that premise) and the op is
    bit-identical to the float64 reference; "recipe" packs bf16 randn with
    DeepGEMM's `per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32,
    use_packed_ue8m0=True)` and uses fp32 randn weights times 2^-6.

    Metadata consumed: none. The launch grid uses DeepGEMM's process-global
    `deep_gemm.get_num_sms()`. For DeepSeek-V4.1: `weights = weights_proj(x) *
    128^-0.5 * 32^-0.5` (exactly 2^-6; 32 is the global index head count even
    on a rank holding fewer heads), `ks[t]` the request's first key row and
    `ke[t] = ks[t] + (p + 1) // ratio` for row `t`'s absolute position `p`. The
    checkpoint's reference scores in bf16; these fp32 logits are more precise,
    so top-k near-ties can resolve differently.

    Output: row stride `align(S + 256, 256)` (compressed: `align(max_seqlen_k,
    256)`), so the result is not contiguous and its consumer must accept the
    padded stride. One allocation of `align(T, 128 / H) * stride * 4` bytes and
    no other workspace; the first call in a process also keeps 32 MiB.

    JIT: the one kernel (`sm100_mqa_logits`, which with `clean_logits=True`
    also writes the -inf columns; there is no separate clean pass) is compiled
    with NVCC on the first call of each configuration (heads, dtypes,
    compressed or not) and cached on disk under
    `$HOME/.deep_gemm` (`DG_JIT_CACHE_DIR`). Make that call outside CUDA-graph
    capture; after it, capture and replay on new inputs are bit-exact.

    Preconditions: `T >= 1` and `S >= 1` (otherwise a non-sticky
    CUDA_ERROR_INVALID_VALUE; skip the call when no key exists yet);
    `0 <= ks[t]` and `ke[t] <= S` (`ke > S` is out of contract); q and k in the
    same byte layout (a layout swapped in both is invisible, in one is wrong
    everywhere); `kv_sf` 16-byte aligned and a `weights` row stride a multiple
    of 4 floats (otherwise CUDA_ERROR_INVALID_VALUE); the index-K gather op's
    float8 / float32 outputs viewed as int8 / int32 first. Every mismatch probed
    raised, so there is no `is_valid`. H must be a multiple of 4 up to 128 (the
    op raises otherwise); H 24 and 128 are measured against the reference by a
    test, without being certified.

    Certified at H 32, `clean_logits=True`, full width (the target's call):
    one call per context step of 1..8192 rows from 1..32 requests of up to
    16384 tokens (S up to 524288 at ratio 1, 262144 at ratio 2; every T in
    1..8192 and every S in 1..32768 for one request), and the row-tiled
    long-context form -- tiles of a step of up to 16384 rows from up to 256
    requests of up to 1,048,590 tokens over up to 2^25 keys, each tile within
    2^28 logits elements, rows passed as row-offset views. A row's logits do
    not depend on the call's extents or on where its keys sit (bitwise, recipe
    regime included), and repeated calls are bitwise identical. One compiled
    variant serves every T and S. The target never captures
    the call (context steps are not graph-replayed).

    Not certified: other architectures (DeepGEMM dispatches on the arch
    major); `q_sf=None` (the FP8-q path); bf16 or fp16 weights; bf16 logits;
    H other than 32, and 8 beyond its cells; `clean_logits=False` and the
    compressed form beyond their cells; anything outside the two envelopes.
    """

    def raw_call(
        self,
        q: torch.Tensor,
        q_sf: Optional[torch.Tensor],
        kv: torch.Tensor,
        kv_sf: torch.Tensor,
        weights: torch.Tensor,
        cu_seq_len_k_start: torch.Tensor,
        cu_seq_len_k_end: torch.Tensor,
        clean_logits: bool,
        max_seqlen_k: int,
        logits_dtype: torch.dtype,
    ) -> torch.Tensor:
        return deep_gemm.fp8_fp4_mqa_logits(
            (q, q_sf),
            (kv, kv_sf),
            weights,
            cu_seq_len_k_start,
            cu_seq_len_k_end,
            clean_logits=clean_logits,
            max_seqlen_k=max_seqlen_k,
            logits_dtype=logits_dtype,
        )

    def reference(
        self,
        q: torch.Tensor,
        q_sf: Optional[torch.Tensor],
        kv: torch.Tensor,
        kv_sf: torch.Tensor,
        weights: torch.Tensor,
        cu_seq_len_k_start: torch.Tensor,
        cu_seq_len_k_end: torch.Tensor,
        clean_logits: bool,
        max_seqlen_k: int,
        logits_dtype: torch.dtype,
    ) -> torch.Tensor:
        """`sum_h relu(q_h . k) * w_h` over exactly decoded operands, in float64, rounded once.

        NaN marks every element the op does not specify: outside `[ks, ke)`
        when `clean_logits=False`, and the compressed form's columns at or past
        a row's length. `compare` skips those. `q_sf` must be given (the MXFP4
        path); nothing is mutated.
        """
        qd = _decode(q, q_sf)
        kd = _decode(kv, kv_sf)
        num_q, num_k = qd.shape[0], kd.shape[0]
        wd = weights.double()
        full = torch.empty(num_q, num_k, dtype=torch.float64, device=q.device)
        for t0 in range(0, num_q, 512):
            dots = torch.einsum("thd,kd->thk", qd[t0 : t0 + 512], kd)
            full[t0 : t0 + 512] = (dots.clamp_min(0) * wd[t0 : t0 + 512].unsqueeze(-1)).sum(1)
        ks = cu_seq_len_k_start.long()
        ke = cu_seq_len_k_end.long()
        if max_seqlen_k > 0:
            j = torch.arange(max_seqlen_k, device=q.device)
            inside = j[None, :] < (ke - ks)[:, None]
            src = (ks[:, None] + j[None, :]).clamp(0, max(num_k - 1, 0))
            out = full.gather(1, src).masked_fill(~inside, float("nan"))
        else:
            col = torch.arange(num_k, device=q.device)
            vis = (col[None, :] >= ks[:, None]) & (col[None, :] < ke[:, None])
            out = full.masked_fill(~vis, float("-inf") if clean_logits else float("nan"))
        return out.to(logits_dtype)

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """`-inf` exactly where the reference has it, default fp32 tolerance on visible elements.

        Elements the reference marks NaN (not specified by the op) are skipped.
        The band is torch's default for fp32 (rtol 1.3e-6, atol 1e-5): on bf16
        randn packed by the recipe with fp32 randn weights (T 512, S 1024, H
        32) the op's largest difference from float64 was 5.9e-7, against 4.9e-7
        for an fp32 torch einsum of the same decoded values, and 43% of visible
        elements were bit-exact. In the exact regime the op is bit-identical,
        which the test asserts on top of this gate. Controls (relu after the
        head sum, an 8-head weight fold, swapped k nibbles, reversed scale bytes)
        put nearly every visible element outside the band.
        """
        assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        specified = ~torch.isnan(ref)
        masked = torch.isneginf(ref)
        wrong_inf = int(((torch.isneginf(out) & specified) != masked).sum())
        assert wrong_inf == 0, f"{wrong_inf} elements disagree on -inf (outside [ks, ke))"
        visible = specified & ~masked
        torch.testing.assert_close(out[visible], ref[visible])


deepgemm_fp8_fp4_mqa_logits = DeepgemmFp8Fp4MqaLogits()
