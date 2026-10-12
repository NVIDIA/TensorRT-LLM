# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection post mapping (hc_post) via the trtllm mhc_post_mapping kernel.

The op is `torch.ops.trtllm.mhc_post_mapping`, registered in `cpp/tensorrt_llm/thop/mhcOp.cpp`.
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_HC = 4
_H_CKPT = 5120

#: Row counts DeepSeek-V4.1 makes its call at, from the target's runtime-value dumps: the trunk's
#: ('mhc', T, tile) records (T 1..16384, engines up to max_batch_size 256 / max_num_tokens 16384,
#: CUDA-graph and eager, dep4 and tp4, 128K and 1M max_seq_len, the long-context envelope) and
#: the DSpark drafter's ('dspark.mhc', N, tile) records (N = 5G); plus the tops of the step range
#: (16384) and of the decoder bounded-replay pass (32768).
_TARGET_ROWS = tuple(
    sorted(
        {
            1, 2, 3, 4, 5, 6, 7, 8, 12, 14, 16, 18, 24, 29, 32, 34, 48, 60, 96, 108, 127, 128,
            129, 134, 144, 168, 192, 228, 240, 288, 330, 336, 384, 432, 462, 480, 528, 546, 576,
            597, 610, 624, 672, 720, 768, 780, 852, 888, 890, 1018, 1152, 1536, 1906, 1935, 2033,
            2038, 2048, 2049, 2252, 2778, 2784, 3870, 4208, 4245, 4388, 4394, 4559, 4591, 4868,
            7835, 7855, 7893, 7908, 8147, 8191, 8192, 8467, 9193, 9240, 9364, 9370, 9524, 10441,
            10444, 10447, 10451, 16298, 16308, 16316, 16320, 16322, 16327, 16328, 16329, 16333,
            16340, 16378, 16384,
        }
        | {
            5, 10, 15, 20, 40, 50, 80, 90, 110, 115, 120, 125, 140, 160, 190, 200, 240, 275, 280,
            320, 360, 370, 385, 400, 440, 455, 480, 520, 560, 580, 585, 600, 640, 650, 665, 710,
            715, 720, 960, 1280,
        }
        | {4095, 8191, 8193, 16383, 16385, 32767, 32768}
    )
)  # fmt: skip


def _spec(T: int, H: int = _H_CKPT, out: str = "none", flat: bool = False) -> dict:
    """`out`: "none" lets the op allocate, "alias" passes `out=residual` (the target's form),
    "view" passes a row-prefix view of a longer buffer. `flat` passes `post_mix` as [T, 4, 1]
    and `comb_mix` as [T, 16]."""
    return dict(T=T, H=H, out=out, flat=flat)


class MhcPostMapping(OpWrapper):
    """The hyper-connection post step (hc_post) for a 4-copy residual stream.

    Expands a sublayer's output back to the 4 copies and mixes the old copies in through the
    per-token `comb` matrix::

        out[t, j, h] = post_mix[t, j] * x[t, h] + sum_{k=0..3} comb_mix[t, k, j] * residual[t, k, h]

    `comb_mix[t, k, j]` is [source copy k][destination copy j]: the residual is mixed through
    `comb` transposed relative to `comb @ residual`, exactly the orientation of the DeepSeek-V4.1
    checkpoint's `Block.hc_post`, whose Sinkhorn output passes unchanged. Per element the kernel
    computes `acc = post[j] * x` (one fp32 multiply), then for k = 0..3 in order
    `acc = fma(comb[k, j], residual[k], acc)`, then one RNE rounding to bf16.

    Fusion boundary: the expansion, the comb mix, the sum and the bf16 cast -- nothing else. The
    caller owns computing `post_mix` / `comb_mix` (the hc mixes GEMM and the Sinkhorn step),
    producing `x` in bf16 (an fp32 all-reduce result must be cast first), and collapsing the new
    stream for the next sublayer. Writes `out` (a fresh contiguous `[T, 4, H]` bf16 tensor when
    None) and returns it; `out` may be `residual` itself.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(why=f"checkpoint width T={t}: H 5120, op-allocated out", spec=_spec(t))
            for t in (1, 5, 6, 32, 4096, 8192)
        )
        + tuple(
            Cell(why=f"width H={h} T=32: op-allocated out", spec=_spec(32, H=h))
            for h in (1032, 4096, 7168)
        )
        + (
            Cell(
                why="flat mixes T=16: post [T, 4, 1], comb [T, 16]",
                spec=_spec(16, flat=True),
            ),
            Cell(
                why="out view T=32: row prefix of a longer buffer",
                spec=_spec(32, out="view"),
            ),
        )
        + tuple(
            Cell(why=f"in place T={t}: out is residual, H 5120", spec=_spec(t, out="alias"))
            for t in (1, 32, 4096)
        )
        + tuple(
            Cell(
                why=f"target call T={t}: aliased, H 5120, sink row",
                spec=_spec(t, out="alias"),
            )
            for t in _TARGET_ROWS
        )
    )

    note = """
    Every cell carries a doubly-stochastic comb built the way the checkpoint builds it and
    (T > 1) a 1e5-scale attention-sink row (the checkpoint's BOS stream reaches |x| ~ 3.5e5).
    The other widths cover a whole number of 2048-element sweeps (4096), another model's width
    (7168), and a width that leaves most threads idle on its last sweep (1032 = 8 * 129).

    The op is bit-exact to its own FMA order in every cell. Against the checkpoint's expression
    (products rounded separately, then summed) it rounds differently on about 5 in 100,000
    elements, and default bf16 `assert_close` against that expression is not a safe gate where
    the sink row cancels to a small output: it rejects 1 element in 8 of 254 target-call and
    capture cases (max absolute difference 1.17e-2, relative 0.185, all in the sink row). Every
    element is within half a bf16 ulp of the exact sum plus 16 fp32 ulps of the term
    magnitudes (max 0.996 of that bound).

    Rows are independent: a NaN / inf in one row's `post_mix`, `comb_mix`, `x` or `residual`
    leaves every other row bit-identical, so CUDA-graph padding rows cannot contaminate real
    rows. Real-data check on the checkpoint's golden hc_post inputs (BOS rows to 3.5e5): the
    golden stream is reproduced bit-exactly on all but about 2 in 100,000 elements; with comb
    transposed only 8% are.

    Metadata consumed: none. Stateless; capturable in a CUDA graph.

    Preconditions: hc width exactly 4 (compiled in). H % 8 == 0 and 16-byte-aligned base
    pointers (8-element vector accesses); a violation is loud but fatal to the CUDA context
    (H = 12 died with "misaligned address") and is not checked. Dtypes are fixed (bf16
    `residual`, `x`, `out`; fp32 `post_mix`, `comb_mix`); a wrong dtype raises RuntimeError
    before launch. The op reads every tensor as a dense buffer of the size T and H imply; the
    layouts it silently miscomputes or writes out of bounds for are in `is_valid`. `out` may
    alias `residual` exactly (the kernel declares its pointers `__restrict__`; exact aliasing is
    a property of the compiled kernel the test re-checks, not one its source promises); partial
    overlap of `out` with any input was not tested. T = 0 returns an empty `[0, 4, H]` tensor.

    DeepSeek-V4.1 calls `(h, sublayer_out, post, comb, out=h)` twice per layer at H = 5120.
    Certified for every T in 1..32768 of that aliased call: each prefix of a 32768-row call is
    bitwise equal to its first T rows; the call launches exactly one `mhcPostMappingKernel`,
    compiled ahead of time, at every T; captured into a CUDA graph at every decode /
    DSpark-verify / drafter row count up to 1536 it replays bitwise equal to eager.

    Alternative measured on sm_103: `flashinfer.mhc.mhc_post` is bit-identical; this op is
    within 0.2 us at decode sizes, faster at prefill sizes (59.0 vs 70.9 us at T = 4096, 112.6
    vs 136.6 us at T = 8192), and about 3x faster eager.

    Not certified: T > 32768; H other than 5120 (every T) and 1032, 4096, 7168 (T = 32); partial
    aliasing of `out`; torch.compile.
    """

    def raw_call(
        self,
        residual: torch.Tensor,
        x: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_tokens, _, hidden_size = residual.shape
        if out is None:
            out = torch.empty_like(residual, memory_format=torch.contiguous_format)
        torch.ops.trtllm.mhc_post_mapping(
            residual, x, post_mix, comb_mix, out, num_tokens, hidden_size
        )
        return out

    def is_valid(
        self,
        residual: torch.Tensor,
        x: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> None:
        # The op takes T and H as integers and reads every tensor as a dense
        # row-major buffer of the size those imply: a non-contiguous view, a wrong
        # hc width, or a short tensor is read (or, for `out`, written) past its
        # bounds with no error. Each of these was measured to return silently wrong
        # results on sm_103, so they are checked here.
        assert residual.dim() == 3 and residual.shape[1] == 4, "residual must be [T, 4, H]"
        num_tokens, _, hidden_size = residual.shape
        assert x.shape == (num_tokens, hidden_size), "x must be [T, H] matching residual"
        assert post_mix.numel() == num_tokens * 4, (
            "post_mix must hold T*4 values ([T, 4] or [T, 4, 1])"
        )
        assert comb_mix.numel() == num_tokens * 16, (
            "comb_mix must hold T*16 values ([T, 4, 4] or [T, 16])"
        )
        # An absent `out` is allocated contiguous and shaped like `residual` by the call.
        if out is not None:
            assert out.shape == residual.shape, "out must be [T, 4, H] like residual"
        for name, t in (
            ("residual", residual),
            ("x", x),
            ("post_mix", post_mix),
            ("comb_mix", comb_mix),
            ("out", out),
        ):
            if t is not None:
                assert t.is_contiguous(), f"{name} must be contiguous"

    def reference(
        self,
        residual: torch.Tensor,
        x: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(out, exact, terms)`, each `[T, 4, H]`.

        `out` is the kernel's contracted order in native torch: `acc = fp32(post[j] * x)`, then
        for k = 0..3 `acc = fp32(comb[k, j] * r[k] + acc)` with the product and the add rounded
        once (the fp64 sum stands in for the fused multiply-add: the fp32 x fp32 product is
        exact in fp64), then one RNE rounding to bf16. `exact` is the five-term sum in fp64 and
        `terms` is `|post[j] * x| + sum_k |comb[k, j] * residual[k]|` in fp64, which the gate in
        `compare` needs. `out` (the argument) is ignored: the expected values do not depend on
        where they are written. Reads `residual`, so when the call aliases `out` to it, take
        the reference before the call.
        """
        t = residual.shape[0]
        post = post_mix.reshape(t, _HC).float()
        comb = comb_mix.reshape(t, _HC, _HC).float()
        acc = post.unsqueeze(-1) * x.float().unsqueeze(-2)
        terms = post.double().unsqueeze(-1) * x.double().unsqueeze(-2)
        exact = terms.clone()
        term_scale = terms.abs()
        for k in range(_HC):
            prod = comb[:, k, :].double().unsqueeze(-1) * residual[:, k, :].double().unsqueeze(1)
            acc = (prod + acc.double()).float()
            exact += prod
            term_scale += prod.abs()
        return acc.to(torch.bfloat16), exact, term_scale

    def compare(
        self,
        out: torch.Tensor,
        ref: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Equal to the kernel-order reference on every element, and within the term bound.

        The op matched its own FMA order on every element of every certified cell, so nothing
        is left to round differently. The term bound is half a bf16 ulp of the exact value (the
        final rounding) plus 16 fp32 ulps of S = |post[j] * x| + sum_k |comb[k, j] * r[k]|:
        any fp32 order errs by a few fp32 ulps of the TERMS, which on a cancelling sink row is
        large next to the result. Max 0.996 of the bound measured on sm_103. Controls: comb
        applied untransposed, and `post` rolled by one copy, both fail.
        """
        expected, exact, terms = ref
        assert out.dtype == expected.dtype == torch.bfloat16, (out.dtype, expected.dtype)
        assert out.shape == expected.shape, (out.shape, expected.shape)
        bad = (out != expected).sum().item()
        assert bad == 0, f"{bad} of {out.numel()} elements differ from the kernel-order reference"
        if out.numel():
            bound = 2.0**-8 * exact.abs() + 16 * 2.0**-24 * terms + torch.finfo(torch.float64).tiny
            ratio = ((out.double() - exact).abs() / bound).max().item()
            assert ratio <= 1.0, f"max err / bound against the exact sum = {ratio:.3f}"


mhc_post_mapping = MhcPostMapping()
