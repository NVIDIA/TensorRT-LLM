# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection mix coefficients via the trtllm mhc_big_fuse op.

From the raw mixes and square sums of the hc mixes GEMM: RMS-scaled sigmoid `post`, a
Sinkhorn-normalized `comb`, and (when `hidden_size > 0`) the stream collapsed with the FRESH
`pre`. DeepSeek-V4.1 lags `pre` by one sublayer, so it consumes only `post_mix` and `comb_mix`.

The op is `torch.ops.trtllm.mhc_big_fuse`, registered in `cpp/tensorrt_llm/thop/mhcOp.cpp`.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Residual-stream copies; compiled into the kernel.
_HC = 4
#: Columns of a mixes row: pre 0:4 | post 4:8 | comb 8:24.
_MIX_COLS = _HC * (2 + _HC)

#: Row counts DeepSeek-V4.1 makes its call at, from the target's runtime-value dumps: the trunk's
#: ('mhc', T, tile) records (T 1..16384, engines up to max_batch_size 256 / max_num_tokens 16384,
#: CUDA-graph and eager, dep4 and tp4, 128K and 1M max_seq_len, the long-context envelope) and
#: the DSpark drafter's ('dspark.mhc', N, tile) records (N = 5G); plus 17 and the tops of the
#: step range (16384) and of the decoder bounded-replay pass (32768).
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
        | {17, 4095, 8191, 8193, 16383, 16385, 32767, 32768}
    )
)  # fmt: skip


def _spec(
    M: int,
    hidden_size: int = 5120,
    num_splits: int = 1,
    block_size: int = 0,
    zero_rows: int = 0,
    sinkhorn_repeat: int = 20,
    hc_post_mult_value: float = 2.0,
    kind: str = "stream",
) -> dict:
    """One cell's spec. `kind` "stream" builds the mixes from a [M, 4, 5120] stream the way the
    mixes GEMM does; "extreme" feeds logits of magnitude ~1e3-1e4 at unit RMS."""
    return dict(
        kind=kind,
        M=M,
        hidden_size=hidden_size,
        num_splits=num_splits,
        block_size=block_size,
        zero_rows=zero_rows,
        sinkhorn_repeat=sinkhorn_repeat,
        hc_post_mult_value=hc_post_mult_value,
    )


def layer_input_error_ratio(
    layer_input: torch.Tensor, exact: torch.Tensor, terms: torch.Tensor
) -> float:
    """max(|layer_input - exact| / bound) under the `layer_input` gate (see `MhcBigFuse.compare`)."""
    bound = 2.0**-7 * exact.abs() + 16 * 2.0**-24 * terms
    bound = (
        bound + torch.finfo(torch.float64).tiny
    )  # an all-zero padding row must come out exactly 0
    return ((layer_input.double() - exact).abs() / bound).max().item()


class MhcBigFuse(OpWrapper):
    """The hyper-connection coefficient step for a 4-copy residual stream, one CTA per token.

    From the hc mixes GEMM's raw mixes `y[t, 0:24]` and square sum `r[t]` (optionally split-K
    partials), for rows `t < M`::

        m       = y * rsqrt(r / K + rms_eps)                 # pre 0:4 | post 4:8 | comb 8:24
        pre     = sigmoid(m[0:4] * hc_scale[0] + hc_base[0:4]) + hc_pre_eps        (not returned)
        post    = sigmoid(m[4:8] * hc_scale[1] + hc_base[4:8]) * hc_post_mult_value -> post_mix
        comb    = Sinkhorn(m[8:24] * hc_scale[2] + hc_base[8:24]) as [4, 4]         -> comb_mix
        layer_input = bf16(sum_j pre[j] * residual[t, j, :])   only when hidden_size > 0

    Sinkhorn is a row softmax plus `hc_sinkhorn_eps`, a column pass, then `sinkhorn_repeat - 1`
    (row, column) passes, every denominator `+ hc_sinkhorn_eps`.

    Fusion boundary: inside the call are the split-K reduction, the RMS scaling, both sigmoids,
    the whole Sinkhorn normalization and (when `hidden_size > 0`) the collapse with the FRESH
    `pre`. The caller owns the mixes GEMM and square sum that produce `y_acc` / `r_acc`, the
    lagged `pre` collapse DeepSeek-V4.1 actually feeds its sublayers, the RMSNorm of the sublayer
    input, and hc_post. Writes rows `[0, M)` of `post_mix`, `comb_mix` and (when
    `hidden_size > 0`) `layer_input` in place and returns `(post_mix, comb_mix, layer_input)`.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"full call M={m}: V4.1 constants"
                + (", one all-zero padding row" if m > 1 else ""),
                spec=_spec(m, zero_rows=1 if m > 1 else 0),
            )
            for m in (1, 5, 6, 64, 65, 1024, 8192)
        )
        + tuple(
            Cell(
                why=f"block_size={bs} M={m}: full-call launch shape",
                spec=_spec(m, block_size=bs),
            )
            for m in (1, 64, 65, 300)
            for bs in (0, 128, 256, 512)
        )
        + tuple(
            Cell(
                why=f"post/comb-only block_size={bs} M={m}: hidden_size 0",
                spec=_spec(m, hidden_size=0, block_size=bs),
            )
            for m in (1, 6, 4096)
            for bs in (0, 128, 256)
        )
        + tuple(
            Cell(
                why=f"num_splits={s} M=33: split-K partials summed in the kernel",
                spec=_spec(33, num_splits=s),
            )
            for s in (1, 2, 4, 8, 16)
        )
        + tuple(
            Cell(
                why=f"sinkhorn_repeat={n} hc_post_mult={p}: M=48 full call",
                spec=_spec(48, sinkhorn_repeat=n, hc_post_mult_value=p),
            )
            for n in (1, 20)
            for p in (1.0, 2.0)
        )
        + (
            Cell(
                why="logits ~1e3-1e4 M=64: softmax and sigmoids stay finite",
                spec=_spec(64, kind="extreme"),
            ),
        )
        + tuple(
            Cell(
                why=f"target call M={m}: hidden_size 0, block_size 128, one split",
                spec=_spec(m, hidden_size=0, block_size=128, zero_rows=1 if m > 2 else 0),
            )
            for m in _TARGET_ROWS
        )
    )

    note = """
    DeepSeek-V4.1 constants: rms_eps 1e-20, hc_pre_eps = hc_sinkhorn_eps = 1e-6,
    hc_post_mult_value 2.0, sinkhorn_repeat 20, K = 4 * 5120 = 20480. With them `post` and
    `comb` are exactly the checkpoint's `hc_split_sinkhorn` (same Sinkhorn order, same eps
    placement). `comb_mix[t, j, k]` is row j (source copy), column k (destination copy); the
    softmax runs along the row, and V4.1's hc_post sums over j, consuming the tensor unchanged.

    `layer_input` is NOT V4.1's hc_pre: the kernel collapses with the `pre` of this same call,
    while V4.1 lags `pre` by one sublayer. Measured on golden layer dumps it is off by relative
    L2 5.3e-2 to 9.0 from the lagged collapse. V4.1 therefore calls with hidden_size 0,
    block_size 128, num_splits 1 and consumes only `post_mix` / `comb_mix`; the caller computes
    the lagged collapse from the previous sublayer's saved `y[:, 0:4]` and `r`, which this call
    does not return. Post/comb-only mode (hidden_size 0) neither reads `residual` nor writes
    `layer_input` (both may be zero-size, `[M, 4, 0]` / `[M, 0]`); `post_mix` / `comb_mix` are
    bit-identical to a hidden_size 5120 call. `K` stays the flattened stream width in that mode.

    Metadata consumed: none. Stateless: no workspace, no autotuner.

    Preconditions: fp32 `y_acc`, `r_acc`, `hc_scale`, `hc_base`, `post_mix`, `comb_mix`; bf16
    `residual`, `layer_input` (a bf16 `y_acc` and an fp32 `residual` raise RuntimeError, the
    other dtypes were not probed). num_splits in {1, 2, 4, 8, 16}, else the op raises
    RuntimeError. With num_splits > 1 split s of token t is read at flat row s * M + t, so
    `y_acc` / `r_acc` must be exactly `[num_splits, M, ...]`. Shapes are trusted beyond dtype;
    hidden_size is a multiple of 8. `sinkhorn_repeat <= 0` computes exactly what 1 does.
    An all-zero token row (a CUDA-graph padding row) gives finite outputs and a zero
    `layer_input` row. Rows at and beyond M are never written; M = 0 touches nothing.

    block_size: 0 or negative = auto (128 for M <= 16, else 256); 1-255 -> 128, 256-511 -> 256,
    >= 512 -> 512; all bit-identical. 512 with num_splits 1 and M <= 64 stages the residual
    through a bulk async copy; that path was not run in post/comb-only mode, use 0, 128 or 256
    there. The V4.1 call launches exactly one kernel at every M,
    `mhcBigFuseKernel<1, 128, false, false, false>`, compiled ahead of time. Measured on sm_103,
    post/comb-only at block_size 128 is the fastest or within 0.1 us of the fastest candidate at
    every M (11.3 us at M = 8192, against 69-75 us for any call that also collapses).

    Certified for every M in 1..32768 of the V4.1 call: each prefix of a 32768-row call is
    bitwise equal to its first M rows, and the call captured into a CUDA graph at every
    decode / DSpark-verify / drafter row count up to 1536 replays bitwise equal to eager.

    The registered op also takes `norm_weight`, `norm_eps`, `pre_mix_ext`, `pre_mix_out` (a
    fused next-layer RMSNorm and a lagged-`pre` collapse); this entry leaves them at their
    defaults and certifies nothing about them. The kernel is launched with PDL when
    TRTLLM_ENABLE_PDL is on. No fake/meta implementation; torch.compile not exercised.

    Not certified: hidden_size other than 5120 and 0; K other than 20480; block_size >= 512
    with hidden_size 0; num_splits > 1 at M other than 33 or with more than M rows per split;
    M > 32768; eps other than V4.1's; sinkhorn_repeat other than 1 and 20; hc_post_mult_value
    other than 1.0 and 2.0; non-contiguous `hc_scale` / `hc_base`.
    """

    def raw_call(
        self,
        y_acc: torch.Tensor,
        r_acc: torch.Tensor,
        residual: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        layer_input: torch.Tensor,
        M: int,
        K: int,
        hidden_size: int,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        num_splits: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.ops.trtllm.mhc_big_fuse(
            y_acc,
            r_acc,
            residual,
            hc_scale,
            hc_base,
            post_mix,
            comb_mix,
            layer_input,
            M,
            K,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            num_splits,
            block_size,
        )
        return post_mix, comb_mix, layer_input

    def is_valid(
        self,
        y_acc: torch.Tensor,
        r_acc: torch.Tensor,
        residual: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        layer_input: torch.Tensor,
        M: int,
        K: int,
        hidden_size: int,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        num_splits: int,
        block_size: int,
    ) -> None:
        # The kernel addresses every one of these as a dense row-major buffer. A strided view of any
        # of the six was accepted and read or wrote the wrong elements with no error.
        assert y_acc.is_contiguous() and r_acc.is_contiguous(), "y_acc and r_acc must be contiguous"
        assert residual.is_contiguous(), "residual must be contiguous [M, 4, hidden_size]"
        assert post_mix.is_contiguous() and comb_mix.is_contiguous(), (
            "post_mix and comb_mix must be contiguous"
        )
        assert layer_input.is_contiguous(), "layer_input must be contiguous"

    def reference(
        self,
        y_acc: torch.Tensor,
        r_acc: torch.Tensor,
        residual: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        layer_input: torch.Tensor,
        M: int,
        K: int,
        hidden_size: int,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        num_splits: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(post_mix, comb_mix, layer_input, layer_input_terms)` expected after the call.

        fp32 torch ops in the kernel's operation order for `pre` / `post` / `comb`; split partials
        are summed in split order, read at flat row `s * M + t` like the kernel. `layer_input` is
        returned as the EXACT fp64 collapse of that fp32 `pre` (not rounded to bf16: the gate in
        `compare` carries the final rounding), and `layer_input_terms` is `sum_j |pre_j * x_j|`
        in fp64, the scale of the kernel's fp32 work, which the gate needs. Each expected output
        is a copy of the passed buffer with rows `[0, M)` replaced, so rows the op must not
        write keep their contents; nothing passed in is mutated. `layer_input` (and
        `layer_input_terms`) rows are only replaced when `hidden_size > 0`. `block_size` is
        ignored (it picks the launch shape only); `residual` is only read when
        `hidden_size > 0`.
        """
        y_rows = y_acc.reshape(-1, _MIX_COLS).float()
        r_rows = r_acc.reshape(-1).float()
        y, r = y_rows[:M].clone(), r_rows[:M].clone()
        for s in range(1, num_splits):
            y += y_rows[s * M : (s + 1) * M]
            r += r_rows[s * M : (s + 1) * M]
        eps = hc_sinkhorn_eps
        mixes = y * torch.rsqrt(r / K + rms_eps)[:, None]
        pre = torch.sigmoid(mixes[:, :_HC] * hc_scale[0] + hc_base[:_HC]) + hc_pre_eps
        post = (
            torch.sigmoid(mixes[:, _HC : 2 * _HC] * hc_scale[1] + hc_base[_HC : 2 * _HC])
            * hc_post_mult_value
        )
        comb = (mixes[:, 2 * _HC :] * hc_scale[2] + hc_base[2 * _HC :]).view(-1, _HC, _HC)
        comb = comb.softmax(-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(sinkhorn_repeat - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)

        exp_post = post_mix.clone()
        exp_post.view(-1, _HC)[:M] = post
        exp_comb = comb_mix.clone()
        exp_comb.view(-1, _HC * _HC)[:M] = comb.reshape(-1, _HC * _HC)
        exp_li = layer_input.to(torch.float64, copy=True)
        terms_sum = torch.zeros_like(exp_li)
        if hidden_size > 0:
            x = residual.reshape(-1, _HC, hidden_size)[:M].double()
            terms = pre.double()[:, :, None] * x
            exp_li.view(-1, hidden_size)[:M] = terms.sum(1)
            terms_sum.view(-1, hidden_size)[:M] = terms.abs().sum(1)
        return exp_post, exp_comb, exp_li, terms_sum

    def compare(
        self,
        out: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ref: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """`post_mix` / `comb_mix` at fp32 `assert_close` defaults; `layer_input` per element.

        `post` / `comb`: the kernel is built with fast math (`expf`, reciprocals), which
        accounts for last-ulp differences; rtol 1.3e-6, atol 1e-5 held in every certified cell.

        `layer_input` against the exact fp64 collapse: one bf16 ulp of the exact value (the final
        rounding, which a few-ulp fp32 difference can push across a rounding midpoint) plus 16
        fp32 ulps of S = sum_j |pre_j * x_j|. The second term is the kernel's fp32 work: a 4-term
        FMA chain rounds each running sum (<= 4 ulps of S), and its fast-math sigmoid differs
        from torch's by a few ulps of each pre_j. A row that cancels to near zero keeps that
        absolute error, which a purely relative bf16 gate misreads: the default bf16
        `assert_close` failed on 2 of 41943040 such elements at M = 8192 (greatest relative
        difference 0.625). Observed max error / bound under this gate: 0.498. Controls at
        M = 256: a 19-pass Sinkhorn lands 350x the fp32 atol away, a transposed comb 97000x,
        and the lagged-`pre` collapse 1.2e6x the `layer_input` bound.
        """
        (post, comb, li), (r_post, r_comb, r_li, r_terms) = out, ref
        torch.testing.assert_close(post, r_post)
        torch.testing.assert_close(comb, r_comb)
        assert li.shape == r_li.shape, (li.shape, r_li.shape)
        if li.numel():
            worst = layer_input_error_ratio(li, r_li, r_terms)
            assert worst <= 1.0, f"layer_input outside the gate: max err/bound = {worst:.3f}"


mhc_big_fuse = MhcBigFuse()
