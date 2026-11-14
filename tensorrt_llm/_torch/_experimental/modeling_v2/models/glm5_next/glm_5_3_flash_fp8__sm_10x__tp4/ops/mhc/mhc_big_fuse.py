# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The mHC pre-mapping's coefficients and layer input, from the mixer product (``mhc_big_fuse``).

The second of the two kernels of an unfused pre-mapping (``mHC.pre_mapping``):
``torch.ops.trtllm.mhc_big_fuse`` (``mhcBigFuseKernel``, one CTA per row). From ``mhc_gemm_sqrsum``'s
raw mixer product ``mixes [M, 24]`` and row sum of squares ``sqrsum [M]`` of the streams
``residual [M, 4, hidden]``, with ``K = 4 * hidden``::

    rstd = rsqrt(sqrsum[:, None] / K + norm_eps)  # the RMS of the flattened streams
    pre = sigmoid(mixes[:, 0:4] * rstd * scale[0] + base[0:4]) + hc_eps
    post = sigmoid(mixes[:, 4:8] * rstd * scale[1] + base[4:8]) * post_mult  # [M, 4]
    logits = (mixes[:, 8:24] * rstd * scale[2] + base[8:24]).view(M, 4, 4)
    comb = sinkhorn(logits, sinkhorn_iters, sinkhorn_eps)  # [M, 4, 4]
    layer_input = bf16((pre[:, :, None] * residual).sum(1))  # [M, hidden]
    if norm_weight is not None:  # the sublayer's input RMSNorm, fused
        rms = rsqrt(mean(layer_input**2) + layer_input_eps)
        layer_input = bf16(layer_input * rms * norm_weight)

``comb`` is ``mixes[:, 8:24]`` read row-major as ``[source stream, destination stream]``.
``sinkhorn`` takes a softmax over each row, adds ``sinkhorn_eps`` and normalizes the columns, then
runs ``sinkhorn_iters - 1`` pairs of row and column normalizations; every normalization divides by
the sum plus ``sinkhorn_eps``. It ends on a column normalization, so each destination stream gets a
convex combination of the sources (up to ``sinkhorn_eps``) and the rows sum to one only
approximately.

Warp 0 computes the coefficients (lanes 0 to 3, one comb row each); the other warps collapse the
streams into the layer input, eight values per thread and step, in FP32 with ``j = 0..3`` as fused
multiply-adds, rounded to BF16. The fused RMSNorm then squares that BF16 layer input, reduces over
the row and rounds the normalized product to BF16 again: the bits a separate RMSNorm of the
unfused pre-mapping's BF16 output would see. The kernels are compiled with fast math (approximate
``exp``, ``rsqrt`` and division).
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: GLM-5.3-Flash's hyper-connection: 4 streams of 4096, 24 mixer outputs.
MULT = 4
HIDDEN = 4096
MIX = (2 + MULT) * MULT

#: The CTA sizes the kernel is instantiated for; 512 at most 64 rows also bulk-copies the row's
#: streams into shared memory first.
BLOCK_SIZE_OPTIONS = (128, 256, 512)

#: GLM-5.3-Flash's mHC scalars, as the config sets them: ``hc_sinkhorn_iters``; ``rms_norm_eps`` for
#: the mixer's RMS and for the sublayer input RMSNorm; ``hc_eps`` for ``pre`` and the Sinkhorn
#: normalization; ``post`` weights of ``2 * sigmoid``.
GLM_SCALARS = dict(
    sinkhorn_iters=20,
    norm_eps=1e-5,
    hc_eps=1e-6,
    sinkhorn_eps=1e-6,
    post_mult=2.0,
    layer_input_eps=1e-5,
)


def big_fuse_block_size(rows: int) -> int:
    """The kernel's own default CTA size for ``rows`` rows (``selectBigFuseBlockSize``).

    128 threads up to 16 rows, 256 above. It is what ``mHC.pre_mapping`` runs when the autotuner
    has no entry for the step.
    """
    return 128 if rows <= 16 else 256


def sinkhorn(logits: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """Sinkhorn normalization of ``[..., 4, 4]`` logits as the mHC kernels define it.

    A softmax over the last dim plus ``eps``, a column normalization, then ``iters - 1`` row and
    column normalization pairs, each dividing by the sum plus ``eps``.
    """
    comb = logits.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return comb


class MhcBigFuse(OpWrapper):
    """A sublayer's mHC coefficients and its (optionally RMS-normalized) layer input.

    Args:
        mixes: ``[M, 24]`` FP32, contiguous: ``mhc_gemm_sqrsum``'s mixer product of ``residual``.
        sqrsum: ``[M]`` FP32, contiguous: the row sum of squares of the flattened ``residual``.
        residual: ``[M, 4, hidden]`` BF16, contiguous: the streams the sublayer reads.
        scale: ``[3]`` FP32, contiguous: the sublayer's ``hc_*_scale`` (pre, post, comb).
        base: ``[24]`` FP32, contiguous: the sublayer's ``hc_*_base``.
        norm_weight: ``[hidden]`` BF16, contiguous, the sublayer's input RMSNorm weight to fuse; or
            None to return the collapsed streams unnormalized.
        sinkhorn_iters: Sinkhorn iterations, the initial column normalization included (20).
        norm_eps: epsilon of the mixer's RMS over the flattened streams (``rms_norm_eps``).
        hc_eps: added to ``pre`` (``hc_eps``).
        sinkhorn_eps: Sinkhorn epsilon (``hc_eps``).
        post_mult: multiplier of the ``post`` sigmoid (2.0).
        layer_input_eps: epsilon of the fused RMSNorm (``rms_norm_eps``); ignored without
            ``norm_weight``.
        block_size: threads per CTA, one of ``BLOCK_SIZE_OPTIONS``; ``big_fuse_block_size(M)`` is
            the kernel's default. It changes the layer input's norm reduction order only.

    Returns:
        ``(post, comb, layer_input)``: new ``[M, 4]`` and ``[M, 4, 4]`` FP32 and ``[M, hidden]``
        BF16 tensors -- what ``mHC.pre_mapping`` returns (``post`` without its trailing unit
        dimension), followed by the input RMSNorm when ``norm_weight`` is given.
    """

    ARCHS = frozenset({Arch.SM_100})

    _GLM = dict(mult=MULT, hidden=HIDDEN, **GLM_SCALARS)
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode of one request without MTP, the input RMSNorm fused",
            spec=dict(rows=1, norm=True, **_GLM),
        ),
        Cell(why="MTP-3 verify of one request (4 rows)", spec=dict(rows=4, norm=True, **_GLM)),
        Cell(why="MTP-3 verify of two requests (8 rows)", spec=dict(rows=8, norm=True, **_GLM)),
        Cell(
            why="MTP-3 verify of 8 requests (32 rows): the default CTA is 256 threads from 17 rows",
            spec=dict(rows=32, norm=True, **_GLM),
        ),
        Cell(
            why="MTP-3 verify of 64 requests (256 rows): no bulk copy, which stops at 64 rows",
            spec=dict(rows=256, norm=True, **_GLM),
        ),
        Cell(
            why="a quarter of a 13,312-token prompt: one rank's rows under sequence parallelism",
            spec=dict(rows=3328, norm=True, **_GLM),
        ),
        Cell(
            why="MTP-3 verify of one request without the fused norm: mHC.pre_mapping's output",
            spec=dict(rows=4, norm=False, **_GLM),
        ),
        Cell(
            why="a quarter of a 13,312-token prompt without the fused norm",
            spec=dict(rows=3328, norm=False, **_GLM),
        ),
    )

    note = """
    The kernel is compiled for 4 streams. This entry runs it with one split of the mixer product
    (`mhc_gemm_sqrsum`'s output); the split-K partial sums the fused boundary feeds it stay inside
    `mhc_fused_hc`. The lagged pre-mix variant (`pre_mix_ext` / `pre_mix_out`) is not used. Launched
    with programmatic dependent launch when `TRTLLM_ENABLE_PDL` allows it; no rows launch nothing.
    `post`, `comb` and, without the fused norm, `layer_input` are the same bits at every block
    size. Outputs are allocated per call, from the graph's private pool under CUDA graph capture.
    """

    def __call__(
        self,
        mixes: torch.Tensor,
        sqrsum: torch.Tensor,
        residual: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm_weight: Optional[torch.Tensor],
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
        sinkhorn_eps: float,
        post_mult: float,
        layer_input_eps: float,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, hidden = residual.shape[0], residual.shape[-1]
        device = residual.device
        post = torch.empty((rows, MULT), dtype=torch.float32, device=device)
        comb = torch.empty((rows, MULT, MULT), dtype=torch.float32, device=device)
        layer_input = torch.empty((rows, hidden), dtype=torch.bfloat16, device=device)
        torch.ops.trtllm.mhc_big_fuse(
            mixes,
            sqrsum,
            residual,
            scale,
            base,
            post,
            comb,
            layer_input,
            rows,
            MULT * hidden,
            hidden,
            norm_eps,
            hc_eps,
            sinkhorn_eps,
            post_mult,
            sinkhorn_iters,
            1,
            block_size,
            norm_weight,
            layer_input_eps,
            None,
            None,
        )
        return post, comb, layer_input

    def reference(
        self,
        mixes: torch.Tensor,
        sqrsum: torch.Tensor,
        residual: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm_weight: Optional[torch.Tensor],
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
        sinkhorn_eps: float,
        post_mult: float,
        layer_input_eps: float,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The same outputs in FP32 torch. ``block_size`` is ignored.

        Rounded to BF16 only where the unfused pre-mapping stores a tensor: the collapsed layer
        input, which a separate RMSNorm reads, and the normalized result.
        """
        rows, hidden = residual.shape[0], residual.shape[-1]
        rstd = torch.rsqrt(sqrsum.float().reshape(rows, 1) / (MULT * hidden) + norm_eps)
        scaled = mixes.float().reshape(rows, MIX) * rstd
        s, b = scale.float().reshape(-1), base.float().reshape(-1)
        pre = torch.sigmoid(scaled[:, :MULT] * s[0] + b[:MULT]) + hc_eps
        post = torch.sigmoid(scaled[:, MULT : 2 * MULT] * s[1] + b[MULT : 2 * MULT]) * post_mult
        logits = (scaled[:, 2 * MULT :] * s[2] + b[2 * MULT : MIX]).reshape(rows, MULT, MULT)
        comb = sinkhorn(logits, sinkhorn_iters, sinkhorn_eps)
        streams = residual.float().reshape(rows, MULT, hidden)
        collapsed = pre[:, :1] * streams[:, 0]
        for j in range(1, MULT):
            collapsed += pre[:, j : j + 1] * streams[:, j]
        layer_input = collapsed.to(torch.bfloat16)
        if norm_weight is not None:
            li = layer_input.float()
            li = li * torch.rsqrt(li.square().mean(-1, keepdim=True) + layer_input_eps)
            layer_input = (li * norm_weight.float()).to(torch.bfloat16)
        return post, comb, layer_input

    def is_valid(
        self,
        mixes: torch.Tensor,
        sqrsum: torch.Tensor,
        residual: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm_weight: Optional[torch.Tensor],
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
        sinkhorn_eps: float,
        post_mult: float,
        layer_input_eps: float,
        block_size: int,
    ) -> None:
        assert residual.dim() == 3 and residual.shape[1] == MULT, (
            f"the kernel is compiled for {MULT} streams: residual is [M, {MULT}, hidden]; "
            f"got {tuple(residual.shape)}"
        )
        rows, hidden = residual.shape[0], residual.shape[2]
        assert hidden % 8 == 0, "each thread moves 8 BF16 values per step; the kernel has no tail"
        assert mixes.shape == (rows, MIX) and sqrsum.shape == (rows,), (
            f"one split: mixes [{rows}, {MIX}] and sqrsum [{rows}]; "
            f"got {tuple(mixes.shape)} and {tuple(sqrsum.shape)}"
        )
        assert scale.numel() >= 3 and base.numel() >= MIX, "scale holds 3 values and base 24"
        for name, t in (
            ("mixes", mixes),
            ("sqrsum", sqrsum),
            ("residual", residual),
            ("scale", scale),
            ("base", base),
        ):
            assert t.is_contiguous(), f"{name} is read with dense row-major strides"


mhc_big_fuse = MhcBigFuse()
