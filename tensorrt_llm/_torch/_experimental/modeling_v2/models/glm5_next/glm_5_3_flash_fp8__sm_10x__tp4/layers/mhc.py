# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One manifold hyper-connection (mHC) site of GLM-5.3-Flash: 4 residual streams around each
attention and FFN sublayer.

Before a sublayer the streams are mixed (``pre_mapping``): a ``[24, 4 * hidden]`` mixer over
the RMS-normalized flattened streams gives the sublayer's input weights ``pre``, the output
weights ``post`` and a Sinkhorn-normalized 4 x 4 stream combination ``comb``. After it the
sublayer's output is written back (``post_mapping``: ``streams' = comb^T streams + post *
out``). ``fused_hc`` is a sublayer's post-mapping fused with the next one's pre-mapping and the
next sublayer's input RMSNorm.

The parameters are the model's (``glm5_next_hyper_connection`` in the checkpoint's release):
the mixer's RMS epsilon is ``rms_norm_eps``, ``pre``'s epsilon and the Sinkhorn epsilon are
``hc_eps``, ``post``'s multiplier is 2 and the Sinkhorn runs ``hc_sinkhorn_iters`` rounds.
Note the two epsilons of ``fused_hc``: ``norm_eps`` here is the mixer's; the sublayer input's
RMSNorm epsilon is the caller's ``norm_eps`` argument (``layer_input_eps`` of the ops).
"""

from __future__ import annotations

from typing import Optional

import torch

from ..dims import Dims
from ..ops.mhc.mhc_big_fuse import big_fuse_block_size, mhc_big_fuse
from ..ops.mhc.mhc_fused_hc import fused_hc_scratch, fused_hc_tactic, mhc_fused_hc
from ..ops.mhc.mhc_gemm_sqrsum import gemm_sqrsum_tile_n, mhc_gemm_sqrsum
from ..ops.mhc.mhc_post_mapping import mhc_post_mapping

#: post's multiplier: post = sigmoid(.) * 2.
_POST_MULT = 2.0


class Mhc:
    """One mHC site's weights and the three operations on them.

    ``post`` travels as ``[rows, mult]`` FP32 and ``comb`` as ``[rows, mult, mult]`` FP32; the
    streams are ``[rows, mult, hidden]`` BF16.
    """

    def __init__(self, w, dims: Dims) -> None:
        self.fn = w.fn  # [24, mult * hidden] fp32
        self.scale = w.scale  # [3] fp32
        self.base = w.base  # [24] fp32
        self.mult = dims.hc_mult
        self.sinkhorn_iters = dims.hc_sinkhorn_iters
        self.norm_eps = dims.rms_eps  # the mixer's RMS epsilon
        self.eps = dims.hc_eps
        self.sinkhorn_eps = dims.hc_eps
        self.post_mult_value = _POST_MULT

    def _scalars(self, layer_input_eps: float) -> tuple:
        return (
            self.sinkhorn_iters,
            self.norm_eps,
            self.eps,
            self.sinkhorn_eps,
            self.post_mult_value,
            layer_input_eps,
        )

    def pre_mapping(self, residual: torch.Tensor):
        """``(post, comb, x)``: the first sublayer's stream mix, ``x`` not yet RMS-normalized."""
        rows = residual.shape[0]
        mixes, sqrsum = mhc_gemm_sqrsum(residual.view(rows, -1), self.fn, gemm_sqrsum_tile_n(rows))
        return mhc_big_fuse(
            mixes,
            sqrsum,
            residual,
            self.scale,
            self.base,
            None,
            *self._scalars(0.0),
            big_fuse_block_size(rows),
        )

    def fused_hc(
        self,
        out: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ):
        """``(residual, post, comb, x)``: ``out``'s post-mapping into the streams, then the next
        sublayer's pre-mapping and its input RMSNorm (``norm_weight``, ``norm_eps``)."""
        args = (
            out,
            residual,
            post,
            comb,
            self.fn,
            self.scale,
            self.base,
            norm_weight,
            *self._scalars(norm_eps),
        )
        tactic = fused_hc_tactic(*args)
        return mhc_fused_hc(*args, tactic, *fused_hc_scratch(out.shape[0], tactic, out.device))

    def post_mapping(
        self,
        out: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        norm_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The streams after ``out``'s post-mapping (the last sublayer's)."""
        del norm_weight
        return mhc_post_mapping(out, residual, post, comb)
