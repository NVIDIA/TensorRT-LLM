# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Multi-Head Hyper-Connection (mHC) module
# Based on: "Hyper-Connections" (https://arxiv.org/abs/2409.19606)
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class HCState:
    """Inter-layer mHC pipeline state.

    Two modes, distinguished by whether ``post_mix`` is populated:

    - **resolved** (``post_mix is None``): ``residual`` is the fully post-mapped
      activation for the next layer's ``pre_mapping``. This is what the prior
      layer returns when fused_hc is disabled (or engram mutated the residual).
      The next layer just runs ``pre_mapping(residual)``.
    - **deferred** (``post_mix is not None``): the prior layer deferred its
      ``post_mapping``; the 4 tensors carry the inputs needed for the next
      layer to absorb it via ``fused_hc``.

    Only ``modeling_deepseekv4.py`` depends on this shape — the kernel-level
    ``mHC.fused_hc`` still returns a 4-tuple so low-level callers (tests,
    benchmarks) stay unchanged.
    """

    residual: torch.Tensor
    post_mix: Optional[torch.Tensor] = None
    comb_mix: Optional[torch.Tensor] = None
    x_prev: Optional[torch.Tensor] = None
    # V4.1 only: the *previous* sublayer's `pre` coefficients, which this
    # sublayer collapses with (`pre_mapping_lagged`). V4 leaves it None, so
    # `is_deferred` and both constructors behave exactly as before.
    pre_mix: Optional[torch.Tensor] = None

    @property
    def is_deferred(self) -> bool:
        return self.post_mix is not None

    @classmethod
    def resolved(cls, residual: torch.Tensor, pre_mix: Optional[torch.Tensor] = None) -> "HCState":
        return cls(residual=residual, pre_mix=pre_mix)

    @classmethod
    def deferred(
        cls,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        x_prev: torch.Tensor,
        pre_mix: Optional[torch.Tensor] = None,
    ) -> "HCState":
        return cls(
            residual=residual,
            post_mix=post_mix,
            comb_mix=comb_mix,
            x_prev=x_prev,
            pre_mix=pre_mix,
        )


try:
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import mhc_fused_hc as mhc_fused_hc_cuda
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import (
        mhc_gemm_rms_fma_cuda,
        mhc_hc_head_cuda,
        mhc_post_mapping_cuda,
    )
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import (
        mhc_pre_mapping_fused as mhc_pre_mapping_fused_cuda,
    )

    _cuda_available = True
except Exception as _e:
    _cuda_available = False
    mhc_hc_head_cuda = None
    mhc_post_mapping_cuda = None
    mhc_pre_mapping_fused_cuda = None
    mhc_fused_hc_cuda = None
    mhc_gemm_rms_fma_cuda = None


class mHC(nn.Module):
    def __init__(
        self,
        mult: int,
        hidden_size: int,
        sinkhorn_iters: int,
        dtype: Optional[torch.dtype] = None,
        eps: float = 1e-6,
        norm_eps: float = 1e-6,
        sinkhorn_eps: float = 1e-6,
        post_mult_value: float = 1.0,
        n_splits: int = 1,
    ):
        super().__init__()
        self.mult = mult
        self.hidden_size = hidden_size
        self.sinkhorn_iters = sinkhorn_iters
        self.dtype = dtype
        self.eps = eps
        self.norm_eps = norm_eps
        self.sinkhorn_eps = sinkhorn_eps
        self.post_mult_value = post_mult_value
        self.n_splits = n_splits
        self.mix_hc = (2 + self.mult) * self.mult
        self.hc_dim = self.mult * self.hidden_size

        # Parameters
        self.fn = nn.Parameter(
            torch.empty((self.mix_hc, self.hc_dim), dtype=torch.float32), requires_grad=False
        )
        self.base = nn.Parameter(
            torch.empty((self.mix_hc,), dtype=torch.float32), requires_grad=False
        )
        self.scale = nn.Parameter(torch.empty((3,), dtype=torch.float32), requires_grad=False)

    def pre_mapping(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [b,s,hc,d], hc_fn: [mix_hc,hc*d], hc_scale: [3], hc_base: [mix_hc], y: [b,s,hc,d]
        if not _cuda_available:
            raise RuntimeError(
                "Raw CUDA backend is unavailable. "
                "Ensure torch.utils.cpp_extension and CUDA toolkit are installed."
            )
        assert x.dtype == torch.bfloat16
        assert self.mult == x.shape[-2]
        assert self.hidden_size == x.shape[-1]
        outer_shape = x.shape[:-2]
        residual_flat = x.view(-1, self.mult, self.hidden_size)
        num_tokens = residual_flat.shape[0]

        post_mix, comb_mix, layer_input = mhc_pre_mapping_fused_cuda(
            residual_flat.view(num_tokens, self.hc_dim),
            self.fn.contiguous(),
            residual_flat,
            self.mult,
            self.scale,
            self.base,
            self.hidden_size,
            self.norm_eps,
            self.eps,
            self.sinkhorn_eps,
            self.post_mult_value,
            self.sinkhorn_iters,
        )

        post_mix = post_mix.view(*outer_shape, self.mult, 1)
        comb_mix = comb_mix.view(*outer_shape, self.mult, self.mult)
        layer_input = layer_input.view(*outer_shape, self.hidden_size)
        return post_mix, comb_mix, layer_input

    # ------------------------------------------------------------------
    # DeepSeek-V4.1: lagged-`pre` path
    # ------------------------------------------------------------------
    # V4 and V4.1 both project one mixer per sublayer into three coefficient
    # sets, but they wire `pre` differently: V4 collapses with the `pre` the
    # same sublayer just produced, V4.1 collapses with the *previous*
    # sublayer's `pre` and hands its own forward. Both run the same fused
    # kernels (GEMM + sqrsum, then bigfuse; or the all-in-one variants); the
    # wiring is a template parameter of the kernel epilogue (`kLaggedPre`),
    # exposed here as `pre_mapping_lagged` / `fused_hc_lagged` next to V4's
    # `pre_mapping` / `fused_hc`. `mixer_projection`, `split_sinkhorn` and
    # `hc_coeffs` below are the eager torch transcription the tests compare
    # those kernels against.

    def mixer_projection(self, x: torch.Tensor) -> torch.Tensor:
        """RMS-scaled mixer projection: ``[..., mult, hidden] -> [M, mix_hc]`` fp32.

        Matches the reference ``hc_mixes`` ordering exactly — the GEMM runs on
        the raw flattened stream and the per-token ``rsqrt`` is applied to the
        *result*, not to the input::

            mixes = F.linear(x.flatten(2).float(), fn) * rsqrt(mean(x ^ 2) + norm_eps)
        """
        y_acc, r_acc = self._mixer_gemm(x)
        # r_acc is the row sum of squares; the reference normalizes by the mean.
        rsqrt = torch.rsqrt(r_acc / self.hc_dim + self.norm_eps).unsqueeze(-1)
        return y_acc * rsqrt

    def _mixer_gemm(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The mixer GEMM and row sum of squares, *before* the per-token rsqrt.

        Split out of :meth:`mixer_projection` because the fused tail in
        :meth:`hc_coeffs` applies that rsqrt inside its own kernel, so it needs
        the raw accumulator. :meth:`mixer_projection` is unchanged: it is this
        plus the same rsqrt it always applied.
        """
        x_flat = x.reshape(-1, self.mult, self.hidden_size)
        m = x_flat.shape[0]
        x_2d = x_flat.reshape(m, self.hc_dim).contiguous()
        w_t = self.fn.to(torch.float32).contiguous()

        if _cuda_available and mhc_gemm_rms_fma_cuda is not None and x_2d.is_cuda:
            # w_t is [N, K] == [mix_hc, hc_dim], i.e. ``self.fn`` as stored.
            return mhc_gemm_rms_fma_cuda(x_2d, None, m, self.mix_hc, self.hc_dim, w_t=w_t)

        # One upcast feeds both outputs -- `x_2d.to(float32)` twice would double
        # the peak fp32 footprint at prefill token counts.
        x_fp32 = x_2d.to(torch.float32)
        return torch.nn.functional.linear(x_fp32, w_t), x_fp32.square().sum(-1)

    def split_sinkhorn(
        self, mixes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split ``mixes`` into ``(pre, post, comb)``; ``comb`` is Sinkhorn-normalized.

        Transcribed from the reference ``hc_split_sinkhorn`` tilelang kernel
        (``<checkpoint>/inference/kernel.py``). What is *pinned by a test* is
        agreement with an independent re-transcription of that kernel on random
        GPU inputs (``test_mhc.py::test_coeffs_match_reference_transcription``,
        rtol 1e-4 on ``pre``/``post``, 1e-3 on ``comb``); the tilelang kernel
        itself is not executed here. Two details are easy to get wrong and are
        load-bearing:

        * ``pre`` gets ``+ eps``, ``post`` does not.
        * the Sinkhorn tail runs ``sinkhorn_iters - 1`` row/column pairs after
          the initial softmax + column normalize, so it *ends on a column
          normalize* — the rows are deliberately not unit-sum.
        """
        n = self.mult
        scale = self.scale.to(torch.float32)
        base = self.base.to(torch.float32)

        pre = torch.sigmoid(mixes[..., :n] * scale[0] + base[:n]) + self.eps
        post = self.post_mult_value * torch.sigmoid(
            mixes[..., n : 2 * n] * scale[1] + base[n : 2 * n]
        )
        comb = (mixes[..., 2 * n :] * scale[2] + base[2 * n :]).unflatten(-1, (n, n))
        comb = comb.softmax(dim=-1) + self.sinkhorn_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.sinkhorn_eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.sinkhorn_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.sinkhorn_eps)
        return pre, post, comb

    def hc_coeffs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eager ``[..., mult, hidden] -> (pre_mix, post_mix, comb_mix)`` (torch reference).

        ``pre_mix`` / ``post_mix`` are ``[..., mult, 1]`` fp32 and ``comb_mix`` is
        ``[..., mult, mult]`` fp32 -- the layout ``pre_mapping`` returns. The serving path
        never calls this: DeepSeek-V4.1 uses :meth:`pre_mapping_lagged` /
        :meth:`fused_hc_lagged` (C++ fused kernels); this is the transcription the tests
        compare them against.
        """
        outer_shape = x.shape[:-2]
        pre, post, comb = self.split_sinkhorn(self.mixer_projection(x))
        return (
            pre.view(*outer_shape, self.mult, 1),
            post.view(*outer_shape, self.mult, 1),
            comb.view(*outer_shape, self.mult, self.mult),
        )

    @staticmethod
    def collapse(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Reference ``hc_pre``: ``[..., mult, hidden] x [..., mult, 1] -> [..., hidden]``.

        Accumulated in fp32 and cast back, matching the reference. Summed one
        copy at a time rather than materializing an fp32 ``[..., mult, hidden]``
        product, which at prefill token counts is several hundred MB.
        """
        mult = x.shape[-2]
        # ``acc`` is the fresh product of the first term, so accumulate into it
        # in place: ``acc = acc + ...`` would allocate a second fp32
        # ``[..., hidden]`` result on each of the remaining ``mult - 1`` steps.
        acc = x[..., 0, :].to(torch.float32) * pre_mix[..., 0, :]
        for i in range(1, mult):
            acc.add_(x[..., i, :].to(torch.float32) * pre_mix[..., i, :])
        return acc.to(x.dtype)

    def pre_mapping_lagged(
        self,
        x: torch.Tensor,
        pre_mix: torch.Tensor,
        norm_weight: Optional[torch.Tensor] = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """V4.1 pre-mapping: collapse with an *external* ``pre_mix``, return our own.

        Args:
            x:       ``[..., mult, hidden]`` bf16 residual stream.
            pre_mix: ``[..., mult, 1]`` or ``[..., mult]`` fp32, produced by the
                     *previous* sublayer's mixer.

        Returns ``(pre_mix_own, post_mix, comb_mix, layer_input)`` where
        ``pre_mix_own`` is this sublayer's own ``pre``, to be carried forward to
        the next sublayer, and the other three match ``pre_mapping``'s contract.

        Self-check available to callers and tests: passing ``pre_mix=pre_mix_own``
        makes this exactly equivalent to ``pre_mapping(x)`` — that is the V4
        (unlagged) wiring, and it is what
        ``tests/unittest/_torch/modules/test_mhc.py::test_lagged_reduces_to_pre_mapping`` asserts.

        Same kernels as V4's :meth:`pre_mapping` (GEMM + sqrsum, FMA or TF32 MMA by the
        autotuner, then bigfuse) instantiated with the lagged pre-mix template variant:
        bigfuse collapses with ``pre_mix`` and writes this sublayer's own ``pre``.
        ``norm_weight`` folds the next RMSNorm into ``layer_input``.
        """
        if not _cuda_available:
            raise RuntimeError(
                "Raw CUDA backend is unavailable. "
                "Ensure torch.utils.cpp_extension and CUDA toolkit are installed."
            )
        assert x.dtype == torch.bfloat16
        assert self.mult == x.shape[-2]
        assert self.hidden_size == x.shape[-1]

        outer_shape = x.shape[:-2]
        residual_flat = x.reshape(-1, self.mult, self.hidden_size).contiguous()
        num_tokens = residual_flat.shape[0]
        post_mix, comb_mix, layer_input, pre_own = mhc_pre_mapping_fused_cuda(
            residual_flat.view(num_tokens, self.hc_dim),
            self.fn.contiguous(),
            residual_flat,
            self.mult,
            self.scale,
            self.base,
            self.hidden_size,
            self.norm_eps,
            self.eps,
            self.sinkhorn_eps,
            self.post_mult_value,
            self.sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            pre_mix_ext=pre_mix.reshape(num_tokens, self.mult),
        )
        return (
            pre_own.view(*outer_shape, self.mult, 1),
            post_mix.view(*outer_shape, self.mult, 1),
            comb_mix.view(*outer_shape, self.mult, self.mult),
            layer_input.view(*outer_shape, self.hidden_size),
        )

    def fused_hc_lagged(
        self,
        x_prev: torch.Tensor,
        residual_prev: torch.Tensor,
        post_mix_prev: torch.Tensor,
        comb_mix_prev: torch.Tensor,
        pre_mix: torch.Tensor,
        norm_weight: Optional[torch.Tensor] = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Post update, lagged collapse and next coefficient prediction in two launches.

        V4's :meth:`fused_hc` (post update + GEMM kernel -- FMA or tcgen05 TF32 by the
        autotuner -- then bigfuse) instantiated with the lagged pre-mix template variant,
        with the optional next RMSNorm folded into ``layer_input``.

        Consumes CUDA BF16 ``x_prev [..., hidden]`` and residual
        ``[..., 4, hidden]`` with FP32 previous ``post [..., 4, 1]`` and
        ``comb [..., 4, 4]``. The external FP32 ``pre_mix [..., 4, 1]`` belongs
        to the previous sublayer; this module's own pre is returned separately.
        Returns ``(residual_cur, pre_cur, post_cur, comb_cur, layer_input)``.
        """
        if not _cuda_available:
            raise RuntimeError(
                "Raw CUDA backend is unavailable. "
                "Ensure torch.utils.cpp_extension and CUDA toolkit are installed."
            )
        n = self.mult
        hidden = self.hidden_size
        outer_shape = residual_prev.shape[:-2]
        residual_prev_flat = residual_prev.reshape(-1, n, hidden).contiguous()
        B = residual_prev_flat.shape[0]
        residual_cur, post_cur, comb_cur, layer_input, pre_cur = mhc_fused_hc_cuda(
            x_prev.reshape(B, hidden).contiguous(),
            residual_prev_flat,
            post_mix_prev.reshape(B, n),
            comb_mix_prev.reshape(B, n, n),
            self.fn.contiguous(),
            self.scale,
            self.base,
            n,
            hidden,
            self.norm_eps,
            self.eps,
            self.sinkhorn_eps,
            self.post_mult_value,
            self.sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            pre_mix_ext=pre_mix.reshape(B, n),
        )
        return (
            residual_cur.view(*outer_shape, n, hidden),
            pre_cur.view(*outer_shape, n, 1),
            post_cur.view(*outer_shape, n, 1),
            comb_cur.view(*outer_shape, n, n),
            layer_input.view(*outer_shape, hidden),
        )

    def fused_hc(
        self,
        x_prev: torch.Tensor,
        residual_prev: torch.Tensor,
        post_mix_prev: torch.Tensor,
        comb_mix_prev: torch.Tensor,
        norm_weight: Optional[torch.Tensor] = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused post_mapping(from previous mHC) + pre_mapping(from self).

        This is the boundary op between two mHC-wrapped blocks. It consumes the
        output of the previous block (``x_prev``) plus the previous block's
        residual and mix matrices, then runs the current block's pre_mapping
        using ``self``'s parameters. Semantically identical to

            residual_cur = prev_mHC.post_mapping(x_prev, residual_prev, ...)
            post_mix, comb_mix, layer_input = self.pre_mapping(residual_cur)

        but exposed as one call so the model forward can say
        ``state = mHC_next.fused_hc(...)`` at every layer boundary.

        When ``norm_weight`` is provided, the next-layer RMSNorm is folded into
        ``layer_input_cur`` (i.e. the returned ``layer_input_cur`` is already
        ``rmsnorm(layer_input_raw, norm_weight, norm_eps)``), saving an extra
        kernel launch on the model's pre-attention norm.

        Args:
            x_prev:        [..., hidden]    bf16  (attn / MoE output of prev block)
            residual_prev: [..., mult, hidden] bf16
            post_mix_prev: [..., mult] or [..., mult, 1] fp32
            comb_mix_prev: [..., mult, mult] fp32
            norm_weight:   [hidden] bf16 / None — when set, fuse next-layer RMSNorm
                           into ``layer_input_cur`` epilogue.
            norm_eps:      RMSNorm epsilon (only consulted when ``norm_weight`` is set).

        Returns:
            residual_cur:    [..., mult, hidden] bf16 (new residual for next post_mapping)
            post_mix_cur:    [..., mult, 1] fp32
            comb_mix_cur:    [..., mult, mult] fp32
            layer_input_cur: [..., hidden] bf16 (RMSNorm-normalized when ``norm_weight`` is set)
        """
        if not _cuda_available:
            raise RuntimeError(
                "Raw CUDA backend is unavailable. "
                "Ensure torch.utils.cpp_extension and CUDA toolkit are installed."
            )
        assert x_prev.dtype == torch.bfloat16
        assert residual_prev.dtype == torch.bfloat16
        n = self.mult
        hidden = self.hidden_size
        outer_shape = residual_prev.shape[:-2]

        residual_prev_flat = residual_prev.reshape(-1, n, hidden).contiguous()
        B = residual_prev_flat.shape[0]
        x_prev_flat = x_prev.reshape(B, hidden).contiguous()
        post_mix_prev_flat = post_mix_prev.reshape(B, n)
        comb_mix_prev_flat = comb_mix_prev.reshape(B, n, n)

        residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur = mhc_fused_hc_cuda(
            x_prev_flat,
            residual_prev_flat,
            post_mix_prev_flat,
            comb_mix_prev_flat,
            self.fn.contiguous(),
            self.scale,
            self.base,
            n,
            hidden,
            self.norm_eps,
            self.eps,
            self.sinkhorn_eps,
            self.post_mult_value,
            self.sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )

        residual_cur = residual_cur.view(*outer_shape, n, hidden)
        post_mix_cur = post_mix_cur.view(*outer_shape, n, 1)
        comb_mix_cur = comb_mix_cur.view(*outer_shape, n, n)
        layer_input_cur = layer_input_cur.view(*outer_shape, hidden)
        return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur

    def post_mapping(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        if not _cuda_available:
            raise RuntimeError(
                "Raw CUDA backend is unavailable. "
                "Ensure torch.utils.cpp_extension and CUDA toolkit are installed."
            )
        outer_shape = residual.shape[:-2]
        n = self.mult
        hidden = residual.shape[-1]
        residual_flat = residual.view(-1, n, hidden)
        B = residual_flat.shape[0]

        out = mhc_post_mapping_cuda(
            residual_flat,
            x.reshape(B, hidden),
            post_layer_mix.view(B, n),
            comb_res_mix.view(B, n, n),
            n,
        )
        return out.view(*outer_shape, n, hidden)


class HCHead(nn.Module):
    def __init__(
        self,
        mult: int,
        hidden_size: int,
        eps: float = 1e-6,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.mult = mult
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm_eps = norm_eps
        self.fn = nn.Parameter(
            torch.empty((self.mult, self.mult * self.hidden_size), dtype=torch.float32),
            requires_grad=False,
        )
        self.base = nn.Parameter(
            torch.empty((self.mult,), dtype=torch.float32), requires_grad=False
        )
        self.scale = nn.Parameter(torch.empty((1,), dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _cuda_available:
            raise RuntimeError("CUDA MHC kernels not available")
        dtype = x.dtype
        # The CUDA head consumes a flat ``[M, mult, hidden]`` tensor. Preserve any
        # leading dims (e.g. ``[batch, block, mult, hidden]`` from the DSpark draft
        # block) by collapsing to a single token axis and restoring afterwards;
        # the RMS-norm + weighted sum is independent per token, so this matches the
        # reference ``hc_head`` which keeps the leading dims.
        lead = x.shape[:-2]
        x_bf16 = x.reshape(-1, self.mult, self.hidden_size).to(torch.bfloat16).contiguous()
        y = mhc_hc_head_cuda(
            x_bf16,
            self.fn,
            self.scale,
            self.base,
            self.mult,
            self.hidden_size,
            norm_eps=self.norm_eps,
            eps=self.eps,
        )
        return y.reshape(*lead, self.hidden_size).to(dtype)

    def skip_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Skip HCHead computation for pipeline parallelism on non-last ranks."""
        return x
