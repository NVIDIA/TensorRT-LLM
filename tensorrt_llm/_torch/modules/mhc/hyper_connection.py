# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Multi-Head Hyper-Connection (mHC) module
# Based on: "Hyper-Connections" (https://arxiv.org/abs/2409.19606)
import functools
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import triton
import triton.language as tl
from torch import nn

from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream, private_stream
from tensorrt_llm._torch.utils import Fp4QuantizedTensor


@triton.jit
def _collapse_kernel(x_ptr, pre_ptr, out_ptr, hidden, MULT: tl.constexpr, BLOCK: tl.constexpr):
    """One row and ``BLOCK`` columns: ``sum_i x[i] * pre[i]`` in FP32, cast to the output type."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < hidden
    acc = tl.load(x_ptr + row * MULT * hidden + cols, mask=mask).to(tl.float32) * tl.load(
        pre_ptr + row * MULT
    )
    for i in tl.static_range(1, MULT):
        x = tl.load(x_ptr + (row * MULT + i) * hidden + cols, mask=mask).to(tl.float32)
        acc += x * tl.load(pre_ptr + row * MULT + i)
    tl.store(out_ptr + row * hidden + cols, acc.to(out_ptr.dtype.element_ty), mask=mask)


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
    # Split boundaries (decode-sized M under CUDA-graph multi-stream) produce ``pre_mix`` /
    # ``post_mix`` / ``comb_mix`` on a side stream; the consumer's stream must wait for this
    # event before reading them (``wait_coefficients``).
    coeff_event: Optional[torch.cuda.Event] = None
    # Set when ``x_prev`` is a tensor-parallel partial sum: the consumer either all-reduces it
    # inside its boundary kernel or calls ``reduced_x_prev`` (which runs this all-reduce).
    x_prev_allreduce: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    @property
    def is_deferred(self) -> bool:
        return self.post_mix is not None

    def wait_coefficients(self) -> None:
        """Make the current stream wait for side-stream coefficients (no-op without them)."""
        if self.coeff_event is not None:
            torch.cuda.current_stream().wait_event(self.coeff_event)
            self.coeff_event = None

    def reduced_x_prev(self) -> Optional[torch.Tensor]:
        """``x_prev`` summed over the tensor-parallel group (all-reduced here if still partial)."""
        if self.x_prev is not None and self.x_prev_allreduce is not None:
            self.x_prev = self.x_prev_allreduce(self.x_prev)
            self.x_prev_allreduce = None
        return self.x_prev

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
        coeff_event: Optional[torch.cuda.Event] = None,
        x_prev_allreduce: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> "HCState":
        return cls(
            residual=residual,
            post_mix=post_mix,
            comb_mix=comb_mix,
            x_prev=x_prev,
            pre_mix=pre_mix,
            coeff_event=coeff_event,
            x_prev_allreduce=x_prev_allreduce,
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


@functools.cache
def _boundary_kernel(device: torch.device, hidden_size: int):
    """The CuTe DSL single-pass boundary module if it runs on ``device`` at ``hidden_size``."""
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
    from tensorrt_llm._utils import get_sm_version

    if device.type != "cuda" or not IS_CUTLASS_DSL_AVAILABLE or get_sm_version() not in (100, 103):
        return None
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.mhc import mhc_boundary

    return mhc_boundary if mhc_boundary.MhcBoundaryKernel.supports(hidden_size) else None


# Side streams with deferred Sinkhorn kernels that no consumer of comb has joined yet.
_sinkhorn_pending: set[torch.cuda.ExternalStream] = set()


def _join_sinkhorn() -> None:
    """Orders the current stream after the deferred Sinkhorn kernels; comb's consumers call it first."""
    while _sinkhorn_pending:
        torch.cuda.current_stream().wait_stream(_sinkhorn_pending.pop())


_COEFF_SCRATCH_PRE: dict = {}


def _coefficient_scratch_pre(m: int, n: int, device: torch.device) -> torch.Tensor:
    """A ``[m, n]`` FP32 ones view of a buffer that is never freed or rewritten.

    Grown only outside CUDA-graph capture (a captured fill would not run before replay); the
    warmup forwards that precede every capture allocate it. Outgrown buffers stay referenced,
    since captured graphs may still read them.
    """
    key = (device, n)
    bufs = _COEFF_SCRATCH_PRE.setdefault(key, [])
    if not bufs or bufs[-1].shape[0] < m:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("mHC coefficient scratch must be allocated before graph capture")
        rows = max(m, 64 if not bufs else 2 * bufs[-1].shape[0])
        bufs.append(torch.ones((rows, n), dtype=torch.float32, device=device))
    return bufs[-1][:m]


class mHC(nn.Module):
    # Every tensor-parallel rank computes this residual stream alike (attention TP without attention
    # DP). Set by the owning layer; the fused boundaries are then deterministic, so the replicas stay
    # bit-identical.
    replicated = False

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
        single_pass_boundary: bool = False,
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
        # With single_pass_boundary, lagged boundaries of up to 4096 tokens run CuTe DSL kernels
        # with a prepared copy of `fn` that the post-load hooks build and refresh.
        self.single_pass_boundary = single_pass_boundary
        self._boundary_weight: Optional[torch.Tensor] = None

    @torch.no_grad()
    def cache_derived_state(self) -> None:
        kernel = self.single_pass_boundary and _boundary_kernel(self.fn.device, self.hidden_size)
        if not kernel or self.mult != kernel.MULT:
            return
        prepared = kernel.prepare_boundary_weight(self.fn)
        if self._boundary_weight is None:
            self._boundary_weight = prepared
        else:
            # In place: captured CUDA graphs keep the address.
            self._boundary_weight.copy_(prepared)
        # The boundary kernel reads this weight, scale and base before its PDL wait, so their
        # writes must be complete before the next launch.
        torch.cuda.current_stream().synchronize()

    def post_load_weights(self) -> None:
        self.cache_derived_state()

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.cache_derived_state()

    def _single_pass_boundary(
        self,
        x_prev: Optional[torch.Tensor],
        residual: torch.Tensor,
        post_mix: Optional[torch.Tensor],
        comb_mix: Optional[torch.Tensor],
        pre_mix: torch.Tensor,
        norm_weight: Optional[torch.Tensor],
        norm_eps: float,
        mxfp8: bool = False,
    ) -> Optional[tuple[torch.Tensor, ...]]:
        """CuTe DSL lagged boundary for up to 1023 tokens (4096 without a post update), else None.

        Same TF32 products as the fused kernels, reduced in a fixed order (deterministic).
        ``x_prev is None`` collapses ``residual`` without a post update.
        Larger batches keep ``mhc_fused_hc`` (``mega_mhc`` from 1024 tokens) and the prefill-size
        collapse kernels. Under CUDA graphs the Sinkhorn iterations of ``comb`` run on a side stream,
        off the critical path; ``fused_hc_lagged`` and ``post_mapping`` join it before reading ``comb``.
        Returns ``(residual_cur, post, comb, layer_input, pre, layer_input_mxfp8)``, the first five
        like ``mhc_fused_hc``; see ``mhc_boundary.mhc_boundary`` for ``mxfp8``.
        """
        if self._boundary_weight is None or norm_weight is None:
            return None
        kernel = _boundary_kernel(residual.device, self.hidden_size)
        tokens = residual.shape[0]
        deferred = do_multi_stream()
        if not 0 < tokens <= (1023 if x_prev is not None else 4096) or (
            torch.cuda.is_current_stream_capturing()
            and not kernel.compiled(
                self.hidden_size, x_prev is not None, self.sinkhorn_iters, tokens, deferred, mxfp8
            )
        ):
            return None
        inputs = [
            None if t is None else t.contiguous()
            for t in (x_prev, residual, post_mix, comb_mix, pre_mix, norm_weight)
        ]
        if any(t is not None and t.data_ptr() % 16 for t in inputs):
            return None
        x_prev, residual, post_mix, comb_mix, pre_mix, norm_weight = inputs
        residual_cur, pre, post, comb, layer_input, layer_input_mxfp8 = kernel.mhc_boundary(
            x_prev,
            residual,
            post_mix,
            comb_mix,
            pre_mix,
            self._boundary_weight,
            self.scale,
            self.base,
            norm_weight,
            self.norm_eps,
            self.eps,
            self.sinkhorn_eps,
            self.post_mult_value,
            norm_eps,
            0 if deferred else self.sinkhorn_iters,
            mxfp8,
        )
        if deferred:
            side = private_stream(residual.device.index, "mhc_sinkhorn")
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                kernel.mhc_sinkhorn(comb, self.sinkhorn_eps, self.sinkhorn_iters)
            _sinkhorn_pending.add(side)
        return residual_cur, post, comb, layer_input, pre, layer_input_mxfp8

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
        if x.is_cuda and x.is_contiguous():
            # One launch with the same per-term rounding and summation order;
            # fp fusion is off so the products are not contracted into FMAs.
            hidden = x.shape[-1]
            out = x.new_empty((*x.shape[:-2], hidden))
            rows = out.numel() // hidden
            block = 1024
            _collapse_kernel[(rows, triton.cdiv(hidden, block))](
                x,
                pre_mix.reshape(rows, mult).to(torch.float32).contiguous(),
                out,
                hidden,
                MULT=mult,
                BLOCK=block,
                enable_fp_fusion=False,
            )
            return out
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
        ``norm_weight`` folds the next RMSNorm into ``layer_input``. Up to 4096 tokens with a
        ``norm_weight``, :meth:`_single_pass_boundary` runs CuTe DSL kernels instead.
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
        pre_mix = pre_mix.reshape(num_tokens, self.mult)
        single = self._single_pass_boundary(
            None, residual_flat, None, None, pre_mix, norm_weight, norm_eps
        )
        if single is not None:
            _, post_mix, comb_mix, layer_input, pre_own, _ = single
        else:
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
                pre_mix_ext=pre_mix,
            )
        return (
            pre_own.view(*outer_shape, self.mult, 1),
            post_mix.view(*outer_shape, self.mult, 1),
            comb_mix.view(*outer_shape, self.mult, self.mult),
            layer_input.view(*outer_shape, self.hidden_size),
        )

    def lagged_coefficients(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(pre, post, comb)`` of this boundary from its BF16 residual ``[M, mult, hidden]``.

        The coefficient half of a split boundary (see ``split_hc_lagged``): the same kernels as
        :meth:`pre_mapping_lagged` (mixer GEMM + sum of squares, then bigfuse), computed from the
        BF16 residual like the reference ``hc_mixes``. Bigfuse's collapse output is discarded, so it
        collapses with a constant pre-mix buffer that outlives any side-stream use.
        """
        n = self.mult
        m = residual.shape[0]
        ones = _coefficient_scratch_pre(m, n, residual.device)
        post_mix, comb_mix, _, pre_own = mhc_pre_mapping_fused_cuda(
            residual.view(m, self.hc_dim),
            self.fn.contiguous(),
            residual,
            n,
            self.scale,
            self.base,
            self.hidden_size,
            self.norm_eps,
            self.eps,
            self.sinkhorn_eps,
            self.post_mult_value,
            self.sinkhorn_iters,
            pre_mix_ext=ones,
        )
        return pre_own.view(m, n, 1), post_mix.view(m, n, 1), comb_mix.view(m, n, n)

    def split_hc_lagged(
        self,
        x_prev: Optional[torch.Tensor],
        residual_prev: torch.Tensor,
        post_mix_prev: Optional[torch.Tensor],
        comb_mix_prev: Optional[torch.Tensor],
        pre_mix: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        coeff_stream: Optional[torch.cuda.Stream] = None,
        events: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None,
        allreduce=None,
        mxfp8: bool = False,
        nvfp4_scale: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Lagged boundary as a critical half and a coefficient half (decode-sized M).

        The critical half (one launch) applies the previous post update when ``x_prev`` is given,
        collapses with the previous ``pre_mix`` and applies the RMSNorm: that is all the next
        sublayer reads. This boundary's own ``pre`` / ``post`` / ``comb`` are only read by the next
        boundary, so with ``coeff_stream`` they are computed there, overlapping the sublayer, and the
        returned event must be waited on before they are read (``HCState.wait_coefficients``).

        With ``allreduce`` (a :class:`~.mhc_lowm.LamportAllReduce`), ``x_prev`` is this rank's
        tensor-parallel partial sum and the critical half all-reduces it in the same launch.

        Returns ``(residual, pre, post, comb, layer_input, coeff_event)``; ``residual`` is
        ``residual_prev`` itself when ``x_prev`` is None. ``mxfp8`` appends ``layer_input``'s
        MXFP8 values and swizzled scales, as ``torch.ops.trtllm.mxfp8_quantize(layer_input, True)``.
        With an FP32 ``nvfp4_scale [1]``, ``layer_input`` is an unswizzled ``Fp4QuantizedTensor``
        equal to ``fp4_quantize(layer_input, nvfp4_scale)``, the BF16 values in its
        ``unquantized_hidden_states``.
        """
        from .mhc_lowm import mhc_allreduce_post_collapse_norm, mhc_post_collapse_norm

        _join_sinkhorn()
        n = self.mult
        res_in = residual_prev.reshape(-1, n, self.hidden_size)
        if allreduce is not None and x_prev is not None:
            residual, layer_input, *quantized = mhc_allreduce_post_collapse_norm(
                x_prev,
                res_in,
                post_mix_prev,
                comb_mix_prev,
                pre_mix,
                norm_weight,
                norm_eps,
                allreduce,
                mxfp8=mxfp8,
                nvfp4_scale=nvfp4_scale,
            )
        else:
            residual, layer_input, *quantized = mhc_post_collapse_norm(
                x_prev,
                res_in,
                post_mix_prev,
                comb_mix_prev,
                pre_mix,
                norm_weight,
                norm_eps,
                mxfp8=mxfp8,
                nvfp4_scale=nvfp4_scale,
            )
        if nvfp4_scale is not None:
            (fp4, scales), quantized = quantized[0], ()
            layer_input = Fp4QuantizedTensor(
                fp4, scales, is_sf_swizzled=False, unquantized_hidden_states=layer_input
            )
        if coeff_stream is None:
            pre, post, comb = self.lagged_coefficients(residual)
            return residual, pre, post, comb, layer_input, None, *quantized
        fork, done = events
        fork.record()
        with torch.cuda.stream(coeff_stream):
            fork.wait()
            pre, post, comb = self.lagged_coefficients(residual)
            done.record()
        return residual, pre, post, comb, layer_input, done, *quantized

    def fused_hc_lagged(
        self,
        x_prev: torch.Tensor,
        residual_prev: torch.Tensor,
        post_mix_prev: torch.Tensor,
        comb_mix_prev: torch.Tensor,
        pre_mix: torch.Tensor,
        norm_weight: Optional[torch.Tensor] = None,
        norm_eps: float = 0.0,
        mxfp8: bool = False,
        nvfp4_scale: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        """Post update, lagged collapse and next coefficient prediction.

        V4's :meth:`fused_hc` (post update + GEMM kernel -- FMA or tcgen05 TF32 by the
        autotuner -- then bigfuse) instantiated with the lagged pre-mix template variant,
        with the optional next RMSNorm folded into ``layer_input``; ``mhc_fused_hc``
        documents when DeepGEMM's ``mega_mhc`` or, for :attr:`replicated` streams, the
        deterministic composition runs instead. Up to 1023 tokens with a ``norm_weight``,
        :meth:`_single_pass_boundary` runs CuTe DSL kernels instead.

        Consumes CUDA BF16 ``x_prev [..., hidden]`` and residual
        ``[..., 4, hidden]`` with FP32 previous ``post [..., 4, 1]`` and
        ``comb [..., 4, 4]``. The external FP32 ``pre_mix [..., 4, 1]`` belongs
        to the previous sublayer; this module's own pre is returned separately.
        Returns ``(residual_cur, pre_cur, post_cur, comb_cur, layer_input)``; ``mxfp8`` appends
        ``layer_input`` as ``torch.ops.trtllm.mxfp8_quantize(layer_input, True)`` returns it when
        the single-pass boundary ran, else None. With an FP32 ``nvfp4_scale [1]``, ``layer_input``
        comes back as an unswizzled ``Fp4QuantizedTensor`` equal to
        ``fp4_quantize(layer_input, nvfp4_scale)``, the BF16 values in its
        ``unquantized_hidden_states``, when the CuTe DSL kernels run on this device.
        """
        if not _cuda_available:
            raise RuntimeError(
                "Raw CUDA backend is unavailable. "
                "Ensure torch.utils.cpp_extension and CUDA toolkit are installed."
            )
        _join_sinkhorn()
        n = self.mult
        hidden = self.hidden_size
        outer_shape = residual_prev.shape[:-2]
        residual_prev_flat = residual_prev.reshape(-1, n, hidden).contiguous()
        B = residual_prev_flat.shape[0]
        single = self._single_pass_boundary(
            x_prev.reshape(B, hidden),
            residual_prev_flat,
            post_mix_prev.reshape(B, n),
            comb_mix_prev.reshape(B, n * n),
            pre_mix.reshape(B, n),
            norm_weight,
            norm_eps,
            mxfp8,
        )
        layer_input_mxfp8 = None
        if single is not None:
            residual_cur, post_cur, comb_cur, layer_input, pre_cur, layer_input_mxfp8 = single
        else:
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
                deterministic=self.replicated,
            )
        nvfp4 = None
        if nvfp4_scale is not None:
            kernel = _boundary_kernel(residual_prev.device, hidden)
            if kernel is not None and (
                not torch.cuda.is_current_stream_capturing()
                or kernel.nvfp4_quantize_compiled(hidden)
            ):
                nvfp4 = kernel.nvfp4_quantize(layer_input.reshape(B, hidden), nvfp4_scale)
        layer_input = layer_input.view(*outer_shape, hidden)
        if nvfp4 is not None:
            layer_input = Fp4QuantizedTensor(
                *nvfp4, is_sf_swizzled=False, unquantized_hidden_states=layer_input
            )
        outputs = (
            residual_cur.view(*outer_shape, n, hidden),
            pre_cur.view(*outer_shape, n, 1),
            post_cur.view(*outer_shape, n, 1),
            comb_cur.view(*outer_shape, n, n),
            layer_input,
        )
        return (*outputs, layer_input_mxfp8) if mxfp8 else outputs

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
            deterministic=self.replicated,
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
        _join_sinkhorn()
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
