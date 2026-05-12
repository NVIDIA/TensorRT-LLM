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

    @property
    def is_deferred(self) -> bool:
        return self.post_mix is not None

    @classmethod
    def resolved(cls, residual: torch.Tensor) -> "HCState":
        return cls(residual=residual)

    @classmethod
    def deferred(
        cls,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
        x_prev: torch.Tensor,
    ) -> "HCState":
        return cls(residual=residual, post_mix=post_mix, comb_mix=comb_mix, x_prev=x_prev)


try:
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import mhc_fused_hc as mhc_fused_hc_cuda
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import mhc_hc_head_cuda, mhc_post_mapping_cuda
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
        x_bf16 = x.to(torch.bfloat16).contiguous()
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
        return y.to(dtype)

    def skip_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Skip HCHead computation for pipeline parallelism on non-last ranks."""
        return x
