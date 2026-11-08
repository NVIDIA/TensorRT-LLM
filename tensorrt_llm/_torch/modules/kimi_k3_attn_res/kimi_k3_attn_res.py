# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KimiK3AttnResidualOp — decoder-layer fused Attention Residual op.

Kimi K3 uses ``attn_res_block_size=12``. Every layer emits a running
``prefix_sum``; every ``attn_res_block_size``-th layer accumulates that
prefix into a growing block-residual stack. Two positions per layer
consume the stack: once before self-attention (``self_attention_res``)
and once after the sub-block that produces the running ``prefix_sum``
(``mlp_res``). The model also emits one final ``output_attn_res`` at the
top of the decoder tower.

Each residual-selection step evaluates the same algebra, expressed by HF
``modeling_kimi._apply_attn_res``:

    v = concat(block_residual, prefix_sum.unsqueeze(1))  # [M, K+1, H]
    variance = v.pow(2).mean(-1, keepdim=True)
    k = v * rsqrt(variance + eps)
    score_weight = norm.weight * proj.weight.squeeze(0)
    scores = (k * score_weight).sum(-1)
    probs = softmax(scores, dim=-1)                       # [M, K+1]
    output = probs @ v                                    # [M, H]

The sm_100 ``attn_res_fwd`` fused kernel does exactly this in one launch
under the documented constraints:

    B == 1
    N = K + 1 in [1, 12]
    T in [1, 16384]
    H in {4096, 5120, 6144, 7168, 8192}   (K3 uses H=7168)
    layer_residual, block_residual: bf16 CUDA contiguous
    res_weight: bf16 [H] or [H, 1]
    rms_weight: bf16 [H]

Interface differences vs HF:
* The kernel packs shapes as [T, B, H] and [K, T, B, H]; HF passes the
  block residual as [M, K, H]. This module handles the reshape and
  contiguity requirements before calling the kernel.
* The kernel returns ``rsigma`` and ``probs`` alongside ``output`` for
  benchmarking / diagnostics; the module discards them on the module
  path but exposes them on the standalone ``forward`` helper for tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from ._attn_res_kernels import (
    intree_attn_res_fwd,
    is_attn_res_optimized_supported,
    is_intree_attn_res_available,
)


class KimiK3AttnResidualKernelPath:
    """Enum-like string tags for the selected kernel path."""

    OPTIMIZED = "optimized"
    REFERENCE = "reference"


# ---------------------------------------------------------------------------
# Reference implementations (chunked torch reference + HF direct).
# ---------------------------------------------------------------------------


def attn_res_fwd_chunked_reference(
    layer_residual: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    max_elements: int = 4 * 1024 * 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked torch reference for packed Attention Residual forward.

    Mirrors ``exisiting_optimization_work/Attention_residual/tests/util/attn_res_ref.py``
    so callers can produce byte-identical reference outputs without loading
    the optimization tree.

    Args mirror :meth:`KimiK3AttnResidualOp.attn_res_fwd`:

        layer_residual: bf16 [T, B, H]
        block_residual: bf16 [K, T, B, H]
        res_weight:    bf16 [H] or [H, 1]
        rms_weight:    bf16 [H]

    Returns ``(output, rsigma, probs, logits)`` with ``output`` in bf16
    ``[T, B, H]`` and the three saved tensors in fp32 ``[K+1, T, B]``.
    """
    T, B, H = layer_residual.shape
    K = int(block_residual.shape[0])
    N = K + 1
    M = T * B
    layer_flat = layer_residual.reshape(M, H)
    block_flat = block_residual.reshape(K, M, H)
    res_w = res_weight.flatten().float()
    rms_w = rms_weight.flatten().float()

    output = torch.empty_like(layer_residual)
    rsigma = torch.empty((N, T, B), device=layer_residual.device, dtype=torch.float32)
    probs = torch.empty_like(rsigma)
    logits = torch.empty_like(rsigma)

    output_flat = output.reshape(M, H)
    rsigma_flat = rsigma.reshape(N, M)
    probs_flat = probs.reshape(N, M)
    logits_flat = logits.reshape(N, M)
    chunk_m = max(1, max_elements // max(N * H, 1))

    for start in range(0, M, chunk_m):
        end = min(start + chunk_m, M)
        if K == 0:
            values = layer_flat[start:end].unsqueeze(0).float()
        else:
            values = torch.cat(
                [block_flat[:, start:end, :], layer_flat[start:end].unsqueeze(0)],
                dim=0,
            ).float()
        rs = (values.square().mean(dim=-1) + rms_eps).rsqrt()
        lg = (values * rs[..., None] * rms_w * res_w).sum(dim=-1)
        pr = torch.softmax(lg, dim=0)
        out = (pr[..., None] * values).sum(dim=0)
        output_flat[start:end].copy_(out.to(layer_residual.dtype))
        rsigma_flat[:, start:end].copy_(rs)
        probs_flat[:, start:end].copy_(pr)
        logits_flat[:, start:end].copy_(lg)

    return output, rsigma, probs, logits


def apply_attn_res_reference(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
) -> torch.Tensor:
    """HF ``_apply_attn_res`` mirror.

    ``prefix_sum``     is ``(num_tokens, hidden_size)`` and comes from the
                       running sub-block sum.
    ``block_residual`` is ``(num_tokens, num_blocks, hidden_size)`` — the
                       HF layout, block-residual axis in the middle.
    ``proj_weight``    is the ``proj.weight.squeeze(0)`` linear weight of
                       shape ``(hidden_size,)``.
    ``rms_weight``     is the RMSNorm learnable weight of shape
                       ``(hidden_size,)``.

    Returns bf16 ``(num_tokens, hidden_size)`` matching the input dtype.
    """
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + rms_eps)
    score_weight = rms_weight.float() * proj_weight.squeeze().float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)


# ---------------------------------------------------------------------------
# Fused-op module.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AttnResFwdInputs:
    layer_residual: torch.Tensor  # bf16 [T, B, H]
    block_residual: torch.Tensor  # bf16 [K, T, B, H]
    res_weight: torch.Tensor  # bf16 [H] or [H, 1]
    rms_weight: torch.Tensor  # bf16 [H]


class KimiK3AttnResidualOp(nn.Module):
    """Decoder-layer Attention Residual fused op wrapper.

    The op owns the ``proj`` (``nn.Linear(hidden_size, 1, bias=False)``)
    and ``norm`` (``KimiRMSNorm(hidden_size)`` — represented here by a raw
    learnable ``weight`` and ``variance_epsilon``) parameters that
    accompany each residual-selection site (self_attention_res, mlp_res,
    output_attn_res). It exposes:

    * :meth:`forward_hf_layout` — the "HF-friendly" entry point that
      accepts ``(prefix_sum: [M, H], block_residual: [M, K, H])`` and
      returns ``[M, H]``. This is the shape the K3 decoder layer wants.
    * :meth:`forward` — the raw kernel entry point mirroring
      ``attn_res_fwd``: ``(layer_residual, block_residual_kthbh, ...)`` →
      ``(output, rsigma, probs)``.

    Kernel path selection follows the same policy as the KDA module:
    Blackwell sm_100 + a resolvable optimization root ⇒ ``OPTIMIZED``;
    otherwise ``REFERENCE`` (pure-torch chunked reference).
    """

    def __init__(
        self,
        hidden_size: int,
        rms_eps: float = 1e-6,
        force_use_fallback_kernel: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.variance_epsilon = float(rms_eps)
        self.force_use_fallback_kernel = bool(force_use_fallback_kernel)

        # Match HF: KimiRMSNorm.weight has shape [H]; the projection is
        # ``nn.Linear(H, 1, bias=False)``. Storing as raw Parameters lets
        # the module test copy weights directly from the HF module.
        self.rms_weight = nn.Parameter(torch.ones(self.hidden_size))
        self.proj_weight = nn.Parameter(torch.zeros(1, self.hidden_size))

        if (
            force_use_fallback_kernel
            or not is_attn_res_optimized_supported()
            or not is_intree_attn_res_available()
        ):
            self.kernel_path = KimiK3AttnResidualKernelPath.REFERENCE
        else:
            self.kernel_path = KimiK3AttnResidualKernelPath.OPTIMIZED
        self._optimized_extension = None

    # ------------------------------------------------------------------
    # Weight loading helpers.
    # ------------------------------------------------------------------

    def copy_weights_from(
        self,
        hf_norm: nn.Module,
        hf_proj: nn.Module,
    ) -> None:
        """Copy weights from an HF ``(KimiRMSNorm, nn.Linear(H,1))`` pair.

        HF wires the residual-selection site with a ``KimiRMSNorm`` and an
        ``nn.Linear(H, 1, bias=False)`` — ``self_attention_res_{norm,proj}``,
        ``mlp_res_{norm,proj}``, ``output_attn_res_{norm,proj}``.
        """
        with torch.no_grad():
            self.rms_weight.data.copy_(
                hf_norm.weight.detach().to(
                    dtype=self.rms_weight.dtype, device=self.rms_weight.device
                )
            )
            # HF proj.weight has shape [1, H]; we store [1, H] so both
            # layouts stay compatible without a squeeze at call time.
            self.proj_weight.data.copy_(
                hf_proj.weight.detach().to(
                    dtype=self.proj_weight.dtype, device=self.proj_weight.device
                )
            )

    # ------------------------------------------------------------------
    # Raw kernel entry.
    # ------------------------------------------------------------------

    def _validate_inputs(self, inputs: _AttnResFwdInputs) -> Tuple[int, int, int, int]:
        lr = inputs.layer_residual
        br = inputs.block_residual
        rw = inputs.res_weight
        nw = inputs.rms_weight
        if lr.ndim != 3:
            raise ValueError("layer_residual must have shape [T, B, H]")
        T, B, H = lr.shape
        if B != 1:
            raise ValueError(f"attn_res_fwd requires B==1 (got B={B})")
        if H != self.hidden_size:
            raise ValueError(
                f"layer_residual H={H} does not match module hidden_size={self.hidden_size}"
            )
        if lr.dtype != torch.bfloat16 or not lr.is_cuda or not lr.is_contiguous():
            raise ValueError("layer_residual must be a CUDA bf16 contiguous tensor")
        if br.ndim != 4:
            raise ValueError("block_residual must have shape [K, T, B, H]")
        K = int(br.shape[0])
        if tuple(br.shape[1:]) != (T, B, H):
            raise ValueError(
                f"block_residual shape {tuple(br.shape)} does not match [K, T, B, H] for T={T}, B={B}, H={H}"
            )
        if br.dtype != torch.bfloat16 or not br.is_cuda or not br.is_contiguous():
            raise ValueError("block_residual must be a CUDA bf16 contiguous tensor")
        if K + 1 > 12:
            raise ValueError(f"attn_res_fwd requires N=K+1 in [1, 12] (got N={K + 1})")
        if T > 16384:
            raise ValueError(f"attn_res_fwd requires T in [1, 16384] (got T={T})")
        if rw.numel() != H or rw.dtype != torch.bfloat16 or not rw.is_cuda:
            raise ValueError("res_weight must be CUDA bf16 with H elements")
        if nw.numel() != H or nw.dtype != torch.bfloat16 or not nw.is_cuda:
            raise ValueError("rms_weight must be CUDA bf16 with H elements")
        return T, B, H, K

    def _attn_res_fwd_optimized(
        self, inputs: _AttnResFwdInputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_inputs(inputs)
        output, rsigma, probs, _logits = intree_attn_res_fwd(
            inputs.layer_residual,
            inputs.block_residual,
            inputs.res_weight,
            inputs.rms_weight,
            self.variance_epsilon,
        )
        return output, rsigma, probs

    def _attn_res_fwd_reference(
        self, inputs: _AttnResFwdInputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_inputs(inputs)
        out, rsigma, probs, _logits = attn_res_fwd_chunked_reference(
            inputs.layer_residual,
            inputs.block_residual,
            inputs.res_weight,
            inputs.rms_weight,
            self.variance_epsilon,
        )
        return out, rsigma, probs

    @torch.no_grad()
    def forward(
        self,
        layer_residual: torch.Tensor,
        block_residual: torch.Tensor,
        res_weight: Optional[torch.Tensor] = None,
        rms_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Kernel-shaped forward.

        Uses this module's own ``proj_weight`` / ``rms_weight`` as
        weights unless overrides are supplied. Returns
        ``(output, rsigma, probs)`` matching the fused kernel's contract.
        """
        rw = res_weight if res_weight is not None else self.proj_weight
        nw = rms_weight if rms_weight is not None else self.rms_weight
        inputs = _AttnResFwdInputs(
            layer_residual=layer_residual,
            block_residual=block_residual,
            res_weight=rw,
            rms_weight=nw,
        )
        if self.kernel_path == KimiK3AttnResidualKernelPath.OPTIMIZED:
            return self._attn_res_fwd_optimized(inputs)
        return self._attn_res_fwd_reference(inputs)

    # ------------------------------------------------------------------
    # HF-friendly entry point.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_hf_layout(
        self,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the fused residual-selection op in the HF shape layout.

        ``prefix_sum``     is ``(num_tokens, hidden_size)`` bf16 CUDA.
        ``block_residual`` is ``(num_tokens, num_blocks, hidden_size)`` bf16
                           CUDA (HF layout with the block axis in the
                           middle).

        Returns ``(num_tokens, hidden_size)`` bf16.

        Internally reshapes to the kernel's ``[T, B=1, H]`` and
        ``[K, T, B=1, H]`` layout, calls the fused op, and reshapes back.
        """
        if prefix_sum.ndim != 2:
            raise ValueError("prefix_sum must have shape [M, H]")
        M, H = prefix_sum.shape
        if H != self.hidden_size:
            raise ValueError(
                f"prefix_sum H={H} does not match module hidden_size={self.hidden_size}"
            )
        if block_residual.ndim != 3 or block_residual.shape[0] != M or block_residual.shape[2] != H:
            raise ValueError("block_residual must have shape [M, K, H] matching prefix_sum")
        K = int(block_residual.shape[1])

        # Kernel expects [T, B=1, H] and [K, T, B=1, H] bf16 contiguous.
        # HF passes num_tokens as the flat batch axis, so we set T=M, B=1.
        layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
        # HF block_residual [M, K, H] → kernel [K, M, 1, H].
        block_kernel = block_residual.transpose(0, 1).reshape(K, M, 1, H).contiguous()

        output, _rsigma, _probs = self.forward(layer_kernel, block_kernel)
        return output.reshape(M, H)

    # ------------------------------------------------------------------
    # Diagnostics.
    # ------------------------------------------------------------------

    def kernel_source(self) -> str:
        """Return a stable string describing the kernel path in use."""
        if self.kernel_path == KimiK3AttnResidualKernelPath.OPTIMIZED:
            return "<trtllm::attn_res_fwd>"
        return "<reference:attn_res_fwd_chunked_reference>"

    def precompile(self, verbose: bool = False) -> None:
        """No-op: the in-tree ``trtllm::attn_res_fwd`` op is pre-compiled."""
        del verbose
