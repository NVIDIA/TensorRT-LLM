# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense MLP helper for the in-tree Kimi K3 MoE block.

Kimi K3 uses the ``situ`` activation (not SiLU/SwiGLU). ``SituAndMul``
computes ``beta * tanh(gate / beta) * sigmoid(gate)`` on the gate half
and optionally applies ``linear_beta * tanh(up / linear_beta)`` on the
up half, then multiplies. ``KimiK3MLP`` is the fused ``gate_up_proj +
down_proj`` layout used by the shared expert stack in HF
``KimiSparseMoeBlock`` — the same shape TRT-LLM's ``GatedMLP`` uses.

Two activation paths coexist:

* the eager fp32 ``SituAndMul`` module — the byte-exact HF reference,
  used by the parity-test MoE block and as the fallback;
* the fused Triton ``trtllm::situ_and_mul`` custom op (same fp32 math
  in a single kernel, modeled on ``modules/swiglu.py``'s
  ``silu_and_mul_kernel``), enabled per ``KimiK3MLP`` instance via
  ``use_fused_activation=True`` (the runtime model opts in). The op is
  CUDA-graph-safe: no host synchronization and no data-dependent
  control flow.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]
import triton.language.extra.libdevice as tldevice  # type: ignore[import]
from torch import nn

from ...flashinfer_utils import IS_FLASHINFER_AVAILABLE

# Route the RMSNorm forward through flashinfer's single-kernel fused RMSNorm
# instead of the eager pow/mean/rsqrt/mul/cast chain. Set to "0" to fall back
# to the eager reference (the exact-parity rollback lever).
_FUSED_RMSNORM = os.environ.get("KIMI_K3_FUSED_RMSNORM", "1") == "1"


class SituAndMul(nn.Module):
    """K3 SiTU activation with gate/up multiplicative gating.

    Byte-identical to HF ``modeling_kimi.py``'s ``SituAndMul`` at
    lines 41-59. Runs the math in fp32 for numerical stability
    (matches HF), then casts back to the input's dtype.
    """

    def __init__(
        self,
        *,
        beta: float = 1.0,
        linear_beta: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d].to(torch.float32)
        up = x[..., d:].to(torch.float32)
        situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (situ_a * up).to(x.dtype)


@triton.jit
def situ_and_mul_kernel(
    o_ptr,
    o_stride,
    x_ptr,
    x_stride,
    d,
    beta,
    linear_beta,
    BLOCK_SIZE: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
) -> None:
    """Fused :class:`SituAndMul` on a packed ``[gate | up]`` row layout.

    Loads ``gate = x[i, :d]`` and ``up = x[i, d:2d]``, computes (fp32)
    ``beta * tanh(gate / beta) * sigmoid(gate) * up'`` with
    ``up' = linear_beta * tanh(up / linear_beta)`` when
    ``HAS_LINEAR_BETA`` else ``up``, and stores the product rounded to
    ``o_ptr``'s element type.
    """
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)

    o_row_ptr = o_ptr + o_stride * i
    x_row_ptr = x_ptr + x_stride * i

    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    gate = tl.load(x_row_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(x_row_ptr + offsets + d, mask=mask).to(tl.float32)

    situ_a = beta * tldevice.tanh(gate / beta) * tl.sigmoid(gate)
    if HAS_LINEAR_BETA:
        up = linear_beta * tldevice.tanh(up / linear_beta)
    result = situ_a * up

    tl.store(o_row_ptr + offsets, result, mask=mask)


@torch.library.custom_op("trtllm::situ_and_mul", mutates_args=())
def situ_and_mul(x: torch.Tensor, beta: float, linear_beta: Optional[float] = None) -> torch.Tensor:
    """Fused SiTU activation (single Triton kernel, fp32 internal math).

    Args:
        x: ``[num_tokens, 2 * d]`` packed ``[gate | up]`` GEMM output
           (fp16/bf16/fp32; the last dim must be contiguous).
        beta: SiTU gate ``beta`` (``activation_situ_beta``).
        linear_beta: optional up-half ``linear_beta``
           (``activation_situ_linear_beta``); ``None`` keeps the up half
           linear.

    Returns:
        ``[num_tokens, d]`` tensor in ``x``'s dtype, numerically matching
        the eager :class:`SituAndMul` reference.
    """
    b, n = x.shape

    assert n % 2 == 0
    d = n // 2

    o = torch.empty((b, d), dtype=x.dtype, device=x.device)

    def grid(meta: Mapping[str, int]) -> tuple[int, int]:
        return (b, triton.cdiv(d, meta["BLOCK_SIZE"]))

    situ_and_mul_kernel[grid](
        o_ptr=o,
        o_stride=o.stride(0),
        x_ptr=x,
        x_stride=x.stride(0),
        d=d,
        beta=float(beta),
        linear_beta=float(linear_beta) if linear_beta is not None else 1.0,
        BLOCK_SIZE=1024,
        HAS_LINEAR_BETA=linear_beta is not None,
    )

    return o


@situ_and_mul.register_fake
def _(x: torch.Tensor, beta: float, linear_beta: Optional[float] = None) -> torch.Tensor:
    b, n = x.shape

    assert n % 2 == 0

    return x.new_empty((b, n // 2))


class NonSituActivation(nn.Module):
    """SiLU/SwiGLU activation used as the non-SiTU mutation control.

    Splits the last dim into gate/up, applies SiLU to the gate, and
    multiplies element-wise. Deliberately does NOT use the SiTU
    ``beta * tanh(gate/beta) * sigmoid(gate)`` recipe.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d]
        up = x[..., d:]
        return torch.nn.functional.silu(gate) * up


class KimiK3MLP(nn.Module):
    """K3 dense/shared-expert MLP module with TRT-LLM-style fused layout.

    Weight layout:

    * ``gate_up_proj``: ``nn.Linear(hidden_size, 2 * intermediate_size, bias=False)``.
      Rows ``[:intermediate_size]`` correspond to HF's ``gate`` (KimiMLP.gate_proj
      or KimiBlockSparseMLP.w1). Rows ``[intermediate_size:]`` correspond to
      HF's ``up`` (KimiMLP.up_proj or KimiBlockSparseMLP.w3).
    * ``down_proj``: ``nn.Linear(intermediate_size, hidden_size, bias=False)``.
      Matches HF ``KimiMLP.down_proj`` or ``KimiBlockSparseMLP.w2``.

    Forward: ``down_proj( activation( gate_up_proj(x) ) )``. Default
    ``activation`` is :class:`SituAndMul`; pass a different callable to
    run mutation controls (e.g. :class:`NonSituActivation` for a
    negative-control test). ``use_fused_activation=True`` routes CUDA
    inputs through the fused Triton ``trtllm::situ_and_mul`` op instead
    of the eager module (only valid with the default SiTU activation).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        situ_beta: float = 4.0,
        situ_linear_beta: Optional[float] = 25.0,
        activation: Optional[nn.Module] = None,
        use_fused_activation: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if use_fused_activation and activation is not None:
            raise ValueError(
                "use_fused_activation only fuses the default SiTU activation; "
                "drop the custom activation module or the flag"
            )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.use_fused_activation = use_fused_activation

        self.gate_up_proj = nn.Linear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.activation = (
            activation
            if activation is not None
            else SituAndMul(beta=situ_beta, linear_beta=situ_linear_beta)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.gate_up_proj(x)
        if self.use_fused_activation and h1.is_cuda:
            act = self.activation
            h2 = torch.ops.trtllm.situ_and_mul(
                h1.reshape(-1, h1.shape[-1]), act.beta, act.linear_beta
            ).reshape(*h1.shape[:-1], self.intermediate_size)
        else:
            h2 = self.activation(h1)
        return self.down_proj(h2)


class KimiK3RMSNorm(nn.Module):
    """RMSNorm matching HF ``KimiRMSNorm`` semantics exactly.

    HF ``KimiRMSNorm.forward``::

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return self.weight * hidden_states.to(input_dtype)

    ``self.weight`` in HF is initialised in the module's ambient dtype
    (bf16 or fp32). Callers pin the weight dtype here too so byte-exact
    parity holds regardless of the ambient dtype.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # flashinfer's fused RMSNorm does the same fp32-accumulate
        # normalization in one kernel, collapsing the eager
        # pow/mean/rsqrt/mul/cast launch chain. Rounding differs by one final
        # cast: flashinfer multiplies by ``weight`` in fp32 and casts once at
        # the end, while the eager path casts the normalized value to the
        # input dtype BEFORE the weight multiply — so outputs can differ by
        # ~1 ulp and byte-exact HF parity requires the eager path. It is only
        # valid for a CUDA fp16/bf16 input whose dtype matches the weight;
        # CPU / fp32 parity paths, meta init, and the KIMI_K3_FUSED_RMSNORM=0
        # rollback keep the exact eager math below.
        if (
            _FUSED_RMSNORM
            and IS_FLASHINFER_AVAILABLE
            and hidden_states.is_cuda
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
            and self.weight.dtype == hidden_states.dtype
        ):
            from ...custom_ops import flashinfer_rmsnorm

            return flashinfer_rmsnorm(hidden_states.contiguous(), self.weight, self.eps)
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.eps)
        return self.weight * h.to(input_dtype)
