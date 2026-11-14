# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``torch.ops.mv2_kimi_k3_vl.rms_norm_gated_token_major``: gated RMSNorm with a token-major gate.

The launch of ``trtllm::rms_norm_gated_token_major``: the multi-row kernel for a power-of-two
width up to 256 with dense per-token gate blocks (PDL when the grid is smaller than the SM count),
the one-row kernel on a packed copy of the gate otherwise.
"""

import torch
import triton

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of

from .rms_norm_gated_kernel import (
    _MULTIROW_MAX_N,
    _MULTIROW_NUM_WARPS,
    _MULTIROW_ROWS,
    _layer_norm_fwd,
    _multirow_pdl,
    _rms_norm_gated_fwd_multirow_kernel,
)

_NS = namespace_of(__name__)


@torch.library.custom_op(f"{_NS}::rms_norm_gated_token_major", mutates_args=())
def rms_norm_gated_token_major(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fp8_scale: torch.Tensor | None = None,
    gate_activation: str = "silu",
) -> torch.Tensor:
    """Gated rmsnorm(x) with z read from a 3D token-major view.

    x: [num_tokens * heads, N] with contiguous rows. z: [num_tokens, heads, N]
    whose (heads, N) block is contiguous per token and whose token stride is
    arbitrary (e.g. a column slice of a wider per-token projection). Falls
    back to the generic kernel on a packed copy of z when the shape is not
    eligible for the multi-row kernel.

    When fp8_scale (a scalar fp32 tensor holding the downstream static input
    scale) is given, the output is quantized to float8_e4m3fn in the same
    kernel. ``gate_activation`` is ``"silu"`` for GDN's ``z * sigmoid(z)``
    gate or ``"sigmoid"`` for KDA's ``sigmoid(z)`` gate.
    """
    if gate_activation not in ("silu", "sigmoid"):
        raise ValueError(f"gate_activation must be 'silu' or 'sigmoid', got {gate_activation!r}")
    gate_sigmoid = gate_activation == "sigmoid"
    M, N = x.shape
    num_tokens, heads, n_z = z.shape
    assert n_z == N and num_tokens * heads == M, (
        f"z shape {tuple(z.shape)} does not match x shape {tuple(x.shape)}"
    )
    weight = weight.contiguous()
    eligible = (
        x.stride(-1) == 1
        and z.stride(2) == 1
        and z.stride(1) == N
        and N <= _MULTIROW_MAX_N
        and (N & (N - 1)) == 0
    )
    if not eligible:
        y, _, _ = _layer_norm_fwd(
            x,
            weight,
            None,
            eps,
            z=z.reshape(M, N),
            norm_before_gate=True,
            is_rms_norm=True,
            fp8_scale=fp8_scale,
            gate_activation=gate_activation,
        )
        return y
    out_dtype = torch.float8_e4m3fn if fp8_scale is not None else x.dtype
    out = torch.empty_like(x, dtype=out_dtype)
    grid_size = triton.cdiv(M, _MULTIROW_ROWS)
    device_index = x.device.index
    launch_with_pdl = _multirow_pdl(grid_size, device_index)
    with torch.cuda.device(device_index):
        _rms_norm_gated_fwd_multirow_kernel[(grid_size,)](
            x,
            out,
            weight,
            z,
            None,
            fp8_scale,
            x.stride(0),
            out.stride(0),
            z.stride(0),
            M,
            eps,
            N=N,
            ROWS=_MULTIROW_ROWS,
            HEADS_PER_TOK=heads,
            GATE_SIGMOID=gate_sigmoid,
            WEIGHT_IS_DELTA=False,
            LAUNCH_WITH_PDL=launch_with_pdl,
            num_warps=_MULTIROW_NUM_WARPS,
            launch_pdl=launch_with_pdl,
        )
    return out


@rms_norm_gated_token_major.register_fake
def _(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fp8_scale: torch.Tensor | None = None,
    gate_activation: str = "silu",
) -> torch.Tensor:
    del z, weight, eps, gate_activation
    out_dtype = torch.float8_e4m3fn if fp8_scale is not None else x.dtype
    return torch.empty_like(x, dtype=out_dtype)
