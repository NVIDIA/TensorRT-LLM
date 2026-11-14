# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/layernorm_gated.py
# Copyright (c) 2024, Tri Dao, Albert Gu.
#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``trtllm::k3_rms_norm_gated_token_major``: the Kimi K3 targets' copy of ``trtllm::rms_norm_gated_token_major``
(``tensorrt_llm/_torch/modules/mamba/layernorm_gated.py``) under its own name, so both can be registered in one
process. Only what that op runs is copied: the token-major multi-row path and its fallback through
``_layer_norm_fwd``.
"""

import functools
import os

import torch
import triton

from .k3_rms_norm_gated_kernel import (
    _layer_norm_fwd_1pass_kernel,
    _rms_norm_gated_fwd_multirow_kernel,
)

# Rows per program of the multi-row gated-RMSNorm kernel. At the GDN decode
# shape (thousands of 128-element rows) one row per CTA leaves the kernel
# launch-limited; 4 rows with 4 warps reproduces the single-row kernel's
# reduction order (bitwise-identical output) at ~2x the throughput.
_MULTIROW_ROWS = 4
_MULTIROW_NUM_WARPS = 4
_MULTIROW_MAX_N = 256


@functools.lru_cache(maxsize=None)
def _pdl_device_policy(device_index: int) -> tuple[bool, int]:
    """Whether PDL is allowed on this device, and its SM count."""
    props = torch.cuda.get_device_properties(device_index)
    sm_version = props.major * 10 + props.minor
    allowed = sm_version >= 90 and os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    return allowed, props.multi_processor_count


def _multirow_pdl(grid_size: int, device_index: int) -> bool:
    """Use PDL only when the launch grid cannot cover all SMs by itself."""
    allowed, num_sms = _pdl_device_policy(device_index)
    return allowed and grid_size < num_sms


def _multirow_gated_rmsnorm_eligible(N, ngroups, bias, z, norm_before_gate, is_rms_norm):
    return (
        is_rms_norm
        and norm_before_gate
        and z is not None
        and bias is None
        and ngroups == 1
        and N <= _MULTIROW_MAX_N
        and (N & (N - 1)) == 0
    )


@torch.library.custom_op("trtllm::k3_rms_norm_gated_token_major", mutates_args=())
def k3_rms_norm_gated_token_major(
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
            # Dependents (out_proj) prefetch their weights; they read Y only
            # after their own griddepcontrol.wait.
            EARLY_TRIGGER=_pdl_device_policy(device_index)[0],
            launch_pdl=launch_with_pdl,
        )
    return out


@k3_rms_norm_gated_token_major.register_fake
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


def _layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    fp8_scale=None,
    gate_activation="silu",
    weight_is_delta=False,
):
    if gate_activation not in ("silu", "sigmoid"):
        raise ValueError(f"gate_activation must be 'silu' or 'sigmoid', got {gate_activation!r}")
    gate_sigmoid = gate_activation == "sigmoid"
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out_dtype = torch.float8_e4m3fn if fp8_scale is not None else x.dtype
        out = torch.empty_like(x, dtype=out_dtype)
    assert out.stride(-1) == 1
    if fp8_scale is not None:
        assert fp8_scale.dtype == torch.float32
        assert fp8_scale.numel() == 1
        assert out.dtype == torch.float8_e4m3fn
    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    if _multirow_gated_rmsnorm_eligible(
        group_size, ngroups, bias, z, norm_before_gate, is_rms_norm
    ):
        grid_size = triton.cdiv(M, _MULTIROW_ROWS)
        device_index = x.device.index
        launch_with_pdl = _multirow_pdl(grid_size, device_index)
        with torch.cuda.device(device_index):
            _rms_norm_gated_fwd_multirow_kernel[(grid_size,)](
                x,
                out,
                weight,
                z,
                rstd,
                fp8_scale,
                x.stride(0),
                out.stride(0),
                z.stride(0),
                M,
                eps,
                N=group_size,
                ROWS=_MULTIROW_ROWS,
                HEADS_PER_TOK=1,
                GATE_SIGMOID=gate_sigmoid,
                WEIGHT_IS_DELTA=weight_is_delta,
                LAUNCH_WITH_PDL=launch_with_pdl,
                num_warps=_MULTIROW_NUM_WARPS,
                launch_pdl=launch_with_pdl,
            )
        return out, mean, rstd
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M, ngroups)
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[grid](
            x,
            out,
            weight,
            bias,
            z,
            mean,
            rstd,
            fp8_scale,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_N=BLOCK_N,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            GATE_SIGMOID=gate_sigmoid,
            WEIGHT_IS_DELTA=weight_is_delta,
            num_warps=num_warps,
        )
    return out, mean, rstd
