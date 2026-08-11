# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch wrapper for KDA decode fusion."""

from __future__ import annotations

import torch

_DUMMY_CACHE: dict = {}


def _require_cuda_bf16(name: str, tensor: torch.Tensor) -> None:
    """Validate that a tensor is CUDA bf16."""
    if not tensor.is_cuda or tensor.dtype is not torch.bfloat16:
        raise TypeError(f"{name} must be a CUDA bfloat16 tensor")


def _require_cuda_fp32(name: str, tensor: torch.Tensor) -> None:
    """Validate that a tensor is CUDA fp32."""
    if not tensor.is_cuda or tensor.dtype is not torch.float32:
        raise TypeError(f"{name} must be a CUDA float32 tensor")


def _dummy_tensor(
    tag: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    fill: float = 0.0,
) -> torch.Tensor:
    """Return a cached dummy tensor for optional CUDA arguments."""
    device = torch.device(device)
    key = (tag, shape, dtype, device.type, device.index, fill)
    tensor = _DUMMY_CACHE.get(key)
    if tensor is None:
        if fill == 0.0:
            tensor = torch.zeros(shape, dtype=dtype, device=device)
        elif fill == 1.0:
            tensor = torch.ones(shape, dtype=dtype, device=device)
        else:
            tensor = torch.full(shape, fill, dtype=dtype, device=device)
        _DUMMY_CACHE[key] = tensor
    return tensor


def run_kda_decode_fusion_cuda(
    *,
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    w_q_t: torch.Tensor,
    w_k_t: torch.Tensor,
    w_v_t: torch.Tensor,
    bias_q: torch.Tensor | None,
    bias_k: torch.Tensor | None,
    bias_v: torch.Tensor | None,
    cs_q: torch.Tensor,
    cs_k: torch.Tensor,
    cs_v: torch.Tensor,
    A_log: torch.Tensor,
    g: torch.Tensor,
    dt_bias: torch.Tensor | None,
    beta: torch.Tensor,
    state: torch.Tensor,
    onorm_g: torch.Tensor | None = None,
    onorm_weight: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = 128**-0.5,
    onorm_eps: float = 1e-5,
    lower_bound: float | None = None,
    use_beta_sigmoid_in_kernel: bool = True,
    verbose: bool = False,
    update_conv_cache: bool = False,
) -> torch.Tensor:
    """Run CUDA KDA decode fusion for the tuned decode shapes.

    ``ssm_state_indices=None`` selects the tuned batch-local static layout,
    while a tensor selects the indexed state-pool layout.
    """
    for name, tensor in (
        ("x_q", x_q),
        ("x_k", x_k),
        ("x_v", x_v),
        ("w_q_t", w_q_t),
        ("w_k_t", w_k_t),
        ("w_v_t", w_v_t),
        ("cs_q", cs_q),
        ("cs_k", cs_k),
        ("cs_v", cs_v),
        ("g", g),
        ("beta", beta),
    ):
        _require_cuda_bf16(name, tensor)
    for name, tensor in (("A_log", A_log), ("state", state)):
        _require_cuda_fp32(name, tensor)

    if x_q.ndim != 4 or x_k.ndim != 4 or x_v.ndim != 4:
        raise ValueError("x_q, x_k, and x_v must be rank-4 decode tensors")
    if x_q.shape[0] != 1 or x_k.shape[0] != 1 or x_v.shape[0] != 1:
        raise ValueError("only T=1 decode inputs are supported")
    if x_q.shape[-1] != 128 or x_k.shape[-1] != 128 or x_v.shape[-1] != 128:
        raise ValueError("only K=128 and V=128 are supported")

    B = x_q.shape[1]
    H = x_q.shape[2]
    HV = x_v.shape[2]
    if x_k.shape[1:3] != (B, H) or x_v.shape[1] != B:
        raise ValueError("x_q, x_k, and x_v batch/head dimensions are inconsistent")
    if H != HV or H not in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96):
        raise ValueError(
            "CUDA KDA decode fusion supports H == HV in {1,2,3,4,6,8,12,16,24,32,48,96}"
        )
    if ssm_state_indices is None and not state.is_contiguous():
        raise ValueError("state must be contiguous because it is updated in place")
    if out is not None:
        _require_cuda_bf16("out", out)
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if tuple(out.shape) != (B, 1, HV, 128):
            raise ValueError("out must have shape [B, 1, HV, 128]")

    device = x_q.device
    apply_onorm = onorm_g is not None
    if bias_q is None:
        bias_q = _dummy_tensor("bias_q", (H * 128,), torch.bfloat16, device)
    if bias_k is None:
        bias_k = _dummy_tensor("bias_k", (H * 128,), torch.bfloat16, device)
    if bias_v is None:
        bias_v = _dummy_tensor("bias_v", (HV * 128,), torch.bfloat16, device)
    if dt_bias is None:
        dt_bias = _dummy_tensor("dt_bias", (H * 128,), torch.float32, device)
    if onorm_g is None:
        onorm_g = _dummy_tensor("onorm_g", (1, B, HV, 128), torch.bfloat16, device)
    if onorm_weight is None:
        onorm_weight = _dummy_tensor("onorm_weight", (128,), torch.float32, device, fill=1.0)

    for name, tensor in (
        ("bias_q", bias_q),
        ("bias_k", bias_k),
        ("bias_v", bias_v),
        ("onorm_g", onorm_g),
    ):
        _require_cuda_bf16(name, tensor)
    for name, tensor in (("dt_bias", dt_bias), ("onorm_weight", onorm_weight)):
        _require_cuda_fp32(name, tensor)

    if update_conv_cache:
        q_stride = H * 128
        v_stride = HV * 128
        if not (
            cs_q.stride(1) == 1
            and cs_k.stride(1) == 1
            and cs_v.stride(1) == 1
            and cs_q.stride(2) == q_stride
            and cs_k.stride(2) == q_stride
            and cs_v.stride(2) == v_stride
        ):
            raise ValueError(
                "update_conv_cache expects transposed conv-state layout: "
                "shape [B, dim, 3], stride(1)=1, stride(2)=dim"
            )

    if cu_seqlens is None:
        cu_seqlens = torch.arange(B + 1, dtype=torch.int32, device=device)
    else:
        if not cu_seqlens.is_cuda or cu_seqlens.dtype is not torch.int32:
            raise TypeError("cu_seqlens must be a CUDA int32 tensor")
        if tuple(cu_seqlens.shape) != (B + 1,):
            raise ValueError("cu_seqlens must have shape [B + 1]")
        cu_seqlens = cu_seqlens.contiguous()

    args = (
        x_q.contiguous(),
        x_k.contiguous(),
        x_v.contiguous(),
        w_q_t.contiguous(),
        w_k_t.contiguous(),
        w_v_t.contiguous(),
        bias_q.contiguous(),
        bias_k.contiguous(),
        bias_v.contiguous(),
        cs_q if update_conv_cache else cs_q.contiguous(),
        cs_k if update_conv_cache else cs_k.contiguous(),
        cs_v if update_conv_cache else cs_v.contiguous(),
        A_log.contiguous(),
        g.contiguous(),
        dt_bias.contiguous(),
        beta.contiguous(),
        onorm_g.contiguous(),
        onorm_weight.contiguous(),
        ssm_state_indices,
        cu_seqlens,
        state,
    )

    use_lower_bound = lower_bound is not None
    lower_bound_value = 0.0 if lower_bound is None else float(lower_bound)
    launch_args = (
        bool(apply_onorm),
        bool(update_conv_cache),
        bool(use_lower_bound),
        bool(use_beta_sigmoid_in_kernel),
        lower_bound_value,
        float(scale),
        float(onorm_eps),
    )
    return torch.ops.trtllm.kda_decode(*args, *launch_args, output=out)
