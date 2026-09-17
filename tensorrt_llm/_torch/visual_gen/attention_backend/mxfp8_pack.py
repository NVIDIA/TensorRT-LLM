# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Automatic strided MXFP8 packing for SM100 BF16 attention tensors."""

import threading
from typing import Literal

import torch

_COMPILED = {}
_COMPILE_LOCK = threading.Lock()


def metadata_eligible(x: torch.Tensor) -> bool:
    """Keep short sequences and unsupported layouts on the compiler-visible native path."""
    return (
        x.ndim == 4
        and x.dtype == torch.bfloat16
        and x.is_cuda
        and x.shape[3] == 128
        and x.shape[2] >= 4096
        # CUDA grid = (four CTAs per 128-row tile, batch * heads, 1).
        and (x.shape[2] + 127) // 128 * 4 <= 2**31 - 1
        and x.shape[0] * x.shape[1] <= 65535
        and all(n > 0 for n in x.shape)
        and x.stride(-1) == 1
        and all(s > 0 and s % 8 == 0 for s in x.stride()[:-1])
        and not x.requires_grad
        and torch.cuda.get_device_capability(x.device) == (10, 0)
    )


def _runtime_eligible(x: torch.Tensor) -> bool:
    # Pointer checks stay inside the opaque operator for torch.compile.
    return (
        metadata_eligible(x)
        and x.data_ptr() % 16 == 0
        and sum((int(n) - 1) * int(s) for n, s in zip(x.shape, x.stride())) < 2**62
    )


def _allocate(x: torch.Tensor, kind: Literal["qk", "v"]) -> tuple[torch.Tensor, torch.Tensor]:
    b, h, s, _ = x.shape
    padded_s = (s + 127) // 128 * 128
    payload = torch.empty(
        (b, h, padded_s if kind == "qk" else s, 128), device=x.device, dtype=torch.uint8
    )
    scales = torch.empty((b * h * padded_s * 4,), device=x.device, dtype=torch.uint8)
    return payload, scales


def _views(
    payload: torch.Tensor, scales: torch.Tensor, s: int, kind: Literal["qk", "v"]
) -> tuple[torch.Tensor, torch.Tensor]:
    b, h = payload.shape[:2]
    padded_s = (s + 127) // 128 * 128
    data = payload.view(torch.float8_e4m3fn)
    if kind == "qk":
        return data[:, :, :s, :], scales.view(b, h, padded_s, 4)
    return data, scales.view(b, h, 128, padded_s // 32).permute(0, 1, 3, 2)


def _pack(x: torch.Tensor, kind: Literal["qk", "v"]) -> tuple[torch.Tensor, torch.Tensor]:
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from ...cute_dsl_kernels.blackwell import mxfp8_attention_pack as kernels

    payload, scales = _allocate(x, kind)
    with torch.cuda.device(x.device):
        key = (kind, tuple(x.shape), tuple(x.stride()), x.device.index)
        compiled = _COMPILED.get(key)
        if compiled is None:
            with _COMPILE_LOCK:
                compiled = _COMPILED.get(key)
                if compiled is None:
                    tensors = [from_dlpack(t, assumed_align=16) for t in (x, payload, scales)]
                    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
                    compiled = cute.compile(
                        kernels.launch_qk if kind == "qk" else kernels.launch_v,
                        *tensors,
                        stream=stream,
                        options="--opt-level 2 --enable-tvm-ffi",
                    )
                    _COMPILED[key] = compiled
        compiled(x, payload, scales)
    return _views(payload, scales, x.shape[2], kind)


@torch.library.custom_op(
    "trtllm::visual_gen_mxfp8_qk",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def pack_qk(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack aligned long BF16 [B, H, S, 128] tensors, with native fallback."""
    if _runtime_eligible(x):
        return _pack(x, "qk")
    from .cudnn import _quantize_mxfp8_qk_native

    return _quantize_mxfp8_qk_native(x)


@pack_qk.register_fake
def _fake_qk(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    payload, scales = _allocate(x, "qk")
    return _views(payload, scales, x.shape[2], "qk")


@torch.library.custom_op(
    "trtllm::visual_gen_mxfp8_v",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def pack_v(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack V along S while preserving the native payload and scale-factor layouts."""
    if _runtime_eligible(x):
        return _pack(x, "v")
    from .cudnn import _quantize_mxfp8_v_native

    return _quantize_mxfp8_v_native(x)


@pack_v.register_fake
def _fake_v(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    payload, scales = _allocate(x, "v")
    return _views(payload, scales, x.shape[2], "v")
