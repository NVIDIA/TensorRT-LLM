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

"""Autotunable MXFP8 GEMMs with the weight and activation operands swapped.

The weight rows occupy MMA M and the activation rows occupy N. The kernel writes
column-major ``[N, M]`` into row-major ``[M, N]`` storage, avoiding a transpose copy.
Both operands use E4M3 values and swizzled UE8M0 scales with 32 values per block.

Supported inputs admit every tile below as an autotuning candidate alongside
FlashInfer, without a model-specific shape policy.
"""

from __future__ import annotations

import torch

from ..autotuner import OptimizationProfile, TunableRunner, TuningConfig
from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

# Candidate (MMA tiler (M, N), cluster (M, N)) configurations.
_Tile = tuple[tuple[int, int], tuple[int, int]]
_TILE_CONFIGS: tuple[_Tile, ...] = (((256, 64), (2, 1)), ((256, 192), (4, 1)))
_COMPILED: dict = {}


def _pad(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _supports_swapab_device(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) in (
        (10, 0),
        (10, 3),
        (10, 7),
    )


class MXFP8SwapABGemmRunner(TunableRunner):
    """Profile the supported Swap-AB tiles as an independent MXFP8 backend."""

    # Use exact input shapes: 192 and 256 rows can require different tile counts.
    tuning_config = TuningConfig()
    kernel_cache = _COMPILED

    def __init__(self, output_dtype: torch.dtype) -> None:
        self.output_dtype = output_dtype

    def unique_id(self) -> tuple[torch.dtype]:
        return (self.output_dtype,)

    def get_valid_tactics(
        self, inputs: list[torch.Tensor], profile: OptimizationProfile | None, **kwargs
    ) -> list[int]:
        act, act_scale, weight, weight_scale = inputs
        if (
            self.output_dtype != torch.bfloat16
            or act.ndim != 2
            or weight.ndim != 2
            or act.shape[1] != weight.shape[1]
        ):
            return []
        m, k = act.shape
        n = weight.shape[0]
        # MXFP8 scales cover 32 K elements. The column-major BF16 epilogue
        # requires a 16-byte-aligned stride (eight output columns).
        if min(m, n, k) <= 0 or k % 32 or n % 8:
            return []
        if (
            not IS_CUTLASS_DSL_AVAILABLE
            or not _supports_swapab_device(act.device)
            or act.dtype != torch.float8_e4m3fn
            or weight.dtype != torch.float8_e4m3fn
            or act_scale.dtype not in (torch.uint8, torch.float8_e8m0fnu)
            or weight_scale.dtype not in (torch.uint8, torch.float8_e8m0fnu)
            or any(t.device != act.device or not t.is_contiguous() for t in inputs)
            or any(t.data_ptr() % alignment for t, alignment in zip(inputs, (32, 16, 32, 16)))
            or act_scale.numel() < _pad(act.shape[0], 128) * _pad(act.shape[1] // 32, 4)
            or weight_scale.numel() < _pad(weight.shape[0], 128) * _pad(weight.shape[1] // 32, 4)
        ):
            return []
        # Distinct tile IDs also let the autotuner prime every cached JIT variant.
        return list(range(len(_TILE_CONFIGS)))

    def forward(self, inputs: list[torch.Tensor], *, tactic: int = -1) -> torch.Tensor:
        valid_tactics = self.get_valid_tactics(inputs, None)
        if not valid_tactics or (tactic != -1 and tactic not in valid_tactics):
            raise ValueError("Unsupported inputs or tactic for the MXFP8 Swap-AB GEMM")
        if tactic == -1:
            tactic = valid_tactics[0]
        act, act_scale, weight, weight_scale = inputs
        return mxfp8_swapab_gemm(
            act,
            act_scale.view(torch.uint8),
            weight,
            weight_scale.view(torch.uint8),
            _TILE_CONFIGS[tactic],
        )


def mxfp8_swapab_gemm(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    tile: _Tile,
) -> torch.Tensor:
    """Compute ``act[M, K] @ weight[N, K].T`` with BF16 output using a swapped GEMM.

    Inputs are contiguous E4M3 tensors with their swizzled UE8M0 block scales.
    The compiled kernel must be warmed up before CUDA graph capture.
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    from ..cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )

    m, k = act.shape
    n = weight.shape[0]
    tiler, cluster = tile
    out = torch.empty((m, n), dtype=torch.bfloat16, device=act.device)
    sf_m, sf_n, sf_k = _pad(m, 128), _pad(n, 128), _pad(k // 32, 4)

    def ptr(t: torch.Tensor, dtype: type[cutlass.Numeric], align: int) -> cute.Pointer:
        return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)

    args = (
        ptr(weight, cutlass.Float8E4M3FN, 32),
        ptr(act, cutlass.Float8E4M3FN, 32),
        ptr(weight_scale, cutlass.Float8E8M0FNU, 16),
        ptr(act_scale, cutlass.Float8E8M0FNU, 16),
        ptr(out, cutlass.BFloat16, 16),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (tiler, cluster, act.device.index)
    entry = _COMPILED.get(key)
    if entry is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Compile the swap-AB MXFP8 GEMM before CUDA graph capture")
        alpha = torch.ones(1, dtype=torch.float32, device=act.device)
        gemm = Sm100BlockScaledPersistentDenseGemmKernel(32, tiler, cluster, False)
        clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1])
        compiled = cute.compile(
            gemm.wrapper,
            n,
            m,
            k,
            sf_n // 128,
            sf_m // 128,
            sf_k // 4,
            1,
            *args,
            cute.runtime.from_dlpack(alpha),
            clusters,
            stream,
            True,
            options="--opt-level 2",
        )
        entry = _COMPILED[key] = (compiled, alpha, cute.runtime.from_dlpack(alpha))
    compiled, _, alpha_cute = entry
    compiled(n, m, k, sf_n // 128, sf_m // 128, sf_k // 4, *args, alpha_cute, stream)
    return out
