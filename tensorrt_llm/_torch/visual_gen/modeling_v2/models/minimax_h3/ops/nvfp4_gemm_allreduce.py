# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch

_native_runners = {}


@torch.library.custom_op("trtllm::minimax_h3_v2_nvfp4_gemm_allreduce", mutates_args=())
def nvfp4_gemm_allreduce(
    act_fp4: torch.Tensor,
    weight_fp4: torch.Tensor,
    act_sf: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    tp_rank: int,
    tp_group: List[int],
) -> torch.Tensor:
    key = (output_dtype, tuple(tp_group))
    if key not in _native_runners:
        _native_runners[key] = torch.classes.trtllm.Fp4GemmAllreduceRunner(
            output_dtype, tp_rank, tp_group
        )
    return _native_runners[key].run_gemm(act_fp4, weight_fp4, act_sf, weight_scale, alpha, -1)


@nvfp4_gemm_allreduce.register_fake
def _fake_nvfp4_gemm_allreduce(
    act_fp4: torch.Tensor,
    weight_fp4: torch.Tensor,
    act_sf: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    tp_rank: int,
    tp_group: List[int],
) -> torch.Tensor:
    return act_fp4.new_empty((act_fp4.size(0), weight_fp4.size(0)), dtype=output_dtype)
