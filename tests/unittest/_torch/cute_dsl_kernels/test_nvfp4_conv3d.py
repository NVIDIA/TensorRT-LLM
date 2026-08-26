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

import math

import cutlass
import pytest
import torch

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.conv.dense_blockscaled_implicit_gemm_fprop import (
    run,
)


def _require_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        pytest.skip(f"NVFP4 Conv3d requires SM100-family, got sm_{major}{minor}")


def test_nvfp4_conv3d_bias_residual_epilogue_matches_reference() -> None:
    """Exercise the product tactic and BF16 epilogue against the kernel reference."""
    _require_sm100()

    runtime_us = run(
        ncdhw=(1, 128, 3, 12, 16),
        ktrs=(256, 3, 3, 3),
        stride_dhw=(1, 1, 1),
        upper_pad_dhw=(0, 1, 1),
        lower_pad_dhw=(0, 1, 1),
        dil_dhw=(1, 1, 1),
        ab_dtype=cutlass.Float4E2M1FN,
        d_dtype=cutlass.BFloat16,
        acc_dtype=cutlass.Float32,
        sf_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=16,
        mma_tiler_mn=(256, 256),
        preferred_cluster_shape_mn=(2, 1),
        fallback_cluster_shape_mn=(2, 1),
        use_2cta_instrs=True,
        use_bias=True,
        beta=1.0,
        tolerance=1e-2,
        warmup_iterations=1,
        iterations=2,
        skip_ref_check=False,
    )

    # ``skip_ref_check=False`` validates the result against the provider's BF16 reference.
    assert runtime_us is not None and math.isfinite(runtime_us) and runtime_us > 0
