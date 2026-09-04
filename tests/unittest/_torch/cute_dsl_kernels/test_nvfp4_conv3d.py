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


def _require_supported_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) not in ((10, 0), (10, 3)):
        pytest.skip(f"NVFP4 Conv3d requires SM100 or SM103, got SM{major}{minor}")


@pytest.mark.parametrize(
    ("ncdhw", "ktrs", "upper_pad_dhw", "lower_pad_dhw", "tactic"),
    [
        pytest.param(
            (1, 128, 3, 8, 10),
            (128, 3, 3, 3),
            (0, 1, 2),
            (0, 1, 0),
            {
                "mma_tiler_mn": (128, 64),
                "preferred_cluster_shape_mn": (1, 1),
                "fallback_cluster_shape_mn": (1, 1),
                "use_2cta_instrs": False,
            },
            id="128x64-1cta-asymmetric-padding",
        ),
        pytest.param(
            (1, 256, 3, 12, 16),
            (256, 3, 3, 3),
            (0, 1, 1),
            (0, 1, 1),
            {
                "mma_tiler_mn": (256, 256),
                "preferred_cluster_shape_mn": (2, 1),
                "fallback_cluster_shape_mn": (2, 1),
                "use_2cta_instrs": True,
            },
            id="256x256-2cta",
        ),
    ],
)
@pytest.mark.parametrize(
    ("use_bias", "beta"),
    [
        pytest.param(False, 0.0, id="no-epilogue"),
        pytest.param(True, 0.0, id="bias"),
        pytest.param(False, 1.0, id="residual"),
        pytest.param(True, 1.0, id="bias-residual"),
    ],
)
def test_nvfp4_conv3d_reference(
    ncdhw: tuple[int, int, int, int, int],
    ktrs: tuple[int, int, int, int],
    upper_pad_dhw: tuple[int, int, int],
    lower_pad_dhw: tuple[int, int, int],
    tactic: dict[str, object],
    use_bias: bool,
    beta: float,
) -> None:
    """Run the built-in BF16 check across representative tactics and epilogues."""
    _require_supported_gpu()

    runtime_us = run(
        ncdhw=ncdhw,
        ktrs=ktrs,
        stride_dhw=(1, 1, 1),
        upper_pad_dhw=upper_pad_dhw,
        lower_pad_dhw=lower_pad_dhw,
        dil_dhw=(1, 1, 1),
        ab_dtype=cutlass.Float4E2M1FN,
        d_dtype=cutlass.BFloat16,
        acc_dtype=cutlass.Float32,
        sf_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=16,
        use_bias=use_bias,
        beta=beta,
        tolerance=1e-2,
        warmup_iterations=1,
        iterations=2,
        skip_ref_check=False,
        **tactic,
    )

    assert runtime_us is not None and math.isfinite(runtime_us) and runtime_us > 0
