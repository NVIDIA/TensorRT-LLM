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

"""The layer-major MLA context latents read strided per-layer slices of one GEMM output.

``MLADSparkForCausalLM.precompute_context_kv_layer_major`` (the fused generation
prepare under ``TLLM_SPEC_TAIL_FUSION=1``) hands the fused RMSNorm+RoPE kernel
``latent[:, layer]``, a view with row stride ``L * head_dim``. The compact
compile rejected it at the first warmup step ("Mismatched x.strides[0]"); the
``strided_rows`` compile must accept it and match the compact kernel bitwise.
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not IS_CUTLASS_DSL_AVAILABLE
    or get_sm_version() not in (100, 103, 107),
    reason="the fused DSpark RMSNorm+RoPE kernel needs an SM100, SM103, or SM107 GPU",
)

_HEAD_DIM, _ROPE_DIM, _KV_LORA = 576, 64, 512


@pytest.mark.parametrize("rows", [1, 7, 256])
def test_strided_layer_slices_match_compact_kernel(rows):
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        _compile_fused_dspark_rmsnorm_rope,
    )

    torch.manual_seed(rows)
    layers = 5
    eps = 1e-6
    latent = torch.randn(rows, layers * _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(layers, _KV_LORA, device="cuda", dtype=torch.bfloat16)
    angles = torch.rand(rows, _ROPE_DIM // 2, device="cuda") * 6.28
    freqs = torch.stack([angles.cos(), angles.sin()], dim=-1).contiguous()
    args = (_HEAD_DIM, _ROPE_DIM, 1, eps, True, True, False, _KV_LORA)
    compact = _compile_fused_dspark_rmsnorm_rope(*args)
    strided = _compile_fused_dspark_rmsnorm_rope(*args, strided_rows=True)
    view = latent.view(rows, layers, _HEAD_DIM)
    for layer in range(layers):
        x = view[:, layer]
        out_strided = torch.empty(rows, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        strided(x, weights[layer], freqs, out_strided)
        out_compact = torch.empty_like(out_strided)
        compact(x.contiguous(), weights[layer], freqs, out_compact)
        torch.cuda.synchronize()
        assert torch.equal(out_strided, out_compact), f"layer {layer}"
    if rows > 1:
        with pytest.raises(ValueError, match="strides"):
            compact(view[:, 1], weights[1], freqs, torch.empty_like(out_compact))
