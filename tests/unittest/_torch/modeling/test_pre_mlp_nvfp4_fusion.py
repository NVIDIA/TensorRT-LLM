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
"""Unit tests for PRE_MLP NVFP4 fusion helpers."""

from types import SimpleNamespace

from tensorrt_llm._torch.models.modeling_utils import (
    EagerFusionConfig,
    gate_up_proj_supports_pre_mlp_nvfp4_fusion,
    reconcile_pre_mlp_nvfp4_fusion,
)


def test_gate_up_proj_supports_pre_mlp_nvfp4_fusion():
    assert gate_up_proj_supports_pre_mlp_nvfp4_fusion(
        SimpleNamespace(has_nvfp4=True, input_scale=1.0))
    assert not gate_up_proj_supports_pre_mlp_nvfp4_fusion(
        SimpleNamespace(has_nvfp4=False))
    assert not gate_up_proj_supports_pre_mlp_nvfp4_fusion(SimpleNamespace())


def test_reconcile_pre_mlp_nvfp4_fusion():
    fusion_config = EagerFusionConfig(PRE_MLP_FUSION=True)
    mlp = SimpleNamespace(
        gate_up_proj=SimpleNamespace(has_nvfp4=False),
    )
    reconcile_pre_mlp_nvfp4_fusion(fusion_config, mlp)
    assert fusion_config.PRE_MLP_FUSION is False

    fusion_config = EagerFusionConfig(PRE_MLP_FUSION=True)
    mlp = SimpleNamespace(
        gate_up_proj=SimpleNamespace(has_nvfp4=True, input_scale=1.0),
    )
    reconcile_pre_mlp_nvfp4_fusion(fusion_config, mlp)
    assert fusion_config.PRE_MLP_FUSION is True
