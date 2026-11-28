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
"""Whole-model gate for the kimi_k31 modeling_v2 target kimi_k31_pro_nvfp4__sm_10x__default.

The NVFP4 checkpoint and the DFlash2 drafter are in ``../_common.py``.
"""

import os

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    DeepSeekSparseAttentionConfig,
    KvCacheConfig,
    MoeConfig,
)

from ....accuracy_core import GSM8K, LlmapiAccuracyTestHarness, assert_acceptance_length_for_llm
from ..._env import scoped_env_overrides
from .._common import (
    NVFP4_DFLASH2_PATH,
    NVFP4_MODEL_NAME,
    NVFP4_PATH,
    nvfp4_dflash2,
    nvfp4_kv_cache,
    nvfp4_sparse_attention,
    skip_not_sm103_or_sm107,
)


def _aggregated(deployment: str) -> dict:
    """An aggregated server of the validated recipe.

    TEP8 or DEP16, CUTEDSL MoE, FP8 KV cache and index keys, FP32 KDA state, DFlash2 (draft length 4), with
    iteration stats for the acceptance length.
    """
    dep16 = deployment == "dep16"
    ranks = 16 if dep16 else 8
    return dict(
        tensor_parallel_size=ranks,
        moe_expert_parallel_size=ranks,
        enable_attention_dp=dep16,
        modeling_v2="require",
        max_batch_size=8,
        max_num_tokens=8192,
        max_seq_len=8192,
        trust_remote_code=True,
        cuda_graph_config=CudaGraphConfig(max_batch_size=8, enable_padding=True),
        kv_cache_config=KvCacheConfig(**nvfp4_kv_cache()),
        moe_config=MoeConfig(backend="CUTEDSL"),
        sparse_attention_config=DeepSeekSparseAttentionConfig(**nvfp4_sparse_attention()),
        speculative_config=nvfp4_dflash2(4),
        enable_iter_perf_stats=True,
    )


@skip_not_sm103_or_sm107
class TestModelingV2KimiK31ProNvfp4Sm10xDefault(LlmapiAccuracyTestHarness):
    """``kimi_k31_pro_nvfp4__sm_10x__default``: the NVIDIA NVFP4 checkpoint on its validated recipe."""

    MODEL_NAME = NVFP4_MODEL_NAME
    MODEL_PATH = NVFP4_PATH

    @pytest.mark.timeout(3600)
    @pytest.mark.skip_less_device_memory(250000)
    @pytest.mark.parametrize(
        "deployment",
        [
            pytest.param("tep8", marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param("dep16", marks=pytest.mark.skip_less_mpi_world_size(16)),
        ],
    )
    def test_gsm8k(self, deployment):
        """GSM8K on an aggregated server of the validated recipe, and at DEP16 the acceptance length."""
        for path in (self.MODEL_PATH, NVFP4_DFLASH2_PATH):
            if not os.path.exists(path):
                pytest.skip(f"Model directory {path} does not exist")
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(self.MODEL_PATH, env_overrides=rank_env, **_aggregated(deployment)) as llm,
        ):
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            if deployment == "dep16":
                assert_acceptance_length_for_llm(
                    "ModelingV2KimiK31ProNvfp4Sm10xDefault::dflash2_w5[dep16]", llm
                )
