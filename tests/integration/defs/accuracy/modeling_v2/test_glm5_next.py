# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Full-GSM8K gates for the NVFP4 GLM-5.3-Flash target on SM100 (B200).

Requiring modeling_v2 prevents an unsupported deployment from silently testing
the built-in model.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MTPDecodingConfig

from ...conftest import llm_models_root
from ..accuracy_core import GSM8K, LlmapiAccuracyTestHarness

_MODEL_NAME = "zai-org/GLM-5.3-Flash"
_MODEL_PATH = f"{llm_models_root()}/GLM-5.3-Flash-NVFP4"


def _llm(model_path: str, speculative_config=None) -> LLM:
    return LLM(
        model_path,
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        moe_expert_parallel_size=4,
        modeling_v2="require",
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5, enable_block_reuse=False),
        max_batch_size=64,
        max_num_tokens=16384,
        max_seq_len=8192,
        cuda_graph_config=CudaGraphConfig(max_batch_size=64, enable_padding=True),
        enable_chunked_prefill=True,
        disable_mm_encoder=True,
        speculative_config=speculative_config,
        env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
    )


@pytest.mark.skipif(
    get_sm_version() != 100, reason="The modeling_v2 GLM-5.3-Flash target requires SM100 (B200)."
)
@pytest.mark.skip_less_mpi_world_size(4)
@pytest.mark.timeout(7200)
class TestModelingV2Glm53FlashNvfp4Sm100Default(LlmapiAccuracyTestHarness):
    """Attention TP4 with expert-parallel MoE (EP4), without and with one-model MTP."""

    MODEL_NAME = _MODEL_NAME
    MODEL_PATH = _MODEL_PATH

    def test_gsm8k(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MODELING_V2_VALIDATE_ENV, "1")
        with _llm(self.MODEL_PATH) as llm:
            GSM8K(self.MODEL_NAME).evaluate(llm)

    def test_gsm8k_mtp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MODELING_V2_VALIDATE_ENV, "1")
        with _llm(self.MODEL_PATH, MTPDecodingConfig(max_draft_len=3)) as llm:
            GSM8K(self.MODEL_NAME).evaluate(llm)
