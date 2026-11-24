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
"""Whole-model gate for the glm_moe_dsa modeling_v2 target glm_5_2_nvfp4__sm_103__tp8.

``TestModelingV2Glm52Nvfp4Sm103Tp8``: GSM8K end to end for the SM103 tp8
low-latency target, at the deployment its ``route()`` admits: the GLM-5.2
modelopt NVFP4 checkpoint (bf16 MTP layer), attention and experts
tensor-parallel over 8 ranks, one-model MTP-Eagle as a linear chain of five
draft tokens.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    DeepSeekSparseAttentionConfig,
    KvCacheConfig,
    MoeConfig,
    MTPDecodingConfig,
)

from .....conftest import llm_models_root
from ....accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from ..._env import scoped_env_overrides
from .._common import skip_not_sm103

# On every rank: check the target's step contract on the first forward, and run the
# indexer key projection in BF16, the form the target's fused indexer paths take.
_SM103_RANK_ENV = {MODELING_V2_VALIDATE_ENV: "1", "TRTLLM_DSA_INDEXER_BF16": "1"}


class TestModelingV2Glm52Nvfp4Sm103Tp8(LlmapiAccuracyTestHarness):
    """glm-5.2-nvfp4 / sm_103 / tp8, at the deployment it was validated at."""

    MODEL_NAME = "zai-org/GLM-5.2"
    MODEL_PATH = f"{llm_models_root()}/GLM-5.2-NVFP4"

    @staticmethod
    def _llm_kwargs(*, draft_len: int, kv_dtype: str, moe_backend: str, batch: int) -> dict:
        return dict(
            modeling_v2="require",
            speculative_config=MTPDecodingConfig(max_draft_len=draft_len),
            kv_cache_config=KvCacheConfig(
                dtype=kv_dtype, enable_block_reuse=False, free_gpu_memory_fraction=0.75
            ),
            moe_config=MoeConfig(backend=moe_backend),
            sparse_attention_config=DeepSeekSparseAttentionConfig(
                enable_heuristic_topk=True,
                use_cute_dsl_paged_mqa_logits=True,
                use_cute_dsl_topk=True,
            ),
            cuda_graph_config=CudaGraphConfig(max_batch_size=batch, enable_padding=True),
            max_batch_size=batch,
            max_seq_len=8192,
            enable_chunked_prefill=True,
        )

    def _evaluate(self, **llm_kwargs) -> None:
        with (
            scoped_env_overrides(_SM103_RANK_ENV) as rank_env,
            LLM(self.MODEL_PATH, env_overrides=rank_env, **llm_kwargs) as llm,
        ):
            GSM8K(self.MODEL_NAME).evaluate(llm)

    @skip_not_sm103
    @pytest.mark.skip_less_mpi_world_size(8)
    def test_gsm8k_tp8(self):
        """tp8, the layout the fast paths are tuned for: attention and experts TP over 8 ranks."""
        self._evaluate(
            tensor_parallel_size=8,
            moe_expert_parallel_size=1,
            **self._llm_kwargs(draft_len=5, kv_dtype="fp8", moe_backend="TRTLLM", batch=1),
        )
