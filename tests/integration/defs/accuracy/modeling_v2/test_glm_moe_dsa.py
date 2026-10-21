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
"""Whole-model gates for the glm_moe_dsa modeling_v2 target (GLM-5.2 NVFP4, sm_103).

One case per layout family the target admits (tp8, tep, dep), each serving GSM8K end
to end with ``modeling_v2="require"``. Under ``"auto"`` a deployment that missed the
route would fall back to the built-in model, pass, and report the built-in's accuracy
as the target's; under ``"require"`` it raises instead, and so does a step a fast
path covers but cannot run. The mode and the environment below are LLM API
arguments, so they reach every rank.

Each deployment is one the family's ``route()`` claims: the GLM-5.2 modelopt NVFP4
checkpoint (bf16 MTP layer) with one-model MTP-Eagle as a linear chain, at the
layout's draft length; the dep case also matches its cell's KV-cache dtype and MoE
backend.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    DeepSeekSparseAttentionConfig,
    KvCacheConfig,
    MoeConfig,
    MTPDecodingConfig,
)

from ...conftest import llm_models_root
from ..accuracy_core import GSM8K, LlmapiAccuracyTestHarness

# The target asserts its own SM at construction.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="the glm_moe_dsa target is certified on sm_103 only"
)

# On every rank: check the target's step contract on the first forward, and run the
# indexer key projection in BF16, the form the family's fused indexer paths take.
# The deployments below are the ones the target was validated at: per-layout batch,
# KV-cache dtype, MoE backend and DSA kernels.
_RANK_ENV = {MODELING_V2_VALIDATE_ENV: "1", "TRTLLM_DSA_INDEXER_BF16": "1"}


class TestModelingV2Glm52Nvfp4Sm103Default(LlmapiAccuracyTestHarness):
    """glm-5.2-nvfp4 / sm_103 / default at tp8, tep4 and dep8."""

    MODEL_NAME = "zai-org/GLM-5.2"
    MODEL_PATH = f"{llm_models_root()}/GLM-5.2-NVFP4"

    @staticmethod
    def _llm_kwargs(*, draft_len: int, kv_dtype: str, moe_backend: str, batch: int) -> dict:
        return dict(
            modeling_v2="require",
            env_overrides=dict(_RANK_ENV),
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
        with LLM(self.MODEL_PATH, **llm_kwargs) as llm:
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

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_gsm8k_tep4(self):
        """Attention tensor-parallel over 4 ranks, experts expert-parallel over them."""
        self._evaluate(
            tensor_parallel_size=4,
            moe_expert_parallel_size=4,
            **self._llm_kwargs(draft_len=5, kv_dtype="fp8", moe_backend="TRTLLM", batch=2),
        )

    @skip_not_sm103
    @pytest.mark.skip_less_mpi_world_size(8)
    def test_gsm8k_dep8(self):
        """Attention data parallel over 8 ranks: the FP8-KV, CuteDSL-MoE, MTP3 cell."""
        self._evaluate(
            tensor_parallel_size=8,
            moe_expert_parallel_size=8,
            enable_attention_dp=True,
            **self._llm_kwargs(draft_len=3, kv_dtype="fp8", moe_backend="CUTEDSL", batch=16),
        )
