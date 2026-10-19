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
"""Whole-model gate for the nemotron_h modeling_v2 targets.

Every test here passes ``modeling_v2="require"`` to ``LLM``: under ``"auto"`` a
configuration that missed the target's criteria would quietly fall back to the
built-in implementation and report its numbers as the target's.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

from ...conftest import llm_models_root
from ..accuracy_core import GSM8K, LlmapiAccuracyTestHarness

# The target asserts its own SM at construction: certification is per GPU
# architecture, and a receipt from another one says nothing here.
skip_not_sm107 = pytest.mark.skipif(
    get_sm_version() != 107, reason="the nemotron_h NVFP4 targets are certified on sm_107 only"
)
skip_not_sm100 = pytest.mark.skipif(
    get_sm_version() != 100, reason="the nemotron_h BF16 target is certified on sm_100 only"
)


@pytest.mark.skip_less_mpi_world_size(4)
class TestModelingV2Nemotron3UltraNvfp4Sm107Tep4(LlmapiAccuracyTestHarness):
    """Nemotron-3 Ultra 550B-A55B NVFP4 / sm_107 / tep4."""

    # The registry key upstream uses for this checkpoint; it carries the
    # MIXED_PRECISION / FP8-KV entry the engine resolves from its quantization config.
    MODEL_NAME = "nvidia/Nemotron-Ultra-V3"
    MODEL_PATH = f"{llm_models_root()}/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"

    # The protocol TestNemotronV3Ultra gates this checkpoint with: thinking off.
    extra_evaluator_kwargs = dict(chat_template_kwargs=dict(enable_thinking=False))

    @skip_not_sm107
    def test_gsm8k(self):
        # The engine configuration of TestNemotronV3Ultra's TEP4 runs: tensor-parallel
        # attention and mamba, expert-parallel MoE, fp16 SSM state, CUDA graphs with
        # padding. The target checks its step contract on the first forward; it
        # travels as ``env_overrides`` because the check runs where the metadata does.
        max_batch_size = 32
        with LLM(
            self.MODEL_PATH,
            modeling_v2="require",
            env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=False,
                mamba_ssm_cache_dtype="float16",
                free_gpu_memory_fraction=0.5,
            ),
            max_batch_size=max_batch_size,
            tensor_parallel_size=4,
            moe_expert_parallel_size=4,
            enable_attention_dp=False,
            cuda_graph_config=CudaGraphConfig(max_batch_size=max_batch_size, enable_padding=True),
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=self.extra_evaluator_kwargs)


@pytest.mark.skip_less_mpi_world_size(4)
class TestModelingV2Nemotron3UltraNvfp4Sm107Dep4(LlmapiAccuracyTestHarness):
    """Nemotron-3 Ultra 550B-A55B NVFP4 / sm_107 / dep4."""

    MODEL_NAME = "nvidia/Nemotron-Ultra-V3"
    MODEL_PATH = f"{llm_models_root()}/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
    extra_evaluator_kwargs = dict(chat_template_kwargs=dict(enable_thinking=False))

    @skip_not_sm107
    def test_gsm8k(self):
        # The tep4 gate's engine configuration with attention data parallelism: every
        # rank serves its own requests (max_batch_size is per rank) and only the routed
        # experts are split across ranks.
        max_batch_size = 32
        with LLM(
            self.MODEL_PATH,
            modeling_v2="require",
            env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=False,
                mamba_ssm_cache_dtype="float16",
                free_gpu_memory_fraction=0.5,
            ),
            max_batch_size=max_batch_size,
            tensor_parallel_size=4,
            moe_expert_parallel_size=4,
            enable_attention_dp=True,
            cuda_graph_config=CudaGraphConfig(max_batch_size=max_batch_size, enable_padding=True),
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=self.extra_evaluator_kwargs)


@pytest.mark.skip_less_mpi_world_size(8)
@pytest.mark.skip_less_device_memory(180000)
class TestModelingV2Nemotron3UltraBf16Sm100Tep8(LlmapiAccuracyTestHarness):
    """Nemotron-3 Ultra 550B-A55B BF16 / sm_100 / tep8."""

    # The registry key upstream uses for this checkpoint; the BF16 release declares no
    # quantization, so the engine resolves the entry that has neither a quant_algo nor
    # a kv_cache_quant_algo.
    MODEL_NAME = "nvidia/Nemotron-Ultra-V3"
    MODEL_PATH = f"{llm_models_root()}/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"

    # The protocol TestNemotronV3Ultra gates the NVFP4 checkpoint with: thinking off.
    extra_evaluator_kwargs = dict(chat_template_kwargs=dict(enable_thinking=False))

    @skip_not_sm100
    def test_gsm8k(self):
        # 1.1 TB of weights: 8 ranks, tensor-parallel attention and mamba, expert-parallel
        # MoE (the replicated-attention layout leaves no room for the state and the cache).
        # The state and the KV cache keep the checkpoint's own dtypes (fp32 SSM state, bf16
        # KV). Block reuse is off and the fraction applies to what is left after about
        # 128 GiB of weights on each 185 GiB GPU. The target checks its step contract on
        # the first forward; it travels as ``env_overrides`` because the check runs where
        # the metadata does.
        max_batch_size = 32
        with LLM(
            self.MODEL_PATH,
            modeling_v2="require",
            env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=False,
                free_gpu_memory_fraction=0.5,
            ),
            max_batch_size=max_batch_size,
            tensor_parallel_size=8,
            moe_expert_parallel_size=8,
            enable_attention_dp=False,
            cuda_graph_config=CudaGraphConfig(max_batch_size=max_batch_size, enable_padding=True),
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=self.extra_evaluator_kwargs)
