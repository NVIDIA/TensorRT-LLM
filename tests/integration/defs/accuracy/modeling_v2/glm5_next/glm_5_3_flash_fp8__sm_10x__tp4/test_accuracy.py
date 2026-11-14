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
"""Whole-model gates for the glm5_next modeling_v2 target glm_5_3_flash_fp8__sm_10x__tp4.

Every test passes ``modeling_v2="require"``: under ``"auto"`` a deployment the target does
not claim would fall back to the built-in model, pass, and report the built-in model's
numbers as the target's. The settings are the target's routed deployment: tensor
parallelism 4 with the experts split by intermediate, an FP8 latent cache, the vision tower
disabled.
"""

import pytest

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MTPDecodingConfig

from .....conftest import llm_models_root
from ....accuracy_core import GSM8K, LlmapiAccuracyTestHarness, assert_acceptance_length_for_llm
from ..._env import scoped_env_overrides

# The target refuses other GPUs at construction: certification is per GPU architecture.
skip_not_sm100 = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next FP8 target is certified on sm_100 only"
)


class TestModelingV2Glm53FlashFp8Sm10xTp4(LlmapiAccuracyTestHarness):
    """GLM-5.3-Flash block-FP8 / sm_100 / tp4, without speculation and with one-model MTP-3.

    The MTP variant also gates the acceptance length: a miscomputed draft layer only slows
    decoding (every draft is rejected), so accuracy alone cannot see it. The acceptance
    anchor is the built-in model's on the same checkpoint and MTP layer.
    """

    MODEL_NAME = "zai-org/GLM-5.3-Flash"
    MODEL_PATH = f"{llm_models_root()}/GLM-5.3-Flash"
    ACCEPTANCE_KEY = "ModelingV2Glm53FlashFp8Sm10xTp4::mtp3"

    @staticmethod
    def _llm_kwargs(**extra) -> dict:
        return dict(
            tensor_parallel_size=4,
            moe_tensor_parallel_size=4,
            moe_expert_parallel_size=1,
            kv_cache_config=KvCacheConfig(
                dtype="fp8", free_gpu_memory_fraction=0.5, enable_block_reuse=False
            ),
            max_batch_size=64,
            max_num_tokens=16384,
            max_seq_len=8192,
            cuda_graph_config=CudaGraphConfig(max_batch_size=64, enable_padding=True),
            enable_chunked_prefill=True,
            disable_mm_encoder=True,
            modeling_v2="require",
            **extra,
        )

    # Each engine also checks, on its first forward, that the metadata fields and cache members
    # the target reads exist (TRTLLM_MODELING_V2_VALIDATE, passed to the ranks).

    @skip_not_sm100
    @pytest.mark.skip_less_device(4)
    def test_gsm8k(self):
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(self.MODEL_PATH, env_overrides=rank_env, **self._llm_kwargs()) as llm,
        ):
            GSM8K(self.MODEL_NAME).evaluate(llm)
            self._check_one_token_prompt(llm)

    @staticmethod
    def _check_one_token_prompt(llm: LLM) -> None:
        """A one-token prompt runs as a one-row prefill step from a zero KDA state.

        Its output must not depend on which request held its slot before. Without speculation
        the engine may run a final one-token context as a decode-graph row, which would skip the
        slot reset the KDA prefill does; the target opts out of that, so the prompt takes the
        one-row prefill path in both tests.
        """
        params = SamplingParams(max_tokens=16, temperature=0)
        token = llm.tokenizer.encode("Hello", add_special_tokens=False)[:1]
        fresh = llm.generate(token, params).outputs[0].token_ids
        llm.generate("Write a long story about a lighthouse keeper. " * 20, params)
        after = llm.generate(token, params).outputs[0].token_ids
        assert after == fresh, (
            f"one-token prompt changed with the slot's history: {fresh} -> {after}"
        )

    @skip_not_sm100
    @pytest.mark.skip_less_device(4)
    def test_gsm8k_mtp3(self):
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(
                self.MODEL_PATH,
                env_overrides=rank_env,
                **self._llm_kwargs(
                    speculative_config=MTPDecodingConfig(max_draft_len=3),
                    max_stats_len=-1,
                    enable_iter_perf_stats=True,
                ),
            ) as llm,
        ):
            GSM8K(self.MODEL_NAME).evaluate(llm)
            assert_acceptance_length_for_llm(self.ACCEPTANCE_KEY, llm)
            self._check_one_token_prompt(llm)
