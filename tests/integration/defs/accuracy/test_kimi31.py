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

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    DeepSeekSparseAttentionConfig,
    DSparkDecodingConfig,
    KvCacheConfig,
    MoeConfig,
    SamplingParams,
)

from ..conftest import llm_models_root, skip_pre_blackwell
from .accuracy_core import (
    GSM8K,
    GPQADiamond,
    LlmapiAccuracyTestHarness,
    assert_acceptance_length_for_llm,
)


@skip_pre_blackwell
class TestKimiK31(LlmapiAccuracyTestHarness):
    """Kimi K3.1 Pro (DSA sparse attention on the MLA layers) with the DSpark drafter.

    The two configurations carry the numerics of the GB300 deployments on the LLM API: DEP16 with
    NVFP4 MLA KV records (``nvfp4_ds_mla``) and FP4 index keys, and TEP8 with FP8 KV and FP8 index
    keys. Both read as ``kv_cache_quant_algo: FP8`` (``nvfp4_ds_mla`` uses a one-byte container),
    so they share one GSM8K reference. Speculative decoding is lossless, so each GSM8K test also
    gates the acceptance length: a broken drafter scores the same and only runs slower.
    """

    MODEL_NAME = "moonshotai/Kimi-K3.1-Pro"
    MODEL_PATH = f"{llm_models_root()}/Kimi-K3.1-Pro"
    DSPARK_MODEL_PATH = f"{llm_models_root()}/Inferact-Kimi-K3-DSpark"

    def _dep16_kwargs(self, max_seq_len: int) -> dict:
        return dict(
            tensor_parallel_size=16,
            moe_expert_parallel_size=16,
            enable_attention_dp=True,
            max_batch_size=8,
            max_num_tokens=8192,
            max_seq_len=max_seq_len,
            trust_remote_code=True,
            # The drafter runs between target steps, and its hidden-state capture needs a
            # whole-sequence prefill.
            disable_overlap_scheduler=True,
            enable_chunked_prefill=False,
            cuda_graph_config=CudaGraphConfig(max_batch_size=8),
            kv_cache_config=KvCacheConfig(
                use_kv_cache_manager_v2=True,
                dtype="nvfp4_ds_mla",
                tokens_per_block=64,
                enable_block_reuse=False,
                free_gpu_memory_fraction=0.5,
            ),
            moe_config=MoeConfig(max_num_tokens=32768),
            sparse_attention_config=DeepSeekSparseAttentionConfig(
                indexer_k_dtype="fp4",
                enable_heuristic_topk=True,
                gvr_deterministic_topk=True,
            ),
            speculative_config=DSparkDecodingConfig(
                max_draft_len=6, speculative_model=self.DSPARK_MODEL_PATH
            ),
            enable_iter_perf_stats=True,
            max_stats_len=-1,
        )

    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_mpi_world_size(16)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_dep16(self):
        """GSM8K plus an acceptance-length gate on the DEP16 configuration."""
        with LLM(self.MODEL_PATH, **self._dep16_kwargs(max_seq_len=8192)) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            assert_acceptance_length_for_llm("TestKimiK31::test_gsm8k_dep16", llm)

    @pytest.mark.timeout(10800)
    @pytest.mark.skip_less_mpi_world_size(16)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gpqa_diamond_dep16(self):
        """GPQA Diamond on the DEP16 configuration, with the model's chat template."""
        max_seq_len = GPQADiamond.MAX_INPUT_LEN + GPQADiamond.MAX_OUTPUT_LEN
        sampling_params = SamplingParams(
            max_tokens=GPQADiamond.MAX_OUTPUT_LEN,
            truncate_prompt_tokens=GPQADiamond.MAX_INPUT_LEN,
        )
        with LLM(self.MODEL_PATH, **self._dep16_kwargs(max_seq_len=max_seq_len)) as llm:
            task = GPQADiamond(self.MODEL_NAME)
            task.evaluate(
                llm,
                sampling_params=sampling_params,
                extra_evaluator_kwargs=dict(apply_chat_template=True),
            )

    @pytest.mark.timeout(3600)
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_tep8(self):
        """GSM8K plus an acceptance-length gate on the TEP8 configuration."""
        with LLM(
            self.MODEL_PATH,
            tensor_parallel_size=8,
            moe_expert_parallel_size=8,
            enable_attention_dp=False,
            max_batch_size=8,
            max_num_tokens=8192,
            max_seq_len=8192,
            trust_remote_code=True,
            disable_overlap_scheduler=True,
            enable_chunked_prefill=False,
            cuda_graph_config=CudaGraphConfig(max_batch_size=8),
            kv_cache_config=KvCacheConfig(
                use_kv_cache_manager_v2=True,
                dtype="fp8",
                tokens_per_block=64,
                enable_block_reuse=False,
                free_gpu_memory_fraction=0.5,
            ),
            moe_config=MoeConfig(max_num_tokens=33024),
            sparse_attention_config=DeepSeekSparseAttentionConfig(
                enable_heuristic_topk=True,
                gvr_deterministic_topk=True,
            ),
            speculative_config=DSparkDecodingConfig(
                max_draft_len=5, speculative_model=self.DSPARK_MODEL_PATH
            ),
            enable_iter_perf_stats=True,
            max_stats_len=-1,
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            assert_acceptance_length_for_llm("TestKimiK31::test_gsm8k_tep8", llm)
