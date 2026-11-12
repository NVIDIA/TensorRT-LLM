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
"""GSM8K accuracy for Muse Spark Open (NVFP4, TP4/EP4) on its documented serving recipe.

The recipe is the one in `examples/musespark_open/serve_config.yaml`, with a shorter
`max_seq_len`, which GSM8K prompts do not need. Thresholds live in
`references/gsm8k.yaml` under `MODEL_NAME`.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig, SADecodingConfig

from ..conftest import llm_models_root, skip_pre_blackwell
from .accuracy_core import GSM8K, LlmapiAccuracyTestHarness


@pytest.mark.timeout(7200)
class TestMuseSparkOpen(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
    MODEL_PATH = f"{llm_models_root()}/Muse-Spark-Open-816B-A42B-NVFP4"

    @staticmethod
    def _llm_kwargs(cuda_graph: bool) -> dict:
        return dict(
            tensor_parallel_size=4,
            moe_expert_parallel_size=4,
            max_batch_size=256,
            max_num_tokens=8192,
            max_seq_len=8192,
            attn_backend="TRTLLM",
            moe_config=MoeConfig(backend="TRTLLM"),
            allreduce_strategy="NCCL",
            enable_chunked_prefill=True,
            # On SM103 the fused paged-context FMHA kernel for head dimension 64 exists only for an
            # FP8 KV cache, and chunked prefill and block reuse both run through it.
            kv_cache_config=KvCacheConfig(
                dtype="fp8",
                enable_block_reuse=True,
                enable_partial_reuse=True,
                free_gpu_memory_fraction=0.8,
            ),
            cuda_graph_config=CudaGraphConfig(max_batch_size=256, enable_padding=True)
            if cuda_graph
            else None,
            disable_overlap_scheduler=not cuda_graph,
        )

    @skip_pre_blackwell
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(180000)
    @pytest.mark.parametrize("cuda_graph", [False, True], ids=["eager", "cuda_graph"])
    def test_nvfp4_4gpus(self, cuda_graph: bool):
        with LLM(self.MODEL_PATH, **self._llm_kwargs(cuda_graph)) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(180000)
    @pytest.mark.parametrize("max_draft_len", [2, 4], ids=["sa2", "sa4"])
    def test_nvfp4_4gpus_sa_spec(self, max_draft_len: int):
        """Suffix-automaton speculative decoding keeps the baseline score."""
        llm_kwargs = self._llm_kwargs(cuda_graph=True)
        llm_kwargs["speculative_config"] = SADecodingConfig(
            max_draft_len=max_draft_len, max_matching_ngram_size=-1
        )
        with LLM(self.MODEL_PATH, **llm_kwargs) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
