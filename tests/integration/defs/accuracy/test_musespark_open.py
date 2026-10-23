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
"""Accuracy suite for the MuseSparkOpen text model (NVFP4, TP4/EP4).

The portable per-model accuracy coverage, in the same style as the other
per-model suites here (``test_kimi3.py``, ``test_glm52.py``): a
``LlmapiAccuracyTestHarness`` subclass whose thresholds live in
``references/gsm8k.yaml`` under ``MODEL_NAME``.
"""

import os

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig, SADecodingConfig

from ..conftest import llm_models_root, skip_pre_blackwell
from .accuracy_core import GSM8K, LlmapiAccuracyTestHarness


@pytest.mark.timeout(7200)
class TestMuseSparkOpen(LlmapiAccuracyTestHarness):
    # Local reference key, deliberately not the Hub id
    # (meta-models/Muse-Spark-Open-816B-A42B-NVFP4): it only has to match
    # the references/*.yaml key.
    MODEL_NAME = "musespark_open"
    # MUSESPARK_REFERENCE_CKPT points at a checkpoint staged outside
    # llm-models; otherwise the llm-models copy (ftp/llm-models!621) is used.
    MODEL_PATH = os.environ.get("MUSESPARK_REFERENCE_CKPT") or (
        f"{llm_models_root()}/Muse-Spark-Open-816B-A42B-NVFP4"
    )

    @staticmethod
    def _llm_kwargs(cuda_graph: bool) -> dict:
        return dict(
            # The registered wrapper `trtllm-eval --custom_tokenizer
            # musespark_open` uses. Without it the runtime builds a plain HF
            # tokenizer with no chat template (the checkpoint keeps it in a
            # separate .jinja), no pad token and only one of the two EOS ids.
            custom_tokenizer="musespark_open",
            tensor_parallel_size=4,
            moe_expert_parallel_size=4,
            max_batch_size=8,
            max_seq_len=8192,
            # head_dim 64 has no fused context FMHA kernel on this SM, so the
            # per-step token budget stays at the value the checkpoint is
            # qualified under.
            max_num_tokens=2048,
            # FlashInfer: the TRTLLM backend's unfused-MHA fallback at
            # head_dim 64 drops cached context on reuse-started requests.
            attn_backend="FLASHINFER",
            moe_config=MoeConfig(backend="TRTLLM"),
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=0.8,
                enable_block_reuse=True,
            ),
            # AUTO dispatches TP4 decode-sized messages to the NCCL symmetric
            # all-reduce path that returned non-finite layer outputs during
            # bring-up; NCCL is the qualified strategy.
            allreduce_strategy="NCCL",
            enable_chunked_prefill=False,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            disable_overlap_scheduler=not cuda_graph,
        )

    @pytest.fixture(autouse=True)
    def _skip_kv_cache_estimation(self, monkeypatch):
        # The KV-cache size estimation pass allocates a temporary cache on top
        # of the activation peak and does not fit next to this checkpoint's
        # weights; the fraction below pins the cache size instead.
        monkeypatch.setenv("TRTLLM_SKIP_KV_CACHE_ESTIMATION", "1")

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
        # Speculation runs with the overlap scheduler off until the
        # FlashInfer KV-length correction covers one-engine drafters.
        llm_kwargs["disable_overlap_scheduler"] = True
        llm_kwargs["speculative_config"] = SADecodingConfig(
            max_draft_len=max_draft_len, max_matching_ngram_size=-1
        )
        with LLM(self.MODEL_PATH, **llm_kwargs) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
