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
"""Whole-model gate for the MiniMax-M3 modeling_v2 targets.

Every test here passes ``modeling_v2="require"`` to ``LLM``: under ``"auto"`` a configuration that missed a target's
criteria would quietly fall back to the built-in implementation, pass, and report the built-in's numbers as the
target's.

The deployment is the one the targets route (``models/minimax_m3/routing.py``): the NVFP4 checkpoint on sm_103,
tensor-parallel attention and experts on 4 or 8 ranks, the MiniMax-M3 MSA sparse attention with the fused five-way
QKV/index projection, an FP8 indexer and an NVFP4 KV cache. Each target runs GSM8K without speculation and with
one-model Eagle3, where a chat-GSM8K acceptance probe also guards the drafter (accuracy alone does not notice a
corrupted draft KV).
"""

import json

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    Eagle3DecodingConfig,
    KvCacheConfig,
    MiniMaxM3SparseAttentionConfig,
    MoeConfig,
    SamplingParams,
)
from tensorrt_llm.quantization import QuantAlgo

from ...conftest import llm_models_root
from ..accuracy_core import GSM8K, LlmapiAccuracyTestHarness

# The targets assert their own SM at construction: certification is per GPU architecture.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103,
    reason="the MiniMax-M3 modeling_v2 targets are certified on sm_103 only",
)

# The chat-GSM8K acceptance floors of the in-tree test on an NVFP4 KV cache (test_llm_api_pytorch.TestMiniMaxM3).
MIN_ACCEPTANCE_RATE = 0.78
MIN_ACCEPTANCE_LENGTH = 3.3


def _drain_spec_stats(llm):
    drafted = accepted = steps = 0
    for s in llm.get_stats(timeout=2):
        s = json.loads(s) if isinstance(s, str) else s
        sd = s.get("specDecodingStats") or {}
        drafted += sd.get("numDraftTokens", 0)
        accepted += sd.get("numAcceptedTokens", 0)
        steps += sd.get("numRequestsWithDraftTokens", 0)
    return drafted, accepted, steps


class _MiniMaxM3Target(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/MiniMax-M3-NVFP4"
    MODEL_PATH = f"{llm_models_root()}/MiniMax-M3-NVFP4"
    EAGLE_PATH = f"{llm_models_root()}/MiniMax-M3-EAGLE3-GQA"
    TP = 4

    def _llm(self, eagle3: bool) -> LLM:
        spec = (
            Eagle3DecodingConfig(max_draft_len=3, speculative_model=self.EAGLE_PATH)
            if eagle3
            else None
        )
        return LLM(
            self.MODEL_PATH,
            tensor_parallel_size=self.TP,
            moe_expert_parallel_size=1,
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=0.6, enable_block_reuse=False, dtype="nvfp4"
            ),
            sparse_attention_config=MiniMaxM3SparseAttentionConfig(
                implementation="msa", indexer_kv_dtype="fp8", fuse_qkv_index_projection=True
            ),
            moe_config=MoeConfig(backend="TRTLLM"),
            max_seq_len=4096,
            max_batch_size=64,
            cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=64),
            speculative_config=spec,
            enable_iter_perf_stats=eagle3,
            max_stats_len=5000,
            trust_remote_code=True,
            modeling_v2="require",
            env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
        )

    def _gsm8k(self, eagle3: bool) -> None:
        with self._llm(eagle3) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            if not eagle3:
                return
            from datasets import load_dataset

            questions = [
                r["question"] for r in load_dataset(GSM8K.DATASET_DIR, "main", split="test")
            ][:200]
            prompts = [
                llm.tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
                )
                for q in questions
            ]
            _drain_spec_stats(llm)
            llm.generate(prompts, SamplingParams(max_tokens=512, temperature=0))
            drafted, accepted, steps = _drain_spec_stats(llm)
            assert steps > 0, "no speculative iterations recorded"
            rate, length = accepted / drafted, 1 + accepted / steps
            print(
                f"MiniMax-M3 tp{self.TP} Eagle3 chat-GSM8K acceptance: rate={rate:.3f}, length={length:.3f}"
            )
            assert rate > MIN_ACCEPTANCE_RATE, (
                f"acceptance rate {rate:.3f} <= {MIN_ACCEPTANCE_RATE}"
            )
            assert length > MIN_ACCEPTANCE_LENGTH, (
                f"acceptance length {length:.3f} <= {MIN_ACCEPTANCE_LENGTH}"
            )


class TestModelingV2MiniMaxM3Nvfp4Sm103Tp4(_MiniMaxM3Target):
    """MiniMax-M3 NVFP4 / sm_103 / tp4: four GB300s."""

    TP = 4

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_gsm8k(self):
        self._gsm8k(eagle3=False)

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_eagle3_gsm8k_and_acceptance(self):
        self._gsm8k(eagle3=True)


class TestModelingV2MiniMaxM3Nvfp4Sm103Tp8(_MiniMaxM3Target):
    """MiniMax-M3 NVFP4 / sm_103 / tp8: eight GB300s over two trays (an 8-rank MPI world across the two nodes)."""

    TP = 8

    @skip_not_sm103
    @pytest.mark.skip_less_mpi_world_size(8)
    def test_gsm8k(self):
        self._gsm8k(eagle3=False)

    @skip_not_sm103
    @pytest.mark.skip_less_mpi_world_size(8)
    def test_eagle3_gsm8k_and_acceptance(self):
        self._gsm8k(eagle3=True)
