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
"""Shared harness for the MiniMax-M3.1-Flash modeling_v2 accuracy targets.

Every test passes ``modeling_v2="require"`` to ``LLM``: under ``"auto"`` a configuration that missed a target's
criteria would fall back to the built-in implementation and report its numbers as the target's. ``require`` also builds
the DEP target, which the default mode leaves to the built-in model.

The deployment is the one ``models/minimax_m3/routing.py`` routes to the M3.1 targets: the MiniMax-M3.1-Flash
checkpoint (MXFP8 dense projections, packed NVFP4 routed experts) on sm_107 (Rubin), the MSA sparse attention with the
fused QKV/index projection and an FP8 indexer, an NVFP4 KV cache, and either no speculation or the checkpoint's
standalone DSpark drafter. Each target subclass sets its parallel layout.

GSM8K runs under the InferenceX protocol (``GSM8KInferenceX``: the chat template with the five exemplars in one user
turn, thinking as the checkpoint's template decides, the 12288-token generation budget and strict "#### N" extraction),
as main scores MiniMax-M3 with thinking: MiniMax-M3.1-Flash is a chat-tuned reasoning checkpoint whose raw-prompt
completions do not follow the plain few-shot format. It is scored against ``references/gsm8k_inferencex.yaml`` under
``MODEL_NAME``; ``evaluate`` raises "Not registered specs" until a measured reference for these specs is registered
there (see ``accuracy/README.md``).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Dict

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    DSparkDecodingConfig,
    KvCacheConfig,
    MiniMaxM3SparseAttentionConfig,
    SamplingParams,
)
from tensorrt_llm.quantization import QuantAlgo

from ....conftest import llm_models_root
from ...accuracy_core import GSM8KInferenceX, LlmapiAccuracyTestHarness
from .._env import scoped_env_overrides
from ._common import _drain_spec_stats

# The targets assert their own SM at construction: certification is per GPU architecture.
skip_not_sm107 = pytest.mark.skipif(
    get_sm_version() != 107,
    reason="the MiniMax-M3.1-Flash modeling_v2 targets are certified on sm_107 only",
)
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103,
    reason="the MiniMax-M3.1-Flash GB300 modeling_v2 targets are certified on sm_103 only",
)


class _MiniMaxM31Target(LlmapiAccuracyTestHarness):
    MODEL_NAME = "MiniMaxAI/MiniMax-M3.1-Flash"
    MODEL_PATH = f"{llm_models_root()}/MiniMax-M3.1-Flash"
    # The standalone DSpark drafter ships inside the checkpoint directory.
    DSPARK_PATH = f"{MODEL_PATH}/dspark"
    # The parallel layout the target routes on (and any backend it requires), as LLM arguments.
    DEPLOYMENT: Dict[str, Any] = dict(tensor_parallel_size=2, moe_expert_parallel_size=1)
    # Requests per rank; with DSpark each one is 1 + max_draft_len query rows in a decode step.
    MAX_BATCH_SIZE = 64
    # DSpark chat-GSM8K acceptance floors (rate = accepted / drafted, length = 1 + accepted per
    # drafting step): GSM8K alone still passes when no draft is accepted. Every target measured a
    # rate of about 0.47 and a length of about 4.3 on Rubin; the floors sit about 10% below.
    MIN_DSPARK_ACCEPTANCE_RATE = 0.42
    MIN_DSPARK_ACCEPTANCE_LENGTH = 3.85

    @contextmanager
    def _llm(self, dspark: bool) -> Iterator[LLM]:
        spec = (
            DSparkDecodingConfig(max_draft_len=7, speculative_model=self.DSPARK_PATH)
            if dspark
            else None
        )
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(
                self.MODEL_PATH,
                **self.DEPLOYMENT,
                kv_cache_config=KvCacheConfig(
                    free_gpu_memory_fraction=0.6, enable_block_reuse=False, dtype="nvfp4"
                ),
                sparse_attention_config=MiniMaxM3SparseAttentionConfig(
                    implementation="msa",
                    indexer_kv_dtype="fp8",
                    fuse_qkv_index_projection=True,
                    sparse_disable_index_value=True,
                ),
                # The prompt plus the protocol's 12288-token generation budget.
                max_seq_len=16384,
                max_batch_size=self.MAX_BATCH_SIZE,
                cuda_graph_config=CudaGraphConfig(
                    enable_padding=True, max_batch_size=self.MAX_BATCH_SIZE
                ),
                speculative_config=spec,
                enable_iter_perf_stats=dspark,
                max_stats_len=5000,
                trust_remote_code=True,
                modeling_v2="require",
                env_overrides=rank_env,
            ) as llm,
        ):
            yield llm

    def _gsm8k(self, dspark: bool) -> None:
        with self._llm(dspark) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.MXFP8
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4
            task = GSM8KInferenceX(self.MODEL_NAME)
            task.evaluate(llm)
            if dspark:
                self._check_dspark_acceptance(llm)

    def _check_dspark_acceptance(self, llm: LLM) -> None:
        from datasets import load_dataset

        questions = [
            r["question"] for r in load_dataset(GSM8KInferenceX.DATASET_DIR, "main", split="test")
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
            f"{type(self).__name__} DSpark chat-GSM8K acceptance: rate={rate:.3f}, length={length:.3f}"
        )
        assert rate > self.MIN_DSPARK_ACCEPTANCE_RATE, (
            f"acceptance rate {rate:.3f} <= {self.MIN_DSPARK_ACCEPTANCE_RATE}"
        )
        assert length > self.MIN_DSPARK_ACCEPTANCE_LENGTH, (
            f"acceptance length {length:.3f} <= {self.MIN_DSPARK_ACCEPTANCE_LENGTH}"
        )
