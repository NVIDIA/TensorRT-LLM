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
"""Shared harness for the deepseek_v41 modeling_v2 accuracy targets (DeepSeek-V4.1-Flash-NVFP4, GB300).

Every test here passes ``modeling_v2="require"`` to ``LLM``. Under ``"auto"`` a configuration that missed
a target's criteria would quietly fall back to the built-in implementation, pass, and report the built-in's
numbers as the target's.

The three targets are the roles of the prefill/decode-disaggregated deployments: ``dep2`` (context: TP2 +
attention DP + MoE EP2, no speculation) and the generation roles ``tep8_moe_tp4_ep2`` (TP8 attention, MoE
tp4 x ep2, DSpark) and ``dep`` (TP = MoE EP = world + attention DP, DSpark, lm_head TP in ADP, at 8 or 32
ranks). Each is gated here as an aggregated engine over the full GSM8K split, so that it runs both of its
step kinds (chunked prefill and decode). The disaggregated pair (ctx dep2 + gen tep8, NIXL) was measured
with the same evaluator and reaches 0.9196 flexible-extract (DSpark on, 1319 samples); the stock
implementation scored 0.9158 +/- 0.0076 on the same pair, so the floor of the reference row
(``references/gsm8k.yaml``, 90.05 = 0.9158 - 2 SE) holds for the target and the built-in model alike. The
dep gates (8 and 32 ranks) read the same rows (same checkpoint, same evaluator, DSpark lossless).

``dep`` at 32 ranks needs eight trays; no CI stage has that many GPUs for accuracy gates yet, so it is
selected by name and not listed in a test-db file. The MoE backend the gates pass is the one the targets
implement (``TRTLLM``); the generation configs' ``MEGAMOE_CUTEDSL`` is the draft model's choice.

Protocol: lm_eval GSM8K, 5-shot, no chat template, greedy, up to 4096 generated tokens,
``exact_match,flexible-extract`` (the answer never arrives in the strict ``#### N`` form often enough for the
strict filter to be meaningful here).
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    DSparkDecodingConfig,
    KvCacheConfig,
    MoeConfig,
    SamplingParams,
)

from ....conftest import llm_models_root
from ...accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from .._env import scoped_env_overrides

# The targets assert their own SM at construction: certification is per GPU architecture.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103,
    reason="the deepseek_v41 modeling_v2 targets are certified on sm_103 (GB300) only",
)

_SCORES_FILTER = {"scores_filter": "exact_match,flexible-extract"}


class _DeepseekV41Gate(LlmapiAccuracyTestHarness):
    # The reference rows (quant_algo MIXED_PRECISION, FP8 KV cache) live under this name.
    MODEL_NAME = "deepseek-ai/DeepSeek-V4.1-Flash"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V4.1-Flash-NVFP4"

    def _gsm8k(self, mocker, **engine_kwargs):
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", 4096)
        mocker.patch.dict(GSM8K.EVALUATE_KWARGS, _SCORES_FILTER)
        # The target checks its step contract on the first forward and only when asked; it travels
        # as ``env_overrides`` because the check runs where the metadata does, on the rank.
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(
                self.MODEL_PATH,
                modeling_v2="require",
                trust_remote_code=True,
                disable_mm_encoder=True,
                enable_chunked_prefill=True,
                # As in the deployed generation configs: decode selects with the hint-free self-sampling GVR top-k.
                sparse_attention_config={"algorithm": "csa2", "enable_heuristic_topk": True},
                env_overrides=rank_env,
                **engine_kwargs,
            ) as llm,
        ):
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, sampling_params=SamplingParams(max_tokens=4096, temperature=0))


class _AttentionDpGenerationGate(_DeepseekV41Gate):
    """The aggregated engine of an attention-DP generation role: TP = EP = world, lm_head split over the ranks."""

    WORLD = 0
    DRAFT_LEN = 0
    BATCH = 16

    def _run(self, mocker, dspark):
        speculative_config = (
            DSparkDecodingConfig(
                max_draft_len=self.DRAFT_LEN,
                block_size=self.DRAFT_LEN,
                speculative_model=self.MODEL_PATH,
            )
            if dspark
            else None
        )
        self._gsm8k(
            mocker,
            tensor_parallel_size=self.WORLD,
            moe_expert_parallel_size=self.WORLD,
            enable_attention_dp=True,
            enable_lm_head_tp_in_adp=True,
            # the TRTLLM runner over the all-gather / reduce-scatter combines in bf16: no low-precision combine
            moe_config=MoeConfig(backend="TRTLLM"),
            cuda_graph_config=CudaGraphConfig(
                batch_sizes=[1, 2, 4, 8, self.BATCH], enable_padding=True
            ),
            max_batch_size=self.BATCH,
            max_num_tokens=8192,
            max_seq_len=8192,
            kv_cache_config=KvCacheConfig(
                dtype="fp8",
                tokens_per_block=128,
                enable_block_reuse=False,
                free_gpu_memory_fraction=0.6,
            ),
            speculative_config=speculative_config,
        )
