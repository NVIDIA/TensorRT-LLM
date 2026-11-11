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
"""Shared GSM8K harness for the nemotron_h modeling_v2 targets.

Every test of these targets passes ``modeling_v2="require"`` to ``LLM``: under
``"auto"`` a configuration that missed a target's criteria would quietly fall back to
the built-in implementation and report its numbers as the target's. The target
checks its step contract on the first forward; the switch travels to the ranks as
``env_overrides``, scoped to the engine.

The NVFP4 targets run the NVFP4 release on 4 Rubin GPUs (sm_107) with its FP8 KV cache
and an fp16 recurrent state. The BF16 targets run the deployment their route admits: the BF16 release over 8
GB200 GPUs (two trays of four), tensor-parallel attention and mamba, the recurrent
state in the checkpoint's fp32, block reuse off. GSM8K runs with thinking off, the
protocol the built-in TestNemotronV3Ultra gates this model with.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

from ....conftest import check_device_contain, llm_models_root
from ...accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from .._env import scoped_env_overrides

# The registry key upstream uses for this model. The BF16 release declares no
# quantization, so the engine resolves the entry with neither a quant_algo nor a
# kv_cache_quant_algo.
MODEL_NAME = "nvidia/Nemotron-Ultra-V3"
BF16_MODEL_PATH = f"{llm_models_root()}/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
NVFP4_MODEL_PATH = f"{llm_models_root()}/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"

# The targets assert their own SM at construction: certification is per GPU
# architecture, and only GB200 trays were measured for the BF16 targets.
skip_not_gb200 = pytest.mark.skipif(
    get_sm_version() != 100 or not check_device_contain(["GB200"]),
    reason="the nemotron_h BF16 targets are certified on GB200 (sm_100) only",
)

skip_not_sm107 = pytest.mark.skipif(
    get_sm_version() != 107,
    reason="the nemotron_h NVFP4 targets are certified on sm_107 only",
)

_EXTRA_EVALUATOR_KWARGS = dict(chat_template_kwargs=dict(enable_thinking=False))


class NemotronHBf16Target(LlmapiAccuracyTestHarness):
    """One BF16 target: the subclass sets the MoE expert-parallel size its layout uses."""

    MODEL_NAME = MODEL_NAME
    MODEL_PATH = BF16_MODEL_PATH
    MOE_EP = 8

    def _gsm8k(
        self, max_batch_size: int, kv_cache_dtype: str = "auto", moe_backend: str | None = None
    ) -> None:
        # `moe_backend` asks for the routed experts' backend by name; None leaves the LLM
        # API's default (AUTO), which runs the target's own fused_moe.
        moe_kwargs = {} if moe_backend is None else dict(moe_config=MoeConfig(backend=moe_backend))
        # 1.1 TB of weights: 8 ranks, about 128 GiB of them on each 185 GiB GPU, and the
        # fraction applies to what is left. The batch ceiling is the serving envelope the
        # targets were gated at: decode graphs up to that many rows, and a recurrent-state
        # pool deep enough (271 slots at 256) that its last slots lie past 2^31 elements of
        # the fp32 state.
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(
                self.MODEL_PATH,
                modeling_v2="require",
                env_overrides=rank_env,
                kv_cache_config=KvCacheConfig(
                    dtype=kv_cache_dtype,
                    enable_block_reuse=False,
                    free_gpu_memory_fraction=0.5,
                    mamba_ssm_cache_dtype="float32",
                ),
                max_batch_size=max_batch_size,
                tensor_parallel_size=8,
                moe_expert_parallel_size=self.MOE_EP,
                enable_attention_dp=False,
                cuda_graph_config=CudaGraphConfig(
                    max_batch_size=max_batch_size, enable_padding=True
                ),
                **moe_kwargs,
            ) as llm,
        ):
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=_EXTRA_EVALUATOR_KWARGS)


class NemotronHNvfp4Target(LlmapiAccuracyTestHarness):
    """One NVFP4 target on 4 Rubin GPUs: the subclass says whether attention is data
    parallel (dep4) or tensor parallel (tep4). The routed experts are expert-parallel over
    the 4 ranks either way, the KV cache is the checkpoint's FP8 and the recurrent state
    fp16, the configuration of the built-in TestNemotronV3Ultra TEP4 runs these targets
    were gated with."""

    MODEL_NAME = MODEL_NAME
    MODEL_PATH = NVFP4_MODEL_PATH
    ATTENTION_DP = False

    def _gsm8k(self, max_batch_size: int = 32) -> None:
        # With attention DP every rank serves its own requests: max_batch_size is per rank.
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(
                self.MODEL_PATH,
                modeling_v2="require",
                env_overrides=rank_env,
                kv_cache_config=KvCacheConfig(
                    enable_block_reuse=False,
                    mamba_ssm_cache_dtype="float16",
                    free_gpu_memory_fraction=0.5,
                ),
                max_batch_size=max_batch_size,
                tensor_parallel_size=4,
                moe_expert_parallel_size=4,
                enable_attention_dp=self.ATTENTION_DP,
                cuda_graph_config=CudaGraphConfig(
                    max_batch_size=max_batch_size, enable_padding=True
                ),
            ) as llm,
        ):
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=_EXTRA_EVALUATOR_KWARGS)
