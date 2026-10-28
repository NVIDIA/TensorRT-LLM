# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Full-GSM8K gates for the two MXFP4 Kimi K3 targets on SM100.

Both layouts require 16 GB200 GPUs in one NVLink domain. Requiring modeling_v2
prevents an unsupported deployment from silently testing the built-in model.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

from ...conftest import check_device_contain, llm_models_root
from ..accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from ._env import scoped_env_overrides

_MODEL_NAME = "moonshotai/Kimi-K3"
_MODEL_PATH = f"{llm_models_root()}/Kimi-K3"

_GB200_ONLY = pytest.mark.skipif(
    get_sm_version() != 100 or not check_device_contain(["GB200"]),
    reason="These modeling_v2 Kimi K3 targets require SM100 (GB200).",
)


def _run_gsm8k(model_path: str, moe_tp_size: int, moe_ep_size: int) -> None:
    with (
        scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
        LLM(
            model_path,
            dtype="bfloat16",
            tensor_parallel_size=16,
            pipeline_parallel_size=1,
            moe_tensor_parallel_size=moe_tp_size,
            moe_expert_parallel_size=moe_ep_size,
            enable_attention_dp=False,
            modeling_v2="require",
            trust_remote_code=True,
            max_batch_size=1,
            max_num_tokens=GSM8K.MAX_INPUT_LEN,
            max_seq_len=8192,
            enable_chunked_prefill=True,
            kv_cache_config=KvCacheConfig(
                dtype="auto",
                tokens_per_block=64,
                enable_block_reuse=False,
                use_kv_cache_manager_v2=True,
                mamba_ssm_cache_dtype="float32",
                max_tokens=8192,
            ),
            cuda_graph_config=CudaGraphConfig(batch_sizes=[1], enable_padding=True),
            moe_config=MoeConfig(backend="TRTLLM", use_low_precision_moe_combine=True),
            env_overrides=rank_env,
        ) as llm,
    ):
        GSM8K(_MODEL_NAME).evaluate(llm)


@_GB200_ONLY
@pytest.mark.skip_less_mpi_world_size(16)
@pytest.mark.timeout(10800)
class TestModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4(LlmapiAccuracyTestHarness):
    """Attention TP16 with routed experts split over TP4 and EP4."""

    MODEL_NAME = _MODEL_NAME
    MODEL_PATH = _MODEL_PATH

    def test_gsm8k(self) -> None:
        _run_gsm8k(self.MODEL_PATH, moe_tp_size=4, moe_ep_size=4)


@_GB200_ONLY
@pytest.mark.skip_less_mpi_world_size(16)
@pytest.mark.timeout(10800)
class TestModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1(LlmapiAccuracyTestHarness):
    """Attention TP16 with an explicitly selected MoE TP16 and EP1 layout."""

    MODEL_NAME = _MODEL_NAME
    MODEL_PATH = _MODEL_PATH

    def test_gsm8k(self) -> None:
        _run_gsm8k(self.MODEL_PATH, moe_tp_size=16, moe_ep_size=1)
