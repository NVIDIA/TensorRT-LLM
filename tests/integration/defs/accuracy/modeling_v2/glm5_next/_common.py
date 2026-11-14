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
"""Full-GSM8K gates for the NVFP4 GLM-5.3-Flash targets on SM100 (B200).

Two targets share the deployment and differ in how speculative decoding runs:
the split target (no speculation, or the engine's speculative worker after its
forward) and the MTP-fused target (one-model MTP-Eagle inside its forward). Each test
picks a configuration that routes to exactly one of them. Requiring
modeling_v2 prevents an unsupported deployment from silently testing the
built-in model.

The ranks build the model in their own processes, so a test cannot read the
built target back. Which target each configuration builds is pinned by the
routing unit test (``test_glm5_next_routing.py``): TP4/EP4 with three-draft
MTP-Eagle routes to the fused target (``tep4-mtp3``), and TP4/EP4 without
speculation or with vanilla MTP to the split target (``no-spec``,
``vanilla-mtp``).
"""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

from ....conftest import llm_models_root
from .._env import scoped_env_overrides

_MODEL_NAME = "zai-org/GLM-5.3-Flash"
_MODEL_PATH = f"{llm_models_root()}/GLM-5.3-Flash-NVFP4"


@contextmanager
def _llm(model_path: str, speculative_config=None, **llm_kwargs) -> Iterator[LLM]:
    """The engine at TP4/EP4, with the targets' step contract checked on the ranks."""
    with (
        scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
        LLM(
            model_path,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            moe_expert_parallel_size=4,
            modeling_v2="require",
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5, enable_block_reuse=False),
            max_batch_size=64,
            max_num_tokens=16384,
            max_seq_len=8192,
            cuda_graph_config=CudaGraphConfig(max_batch_size=64, enable_padding=True),
            enable_chunked_prefill=True,
            disable_mm_encoder=True,
            speculative_config=speculative_config,
            env_overrides=rank_env,
            **llm_kwargs,
        ) as llm,
    ):
        yield llm


_SM100_TP4 = (
    pytest.mark.skipif(
        get_sm_version() != 100,
        reason="The modeling_v2 GLM-5.3-Flash targets require SM100 (B200).",
    ),
    pytest.mark.skip_less_mpi_world_size(4),
    pytest.mark.timeout(7200),
)


def _sm100_tp4(cls):
    for mark in _SM100_TP4:
        cls = mark(cls)
    return cls
