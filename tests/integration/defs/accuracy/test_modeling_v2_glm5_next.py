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
"""Whole-model gates for the glm5_next modeling_v2 targets.

Every engine here passes ``modeling_v2="require"`` (the stock leg of the acceptance gate asks
for ``"off"`` on purpose): under ``"auto"`` a configuration that missed the target's criteria
would fall back to the built-in implementation and report its numbers as the target's.

The checkpoint is the text half of a ``Glm5NextForConditionalGeneration`` export; the target
builds no vision tower, so the engines run with ``disable_mm_encoder``.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MTPDecodingConfig

from ..conftest import llm_models_root
from .accuracy_core import GSM8K, LlmapiAccuracyTestHarness

# The target's kernels are sm_100a / sm_103a code, built at first use for the device's SM.
skip_not_sm10x = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="the glm5_next modeling_v2 target is built for sm_100 and sm_103 only",
)


class TestModelingV2Glm53FlashNvfp4Sm10xTp4(LlmapiAccuracyTestHarness):
    """glm-5.3-flash-nvfp4 / sm_10x / tp4, with the MTP-Eagle drafter at three drafts."""

    MODEL_NAME = "zai-org/GLM-5.3-Flash"
    MODEL_PATH = f"{llm_models_root()}/GLM-5.3-Flash-NVFP4"

    # The parallel topologies routing admits (tensor-parallel attention; the routed experts split
    # by intermediate or by expert), and what the target cannot serve without: 128-token pages
    # (the latent and INDEX_KEY page layouts its kernels address) and at most 256 sequences per
    # step. Block reuse with a KDA snapshot at the end of every prompt is the serving shape; the
    # 5-shot prompts share a long prefix, so most requests exercise it.
    LAYOUTS = {
        "tp4": dict(tensor_parallel_size=4, moe_tensor_parallel_size=4, moe_expert_parallel_size=1),
        "tp4_ep4": dict(
            tensor_parallel_size=4, moe_tensor_parallel_size=1, moe_expert_parallel_size=4
        ),
    }
    ENGINE = dict(
        max_batch_size=32,
        max_num_tokens=8192,
        max_seq_len=16384,
        enable_chunked_prefill=True,
        disable_mm_encoder=True,
        cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=32),
        kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=0.8,
            tokens_per_block=128,
            enable_block_reuse=True,
            mamba_state_config={"additional_snapshot_offsets_from_end": [0]},
        ),
    )
    MTP3 = MTPDecodingConfig(max_draft_len=3)

    @skip_not_sm10x
    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize("layout", ["tp4", "tp4_ep4"])
    def test_gsm8k_mtp3(self, layout):
        """The accuracy gate, on the configurations the target is served in.

        The engine also checks its step contract on its first forward of each phase: the
        metadata it reads is private runtime surface, and a run that prepared it is the only
        honest place to check it.
        """
        with LLM(
            self.MODEL_PATH,
            modeling_v2="require",
            env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
            speculative_config=self.MTP3,
            **self.LAYOUTS[layout],
            **self.ENGINE,
        ) as llm:
            GSM8K(self.MODEL_NAME).evaluate(llm)
