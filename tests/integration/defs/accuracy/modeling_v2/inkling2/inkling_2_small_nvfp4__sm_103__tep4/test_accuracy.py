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
"""Whole-model gate for the inkling2 modeling_v2 target inkling_2_small_nvfp4__sm_103__tep4.

The test passes `modeling_v2="require"`: there is no built-in implementation of this
architecture, so `auto` would resolve to the same target, but `require` states that this is the
target under test.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import KvCacheConfig

from .....conftest import llm_models_root
from ....accuracy_core import GSM8KInferenceX, LlmapiAccuracyTestHarness

skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="the target is certified on sm_103 (GB300) only"
)


class TestModelingV2Inkling2SmallNvfp4Sm103Tep4(LlmapiAccuracyTestHarness):
    """Inkling-2-Small as released, on one GB300 tray with attention TP4 and MoE EP4, eager.

    The checkpoint stores NVFP4 experts in layers 0-39 and MXFP8 in 40-47; the KV cache is BF16.
    The protocol is GSM8K under InferenceX (`GSM8KInferenceX`): the chat template with the
    template's default reasoning effort, strict-match scoring. The reference is the model's
    SGLang implementation measured under the same protocol on the same tray.
    """

    MODEL_NAME = "thinkingmachines/Inkling-2-Small"
    MODEL_PATH = f"{llm_models_root()}/Inkling-2-Small"

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_gsm8k(self):
        with LLM(
            self.MODEL_PATH,
            tensor_parallel_size=4,
            moe_expert_parallel_size=4,
            modeling_v2="require",
            max_seq_len=32768,
            max_batch_size=64,
            max_num_tokens=16384,
            kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.6),
            cuda_graph_config=None,
            disable_overlap_scheduler=True,
            enable_chunked_prefill=False,
        ) as llm:
            task = GSM8KInferenceX(self.MODEL_NAME)
            task.evaluate(llm)
