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
"""Whole-model gate for the deepseek_v41 modeling_v2 target flash_nvfp4__sm_103__dep2.

The harness is in ``../_common.py``.
"""

import pytest

from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

from .._common import _DeepseekV41Gate, skip_not_sm103


class TestModelingV2DeepseekV41FlashNvfp4Sm103Dep2(_DeepseekV41Gate):
    """deepseek-v4.1-flash-nvfp4 / sm_103 / dep2: TP2 + attention DP + MoE EP2, no speculation (ctx role)."""

    @skip_not_sm103
    @pytest.mark.skip_less_device(2)
    @pytest.mark.timeout(7200)
    def test_gsm8k(self, mocker):
        self._gsm8k(
            mocker,
            tensor_parallel_size=2,
            moe_expert_parallel_size=2,
            enable_attention_dp=True,
            moe_config=MoeConfig(backend="CUTEDSL"),
            disable_overlap_scheduler=True,
            max_batch_size=16,
            max_num_tokens=8192,
            max_seq_len=8192,
            kv_cache_config=KvCacheConfig(
                dtype="fp8",
                tokens_per_block=128,
                enable_block_reuse=False,
                free_gpu_memory_fraction=0.8,
            ),
        )
