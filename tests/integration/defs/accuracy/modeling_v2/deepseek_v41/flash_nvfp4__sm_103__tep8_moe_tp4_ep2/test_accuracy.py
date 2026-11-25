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
"""Whole-model gate for the deepseek_v41 modeling_v2 target flash_nvfp4__sm_103__tep8_moe_tp4_ep2.

The harness is in ``../_common.py``.
"""

import pytest

from tensorrt_llm.llmapi import CudaGraphConfig, DSparkDecodingConfig, KvCacheConfig, MoeConfig

from .._common import _DeepseekV41Gate, skip_not_sm103


class TestModelingV2DeepseekV41FlashNvfp4Sm103Tep8MoeTp4Ep2(_DeepseekV41Gate):
    """deepseek-v4.1-flash-nvfp4 / sm_103 / tep8_moe_tp4_ep2: TP8 attention, MoE tp4 x ep2, DSpark (gen role).

    Two GB300 trays (8 ranks). Run with and without speculation: DSpark is lossless, so both rows share one
    floor, and the speculative row proves the target's verification path. Whether the draft reads the right
    hidden states shows in the acceptance length, not the accuracy; the acceptance length is gated against the
    stock implementation in the disaggregated measurement, not here.
    """

    @skip_not_sm103
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.timeout(7200)
    @pytest.mark.parametrize("dspark", [False, True], ids=["no_spec", "dspark"])
    def test_gsm8k(self, mocker, dspark):
        speculative_config = (
            DSparkDecodingConfig(max_draft_len=5, speculative_model=self.MODEL_PATH)
            if dspark
            else None
        )
        self._gsm8k(
            mocker,
            tensor_parallel_size=8,
            moe_expert_parallel_size=2,
            enable_attention_dp=False,
            moe_config=MoeConfig(backend="TRTLLM", use_low_precision_moe_combine=True),
            cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4, 8, 16], enable_padding=True),
            max_batch_size=16,
            max_num_tokens=8192,
            max_seq_len=8192,
            kv_cache_config=KvCacheConfig(
                dtype="fp8",
                tokens_per_block=128,
                enable_block_reuse=False,
                free_gpu_memory_fraction=0.5,
            ),
            speculative_config=speculative_config,
        )
