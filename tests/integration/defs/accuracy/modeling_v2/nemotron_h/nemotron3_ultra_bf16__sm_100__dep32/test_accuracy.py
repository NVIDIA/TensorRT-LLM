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
"""Whole-model gate for nemotron3_ultra_bf16__sm_100__dep32: GSM8K on eight GB200 trays.

Not in any test list yet: the target is not GPU-certified until these gates pass on 32 GB200
GPUs. Once they do, list them in tests/integration/test_lists/qa/llm_function_multinode.txt
beside dep16's and drop this directory from ``_ACCURACY_NOT_LISTED_YET`` in
tests/unittest/_torch/modeling_v2/test_modeling_v2_ci_selection.py.
"""

import pytest

from .._common import NemotronHBf16Target, skip_not_gb200

# Prefill CUDA graph buckets: powers of two to 512, then coarser steps to the 4096-token step.
_BCG_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]


@skip_not_gb200
@pytest.mark.skip_less_mpi_world_size(32)
class TestModelingV2Nemotron3UltraBf16Sm100Dep32(NemotronHBf16Target):
    """Nemotron-3 Ultra 550B-A55B BF16 / sm_100 / dep32.

    Attention data parallel over 32 ranks, everything but the routed experts whole on every
    rank; the routed experts in thirty-two 16-expert windows, exchanged between ranks by the
    engine's MoE layer. The deployment of the disaggregated context tier at DEP32. Per rank:
    up to 32 requests (decode graphs to 32 rows) and 2048 tokens a step (4096 on the MegaMoE
    path, the disaggregated context tier's chunk), as dep16 was gated.
    """

    WORLD = 32
    MOE_EP = 32
    ATTENTION_DP = True
    # The CUTLASS and TRTLLM gates keep the step at 2048 tokens a rank: their default MoE launch
    # cap is the step times 32 (65536).
    MAX_NUM_TOKENS = 2048

    def test_gsm8k_megamoe_cutedsl(self):
        # The serving path: CuTe DSL MegaMoE (BF16 Relu2, top-22) at the disaggregated context
        # tier's 4096 tokens a rank and its cap of 1024 rows a rank per launch, so a full prefill
        # step runs four launches (the multi-chunk path).
        self._gsm8k(
            max_batch_size=32,
            moe_backend="MEGAMOE_CUTEDSL",
            moe_max_num_tokens=1024,
            max_num_tokens=4096,
        )

    def test_gsm8k_megamoe_cutedsl_bcg(self):
        # The context tier's serving path with breakable prefill CUDA graphs: every bucket up to
        # the 4096-token step captured, the mamba core and attention eager between segments.
        self._gsm8k(
            max_batch_size=32,
            moe_backend="MEGAMOE_CUTEDSL",
            moe_max_num_tokens=1024,
            max_num_tokens=4096,
            prefill_capture_num_tokens=_BCG_BUCKETS,
        )

    def test_gsm8k(self):
        # AUTO: the CUTLASS leaf over the engine's all-to-all.
        self._gsm8k(max_batch_size=32)

    def test_gsm8k_trtllm_moe(self):
        # FlashInfer's TRTLLM-Gen BF16 MoE over the engine's all-to-all.
        self._gsm8k(max_batch_size=32, moe_backend="TRTLLM")
