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
"""DWDP accuracy tests in aggregated serving.

DWDP was originally built for the context phase of disaggregated serving, where
each context worker is its own TP=1 instance and the workers are joined into one
MPI world by ``trtllm-serve disaggregated_mpi_worker``. The same invariant --
every rank is a complete model replica owning one expert slice -- also holds for
a single aggregated instance running attention DP, because ``Mapping.dp_size ==
tp_size`` there and attention is replicated rather than tensor-sharded.

Running DWDP aggregated keeps the expert-sharing paths under test while removing
the disaggregated KV cache transceiver from the picture, so these tests do not
depend on the UCX transport configuration of the cluster they run on.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig
from tensorrt_llm.llmapi.llm_args import DwdpConfig

from ..conftest import llm_models_root, skip_post_blackwell_ultra, skip_pre_blackwell
from .accuracy_core import GSM8K, LlmapiAccuracyTestHarness

# DeepSeek-V3-Lite has 72 routed experts, partitioned across ``DWDP_SIZE``
# workers. Mode A is the uniform partition (``size == stride == 72 //
# DWDP_SIZE``); Mode B uses ``size > stride`` so adjacent peer ranges overlap,
# and ``(DWDP_SIZE - 1) * stride + size == 72`` must hold exactly.
#
# DWDP_SIZE is 4 rather than the 2 the disaggregated tests used: aggregated
# serving has no generation server, so the whole allocation goes to DWDP peers.
# Three remote peers per rank instead of one is also what makes
# ``contention_opt`` meaningful, since it interleaves prefetch slices across
# peers to spread them over several NVLink links.
#
# The MPI world must be exactly ``num_groups * dwdp_size`` ranks -- a rank
# computes ``group_id = rank // dwdp_size`` and DwdpManager rejects
# ``group_id >= num_groups``. ``tensor_parallel_size`` below is that world size,
# so it has to track DWDP_SIZE and the ``num_groups=1`` passed to DwdpConfig.
DWDP_SIZE = 4


class TestDwdpAggDeepSeekV3Lite(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-V3-Lite"

    @pytest.mark.skip_less_device(DWDP_SIZE)
    @skip_pre_blackwell
    @skip_post_blackwell_ultra
    @pytest.mark.parametrize(
        "num_experts_per_worker,num_prefetch_experts,contention_opt",
        [
            (18, 18, False),
            (24, 16, False),
            (18, 18, True),
        ],
        ids=["mode_a_uniform", "mode_b_overlap", "mode_a_uniform_contention_opt"],
    )
    def test_dwdp_agg_accuracy(
        self,
        num_experts_per_worker: int,
        num_prefetch_experts: int,
        contention_opt: bool,
    ) -> None:
        dwdp_config = DwdpConfig(
            dwdp_size=DWDP_SIZE,
            num_groups=1,
            num_experts_per_worker=num_experts_per_worker,
            num_prefetch_experts=num_prefetch_experts,
            contention_opt=contention_opt,
        )

        # Attention DP makes each of the DWDP_SIZE ranks an independent replica
        # serving its own requests; DWDP supplies the expert weights a rank does
        # not hold locally. Overlap scheduling is not supported with DWDP.
        with LLM(
            f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only_mtp",
            tensor_parallel_size=DWDP_SIZE,
            enable_attention_dp=True,
            dwdp_config=dwdp_config,
            moe_config=MoeConfig(backend="CUTEDSL"),
            disable_overlap_scheduler=True,
            enable_autotuner=False,
            enable_chunked_prefill=False,
            cuda_graph_config=None,
            max_batch_size=16,
            max_num_tokens=8192,
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=0.4,
                enable_block_reuse=False,
                enable_partial_reuse=False,
                tokens_per_block=32,
            ),
        ) as llm:
            # Guard against DWDP silently not running at all. If the config were
            # dropped before create_py_executor, MoE would fall back to the normal
            # parallel path, which is also correct and scores the same, so the
            # accuracy check below could not tell the two apart. create_py_executor
            # either honours dwdp_config or raises -- it has no branch that ignores
            # it -- so an LLM that constructed successfully while still carrying the
            # config had a DwdpManager built for it.
            assert llm.args.dwdp_config == dwdp_config

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
