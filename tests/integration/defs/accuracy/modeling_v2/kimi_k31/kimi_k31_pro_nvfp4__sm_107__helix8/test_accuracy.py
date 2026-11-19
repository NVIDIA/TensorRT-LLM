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
"""Whole-model gate for the kimi_k31 modeling_v2 target kimi_k31_pro_nvfp4__sm_107__helix8.

The recipe of the sm_107 targets is in ``../_common.py``.
"""

import pytest

from tensorrt_llm.quantization import QuantAlgo

from .....conftest import skip_no_rubin
from ....accuracy_core import LlmapiAccuracyTestHarness
from ....test_disaggregated_serving import (
    DEFAULT_ACC_EVALUATION_TIMEOUT,
    launch_disaggregated_llm,
    run_accuracy_test,
)
from .._common import (
    NVFP4_DFLASH2_PATH,
    NVFP4_MODEL_NAME,
    NVFP4_PATH,
    nvfp4_dflash2,
    nvfp4_kv_cache,
    nvfp4_sparse_attention,
)


@skip_no_rubin
class TestModelingV2KimiK31ProNvfp4Sm107Helix8(LlmapiAccuracyTestHarness):
    """``kimi_k31_pro_nvfp4__sm_107__helix8``: Helix context parallelism over 8 ranks.

    A generation layout: the gate runs it disaggregated behind a DEP16 context server, 24 GPUs on
    one host. No Rubin host has 24 GPUs, so this test skips there; a disaggregated helix8
    deployment over several hosts is the gate in practice.
    """

    MODEL_NAME = NVFP4_MODEL_NAME
    MODEL_PATH = NVFP4_PATH

    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_device(24)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_helix8(self):
        """GSM8K through a helix8 generation server at batch 16, draft length 5, MoE TP 2 x EP 4.

        The generation server runs eagerly, without the overlap scheduler.
        """
        transceiver = {"backend": "NIXL", "transceiver_runtime": "PYTHON"}
        spec = {
            "decoding_type": "DFlash",
            "max_draft_len": 5,
            "speculative_model": NVFP4_DFLASH2_PATH,
        }
        ctx_server_config = {
            "tensor_parallel_size": 16,
            "moe_expert_parallel_size": 16,
            "enable_attention_dp": True,
            "modeling_v2": "require",
            "max_batch_size": 16,
            "max_num_tokens": 8192,
            "max_seq_len": 8192,
            "trust_remote_code": True,
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "kv_cache_config": nvfp4_kv_cache(),
            "moe_config": {"backend": "CUTEDSL", "use_low_precision_moe_combine": True},
            "sparse_attention_config": nvfp4_sparse_attention(),
            "speculative_config": spec,
            "cache_transceiver_config": transceiver,
        }
        gen_server_config = {
            "tensor_parallel_size": 1,
            "context_parallel_size": 8,
            "moe_tensor_parallel_size": 2,
            "moe_expert_parallel_size": 4,
            "cp_config": {
                "cp_type": "HELIX",
                "tokens_per_block": 64,
                "use_nccl_for_alltoall": False,
                "fifo_version": 2,
            },
            "enable_attention_dp": False,
            "modeling_v2": "require",
            "max_batch_size": 16,
            "max_num_tokens": 96,
            "max_seq_len": 8192,
            "trust_remote_code": True,
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "kv_cache_config": nvfp4_kv_cache(),
            "moe_config": {"backend": "CUTEDSL_FC12", "use_low_precision_moe_combine": True},
            "sparse_attention_config": nvfp4_sparse_attention(),
            "speculative_config": spec,
            "cache_transceiver_config": transceiver,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "port": 8000,
            "backend": "pytorch",
            "context_servers": {"num_instances": 1, "urls": ["localhost:8001"]},
            "generation_servers": {"num_instances": 1, "urls": ["localhost:8002"]},
        }
        with launch_disaggregated_llm(
            disaggregated_server_config,
            ctx_server_config,
            gen_server_config,
            self.MODEL_PATH,
            request_timeout_s=DEFAULT_ACC_EVALUATION_TIMEOUT,
        ) as llm:
            # The DuckLLM arguments do not carry what the servers run; set the specs the reference
            # lookup matches on (the export, an FP8 KV cache, DFlash2).
            llm.args.quant_config.quant_algo = QuantAlgo.MIXED_PRECISION
            llm.args.quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            llm.args.speculative_config = nvfp4_dflash2(5)
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])
