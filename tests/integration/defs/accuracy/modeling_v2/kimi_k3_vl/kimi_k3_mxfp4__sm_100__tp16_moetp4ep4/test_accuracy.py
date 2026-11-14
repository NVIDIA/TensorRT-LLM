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
"""GSM8K and acceptance-length gates for the MXFP4 Kimi K3 target kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 (SM100)."""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    DSparkDecodingConfig,
    KvCacheConfig,
    MoeConfig,
    SADecodingConfig,
)

from .....conftest import llm_models_root
from ....accuracy_core import (
    GSM8K,
    LlmapiAccuracyTestHarness,
    assert_acceptance_length,
    compute_acceptance_length,
)
from ..._env import scoped_env_overrides
from .._common import _GB200_ONLY, _MODEL_NAME, _MODEL_PATH, _run_gsm8k

# Both legs of each acceptance gate run this, the stock leg with modeling_v2 "off".
_MODES = pytest.mark.parametrize("mode", ["require", "off"], ids=["modeling_v2", "stock"])


def _run_spec_gsm8k_and_acceptance(spec_config, mode: str, acceptance_key: str) -> None:
    """Full GSM8K with speculative decoding, then the mean acceptance length against ``acceptance_key``.

    The settings of the stock Kimi K3 speculative tests (``test_kimi3.py::TestKimiK3::test_w4a16_mxfp4[sa|dspark]``)
    at this target's layout: batch 8, overlap scheduler and chunked prefill off, iteration stats on. The KV fraction
    leaves the DSpark drafter and its captured target states their headroom; batch 8 does not fit the no-spec test's
    8192-token cache. The modeling_v2 leg captures CUDA graphs at every batch size it serves; the stock leg runs eager,
    the only mode the built-in Kimi K3 DSpark verification is qualified in.
    """
    cuda_graph_config = (
        CudaGraphConfig(batch_sizes=list(range(1, 9)), enable_padding=True)
        if mode == "require"
        else None
    )
    with (
        scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
        LLM(
            _MODEL_PATH,
            tensor_parallel_size=16,
            pipeline_parallel_size=1,
            moe_tensor_parallel_size=4,
            moe_expert_parallel_size=4,
            enable_attention_dp=False,
            modeling_v2=mode,
            trust_remote_code=True,
            max_batch_size=8,
            max_num_tokens=8192,
            max_seq_len=8192,
            disable_overlap_scheduler=True,
            enable_chunked_prefill=False,
            kv_cache_config=KvCacheConfig(
                dtype="auto",
                tokens_per_block=64,
                enable_block_reuse=False,
                use_kv_cache_manager_v2=True,
                mamba_ssm_cache_dtype="float32",
                free_gpu_memory_fraction=0.20,
            ),
            cuda_graph_config=cuda_graph_config,
            moe_config=MoeConfig(backend="TRTLLM", use_low_precision_moe_combine=True),
            speculative_config=spec_config,
            enable_iter_perf_stats=True,
            max_stats_len=-1,
            env_overrides=rank_env,
        ) as llm,
    ):
        GSM8K(_MODEL_NAME).evaluate(llm)
        acceptance_length = compute_acceptance_length(llm)
        print(f"[AL] {acceptance_key} {mode} acceptance_length = {acceptance_length:.3f}")
        assert_acceptance_length(acceptance_key, acceptance_length)


@_GB200_ONLY
@pytest.mark.skip_less_mpi_world_size(16)
@pytest.mark.timeout(10800)
class TestModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4(LlmapiAccuracyTestHarness):
    """Attention TP16 with routed experts split over TP4 and EP4."""

    MODEL_NAME = _MODEL_NAME
    MODEL_PATH = _MODEL_PATH

    # One anchor per variant, shared by both legs of its gate. It names the target and the variant, not a test
    # function, because that is what the number is a property of. Populate it from the STOCK leg where that leg has a
    # reading: taking it from the target's own number would make the gate self-referential. An anchor without a
    # STOCK reading says so in the reference file.
    DSPARK7_KEY = "ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4::dspark7"
    DSPARK7_MLA_KEY = "ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4::dspark7_mla"
    SA2_KEY = "ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4::sa2"

    def test_gsm8k(self) -> None:
        _run_gsm8k(self.MODEL_PATH, moe_tp_size=4, moe_ep_size=4)

    @_MODES
    def test_dspark7_gsm8k_and_acceptance(self, mode: str) -> None:
        """DSpark with the RadixArk GQA drafter, 7 drafts.

        Rejection sampling makes a wrong draft path slower, not wrong, so the GSM8K gate stays green while every
        draft is rejected; acceptance length is the only detector. Both legs read the same anchor:

          modeling_v2 fails, stock passes -> the target's draft path regressed
          both fail                       -> the anchor is stale; re-derive it
        """
        spec_config = DSparkDecodingConfig(
            max_draft_len=7,
            speculative_model=f"{llm_models_root()}/RadixArk-Kimi-K3-DSpark",
        )
        _run_spec_gsm8k_and_acceptance(spec_config, mode, self.DSPARK7_KEY)

    @_MODES
    def test_dspark7_mla_gsm8k_and_acceptance(self, mode: str) -> None:
        """DSpark with the Inferact MLA drafter (K3DSparkModel), 7 drafts, at the config's default attention backend.

        The MLA drafter has its own draft path and its own anchor; the two legs read it as for the GQA drafter.
        """
        spec_config = DSparkDecodingConfig(
            max_draft_len=7,
            speculative_model=f"{llm_models_root()}/Inferact-Kimi-K3-DSpark",
        )
        _run_spec_gsm8k_and_acceptance(spec_config, mode, self.DSPARK7_MLA_KEY)

    @_MODES
    def test_sa2_gsm8k_and_acceptance(self, mode: str) -> None:
        """Suffix-automaton drafting, 2 drafts; the two legs read one anchor, as for DSpark."""
        spec_config = SADecodingConfig(max_draft_len=2)
        _run_spec_gsm8k_and_acceptance(spec_config, mode, self.SA2_KEY)
