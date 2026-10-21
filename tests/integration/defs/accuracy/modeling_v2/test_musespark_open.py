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
"""Whole-model gate for the musespark_open modeling_v2 TEP target.

The test passes ``modeling_v2="require"``. A deployment outside the target's
compositions (GPU architecture, tensor-parallel size, DFlash with seven draft
tokens) then raises instead of running the built-in model, and a composition
whose preconditions are not met raises under any mode, so a pass is the
target's fast paths end to end.
"""

import os

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, DFlashDecodingConfig, KvCacheConfig, MoeConfig

from ...conftest import llm_models_root
from ..accuracy_core import GSM8K, LlmapiAccuracyTestHarness, assert_acceptance_length_for_llm

# The TEP4 compositions are written for sm103 (GB300) and sm107 (Vera Rubin).
skip_not_served_sm = pytest.mark.skipif(
    get_sm_version() not in (103, 107),
    reason="the musespark_open TEP target serves sm103 and sm107 only",
)

# The public opt-ins the compositions are written against: PDL launches, the
# MNNVL one-shot attention all-reduce, the fused MoE finalize all-reduce, the
# sharded projections, the low-M GEMM dispatcher and the windowed drafter context
# arena. The target refuses to build without them. The KV-cache estimation pass is skipped because its
# temporary cache does not fit next to this checkpoint's weights.
_SERVED_ENV = {
    "TRTLLM_ENABLE_PDL": "1",
    "MUSESPARK_OPEN_ALLREDUCE": "MNNVL",
    "MUSESPARK_OPEN_FUSED_FINALIZE_AR_MNNVL": "1",
    "MUSESPARK_OPEN_SHARDED_PROJECTIONS": "1",
    "TRTLLM_LOW_M_GEMM_BACKEND": "auto",
    "TRTLLM_DFLASH_WINDOWED_CTX_ARENA": "1",
    "TRTLLM_SKIP_KV_CACHE_ESTIMATION": "1",
    MODELING_V2_VALIDATE_ENV: "1",
}


def _require_target(monkeypatch: pytest.MonkeyPatch, **env: str) -> dict:
    """``LLM(...)`` kwargs that build the target, with ``env`` on the ranks.

    ``LLM`` applies ``env_overrides`` to this process and never restores it, so
    ``monkeypatch`` sets it too, keeping it from leaking into later tests.
    """
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return dict(modeling_v2="require", env_overrides=dict(env))


@pytest.mark.timeout(7200)
class TestModelingV2MuseSparkOpenNvfp4Tep4(LlmapiAccuracyTestHarness):
    """muse-spark-open-816b-a42b-nvfp4 / sm103 and sm107 / TEP4, DFlash2 with seven draft tokens."""

    # The reference key of the built-in model's suite (test_musespark_open.py).
    MODEL_NAME = "musespark_open"
    # Pre-release the checkpoints are staged outside llm-models and named by
    # these variables; the llm_models_root() paths take over once they land there.
    MODEL_PATH = os.environ.get("MUSESPARK_REFERENCE_CKPT") or (
        f"{llm_models_root()}/MuseSparkOpen"
    )
    DFLASH2_MODEL_PATH = os.environ.get("MUSESPARK_DFLASH2_CKPT") or (
        f"{llm_models_root()}/Muse-Spark-Open-816B-A42B-DFlash2"
    )

    # Acceptance-length anchor, keyed by target and variant.
    ACCEPTANCE_KEY = "ModelingV2MuseSparkOpenNvfp4Tep4::dflash2"

    ENGINE = dict(
        # The registered tokenizer wrapper: the chat template, the pad token and
        # both EOS ids (see test_musespark_open.py).
        custom_tokenizer="musespark_open",
        trust_remote_code=True,
        tensor_parallel_size=4,
        moe_expert_parallel_size=4,
        enable_attention_dp=False,
        attn_backend="TRTLLM",
        allreduce_strategy="NCCL",
        moe_config=MoeConfig(backend="TRTLLM"),
        max_batch_size=16,
        max_num_tokens=8192,
        max_seq_len=8192,
        enable_chunked_prefill=True,
        kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=0.8,
            enable_block_reuse=True,
            dtype="fp8",
            tokens_per_block=32,
            use_kv_cache_manager_v2=True,
        ),
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4, 8, 16], enable_padding=True),
        enable_iter_perf_stats=True,
        max_stats_len=-1,
    )

    @skip_not_served_sm
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(180000)
    def test_dflash2_gsm8k_and_acceptance(self, monkeypatch):
        """GSM8K against the model's DFlash floor, plus acceptance length.

        DFlash verification is lossless, so a wrong target forward shows up in
        the GSM8K score. A wrong drafter forward only rejects drafts, so
        acceptance length is its detector.
        """
        with LLM(
            self.MODEL_PATH,
            speculative_config=DFlashDecodingConfig(
                max_draft_len=7, speculative_model=self.DFLASH2_MODEL_PATH
            ),
            **self.ENGINE,
            **_require_target(monkeypatch, **_SERVED_ENV),
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            assert_acceptance_length_for_llm(self.ACCEPTANCE_KEY, llm)
