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
"""Whole-model gates for the glm5_next tp4 target (``glm_5_3_flash_nvfp4__sm_10x__tp4``).

Every engine here passes ``modeling_v2="require"``: under ``"auto"`` a configuration no target
takes would fall back to the built-in implementation and report its numbers as the target's.
The ranks build the model in their own processes, so a test cannot read the built target back.
That the engine arguments below route to this target, and not to the family's sm_100 default
targets, is pinned by the routing unit test (``test_glm5_next_routing.py``,
``test_tp4_engine_routes_to_the_tp4_target``).

The checkpoint is the text half of a ``Glm5NextForConditionalGeneration`` export; the target
builds no vision tower, so the engines run with ``disable_mm_encoder``.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MTPDecodingConfig

from .....conftest import llm_models_root
from ....accuracy_core import (
    GSM8K,
    LlmapiAccuracyTestHarness,
    assert_acceptance_length,
    compute_acceptance_length,
)
from ..._env import scoped_env_overrides

# The target's kernels are sm_100a / sm_103a code; the family library carries both builds.
skip_not_sm10x = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="the glm5_next modeling_v2 target is built for sm_100 and sm_103 only",
)


class TestModelingV2Glm53FlashNvfp4Sm10xTp4(LlmapiAccuracyTestHarness):
    """glm-5.3-flash-nvfp4 / sm_10x / tp4, without speculation and with MTP 3."""

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

    # One anchor for the acceptance check at both layouts: the draft layer computes the same
    # function at either. It is the built-in implementation's reading (modeling_v2="off") under
    # the same speculative config, so the gate is not read against the target's own number.
    ACCEPTANCE_KEY = "ModelingV2Glm53FlashNvfp4Sm10xTp4::mtp3"

    @skip_not_sm10x
    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize("layout", ["tp4", "tp4_ep4"])
    def test_gsm8k_identity_vs_mtp3_and_acceptance(self, layout):
        """GSM8K without and with MTP, MTP not moving the score, and the MTP acceptance length.

        The gates run on the configurations the target is served in. The identity leg is the
        forward without speculation, whose head and decode steps are not the MTP step's. Strict
        acceptance keeps a wrong draft chain from changing any answer: its drafts are rejected and
        the steps only get slower, so both accuracy gates pass, and the acceptance length is the
        one check that sees it.

        Both legs run in this one test: a delta against a score from another session would carry
        that session's variance. Each engine also checks its step contract on its first forward
        of each phase: the metadata it reads is private runtime surface, and a run that prepared
        it is the only honest place to check it.
        """
        task = GSM8K(self.MODEL_NAME)
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(
                self.MODEL_PATH,
                modeling_v2="require",
                env_overrides=rank_env,
                **self.LAYOUTS[layout],
                **self.ENGINE,
            ) as llm,
        ):
            identity = task.evaluate(llm)

        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(
                self.MODEL_PATH,
                modeling_v2="require",
                env_overrides=rank_env,
                speculative_config=self.MTP3,
                enable_iter_perf_stats=True,
                **self.LAYOUTS[layout],
                **self.ENGINE,
            ) as llm,
        ):
            mtp3 = task.evaluate(llm)
            acceptance_length = compute_acceptance_length(llm)

        delta = mtp3 - identity
        print(f"[modeling_v2] gsm8k identity={identity:.4f} mtp3={mtp3:.4f} delta={delta:+.4f}")
        # 2 sigma at the ~0.6 stderr this benchmark reports at n=1319.
        assert abs(delta) < 1.2, (
            f"MTP moved gsm8k by {delta:+.4f} (identity={identity:.4f}, mtp3={mtp3:.4f}); "
            "strict acceptance emits the target's own tokens, so this is not decoding noise"
        )
        print(f"[AL] mtp3 acceptance_length = {acceptance_length:.3f}")
        assert_acceptance_length(self.ACCEPTANCE_KEY, acceptance_length)
