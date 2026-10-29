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
"""Whole-model identity, MTP1 accuracy and acceptance for the GLM SM103 target."""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MTPDecodingConfig

from ...conftest import llm_models_root
from ..accuracy_core import (
    GSM8K,
    LlmapiAccuracyTestHarness,
    assert_acceptance_length,
    compute_acceptance_length,
)

# The targets assert their own SM at construction: certification is per GPU
# architecture, and a receipt from another one says nothing here.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="modeling_v2 targets in this batch are certified on sm_103 only"
)


class TestModelingV2Glm52Nvfp4Sm103Tep4(LlmapiAccuracyTestHarness):
    """glm-5.2-nvfp4 / sm_103 / tep4: identity, the mtp1 variant and its acceptance."""

    MODEL_NAME = "zai-org/GLM-5.2"
    MODEL_PATH = f"{llm_models_root()}/GLM-5.2-NVFP4"

    # The parallel topology the path's tep4 segment declares: attention heads
    # and experts both sharded four ways, attention DP off. Routing derives the
    # target *from* these; attention DP would be a different weight layout and
    # so a different target.
    TEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=False)

    # The engine bounds the target's kernels are compiled for. Routing does not
    # read them, so they are set here rather than left to the defaults:
    #   max_batch_size 32  the CUDA-graph capture ceiling the target was
    #                      validated at. The sparse-MLA decode kernel is
    #                      compiled for the engine's max_batch_size.
    #   max_num_tokens 8192  the routed-expert kernels are warmed for, and
    #                      chunk a step at, 8192 tokens.
    #   max_seq_len 8192   the decode indexer's column count is fixed from it,
    #                      and GSM8K never comes near it.
    BOUNDS = dict(
        max_batch_size=32,
        max_num_tokens=8192,
        max_seq_len=8192,
        cuda_graph_config=CudaGraphConfig(max_batch_size=32, enable_padding=True),
    )

    # The checkpoint carries one MTP layer, so a draft length of 1 resolves to
    # the one-model MTP-Eagle loop, the only draft path the target implements;
    # its specialised rope kernel is built for that two-token width.
    MTP1 = MTPDecodingConfig(max_draft_len=1)

    # The kv-cache fractions are run requirements rather than tuning choices.
    # The MTP module and its separate draft pool take several GiB per rank, and
    # an MTP engine on this checkpoint at the identity leg's 0.7 has been seen
    # to stall rather than finish.
    IDENTITY_KV = KvCacheConfig(free_gpu_memory_fraction=0.7)
    MTP1_KV = KvCacheConfig(free_gpu_memory_fraction=0.55)

    # The acceptance anchor. It names the target and variant, not a test
    # function, because that is what the number is a property of. Its value was
    # measured on stock modeling (modeling_v2="off") under the same
    # speculative_config, so the gate is not read against the target's own number.
    ACCEPTANCE_KEY = "ModelingV2Glm52Nvfp4Sm103Tep4::mtp1"

    # There is deliberately one test. It evaluates the identity config as its
    # first leg and the mtp1 config as its second, and ``task.evaluate``
    # asserts accuracy against the reference on the way past; the mtp1 leg also
    # reads its acceptance length. Separate cases would gate nothing new and
    # would each cost another engine boot of a 78-layer, 4-rank model.

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_gsm8k_identity_vs_mtp1_and_acceptance(self):
        """The identity accuracy gate, the gate on MTP not moving it, and acceptance.

        Turning MTP on must not move the answers.

        Rejection sampling holds the emitted distribution to the target
        model's, so the two scores should differ only by sampling noise.

        Both legs assert accuracy against the registered reference as they
        run -- this test is therefore the identity gate as well as the
        comparison.

        Both measurements are taken **in this one test** on purpose: a delta
        against a score recorded in some earlier session carries that
        session's variance into the judgement. Paired, the variance is common
        to both and cancels.

        The score is the harness default, with no ``scores_filter``. That is
        the protocol the registered GLM-5.2 references were set under and the
        one this target was validated with, so the anchor and the measurement
        read the same metric.

        Acceptance length is the only gate that can see a miscomputed draft
        layer. Rejection sampling makes a wrong draft path *slower*, not wrong:
        every draft is rejected, the text stays correct, and the accuracy gates
        pass. The mtp1 leg therefore records iteration stats and checks its
        acceptance length against ``ACCEPTANCE_KEY``.
        """
        task = GSM8K(self.MODEL_NAME)

        # One dict for both legs: the comparison only means anything if the two
        # engines were built under the same mode. It also asks each engine to
        # check its step contract on its first forward -- the contract is about
        # metadata that engine prepared, so the only place it can be checked
        # honestly is a run that prepared it; a synthesized metadata object
        # would be checking the synthesizer.
        modeling_v2 = dict(modeling_v2="require", env_overrides={MODELING_V2_VALIDATE_ENV: "1"})

        with LLM(
            self.MODEL_PATH,
            kv_cache_config=self.IDENTITY_KV,
            **modeling_v2,
            **self.TEP4,
            **self.BOUNDS,
        ) as llm:
            identity = task.evaluate(llm)

        with LLM(
            self.MODEL_PATH,
            speculative_config=self.MTP1,
            kv_cache_config=self.MTP1_KV,
            enable_iter_perf_stats=True,
            **modeling_v2,
            **self.TEP4,
            **self.BOUNDS,
        ) as llm:
            mtp1 = task.evaluate(llm)
            acceptance_length = compute_acceptance_length(llm)

        delta = mtp1 - identity
        print(f"[modeling_v2] gsm8k identity={identity:.4f} mtp1={mtp1:.4f} delta={delta:+.4f}")
        # 2 sigma at the ~0.6 stderr this benchmark reports at n=1319.
        assert abs(delta) < 1.2, (
            f"MTP moved gsm8k by {delta:+.4f} (identity={identity:.4f}, "
            f"mtp1={mtp1:.4f}); rejection sampling should have held the "
            f"distribution, so this is not sampling noise"
        )
        print(f"[AL] mtp1 acceptance_length = {acceptance_length:.3f}")
        assert_acceptance_length(self.ACCEPTANCE_KEY, acceptance_length)
