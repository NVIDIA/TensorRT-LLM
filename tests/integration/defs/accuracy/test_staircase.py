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
"""Whole-model gates for the staircase targets.

Separate file rather than entries in test_llm_api_pytorch.py, for the same
reason the targets are separate codebases: these gate a parallel
implementation, and reading them next to the built-in model's tests would
invite treating one as a variant of the other.

Every test here needs ``TRTLLM_STAIRCASE=require``. Under ``"auto"`` a
configuration that missed a target's criteria would quietly fall back to the
built-in implementation, pass, and report the built-in's numbers as the
target's -- which is the one failure this whole system exists to prevent. The
one exception is the stock leg of the acceptance gate, which asks for
``"off"`` on purpose.

The switch is an environment variable, and worker ranks read it as it stood
when they started. At world size 1 that is this process. Above it the ranks
are already running by the time a test body executes, so a multi-rank case
cannot choose its own mode -- it can only assert that the environment it was
given is the one it needs, which is what ``_require_mode`` does.
"""

import os

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch.staircase import STAIRCASE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MTPDecodingConfig

from ..conftest import llm_models_root
from .accuracy_core import (
    GSM8K,
    LlmapiAccuracyTestHarness,
    assert_acceptance_length,
    compute_acceptance_length,
)

# The targets assert their own SM at construction: certification is per GPU
# architecture, and a receipt from another one says nothing here.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="staircase targets in this batch are certified on sm_103 only"
)


def _require_mode(expected: str) -> None:
    """Skip unless the ranks were started with the mode this case needs.

    Not a failure: which mode a multi-rank job runs under is a property of how
    it was launched, so a case that wants the other one has nothing to say. It
    must not silently measure the wrong system either, which is what reading
    the variable here rules out.
    """
    actual = os.environ.get(STAIRCASE_ENV, "off")
    if actual != expected:
        pytest.skip(f"{STAIRCASE_ENV}={actual!r}, this case needs {expected!r}")


class _StaircaseGSM8K(GSM8K):
    """GSM8K reading the filter the staircase anchors were measured on.

    Unset, the evaluator averages every metric the task reports, which for
    GSM8K means the mean of ``strict-match`` and ``flexible-extract``. Those
    measure different things here: neither of these checkpoints answers purely
    in the strict ``#### N`` form, so the average is a number no reference was
    ever taken at -- gpt-oss scores ~90 flexible, ~25 strict, and the mean of
    56 reads as a catastrophic failure of a model that is answering correctly.
    """

    EVALUATE_KWARGS = {"scores_filter": "exact_match,flexible-extract"}


class _GSM8KWithRoomToReason(_StaircaseGSM8K):
    """The above, with the output budget a reasoning model needs.

    The stock 256 tokens truncate this checkpoint mid-chain-of-thought, before
    it ever reaches an answer, and the gate then reads as an assembly defect
    rather than as the protocol being wrong for the model.
    """

    MAX_OUTPUT_LEN = 8192


class TestStaircaseGptOss120bSm103Tp1(LlmapiAccuracyTestHarness):
    """gpt-oss-120b / sm_103 / tp1."""

    # The registry key upstream uses for this checkpoint; it carries the
    # W4A8_MXFP4_MXFP8 entry the engine resolves from its quantization_config.
    MODEL_NAME = "GPT-OSS/120B-MXFP4"
    MODEL_PATH = f"{llm_models_root()}/gpt_oss/gpt-oss-120b"

    # This checkpoint is gated as a reasoning model: its answer never arrives
    # in the strict "#### N" form, so the protocol applies the chat template
    # and gives the model room to reason. Matches the protocol recorded in
    # _torch/staircase/references/accuracy.yaml.
    extra_evaluator_kwargs = {
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
    }

    @skip_not_sm103
    def test_gsm8k(self):
        _require_mode("require")
        with LLM(self.MODEL_PATH) as llm:
            task = _GSM8KWithRoomToReason(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=self.extra_evaluator_kwargs)


class TestStaircaseDeepseekR10528Nvfp4Sm103Dep4(LlmapiAccuracyTestHarness):
    """deepseek-r1-0528-nvfp4 / sm_103 / dep4, identity and the mtp3 variant."""

    MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1-0528-FP4"

    # The parallel topology the path's dep4 segment declares. Routing derives
    # the target *from* these, and the target then asserts every one of them
    # against the mapping the engine actually built.
    DEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=True)

    # configs/mtp3.yaml. The kv-cache fraction is a boot requirement of the
    # variant rather than a tuning choice: the drafting forward's post-pool
    # transient does not fit what the default 0.9 leaves.
    MTP3 = MTPDecodingConfig(max_draft_len=3)
    MTP3_KV = KvCacheConfig(free_gpu_memory_fraction=0.75)

    # One anchor shared by both legs of the acceptance gate below. It names the
    # target and variant, not a test function, because that is what the number
    # is a property of.
    ACCEPTANCE_KEY = "StaircaseDeepseekR10528Nvfp4Sm103Dep4::mtp3"

    # There is deliberately no standalone identity gsm8k case. The paired test
    # below evaluates the identity config as its first leg, and
    # ``task.evaluate`` asserts accuracy against the reference on the way past,
    # so a separate one would gate nothing new and would cost a fifth engine
    # boot of a 61-layer, 4-rank model.

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_gsm8k_identity_vs_mtp3(self):
        """The identity accuracy gate, and the gate on MTP not moving it.

        Turning MTP on must not move the answers.

        Rejection sampling holds the emitted distribution to the target
        model's, so the two scores should differ only by sampling noise.

        Both legs assert accuracy against the registered reference as they
        run -- this test is therefore the identity gate as well as the
        comparison.

        Both measurements are taken **in this one test** on purpose. The same
        identity forward measured 94.7688 and 95.0720 on consecutive days --
        0.30 apart, on bit-identical code -- so a delta against a score
        recorded in some earlier session carries that session's variance into
        the judgement. Paired, the variance is common to both and cancels.
        """
        task = _StaircaseGSM8K(self.MODEL_NAME)

        _require_mode("require")
        with LLM(self.MODEL_PATH, **self.DEP4) as llm:
            identity = task.evaluate(llm)

        with LLM(
            self.MODEL_PATH,
            speculative_config=self.MTP3,
            kv_cache_config=self.MTP3_KV,
            **self.DEP4,
        ) as llm:
            mtp3 = task.evaluate(llm)

        delta = mtp3 - identity
        print(f"[staircase] gsm8k identity={identity:.4f} mtp3={mtp3:.4f} delta={delta:+.4f}")
        # 2 sigma at the ~0.6 stderr this benchmark reports at n=1319.
        assert abs(delta) < 1.2, (
            f"MTP moved gsm8k by {delta:+.4f} (identity={identity:.4f}, "
            f"mtp3={mtp3:.4f}); rejection sampling should have held the "
            f"distribution, so this is not sampling noise"
        )

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize("mode", ["require", "off"], ids=["staircase", "stock"])
    def test_mtp3_acceptance(self, mode):
        """The only gate that can see a miscomputed draft layer.

        Rejection sampling makes a wrong draft path *slower*, not wrong: every
        draft is rejected, the text stays correct, and the boot and accuracy
        gates both pass. Acceptance length is the sole detector.

        Two independent cases rather than one that compares them in-session.
        They share ``ACCEPTANCE_KEY``, so both are read against the same
        recorded minimum, which is what makes the pair informative:

          staircase fails, stock passes -> the draft path regressed
          both fail                     -> the anchor is stale; re-derive it
                                           rather than blaming the target

        The anchor is populated from the **stock** leg. Populating it from the
        target's own number would make the gate self-referential, which is the
        same rule references/accuracy.yaml states for accuracy anchors.
        """
        _require_mode(mode)
        if mode == "off":
            # Stock cannot boot this checkpoint at dep4 with MTP otherwise:
            # under attention DP + EP the MoE communication factory lands on
            # DeepEPLowLatency, whose dispatch takes only NVFP4 uint8 hidden
            # states, and the MTP layer is bf16 because modelopt excludes
            # model.layers.61* from quantization. Disabling DeepEP lands on
            # AllGatherReduceScatter -- which is the strategy the staircase
            # target implements by hand, so it makes the two comparable rather
            # than less so. Set in the launching environment, like the switch.
            assert os.environ.get("TRTLLM_CAN_USE_DEEP_EP") == "0", (
                "the stock leg needs TRTLLM_CAN_USE_DEEP_EP=0 exported; without "
                "it stock cannot boot this checkpoint at dep4 with MTP"
            )

        with LLM(
            self.MODEL_PATH,
            speculative_config=self.MTP3,
            kv_cache_config=self.MTP3_KV,
            cuda_graph_config=CudaGraphConfig(),
            enable_iter_perf_stats=True,
            **self.DEP4,
        ) as llm:
            task = _StaircaseGSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            acceptance_length = compute_acceptance_length(llm)
            print(f"[AL] {mode} acceptance_length = {acceptance_length:.3f}")
            assert_acceptance_length(self.ACCEPTANCE_KEY, acceptance_length)


def test_staircase_off_is_the_default(monkeypatch):
    """Unset means off, on the code path the engine actually takes.

    Cheap, GPU-free, and the thing most worth never regressing: everything in
    this file rests on staircase being opt-in.
    """
    from tensorrt_llm._torch.staircase import StaircaseMode

    monkeypatch.delenv(STAIRCASE_ENV, raising=False)
    assert StaircaseMode.from_env() is StaircaseMode.OFF
    assert os.environ.get("STAIRCASE_TARGET") is None, (
        "STAIRCASE_TARGET was retired with the move in-tree; it named a "
        "target, where TRTLLM_STAIRCASE names only a mode and lets routing "
        "pick the target from the configuration"
    )
