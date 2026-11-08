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
"""Whole-model gates for the deepseek-v3 modeling_v2 targets.

One file per model family, beside the other accuracy suites rather than inside
test_llm_api_pytorch.py: these gate a parallel implementation, and reading them
next to the built-in model's tests would invite treating one as a variant of
the other.

Every test here passes ``modeling_v2="require"`` to ``LLM``. Under ``"auto"`` a
configuration that missed a target's criteria -- or fell outside its bounds --
would quietly fall back to the built-in implementation, pass, and report the
built-in's numbers as the target's, which is the one failure this whole
system exists to prevent. The one exception is the stock leg of the acceptance
gate, which asks for ``"off"`` on purpose.

These targets are multi-rank, and the mode has to hold on every rank rather
than only on the one running the test body. The switch is an LLM API argument,
so it reaches the ranks the way the rest of the arguments do.
"""

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
from ._env import scoped_env_overrides

# The targets assert their own SM at construction: certification is per GPU
# architecture, and a receipt from another one says nothing here.
skip_not_sm10x = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="modeling_v2 targets in this batch are certified on sm_100 and sm_103 only",
)

# The filter the modeling_v2 anchors were measured on. Unset, the evaluator
# averages every metric GSM8K reports, which means the mean of strict-match and
# flexible-extract -- two numbers measuring different things on a checkpoint
# that does not answer purely in the strict "#### N" form.
_SCORES_FILTER = {"scores_filter": "exact_match,flexible-extract"}


def _layout(layout_id, world_size, moe_tp_size, moe_ep_size, attention_dp):
    """One parallel layout as a pytest param, with the device count it needs.

    A GB200/GB300 tray holds four GPUs and a B200/B300 node eight; larger
    layouts span nodes.
    """
    need = (
        pytest.mark.skip_less_device(world_size)
        if world_size <= 4
        else pytest.mark.skip_less_mpi_world_size(world_size)
    )
    return pytest.param(
        dict(
            tensor_parallel_size=world_size,
            moe_tensor_parallel_size=moe_tp_size,
            moe_expert_parallel_size=moe_ep_size,
            enable_attention_dp=attention_dp,
        ),
        id=layout_id,
        marks=need,
    )


# Every layout the default target's routing admits.
_LAYOUTS = [
    _layout("tp2", 2, 2, 1, False),
    _layout("tp2_ep2", 2, 1, 2, False),
    _layout("dep2", 2, 1, 2, True),
    _layout("tp4", 4, 4, 1, False),
    _layout("tp4_ep4", 4, 1, 4, False),
    _layout("tp4_moetp2_ep2", 4, 2, 2, False),
    _layout("dep4", 4, 1, 4, True),
    _layout("dep4_moetp2_ep2", 4, 2, 2, True),
    _layout("dep4_moetp4", 4, 4, 1, True),
    _layout("tp8", 8, 8, 1, False),
    _layout("tp8_ep8", 8, 1, 8, False),
    _layout("tp8_moetp4_ep2", 8, 4, 2, False),
    _layout("tp8_moetp2_ep4", 8, 2, 4, False),
    _layout("dep8", 8, 1, 8, True),
    _layout("dep8_moetp8", 8, 8, 1, True),
    _layout("dep8_moetp2_ep4", 8, 2, 4, True),
    _layout("dep16", 16, 1, 16, True),
]

# The layouts routing admits MTP at: each residual-norm path (attention TP in
# one tray and across trays, attention DP) and each expert split.
_MTP_LAYOUT_IDS = ("dep4", "tp4", "tp4_ep4", "dep4_moetp4", "tp8_moetp2_ep4", "dep8")
_MTP_LAYOUTS = [p for i in _MTP_LAYOUT_IDS for p in _LAYOUTS if p.id == i]


class TestModelingV2DeepseekR10528Nvfp4Sm10xDefault(LlmapiAccuracyTestHarness):
    """deepseek-r1-0528-nvfp4 / sm_10x / default, identity and the mtp3 variant.

    Identity runs at every layout its routing admits, the mtp3 variant at the
    layouts MTP is admitted at.

    Each layout cuts the weights differently (heads, MLP and expert
    intermediates, expert windows) or crosses ranks differently (fused
    allreduce in or across trays, or allgather + reduce-scatter under attention
    DP), and a wrong cut still produces fluent text, so every layout was
    measured before routing admitted it. CI runs a representative subset (the
    l0_gb300 test lists): each residual-norm path and each expert split. The
    cut of every layout is pinned on the CPU by
    test_modeling_v2_deepseek_default_shards.py.
    """

    MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1-0528-FP4"

    # The kv-cache fraction is a boot requirement of the variant rather than a
    # tuning choice: the drafting forward's post-pool transient does not fit
    # what the default 0.9 leaves.
    MTP3 = MTPDecodingConfig(max_draft_len=3)
    MTP3_KV = KvCacheConfig(free_gpu_memory_fraction=0.75)

    # One anchor shared by both legs of the acceptance gate below, at every
    # layout. It names the target and variant, not a test function or a
    # layout, because that is what the number is a property of: the draft
    # layer computes the same function at every layout.
    ACCEPTANCE_KEY = "ModelingV2DeepseekR10528Nvfp4Sm10xDefault::mtp3"

    @skip_not_sm10x
    @pytest.mark.parametrize("parallel", _LAYOUTS)
    def test_gsm8k(self, mocker, parallel):
        mocker.patch.dict(GSM8K.EVALUATE_KWARGS, _SCORES_FILTER)

        # `require` so a layout the routing does not admit fails instead of
        # quietly measuring the built-in model. It also asks the engine to
        # check its step contract on its first forward -- the contract is
        # about metadata that engine prepared, so the only place it can be
        # checked honestly is a run that prepared it; a synthesized metadata
        # object would be checking the synthesizer.
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(
                self.MODEL_PATH,
                modeling_v2="require",
                env_overrides=rank_env,
                **parallel,
            ) as llm,
        ):
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_not_sm10x
    @pytest.mark.parametrize("parallel", _MTP_LAYOUTS)
    def test_gsm8k_identity_vs_mtp3(self, mocker, parallel):
        """The gate on MTP not moving the answers.

        Rejection sampling holds the emitted distribution to the target
        model's, so the two scores should differ only by sampling noise.

        Both legs assert accuracy against the registered reference as they
        run -- this test is therefore an identity gate at its layout as well
        as the comparison.

        Both measurements are taken **in this one test** on purpose. The same
        identity forward measured 94.7688 and 95.0720 on consecutive days --
        0.30 apart, on bit-identical code -- so a delta against a score
        recorded in some earlier session carries that session's variance into
        the judgement. Paired, the variance is common to both and cancels.
        """
        mocker.patch.dict(GSM8K.EVALUATE_KWARGS, _SCORES_FILTER)
        task = GSM8K(self.MODEL_NAME)

        # One dict for both legs: the comparison only means anything if the two
        # engines were built under the same mode.
        modeling_v2 = dict(modeling_v2="require", env_overrides={MODELING_V2_VALIDATE_ENV: "1"})

        with (
            scoped_env_overrides(modeling_v2["env_overrides"]),
            LLM(self.MODEL_PATH, **modeling_v2, **parallel) as llm,
        ):
            identity = task.evaluate(llm)

        with (
            scoped_env_overrides(modeling_v2["env_overrides"]),
            LLM(
                self.MODEL_PATH,
                speculative_config=self.MTP3,
                kv_cache_config=self.MTP3_KV,
                **modeling_v2,
                **parallel,
            ) as llm,
        ):
            mtp3 = task.evaluate(llm)

        delta = mtp3 - identity
        print(f"[modeling_v2] gsm8k identity={identity:.4f} mtp3={mtp3:.4f} delta={delta:+.4f}")
        # 2 sigma at the ~0.6 stderr this benchmark reports at n=1319.
        assert abs(delta) < 1.2, (
            f"MTP moved gsm8k by {delta:+.4f} (identity={identity:.4f}, "
            f"mtp3={mtp3:.4f}); rejection sampling should have held the "
            f"distribution, so this is not sampling noise"
        )

    @skip_not_sm10x
    @pytest.mark.parametrize("parallel", _MTP_LAYOUTS)
    @pytest.mark.parametrize("mode", ["require", "off"], ids=["modeling_v2", "stock"])
    def test_mtp3_acceptance(self, mode, parallel, mocker):
        """The only gate that can see a miscomputed draft layer.

        Rejection sampling makes a wrong draft path *slower*, not wrong: every
        draft is rejected, the text stays correct, and the boot and accuracy
        gates both pass. Acceptance length is the sole detector.

        Two independent cases per layout rather than one that compares them
        in-session. They share ``ACCEPTANCE_KEY``, so both are read against the
        same recorded minimum, which is what makes the pair informative:

          modeling_v2 fails, stock passes -> the draft path regressed
          both fail                     -> the anchor is stale; re-derive it
                                           rather than blaming the target

        The anchor is populated from the **stock** leg. Populating it from the
        target's own number would make the gate self-referential.
        """
        # Stock cannot boot this checkpoint under attention DP + EP with MTP
        # otherwise: the MoE communication factory lands on DeepEPLowLatency,
        # whose dispatch takes only NVFP4 uint8 hidden states, and the MTP layer
        # is bf16 because modelopt excludes model.layers.61* from quantization.
        # Disabling DeepEP moves stock off it; the modeling_v2 target builds its
        # own allgather + reduce-scatter and never consults the factory. Set at
        # every layout, where it is inert under attention TP. It travels as
        # ``env_overrides`` because it is read on the ranks.
        rank_env = {MODELING_V2_VALIDATE_ENV: "1"}
        if mode == "off":
            rank_env["TRTLLM_CAN_USE_DEEP_EP"] = "0"

        mocker.patch.dict(GSM8K.EVALUATE_KWARGS, _SCORES_FILTER)

        with (
            scoped_env_overrides(rank_env),
            LLM(
                self.MODEL_PATH,
                speculative_config=self.MTP3,
                kv_cache_config=self.MTP3_KV,
                cuda_graph_config=CudaGraphConfig(),
                enable_iter_perf_stats=True,
                modeling_v2=mode,
                env_overrides=rank_env,
                **parallel,
            ) as llm,
        ):
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            acceptance_length = compute_acceptance_length(llm)
            print(f"[AL] {mode} acceptance_length = {acceptance_length:.3f}")
            assert_acceptance_length(self.ACCEPTANCE_KEY, acceptance_length)

    @skip_not_sm10x
    @pytest.mark.parametrize(
        "parallel", [p for i in ("tp4_ep4", "dep4") for p in _LAYOUTS if p.id == i]
    )
    def test_gsm8k_chunked_prefill(self, mocker, parallel):
        """The prefill target's cached-context flavor, under attention TP and DP.

        With chunked prefill every chunk after the first attends to KV it did
        not compute this step, so the forward takes the MLA path that appends
        to the paged cache and reads the whole context back
        (`mla_rope_append_paged_kv_assign_q`, `load_paged_kv_cache_for_mla`)
        instead of the fresh-context one the other cases run. The harness's
        GSM8K prompts are lm-eval's 5-shot chain-of-thought prompts, several
        hundred tokens each, so `max_num_tokens=256` splits every one of them
        into several chunks.
        """
        mocker.patch.dict(GSM8K.EVALUATE_KWARGS, _SCORES_FILTER)

        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(
                self.MODEL_PATH,
                enable_chunked_prefill=True,
                max_num_tokens=256,
                modeling_v2="require",
                env_overrides=rank_env,
                **parallel,
            ) as llm,
        ):
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
