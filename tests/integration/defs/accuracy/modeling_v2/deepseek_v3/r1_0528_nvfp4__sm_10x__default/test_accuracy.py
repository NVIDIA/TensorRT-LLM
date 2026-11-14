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
"""Whole-model gates for the deepseek-v3 modeling_v2 target r1_0528_nvfp4__sm_10x__default.

Every test here passes ``modeling_v2="require"`` to ``LLM``. Under ``"auto"`` a
configuration that missed a target's criteria -- or fell outside its bounds --
would quietly fall back to the built-in implementation, pass, and report the
built-in's numbers as the target's, which is the one failure this whole
system exists to prevent. The one exception is the stock leg of the acceptance
gate, which asks for ``"off"`` on purpose.

This target is multi-rank, and the mode has to hold on every rank rather
than only on the one running the test body. The switch is an LLM API argument,
so it reaches the ranks the way the rest of the arguments do.
"""

import random

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig, MTPDecodingConfig
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig

from .....conftest import llm_models_root
from ....accuracy_core import (
    GSM8K,
    LlmapiAccuracyTestHarness,
    assert_acceptance_length,
    compute_acceptance_length,
)
from ..._env import scoped_env_overrides

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


# The layouts the target runs EPLB at: attention DP over ranks holding whole experts.
_EPLB_LAYOUTS = [p for i in ("dep4", "dep8") for p in _LAYOUTS if p.id == i]


def _eplb_placement(layer: int, experts: int, slots: int, ep_size: int) -> list:
    """A static EPLB placement unlike the built-in default, drawn per layer.

    The experts shuffled by layer, each rank holding its share once; then each rank's
    spare slots hold experts again: first one it already holds, then experts of the
    ranks after it. A slot loaded with the wrong expert, or an id routed to a slot that
    does not hold its expert, still produces fluent text, so only a placement this far
    from the default shows up in the score.
    """
    order = list(range(experts))
    random.Random(layer).shuffle(order)
    per_rank, spare = experts // ep_size, (slots - experts) // ep_size
    out = []
    for rank in range(ep_size):
        own = order[rank * per_rank : (rank + 1) * per_rank]
        extra = [order[(rank * per_rank + per_rank + k) % experts] for k in range(spare)]
        extra[0] = own[0]
        out += own + extra
    return out


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

    @skip_not_sm10x
    @pytest.mark.parametrize("mtp", [False, True], ids=["no_spec", "mtp3"])
    @pytest.mark.parametrize("parallel", _EPLB_LAYOUTS)
    def test_gsm8k_static_eplb(self, mocker, parallel, mtp):
        """Static EPLB at a placement drawn per layer, the MTP layer included.

        288 slots hold the 256 experts of every routed layer. Every layer registers
        with the engine's load balancer, loads its experts by slot and routes the
        router's ids to slots, so this is the whole EPLB path of the target against
        the same GSM8K reference as the identity cases.
        """
        mocker.patch.dict(GSM8K.EVALUATE_KWARGS, _SCORES_FILTER)
        cfg_layers, first_moe, num_slots = 61, 3, 288
        ep_size = parallel["moe_expert_parallel_size"]
        assignments = {
            i: _eplb_placement(i, 256, num_slots, ep_size)
            for i in range(first_moe, cfg_layers + int(mtp))
        }
        load_balancer = MoeLoadBalancerConfig(
            num_slots=num_slots, initial_global_assignments=assignments
        )
        spec = dict(speculative_config=self.MTP3, kv_cache_config=self.MTP3_KV) if mtp else {}

        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(
                self.MODEL_PATH,
                moe_config=MoeConfig(load_balancer=load_balancer),
                modeling_v2="require",
                env_overrides=rank_env,
                **spec,
                **parallel,
            ) as llm,
        ):
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_not_sm10x
    @pytest.mark.parametrize("mtp", [False, True], ids=["no_spec", "mtp3"])
    @pytest.mark.parametrize("parallel", _EPLB_LAYOUTS)
    def test_gsm8k_online_eplb(self, mocker, parallel, mtp):
        """Online EPLB from the built-in default placement, the MTP layer included.

        Two layers are rebalanced per forward. Every routed layer hands the engine's load
        balancer host copies of its share of the experts and binds its slot weights to it,
        every forward records which experts its tokens chose, and between forwards the
        balancer copies experts into slots and reroutes to them. An expert copied wrong, a
        statistic that stalls the balancer or a view left on the storage the balancer no
        longer writes shows up here, against the same GSM8K reference as the identity cases.
        """
        mocker.patch.dict(GSM8K.EVALUATE_KWARGS, _SCORES_FILTER)
        load_balancer = MoeLoadBalancerConfig(num_slots=288, layer_updates_per_iter=2)
        spec = dict(speculative_config=self.MTP3, kv_cache_config=self.MTP3_KV) if mtp else {}

        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(
                self.MODEL_PATH,
                moe_config=MoeConfig(load_balancer=load_balancer),
                modeling_v2="require",
                env_overrides=rank_env,
                **spec,
                **parallel,
            ) as llm,
        ):
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
