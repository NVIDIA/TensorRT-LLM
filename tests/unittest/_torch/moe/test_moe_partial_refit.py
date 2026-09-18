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
"""Regression tests for bucketed partial weight refit (RL-style update_weights).

The RLHF refit path streams weights bucket by bucket via
``load_weights(allow_partial_loading=True)`` and finalizes once at the end
(``process_weights_after_loading`` + ``post_load_weights``). Quant methods
whose FC1 layout transform is not an involution (e.g. the CuteDsl BF16
gate/up interleave) must therefore stage plain bytes during loading and
apply the transform exactly once at finalize; applying it inside
``load_expert_w3_w1_weight`` re-transforms the already-transformed buffer
whenever a later bucket touches the module (e.g. a down_proj-only bucket
after the gate_up bucket) and scrambles the weights.

These tests replicate the bucket-order scenarios at small scale and assert
bitwise weight equality against a single-shot load.
"""

from typing import Dict, List

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig
from utils.util import getSMVersion

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe
from tensorrt_llm._torch.moe.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm._torch.moe.fused_moe.quantization import BF16CuteDslFusedMoEMethod

NUM_EXPERTS = 8
TOP_K = 2
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 128  # 2*I = 256 is divisible by the interleave group of 2*32
DTYPE = torch.bfloat16

cutedsl_param = pytest.param(
    "CUTEDSL",
    marks=pytest.mark.skipif(
        getSMVersion() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
        reason="BF16 CuteDsl MoE quant method requires SM107 + internal CUTLASS DSL",
    ),
)


def _make_module(moe_backend: str, weight_loading_mode: MoEWeightLoadingMode):
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = NUM_EXPERTS
    pretrained_config.hidden_size = HIDDEN_SIZE
    pretrained_config.moe_intermediate_size = INTERMEDIATE_SIZE
    pretrained_config.intermediate_size = INTERMEDIATE_SIZE
    pretrained_config.torch_dtype = DTYPE
    model_config = ModelConfig(pretrained_config=pretrained_config, moe_backend=moe_backend)
    module = create_moe(
        routing_method=RenormalizeMoeRoutingMethod(top_k=TOP_K),
        reduce_results=True,
        model_config=model_config,
        weight_loading_mode=weight_loading_mode,
    )
    module.cuda()
    return module


def _module_backend(module):
    # create_moe may return a ConfigurableMoE wrapper owning a backend module.
    return getattr(module, "backend", module)


def _rlhf_finalize(module) -> None:
    # Mirrors rlhf_utils.WorkerExtension.finalize_weight_update: the walk
    # invokes BOTH process_weights_after_loading and post_load_weights on
    # every eligible module.
    for mod in module.modules():
        if hasattr(mod, "process_weights_after_loading") and not getattr(
            mod, "_weights_removed", False
        ):
            mod.process_weights_after_loading()
        if hasattr(mod, "post_load_weights") and not getattr(mod, "_weights_removed", False):
            mod.post_load_weights()


def _refit(module, buckets: List[Dict[str, torch.Tensor]]) -> None:
    # Mirrors rlhf_utils.WorkerExtension.update_weights: one
    # pre_reload_weights walk, then one partial load per bucket, then a
    # single finalize.
    for mod in module.modules():
        if hasattr(mod, "pre_reload_weights") and not getattr(mod, "_weights_removed", False):
            mod.pre_reload_weights()
    for bucket in buckets:
        module.load_weights([dict(bucket)], allow_partial_loading=True)
    _rlhf_finalize(module)
    torch.cuda.synchronize()


def _fresh_load(module, weights: Dict[str, torch.Tensor]):
    module.load_weights([dict(weights)])
    module.post_load_weights()
    torch.cuda.synchronize()
    backend = _module_backend(module)
    return (backend.w3_w1_weight.data.clone(), backend.w2_weight.data.clone())


def _assert_refit_matches_fresh(module, buckets, w3_w1_fresh, w2_fresh):
    _refit(module, buckets)
    backend = _module_backend(module)
    assert torch.equal(backend.w3_w1_weight.data, w3_w1_fresh), (
        "w3_w1_weight after bucketed refit differs from single-shot load "
        "(FC1 transform applied a wrong number of times)"
    )
    assert torch.equal(backend.w2_weight.data, w2_fresh), (
        "w2_weight after bucketed refit differs from single-shot load"
    )
    # Repeated finalize must be a no-op (the RLHF finalize walk itself calls
    # both finalize hooks; an extra walk must not re-transform).
    _rlhf_finalize(module)
    torch.cuda.synchronize()
    assert torch.equal(backend.w3_w1_weight.data, w3_w1_fresh), (
        "repeated finalize re-transformed w3_w1_weight"
    )


def _stacked_expert_weights(generator: torch.Generator):
    # Trainer-side HF layout: gate_up [E, 2I, H] (rows 0:I = w1/gate,
    # I:2I = w3/up), down [E, H, I]; the Qwen3.5 mapper hands the module
    # transposed stacked tensors (see qwen3_5_weight_mapper.py).
    gate_up = (
        torch.randn(
            NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, generator=generator, device="cuda"
        )
        * 0.02
    ).to(DTYPE)
    down = (
        torch.randn(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, generator=generator, device="cuda")
        * 0.02
    ).to(DTYPE)
    mapped_gate_up = gate_up.transpose(-1, -2).contiguous()
    mapped_down = down.transpose(-1, -2).contiguous()
    return mapped_gate_up, mapped_down


@pytest.mark.parametrize("moe_backend", [cutedsl_param, "CUTLASS"])
@pytest.mark.parametrize("bucket_order", ["gateup_then_down", "down_then_gateup", "single_bucket"])
def test_fused_gate_up_bucketed_refit_matches_single_shot(moe_backend, bucket_order):
    """FUSED_GATE_UP_PROJ refit: every bucket order must land bitwise on the
    single-shot load result (the gateup_then_down order is the observed RL
    corruption case: a down_proj-only bucket after the gate_up bucket)."""
    torch.manual_seed(20260815)
    generator = torch.Generator(device="cuda").manual_seed(777)
    mapped_gate_up, mapped_down = _stacked_expert_weights(generator)

    module = _make_module(moe_backend, MoEWeightLoadingMode.FUSED_GATE_UP_PROJ)
    if moe_backend == "CUTEDSL":
        assert isinstance(_module_backend(module).quant_method, BF16CuteDslFusedMoEMethod), (
            "expected the BF16 CuteDsl quant method; the regression "
            "target is its deferred FC1 interleave"
        )

    w3_w1_fresh, w2_fresh = _fresh_load(
        module,
        {
            "gate_up_proj": mapped_gate_up,
            "down_proj": mapped_down,
        },
    )

    buckets = {
        "gateup_then_down": [
            {"gate_up_proj": mapped_gate_up},
            {"down_proj": mapped_down},
        ],
        "down_then_gateup": [
            {"down_proj": mapped_down},
            {"gate_up_proj": mapped_gate_up},
        ],
        "single_bucket": [
            {"gate_up_proj": mapped_gate_up, "down_proj": mapped_down},
        ],
    }[bucket_order]
    _assert_refit_matches_fresh(module, buckets, w3_w1_fresh, w2_fresh)


@pytest.mark.parametrize("moe_backend", [cutedsl_param, "CUTLASS"])
@pytest.mark.parametrize("bucket_order", ["w1_w3_w2", "w3_w2_w1"])
def test_vanilla_split_bucketed_refit_matches_single_shot(moe_backend, bucket_order):
    """VANILLA refit where w1 (gate), w3 (up) and w2 arrive in different
    buckets: FC1 halves are staged plain across buckets and the layout
    transform must still be applied exactly once at finalize."""
    torch.manual_seed(20260815)
    generator = torch.Generator(device="cuda").manual_seed(777)
    w1 = (
        torch.randn(NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE, generator=generator, device="cuda")
        * 0.02
    ).to(DTYPE)
    w3 = torch.randn_like(w1) * 0.02
    w2 = (
        torch.randn(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, generator=generator, device="cuda")
        * 0.02
    ).to(DTYPE)

    module = _make_module(moe_backend, MoEWeightLoadingMode.VANILLA)
    if moe_backend == "CUTEDSL":
        assert isinstance(_module_backend(module).quant_method, BF16CuteDslFusedMoEMethod)

    full = {}
    for expert_id in range(NUM_EXPERTS):
        full[f"{expert_id}.w1.weight"] = w1[expert_id]
        full[f"{expert_id}.w3.weight"] = w3[expert_id]
        full[f"{expert_id}.w2.weight"] = w2[expert_id]
    w3_w1_fresh, w2_fresh = _fresh_load(module, full)

    bucket_w1 = {f"{e}.w1.weight": w1[e] for e in range(NUM_EXPERTS)}
    bucket_w3 = {f"{e}.w3.weight": w3[e] for e in range(NUM_EXPERTS)}
    bucket_w2 = {f"{e}.w2.weight": w2[e] for e in range(NUM_EXPERTS)}
    buckets = {
        "w1_w3_w2": [bucket_w1, bucket_w3, bucket_w2],
        "w3_w2_w1": [bucket_w3, bucket_w2, bucket_w1],
    }[bucket_order]
    _assert_refit_matches_fresh(module, buckets, w3_w1_fresh, w2_fresh)
