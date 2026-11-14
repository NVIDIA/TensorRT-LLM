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
"""Bucketed RL weight refit for the TRTLLM-Gen BF16 MoE.

``BF16TRTLLMGenFusedMoEMethod`` keeps the experts in the kernel layout (row
shuffle + BlockMajorK). A refit used to re-create every expert of the layer in
the checkpoint layout before converting it again, holding the routed weights
twice. For a non-gated activation each expert is now converted into its slot
as it arrives. These tests pin that the result is bitwise the single-shot load
for every bucket order, that the kernel-layout storage is kept (CUDA graphs
captured it), that no second copy of the layer is allocated, and that gated
layers and the TLLM_TRTLLM_GEN_BF16_INPLACE_REFIT=0 opt-out keep the original
path.
"""

from typing import Dict, List

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig
from utils.util import getSMVersion

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod, SimpleActivation
from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe
from tensorrt_llm._torch.moe.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm._torch.moe.fused_moe.quantization import BF16TRTLLMGenFusedMoEMethod
from tensorrt_llm._torch.utils import ActivationType

NUM_EXPERTS = 16
TOP_K = 2
HIDDEN_SIZE = 1024
# The FlashInfer bf16 kernels need the intermediate size aligned to 128.
INTERMEDIATE_SIZE = 1024
OPT_OUT_ENV = "TLLM_TRTLLM_GEN_BF16_INPLACE_REFIT"

requires_trtllm_gen_bf16_moe = pytest.mark.skipif(
    not torch.cuda.is_available() or getSMVersion() not in (100, 103),
    reason="TRTLLM-Gen BF16 MoE requires an SM100-family GPU",
)


def _make_module(gated: bool):
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = NUM_EXPERTS
    pretrained_config.hidden_size = HIDDEN_SIZE
    pretrained_config.moe_intermediate_size = INTERMEDIATE_SIZE
    pretrained_config.intermediate_size = INTERMEDIATE_SIZE
    pretrained_config.torch_dtype = torch.bfloat16
    model_config = ModelConfig(pretrained_config=pretrained_config, moe_backend="TRTLLM")
    kwargs = {} if gated else {"activation": SimpleActivation(kind=ActivationType.Relu2)}
    module = create_moe(
        routing_method=RenormalizeMoeRoutingMethod(top_k=TOP_K),
        reduce_results=True,
        model_config=model_config,
        weight_loading_mode=MoEWeightLoadingMode.VANILLA,
        allow_backend_degradation=False,
        **kwargs,
    )
    module.cuda()
    return module


def _backend(module):
    # create_moe may return a ConfigurableMoE wrapper owning a backend module.
    return getattr(module, "backend", module)


def _pre_reload(module) -> None:
    # Mirrors rlhf_utils.WorkerExtension.begin_weight_update.
    for mod in module.modules():
        if hasattr(mod, "pre_reload_weights") and not getattr(mod, "_weights_removed", False):
            mod.pre_reload_weights()


def _finalize(module) -> None:
    # Mirrors rlhf_utils.WorkerExtension.finalize_weight_update.
    for mod in module.modules():
        if getattr(mod, "_weights_removed", False):
            continue
        if hasattr(mod, "process_weights_after_loading"):
            mod.process_weights_after_loading()
        if hasattr(mod, "post_load_weights"):
            mod.post_load_weights()
    torch.cuda.synchronize()


def _refit(module, buckets: List[Dict[str, torch.Tensor]]) -> None:
    # One pre_reload walk, one partial load per bucket, then one finalize walk.
    _pre_reload(module)
    for bucket in buckets:
        module.load_weights([dict(bucket)], allow_partial_loading=True)
    _finalize(module)


def _expert_weights(gated: bool, seed: int) -> Dict[str, torch.Tensor]:
    """VANILLA per-expert weights: w1/w3 [I, H], w2 [H, I]. A non-gated layer
    gets the empty w3 the Nemotron-H weight mapper passes."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=generator, device="cuda").to(torch.bfloat16)

    weights: Dict[str, torch.Tensor] = {}
    for e in range(NUM_EXPERTS):
        w1 = rand(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        weights[f"{e}.w1.weight"] = w1
        weights[f"{e}.w3.weight"] = rand(INTERMEDIATE_SIZE, HIDDEN_SIZE) if gated else w1[:0]
        weights[f"{e}.w2.weight"] = rand(HIDDEN_SIZE, INTERMEDIATE_SIZE)
    return weights


def _buckets(weights, order: str):
    def pick(experts, leaves):
        return {
            k: v
            for k, v in weights.items()
            if int(k.split(".")[0]) in experts and k.split(".")[1] in leaves
        }

    everyone = set(range(NUM_EXPERTS))
    low, high = set(range(NUM_EXPERTS // 2)), set(range(NUM_EXPERTS // 2, NUM_EXPERTS))
    return {
        "fc1_then_fc2": [pick(everyone, {"w1", "w3"}), pick(everyone, {"w2"})],
        "fc2_then_fc1": [pick(everyone, {"w2"}), pick(everyone, {"w1", "w3"})],
        "single_bucket": [weights],
        # fc1 and fc2 of one expert in different buckets, plus an empty bucket.
        "split_experts": [
            pick(high, {"w1", "w3"}),
            {},
            pick(low, {"w1", "w3"}) | pick(high, {"w2"}),
            pick(low, {"w2"}),
        ],
    }[order]


def _snapshot(module):
    b = _backend(module)
    return b.w3_w1_weight.data.clone(), b.w2_weight.data.clone()


def _fresh_load(gated: bool, weights):
    module = _make_module(gated)
    module.load_weights([dict(weights)])
    module.post_load_weights()
    torch.cuda.synchronize()
    return _snapshot(module)


def _assert_matches(module, expected, label):
    for name, got, want in zip(("w3_w1_weight", "w2_weight"), _snapshot(module), expected):
        assert torch.equal(got, want), f"{label}: {name} after refit differs from a fresh load"


def _layer_bytes(weights) -> int:
    return sum(t.numel() * t.element_size() for t in weights.values())


def _refit_peak(module, buckets) -> int:
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _refit(module, buckets)
    return torch.cuda.max_memory_allocated() - base


@requires_trtllm_gen_bf16_moe
@pytest.mark.parametrize("gated", [False, True], ids=["relu2", "swiglu"])
def test_quant_method_is_trtllm_gen_bf16(gated):
    """Guard the premise: the module must use the TRTLLM-Gen BF16 method."""
    assert isinstance(_backend(_make_module(gated)).quant_method, BF16TRTLLMGenFusedMoEMethod)


@requires_trtllm_gen_bf16_moe
@pytest.mark.parametrize(
    "order", ["fc1_then_fc2", "fc2_then_fc1", "single_bucket", "split_experts"]
)
def test_inplace_refit_matches_fresh_load(order):
    initial, update = _expert_weights(False, seed=1), _expert_weights(False, seed=2)
    expected = _fresh_load(False, update)

    module = _make_module(False)
    module.load_weights([dict(initial)])
    module.post_load_weights()
    backend = _backend(module)
    storage = (backend.w3_w1_weight.data_ptr(), backend.w2_weight.data_ptr())

    _pre_reload(module)
    assert backend._trtllm_gen_inplace_refit, "a non-gated layer must refit in place"
    for bucket in _buckets(update, order):
        module.load_weights([dict(bucket)], allow_partial_loading=True)
    _finalize(module)
    _assert_matches(module, expected, order)
    assert not backend._trtllm_gen_inplace_refit
    assert (backend.w3_w1_weight.data_ptr(), backend.w2_weight.data_ptr()) == storage, (
        "the kernel-layout storage must survive the refit (CUDA graphs captured it)"
    )


@requires_trtllm_gen_bf16_moe
def test_inplace_refit_does_not_restage_the_layer(monkeypatch):
    initial, update = _expert_weights(False, seed=1), _expert_weights(False, seed=2)
    layer_bytes = _layer_bytes(update)

    module = _make_module(False)
    module.load_weights([dict(initial)])
    module.post_load_weights()
    peak = _refit_peak(module, _buckets(update, "split_experts"))
    assert peak < layer_bytes // 4, f"in-place refit peak {peak} B vs layer {layer_bytes} B"

    monkeypatch.setenv(OPT_OUT_ENV, "0")
    module = _make_module(False)
    module.load_weights([dict(initial)])
    module.post_load_weights()
    peak = _refit_peak(module, _buckets(update, "split_experts"))
    assert peak >= layer_bytes, "the opt-out must take the original re-creating path"
    _assert_matches(module, _fresh_load(False, update), "opt-out")


@requires_trtllm_gen_bf16_moe
def test_repeated_inplace_refit():
    module = _make_module(False)
    module.load_weights([dict(_expert_weights(False, seed=1))])
    module.post_load_weights()
    for seed in (2, 3):
        update = _expert_weights(False, seed=seed)
        _refit(module, _buckets(update, "split_experts"))
        _assert_matches(module, _fresh_load(False, update), f"refit seed {seed}")


@requires_trtllm_gen_bf16_moe
def test_gated_layer_keeps_original_path():
    """w1 and w3 of one expert are interleaved, and a bucket may carry only one
    of them, so a gated layer must still re-create and convert at finalize."""
    initial, update = _expert_weights(True, seed=1), _expert_weights(True, seed=2)
    module = _make_module(True)
    module.load_weights([dict(initial)])
    module.post_load_weights()
    backend = _backend(module)
    _pre_reload(module)
    assert not backend._trtllm_gen_inplace_refit
    for bucket in _buckets(update, "split_experts"):
        module.load_weights([dict(bucket)], allow_partial_loading=True)
    _finalize(module)
    _assert_matches(module, _fresh_load(True, update), "gated")
