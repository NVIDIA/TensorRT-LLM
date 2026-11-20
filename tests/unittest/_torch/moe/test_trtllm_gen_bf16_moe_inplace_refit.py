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
shuffle + BlockMajorK). Non-gated layers convert each expert into its
kernel-layout slot as it arrives; gated layers re-create the layer in the
creation layout and convert it at finalize (the re-creating path).

For a non-gated layer these tests pin that a refit is bitwise the single-shot
load for every bucket order, that it stays in place on every sync (the
kernel-layout storage CUDA graphs captured is kept, and no second copy of the
layer is allocated), that experts a refit does not deliver keep their weights,
and that a missing expert fails a non-partial load. For a gated layer they pin
that a reload takes the re-creating path, keeps the captured storage and still
matches a single-shot load, and that a gated layer in the kernel layout refuses
w1 and w3 in separate calls. A reload before the first finalize loads into the
creation layout.
"""

import pytest
import torch
from _torch.moe.moe_test_utils import begin_moe_refit, finalize_moe_refit
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
# TrtllmTrtllmGenBf16Impl.can_implement needs hidden_size and the per-rank
# intermediate size to be multiples of 128 (_SHAPE_ALIGNMENT); create_moe raises
# otherwise because backend degradation is off.
INTERMEDIATE_SIZE = 1024
# Bytes of one expert's projection ([I, H] or [H, I], bf16).
EXPERT_BYTES = INTERMEDIATE_SIZE * HIDDEN_SIZE * 2

requires_trtllm_gen_bf16_moe = pytest.mark.skipif(
    not torch.cuda.is_available() or getSMVersion() not in (100, 103),
    reason="TRTLLM-Gen BF16 MoE requires SM100 or SM103",
)

Weights = dict[str, torch.Tensor]


def _make_module(gated: bool) -> torch.nn.Module:
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
    # The premise of every test: create_moe wraps a TRTLLM-Gen BF16 leaf.
    assert isinstance(module.backend.quant_method, BF16TRTLLMGenFusedMoEMethod)
    return module


def _loaded_module(gated: bool, weights: Weights) -> torch.nn.Module:
    module = _make_module(gated)
    module.load_weights([dict(weights)])
    module.post_load_weights()
    return module


def _refit(
    module: torch.nn.Module, buckets: list[Weights], storage: tuple[int, int] | None = None
) -> None:
    """The engine's begin walk, one partial load per bucket, then its finalize walk.

    With ``storage``, also assert that the refit stays in place: the
    kernel-layout tensors are still registered after the begin walk and after
    finalize.
    """
    begin_moe_refit(module)
    if storage is not None:
        _assert_kept_in_kernel_layout(module, storage)
    for bucket in buckets:
        module.load_weights([dict(bucket)], allow_partial_loading=True)
    finalize_moe_refit(module)
    if storage is not None:
        _assert_kept_in_kernel_layout(module, storage)


def _expert_weights(gated: bool, seed: int) -> Weights:
    """VANILLA per-expert weights: w1/w3 [I, H], w2 [H, I]. A non-gated layer
    gets the empty w3 the Nemotron-H weight mapper passes."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, device="cuda").to(torch.bfloat16)

    weights: Weights = {}
    for e in range(NUM_EXPERTS):
        w1 = rand(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        weights[f"{e}.w1.weight"] = w1
        weights[f"{e}.w3.weight"] = rand(INTERMEDIATE_SIZE, HIDDEN_SIZE) if gated else w1[:0]
        weights[f"{e}.w2.weight"] = rand(HIDDEN_SIZE, INTERMEDIATE_SIZE)
    return weights


def _pick(weights: Weights, experts: set[int], leaves: set[str]) -> Weights:
    return {
        k: v
        for k, v in weights.items()
        if int(k.split(".")[0]) in experts and k.split(".")[1] in leaves
    }


def _buckets(weights: Weights, order: str) -> list[Weights]:
    everyone = set(range(NUM_EXPERTS))
    low, high = set(range(NUM_EXPERTS // 2)), set(range(NUM_EXPERTS // 2, NUM_EXPERTS))
    return {
        "fc1_then_fc2": [_pick(weights, everyone, {"w1", "w3"}), _pick(weights, everyone, {"w2"})],
        "fc2_then_fc1": [_pick(weights, everyone, {"w2"}), _pick(weights, everyone, {"w1", "w3"})],
        "single_bucket": [weights],
        # fc1 and fc2 of one expert in different buckets, plus an empty bucket.
        "split_experts": [
            _pick(weights, high, {"w1", "w3"}),
            {},
            _pick(weights, low, {"w1", "w3"}) | _pick(weights, high, {"w2"}),
            _pick(weights, low, {"w2"}),
        ],
        # w1 and w3 of one expert in different buckets.
        "split_gate_up": [
            _pick(weights, everyone, {"w1"}),
            _pick(weights, everyone, {"w3"}),
            _pick(weights, everyone, {"w2"}),
        ],
    }[order]


def _snapshot(module: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    backend = module.backend
    return backend.w3_w1_weight.data.clone(), backend.w2_weight.data.clone()


def _storage(module: torch.nn.Module) -> tuple[int, int]:
    backend = module.backend
    return backend.w3_w1_weight.data_ptr(), backend.w2_weight.data_ptr()


def _fresh_load(gated: bool, weights: Weights) -> tuple[torch.Tensor, torch.Tensor]:
    module = _loaded_module(gated, weights)
    torch.cuda.synchronize()
    return _snapshot(module)


def _assert_matches(
    module: torch.nn.Module, expected: tuple[torch.Tensor, torch.Tensor], label: str
) -> None:
    for name, got, want in zip(("w3_w1_weight", "w2_weight"), _snapshot(module), expected):
        assert torch.equal(got, want), f"{label}: {name} after refit differs from a fresh load"


def _assert_kept_in_kernel_layout(module: torch.nn.Module, storage: tuple[int, int]) -> None:
    backend = module.backend
    assert backend.w3_w1_weight.dim() == 4 and backend.w2_weight.dim() == 4, (
        "a non-gated layer must keep its kernel-layout experts during a refit"
    )
    assert _storage(module) == storage, (
        "the kernel-layout storage must survive the refit (CUDA graphs captured it)"
    )


def _refit_peak(
    module: torch.nn.Module, buckets: list[Weights], storage: tuple[int, int] | None = None
) -> int:
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _refit(module, buckets, storage)
    return torch.cuda.max_memory_allocated() - base


@requires_trtllm_gen_bf16_moe
@pytest.mark.parametrize(
    "order", ["fc1_then_fc2", "fc2_then_fc1", "single_bucket", "split_experts", "split_gate_up"]
)
def test_inplace_refit_matches_fresh_load(order: str) -> None:
    initial, update = _expert_weights(False, seed=1), _expert_weights(False, seed=2)
    expected = _fresh_load(False, update)

    module = _loaded_module(False, initial)
    storage = _storage(module)
    begin_moe_refit(module)
    _assert_kept_in_kernel_layout(module, storage)
    for bucket in _buckets(update, order):
        module.load_weights([dict(bucket)], allow_partial_loading=True)
    finalize_moe_refit(module)
    _assert_matches(module, expected, order)
    _assert_kept_in_kernel_layout(module, storage)


@requires_trtllm_gen_bf16_moe
def test_repeated_inplace_refit_stays_in_place() -> None:
    """Every sync of an RL run takes the in-place path, not only the first."""
    module = _loaded_module(False, _expert_weights(False, seed=1))
    storage = _storage(module)
    for seed in (2, 3):
        update = _expert_weights(False, seed=seed)
        peak = _refit_peak(module, _buckets(update, "split_experts"), storage)
        # A few expert-sized temporaries, never a second copy of the layer.
        assert peak <= 4 * EXPERT_BYTES, f"refit seed {seed}: peak {peak} B"
        _assert_matches(module, _fresh_load(False, update), f"refit seed {seed}")


@requires_trtllm_gen_bf16_moe
@pytest.mark.parametrize("delivered", ["half_the_experts", "w2_only"])
def test_undelivered_experts_keep_their_weights(delivered: str) -> None:
    initial, update = _expert_weights(False, seed=1), _expert_weights(False, seed=2)
    everyone, low = set(range(NUM_EXPERTS)), set(range(NUM_EXPERTS // 2))
    if delivered == "half_the_experts":
        bucket = _pick(update, low, {"w1", "w3", "w2"})
        mixed = {k: (update if int(k.split(".")[0]) in low else initial)[k] for k in initial}
    else:
        bucket = _pick(update, everyone, {"w2"})
        mixed = {k: (update if k.split(".")[1] == "w2" else initial)[k] for k in initial}

    module = _loaded_module(False, initial)
    _refit(module, [bucket], _storage(module))
    _assert_matches(module, _fresh_load(False, mixed), delivered)


@requires_trtllm_gen_bf16_moe
def test_untouched_layer_keeps_its_weights() -> None:
    """A layer no bucket touches: only the pre_reload and finalize walks run."""
    initial = _expert_weights(False, seed=1)
    module = _loaded_module(False, initial)
    _refit(module, [], _storage(module))
    _assert_matches(module, _fresh_load(False, initial), "untouched")


@requires_trtllm_gen_bf16_moe
@pytest.mark.parametrize("gated", [False, True], ids=["relu2", "swiglu"])
def test_repeated_finalize_is_a_no_op(gated: bool) -> None:
    """A layer already in the kernel layout is not converted again."""
    module = _loaded_module(gated, _expert_weights(gated, seed=1))
    storage, before = _storage(module), _snapshot(module)
    finalize_moe_refit(module)
    _assert_kept_in_kernel_layout(module, storage)
    _assert_matches(module, before, "second finalize")


@requires_trtllm_gen_bf16_moe
def test_reload_before_the_first_finalize_loads_in_the_creation_layout() -> None:
    """A layer not converted yet has nothing to update in place: the reload
    writes its creation-layout tensors and finalize converts them."""
    update = _expert_weights(False, seed=2)
    module = _make_module(False)
    begin_moe_refit(module)
    assert module.backend.w3_w1_weight.dim() == 3, "nothing to keep before the first finalize"
    module.load_weights([dict(update)])
    finalize_moe_refit(module)
    _assert_matches(module, _fresh_load(False, update), "reload before the first finalize")


@requires_trtllm_gen_bf16_moe
@pytest.mark.parametrize("projection", ["w3_w1", "w2"])
def test_missing_expert_fails_a_non_partial_load(projection: str) -> None:
    module = _loaded_module(False, _expert_weights(False, seed=1))
    begin_moe_refit(module)
    leaves = {"w1", "w3"} if projection == "w3_w1" else {"w2"}
    update = _expert_weights(False, seed=2)
    bucket = {
        k: v
        for k, v in update.items()
        if not (k.split(".")[0] == "0" and k.split(".")[1] in leaves)
    }
    with pytest.raises(ValueError, match="allow_partial_loading is not set"):
        module.load_weights([bucket])


@requires_trtllm_gen_bf16_moe
@pytest.mark.parametrize("order", ["split_experts", "split_gate_up"])
def test_gated_layer_takes_the_re_creating_path(order: str) -> None:
    """w3 and w1 rows of one expert are interleaved, and a bucket may carry only
    one of them, so a gated layer re-creates and converts at finalize."""
    initial, update = _expert_weights(True, seed=1), _expert_weights(True, seed=2)
    module = _loaded_module(True, initial)
    storage = _storage(module)
    begin_moe_refit(module)
    assert module.backend.w3_w1_weight.dim() == 3, "a gated layer must take the re-creating path"
    for bucket in _buckets(update, order):
        module.load_weights([dict(bucket)], allow_partial_loading=True)
    finalize_moe_refit(module)
    _assert_matches(module, _fresh_load(True, update), f"gated {order}")
    assert module.backend.w3_w1_weight.dim() == 4 and _storage(module) == storage, (
        "the re-creating path must copy back into the storage CUDA graphs captured"
    )


@requires_trtllm_gen_bf16_moe
def test_gated_layer_in_kernel_layout_refuses_a_split_gate_up_load() -> None:
    """Without pre_reload_weights a gated layer stays in the kernel layout, where
    one half of an expert cannot be written alone."""
    module = _loaded_module(True, _expert_weights(True, seed=1))
    w1_only = _buckets(_expert_weights(True, seed=2), "split_gate_up")[0]
    with pytest.raises(ValueError, match="needs w1 and w3 of an expert in one load"):
        module.load_weights([dict(w1_only)], allow_partial_loading=True)
