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
"""Bucketed partial weight refit (RL-style update_weights) for MXFP8 MoE.

``MXFP8CutlassFusedMoEMethod`` stores its UE8M0 block scales in the CUTLASS
Mxf8f6f4 swizzled layout produced by ``block_scale_interleave``. That transform
is applied *in place* on the destination buffer, so it is a read-modify-write
and NOT an involution: if it runs once per bucket, a second bucket re-swizzles
the bytes an earlier bucket already converted and the scales are silently
scrambled -- the same failure mode ``test_moe_partial_refit.py`` documents for
the CuteDsl BF16 gate/up interleave.

The loader therefore stages raw UE8M0 bytes during ``load_quant_scales`` and
applies the swizzle exactly once in ``process_weights_after_loading``. These
tests pin that contract by asserting bitwise equality against a single-shot
load across every bucket order, including the case that regressed the BF16
path (a w2-only bucket arriving after the w3/w1 bucket).

``MXFP8CuteDslFusedMoEMethod`` (the CUTEDSL backend on Rubin/SM107) layers a
second non-involutive transform on top -- the SwiGLU gate/up interleave of the
FC1 weight and its scales for the fused FC12 kernel -- so the same matrix runs
for both backends, plus a layout check that the CuTe DSL storage is exactly the
interleave of the Cutlass storage (i.e. the interleave was applied once).
"""

from typing import Dict, List

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_FUSED_FC12_AVAILABLE
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe
from tensorrt_llm._torch.moe.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm._torch.moe.fused_moe.quantization import (
    MXFP8CuteDslFusedMoEMethod,
    MXFP8CutlassFusedMoEMethod,
    interleave_linear_and_gate,
)
from tensorrt_llm._torch.utils import unswizzle_sf
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

NUM_EXPERTS = 4
TOP_K = 2
# create_weights asserts hidden_size and intermediate_size_per_partition are
# multiples of BLOCK_SIZE * BLOCK_SCALES_VEC_SIZE (= 128) for the int32 SF packing;
# the CuTe DSL fused FC12 method additionally needs hidden_size % 512 == 0.
HIDDEN_SIZE = 512
INTERMEDIATE_SIZE = 128
BLOCK_SIZE = 32

requires_mxfp8_moe = pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(torch.ops.trtllm, "block_scale_interleave"),
    reason="MXFP8 MoE scale staging requires CUDA + block_scale_interleave",
)


BACKENDS = ["CUTLASS", "CUTEDSL"]


def _skip_if_backend_unavailable(backend: str) -> None:
    if backend == "CUTLASS" and get_sm_version() not in (100, 103):
        # CutlassFusedMoE.can_implement: MXFP8 is SM100/SM103 only (the
        # SM107 fused kernel is a later milestone); on Rubin the CUTEDSL
        # backend serves MXFP8 and the CUTLASS cases below cannot be built.
        pytest.skip("CUTLASS MXFP8 MoE is limited to SM100/SM103 by can_implement")
    if backend == "CUTEDSL" and not (
        get_sm_version() == 107 and IS_CUTLASS_DSL_FUSED_FC12_AVAILABLE
    ):
        pytest.skip(
            "CUTEDSL MXFP8 MoE requires Rubin (SM107) with a CuTe DSL "
            "build that supports the fused FC12 kernel"
        )


def _make_module(backend: str = "CUTLASS"):
    _skip_if_backend_unavailable(backend)
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = NUM_EXPERTS
    pretrained_config.hidden_size = HIDDEN_SIZE
    pretrained_config.moe_intermediate_size = INTERMEDIATE_SIZE
    pretrained_config.intermediate_size = INTERMEDIATE_SIZE
    pretrained_config.torch_dtype = torch.bfloat16
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        moe_backend=backend,
        quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=BLOCK_SIZE),
    )
    module = create_moe(
        routing_method=RenormalizeMoeRoutingMethod(top_k=TOP_K),
        reduce_results=True,
        model_config=model_config,
        weight_loading_mode=MoEWeightLoadingMode.VANILLA,
    )
    module.cuda()
    return module


def _backend(module):
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
    # Mirrors rlhf_utils.WorkerExtension.update_weights: one pre_reload_weights
    # walk, then one partial load per bucket, then a single finalize.
    for mod in module.modules():
        if hasattr(mod, "pre_reload_weights") and not getattr(mod, "_weights_removed", False):
            mod.pre_reload_weights()
    for bucket in buckets:
        module.load_weights([dict(bucket)], allow_partial_loading=True)
    _rlhf_finalize(module)
    torch.cuda.synchronize()


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Bit pattern of a tensor, as uint8.

    The comparisons here are exact-reload checks, so they must be bitwise.
    ``torch.equal`` is the wrong tool for the e4m3 weights: 0x7F/0xFF are NaN
    in float8_e4m3fn and NaN != NaN, so identical buffers would compare unequal
    purely because of the payload bytes.
    """
    return t.view(torch.uint8) if t.dtype.itemsize == 1 else t.contiguous().view(torch.uint8)


def _snapshot(module):
    b = _backend(module)
    return (
        _bits(b.w3_w1_weight.data).clone(),
        _bits(b.w2_weight.data).clone(),
        _bits(b.w3_w1_weight_scale.data).clone(),
        _bits(b.w2_weight_scale.data).clone(),
    )


def _fresh_load(module, weights: Dict[str, torch.Tensor]):
    module.load_weights([dict(weights)])
    module.post_load_weights()
    torch.cuda.synchronize()
    return _snapshot(module)


def _assert_matches(module, expected, label):
    got = _snapshot(module)
    names = ("w3_w1_weight", "w2_weight", "w3_w1_weight_scale", "w2_weight_scale")
    for name, g, e in zip(names, got, expected):
        assert torch.equal(g, e), (
            f"{label}: {name} after bucketed refit differs from single-shot load "
            f"(swizzle applied a wrong number of times)"
        )


def _mxfp8_expert_weights(generator: torch.Generator):
    """VANILLA-layout per-expert MXFP8 weights + UE8M0 block scales.

    w1/w3 are [I, H] (column-parallel), w2 is [H, I] (row-parallel); each scale
    covers one 1x32 block along K, so it drops the last dim by BLOCK_SIZE.
    """
    weights: Dict[str, torch.Tensor] = {}
    for e in range(NUM_EXPERTS):
        for leaf, shape in (
            ("w1", (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
            ("w3", (INTERMEDIATE_SIZE, HIDDEN_SIZE)),
            ("w2", (HIDDEN_SIZE, INTERMEDIATE_SIZE)),
        ):
            w = torch.randint(
                0, 240, shape, generator=generator, device="cuda", dtype=torch.uint8
            ).view(torch.float8_e4m3fn)
            # UE8M0 exponents; keep them in a sane range so the values are
            # representative rather than denormal/inf.
            sf = torch.randint(
                110,
                140,
                (shape[0], shape[1] // BLOCK_SIZE),
                generator=generator,
                device="cuda",
                dtype=torch.uint8,
            )
            weights[f"{e}.{leaf}.weight"] = w
            weights[f"{e}.{leaf}.weight_scale_inv"] = sf
    return weights


def _split(weights, leaves):
    return {k: v for k, v in weights.items() if k.split(".")[1] in leaves}


@requires_mxfp8_moe
@pytest.mark.parametrize("backend", BACKENDS)
def test_mxfp8_moe_quant_method_is_selected(backend):
    """Guard the premise: this module must actually use the MXFP8 method."""
    module = _make_module(backend)
    expected = MXFP8CuteDslFusedMoEMethod if backend == "CUTEDSL" else MXFP8CutlassFusedMoEMethod
    assert type(_backend(module).quant_method) is expected


@requires_mxfp8_moe
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "bucket_order",
    ["w3w1_then_w2", "w2_then_w3w1", "single_bucket", "per_leaf"],
)
def test_mxfp8_bucketed_refit_matches_single_shot(bucket_order, backend):
    """Every bucket order must land bitwise on the single-shot load result.

    ``w3w1_then_w2`` is the regression case: with an eagerly-applied swizzle the
    trailing w2-only bucket re-swizzles the already-converted w3_w1 scales.
    """
    torch.manual_seed(20260821)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weights = _mxfp8_expert_weights(generator)

    expected = _fresh_load(_make_module(backend), weights)

    buckets = {
        "w3w1_then_w2": [_split(weights, {"w1", "w3"}), _split(weights, {"w2"})],
        "w2_then_w3w1": [_split(weights, {"w2"}), _split(weights, {"w1", "w3"})],
        "single_bucket": [weights],
        "per_leaf": [_split(weights, {"w1"}), _split(weights, {"w3"}), _split(weights, {"w2"})],
    }[bucket_order]

    module = _make_module(backend)
    _refit(module, buckets)
    _assert_matches(module, expected, bucket_order)


@requires_mxfp8_moe
@pytest.mark.parametrize("backend", BACKENDS)
def test_mxfp8_repeated_finalize_is_noop(backend):
    """The RLHF finalize walk calls process_weights_after_loading AND
    post_load_weights; draining the pending-slot sets must make the second
    application a no-op rather than a second swizzle."""
    torch.manual_seed(20260821)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weights = _mxfp8_expert_weights(generator)

    expected = _fresh_load(_make_module(backend), weights)

    module = _make_module(backend)
    _refit(module, [_split(weights, {"w1", "w3"}), _split(weights, {"w2"})])
    _assert_matches(module, expected, "after first finalize")

    for _ in range(3):
        _rlhf_finalize(module)
        torch.cuda.synchronize()
    _assert_matches(module, expected, "after repeated finalize")


@requires_mxfp8_moe
def test_mxfp8_pending_slots_drained_after_finalize():
    """White-box check on the mechanism itself: staging arms the pending sets,
    finalize drains them. A non-empty set after finalize means a later bucket
    would re-swizzle."""
    torch.manual_seed(20260821)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weights = _mxfp8_expert_weights(generator)

    module = _make_module()
    backend = _backend(module)
    module.load_weights([_split(weights, {"w1", "w3"})], allow_partial_loading=True)
    assert backend._mxfp8_w3_w1_sf_pending_slots, (
        "staging w3/w1 scales must arm the w3_w1 pending set"
    )
    assert not backend._mxfp8_w2_sf_pending_slots, (
        "a w3/w1-only bucket must NOT arm the w2 pending set"
    )

    module.load_weights([_split(weights, {"w2"})], allow_partial_loading=True)
    assert backend._mxfp8_w2_sf_pending_slots

    _rlhf_finalize(module)
    torch.cuda.synchronize()
    assert not backend._mxfp8_w3_w1_sf_pending_slots
    assert not backend._mxfp8_w2_sf_pending_slots


@requires_mxfp8_moe
@pytest.mark.parametrize("backend", BACKENDS)
def test_mxfp8_two_sequential_refits_match_single_shot(backend):
    """A second full update_weights cycle must land on the same bytes as a
    fresh load of the same tensors -- i.e. refits do not accumulate state."""
    torch.manual_seed(20260821)
    gen_a = torch.Generator(device="cuda").manual_seed(1)
    gen_b = torch.Generator(device="cuda").manual_seed(2)
    weights_a = _mxfp8_expert_weights(gen_a)
    weights_b = _mxfp8_expert_weights(gen_b)

    expected_b = _fresh_load(_make_module(backend), weights_b)

    module = _make_module(backend)
    _refit(module, [_split(weights_a, {"w1", "w3"}), _split(weights_a, {"w2"})])
    _refit(module, [_split(weights_b, {"w1", "w3"}), _split(weights_b, {"w2"})])
    _assert_matches(module, expected_b, "second refit")


@requires_mxfp8_moe
def test_mxfp8_cutedsl_layout_is_single_interleave_of_cutlass_layout():
    """The CuTe DSL storage must be exactly one gate/up interleave (granularity
    64 along the expanded intermediate dim) of the Cutlass storage, for the
    FC1 weight and its block scales alike; w2 is untouched. A double
    interleave or a missing one would break the fused FC12 kernel silently."""
    torch.manual_seed(20260907)
    generator = torch.Generator(device="cuda").manual_seed(4321)
    weights = _mxfp8_expert_weights(generator)

    cutlass = _backend(_make_module("CUTLASS"))
    cutedsl = _backend(_make_module("CUTEDSL"))
    for module in (cutlass, cutedsl):
        module.load_weights([dict(weights)])
        module.post_load_weights()
    torch.cuda.synchronize()

    rows = cutedsl.expand_intermediate_size_per_partition
    k = HIDDEN_SIZE
    group = MXFP8CuteDslFusedMoEMethod.INTERLEAVE_GROUP_SIZE
    for slot in range(cutedsl.w3_w1_weight.shape[0]):
        ref_w = interleave_linear_and_gate(
            cutlass.w3_w1_weight.data[slot].view(torch.uint8), group_size=group, dim=0
        )
        assert torch.equal(cutedsl.w3_w1_weight.data[slot].view(torch.uint8), ref_w), (
            f"slot {slot}: CuTe DSL FC1 weight is not the interleave of the Cutlass layout"
        )

        ref_sf = interleave_linear_and_gate(
            unswizzle_sf(
                cutlass.w3_w1_weight_scale.data[slot].view(torch.uint8), rows, k, BLOCK_SIZE
            ).view(rows, k // BLOCK_SIZE),
            group_size=group,
            dim=0,
        )
        got_sf = unswizzle_sf(
            cutedsl.w3_w1_weight_scale.data[slot].view(torch.uint8), rows, k, BLOCK_SIZE
        ).view(rows, k // BLOCK_SIZE)
        assert torch.equal(got_sf, ref_sf), (
            f"slot {slot}: CuTe DSL FC1 block scales are not the interleave of the Cutlass layout"
        )

    assert torch.equal(_bits(cutedsl.w2_weight.data), _bits(cutlass.w2_weight.data))
    assert torch.equal(_bits(cutedsl.w2_weight_scale.data), _bits(cutlass.w2_weight_scale.data))


@requires_mxfp8_moe
def test_mxfp8_cutedsl_interleave_pending_slots_drained_after_finalize():
    """White-box: staging FC1 weights arms the CuTe DSL interleave set for
    exactly the touched slots; a w2-only bucket must not; finalize drains it."""
    torch.manual_seed(20260907)
    generator = torch.Generator(device="cuda").manual_seed(4321)
    weights = _mxfp8_expert_weights(generator)

    module = _make_module("CUTEDSL")
    backend = _backend(module)
    module.load_weights([_split(weights, {"w2"})], allow_partial_loading=True)
    assert not backend._cute_dsl_mxfp8_w3_w1_interleave_pending, (
        "a w2-only bucket must NOT arm the FC1 interleave set"
    )

    module.load_weights([_split(weights, {"w1", "w3"})], allow_partial_loading=True)
    assert backend._cute_dsl_mxfp8_w3_w1_interleave_pending == set(
        range(backend.w3_w1_weight.shape[0])
    )

    _rlhf_finalize(module)
    torch.cuda.synchronize()
    assert not backend._cute_dsl_mxfp8_w3_w1_interleave_pending
    assert not backend._mxfp8_w3_w1_sf_pending_slots
