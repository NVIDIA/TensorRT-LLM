# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Beam NVFP4 target's device-resident expert provider (``routed_experts``), generated data.

CPU: the combination equals the source model's (the catalog ``beam_dispatch_combine`` op and
its fp64-product reference) bit for bit, and each way of dropping one of its roundings or its
order is caught; the grouped GEMM's operands are views of the fused banks, rebuilt when a bank
is replaced and refused when the FC1 halves are no longer adjacent.

CUDA (sm_103), on generated checkpoint-scale experts (hidden 5120, intermediate 640 per rank, 16
experts): a batch's routes and partials equal each token's alone, bit for bit, and combine
exactly as the source does; where every gate output saturates SiLU (silu(x) = x exactly in
fp32) the provider equals the host-enumerated reference provider (``expert.nvfp4_swiglu_expert``
under ``dispatch_combine``) bit for bit per route and after the combination, with empty
experts, zero weights, one expert selected by every token and few experts; slot order and
padding rows leave the real rows' results unchanged; the call has no host synchronization,
captures in a CUDA graph and replays current inputs; consecutive calls do not share output
storage.

The operator's fused SiLU and the reference's may differ in the last fp32 bit elsewhere. The
real-checkpoint replays (layers 3, 27, 51 on four ranks, against the entry provider and the
approved reference under the approved MoE own-input rule) are not part of this file.

Invocation::

    env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE -u NVIDIA_TF32_OVERRIDE python -m pytest \\
        tests/unittest/_torch/modeling_v2/models/beam/full_model_nvfp4__sm_103__tp4/\\
test_routed_experts.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.beam_dispatch_combine import (
    beam_dispatch_combine_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (  # noqa: E501
    component as nvfp4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (  # noqa: E501
    routed_experts,
)

HIDDEN, INTER, TOP_K, SCALING = 5120, 640, 8, 2.5

cpu_only = pytest.mark.cpu_only


# ------------------------------------------------------------------ CPU: the combination


def _token_hidden(tokens: int, width: int) -> torch.Tensor:
    """Hidden rows whose first element is the token index (exact in bf16 below 256)."""
    h = torch.zeros(tokens, width, dtype=torch.bfloat16)
    h[:, 0] = torch.arange(tokens, dtype=torch.bfloat16)
    return h


def _routes_provider(routes: torch.Tensor, ids: torch.Tensor):
    """An expert provider returning, for each routed row, that token's route to the expert."""

    def provider(expert_id: int, rows: torch.Tensor) -> torch.Tensor:
        tokens = rows[:, 0].long()
        slots = (ids[tokens] == expert_id).int().argmax(dim=-1)
        return routes[tokens, slots]

    return provider


def _routing(tokens: int, experts: int, seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(experts, generator=g)[:TOP_K] for _ in range(tokens)])
    weights = torch.rand(tokens, TOP_K, generator=g)
    weights = weights / weights.sum(-1, keepdim=True)
    return ids, weights


def _cancelling_routes(tokens: int, width: int, seed: int) -> torch.Tensor:
    """bf16 routes spanning many binades, so fp32 sums depend on order and rounding."""
    g = torch.Generator().manual_seed(seed)
    magnitude = torch.pow(2.0, torch.randint(-12, 12, (tokens, TOP_K, width), generator=g).float())
    return (torch.randn(tokens, TOP_K, width, generator=g) * magnitude).to(torch.bfloat16)


@cpu_only
@pytest.mark.parametrize("tokens,experts,seed", [(1, 8, 0), (17, 32, 1), (129, 256, 2)])
def test_combination_is_the_source_combination(tokens, experts, seed):
    width = 64
    ids, weights = _routing(tokens, experts, seed)
    weights[0, 3] = 0.0  # a zero weight still contributes its (zero) route
    routes = _cancelling_routes(tokens, width, seed)
    got = routed_experts.ordered_combine(routes, ids, weights, SCALING)
    h = _token_hidden(tokens, width)
    provider = _routes_provider(routes, ids)
    assert torch.equal(got, beam_dispatch_combine_op(h, ids, weights, provider, SCALING))
    assert torch.equal(got, beam_dispatch_combine_op.reference(h, ids, weights, provider, SCALING))


def _variant(routes, ids, weights, kind):
    """The combination with one required step changed, for the negative controls."""
    tokens, top_k, width = routes.shape
    order = ids.argsort(-1) if kind != "slot_order" else torch.arange(top_k).expand(tokens, -1)
    acc_dtype = torch.bfloat16 if kind == "bf16_accumulator" else torch.float32
    acc = torch.zeros(tokens, width, dtype=acc_dtype)
    for j in range(top_k):
        route = routes[torch.arange(tokens), order[:, j]].float()
        w = weights.gather(1, order[:, j : j + 1])
        if kind == "scale_in_weights":
            weighted = (route * (w * SCALING)).to(torch.bfloat16).float()
        elif kind == "fp32_routes":
            weighted = route * w
        else:
            weighted = (route * w).to(torch.bfloat16).float()
        acc = (acc.float() + weighted).to(acc_dtype)
    scale = 1.0 if kind == "scale_in_weights" else SCALING
    return (acc.float() * scale).to(torch.bfloat16)


@cpu_only
@pytest.mark.parametrize(
    "kind", ["slot_order", "fp32_routes", "scale_in_weights", "bf16_accumulator"]
)
def test_combination_check_catches_a_dropped_step(kind):
    tokens, width = 64, 64
    ids, weights = _routing(tokens, 256, 7)
    routes = _cancelling_routes(tokens, width, 7)
    # Token 0 cancels exactly in ascending expert order and not in slot order.
    ids[0] = torch.tensor([30, 10, 20, 40, 50, 60, 70, 80])
    weights[0] = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    routes[0, :3, 0] = torch.tensor([1.0, 2.0**24, -(2.0**24)], dtype=torch.bfloat16)
    reference = beam_dispatch_combine_op.reference(
        _token_hidden(tokens, width), ids, weights, _routes_provider(routes, ids), SCALING
    )
    assert torch.equal(routed_experts.ordered_combine(routes, ids, weights, SCALING), reference)
    assert not torch.equal(_variant(routes, ids, weights, kind), reference), kind


# ------------------------------------------------------------------ CPU: operand views


def _fused_banks(experts: int, rows: int, half: int, scale_bytes: int) -> dict:
    """Banks laid out as the component allocates them: up and gate halves of one storage."""
    codes = torch.randint(0, 256, (experts, 2 * rows, half), dtype=torch.uint8)
    scales = torch.randint(0x28, 0x48, (experts, 2 * scale_bytes), dtype=torch.uint8)
    bank = {}
    for part, kind in enumerate(nvfp4.FC1_KINDS):
        bank[kind] = {
            "weight": codes[:, part * rows : (part + 1) * rows],
            "weight_scale": scales[:, part * scale_bytes : (part + 1) * scale_bytes],
        }
    bank["down"] = {
        "weight": torch.randint(0, 256, (experts, 4 * rows, rows // 4), dtype=torch.uint8),
        "weight_scale": torch.randint(0x28, 0x48, (experts, rows * rows // 4), dtype=torch.uint8),
    }
    for kind in bank:
        for field in ("input_scale", "weight_scale_2", "g", "alpha"):
            bank[kind][field] = torch.rand(experts, dtype=torch.float32) + 0.5
    return {"codes": codes, "scales": scales, "bank": bank}


def _operand_state(banks: dict, layer: int = 3) -> SimpleNamespace:
    state = SimpleNamespace(_operands={})
    state.expert_bank = lambda requested: banks["bank"] if requested == layer else None
    return state


@cpu_only
def test_operands_are_views_of_the_fused_banks():
    experts, rows, half, scale_bytes = 4, 128, 256, 128 * 32
    banks = _fused_banks(experts, rows, half, scale_bytes)
    state = _operand_state(banks)
    ops = nvfp4.BeamCheckpointLayoutNvfp4Sm103Tp4.expert_operands(state, 3)
    bank = banks["bank"]
    assert ops.fc1_weight.dtype == torch.int64 and ops.fc1_weight.shape == (experts, 2 * rows, 32)
    assert ops.fc1_weight.data_ptr() == banks["codes"].data_ptr()
    assert ops.fc1_block_scales.dtype == torch.int32
    assert ops.fc1_block_scales.shape == (experts, 2 * rows, scale_bytes // rows // 4)
    assert ops.fc1_block_scales.data_ptr() == banks["scales"].data_ptr()
    # The [up; gate] order the grouped GEMM reads.
    assert torch.equal(ops.fc1_weight.view(torch.uint8)[:, :rows], bank["up"]["weight"])
    assert torch.equal(ops.fc1_weight.view(torch.uint8)[:, rows:], bank["gate"]["weight"])
    assert ops.fc1_alpha is bank["gate"]["alpha"]
    assert ops.fc1_input_scale_inv.dim() == 0
    assert ops.fc1_input_scale_inv.data_ptr() == bank["gate"]["g"].data_ptr()
    assert ops.fc2_weight.data_ptr() == bank["down"]["weight"].data_ptr()
    assert ops.fc2_weight.dtype == torch.int64 and ops.fc2_block_scales.dtype == torch.int32
    assert ops.fc2_input_scale_inv is bank["down"]["g"] and ops.fc2_alpha is bank["down"]["alpha"]
    # Cached while the parameter objects stay; rebuilt from a replaced (still adjacent) bank.
    assert nvfp4.BeamCheckpointLayoutNvfp4Sm103Tp4.expert_operands(state, 3) is ops
    bank["down"]["alpha"] = bank["down"]["alpha"].clone()
    rebuilt = nvfp4.BeamCheckpointLayoutNvfp4Sm103Tp4.expert_operands(state, 3)
    assert rebuilt is not ops and rebuilt.fc2_alpha is bank["down"]["alpha"]


@cpu_only
@pytest.mark.parametrize("field", ["weight", "weight_scale"])
def test_operands_refuse_fc1_halves_that_are_not_adjacent(field):
    banks = _fused_banks(4, 128, 256, 128 * 32)
    banks["bank"]["up"][field] = banks["bank"]["up"][field].clone()
    with pytest.raises(RuntimeError, match="two halves of one FC1 storage"):
        nvfp4.BeamCheckpointLayoutNvfp4Sm103Tp4.expert_operands(_operand_state(banks), 3)


# ------------------------------------------------------------------ CUDA: generated experts


def _sm103() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3)


needs_sm103 = pytest.mark.skipif(not _sm103(), reason="the NVFP4 grouped GEMM runs on sm_103")

EXPERTS = 16


def _swizzled(linear: torch.Tensor) -> torch.Tensor:
    """Per-expert 128x4-swizzled bytes of linear block scales ``[E, rows, cols]``."""
    return torch.stack(
        [torch.ops.trtllm.block_scale_interleave(sf).reshape(-1) for sf in linear.unbind(0)]
    )


#: From here up, silu(x) == x in fp32 for every formulation (exp(-x) < 2**-43, so 1 + exp(-x)
#: and tanh(x / 2) round to 1): the operator's activation and the reference's agree exactly.
SILU_EXACT_FROM = 30.0


def _make_layer(seed: int, saturated: bool) -> SimpleNamespace:
    """One generated sparse layer at checkpoint scale: banks, operands and scalars.

    ``saturated``: gate codes without sign and a larger gate/up ``weight_scale_2``, so that with
    non-negative inputs every gate output is at least ``SILU_EXACT_FROM``; the FC2 input scale
    keeps the larger activations' block scales inside the E4M3 range.
    """
    g = torch.Generator().manual_seed(seed)
    dev = torch.device("cuda")

    def codes(*shape, unsigned=False):
        c = torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)
        if unsigned:
            c &= 0x77  # both E2M1 sign bits clear
        return c.to(dev)

    def linear_sf(*shape):
        sf = torch.randint(0x28, 0x48, shape, dtype=torch.uint8, generator=g)
        sf[:, 0, 0] = 0  # one all-zero block scale per expert
        return sf.to(dev)

    up_sf, gate_sf = (
        linear_sf(EXPERTS, INTER, HIDDEN // 16),
        linear_sf(EXPERTS, INTER, HIDDEN // 16),
    )
    fc1_codes = torch.cat(
        [
            codes(EXPERTS, INTER, HIDDEN // 2),
            codes(EXPERTS, INTER, HIDDEN // 2, unsigned=saturated),
        ],
        1,
    )
    fc1_scales = torch.cat([_swizzled(up_sf), _swizzled(gate_sf)], 1)
    down_codes = codes(EXPERTS, HIDDEN, INTER // 2)
    down_scales = _swizzled(linear_sf(EXPERTS, HIDDEN, INTER // 16))
    one = torch.ones(EXPERTS, dtype=torch.float32)
    s1 = one * 0.0123
    s2 = one * (0.5 if saturated else 0.0625)
    w2_fc1 = (torch.rand(EXPERTS, generator=g) * 3e-3 + (1e-2 if saturated else 1e-3)).float()
    w2_fc2 = (torch.rand(EXPERTS, generator=g) * 1e-3 + 1e-3).float()
    nvfp4.check_expert_scales(
        3,
        {
            "gate": {"input_scale": s1, "weight_scale_2": w2_fc1},
            "up": {"input_scale": s1, "weight_scale_2": w2_fc1},
            "down": {"input_scale": s2, "weight_scale_2": w2_fc2},
        },
    )
    fc1 = {"g": (one / s1).to(dev), "alpha": (s1 * w2_fc1).to(dev)}
    fc1.update(input_scale=s1.to(dev), weight_scale_2=w2_fc1.to(dev))
    bank = {
        "up": dict(fc1, weight=fc1_codes[:, :INTER], weight_scale=fc1_scales[:, : INTER * 320]),
        "gate": dict(fc1, weight=fc1_codes[:, INTER:], weight_scale=fc1_scales[:, INTER * 320 :]),
        "down": {
            "weight": down_codes,
            "weight_scale": down_scales,
            "g": (one / s2).to(dev),
            "alpha": (s2 * w2_fc2).to(dev),
            "input_scale": s2.to(dev),
            "weight_scale_2": w2_fc2.to(dev),
        },
    }
    state = SimpleNamespace(_operands={}, expert_bank=lambda layer: bank)
    operands = nvfp4.BeamCheckpointLayoutNvfp4Sm103Tp4.expert_operands(state, 3)
    return SimpleNamespace(bank=bank, operands=operands)


@pytest.fixture(scope="module")
def generated_layer():
    if not _sm103():
        pytest.skip("the NVFP4 grouped GEMM runs on sm_103")
    return _make_layer(11, saturated=False)


@pytest.fixture(scope="module")
def saturated_layer():
    if not _sm103():
        pytest.skip("the NVFP4 grouped GEMM runs on sm_103")
    return _make_layer(12, saturated=True)


def _reference(layer, h, ids, weights):
    """The host-enumerated reference provider: per-route outputs, combined partial, gate outputs."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (  # noqa: E501
        dispatch_combine,
        expert,
    )

    routes = torch.empty(h.shape[0], TOP_K, HIDDEN, dtype=torch.bfloat16, device=h.device)
    calls = []
    for e in ids.unique().tolist():
        tokens, slots = (ids == e).nonzero(as_tuple=True)
        routes[tokens, slots] = expert.nvfp4_swiglu_expert(
            h[tokens], layer.bank, e, True, record=calls
        )

    def provider(expert_id, rows):
        return expert.nvfp4_swiglu_expert(rows, layer.bank, expert_id, True)

    local = dispatch_combine.beam_dispatch_combine(h, ids, weights, provider)
    return routes, local, torch.cat([call["gate"].reshape(-1) for call in calls])


def _case(tokens: int, pattern: str, seed: int, device="cuda", non_negative=False) -> tuple:
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(tokens, HIDDEN, generator=g) * 0.7
    h = (h.abs() + 0.05 if non_negative else h).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(EXPERTS, generator=g)[:TOP_K] for _ in range(tokens)])
    weights = torch.rand(tokens, TOP_K, generator=g)
    if pattern == "concentrated":  # every token selects expert 0, in different slots
        for t in range(tokens):
            if not bool((ids[t] == 0).any()):
                ids[t, t % TOP_K] = 0
    elif pattern == "few_experts":  # experts 8..15 selected by no token
        ids = torch.stack([torch.randperm(8, generator=g) for _ in range(tokens)])
    elif pattern == "zero_weights":
        weights[:, ::3] = 0.0
    weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-20)
    return h.to(device), ids.to(device), weights.float().to(device)


PATTERNS = ["random", "concentrated", "few_experts", "zero_weights"]


@needs_sm103
@pytest.mark.parametrize("tokens", [1, 32, 129])
@pytest.mark.parametrize("pattern", PATTERNS)
def test_batched_routes_equal_each_token_alone_and_combine_exactly(
    generated_layer, tokens, pattern
):
    h, ids, weights = _case(tokens, pattern, seed=tokens * 7 + len(pattern))
    record = {}
    local = routed_experts.nvfp4_routed_experts(
        h, ids, weights, generated_layer.operands, tune_max_rows=8 * 1024, record=record
    )
    # Each token alone through the provider: the batch's expansion and dispatch neither mix
    # tokens nor lose a route, and other rows do not change a row's result.
    for t in range(tokens):
        alone = {}
        partial = routed_experts.nvfp4_routed_experts(
            h[t : t + 1],
            ids[t : t + 1],
            weights[t : t + 1],
            generated_layer.operands,
            tune_max_rows=8 * 1024,
            record=alone,
        )
        assert torch.equal(alone["routes"][0], record["routes"][t]), t
        assert torch.equal(partial[0], local[t]), t
    # The combination of the provider's own routes is the source's, exactly.
    provider_routes = _routes_provider(record["routes"].cpu(), ids.cpu())
    exact = beam_dispatch_combine_op.reference(
        _token_hidden(tokens, HIDDEN), ids.cpu(), weights.cpu(), provider_routes, SCALING
    )
    assert torch.equal(local.cpu(), exact)


@needs_sm103
@pytest.mark.parametrize("tokens", [1, 32, 129])
@pytest.mark.parametrize("pattern", PATTERNS)
def test_provider_equals_the_reference_provider_where_silu_is_exact(
    saturated_layer, tokens, pattern
):
    """Every gate output at least ``SILU_EXACT_FROM``: both activations compute silu(x) = x.

    The operator's fused activation and the reference's ``swiglu_fp32`` then agree exactly, so
    any difference is in the operands, the projections, the rounding points or the activation
    quantization. Outside that range the two SiLU evaluations may differ in the last fp32 bit;
    the real-checkpoint replays compare that path against the entry provider.
    """
    h, ids, weights = _case(tokens, pattern, seed=tokens * 7 + len(pattern), non_negative=True)
    record = {}
    local = routed_experts.nvfp4_routed_experts(
        h, ids, weights, saturated_layer.operands, tune_max_rows=8 * 1024, record=record
    )
    ref_routes, ref_local, gates = _reference(saturated_layer, h, ids, weights)
    assert float(gates.float().min()) >= SILU_EXACT_FROM, float(gates.float().min())
    assert torch.equal(record["routes"], ref_routes)
    assert torch.equal(local, ref_local)


@needs_sm103
def test_padding_and_slot_order_leave_real_rows_unchanged(generated_layer):
    h, ids, weights = _case(24, "random", seed=3)
    base = routed_experts.nvfp4_routed_experts(
        h, ids, weights, generated_layer.operands, tune_max_rows=8 * 1024
    )
    # Executor padding: extra rows of a dummy token appended after the real ones.
    pad_h, pad_ids, pad_w = _case(8, "random", seed=4)
    padded = routed_experts.nvfp4_routed_experts(
        torch.cat([h, torch.zeros_like(pad_h)]),
        torch.cat([ids, pad_ids]),
        torch.cat([weights, pad_w]),
        generated_layer.operands,
        tune_max_rows=8 * 1024,
    )
    assert torch.equal(padded[:24], base)
    # The router's slot order is not the combination's order.
    perm = torch.stack([torch.randperm(TOP_K) for _ in range(24)]).cuda()
    permuted = routed_experts.nvfp4_routed_experts(
        h,
        ids.gather(1, perm),
        weights.gather(1, perm),
        generated_layer.operands,
        tune_max_rows=8 * 1024,
    )
    assert torch.equal(permuted, base)


@needs_sm103
def test_provider_reads_nothing_back_captures_and_replays_current_inputs(generated_layer):
    tokens = 32
    h, ids, weights = _case(tokens, "random", seed=5)
    static = {"h": h.clone(), "ids": ids.clone(), "w": weights.clone()}
    out = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")

    def step():
        out.copy_(
            routed_experts.nvfp4_routed_experts(
                static["h"],
                static["ids"],
                static["w"],
                generated_layer.operands,
                tune_max_rows=8 * 1024,
            )
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
        torch.cuda.set_sync_debug_mode("error")
        try:
            step()  # warm: no synchronizing call anywhere in the provider
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for seed, pattern in ((6, "random"), (7, "concentrated"), (8, "zero_weights")):
        h2, ids2, w2 = _case(tokens, pattern, seed=seed)
        static["h"].copy_(h2)
        static["ids"].copy_(ids2)
        static["w"].copy_(w2)
        graph.replay()
        eager = routed_experts.nvfp4_routed_experts(
            h2, ids2, w2, generated_layer.operands, tune_max_rows=8 * 1024
        )
        torch.cuda.synchronize()
        assert torch.equal(out, eager), pattern


@needs_sm103
def test_consecutive_calls_do_not_share_output_storage(generated_layer):
    h, ids, weights = _case(16, "random", seed=9)
    first = routed_experts.nvfp4_routed_experts(
        h, ids, weights, generated_layer.operands, tune_max_rows=8 * 1024
    )
    kept = first.clone()
    h2, ids2, w2 = _case(16, "concentrated", seed=10)
    second = routed_experts.nvfp4_routed_experts(
        h2, ids2, w2, generated_layer.operands, tune_max_rows=8 * 1024
    )
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first, kept)
