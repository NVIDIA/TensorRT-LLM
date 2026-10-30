# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the Beam MoE dispatch and weighted combination of block 3: bf16 / sm_103 / tp1.

Configuration under test, fixed in ``_dispatch_combine_cases.py``: hidden 5120, 256
experts, eight per token, FP32 mixing weights, routed scaling factor 2.5; one sm_103 GPU;
eager; controlled affine experts instead of expert matrices; seeds {17, 41} x tokens
{1, 17, 129}, and focused fixtures.

What is compared. Every call goes through the public ``beam_dispatch_combine``. Its output
must equal, bit for bit, the plain-torch golden of the source combination in
``_dispatch_combine_cases.py``: the steps are exact fp32 products, bf16 roundings and fp32
additions in a fixed order, so any correct implementation reproduces them exactly. The
expert calls themselves -- which experts, in which order, with which rows -- are checked
against the routing table.

Invocation::

    python -m pytest tests/unittest/_torch/modeling_v2/models/beam/moe_dispatch_combine_bf16__sm_103__tp1

Dependencies: a TensorRT-LLM build for sm_103. No model files.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from torch.overrides import TorchFunctionMode, resolve_name

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.beam_dispatch_combine import (  # noqa: E501
    beam_dispatch_combine_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (  # noqa: E501
    dispatch_combine,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.dispatch_combine import (  # noqa: E501
    beam_dispatch_combine,
)

__extra_import_path__ = ["."]
import _dispatch_combine_cases as cases  # noqa: E402

assert torch.cuda.is_available(), "the Beam MoE dispatch/combine tests require a CUDA device"

BF16, FP32, I64 = torch.bfloat16, torch.float32, torch.int64


@pytest.fixture(scope="module")
def experts():
    return cases.affine_experts()


def call(x, ids, w, experts):
    """One public call on fresh copies of the experts' log; returns (output, calls)."""
    provider = experts.fresh()
    out = beam_dispatch_combine(x, ids, w, provider)
    return out, provider.calls


def check_calls(calls, x, ids):
    """Exactly the selected experts, ascending, once each, each with its tokens' rows in
    ascending token order: every one of the T*8 assignments once and nothing else."""
    selected = sorted(set(ids.reshape(-1).tolist()))
    assert [e for e, _, _ in calls] == selected
    assignments = 0
    for e, rows, _ in calls:
        tokens = (ids == e).any(dim=1).nonzero().squeeze(1)
        assert rows.shape[0] == tokens.numel() >= 1, e
        assert cases.bitwise_equal(rows, x[tokens]), f"expert {e} received the wrong rows"
        assignments += rows.shape[0]
    assert assignments == ids.numel()


def run_checked(x, ids, w, experts, zero=False):
    """Public call on the GPU with every output and immutability check; returns
    (output, calls, golden)."""
    x, ids, w = x.cuda(), ids.cuda(), w.cuda()
    before = (x.clone(), ids.clone(), w.clone())
    out, calls = call(x, ids, w, experts)
    assert out.shape == (x.shape[0], 5120) and out.dtype == BF16 and out.device == x.device
    assert bool(torch.isfinite(out).all())
    for name, now, then in zip(
        ("hidden_states", "expert_ids", "mixing_weights"), (x, ids, w), before
    ):
        assert cases.bitwise_equal(now, then), f"{name} was modified"
    check_calls(calls, x, ids)
    gold = cases.golden(x, ids, w, experts.fresh())
    assert [e for e, _, _ in calls] == [e for e, _ in gold["calls"]]
    return out, calls, gold


# ----------------------------------------------------------------------------- the interface


def test_public_callable_and_provider_interface():
    params = list(inspect.signature(beam_dispatch_combine).parameters)
    assert params == ["hidden_states", "expert_ids", "mixing_weights", "expert_provider"]
    doc = " ".join(inspect.getdoc(dispatch_combine).split())
    for phrase in (
        "expert_provider(expert_id, rows)",
        "owns dispatch and combination",
        "owns the expert computation",
        "[n, 5120]",
    ):
        assert phrase in doc, phrase
    assert (
        dispatch_combine.HIDDEN_SIZE,
        dispatch_combine.NUM_EXPERTS,
        dispatch_combine.EXPERTS_PER_TOKEN,
        dispatch_combine.ROUTED_SCALING_FACTOR,
    ) == (5120, 256, 8, 2.5)


def test_component_holds_no_parameters_and_allocates_nothing_persistent(experts):
    """No router parameter, expert matrix, module or configuration object exists to build:
    the component is a function, its entry holds no state, and after a call only the
    returned tensor remains allocated."""
    assert inspect.isfunction(beam_dispatch_combine)
    assert vars(beam_dispatch_combine_op) == {}
    assert not any(
        isinstance(v, (torch.nn.Module, torch.Tensor)) for v in vars(dispatch_combine).values()
    )
    x, ids, w = (t.cuda() for t in cases.matrix_case(17, 17))
    a, b = experts.a, experts.b
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["requested_bytes.all.current"]
    out = beam_dispatch_combine(x, ids, w, lambda e, rows: (rows.float() * a[e] + b[e]).to(BF16))
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()["requested_bytes.all.current"]
    assert after - before == out.numel() * out.element_size()


_T = 17


@pytest.mark.parametrize(
    "case,error,match",
    [
        ("hidden_rank_1", ValueError, r"hidden_states must be \[tokens, 5120\]"),
        ("hidden_rank_3", ValueError, r"hidden_states must be \[tokens, 5120\]"),
        ("hidden_width_5119", ValueError, r"hidden_states must be \[tokens, 5120\]"),
        ("zero_tokens", ValueError, "at least one token"),
        ("hidden_fp32", TypeError, "hidden_states must be torch.bfloat16"),
        ("hidden_fp16", TypeError, "hidden_states must be torch.bfloat16"),
        ("hidden_not_a_tensor", TypeError, "hidden_states must be a torch.Tensor"),
        ("hidden_on_host", ValueError, "sm_103 CUDA device"),
        ("ids_rank_1", ValueError, r"expert_ids must be \[tokens, 8\]"),
        ("ids_7_slots", ValueError, r"expert_ids must be \[tokens, 8\]"),
        ("ids_extra_row", ValueError, r"expert_ids must be \[tokens, 8\]"),
        ("ids_int32", TypeError, "expert_ids must be torch.int64"),
        ("ids_float", TypeError, "expert_ids must be torch.int64"),
        ("ids_on_host", ValueError, "expert_ids must be on hidden_states' device"),
        ("weights_7_slots", ValueError, r"mixing_weights must be \[tokens, 8\]"),
        ("weights_missing_row", ValueError, r"mixing_weights must be \[tokens, 8\]"),
        ("weights_transposed", ValueError, r"mixing_weights must be \[tokens, 8\]"),
        ("weights_bf16", TypeError, "mixing_weights must be torch.float32"),
        ("weights_fp64", TypeError, "mixing_weights must be torch.float32"),
        ("weights_on_host", ValueError, "mixing_weights must be on hidden_states' device"),
        ("provider_not_callable", TypeError, "expert_provider must be callable"),
    ],
)
def test_rejects_malformed_inputs_before_any_device_work(case, error, match, experts):
    """Each argument is named in its error, nothing is computed and no provider is called;
    the checks read metadata only, so they never synchronize with the device."""
    x, ids, w = (t.cuda() for t in cases.matrix_case(41, _T))
    provider = experts.fresh()
    args = {"hidden_states": x, "expert_ids": ids, "mixing_weights": w, "expert_provider": provider}
    change = {
        "hidden_rank_1": ("hidden_states", x[0]),
        "hidden_rank_3": ("hidden_states", x.unsqueeze(0)),
        "hidden_width_5119": ("hidden_states", x[:, :5119]),
        "zero_tokens": ("hidden_states", x[:0]),
        "hidden_fp32": ("hidden_states", x.float()),
        "hidden_fp16": ("hidden_states", x.half()),
        "hidden_not_a_tensor": ("hidden_states", x.tolist()),
        "hidden_on_host": ("hidden_states", x.cpu()),
        "ids_rank_1": ("expert_ids", ids.reshape(-1)),
        "ids_7_slots": ("expert_ids", ids[:, :7]),
        "ids_extra_row": ("expert_ids", torch.cat([ids, ids[:1]])),
        "ids_int32": ("expert_ids", ids.int()),
        "ids_float": ("expert_ids", ids.float()),
        "ids_on_host": ("expert_ids", ids.cpu()),
        "weights_7_slots": ("mixing_weights", w[:, :7]),
        "weights_missing_row": ("mixing_weights", w[:-1]),
        "weights_transposed": ("mixing_weights", w.t()),
        "weights_bf16": ("mixing_weights", w.to(BF16)),
        "weights_fp64": ("mixing_weights", w.double()),
        "weights_on_host": ("mixing_weights", w.cpu()),
        "provider_not_callable": ("expert_provider", None),
    }[case]
    args[change[0]] = change[1]
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(error, match=match):
            beam_dispatch_combine(**args)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert provider.calls == []


@pytest.mark.parametrize(
    "case,error,match",
    [
        ("fp32", TypeError, r"expert_provider\(\d+\) must return torch.bfloat16"),
        ("narrow", ValueError, r"expert_provider\(\d+\) must return \[\d+, 5120\]"),
        ("extra_row", ValueError, r"expert_provider\(\d+\) must return \[\d+, 5120\]"),
        ("host", ValueError, r"expert_provider\(\d+\) must return a dense tensor on cuda"),
        ("not_a_tensor", TypeError, r"expert_provider\(\d+\) must return a torch.Tensor"),
    ],
)
def test_rejects_provider_outputs_outside_the_contract(case, error, match, experts):
    x, ids, w = (t.cuda() for t in cases.matrix_case(17, _T))
    good = experts.fresh()
    bad = {
        "fp32": lambda e, rows: good(e, rows).float(),
        "narrow": lambda e, rows: good(e, rows)[:, :5119],
        "extra_row": lambda e, rows: torch.cat([good(e, rows), good(e, rows[:1])]),
        "host": lambda e, rows: good(e, rows).cpu(),
        "not_a_tensor": lambda e, rows: good(e, rows).tolist(),
    }[case]
    with pytest.raises(error, match=match):
        beam_dispatch_combine(x, ids, w, bad)
    assert len(good.calls) in (1, 2)  # refused at the first expert's output


# ----------------------------------------------------------------------------- the matrix


@pytest.mark.parametrize("tokens", cases.TOKENS)
@pytest.mark.parametrize("seed", cases.SEEDS)
def test_matrix_matches_the_source_combination(seed, tokens, experts):
    out, calls, gold = run_checked(*cases.matrix_case(seed, tokens), experts)
    assert cases.bitwise_equal(out, gold["output"]), f"seed {seed}, {tokens} tokens"
    # Each expert's output went back to its own tokens: the scattered golden is different.
    if tokens > 1:
        wrong = cases.golden(
            *(t.cuda() for t in cases.matrix_case(seed, tokens)),
            experts.fresh(),
            fault="wrong_scatter_mapping",
        )
        assert not torch.equal(wrong["output"], out)


# ----------------------------------------------------------------------------- fixtures


def test_shared_extreme_experts_keep_tokens_and_experts_apart(experts):
    x, ids, w, _ = cases.fixtures()["shared_extremes"]
    out, calls, gold = run_checked(x, ids, w, experts)
    assert cases.bitwise_equal(out, gold["output"])
    by_expert = {e: rows for e, rows, _ in calls}
    for e in (0, 255):  # every token selects both ends of the id range, in different slots
        assert cases.bitwise_equal(by_expert[e], x.cuda())
    assert calls[0][0] == 0 and calls[-1][0] == 255
    # Leakage across tokens, across slots and a re-normalization each change the output.
    for fault in ("wrong_scatter_mapping", "weights_slot_rotated", "renormalized_weights"):
        wrong = cases.golden(x.cuda(), ids.cuda(), w.cuda(), experts.fresh(), fault=fault)
        assert not torch.equal(wrong["output"], out), fault


def test_cancellation_witness_needs_every_rounding_step_in_place(experts):
    """Signed routes of unequal size: the rounded route, the fp32 accumulation and the single
    final scale are each visible in the output."""
    x, ids, w, _ = cases.fixtures()["cancellation_witness"]
    out, calls, gold = run_checked(x, ids, w, experts)
    assert cases.bitwise_equal(out, gold["output"])
    route = gold["route"].float()
    assert int((route > 0).sum()) > 0 and int((route < 0).sum()) > 0
    assert bool((w.cuda() != w.cuda().to(BF16).float()).all())  # no weight is a bf16 value
    for fault in cases.FAULTS:
        if fault == "renormalized_weights":
            continue  # these weights are already normalized
        wrong = cases.golden(x.cuda(), ids.cuda(), w.cuda(), experts.fresh(), fault=fault)
        assert not torch.equal(wrong["output"], out), fault


def test_zero_weight_routes_contribute_exactly_zero(experts):
    x, ids, w, _ = cases.fixtures()["zero_weight_route"]
    out, calls, gold = run_checked(x, ids, w, experts)
    assert cases.bitwise_equal(out, gold["output"])
    # A zero-weight route still calls its expert, and removing it changes nothing.
    called = [e for e, _, _ in calls]
    for t, j in enumerate(cases.ZERO_ROUTE_SLOT):
        assert int(ids[t, j]) in called
    without = cases.golden(
        x.cuda(), ids.cuda(), w.cuda(), experts.fresh(), skip_slots=cases.ZERO_ROUTE_SLOT
    )
    assert cases.bitwise_equal(out, without["output"])
    assert bool((out != 0).any(dim=1).all())


@pytest.mark.parametrize("name", ["all_zero_weights", "all_zero_outputs"])
def test_all_zero_weights_or_outputs_give_exact_zero(name, experts):
    x, ids, w, zero = cases.fixtures()[name]
    provider = cases.affine_experts(zero=True) if zero else experts
    out, _, _ = run_checked(x, ids, w, provider)
    assert bool((out == 0).all())
    # The hidden states are nonzero, so no residual or bias term reaches the output.
    assert bool((x != 0).any())


@pytest.mark.parametrize("seed", cases.SEEDS)
@pytest.mark.parametrize("tokens", [t for t in cases.TOKENS if t > 1])
def test_row_permutation_keeps_each_token_its_output(seed, tokens, experts):
    x, ids, w = (t.cuda() for t in cases.matrix_case(seed, tokens))
    perm = cases.permutation(seed, tokens).cuda()
    base, _ = call(x, ids, w, experts)
    permuted, _ = call(x[perm], ids[perm], w[perm], experts)
    assert cases.bitwise_equal(permuted[torch.argsort(perm)], base)


# ----------------------------------------------------------------------------- execution


class TorchCallTrace(TorchFunctionMode):
    """The name of every torch call made inside the block and the devices of its outputs."""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[str, list[str]]] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        outs = out if isinstance(out, (tuple, list)) else [out]
        devices = [str(t.device) for t in outs if isinstance(t, torch.Tensor)]
        self.calls.append((resolve_name(func) or repr(func), devices))
        return out


def test_token_arithmetic_runs_on_the_gpu(experts):
    """Gathers, products, roundings, the accumulation and the scale all produce tensors on
    the input's device; only the expert list itself comes to the host (``tolist``)."""
    x, ids, w = (t.cuda() for t in cases.matrix_case(41, 129))
    with TorchCallTrace() as trace:
        beam_dispatch_combine(x, ids, w, experts.fresh())
    names = [name for name, _ in trace.calls]
    short = {name.rsplit(".", 1)[-1] for name in names}
    assert {"__getitem__", "to", "index_add_"} <= short and {"__mul__", "mul"} & short, short
    produced = [device for _, devices in trace.calls for device in devices]
    assert produced and all(device == str(x.device) for device in produced), trace.calls
    assert not any("reduce" in n or "all_gather" in n or "distributed" in n for n in names)
