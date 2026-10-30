# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for one Beam MoE routed expert of block 3: bf16 / sm_103 / tp1.

Configuration under test, fixed in ``_expert_cases.py``: hidden 5120, intermediate 2560,
bias-free BF16 weights [out, in], BF16 already-normalized input; one sm_103 GPU; eager;
generated parameters; seeds {17, 41} x tokens {1, 17, 129}, and exact fixtures.

What is compared. Every call goes through the component's ``forward``. The three
``cublas_mm`` results and the activation op's mapping of that same call are captured, and
every step -- gate, up, fp32 SiLU, fp32 product, bf16 activation, output -- is compared
with the source expert written in plain torch under ``_expert_cases.MATRIX_RULES``, or
exactly on the fixtures built so that every step is known exactly.

Invocation (fp32 steps need the TF32 overrides removed before torch is imported)::

    env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE -u NVIDIA_TF32_OVERRIDE \\
        python -m pytest tests/unittest/_torch/modeling_v2/models/beam/moe_expert_bf16__sm_103__tp1

Dependencies: a TensorRT-LLM build for sm_103. No model files.
"""

from __future__ import annotations

import contextlib
import gc

import pytest
import torch
from torch.overrides import TorchFunctionMode, resolve_name

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.swiglu_fp32 import (  # noqa: E501
    SwigluFp32,
    swiglu_fp32,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import CublasMm, cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.moe_expert_bf16__sm_103__tp1.expert import (  # noqa: E501
    WEIGHT_SHAPES,
    BeamMoeExpertBf16Sm103Tp1,
)

__extra_import_path__ = [".", "../../.."]
import _expert_cases as cases  # noqa: E402
from _validating import validating  # noqa: E402

assert torch.cuda.is_available(), "the Beam MoE expert tests require a CUDA device"

BF16 = torch.bfloat16
_WEIGHT_BYTES = sum(2 * rows * cols for rows, cols in WEIGHT_SHAPES.values())


@pytest.fixture(scope="module", autouse=True)
def controls():
    """IEEE fp32, no reduced-precision bf16 reduction and no autocast for the whole module."""
    with cases.numerical_controls():
        yield


# ----------------------------------------------------------------------------- observing a call


@contextlib.contextmanager
def observed_calls():
    """Every cublas_mm result and activation mapping made inside the block, unchanged."""
    seen = {"mm": [], "act": []}
    mm_call, act_call = CublasMm.__call__, SwigluFp32.__call__

    def mm(self, *args, **kwargs):
        out = mm_call(self, *args, **kwargs)
        seen["mm"].append((args, out))
        return out

    def act(self, *args, **kwargs):
        out = act_call(self, *args, **kwargs)
        seen["act"].append((args, out))
        return out

    CublasMm.__call__, SwigluFp32.__call__ = mm, act
    try:
        yield seen
    finally:
        CublasMm.__call__, SwigluFp32.__call__ = mm_call, act_call


def run(expert: BeamMoeExpertBf16Sm103Tp1, x: torch.Tensor) -> dict:
    """Every step of one forward call, by name; ``y`` is what forward returned."""
    with validating(cublas_mm, swiglu_fp32), observed_calls() as seen:
        y = expert(x)
    assert len(seen["mm"]) == 3 and len(seen["act"]) == 1, seen
    (gate_args, gate), (up_args, up), (down_args, out) = seen["mm"]
    ((act_args, steps),) = seen["act"]
    for args, weight in (
        (gate_args, expert.gate_proj),
        (up_args, expert.up_proj),
        (down_args, expert.down_proj),
    ):
        # The stored [out, in] weight itself, read column-major: no copy, no transpose kernel.
        operand = args[1]
        assert operand.data_ptr() == weight.data_ptr() and operand.stride() == (1, weight.shape[1])
    assert act_args[0] is gate and act_args[1] is up and down_args[0] is steps["act"]
    assert out is y
    return dict(gate=gate, up=up, **{k: steps[k] for k in ("silu", "product", "act")}, y=y)


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


def loaded(weights: dict) -> BeamMoeExpertBf16Sm103Tp1:
    expert = BeamMoeExpertBf16Sm103Tp1(device="cuda")
    expert.load_weights(weights)
    return expert


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(a.view(torch.int16), b.view(torch.int16))
    )


# ----------------------------------------------------------------------------- state


def _requested_cuda_bytes() -> int:
    """Bytes the live tensors asked for; unlike memory_allocated, free of block rounding."""
    return torch.cuda.memory_stats()["requested_bytes.all.current"]


def test_construction_holds_one_expert_and_nothing_else():
    gc.collect()
    torch.cuda.synchronize()
    before = _requested_cuda_bytes()
    expert = BeamMoeExpertBf16Sm103Tp1(device="cuda")
    torch.cuda.synchronize()
    assert _requested_cuda_bytes() - before == _WEIGHT_BYTES == 78_643_200
    inventory = {
        name: (tuple(p.shape), p.dtype, p.device.type, p.requires_grad)
        for name, p in expert.named_parameters()
    }
    assert inventory == {
        "gate_proj": ((2560, 5120), BF16, "cuda", False),
        "up_proj": ((2560, 5120), BF16, "cuda", False),
        "down_proj": ((5120, 2560), BF16, "cuda", False),
    }
    assert sum(p.numel() for p in expert.parameters()) == 39_321_600
    assert list(expert.buffers()) == [] and list(expert.children()) == []
    assert list(expert.checkpoint_tensors()) == list(WEIGHT_SHAPES)


def test_construction_rejects_a_host_device():
    with pytest.raises(ValueError, match="sm_103 CUDA devices"):
        BeamMoeExpertBf16Sm103Tp1(device="cpu")


@pytest.mark.parametrize("source_device", ["cpu", "cuda"])
def test_load_copies_exact_values(source_device):
    weights = {k: v.to(source_device) for k, v in cases.matrix_case(17)[0].items()}
    expert = loaded(weights)
    for name, tensor in expert.checkpoint_tensors().items():
        assert tensor.device.type == "cuda" and tensor.dtype == BF16
        assert tuple(tensor.shape) == WEIGHT_SHAPES[name]
        assert bitwise_equal(tensor.cpu(), weights[name].cpu()), name
        assert tensor.data_ptr() != weights[name].data_ptr()


@pytest.mark.parametrize(
    "case,error,match",
    [
        ("not_loaded", RuntimeError, "load_weights must be called"),
        ("rank_1", ValueError, r"must be \[tokens, 5120\]"),
        ("rank_3", ValueError, r"must be \[tokens, 5120\]"),
        ("width_5119", ValueError, r"must be \[tokens, 5120\]"),
        ("zero_tokens", ValueError, "at least one token"),
        ("fp32", TypeError, "must be torch.bfloat16"),
        ("fp16", TypeError, "must be torch.bfloat16"),
        ("host", ValueError, "expert's device"),
        ("not_a_tensor", TypeError, "must be a torch.Tensor"),
    ],
)
def test_rejects_unsupported_inputs_before_any_kernel(case, error, match):
    weights, x_all = cases.matrix_case(17)
    expert = BeamMoeExpertBf16Sm103Tp1(device="cuda")
    if case != "not_loaded":
        expert.load_weights(weights)
    x = x_all[:17].cuda()
    bad = {
        "not_loaded": x,
        "rank_1": x[0],
        "rank_3": x.unsqueeze(0),
        "width_5119": x[:, :5119],
        "zero_tokens": x[:0],
        "fp32": x.float(),
        "fp16": x.half(),
        "host": x.cpu(),
        "not_a_tensor": x.tolist(),
    }[case]
    with observed_calls() as seen, pytest.raises(error, match=match):
        expert(bad)
    assert seen["mm"] == [] and seen["act"] == []


@pytest.mark.parametrize(
    "case,error,match",
    [
        ("down_fp32", TypeError, "down_proj.weight must be torch.bfloat16"),
        ("down_transposed", ValueError, r"down_proj.weight must be \[5120, 2560\]"),
        ("gate_fp16", TypeError, "gate_proj.weight must be torch.bfloat16"),
        ("up_transposed", ValueError, r"up_proj.weight must be \[2560, 5120\]"),
        ("missing_down", ValueError, r"missing \['down_proj.weight'\]"),
        ("unexpected_bias", ValueError, r"unexpected \['down_proj.bias'\]"),
        ("down_not_a_tensor", TypeError, "down_proj.weight must be a torch.Tensor"),
        ("down_meta", ValueError, "dense host or CUDA tensor"),
        ("not_a_mapping", TypeError, "weights must be a mapping"),
    ],
)
def test_rejected_load_leaves_the_expert_unchanged(case, error, match):
    """Every other tensor is a valid replacement; a bad down tensor is the last one checked."""
    weights, x_all = cases.matrix_case(17)
    expert = loaded(weights)
    x = x_all[:17].cuda()
    before = {k: v.clone() for k, v in expert.checkpoint_tensors().items()}
    y_before = expert(x)
    valid = cases.matrix_case(41)[0]
    gate, up, down = valid.values()
    replacement = {
        "down_fp32": dict(valid, **{"down_proj.weight": down.float()}),
        "down_transposed": dict(valid, **{"down_proj.weight": down.reshape(2560, 5120)}),
        "gate_fp16": dict(valid, **{"gate_proj.weight": gate.half()}),
        "up_transposed": dict(valid, **{"up_proj.weight": up.reshape(5120, 2560)}),
        "missing_down": {k: v for k, v in valid.items() if k != "down_proj.weight"},
        "unexpected_bias": dict(valid, **{"down_proj.bias": down[:, 0]}),
        "down_not_a_tensor": dict(valid, **{"down_proj.weight": down.float().numpy()}),
        "down_meta": dict(valid, **{"down_proj.weight": torch.empty_like(down, device="meta")}),
        "not_a_mapping": list(valid.items()),
    }[case]
    with pytest.raises(error, match=match):
        expert.load_weights(replacement)
    for name, tensor in expert.checkpoint_tensors().items():
        assert bitwise_equal(tensor, before[name]), name
    assert bitwise_equal(expert(x), y_before)


def test_rejected_first_load_leaves_the_expert_unloaded():
    weights = dict(cases.matrix_case(17)[0])
    weights["down_proj.weight"] = weights["down_proj.weight"].float()
    expert = BeamMoeExpertBf16Sm103Tp1(device="cuda")
    with pytest.raises(TypeError):
        expert.load_weights(weights)
    with pytest.raises(RuntimeError, match="load_weights must be called"):
        expert(torch.zeros(1, 5120, dtype=BF16, device="cuda"))


# ----------------------------------------------------------------------------- the matrix


@pytest.mark.parametrize("tokens", cases.TOKENS)
@pytest.mark.parametrize("seed", cases.SEEDS)
def test_matrix_every_step_matches_the_source_expert(seed, tokens):
    weights, x_all = cases.matrix_case(seed)
    expert = loaded(weights)
    x = x_all[:tokens].cuda()
    x_before = x.clone()
    params_before = {k: v.clone() for k, v in expert.checkpoint_tensors().items()}

    steps = run(expert, x)
    y = steps["y"]
    assert y.shape == (tokens, 5120) and y.dtype == BF16 and y.device == x.device
    assert bool(torch.isfinite(y).all())
    assert bitwise_equal(x, x_before)
    for name, tensor in expert.checkpoint_tensors().items():
        assert bitwise_equal(tensor, params_before[name]), name
    # Observing the call does not change it.
    assert bitwise_equal(expert(x), y)

    gold = cases.golden(x, *(t.cuda() for t in weights.values()))
    cases.assert_stages(steps, gold, cases.MATRIX_RULES, f"seed {seed}, {tokens} tokens")


# ----------------------------------------------------------------------------- fixtures


def test_zero_input_gives_exact_zero():
    expert = loaded(cases.matrix_case(17)[0])
    steps = run(expert, torch.zeros(17, 5120, dtype=BF16, device="cuda"))
    for stage, tensor in steps.items():
        assert tensor.dtype == cases.STAGE_DTYPES[stage]
        assert bool((tensor == 0).all()), stage


def test_signed_gates_exercise_silu_exactly():
    weights, x, expected = cases.signed_gate_fixture()
    gates = expected["gate"].double()
    assert int((gates < 0).sum()) >= 2000 and int((gates > 0).sum()) >= 2000
    assert int((gates == 0).sum()) > 0 and int((gates.abs() >= 24).sum()) > 0
    steps = run(loaded(weights), x.cuda())
    cases.assert_stages(steps, expected, cases.EXACT_RULES, "signed gates")
    # SiLU is not linear on these gates: it is neither the gate nor half of it.
    silu = steps["silu"].double().cpu()
    assert not torch.equal(silu, gates) and not torch.equal(silu, gates / 2)


def test_precision_fixture_tells_fp32_activation_from_bf16_steps():
    weights, x, expected, premature_act = cases.precision_fixture()
    steps = run(loaded(weights), x.cuda())
    cases.assert_stages(steps, expected, cases.EXACT_RULES, "precision fixture")
    # Every element of the exact activation differs from the twice-rounded one.
    assert bool((steps["act"].cpu().double() != premature_act.double()).all())


def test_sparse_mapping_fixes_orientation_and_feature_maps():
    weights, x, expected = cases.sparse_mapping_fixture()
    steps = run(loaded(weights), x.cuda())
    cases.assert_stages(steps, expected, cases.EXACT_RULES, "sparse mapping")
    # The fixture separates the maps it is meant to check: each wrong reading moves y.
    gate_w, up_w, down_w = (t.cuda() for t in weights.values())
    xd = x.cuda()
    wrong = {
        "gate/up swapped": cases.golden(xd, up_w, gate_w, down_w),
        "gate read transposed": cases.golden(xd, gate_w.reshape(5120, 2560).t(), up_w, down_w),
        "down read transposed": cases.golden(xd, gate_w, up_w, down_w.reshape(2560, 5120).t()),
    }
    for name, gold in wrong.items():
        assert not torch.equal(gold["y"].cpu(), expected["y"]), name


def test_row_permutation_keeps_tokens_apart():
    weights, x_all = cases.matrix_case(17)
    expert = loaded(weights)
    x = x_all.cuda()
    perm = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(20261019)).cuda()
    inverse = torch.argsort(perm)
    base = run(expert, x)
    permuted = run(expert, x[perm])
    back = {stage: t[inverse] for stage, t in permuted.items()}
    cases.assert_stages(back, base, cases.MATRIX_RULES, "row permutation")
    # A token-mixing expert would fail the same rule.
    mixed = torch.roll(permuted["y"], 1, dims=0)[inverse]
    assert cases.violations(mixed, base["y"], *cases.MATRIX_RULES["y"])["failing"] > 0


def test_no_residual_weighting_or_collective():
    """Zero down weights with a nonzero input: an added residual, bias or any other term
    would leave the output nonzero. Exact outputs on the fixtures above rule out a mixing
    weight or the routed scaling factor."""
    weights = dict(cases.matrix_case(17)[0])
    weights["down_proj.weight"] = torch.zeros_like(weights["down_proj.weight"])
    expert = loaded(weights)
    x = cases.matrix_case(17)[1][:17].cuda()
    with TorchCallTrace() as trace:
        steps = run(expert, x)
    assert bool((steps["y"] == 0).all())
    assert all(bool((steps[s] != 0).any()) for s in ("gate", "up", "act"))
    names = [name for name, _ in trace.calls]
    assert not any("reduce" in n or "all_gather" in n or "distributed" in n for n in names), names


def test_forward_runs_on_the_gpu():
    weights, x_all = cases.matrix_case(41)
    expert = loaded(weights)
    x = x_all[:129].cuda()
    with TorchCallTrace() as trace:
        expert(x)
    names = [name for name, _ in trace.calls]
    assert sum("cublas_mm" in name for name in names) == 3, names
    assert "torch.nn.functional.silu" in names, names
    produced = [device for _, devices in trace.calls for device in devices]
    assert produced and all(device == str(x.device) for device in produced), trace.calls
