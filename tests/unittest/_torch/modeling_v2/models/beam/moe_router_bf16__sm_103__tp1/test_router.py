# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the Beam MoE router component of block 3: bf16 / sm_103 / tp1.

Configuration under test, fixed in ``_router_cases.py`` and nowhere else: 256 experts in
8 groups of 32, 4 groups kept, 8 experts per token, hidden 5120; FP32 router weight and
correction bias, BF16 already-normalized input; one sm_103 GPU; eager; generated
parameters; seeds {17, 41} x tokens {1, 17, 129}, and the controlled fixtures there.

What is compared. Every call goes through the component's ``forward``; the op's mapping
of that same call is captured, and each of its steps is compared with the op's own
``reference`` on the same input under the rules in ``_router_cases.RULES``. Groups and
expert ids are compared exactly, in order. Each fixture also states the decision it was
built to force, independently of either implementation.

Invocation (fp32 needs the TF32 overrides removed before torch is imported)::

    env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE -u NVIDIA_TF32_OVERRIDE \\
        python -m pytest tests/unittest/_torch/modeling_v2/models/beam/moe_router_bf16__sm_103__tp1

Optional environment:

* ``BEAM_MOE_ROUTER_SOURCE_CAPTURES``: a directory of routes captured from the source
  model's router: ``seed<S>/params.pt`` (``weight``, ``bias``), ``seed<S>/T<T>.pt``
  (``hidden`` and ``native``, a dict of ``logits``, ``scores``, ``corrected``,
  ``group_score``, ``top_groups``, ``masked``, ``ids``, ``weights``) and
  ``fixtures/<name>.pt`` (``hidden``, ``bias`` and the same names). The generated inputs
  must then equal the captured ones bit for bit, the op's ``reference`` must reproduce
  every captured tensor bit for bit, and the component is compared with the captures
  under the same rules as with the reference.
* ``BEAM_MOE_ROUTER_EVIDENCE``: a directory that receives one JSON record per test case:
  every comparison with its rule, hashes of the routed tensors, the torch calls and CUDA
  kernels of the forward, and the device and software it ran on.

Dependencies: a TensorRT-LLM build for sm_103. No model files.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import sys
import types
from pathlib import Path

import pytest
import torch
from torch.overrides import TorchFunctionMode, resolve_name

import tensorrt_llm
import tensorrt_llm._torch.models.modeling_beam as builtin_beam
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.beam_grouped_topk_router import (  # noqa: E501
    BeamGroupedTopkRouter,
    beam_grouped_topk_router,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.moe_router_bf16__sm_103__tp1.router import (  # noqa: E501
    NORM_EPS,
    BeamMoeRouterBf16Sm103Tp1,
)

__extra_import_path__ = [".", "../../.."]
import _router_cases as cases  # noqa: E402
from _validating import validating  # noqa: E402

assert torch.cuda.is_available(), "the Beam MoE router tests require a CUDA device"

#: The router as the checkpoint configures it. Only these fields are read.
ROUTER_CONFIG = types.MappingProxyType(
    dict(
        hidden_size=5120,
        num_experts=256,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        norm_topk_prob=True,
        mlp_layer_types=("dense", "dense", "dense", "sparse"),
    )
)
LAYER_IDX = 3
SM = (10, 3)
#: Eager, one GPU, one call per case.
EXECUTION = types.MappingProxyType(dict(eager=True, cuda_graph=False, devices=1))
OP_KWARGS = types.MappingProxyType(
    dict(n_group=cases.GROUPS, topk_group=cases.TOPK_GROUP, top_k=cases.TOP_K, eps=cases.NORM_EPS)
)
#: Op names -> the names a source capture stores them under.
CAPTURE_NAMES = types.MappingProxyType(
    dict(
        logits="logits",
        scores="scores",
        selection_scores="corrected",
        group_scores="group_score",
        groups="top_groups",
        masked_selection_scores="masked",
        ids="ids",
        weights="weights",
    )
)
#: Calls that move token data to the host, and with it a routing decision.
HOST_TRANSFERS = frozenset(
    "torch.Tensor." + name
    for name in ("cpu", "item", "tolist", "numpy", "__bool__", "__int__", "__float__", "__index__")
)


def router_config(**overrides) -> types.SimpleNamespace:
    fields = dict(ROUTER_CONFIG)
    fields["mlp_layer_types"] = list(fields["mlp_layer_types"])
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


@pytest.fixture(scope="module", autouse=True)
def no_tf32_override():
    """The router's fp32 steps are IEEE fp32; these variables substitute TF32 regardless of
    what torch is told."""
    for var in ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE"):
        assert var not in os.environ, (
            f"{var} is set and substitutes TF32 for fp32 GEMMs; run with `env -u {var}`"
        )


# ----------------------------------------------------------------------------- evidence


def _environment() -> dict:
    device = torch.cuda.current_device()
    return {
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "tensorrt_llm": tensorrt_llm.__version__,
        "tensorrt_llm_file": tensorrt_llm.__file__,
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "argv": sys.argv,
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "execution": dict(EXECUTION),
        "ambient_math": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "source_captures": os.environ.get("BEAM_MOE_ROUTER_SOURCE_CAPTURES"),
    }


def _write_evidence(name: str, record: dict) -> None:
    directory = os.environ.get("BEAM_MOE_ROUTER_EVIDENCE")
    if not directory:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{name}.json").write_text(json.dumps(record, indent=1, default=str) + "\n")


def tensor_sha256(t: torch.Tensor) -> str:
    flat = t.detach().contiguous().cpu().reshape(-1)
    h = hashlib.sha256()
    h.update(f"{t.dtype}|{tuple(t.shape)}|".encode())
    h.update(flat.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def route_hashes(route: dict) -> dict:
    return {name: tensor_sha256(t) for name, t in sorted(route.items())}


# ----------------------------------------------------------------------------- source captures


def _captures_dir() -> Path | None:
    directory = os.environ.get("BEAM_MOE_ROUTER_SOURCE_CAPTURES")
    return Path(directory) if directory else None


def _as_route(captured: dict) -> dict:
    return {name: captured[key] for name, key in CAPTURE_NAMES.items()}


def matrix_captures(seed: int, tokens: int) -> dict | None:
    root = _captures_dir()
    if root is None:
        return None
    params = torch.load(root / f"seed{seed}" / "params.pt", weights_only=True)
    blob = torch.load(root / f"seed{seed}" / f"T{tokens}.pt", weights_only=True)
    return dict(
        weight=params["weight"],
        bias=params["bias"],
        hidden=blob["hidden"],
        route=_as_route(blob["native"]),
    )


def fixture_captures(name: str) -> dict | None:
    root = _captures_dir()
    if root is None:
        return None
    blob = torch.load(root / "fixtures" / f"{name}.pt", weights_only=True)
    return dict(bias=blob["bias"], hidden=blob["hidden"], route=_as_route(blob))


def check_against_captures(
    captured: dict | None, inputs: dict, route: dict, reference: dict
) -> list:
    """With captures: same input bytes, the reference reproduces them bitwise, and the
    component meets the rules against them. Without: nothing."""
    if captured is None:
        return []
    results = []
    for name, tensor in inputs.items():
        same = torch.equal(tensor.cpu(), captured[name].cpu())
        results.append(
            {
                "tensor": f"input:{name}",
                "against": "source_captures",
                "rule": "bitwise",
                "passed": same,
                "problems": [] if same else [f"generated {name} differs from the capture"],
            }
        )
    for name, tensor in captured["route"].items():
        same = torch.equal(reference[name].cpu(), tensor.cpu())
        results.append(
            {
                "tensor": f"reference:{name}",
                "against": "source_captures",
                "rule": "bitwise",
                "passed": same,
                "problems": [] if same else [f"reference {name} is not the captured {name}"],
            }
        )
    device_route = {k: v.to(route["ids"].device) for k, v in captured["route"].items()}
    results.extend(cases.compare_routes(route, device_route, "source_captures"))
    return results


# ----------------------------------------------------------------------------- observing a call


@contextlib.contextmanager
def recording_op_calls():
    """Capture the mapping each op call returns, without changing what it returns."""
    calls: list[dict] = []
    original = BeamGroupedTopkRouter.__call__

    def recorded(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        calls.append(out)
        return out

    BeamGroupedTopkRouter.__call__ = recorded
    try:
        yield calls
    finally:
        BeamGroupedTopkRouter.__call__ = original


@contextlib.contextmanager
def builtin_router_forbidden():
    """The built-in Beam router must not compute any part of this component's routing."""
    saved = {name: getattr(builtin_beam.BeamRouter, name) for name in ("forward", "route")}

    def refuse(*args, **kwargs):
        raise AssertionError("the built-in BeamRouter was called")

    for name in saved:
        setattr(builtin_beam.BeamRouter, name, refuse)
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(builtin_beam.BeamRouter, name, fn)


def _tensors_of(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in _tensors_of(v)]
    if isinstance(value, dict):
        return [t for v in value.values() for t in _tensors_of(v)]
    return []


class TorchCallTrace(TorchFunctionMode):
    """Every torch call made inside the block, with the device and dtype of what it returned,
    and the fp32 matmul and autocast state at each projection."""

    def __init__(self):
        super().__init__()
        self.calls: list[dict] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = resolve_name(func) or getattr(func, "__qualname__", None) or repr(func)
        entry = {"func": name}
        if name == "torch.nn.functional.linear":
            entry["math"] = {
                "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
                "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "autocast_cuda": torch.is_autocast_enabled("cuda"),
            }
            entry["operand_dtypes"] = [str(t.dtype) for t in _tensors_of(args)]
        out = func(*args, **kwargs)
        produced = _tensors_of(out)
        if produced:
            entry["outputs"] = [[str(t.device), str(t.dtype), list(t.shape)] for t in produced]
        self.calls.append(entry)
        return out


def loaded_router(weight: torch.Tensor, bias: torch.Tensor) -> BeamMoeRouterBf16Sm103Tp1:
    router = BeamMoeRouterBf16Sm103Tp1(router_config(), LAYER_IDX, device="cuda")
    router.load_weights(weight.cuda(), bias.cuda())
    return router


def routed(router: BeamMoeRouterBf16Sm103Tp1, hidden: torch.Tensor) -> tuple:
    """(weights, ids, the op's mapping of that same forward call)."""
    with builtin_router_forbidden(), validating(beam_grouped_topk_router):
        with recording_op_calls() as calls:
            weights, ids = router(hidden)
    assert len(calls) == 1, f"forward made {len(calls)} op calls, expected 1"
    (route,) = calls
    assert weights is route["weights"] and ids is route["ids"], (
        "forward did not return the op call's own weights and ids"
    )
    return weights, ids, route


def reference_route(router: BeamMoeRouterBf16Sm103Tp1, hidden: torch.Tensor) -> dict:
    return beam_grouped_topk_router.reference(
        hidden, router.weight, router.e_score_correction_bias, **OP_KWARGS
    )


def check_output_contract(weights, ids, hidden) -> list:
    tokens = hidden.shape[0]
    problems = []
    if tuple(weights.shape) != (tokens, 8) or weights.dtype != torch.float32:
        problems.append(f"weights {tuple(weights.shape)} {weights.dtype}")
    if tuple(ids.shape) != (tokens, 8) or ids.dtype != torch.int64:
        problems.append(f"ids {tuple(ids.shape)} {ids.dtype}")
    if weights.device != hidden.device or ids.device != hidden.device:
        problems.append(f"outputs on {weights.device}/{ids.device}, input on {hidden.device}")
    return [
        {
            "tensor": "output_contract",
            "against": "contract",
            "rule": "[T, 8] float32 weights and int64 ids on the input device",
            "passed": not problems,
            "problems": problems,
        }
    ]


def check_invariants(route: dict, sum_to_one: bool) -> list:
    """Per row: eight distinct ids in range, each in a selected group; finite, nonnegative
    weights; with `sum_to_one`, |sum(weights) - 1| <= NORM_ATOL."""
    ids, groups, weights = route["ids"], route["groups"], route["weights"]
    distinct = (ids.sort(dim=-1).values.diff(dim=-1) != 0).all(dim=-1)
    in_range = ((ids >= 0) & (ids < cases.EXPERTS)).all(dim=-1)
    member = ((ids // cases.PER_GROUP).unsqueeze(-1) == groups.unsqueeze(1)).any(dim=-1).all(-1)
    excluded = route["masked_selection_scores"] == float("-inf")
    expert_selected_group = (
        (torch.arange(cases.EXPERTS, device=ids.device) // cases.PER_GROUP).view(1, -1, 1)
        == groups.unsqueeze(1)
    ).any(dim=-1)
    mask_exact = (excluded == ~expert_selected_group).all(dim=-1)
    finite = (torch.isfinite(weights) & (weights >= 0)).all(dim=-1)
    deviation = (weights.sum(dim=-1).double() - 1.0).abs()
    checks = {
        "eight_distinct_ids": distinct,
        "ids_in_range": in_range,
        "ids_in_selected_groups": member,
        "excluded_exactly_outside_selected_groups": mask_exact,
        "weights_finite_nonnegative": finite,
    }
    if sum_to_one:
        checks["weights_sum_to_one"] = deviation <= cases.NORM_ATOL
    results = []
    for name, ok in checks.items():
        bad = [int(i) for i in torch.nonzero(~ok).flatten().cpu()]
        results.append(
            {
                "tensor": name,
                "against": "invariant",
                "rule": f"|sum - 1| <= {cases.NORM_ATOL:g}"
                if name == "weights_sum_to_one"
                else name,
                "passed": not bad,
                "problems": [f"rows {bad[:8]}"] if bad else [],
            }
        )
    results[-1]["max_abs_sum_minus_one"] = float(deviation.max())
    return results


# ----------------------------------------------------------------------------- construction and loading


def test_construction_allocates_only_the_router_state() -> None:
    """Construction allocates exactly the two FP32 router tensors: no expert, attention,
    cache or model state, and no submodule."""
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    router = BeamMoeRouterBf16Sm103Tp1(router_config(), LAYER_IDX, device="cuda")
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() - before
    params = {n: (tuple(p.shape), p.dtype, p.requires_grad) for n, p in router.named_parameters()}
    buffers = {n: (tuple(b.shape), b.dtype) for n, b in router.named_buffers()}
    elements = sum(p.numel() for p in router.parameters()) + sum(
        b.numel() for b in router.buffers()
    )
    record = {
        "test": "test_construction_allocates_only_the_router_state",
        "parameters": params,
        "buffers": buffers,
        "submodules": [n for n, _ in router.named_modules() if n],
        "elements": elements,
        "cuda_bytes_allocated": allocated,
        "environment": _environment(),
    }
    _write_evidence("construction", record)
    assert params == {"weight": ((256, 5120), torch.float32, False)}
    assert buffers == {"e_score_correction_bias": ((256,), torch.float32)}
    assert record["submodules"] == []
    assert elements == 256 * 5120 + 256
    assert allocated == (256 * 5120 + 256) * 4, allocated
    assert router.weight.device.type == "cuda"
    assert torch.cuda.get_device_capability(router.weight.device) == SM


@pytest.mark.parametrize(
    "overrides,layer_idx,match",
    [
        (dict(n_group=4), LAYER_IDX, "unsupported n_group"),
        (dict(topk_group=3), LAYER_IDX, "unsupported topk_group"),
        (dict(num_experts_per_tok=6), LAYER_IDX, "unsupported num_experts_per_tok"),
        (dict(num_experts=128), LAYER_IDX, "unsupported num_experts"),
        (dict(hidden_size=4096), LAYER_IDX, "unsupported hidden_size"),
        (dict(norm_topk_prob=False), LAYER_IDX, "unsupported norm_topk_prob"),
        (dict(mlp_layer_types=["dense"] * 4), LAYER_IDX, "sparse"),
        ({}, 0, "only the routers of sparse blocks"),
    ],
)
def test_construction_rejects_unsupported_configs(overrides, layer_idx, match) -> None:
    with pytest.raises(ValueError, match=match):
        BeamMoeRouterBf16Sm103Tp1(router_config(**overrides), layer_idx, device="cuda")


def test_construction_rejects_a_cpu_device() -> None:
    with pytest.raises(ValueError, match="sm_103 CUDA devices"):
        BeamMoeRouterBf16Sm103Tp1(router_config(), LAYER_IDX, device="cpu")


@pytest.mark.parametrize("seed", cases.SEEDS)
def test_loading_copies_both_tensors_exactly(seed) -> None:
    weight, bias, _ = cases.matrix_case(seed, 1)
    router = BeamMoeRouterBf16Sm103Tp1(router_config(), LAYER_IDX, device="cuda")
    router.load_weights(weight, bias)
    _write_evidence(
        f"loading-seed{seed}",
        {
            "test": "test_loading_copies_both_tensors_exactly",
            "weight": [
                list(router.weight.shape),
                str(router.weight.dtype),
                tensor_sha256(router.weight),
            ],
            "bias": [
                list(router.e_score_correction_bias.shape),
                str(router.e_score_correction_bias.dtype),
                tensor_sha256(router.e_score_correction_bias),
            ],
            "generated": [tensor_sha256(weight), tensor_sha256(bias)],
        },
    )
    assert router.weight.dtype == torch.float32 and tuple(router.weight.shape) == (256, 5120)
    assert router.e_score_correction_bias.dtype == torch.float32
    assert tuple(router.e_score_correction_bias.shape) == (256,)
    assert torch.equal(router.weight.cpu(), weight)
    assert torch.equal(router.e_score_correction_bias.cpu(), bias)


@pytest.mark.parametrize(
    "case,error,match",
    [
        ("weight_rows", ValueError, r"weight must be \[256, 5120\]"),
        ("weight_width", ValueError, r"weight must be \[256, 5120\]"),
        ("weight_transposed", ValueError, r"weight must be \[256, 5120\]"),
        ("bias_length", ValueError, r"correction_bias must be \[256\]"),
        ("bias_length_one", ValueError, r"correction_bias must be \[256\]"),
        ("weight_bf16", TypeError, "weight must be torch.float32"),
        ("bias_fp64", TypeError, "correction_bias must be torch.float32"),
        ("weight_not_tensor", TypeError, "weight must be a torch.Tensor"),
    ],
)
def test_loading_rejects_invalid_parameters(case, error, match) -> None:
    """A rejected load raises before copying either tensor: the router keeps its state."""
    weight, bias, _ = cases.matrix_case(17, 1)
    router = loaded_router(weight, bias)
    good_weight, good_bias = router.weight.clone(), router.e_score_correction_bias.clone()
    other_weight, other_bias, _ = cases.matrix_case(41, 1)
    bad = {
        "weight_rows": (other_weight[:255], other_bias),
        "weight_width": (other_weight[:, :5119], other_bias),
        "weight_transposed": (other_weight.t().contiguous(), other_bias),
        "bias_length": (other_weight, other_bias[:255]),
        "bias_length_one": (other_weight, other_bias[:1]),
        "weight_bf16": (other_weight.to(torch.bfloat16), other_bias),
        "bias_fp64": (other_weight, other_bias.double()),
        "weight_not_tensor": ([0.0] * 5120, other_bias),
    }[case]
    with pytest.raises(error, match=match):
        router.load_weights(*bad)
    assert torch.equal(router.weight, good_weight)
    assert torch.equal(router.e_score_correction_bias, good_bias)


@pytest.mark.parametrize(
    "case,error,match",
    [
        ("width_5119", ValueError, r"hidden_states must be \[tokens, 5120\]"),
        ("width_5121", ValueError, r"hidden_states must be \[tokens, 5120\]"),
        ("batched_3d", ValueError, r"hidden_states must be \[tokens, 5120\]"),
        ("one_dim", ValueError, r"hidden_states must be \[tokens, 5120\]"),
        ("fp32_input", TypeError, "hidden_states must be torch.bfloat16"),
        ("fp16_input", TypeError, "hidden_states must be torch.bfloat16"),
        ("no_tokens", ValueError, "at least one token"),
        ("cpu_input", ValueError, "router's device"),
        ("not_a_tensor", TypeError, "hidden_states must be a torch.Tensor"),
        ("not_loaded", RuntimeError, "load_weights must be called"),
    ],
)
def test_forward_rejects_invalid_input(case, error, match) -> None:
    weight, bias, hidden = cases.matrix_case(17, 17)
    if case == "not_loaded":
        router = BeamMoeRouterBf16Sm103Tp1(router_config(), LAYER_IDX, device="cuda")
    else:
        router = loaded_router(weight, bias)
    hidden = hidden.cuda()
    bad = {
        "width_5119": hidden[:, :5119],
        "width_5121": torch.cat([hidden, hidden[:, :1]], dim=1),
        "batched_3d": hidden.unsqueeze(0),
        "one_dim": hidden[0],
        "fp32_input": hidden.float(),
        "fp16_input": hidden.half(),
        "no_tokens": hidden[:0],
        "cpu_input": hidden.cpu(),
        "not_a_tensor": [0.0] * 5120,
        "not_loaded": hidden,
    }[case]
    with pytest.raises(error, match=match):
        router(bad)


# ----------------------------------------------------------------------------- precision


def test_precision_path_is_fp32_without_tf32_or_autocast() -> None:
    """Projection, sigmoid, selection scores, group scores and normalization are fp32, the
    projection is IEEE fp32 even when the caller has TF32 or bf16 autocast on, and the
    caller's settings are restored afterwards."""
    weight, bias, hidden = cases.matrix_case(17, 17)
    router = loaded_router(weight, bias)
    hidden = hidden.cuda()
    plain_weights, plain_ids, plain = routed(router, hidden)

    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cuda.matmul.fp32_precision)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    try:
        ambient = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cuda.matmul.fp32_precision)
        with torch.autocast("cuda", dtype=torch.bfloat16), TorchCallTrace() as trace:
            weights, ids, route = routed(router, hidden)
        after = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cuda.matmul.fp32_precision)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cuda.matmul.fp32_precision = saved

    # TF32 operands are ~1e-3 off; an IEEE fp32 product of these operands is ~1e-6 off fp64.
    fp64_logits = hidden.double() @ router.weight.double().t()
    fp64_error = float((route["logits"].double() - fp64_logits).abs().max())
    linear = [c for c in trace.calls if c["func"] == "torch.nn.functional.linear"]
    dtypes = {name: str(t.dtype) for name, t in route.items()}
    _write_evidence(
        "precision",
        {
            "test": "test_precision_path_is_fp32_without_tf32_or_autocast",
            "ambient_during_call": {"tf32": list(ambient), "autocast_bf16": True},
            "ambient_after_call": list(after),
            "projection_calls": linear,
            "route_dtypes": dtypes,
            "logits_vs_fp64_max_abs": fp64_error,
            "bitwise_equal_to_plain_call": {k: torch.equal(route[k], plain[k]) for k in route},
            "torch_calls": [c["func"] for c in trace.calls],
            "environment": _environment(),
        },
    )
    assert ambient == (True, "tf32") and after == (True, "tf32"), (ambient, after)
    assert len(linear) == 1, linear
    assert linear[0]["math"] == {
        "cuda_matmul_fp32_precision": "ieee",
        "cuda_matmul_allow_tf32": False,
        "autocast_cuda": False,
    }, linear[0]
    assert linear[0]["operand_dtypes"] == ["torch.float32", "torch.float32"], linear[0]
    assert linear[0]["outputs"][0][1] == "torch.float32"
    for name in ("logits", "scores", "selection_scores", "group_scores", "weights"):
        assert dtypes[name] == "torch.float32", (name, dtypes[name])
    assert dtypes["masked_selection_scores"] == "torch.float32"
    assert dtypes["ids"] == dtypes["groups"] == "torch.int64"
    assert fp64_error <= 1e-5, f"projection is not IEEE fp32: {fp64_error:.3e} from fp64"
    for name in route:
        assert torch.equal(route[name], plain[name]), f"{name} changed under TF32/autocast"
    assert torch.equal(weights, plain_weights) and torch.equal(ids, plain_ids)


# ----------------------------------------------------------------------------- the fixed matrix


@pytest.mark.parametrize("seed,tokens", cases.CASES, ids=cases.CASE_IDS)
def test_matrix(seed, tokens) -> None:
    """One case of the fixed matrix: the output contract, every step against the reference
    (and the source captures when given), the routing invariants, and the evidence that
    the routing ran on the GPU in this component's op with no host synchronization."""
    weight, bias, hidden_cpu = cases.matrix_case(seed, tokens)
    router = loaded_router(weight, bias)
    hidden = hidden_cpu.cuda()
    torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        with TorchCallTrace() as trace:
            previous = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                weights, ids, route = routed(router, hidden)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]

    reference = reference_route(router, hidden)
    results = check_output_contract(weights, ids, hidden)
    results += cases.compare_routes(route, reference, "reference")
    results += check_invariants(route, sum_to_one=True)
    results += check_against_captures(
        matrix_captures(seed, tokens),
        {"weight": weight, "bias": bias, "hidden": hidden_cpu},
        route,
        reference,
    )

    host = [c["func"] for c in trace.calls if c["func"] in HOST_TRANSFERS]
    off_device = [
        c for c in trace.calls for o in c.get("outputs", []) if not o[0].startswith("cuda")
    ]
    called = {c["func"] for c in trace.calls}
    routing_calls = {
        "torch.nn.functional.linear",
        "torch.Tensor.sigmoid",
        "torch.Tensor.topk",
        "torch.Tensor.scatter_",
        "torch.Tensor.masked_fill",
        "torch.Tensor.gather",
        "torch.Tensor.sum",
    }
    _write_evidence(
        f"matrix-seed{seed}-T{tokens}",
        {
            "test": "test_matrix",
            "case": {"seed": seed, "tokens": tokens},
            "component": f"{type(router).__module__}.{type(router).__qualname__}",
            "op": f"{BeamGroupedTopkRouter.__module__}.{BeamGroupedTopkRouter.__qualname__}",
            "rules": dict(cases.RULES, norm_atol=cases.NORM_ATOL),
            "results": results,
            "hashes": route_hashes(route),
            "inputs": {
                "weight": tensor_sha256(weight),
                "bias": tensor_sha256(bias),
                "hidden": tensor_sha256(hidden_cpu),
            },
            "ids": route["ids"].tolist(),
            "groups": route["groups"].tolist(),
            "torch_calls": trace.calls,
            "host_transfers": host,
            "off_device_outputs": off_device,
            "sync_debug_mode": "error",
            "cuda_kernels": kernels,
            "environment": _environment(),
        },
    )
    assert type(router).__module__.endswith("moe_router_bf16__sm_103__tp1.router")
    assert torch.cuda.get_device_capability(hidden.device) == SM
    assert not host, f"routing moved token data to the host: {host}"
    assert not off_device, f"routing produced tensors off the GPU: {off_device[:3]}"
    assert routing_calls <= called, sorted(routing_calls - called)
    assert kernels, "no CUDA kernel ran"
    cases.assert_all_pass(results)


# ----------------------------------------------------------------------------- controlled fixtures


def route_fixture(name: str) -> tuple[dict, dict, list]:
    """(component route, reference route, comparisons) for one fixture, captures included."""
    spec = cases.fixtures()[name]
    weight, bias, hidden_cpu = cases.fixture_tensors(spec)
    router = loaded_router(weight, bias)
    hidden = hidden_cpu.cuda()
    _, _, route = routed(router, hidden)
    reference = reference_route(router, hidden)
    results = cases.compare_routes(route, reference, "reference")
    results += check_invariants(route, sum_to_one=False)
    results += check_against_captures(
        fixture_captures(name), {"bias": bias, "hidden": hidden_cpu}, route, reference
    )
    exact = torch.equal(route["logits"].cpu(), spec["logits"].float())
    results.append(
        {
            "tensor": "fixture_logits",
            "against": "fixture",
            "rule": "the one-hot rows reproduce the fixture's logits exactly",
            "passed": exact,
            "problems": [] if exact else ["logits differ from the fixture's"],
        }
    )
    return route, reference, results


def _record_fixture(name: str, test: str, route: dict, results: list, extra: dict) -> None:
    _write_evidence(
        f"fixture-{name}",
        {
            "test": test,
            "fixture": name,
            "intent": cases.fixtures()[name]["intent"],
            "rows": list(cases.fixtures()[name]["rows"]),
            "ids": route["ids"].tolist(),
            "groups": route["groups"].tolist(),
            "weights": route["weights"].tolist(),
            "row_sums": route["weights"].sum(dim=-1).tolist(),
            "hashes": route_hashes(route),
            "results": results,
            **extra,
            "environment": _environment(),
        },
    )


def test_bias_selects_but_never_weights() -> None:
    """The correction bias changes which experts are chosen, and the returned weights are
    still the normalized unbiased scores of the chosen experts."""
    zero, _, zero_results = route_fixture("bias_zero")
    shift, _, shift_results = route_fixture("bias_shift")
    ids, scores = shift["ids"], shift["scores"]
    unbiased = scores.gather(-1, ids)
    unbiased = unbiased / (unbiased.sum(dim=-1, keepdim=True) + NORM_EPS)
    biased = shift["selection_scores"].gather(-1, ids)
    biased = biased / (biased.sum(dim=-1, keepdim=True) + NORM_EPS)
    selected_bias = cases.fixtures()["bias_shift"]["bias"][ids[0].cpu()]
    leak_gap = float((biased - shift["weights"]).abs().max())
    extra = {
        "ids_without_bias": zero["ids"].tolist(),
        "ids_with_bias": ids.tolist(),
        "groups_with_bias": shift["groups"].tolist(),
        "selected_bias": selected_bias.tolist(),
        "weights_vs_normalized_unbiased": cases.compare(
            "weights", shift["weights"], unbiased, cases.RULES["weights"]
        ),
        "weights_vs_normalized_biased_max_abs": leak_gap,
    }
    _record_fixture("bias_zero", "test_bias_selects_but_never_weights", zero, zero_results, {})
    _record_fixture(
        "bias_shift", "test_bias_selects_but_never_weights", shift, shift_results, extra
    )
    cases.assert_all_pass(zero_results + shift_results)
    assert zero["groups"].tolist() == [[0, 1, 2, 3]]
    assert zero["ids"].tolist() == [[0, 1, 32, 33, 64, 65, 96, 97]]
    assert shift["groups"].tolist() == [[6, 0, 1, 2]]
    assert ids.tolist() == [[193, 192, 0, 1, 32, 33, 64, 65]]
    assert torch.equal(zero["logits"], shift["logits"]) and torch.equal(zero["scores"], scores)
    assert bool((selected_bias != 0).any()) and len(set(selected_bias.tolist())) > 1
    assert extra["weights_vs_normalized_unbiased"]["passed"], extra[
        "weights_vs_normalized_unbiased"
    ]
    assert leak_gap > cases.RULES["weights"], f"bias in the weights would be invisible: {leak_gap}"


def test_group_boundary_moves_with_the_top_two_sum() -> None:
    """Only the challenger group's second expert moves, and the group crosses the 4th/5th
    boundary; the group holding the single best expert stays out."""
    route, _, results = route_fixture("group_boundary")
    group_scores = route["group_scores"]
    _record_fixture(
        "group_boundary",
        "test_group_boundary_moves_with_the_top_two_sum",
        route,
        results,
        {"group_scores": group_scores.tolist()},
    )
    cases.assert_all_pass(results)
    assert route["groups"].tolist() == [[0, 1, 2, 3], [0, 1, 2, 4]]
    assert route["ids"].tolist() == [
        [0, 1, 32, 33, 64, 65, 96, 97],
        [0, 1, 32, 33, 64, 65, 129, 128],
    ]
    # Group 4 scores 0.66 + 0.62 then 0.66 + 0.67 against group 3's 0.655 + 0.645.
    assert float(group_scores[0, 4]) < float(group_scores[0, 3]) < float(group_scores[1, 4])
    assert 5 not in route["groups"].flatten().tolist()


def test_exclusion_masks_every_expert_of_dropped_groups() -> None:
    """The best single expert sits in a group that loses; it and every other expert of the
    four dropped groups are -inf, although every retained selection score is negative."""
    route, _, results = route_fixture("exclusion")
    ids = route["ids"]
    selected_scores = route["selection_scores"].gather(-1, ids)
    spike_score = float(route["selection_scores"][0, cases.EXCLUSION_SPIKE])
    _record_fixture(
        "exclusion",
        "test_exclusion_masks_every_expert_of_dropped_groups",
        route,
        results,
        {"spike": cases.EXCLUSION_SPIKE, "spike_selection_score": spike_score},
    )
    cases.assert_all_pass(results)
    assert sorted(route["groups"][0].tolist()) == [0, 1, 2, 3]
    assert cases.EXCLUSION_SPIKE not in ids[0].tolist()
    assert spike_score == float(route["selection_scores"].max())
    assert bool((selected_scores < 0).all())
    assert all(e // cases.PER_GROUP in route["groups"][0].tolist() for e in ids[0].tolist())


def test_expert_boundary_moves_one_expert() -> None:
    """Expert 2 crosses the 8th/9th boundary within an unchanged group selection: it enters
    last and expert 97 leaves, every other position unchanged."""
    route, _, results = route_fixture("expert_boundary")
    _record_fixture("expert_boundary", "test_expert_boundary_moves_one_expert", route, results, {})
    cases.assert_all_pass(results)
    assert route["groups"].tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert route["ids"].tolist() == [
        [0, 1, 32, 33, 64, 65, 96, 97],
        [0, 1, 32, 33, 64, 65, 96, 2],
    ]


def test_ties_follow_the_pinned_topk_order() -> None:
    """Exactly tied scores are routed in the order `torch.topk(..., sorted=True)` returns
    -- the source's order, stated in `_router_cases.TIE_ROUTES` -- whether a row is routed
    in a batch or alone. Ids are compared in order, never sorted."""
    route, _, results = route_fixture("ties")
    spec = cases.fixtures()["ties"]
    weight, bias, hidden = cases.fixture_tensors(spec)
    router = loaded_router(weight, bias)
    alone = {}
    for i, name in enumerate(spec["rows"]):
        _, _, single = routed(router, hidden[i : i + 1].cuda())
        alone[name] = (single["groups"][0].tolist(), single["ids"][0].tolist())
    batched = {
        name: (route["groups"][i].tolist(), route["ids"][i].tolist())
        for i, name in enumerate(spec["rows"])
    }
    _record_fixture(
        "ties",
        "test_ties_follow_the_pinned_topk_order",
        route,
        results,
        {"batched": batched, "alone": alone, "pinned": cases.TIE_ROUTES},
    )
    cases.assert_all_pass(results)
    for name, pinned in cases.TIE_ROUTES.items():
        assert batched[name] == (list(pinned[0]), list(pinned[1])), (name, batched[name])
        assert alone[name] == batched[name], (name, alone[name])


def test_epsilon_is_the_source_denominator() -> None:
    """Weights divide by sum + 1e-20: a row whose scores underflow to 0 gets zero weights,
    not NaN, and a row whose selected scores sum to ~1e-20 sums to S / (S + 1e-20), well
    below 1 -- a denominator without the term, or with 1e-12 or 1e-6, gives 1 or ~0."""
    route, _, results = route_fixture("epsilon")
    selected = route["scores"].gather(-1, route["ids"])
    exact = torch.sigmoid(route["logits"].double()).gather(-1, route["ids"]).sum(dim=-1)
    expected_sum = (exact / (exact + 1e-20)).tolist()
    sums = route["weights"].sum(dim=-1).double().tolist()
    _record_fixture(
        "epsilon",
        "test_epsilon_is_the_source_denominator",
        route,
        results,
        {
            "selected_score_sums": selected.sum(dim=-1).tolist(),
            "expected_row_sums_fp64": expected_sum,
        },
    )
    cases.assert_all_pass(results)
    assert route["ids"].tolist() == [cases.EPSILON_IDS, cases.EPSILON_IDS]
    assert route["groups"].tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert bool((route["weights"][0] == 0).all()), route["weights"][0]
    assert float(selected[1].sum()) > 0
    assert sums[1] < 0.9, sums[1]
    assert abs(sums[1] - expected_sum[1]) <= cases.NORM_ATOL, (sums[1], expected_sum[1])
