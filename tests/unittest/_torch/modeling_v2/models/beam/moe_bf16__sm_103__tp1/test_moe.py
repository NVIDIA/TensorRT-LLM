# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the Beam MoE component of decoder block 3: bf16 / sm_103 / tp1.

Configuration under test, fixed below and nowhere else: block 3's MoE at hidden 5120 --
256 experts of intermediate 2560 in 8 groups of 32, 4 groups and 8 experts per token,
routed scaling factor 2.5; an fp32 router and bf16 experts with the source model's fp32
steps; one sm_103 GPU; eager, one call per case on already-normalized bf16 hidden
states; generated weights; seeds {17, 41} x lengths {1, 17, 129}, and designed fixtures.

What is compared. ``_moe_reference.py`` restates the source MoE in plain torch and
generates its state and input deterministically on the GPU. For every case the forward's
own routing must select exactly the reference's groups and experts, in order, and its
router steps and its output are held to the rules in `RULES`. The forward's op and
expert calls, and fixtures whose routing is fixed by design, show that each token
reaches its own experts and that expert ``e`` is the one loaded under ``experts.<e>``.

Invocation (fp32 steps need the TF32 overrides removed before torch is imported)::

    env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE -u NVIDIA_TF32_OVERRIDE \\
        python -m pytest tests/unittest/_torch/modeling_v2/models/beam/moe_bf16__sm_103__tp1

Optional environment:

* ``BEAM_MOE_EVIDENCE``: a directory that receives one JSON record per compared case:
  every comparison with its rule, the op and expert calls where they were recorded, and
  the device and software it ran on; and ``fp32-session.json``, the fp32 GEMM checks made
  before and after the module's tests.

Dependencies: a TensorRT-LLM build for sm_103 and about 60 GiB of free device memory:
one seed's generated state and the loaded MoE hold 18.75 GiB of expert weights each, and
one case builds a second, unloaded MoE. No model files.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import types
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

import tensorrt_llm
import tensorrt_llm._torch.models.modeling_beam as builtin_beam
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.swiglu_fp32 import (  # noqa: E501
    swiglu_fp32,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.beam_dispatch_combine import (  # noqa: E501
    beam_dispatch_combine_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.beam_grouped_topk_router import (  # noqa: E501
    beam_grouped_topk_router,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (  # noqa: E501
    dispatch_combine,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.moe_bf16__sm_103__tp1 import (
    moe as moe_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.moe_bf16__sm_103__tp1.moe import (
    BeamMoeBf16Sm103Tp1,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.moe_expert_bf16__sm_103__tp1.expert import (  # noqa: E501
    BeamMoeExpertBf16Sm103Tp1,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.moe_router_bf16__sm_103__tp1.router import (  # noqa: E501
    BeamMoeRouterBf16Sm103Tp1,
)

__extra_import_path__ = [".", "../../.."]
import _moe_reference as reference  # noqa: E402
from _validating import validating  # noqa: E402

assert torch.cuda.is_available(), "the Beam MoE tests require a CUDA device"

pytestmark = pytest.mark.high_cuda_memory

# -------------------------------------------------- the fixed configuration

#: Block 3's MoE as the checkpoint configures it. Only these fields are read.
MOE_CONFIG = types.MappingProxyType(
    dict(
        hidden_size=5120,
        moe_intermediate_size=2560,
        num_experts=256,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        hidden_act="silu",
        mlp_layer_types=("dense", "dense", "dense", "sparse"),
    )
)
LAYER_IDX = 3
DTYPE = torch.bfloat16
SM = (10, 3)
SEEDS = (17, 41)
LENGTHS = (1, 17, 129)
CASES = tuple((seed, length) for seed in SEEDS for length in LENGTHS)
CASE_IDS = [f"seed{seed}-T{length}" for seed, length in CASES]
#: Eager, one GPU, one call per case on already-normalized hidden states.
EXECUTION = types.MappingProxyType(dict(eager=True, cuda_graph=False, devices=1))

# -------------------------------------------------- comparison rules

_RTOL = 2.0**-7

#: (atol, rtol, dtype) per compared tensor. An element passes when
#: |actual - reference| <= atol + rtol * |reference|, both widened to fp64; every element
#: of every compared tensor must pass, and no NaN or Inf is accepted. rtol 2^-7 is one bf16
#: unit at any magnitude. Routing is never excused by a tolerance: the groups and the
#: expert ids must equal the reference's exactly, in order.
RULES = types.MappingProxyType(
    {
        "moe_output": (0.03, _RTOL, torch.bfloat16),
        "router_logits": (2e-5, 0.0, torch.float32),
        "router_scores": (3e-6, 0.0, torch.float32),
        "router_selection_scores": (3e-6, 0.0, torch.float32),
        "router_weights": (8e-7, 0.0, torch.float32),
    }
)
#: Per token, |sum(mixing weights) - 1| may not exceed this and no weight may reach
#: `MAX_WEIGHT`: the routed scaling factor 2.5 folded into the weights breaks both.
WEIGHT_SUM_ATOL = 1e-6
MAX_WEIGHT = 1.5

#: Every catalog entry the MoE's forward can call.
OPS = (beam_grouped_topk_router, cublas_mm, swiglu_fp32, beam_dispatch_combine_op)
#: What each selected expert runs, in order, inside BeamDispatchCombine.
EXPERT_CALLS = ("expert", "CublasMm", "CublasMm", "SwigluFp32", "CublasMm")
#: Keyword arguments of the entries' calls that identify the path.
_RECORDED_KWARGS = ("n_group", "topk_group", "top_k", "eps", "routed_scaling_factor")

# -------------------------------------------------- designed routing

#: Two sets of eight experts, the first and last expert of four groups each: A holds groups
#: 0, 1, 4 and 7, both ends of the id range included; B holds groups 2, 3, 5 and 6.
SET_A = (0, 31, 32, 63, 128, 159, 224, 255)
SET_B = (64, 95, 96, 127, 160, 191, 192, 223)
#: Per designed token, the rank of each member of its set, in the set's order. Rank r is the
#: logit -1 + r / 2; every other expert's logit is -2.
RANKS = (
    (0, 1, 2, 3, 4, 5, 6, 7),
    (7, 6, 5, 4, 3, 2, 1, 0),
    (3, 7, 1, 5, 0, 4, 2, 6),
    (6, 2, 4, 0, 5, 1, 7, 3),
)
#: The hidden column that carries each designed token's logits, per set.
COLUMNS_A = tuple(5 + t * (5120 // 4) for t in range(len(RANKS)))
COLUMNS_B = tuple(5 + 5120 // 8 + t * (5120 // 4) for t in range(len(RANKS)))
SPIKE = 16.0
OTHER_LOGIT = -2.0
FIXTURE_SEED = 17


def designed_logit(rank: int) -> float:
    return -1.0 + 0.5 * rank


def moe_config(**overrides) -> types.SimpleNamespace:
    fields = dict(MOE_CONFIG)
    fields["mlp_layer_types"] = list(fields["mlp_layer_types"])
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


# -------------------------------------------------- precision and environment


def _fp32_gemm_probe() -> float:
    """Worst relative error of an fp32 GEMM against fp64; TF32 substitution shows as ~1e-3."""
    gen = torch.Generator().manual_seed(1234)
    x = torch.randn(64, 5120, generator=gen).to(torch.bfloat16).float().cuda()
    w = (torch.randn(512, 5120, generator=gen) * 5120**-0.5).float().cuda()
    y = torch.nn.functional.linear(x, w)
    ref = x.double() @ w.double().t()
    return float((y.double() - ref).abs().max()) / float(ref.pow(2).mean().sqrt())


def _math_state() -> dict:
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


@pytest.fixture(scope="module", autouse=True)
def ieee_fp32():
    """The source's fp32 steps are IEEE fp32: refuse a process that substitutes TF32, both
    before the module's tests and after them."""
    for var in ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE"):
        assert var not in os.environ, (
            f"{var} is set and substitutes TF32 for fp32 GEMMs; run with `env -u {var}`"
        )
    saved = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    before = _fp32_gemm_probe()
    assert before <= 1e-5, f"fp32 GEMM is not IEEE fp32 (relative error {before:.3g})"
    state = {"math": _math_state(), "fp32_gemm_probe_before": before}
    yield state
    after = _fp32_gemm_probe()
    state.update(math_after=_math_state(), fp32_gemm_probe_after=after)
    _write_evidence("fp32-session", {"fixture": "ieee_fp32", "math": state})
    precision, allow_matmul, allow_cudnn = saved
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = allow_matmul
    torch.backends.cudnn.allow_tf32 = allow_cudnn
    assert after <= 1e-5, f"fp32 GEMM stopped being IEEE fp32 (relative error {after:.3g})"


@pytest.fixture(scope="module", autouse=True)
def device_memory():
    """Hand the module's cached state and MoE back to the device when its tests are done."""
    yield
    release_cache()


def _git_revision() -> str | None:
    root = Path(tensorrt_llm.__file__).resolve().parents[1]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _environment(ieee: dict) -> dict:
    device = torch.cuda.current_device()
    return {
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "tensorrt_llm": tensorrt_llm.__version__,
        "tensorrt_llm_file": tensorrt_llm.__file__,
        "source_revision": _git_revision(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "argv": sys.argv,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "execution": dict(EXECUTION),
        "math": ieee,
    }


def _write_evidence(name: str, record: dict) -> None:
    directory = os.environ.get("BEAM_MOE_EVIDENCE")
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


# -------------------------------------------------- comparison


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor, rule: str | None) -> dict:
    """Score one tensor against one rule; `rule=None` measures without judging."""
    result = {
        "tensor": name,
        "rule": rule,
        "shape": list(expected.shape),
        "dtype": str(actual.dtype),
    }
    problems = []
    dtype = expected.dtype if rule is None else RULES[rule][2]
    if tuple(actual.shape) != tuple(expected.shape):
        problems.append(f"shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    if actual.dtype != dtype or expected.dtype != dtype:
        problems.append(f"dtypes {actual.dtype}/{expected.dtype}, need {dtype}")
    if problems:
        result.update(passed=False, problems=problems)
        return result
    a, r = actual.double(), expected.to(actual.device).double()
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(r).all()):
        result.update(passed=False, problems=["non-finite values"])
        return result
    diff = (a - r).abs()
    numel = diff.numel()
    result.update(numel=numel, unequal=int((a != r).sum()), max_abs=0.0)
    if numel:
        worst = int(diff.argmax())
        result.update(
            max_abs=float(diff.max()),
            mean_abs=float(diff.mean()),
            worst_index=[int(i) for i in torch.unravel_index(torch.tensor(worst), diff.shape)],
            worst_actual=float(a.reshape(-1)[worst]),
            worst_expected=float(r.reshape(-1)[worst]),
        )
    if rule is None:
        result["passed"] = None
        return result
    atol, rtol, _ = RULES[rule]
    violations = int((diff > atol + rtol * r.abs()).sum())
    result.update(atol=atol, rtol=rtol, violations=violations)
    if violations:
        problems.append(
            f"{violations} of {numel} elements outside atol {atol} + rtol {rtol} * |ref| "
            f"(max abs {result['max_abs']:.3g})"
        )
    result.update(passed=not problems, problems=problems)
    return result


def compare_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Integer selections: equal element for element and in order, never sorted first."""
    same_kind = tuple(actual.shape) == tuple(expected.shape) and actual.dtype == expected.dtype
    if same_kind:
        rows = (actual == expected).reshape(expected.shape[0], -1).all(dim=-1)
        mismatched = [int(i) for i in torch.nonzero(~rows).flatten().cpu()]
    else:
        mismatched = list(range(expected.shape[0]))
    problems = []
    if not same_kind:
        problems.append(
            f"{tuple(actual.shape)} {actual.dtype} != {tuple(expected.shape)} {expected.dtype}"
        )
    elif mismatched:
        problems.append(f"rows {mismatched[:8]} differ")
    return {
        "tensor": name,
        "rule": "exact, in order",
        "shape": list(expected.shape),
        "mismatched_rows": mismatched,
        "passed": not problems,
        "problems": problems,
    }


def compare_routes(route: dict, ref: dict) -> list[dict]:
    """The router op's mapping of one call against the reference route of the same input."""
    results = [
        compare_exact("groups", route["groups"], ref["groups"]),
        compare_exact("ids", route["ids"], ref["ids"]),
        compare("router_logits", route["logits"], ref["logits"], "router_logits"),
        compare("router_scores", route["scores"], ref["scores"], "router_scores"),
        compare(
            "router_selection_scores",
            route["selection_scores"],
            ref["selection"],
            "router_selection_scores",
        ),
        compare("router_group_scores", route["group_scores"], ref["group_scores"], None),
        compare("router_weights", route["weights"], ref["weights"], "router_weights"),
    ]
    same = torch.equal(torch.isinf(route["masked_selection_scores"]), torch.isinf(ref["masked"]))
    results.append(
        {
            "tensor": "excluded_experts",
            "rule": "exactly the reference's -inf exclusions",
            "passed": same,
            "problems": [] if same else ["the excluded experts differ"],
        }
    )
    return results


def weight_invariants(route: dict) -> dict:
    """Mixing weights sum to one per token and carry no routed scaling factor."""
    weights = route["weights"]
    deviation = float((weights.double().sum(dim=-1) - 1.0).abs().max())
    largest = float(weights.max())
    problems = []
    if deviation > WEIGHT_SUM_ATOL:
        problems.append(f"a token's weights sum to 1 +- {deviation:.3g}")
    if largest >= MAX_WEIGHT:
        problems.append(f"a weight of {largest:.4g}")
    return {
        "tensor": "mixing_weights",
        "rule": f"|sum - 1| <= {WEIGHT_SUM_ATOL:g} per token, every weight < {MAX_WEIGHT}",
        "max_abs_sum_minus_one": deviation,
        "max_weight": largest,
        "passed": not problems,
        "problems": problems,
    }


def output_contract(out, tokens: int) -> dict:
    problems = []
    if not isinstance(out, torch.Tensor):
        problems.append(f"returned {type(out).__name__}")
    elif tuple(out.shape) != (tokens, 5120) or out.dtype != DTYPE or not out.is_cuda:
        problems.append(f"returned {tuple(out.shape)} {out.dtype} on {out.device}")
    elif torch.cuda.get_device_capability(out.device) != SM:
        problems.append(f"on {torch.cuda.get_device_name(out.device)}")
    elif not bool(torch.isfinite(out).all()):
        problems.append("non-finite values")
    return {
        "tensor": "output_contract",
        "rule": f"[{tokens}, 5120] bf16 on the sm_103 device, finite",
        "passed": not problems,
        "problems": problems,
    }


def tensor_versions(module: torch.nn.Module) -> dict[str, int]:
    """The in-place write counter of every parameter and buffer, by name."""
    versions = {name: p._version for name, p in module.named_parameters()}
    versions.update({name: b._version for name, b in module.named_buffers()})
    return versions


def unchanged(x: torch.Tensor, x_before: torch.Tensor, before: dict, after: dict) -> dict:
    written = sorted(name for name, version in before.items() if after.get(name) != version)
    problems = []
    if not bitwise_equal(x, x_before):
        problems.append("the input was modified")
    if written:
        problems.append(f"{len(written)} parameters or buffers were written: {written[:4]}")
    return {
        "tensor": "input_and_parameters",
        "rule": "input bits and parameter write counters unchanged by the call",
        "passed": not problems,
        "problems": problems,
    }


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same dtype, shape and bits (so -0.0 differs from +0.0)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = {torch.bfloat16: torch.int16, torch.float32: torch.int32}.get(a.dtype)
    if view is None:
        return bool(torch.equal(a, b))
    return bool(torch.equal(a.contiguous().view(view), b.contiguous().view(view)))


def assert_all_pass(results: list[dict]) -> None:
    failed = [r for r in results if r["passed"] is False]
    assert not failed, "\n".join(
        f"{r['tensor']} [{r['rule']}]: {'; '.join(r['problems'])}" for r in failed
    )


# -------------------------------------------------- state, inputs and the loaded MoE

_STATE_CACHE: dict[int, dict] = {}


def release_cache() -> None:
    """Drop the cached state and MoE and hand their memory back to the device."""
    _STATE_CACHE.clear()
    gc.collect()
    torch.cuda.empty_cache()


def generated(seed: int) -> dict:
    """The seed's state and [129, 5120] input, generated once on the GPU."""
    if seed not in _STATE_CACHE:
        release_cache()  # one seed's 18.75 GiB of expert weights, and its MoE, at a time
        state = reference.generate_state(seed)
        problems = reference.state_problems(state, seed)
        assert not problems, problems
        _STATE_CACHE[seed] = {
            "state": state,
            "input": reference.generate_input(seed),
            "designed_state": None,
            "moe": None,
            "router": None,
        }
    return _STATE_CACHE[seed]


def case_input(seed: int, length: int) -> torch.Tensor:
    return generated(seed)["input"][:length]


def designed_router() -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 router weight [256, 5120] and correction bias [256] for the designed rows.

    Noise of at most 2^-11 everywhere but the eight designed columns. Designed token t's
    column holds (-1 + rank / 2) / 16 for each member of its set and -2 / 16 for every
    other expert, so a row with 16 there and zero at every other designed column has the
    designed logits plus noise of standard deviation about 0.02, far inside the 0.5 between
    ranks. The bias, at most 2^-7, moves no selection.
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(reference.tensor_seed(FIXTURE_SEED, "designed.gate.weight"))
    weight = (2.0 * torch.rand(256, 5120, generator=gen, device="cuda") - 1.0) * 2.0**-11
    for columns, members in ((COLUMNS_A, SET_A), (COLUMNS_B, SET_B)):
        for token, column in enumerate(columns):
            weight[:, column] = OTHER_LOGIT / SPIKE
            for expert_id, rank in zip(members, RANKS[token]):
                weight[expert_id, column] = designed_logit(rank) / SPIKE
    gen.manual_seed(reference.tensor_seed(FIXTURE_SEED, "designed.gate.e_score_correction_bias"))
    bias = (2.0 * torch.rand(256, generator=gen, device="cuda") - 1.0) * 2.0**-7
    return weight.contiguous(), bias.contiguous()


def designed_rows(columns: tuple[int, ...], name: str) -> torch.Tensor:
    """[4, 5120] bf16: unit normal noise, zero at every designed column but row t's own
    ``columns[t]``, which holds 16."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(reference.tensor_seed(FIXTURE_SEED, name))
    rows = torch.randn(len(columns), 5120, generator=gen, device="cuda")
    rows[:, list(COLUMNS_A + COLUMNS_B)] = 0.0
    rows[torch.arange(len(columns)), list(columns)] = SPIKE
    return rows.to(DTYPE)


def designed_ids(members: tuple[int, ...], token: int) -> list[int]:
    """The members by descending designed logit: the routing the design forces on a token."""
    return [expert_id for _, expert_id in sorted(zip(RANKS[token], members), reverse=True)]


def designed_state(seed: int) -> "OrderedDict[str, torch.Tensor]":
    """The seed's generated experts with the designed router."""
    entry = generated(seed)
    if entry["designed_state"] is None:
        weight, bias = designed_router()
        router = {"gate.weight": weight, "gate.e_score_correction_bias": bias}
        entry["designed_state"] = OrderedDict(entry["state"], **router)
    return entry["designed_state"]


def loaded_moe(seed: int, router: str = "generated") -> BeamMoeBf16Sm103Tp1:
    """The seed's cached MoE, loaded with the generated router or the "designed" one."""
    entry = generated(seed)
    if entry["moe"] is None:
        entry["moe"] = BeamMoeBf16Sm103Tp1(moe_config(), LAYER_IDX, device="cuda")
    if entry["router"] != router:
        entry["router"] = None  # unknown until the load completes
        entry["moe"].load_weights(entry["state"] if router == "generated" else designed_state(seed))
        entry["router"] = router
    return entry["moe"]


def cheap_weights() -> "OrderedDict[str, torch.Tensor]":
    """A valid 770-tensor mapping that allocates nothing: one expanded constant per dtype,
    so a load that copied any of it would change the generated values it replaced."""
    constants = {
        torch.float32: torch.full((), 0.5, dtype=torch.float32, device="cuda"),
        torch.bfloat16: torch.full((), 0.25, dtype=torch.bfloat16, device="cuda"),
    }
    return OrderedDict(
        (name, constants[dtype].expand(shape))
        for name, (shape, dtype) in reference.param_shapes().items()
    )


# -------------------------------------------------- op-path instrumentation


@contextlib.contextmanager
def recording(*wrappers, experts: bool = False):
    """Log every call of these entries, in order, with its nesting depth, the arguments that
    identify the path and its output; with ``experts``, also every expert the MoE asks for
    and the rows it passes."""
    calls: list[dict] = []
    depth = [0]
    saved = []
    for w in wrappers:
        cls = type(w)
        original = cls.__call__
        saved.append((cls, "__call__", original))

        def logged(self, *args, _original=original, _name=cls.__name__, **kwargs):
            tensors = [a for a in args if isinstance(a, torch.Tensor)]
            entry = {
                "op": _name,
                "depth": depth[0],
                "shapes": [list(t.shape) for t in tensors],
                "data_ptrs": [t.data_ptr() for t in tensors],
            }
            entry.update({key: kwargs[key] for key in _RECORDED_KWARGS if key in kwargs})
            calls.append(entry)
            depth[0] += 1
            try:
                out = _original(self, *args, **kwargs)
            finally:
                depth[0] -= 1
            entry["output"] = out
            return out

        cls.__call__ = logged
    if experts:
        original_expert = BeamMoeBf16Sm103Tp1._expert
        saved.append((BeamMoeBf16Sm103Tp1, "_expert", original_expert))

        def asked(self, expert_id, rows, _original=original_expert):
            calls.append({"op": "expert", "depth": depth[0], "expert_id": expert_id, "rows": rows})
            return _original(self, expert_id, rows)

        BeamMoeBf16Sm103Tp1._expert = asked
    try:
        yield calls
    finally:
        for cls, name, original in reversed(saved):
            setattr(cls, name, original)


def loggable(calls: list[dict]) -> list[dict]:
    """The recorded calls without their tensors, for the evidence record."""
    keep = ("op", "depth", "shapes", "expert_id", *_RECORDED_KWARGS)
    out = []
    for call in calls:
        entry = {key: call[key] for key in keep if key in call}
        if "rows" in call:
            entry["rows"] = list(call["rows"].shape)
        out.append(entry)
    return out


@contextlib.contextmanager
def builtin_beam_forbidden():
    """The built-in Beam modeling must not run while the component does."""
    targets = [
        (builtin_beam.BeamMoE, "forward"),
        (builtin_beam.BeamMoE, "combine"),
        (builtin_beam.BeamRouter, "forward"),
        (builtin_beam.BeamRouter, "route"),
        (builtin_beam.BeamSwiGLU, "forward"),
        (builtin_beam.BeamRMSNorm, "forward"),
        (builtin_beam.BeamAttention, "forward"),
        (builtin_beam.BeamDecoderLayer, "forward"),
    ]
    saved = [(cls, name, getattr(cls, name)) for cls, name in targets]

    def refuse(self, *args, **kwargs):
        raise AssertionError(f"the built-in {type(self).__name__} ran inside the component")

    for cls, name, _ in saved:
        setattr(cls, name, refuse)
    try:
        yield
    finally:
        for cls, name, original in reversed(saved):
            setattr(cls, name, original)


@contextlib.contextmanager
def no_dispatch():
    """Fail on any op call: rejected inputs must be rejected before a kernel runs."""
    saved = []
    for w in OPS:
        cls = type(w)
        saved.append((cls, cls.__call__))

        def refuse(self, *args, _name=cls.__name__, **kwargs):
            raise AssertionError(f"{_name} dispatched before the input was rejected")

        cls.__call__ = refuse
    try:
        yield
    finally:
        for cls, original in reversed(saved):
            cls.__call__ = original


# -------------------------------------------------- tests: configuration and construction


def test_configuration_is_the_fixed_scope() -> None:
    """One configuration, no open choices: what the component accepts is what is tested."""
    assert CASES == ((17, 1), (17, 17), (17, 129), (41, 1), (41, 17), (41, 129))
    assert moe_module.SUPPORTED_LAYER_IDX == LAYER_IDX
    for field, value in moe_module.SUPPORTED_CONFIG.items():
        assert MOE_CONFIG[field] == value, field
    assert MOE_CONFIG["mlp_layer_types"][LAYER_IDX] == "sparse"
    shapes = moe_module.weight_shapes(moe_config())
    assert list(shapes.items()) == list(reference.param_shapes().items())
    assert (reference.HIDDEN, reference.INTERMEDIATE, reference.NUM_EXPERTS) == (
        MOE_CONFIG["hidden_size"],
        MOE_CONFIG["moe_intermediate_size"],
        MOE_CONFIG["num_experts"],
    )
    assert (reference.N_GROUP, reference.TOPK_GROUP, reference.TOP_K) == (
        MOE_CONFIG["n_group"],
        MOE_CONFIG["topk_group"],
        MOE_CONFIG["num_experts_per_tok"],
    )
    assert (
        reference.ROUTED_SCALING_FACTOR
        == MOE_CONFIG["routed_scaling_factor"]
        == dispatch_combine.ROUTED_SCALING_FACTOR
    )
    assert (reference.SEEDS, reference.LENGTHS) == (SEEDS, LENGTHS)
    assert reference.MAX_TOKENS == max(LENGTHS)
    assert torch.cuda.get_device_capability() == SM, "this suite certifies sm_103 only"


def test_moe_constructs_alone() -> None:
    """The MoE is built directly from its configuration, without the model around it: a
    router and 256 experts holding 770 tensors, every expert matrix its own storage."""

    def refuse(self, *args, **kwargs):
        raise AssertionError(f"{type(self).__name__} constructed")

    with pytest.MonkeyPatch.context() as mp:
        for cls in (
            builtin_beam.BeamForCausalLM,
            builtin_beam.BeamModel,
            builtin_beam.BeamDecoderLayer,
            builtin_beam.BeamMoE,
            builtin_beam.BeamRouter,
            builtin_beam.BeamSwiGLU,
        ):
            mp.setattr(cls, "__init__", refuse)
        moe = BeamMoeBf16Sm103Tp1(moe_config(), LAYER_IDX, device="cuda")
    assert isinstance(moe.gate, BeamMoeRouterBf16Sm103Tp1)
    assert len(moe.experts) == 256
    assert all(isinstance(expert, BeamMoeExpertBf16Sm103Tp1) for expert in moe.experts)
    assert [name for name, _ in moe.named_children()] == ["gate", "experts"]
    tensors = moe.checkpoint_tensors()
    assert list(tensors) == list(reference.param_shapes())
    for name, (shape, dtype) in reference.param_shapes().items():
        t = tensors[name]
        assert tuple(t.shape) == shape and t.dtype == dtype and t.is_cuda, name
    assert tensors["gate.weight"] is moe.gate.weight
    assert tensors["gate.e_score_correction_bias"] is moe.gate.e_score_correction_bias
    for expert_id in (0, 1, 31, 32, 128, 255):
        expert = moe.experts[expert_id]
        for attr in ("gate_proj", "up_proj", "down_proj"):
            assert tensors[f"experts.{expert_id}.{attr}.weight"] is getattr(expert, attr)
    pointers = [t.data_ptr() for name, t in tensors.items() if name.startswith("experts.")]
    assert len(pointers) == 768 and len(set(pointers)) == 768
    assert "modeling_beam" not in Path(moe_module.__file__).read_text()


# -------------------------------------------------- tests: declared limits

_BAD_CONFIGS = [
    ("moe_intermediate_size", dict(moe_intermediate_size=1536), LAYER_IDX, "moe_intermediate"),
    ("routed_scaling_factor", dict(routed_scaling_factor=1.0), LAYER_IDX, "routed_scaling"),
    ("hidden_act", dict(hidden_act="gelu"), LAYER_IDX, "unsupported hidden_act"),
    ("num_experts", dict(num_experts=128), LAYER_IDX, "unsupported num_experts"),
    ("dense_block_3", dict(mlp_layer_types=["dense"] * 4), LAYER_IDX, "sparse"),
    ("block_2", {}, 2, "only the routers of sparse blocks"),
]


@pytest.mark.parametrize(
    "case,overrides,layer_idx,match", _BAD_CONFIGS, ids=[c[0] for c in _BAD_CONFIGS]
)
def test_rejects_unsupported_configuration(case, overrides, layer_idx, match) -> None:
    """Anything outside the one configuration is refused at construction."""
    with pytest.raises(ValueError, match=match):
        BeamMoeBf16Sm103Tp1(moe_config(**overrides), layer_idx, device="cuda")


def test_rejects_a_host_device() -> None:
    with pytest.raises(ValueError, match="sm_103"):
        BeamMoeBf16Sm103Tp1(moe_config(), LAYER_IDX, device="cpu")


# -------------------------------------------------- tests: loading


def test_load_weights_maps_every_expert() -> None:
    """Every checkpoint name lands in its own tensor, exactly: expert e's matrices in
    ``experts[e]`` for all 256, the router's fp32 weight in ``gate.weight`` and the
    selection bias in the router's ``e_score_correction_bias`` buffer."""
    seed = 17
    state = generated(seed)["state"]
    moe = loaded_moe(seed)
    loaded = moe.checkpoint_tensors()
    assert list(loaded) == list(state) == list(reference.param_shapes())
    wrong = []
    for name, tensor in loaded.items():
        given = state[name]
        if tensor.dtype != given.dtype or tensor.shape != given.shape:
            wrong.append(f"{name}: {tuple(tensor.shape)} {tensor.dtype}")
        elif not torch.equal(tensor, given):
            wrong.append(f"{name}: values differ")
        assert tensor.data_ptr() != given.data_ptr(), f"{name} aliases the given tensor"
    assert not wrong, wrong[:8]
    for expert_id, expert in enumerate(moe.experts):
        for attr in ("gate_proj", "up_proj", "down_proj"):
            given = state[f"experts.{expert_id}.{attr}.weight"]
            assert torch.equal(getattr(expert, attr), given), (expert_id, attr)
    router = moe.gate
    assert router.weight.dtype == torch.float32
    assert torch.equal(router.weight, state["gate.weight"])
    assert [name for name, _ in router.named_buffers()] == ["e_score_correction_bias"]
    assert [name for name, _ in router.named_parameters()] == ["weight"]
    bias = router.e_score_correction_bias
    assert bias.dtype == torch.float32
    assert torch.equal(bias, state["gate.e_score_correction_bias"])


_BAD_LOADS = [
    ("last_down_fp32", TypeError, r"experts\.255\.down_proj\.weight must be torch\.bfloat16"),
    ("last_gate_transposed", ValueError, r"experts\.255\.gate_proj\.weight must be \[2560, 5120\]"),
    ("router_weight_bf16", TypeError, r"gate\.weight must be torch\.float32"),
    ("missing_bias", ValueError, r"missing \['gate\.e_score_correction_bias'\]"),
    ("unexpected_name", ValueError, r"unexpected \['experts\.256\.gate_proj\.weight'\]"),
]


@pytest.mark.parametrize("case,error,match", _BAD_LOADS, ids=[c for c, _, _ in _BAD_LOADS])
def test_rejected_load_changes_nothing(case, error, match) -> None:
    """Every tensor is checked before any is copied: a rejected load writes no parameter or
    buffer, leaves every value as it was, and the MoE still runs."""
    seed = 17
    entry = generated(seed)
    moe = loaded_moe(seed)
    bad = cheap_weights()
    if case == "last_down_fp32":
        bad["experts.255.down_proj.weight"] = torch.zeros((), device="cuda").expand(5120, 2560)
    elif case == "last_gate_transposed":
        bad["experts.255.gate_proj.weight"] = bad["experts.255.gate_proj.weight"].t()
    elif case == "router_weight_bf16":
        bad["gate.weight"] = bad["gate.weight"].to(DTYPE)
    elif case == "missing_bias":
        del bad["gate.e_score_correction_bias"]
    elif case == "unexpected_name":
        bad["experts.256.gate_proj.weight"] = bad["experts.255.gate_proj.weight"]
    names = ("gate.weight", "gate.e_score_correction_bias") + tuple(
        f"experts.255.{n}" for n in reference.EXPERT_SHAPES
    )
    snapshots = {name: moe.checkpoint_tensors()[name].clone() for name in names}
    versions = tensor_versions(moe)
    try:
        with no_dispatch(), pytest.raises(error, match=match):
            moe.load_weights(bad)
    finally:
        written = sorted(n for n, v in tensor_versions(moe).items() if v != versions[n])
        if written:
            entry["router"] = None  # reload the generated state before the MoE is used again
    current = moe.checkpoint_tensors()
    changed = [name for name, value in snapshots.items() if not torch.equal(current[name], value)]
    assert not written, f"{len(written)} tensors written by a rejected load: {written[:8]}"
    assert not changed, changed
    out = moe(case_input(seed, 1))
    assert out.shape == (1, 5120) and bool(torch.isfinite(out).all())


_BAD_INPUTS = [
    ("fp32_input", TypeError, "must be torch.bfloat16"),
    ("fp16_input", TypeError, "must be torch.bfloat16"),
    ("batched_3d", ValueError, r"must be \[tokens, 5120\]"),
    ("width_5119", ValueError, r"must be \[tokens, 5120\]"),
    ("zero_tokens", ValueError, "at least one token"),
    ("host_input", ValueError, "MoE's device"),
    ("not_a_tensor", TypeError, "must be a torch.Tensor"),
    ("not_loaded", RuntimeError, "load_weights must be called"),
]


@pytest.mark.parametrize("case,error,match", _BAD_INPUTS, ids=[c for c, _, _ in _BAD_INPUTS])
def test_rejects_unsupported_inputs_before_dispatch(case, error, match) -> None:
    """Each declared restriction raises before any op runs and before any expert is asked
    for."""
    if case == "not_loaded":
        moe = BeamMoeBf16Sm103Tp1(moe_config(), LAYER_IDX, device="cuda")
        x = torch.randn(17, 5120, device="cuda").to(DTYPE)
    else:
        moe = loaded_moe(17)
        x = case_input(17, 17)
    bad = {
        "fp32_input": lambda: x.float(),
        "fp16_input": lambda: x.half(),
        "batched_3d": lambda: x.unsqueeze(0),
        "width_5119": lambda: x[:, :5119],
        "zero_tokens": lambda: x[:0],
        "host_input": lambda: x.cpu(),
        "not_a_tensor": lambda: x.tolist(),
        "not_loaded": lambda: x,
    }[case]()
    with recording(experts=True) as asked, no_dispatch(), pytest.raises(error, match=match):
        moe(bad)
    assert asked == [], loggable(asked)


# -------------------------------------------------- tests: the six cases


@pytest.mark.parametrize("seed,length", CASES, ids=CASE_IDS)
def test_moe_matches_reference(seed, length, ieee_fp32) -> None:
    """One case: the forward's own routing (the router op's mapping of that call) selects
    exactly the reference's groups and experts in order, its fp32 steps and the output are
    within their rules, and the call leaves its input and the parameters untouched."""
    moe = loaded_moe(seed)
    state = generated(seed)["state"]
    x = case_input(seed, length)
    x_before = x.clone()
    versions = tensor_versions(moe)
    with recording(beam_grouped_topk_router) as calls:
        out = moe(x)
    torch.cuda.synchronize()
    after = tensor_versions(moe)
    (router_call,) = calls
    route = router_call["output"]
    ref_route = reference.route(x, state["gate.weight"], state["gate.e_score_correction_bias"])
    ref_out = reference.moe(x, state)
    results = [output_contract(out, length)]
    results.append(compare("moe_output", out, ref_out, "moe_output"))
    results.extend(compare_routes(route, ref_route))
    results.append(weight_invariants(route))
    results.append(unchanged(x, x_before, versions, after))
    _write_evidence(
        f"moe-seed{seed}-T{length}",
        {
            "test": "test_moe_matches_reference",
            "case": {"seed": seed, "length": length},
            "rules": {k: [v[0], v[1], str(v[2])] for k, v in RULES.items()},
            "results": results,
            "selected_experts": sorted(set(route["ids"].reshape(-1).tolist())),
            "input_sha256": tensor_sha256(x),
            "environment": _environment(ieee_fp32),
        },
    )
    assert_all_pass(results)


# -------------------------------------------------- tests: the component path


def test_forward_runs_the_component_path(ieee_fp32) -> None:
    """One case: the router op once, then BeamDispatchCombine, inside which every selected
    expert -- ascending, once each -- receives exactly its tokens' rows in token order and
    runs its own three projections and SwiGLU; every op's guard admits its arguments and no
    built-in Beam module runs."""
    seed, length = 17, 17
    moe = loaded_moe(seed)
    x = case_input(seed, length)
    with builtin_beam_forbidden(), validating(*OPS), recording(*OPS, experts=True) as calls:
        out = moe(x)
        torch.cuda.synchronize()
    _write_evidence(
        f"path-seed{seed}-T{length}",
        {
            "test": "test_forward_runs_the_component_path",
            "case": {"seed": seed, "length": length},
            "moe": f"{type(moe).__module__}.{type(moe).__qualname__}",
            "op_calls": loggable(calls),
            "environment": _environment(ieee_fp32),
        },
    )

    assert type(moe).__module__.endswith("moe_bf16__sm_103__tp1.moe")
    assert [c["op"] for c in calls[:2]] == ["BeamGroupedTopkRouter", "BeamDispatchCombine"]
    router, dispatch = calls[:2]
    assert router["depth"] == dispatch["depth"] == 0
    assert {k: router[k] for k in ("n_group", "topk_group", "top_k", "eps")} == {
        "n_group": 8,
        "topk_group": 4,
        "top_k": 8,
        "eps": 1e-20,
    }
    assert router["shapes"] == [[length, 5120], [256, 5120], [256]]
    assert dispatch["shapes"] == [[length, 5120], [length, 8], [length, 8]]
    assert dispatch["routed_scaling_factor"] == 2.5
    assert dispatch["output"] is out
    # Everything after the dispatch entry ran inside it.
    nested = calls[2:]
    assert all(c["depth"] == 1 for c in nested), loggable(nested)
    ids = router["output"]["ids"]
    selected = sorted(set(ids.reshape(-1).tolist()))
    assert [c["op"] for c in nested] == list(EXPERT_CALLS) * len(selected), loggable(nested)
    rows_total = 0
    for i, expert_id in enumerate(selected):
        asked, gate, up, act, down = nested[len(EXPERT_CALLS) * i : len(EXPERT_CALLS) * (i + 1)]
        assert asked["expert_id"] == expert_id
        tokens = (ids == expert_id).any(dim=-1).nonzero().squeeze(-1)
        n = int(tokens.numel())
        assert n >= 1 and bitwise_equal(asked["rows"], x[tokens]), expert_id
        assert gate["shapes"] == up["shapes"] == [[n, 5120], [5120, 2560]], expert_id
        assert act["shapes"] == [[n, 2560], [n, 2560]], expert_id
        assert down["shapes"] == [[n, 2560], [2560, 5120]], expert_id
        # Expert e's own stored matrices, read in place, on the rows it was given.
        expert = moe.experts[expert_id]
        assert gate["data_ptrs"] == [asked["rows"].data_ptr(), expert.gate_proj.data_ptr()]
        assert up["data_ptrs"] == [asked["rows"].data_ptr(), expert.up_proj.data_ptr()]
        assert down["data_ptrs"][1] == expert.down_proj.data_ptr()
        rows_total += n
    assert rows_total == 8 * length
    assert out.shape == (length, 5120) and out.dtype == DTYPE and out.is_cuda
    assert torch.cuda.get_device_capability(out.device) == SM


# -------------------------------------------------- tests: designed fixtures


def test_expert_identity_fixture(ieee_fp32, monkeypatch) -> None:
    """Designed logits route four tokens to the same eight experts -- both ends of the id
    range and of four groups -- in four different orders. Every one of the eight gets all
    four rows, the other 248 are never asked for, the output matches the reference, and
    putting expert 0 in expert 255's place fails the same comparison."""
    seed = 17
    moe = loaded_moe(seed, router="designed")
    state = designed_state(seed)
    x = designed_rows(COLUMNS_A, "designed.rows_a")
    ref_route = reference.route(x, state["gate.weight"], state["gate.e_score_correction_bias"])
    ref_out = reference.moe(x, state)
    with recording(beam_grouped_topk_router, experts=True) as calls:
        out = moe(x)
    route = calls[0]["output"]
    asked = [c for c in calls if c["op"] == "expert"]
    designed = [designed_ids(SET_A, token) for token in range(len(RANKS))]
    runs = {"unchanged": compare("moe_output", out, ref_out, "moe_output")}
    # experts[255] = experts[0], as ModuleList item assignment sets it, undone below.
    monkeypatch.setattr(moe.experts, "255", moe.experts[0])
    assert moe.experts[255] is moe.experts[0]
    runs["expert_255_is_expert_0"] = compare("moe_output", moe(x), ref_out, "moe_output")
    monkeypatch.undo()
    assert moe.experts[255] is not moe.experts[0]
    runs["restored"] = compare("moe_output", moe(x), ref_out, "moe_output")
    _write_evidence(
        "fixture-expert-identity",
        {
            "test": "test_expert_identity_fixture",
            "designed_ids": designed,
            "ids": route["ids"].tolist(),
            "weights": route["weights"].tolist(),
            "expert_calls": loggable(asked),
            "runs": runs,
            "environment": _environment(ieee_fp32),
        },
    )
    assert route["ids"].tolist() == designed, route["ids"].tolist()
    assert ref_route["ids"].tolist() == designed, ref_route["ids"].tolist()
    assert torch.equal(route["groups"], ref_route["groups"])
    assert sorted(route["groups"][0].tolist()) == [0, 1, 4, 7]
    assert [c["expert_id"] for c in asked] == sorted(SET_A)
    for call in asked:
        assert bitwise_equal(call["rows"], x), call["expert_id"]
    assert runs["unchanged"]["passed"] is True, runs["unchanged"]["problems"]
    faulted = runs["expert_255_is_expert_0"]
    assert faulted["passed"] is False and faulted["violations"] > 0, faulted
    assert runs["restored"]["passed"] is True, runs["restored"]["problems"]


def test_changed_routing_between_calls(ieee_fp32) -> None:
    """Rows routed to expert set A, then rows routed to the disjoint set B, then A again:
    B's call routes as designed and asks for no expert of A, and the repeated A call
    reproduces the first one bit for bit -- nothing of one call's routing survives into
    the next."""
    seed = 17
    moe = loaded_moe(seed, router="designed")
    state = designed_state(seed)
    rows_a = designed_rows(COLUMNS_A, "designed.rows_a")
    rows_b = designed_rows(COLUMNS_B, "designed.rows_b")
    runs = []
    for x in (rows_a, rows_b, rows_a):
        with recording(beam_grouped_topk_router, experts=True) as calls:
            out = moe(x)
        runs.append(
            {
                "out": out,
                "ids": calls[0]["output"]["ids"],
                "experts": [c["expert_id"] for c in calls if c["op"] == "expert"],
                "vs_reference": compare("moe_output", out, reference.moe(x, state), "moe_output"),
            }
        )
    first, second, third = runs
    _write_evidence(
        "fixture-changed-routing",
        {
            "test": "test_changed_routing_between_calls",
            "calls": [
                {"ids": r["ids"].tolist(), "experts": r["experts"], "result": r["vs_reference"]}
                for r in runs
            ],
            "environment": _environment(ieee_fp32),
        },
    )
    assert first["ids"].tolist() == [designed_ids(SET_A, t) for t in range(len(RANKS))]
    assert second["ids"].tolist() == [designed_ids(SET_B, t) for t in range(len(RANKS))]
    assert first["experts"] == third["experts"] == sorted(SET_A)
    assert second["experts"] == sorted(SET_B)
    assert not set(second["experts"]) & set(SET_A)
    assert torch.equal(third["ids"], first["ids"])
    assert bitwise_equal(third["out"], first["out"])
    for run in runs:
        assert run["vs_reference"]["passed"] is True, run["vs_reference"]["problems"]


def test_zero_input_gives_exact_zero() -> None:
    """Zero hidden states: every score is 1/2, so every mixing weight is exactly 1/8, every
    selected expert runs on zero rows, and the output is exactly zero -- no residual, bias
    or other term reaches it."""
    moe = loaded_moe(17)
    x = torch.zeros(17, 5120, dtype=DTYPE, device="cuda")
    with recording(beam_grouped_topk_router, experts=True) as calls:
        out = moe(x)
    route = calls[0]["output"]
    asked = [c for c in calls if c["op"] == "expert"]
    assert out.shape == (17, 5120) and out.dtype == DTYPE
    assert bool((out == 0).all())
    assert bool((route["scores"] == 0.5).all())
    assert bool((route["weights"] == 0.125).all()), route["weights"]
    assert [c["expert_id"] for c in asked] == sorted(set(route["ids"].reshape(-1).tolist()))


def test_token_permutation_preserves_correspondence(ieee_fp32) -> None:
    """Reversing the tokens reverses the routing and the output: each token keeps its own
    experts and its own output row."""
    seed, length = 17, 17
    moe = loaded_moe(seed)
    state = generated(seed)["state"]
    x = case_input(seed, length)
    perm = torch.arange(length - 1, -1, -1, device="cuda")
    inverse = torch.argsort(perm)
    outputs, routes = [], []
    for rows in (x, x[perm]):
        with recording(beam_grouped_topk_router) as calls:
            outputs.append(moe(rows))
        routes.append(calls[0]["output"])
    base, permuted = outputs
    ref_out = reference.moe(x, state)
    results = [
        compare_exact("ids_back_in_token_order", routes[1]["ids"][inverse], routes[0]["ids"]),
        compare("output_back_in_token_order", permuted[inverse], base, "moe_output"),
        compare("output_vs_reference", base, ref_out, "moe_output"),
        compare("permuted_output_vs_reference", permuted, ref_out[perm], "moe_output"),
    ]
    _write_evidence(
        f"permutation-seed{seed}-T{length}",
        {
            "test": "test_token_permutation_preserves_correspondence",
            "results": results,
            "environment": _environment(ieee_fp32),
        },
    )
    assert_all_pass(results)


def test_detects_routed_scaling_faults(ieee_fp32, monkeypatch) -> None:
    """The output comparison rejects the combination scaled by 1.0 or by 2.5 twice, and
    accepts the unchanged and the restored MoE, on the same case and the same rule."""
    seed, length = 17, 17
    moe = loaded_moe(seed)
    x = case_input(seed, length)
    ref_out = reference.moe(x, generated(seed)["state"])
    runs = {"unchanged": compare("moe_output", moe(x), ref_out, "moe_output")}
    for factor in (1.0, 6.25):
        monkeypatch.setattr(dispatch_combine, "ROUTED_SCALING_FACTOR", factor)
        runs[f"scaled_by_{factor}"] = compare("moe_output", moe(x), ref_out, "moe_output")
        monkeypatch.undo()
    runs["restored"] = compare("moe_output", moe(x), ref_out, "moe_output")
    _write_evidence(
        f"fault-scaling-seed{seed}-T{length}",
        {
            "test": "test_detects_routed_scaling_faults",
            "runs": runs,
            "environment": _environment(ieee_fp32),
        },
    )
    assert dispatch_combine.ROUTED_SCALING_FACTOR == 2.5
    assert runs["unchanged"]["passed"] is True, runs["unchanged"]["problems"]
    for factor in (1.0, 6.25):
        faulted = runs[f"scaled_by_{factor}"]
        assert faulted["passed"] is False and faulted["violations"] > 0, factor
    assert runs["restored"]["passed"] is True, runs["restored"]["problems"]
