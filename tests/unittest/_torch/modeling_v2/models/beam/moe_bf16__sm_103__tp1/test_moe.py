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

import functools
import gc
import types
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

try:
    import tensorrt_llm._torch.models.modeling_beam as builtin_beam
except ModuleNotFoundError as exc:
    if exc.name != "tensorrt_llm._torch.models.modeling_beam":
        raise
    builtin_beam = None  # The standard-serving implementation is a separate PR.
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.models.beam import dispatch_combine  # noqa: E501
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
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.ops.activation.swiglu_fp32 import (  # noqa: E501
    swiglu_fp32,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.ops.moe.beam_dispatch_combine import (  # noqa: E501
    beam_dispatch_combine_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.ops.moe.beam_grouped_topk_router import (  # noqa: E501
    beam_grouped_topk_router,
)

__extra_import_path__ = [".", "..", "../../.."]
import _beam_component_support as support  # noqa: E402
import _moe_reference as reference  # noqa: E402
from _beam_component_support import (  # noqa: E402
    assert_all_pass,
    bitwise_equal,
    compare_exact,
    tensor_sha256,
    tensor_versions,
    unchanged,
)
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


@pytest.fixture(scope="module", autouse=True)
def ieee_fp32():
    """The source's fp32 steps are IEEE fp32: refuse a process that substitutes TF32, both
    before the module's tests and after them."""
    yield from support.ieee_fp32_session(_write_evidence)


@pytest.fixture(scope="module", autouse=True)
def device_memory():
    """Hand the module's cached state and MoE back to the device when its tests are done."""
    yield
    release_cache()


def _environment(ieee: dict) -> dict:
    return support.environment(ieee, EXECUTION)


_write_evidence = functools.partial(support.write_evidence, "BEAM_MOE_EVIDENCE")


# -------------------------------------------------- comparison


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor, rule: str | None) -> dict:
    """Score one tensor against one rule; `rule=None` measures without judging."""
    if rule is None:
        return support.compare(name, actual, expected, None, None, expected.dtype)
    atol, rtol, dtype = RULES[rule]
    return support.compare(name, actual, expected, rule, (atol, rtol), dtype)


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
    return support.output_contract(out, (tokens, 5120), DTYPE, SM)


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
    """A valid 770-tensor mapping that allocates nothing (see support.cheap_weights)."""
    return support.cheap_weights(reference.param_shapes())


# -------------------------------------------------- op-path instrumentation


def recording(*wrappers, experts: bool = False):
    """Log every call of these entries, in order, with its nesting depth, the arguments that
    identify the path, its positional tensors' data pointers and its output; with
    ``experts``, also every expert the MoE asks for and the rows it passes."""
    return support.recording(
        *wrappers,
        recorded_kwargs=_RECORDED_KWARGS,
        experts=BeamMoeBf16Sm103Tp1 if experts else None,
        data_ptrs=True,
    )


def loggable(calls: list[dict]) -> list[dict]:
    """The recorded calls without their tensors, for the evidence record."""
    return support.loggable(calls, ("op", "depth", "shapes", "expert_id", *_RECORDED_KWARGS))


def builtin_beam_forbidden():
    """The built-in Beam modeling must not run while the component does."""
    if builtin_beam is None:
        return support.builtin_forbidden([])
    return support.builtin_forbidden(
        [
            (builtin_beam.BeamMoE, "forward"),
            (builtin_beam.BeamMoE, "combine"),
            (builtin_beam.BeamRouter, "forward"),
            (builtin_beam.BeamRouter, "route"),
            (builtin_beam.BeamSwiGLU, "forward"),
            (builtin_beam.BeamRMSNorm, "forward"),
            (builtin_beam.BeamAttention, "forward"),
            (builtin_beam.BeamDecoderLayer, "forward"),
        ]
    )


def no_dispatch():
    """Fail on any op call: rejected inputs must be rejected before a kernel runs."""
    return support.no_dispatch(OPS)


# -------------------------------------------------- tests: configuration and construction


def test_moe_constructs_alone() -> None:
    """The MoE is built directly from its configuration, without the model around it: a
    router and 256 experts holding 770 tensors, every expert matrix its own storage."""

    def refuse(self, *args, **kwargs):
        raise AssertionError(f"{type(self).__name__} constructed")

    with pytest.MonkeyPatch.context() as mp:
        if builtin_beam is not None:
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
