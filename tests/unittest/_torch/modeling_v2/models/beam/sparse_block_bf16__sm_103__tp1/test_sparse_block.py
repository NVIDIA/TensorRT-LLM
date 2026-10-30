# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the Beam sparse decoder block 3 component: bf16 / sm_103 / tp1.

Configuration under test, fixed below and nowhere else: block 3 of the 52-block model
(sliding attention, MoE feed-forward) at hidden 5120 with 48 q / 8 kv heads of 128, and
block 3's MoE: 256 experts of intermediate 2560, 8 per token from 4 of 8 groups, routed
scaling factor 2.5; bf16 weights and activations with the source model's fp32 steps and
an fp32 router; one sm_103 GPU; eager, batch 1, one fresh context per call on a
KVCacheManagerV2 whose layer mask selects logical layer 3 alone; generated weights;
seeds {17, 41} x lengths {1, 17, 129}.

What is compared. ``_sparse_block_reference.py`` restates the source block in plain
torch, with the MoE of ``../moe_bf16__sm_103__tp1/_moe_reference.py``, and generates its
state and input deterministically on the GPU. Routing is judged apart from numerical
error. On the component's own feed-forward input, captured from the forward, the
reference must route exactly as the component did. Against the whole reference block,
whose feed-forward input differs from the component's by rounding, every routing
decision the two take differently must sit on a near tie of the reference's own scores;
the block output is compared on the tokens routed to the same experts, and the MoE
output on the component's own feed-forward input. Each compared tensor is held to one
rule in `RULES`.

Invocation (fp32 steps need the TF32 overrides removed before torch is imported)::

    env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE -u NVIDIA_TF32_OVERRIDE python -m pytest \\
        tests/unittest/_torch/modeling_v2/models/beam/sparse_block_bf16__sm_103__tp1

Optional environment:

* ``BEAM_SPARSE_BLOCK_EVIDENCE``: a directory that receives one JSON record per test
  case: every comparison with its rule, the op calls and CUDA kernels of the forward
  where they were recorded, and the device and software it ran on; and
  ``fp32-session.json``, the fp32 GEMM checks made before and after the module's tests.

Dependencies: a TensorRT-LLM build for sm_103 with flashinfer importable and about 60 GiB
of free device memory: one seed's generated state and the loaded block hold 18.75 GiB of
expert weights each, and one test builds a second block. No model files.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import re
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
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.beam_neox_rope import (
    beam_neox_rope,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.beam_output_gate import (
    beam_output_gate,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.beam_dispatch_combine import (  # noqa: E501
    beam_dispatch_combine_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.beam_grouped_topk_router import (  # noqa: E501
    beam_grouped_topk_router,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.residual.beam_depth_residual import (
    beam_depth_residual,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.moe_bf16__sm_103__tp1.moe import (
    BeamMoeBf16Sm103Tp1,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.sparse_block_bf16__sm_103__tp1 import (  # noqa: E501
    block as block_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.sparse_block_bf16__sm_103__tp1.block import (  # noqa: E501
    BeamSparseBlockBf16Sm103Tp1,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

__extra_import_path__ = [".", "../moe_bf16__sm_103__tp1", "../../.."]
import _moe_reference as moe_reference  # noqa: E402
import _sparse_block_reference as reference  # noqa: E402
from _validating import validating  # noqa: E402

assert torch.cuda.is_available(), "the Beam sparse block requires a CUDA device"

pytestmark = pytest.mark.high_cuda_memory

# -------------------------------------------------- the fixed configuration

NUM_LAYERS = 52
LAYER_IDX = 3
_FULL_ATTENTION_LAYERS = frozenset({5, 11, 17, 23, 29, 35, 41, 47, 51})

#: The block as the checkpoint configures it: the 52-block model's layer pattern and
#: block 3's attention and MoE. Only these fields are read.
BLOCK_CONFIG = types.MappingProxyType(
    dict(
        hidden_size=5120,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=128,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=512,
        num_hidden_layers=NUM_LAYERS,
        layer_types=tuple(
            "full_attention" if i in _FULL_ATTENTION_LAYERS else "sliding_attention"
            for i in range(NUM_LAYERS)
        ),
        mlp_layer_types=("dense",) * 3 + ("sparse",) * (NUM_LAYERS - 3),
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": None,
        },
        moe_intermediate_size=2560,
        num_experts=256,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        hidden_act="silu",
    )
)
DTYPE = torch.bfloat16
SM = (10, 3)
SEEDS = (17, 41)
LENGTHS = (1, 17, 129)
CASES = tuple((seed, length) for seed in SEEDS for length in LENGTHS)
CASE_IDS = [f"seed{seed}-T{length}" for seed, length in CASES]
#: Eager, batch 1, one fresh context: no CUDA graph, no overlap scheduler, no cached prefix.
EXECUTION = types.MappingProxyType(
    dict(eager=True, batch_size=1, fresh_context=True, cuda_graph=False, overlap_scheduler=False)
)
#: The KV cache the attention op appends to: logical layer 3 of 52 selected by the layer
#: mask, bf16, 32-token pages, 512 tokens per sequence (the window), one request.
KV_CACHE = types.MappingProxyType(dict(tokens_per_block=32, max_seq_len=512, num_blocks=16))
LAYER_MASK = tuple(i == LAYER_IDX for i in range(NUM_LAYERS))
REQUEST_ID = 0

# -------------------------------------------------- comparison rules

_RTOL = 2.0**-7

#: (atol, rtol) per kind of compared bf16 tensor. An element passes when
#: |actual - reference| <= atol + rtol * |reference|, both widened to fp64; every element of
#: every compared tensor must pass, and no NaN or Inf is accepted. rtol 2^-7 is one bf16 unit
#: at any magnitude. "block_output" holds the block output on the tokens routed to the same
#: experts as in the reference block, and the two prefix-identical inputs' common rows;
#: "moe_output" holds the MoE output against the reference MoE on the same input.
RULES = types.MappingProxyType({"block_output": (0.1, _RTOL), "moe_output": (0.04, _RTOL)})
#: A routing decision the component takes differently from the whole reference block -- a
#: group kept or ordered differently, or an expert selected or ordered differently within
#: the groups both kept -- passes only on a margin of at most this in the reference's own
#: group scores or selection scores. Routing on the same input is never excused: it is
#: compared exactly.
ROUTE_GAP = 3e-3

#: The forward's top-level op calls, in order: the composition this component claims to
#: run. The experts' calls nest inside BeamDispatchCombine.
EXPECTED_CALLS = (
    "FlashinferRmsnorm",
    "CublasMm",
    "CublasMm",
    "CublasMm",
    "CublasMm",
    "FlashinferRmsnorm",
    "FlashinferRmsnorm",
    "BeamNeoxRope",
    "ThopAttention",
    "BeamOutputGate",
    "CublasMm",
    "FlashinferRmsnorm",
    "BeamDepthResidual",
    "FlashinferRmsnorm",
    "BeamGroupedTopkRouter",
    "BeamDispatchCombine",
    "FlashinferRmsnorm",
    "BeamDepthResidual",
)
#: What each selected expert runs, in order, inside BeamDispatchCombine.
EXPERT_CALLS = ("expert", "CublasMm", "CublasMm", "SwigluFp32", "CublasMm")
#: Every catalog entry the block's forward can call.
OPS = (
    flashinfer_rmsnorm,
    cublas_mm,
    beam_neox_rope,
    thop_attention,
    beam_output_gate,
    beam_depth_residual,
    beam_grouped_topk_router,
    swiglu_fp32,
    beam_dispatch_combine_op,
)
#: Keyword arguments of the entries' calls that identify the path.
_RECORDED_KWARGS = (
    "num_heads",
    "num_kv_heads",
    "head_size",
    "attention_window_size",
    "max_seq_len",
    "tokens_per_block",
    "mask_type",
    "is_fused_qkv",
    "num_contexts",
    "num_ctx_tokens",
    "use_paged_context_fmha",
    "position_embedding_type",
    "local_layer_idx",
    "quant_mode",
    "n_group",
    "topk_group",
    "top_k",
    "eps",
    "routed_scaling_factor",
)

_SDPA_OPS = re.compile(r"scaled_dot_product|_flash_attention|_efficient_attention|_cudnn_attention")
_ATTENTION_KERNEL = re.compile(r"(?i)fmha|attention")


def block_config(**overrides) -> types.SimpleNamespace:
    fields = dict(BLOCK_CONFIG)
    fields["layer_types"] = list(fields["layer_types"])
    fields["mlp_layer_types"] = list(fields["mlp_layer_types"])
    fields["rope_parameters"] = {
        k: (dict(v) if v else v) for k, v in fields["rope_parameters"].items()
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def replaced(items, index: int, value) -> list:
    """A list copy of ``items`` with one entry replaced."""
    out = list(items)
    out[index] = value
    return out


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
    """Hand the module's cached state and block back to the device when its tests are done."""
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
        "kv_cache": dict(KV_CACHE, layer_mask=[i for i, kept in enumerate(LAYER_MASK) if kept]),
        "math": ieee,
    }


def _write_evidence(name: str, record: dict) -> None:
    directory = os.environ.get("BEAM_SPARSE_BLOCK_EVIDENCE")
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
    """Score one bf16 tensor against one rule; `rule=None` measures without judging."""
    result = {
        "tensor": name,
        "rule": rule,
        "shape": list(expected.shape),
        "dtype": str(actual.dtype),
    }
    problems = []
    if tuple(actual.shape) != tuple(expected.shape):
        problems.append(f"shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    if actual.dtype != DTYPE or expected.dtype != DTYPE:
        problems.append(f"dtypes {actual.dtype}/{expected.dtype}, need {DTYPE}")
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
    atol, rtol = RULES[rule]
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


def inversion_gap(candidate: list, ref: list, scores: list, members: set) -> float:
    """Largest ``scores[y] - scores[x]`` over the pairs of ``members`` that the candidate
    ranks x before y and the reference ranks y before x; 0.0 when there is none. A rank is
    the position in the ordered selection, and a member a selection lacks ranks after all of
    it (8 for experts, 4 for groups)."""
    rank_candidate = {m: i for i, m in enumerate(candidate)}
    rank_ref = {m: i for i, m in enumerate(ref)}
    last_candidate, last_ref = len(candidate), len(ref)
    gap = 0.0
    for x in members:
        for y in members:
            before = rank_candidate.get(x, last_candidate) < rank_candidate.get(y, last_candidate)
            after = rank_ref.get(y, last_ref) < rank_ref.get(x, last_ref)
            if before and after:
                gap = max(gap, scores[y] - scores[x])
    return gap


def judge_routing(own: dict, ref: dict) -> tuple[dict, torch.Tensor]:
    """The component's routing against the whole reference block's, token by token.

    A token routed identically passes. Otherwise each decision taken differently has to be
    a near tie in the reference's own scores: groups kept or ordered differently by their
    ``group_scores``, experts selected or ordered differently by their ``selection`` scores
    among the experts of the groups both kept (an expert of a group only one side kept is
    the group decision's consequence). Returns the judgement and the mask of the tokens
    routed to the same expert set.
    """
    own_ids, ref_ids = own["ids"].tolist(), ref["ids"].tolist()
    own_groups, ref_groups = own["groups"].tolist(), ref["groups"].tolist()
    selection = ref["selection"].tolist()
    group_scores = ref["group_scores"].tolist()
    per_group = moe_reference.PER_GROUP
    differing = []
    for t, (ids, ref_t) in enumerate(zip(own_ids, ref_ids)):
        if ids == ref_t and own_groups[t] == ref_groups[t]:
            continue
        common = set(own_groups[t]) & set(ref_groups[t])
        members = {e for e in set(ids) | set(ref_t) if e // per_group in common}
        differing.append(
            {
                "token": t,
                "ids": ids,
                "reference_ids": ref_t,
                "groups": own_groups[t],
                "reference_groups": ref_groups[t],
                "group_gap": inversion_gap(
                    own_groups[t], ref_groups[t], group_scores[t], set(range(len(group_scores[t])))
                ),
                "expert_gap": inversion_gap(ids, ref_t, selection[t], members),
            }
        )
    worst = max([0.0] + [max(d["group_gap"], d["expert_gap"]) for d in differing])
    beyond = [d["token"] for d in differing if max(d["group_gap"], d["expert_gap"]) > ROUTE_GAP]
    matched = torch.tensor(
        [set(ids) == set(ref_t) for ids, ref_t in zip(own_ids, ref_ids)],
        dtype=torch.bool,
        device=own["ids"].device,
    )
    problems = []
    if beyond:
        problems.append(f"tokens {beyond[:8]} route differently by up to {worst:.3g}")
    judgement = {
        "tensor": "routing_vs_reference_block",
        "rule": f"every differing decision within {ROUTE_GAP:g} of the reference's own scores",
        "tokens": len(ref_ids),
        "differing_tokens": differing,
        "set_matched_tokens": int(matched.sum()),
        "max_gap": worst,
        "passed": not problems,
        "problems": problems,
    }
    return judgement, matched


def output_contract(out, tokens: int) -> dict:
    problems = []
    if not isinstance(out, torch.Tensor):
        problems.append(f"returned {type(out).__name__}")
    elif tuple(out.shape) != (1, tokens, 5120) or out.dtype != DTYPE or not out.is_cuda:
        problems.append(f"returned {tuple(out.shape)} {out.dtype} on {out.device}")
    elif torch.cuda.get_device_capability(out.device) != SM:
        problems.append(f"on {torch.cuda.get_device_name(out.device)}")
    elif not bool(torch.isfinite(out).all()):
        problems.append("non-finite values")
    return {
        "tensor": "output_contract",
        "rule": f"[1, {tokens}, 5120] bf16 on the sm_103 device, finite",
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


# -------------------------------------------------- state, inputs, fresh context

_STATE_CACHE: dict[int, dict] = {}


def release_cache() -> None:
    """Drop the cached state and block and hand their memory back to the device."""
    _STATE_CACHE.clear()
    gc.collect()
    torch.cuda.empty_cache()


def generated(seed: int) -> dict:
    """The seed's 782 tensors and [1, 129, 5120] input, generated once on the GPU."""
    if seed not in _STATE_CACHE:
        release_cache()  # one seed's 18.75 GiB of expert weights, and its block, at a time
        state = reference.generate_state(seed)
        problems = reference.state_problems(state, seed)
        assert not problems, problems
        _STATE_CACHE[seed] = {
            "state": state,
            "moe_state": reference.moe_state(state),
            "input": reference.generate_input(seed),
            "block": None,
            "loaded": False,
        }
    return _STATE_CACHE[seed]


def loaded_block(seed: int) -> BeamSparseBlockBf16Sm103Tp1:
    """The seed's cached block, loaded with the seed's state."""
    entry = generated(seed)
    if entry["block"] is None:
        entry["block"] = BeamSparseBlockBf16Sm103Tp1(block_config(), LAYER_IDX, device="cuda")
    if not entry["loaded"]:
        entry["block"].load_weights(entry["state"])
        entry["loaded"] = True
    return entry["block"]


def cheap_weights() -> "OrderedDict[str, torch.Tensor]":
    """A valid 782-tensor mapping that allocates nothing: one expanded constant per dtype,
    so a load that copied any of it would change the generated values it replaced."""
    constants = {
        torch.float32: torch.full((), 0.5, dtype=torch.float32, device="cuda"),
        torch.bfloat16: torch.full((), 0.25, dtype=torch.bfloat16, device="cuda"),
    }
    return OrderedDict(
        (name, constants[dtype].expand(shape))
        for name, (shape, dtype) in reference.param_shapes().items()
    )


@contextlib.contextmanager
def fresh_context(num_tokens: int, cached_tokens: int = 0, num_requests: int = 1, **mgr_overrides):
    """A KVCacheManagerV2 whose layer mask selects logical layer 3 alone, and prepared
    TrtllmAttentionMetadata for one new request."""
    mgr_args = dict(
        num_layers=1,
        layer_mask=list(LAYER_MASK),
        num_kv_heads=BLOCK_CONFIG["num_key_value_heads"],
        head_dim=BLOCK_CONFIG["head_dim"],
        tokens_per_block=KV_CACHE["tokens_per_block"],
        max_seq_len=KV_CACHE["max_seq_len"],
        max_batch_size=num_requests,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
    )
    mgr_args.update(mgr_overrides)
    mgr = KVCacheManagerV2(
        KvCacheConfig(
            max_tokens=KV_CACHE["num_blocks"] * KV_CACHE["tokens_per_block"] * num_requests,
            enable_block_reuse=False,
        ),
        CacheTypeCpp.SELF,
        **mgr_args,
    )
    try:
        request_ids = [REQUEST_ID + i for i in range(num_requests)]
        mgr.add_dummy_requests(request_ids, token_nums=[cached_tokens + num_tokens] * num_requests)
        md = TrtllmAttentionMetadata(
            max_num_requests=num_requests,
            max_num_tokens=block_module.MAX_TOKENS * num_requests,
            kv_cache_manager=mgr,
            num_contexts=num_requests,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[cached_tokens] * num_requests
            ),
            seq_lens=torch.tensor([num_tokens] * num_requests, dtype=torch.int),
            request_ids=request_ids,
            prompt_lens=[cached_tokens + num_tokens] * num_requests,
        )
        md.prepare()
        yield md
    finally:
        mgr.shutdown()


def case_inputs(seed: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = generated(seed)["input"][:, :length].contiguous()
    positions = torch.arange(length, device="cuda").view(1, length)
    return x, positions


def reference_record(seed: int, x: torch.Tensor, positions: torch.Tensor) -> dict:
    """The reference block of the seed's state on ``x``: its output and every named
    intermediate, ``[T, ...]``, on the GPU."""
    record = {}
    with torch.no_grad():
        out = reference.block(x, positions, generated(seed)["state"], record=record.__setitem__)
    assert torch.equal(out[0], record["block_output"])
    return record


def run_block(
    blk: BeamSparseBlockBf16Sm103Tp1, x: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, dict]:
    """One forward on a fresh context: the output, the MoE's input (the pre-feed-forward
    norm's output) and output from a forward hook, and the router op's mapping of the
    call."""
    seen = []

    def hook(module, args, output):
        seen.append((args[0].clone(), output.clone()))

    handle = blk.mlp.register_forward_hook(hook)
    try:
        with fresh_context(x.shape[1]) as md, recording(beam_grouped_topk_router) as calls:
            out = blk(x, positions, md)
            torch.cuda.synchronize()
    finally:
        handle.remove()
    assert len(seen) == 1 and len(calls) == 1, (len(seen), len(calls))
    ((pre_ff, moe_out),) = seen
    return out, {"pre_ff": pre_ff, "moe_out": moe_out, "routing": calls[0]["output"]}


def block_results(seed: int, out: torch.Tensor, captured: dict, ref: dict) -> list[dict]:
    """Every comparison of one forward against the reference block of the same input,
    routing judged apart from numerical error."""
    moe_state = generated(seed)["moe_state"]
    pre_ff, moe_out, routing = captured["pre_ff"], captured["moe_out"], captured["routing"]
    tokens = pre_ff.shape[0]
    results = [output_contract(out, tokens)]
    # Same input, same routing: the reference routes the component's own MoE input exactly
    # as the component did.
    same_input = moe_reference.route(
        pre_ff, moe_state["gate.weight"], moe_state["gate.e_score_correction_bias"]
    )
    results.append(compare_exact("ids_on_own_moe_input", routing["ids"], same_input["ids"]))
    results.append(
        compare_exact("groups_on_own_moe_input", routing["groups"], same_input["groups"])
    )
    # Against the whole reference block: differing decisions are near ties.
    judgement, matched = judge_routing(routing, ref["moe.routing"])
    results.append(judgement)
    final = compare(
        "block_output_on_set_matched_tokens",
        out[0][matched],
        ref["block_output"][matched],
        "block_output",
    )
    final["tokens"] = int(matched.sum())
    results.append(final)
    results.append(
        compare(
            "moe_output_on_own_moe_input",
            moe_out,
            moe_reference.moe(pre_ff, moe_state),
            "moe_output",
        )
    )
    # Measured, not judged: how far the component's MoE input is from the reference's.
    results.append(
        compare("pre_feedforward_layernorm", pre_ff, ref["pre_feedforward_layernorm"], None)
    )
    return results


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
            }
            entry.update({key: kwargs[key] for key in _RECORDED_KWARGS if key in kwargs})
            if "q" in kwargs:
                entry["q_shape"] = list(kwargs["q"].shape)
                entry["qkv"] = kwargs["q"]
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
    keep = ("op", "depth", "shapes", "q_shape", "expert_id", *_RECORDED_KWARGS)
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
        (builtin_beam.BeamDecoderLayer, "forward"),
        (builtin_beam.BeamAttention, "forward"),
        (builtin_beam.BeamMoE, "forward"),
        (builtin_beam.BeamMoE, "combine"),
        (builtin_beam.BeamRouter, "forward"),
        (builtin_beam.BeamRouter, "route"),
        (builtin_beam.BeamSwiGLU, "forward"),
        (builtin_beam.BeamRMSNorm, "forward"),
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


def cache_rows(md: TrtllmAttentionMetadata, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """This request's K and V rows read back from the pool, [T, Hkv, D] each. The manager's
    APIs take the logical layer, 3, and translate it to the local slot themselves."""
    mgr = md.kv_cache_manager
    view = mgr.get_buffers(LAYER_IDX, kv_layout="HND")
    view = view[:: mgr.get_layer_page_index_scale(LAYER_IDX) // 2]
    tpb = KV_CACHE["tokens_per_block"]
    keys, values = [], []
    for page in range((length + tpb - 1) // tpb):
        k_off = int(md.kv_cache_block_offsets[0, 0, 0, page])
        rows = min(tpb, length - page * tpb)
        keys.append(view[k_off // 2, 0, :, :rows, :].permute(1, 0, 2))
        values.append(view[k_off // 2, 1, :, :rows, :].permute(1, 0, 2))
    return torch.cat(keys), torch.cat(values)


# -------------------------------------------------- tests: configuration and construction


def test_configuration_is_the_fixed_scope() -> None:
    """One configuration, no open choices: what the component accepts is what is tested."""
    assert CASES == ((17, 1), (17, 17), (17, 129), (41, 1), (41, 17), (41, 129))
    assert block_module.SUPPORTED_LAYER_IDX == LAYER_IDX
    for field, value in block_module.SUPPORTED_CONFIG.items():
        assert BLOCK_CONFIG[field] == value, field
    assert BLOCK_CONFIG["rope_parameters"]["sliding_attention"] == block_module.SUPPORTED_ROPE
    assert BLOCK_CONFIG["layer_types"][LAYER_IDX] == "sliding_attention"
    assert BLOCK_CONFIG["mlp_layer_types"][LAYER_IDX] == "sparse"
    assert len(BLOCK_CONFIG["layer_types"]) == len(BLOCK_CONFIG["mlp_layer_types"]) == NUM_LAYERS
    assert block_module.MAX_TOKENS == 1024 and max(LENGTHS) == reference.MAX_TOKENS
    own = block_module.weight_shapes(block_config())
    assert list(own.items()) == list(reference.own_shapes().items())
    assert (reference.HEADS, reference.KV_HEADS, reference.HEAD_DIM) == (48, 8, 128)
    assert reference.WINDOW == BLOCK_CONFIG["sliding_window"] == KV_CACHE["max_seq_len"]
    assert reference.LAYER_IDX == LAYER_IDX and sum(LAYER_MASK) == 1 and LAYER_MASK[LAYER_IDX]
    assert (reference.SEEDS, reference.LENGTHS) == (SEEDS, LENGTHS)
    assert torch.cuda.get_device_capability() == SM, "this suite certifies sm_103 only"


def test_block_constructs_alone() -> None:
    """The block is built directly from its configuration, without the model around it: its
    twelve attention and norm tensors and the MoE component, no dense MLP, and block 3's
    residual coefficients."""

    def refuse(self, *args, **kwargs):
        raise AssertionError(f"{type(self).__name__} constructed")

    with pytest.MonkeyPatch.context() as mp:
        for cls in (
            builtin_beam.BeamForCausalLM,
            builtin_beam.BeamModel,
            builtin_beam.BeamDecoderLayer,
            builtin_beam.BeamAttention,
            builtin_beam.BeamMoE,
            builtin_beam.BeamRouter,
            builtin_beam.BeamSwiGLU,
        ):
            mp.setattr(cls, "__init__", refuse)
        blk = BeamSparseBlockBf16Sm103Tp1(block_config(), LAYER_IDX, device="cuda")
    assert isinstance(blk.mlp, BeamMoeBf16Sm103Tp1)
    assert [name for name, _ in blk.named_children()] == ["mlp"]
    tensors = blk.checkpoint_tensors()
    names = list(tensors)
    assert len(names) == 782
    assert names[:12] == list(reference.own_shapes())
    assert names[12:] == [reference.MLP_PREFIX + n for n in moe_reference.param_shapes()]
    assert list(tensors) == list(reference.param_shapes())
    for dense in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
        assert dense not in tensors, dense
    for name, (shape, dtype) in reference.param_shapes().items():
        t = tensors[name]
        assert tuple(t.shape) == shape and t.dtype == dtype and t.is_cuda, name
    assert tensors["mlp.gate.weight"] is blk.mlp.gate.weight
    assert tensors["mlp.experts.255.down_proj.weight"] is blk.mlp.experts[255].down_proj
    assert blk.attention_scales == (1 / math.sqrt(8), math.sqrt(7) / math.sqrt(8))
    assert blk.feed_forward_scales == (1 / 3, math.sqrt(8) / 3)
    assert blk.attention_scales != block_module.depth_scales(0, True)
    assert blk.feed_forward_scales != block_module.depth_scales(0, False)
    assert "modeling_beam" not in Path(block_module.__file__).read_text()


# -------------------------------------------------- tests: declared limits

_SLIDING_ROPE_1E6 = {
    "sliding_attention": {"rope_type": "default", "rope_theta": 1e6},
    "full_attention": None,
}
_BAD_CONFIGS = [
    (
        "dense_block_3",
        dict(mlp_layer_types=replaced(BLOCK_CONFIG["mlp_layer_types"], LAYER_IDX, "dense")),
        "sparse",
    ),
    (
        "full_attention_block_3",
        dict(layer_types=replaced(BLOCK_CONFIG["layer_types"], LAYER_IDX, "full_attention")),
        "sliding_attention",
    ),
    ("rope_theta", dict(rope_parameters=_SLIDING_ROPE_1E6), "rope parameters"),
    ("sliding_window", dict(sliding_window=1024), "unsupported sliding_window"),
    ("head_dim", dict(head_dim=64), "unsupported head_dim"),
]


@pytest.mark.parametrize("case,overrides,match", _BAD_CONFIGS, ids=[c[0] for c in _BAD_CONFIGS])
def test_rejects_unsupported_configuration(case, overrides, match) -> None:
    """Anything outside the one configuration is refused at construction."""
    with pytest.raises(ValueError, match=match):
        BeamSparseBlockBf16Sm103Tp1(block_config(**overrides), LAYER_IDX, device="cuda")


def test_rejects_other_blocks_and_devices() -> None:
    for layer_idx in (0, 2, 6):
        with pytest.raises(ValueError, match="only sparse blocks"):
            BeamSparseBlockBf16Sm103Tp1(block_config(), layer_idx, device="cuda")
    with pytest.raises(ValueError, match="sm_103"):
        BeamSparseBlockBf16Sm103Tp1(block_config(), LAYER_IDX, device="cpu")


# -------------------------------------------------- tests: loading


def test_load_weights_maps_every_tensor() -> None:
    """Exactly the 782 names, each copied exactly into its own tensor: the twelve attention
    and norm tensors into the block, the ``mlp.`` ones into the MoE's router and into expert
    e for every e."""
    seed = 17
    state = generated(seed)["state"]
    blk = loaded_block(seed)
    loaded = blk.checkpoint_tensors()
    assert list(loaded) == list(state) == list(reference.param_shapes())
    wrong = []
    for name, (shape, dtype) in reference.param_shapes().items():
        t = loaded[name]
        if tuple(t.shape) != shape or t.dtype != dtype or not t.is_cuda:
            wrong.append(f"{name}: {tuple(t.shape)} {t.dtype} on {t.device}")
        elif not torch.equal(t, state[name]):
            wrong.append(f"{name}: values differ")
        assert t.data_ptr() != state[name].data_ptr(), f"{name} aliases the given tensor"
    assert not wrong, wrong[:8]
    moe = blk.mlp
    for expert_id, expert in enumerate(moe.experts):
        for attr in ("gate_proj", "up_proj", "down_proj"):
            given = state[f"mlp.experts.{expert_id}.{attr}.weight"]
            assert torch.equal(getattr(expert, attr), given), (expert_id, attr)
    assert moe.gate.weight.dtype == moe.gate.e_score_correction_bias.dtype == torch.float32
    assert torch.equal(moe.gate.weight, state["mlp.gate.weight"])
    assert torch.equal(moe.gate.e_score_correction_bias, state["mlp.gate.e_score_correction_bias"])


_BAD_LOADS = [
    ("late_expert_shape", ValueError, r"experts\.255\.up_proj\.weight must be \[2560, 5120\]"),
    ("q_norm_fp32", TypeError, r"self_attn\.q_norm\.weight must be torch\.bfloat16"),
    ("router_weight_bf16", TypeError, r"gate\.weight must be torch\.float32"),
]


@pytest.mark.parametrize("case,error,match", _BAD_LOADS, ids=[c for c, _, _ in _BAD_LOADS])
def test_rejected_load_changes_nothing(case, error, match) -> None:
    """Every one of the 782 tensors is checked before any is copied: a mapping malformed
    late is rejected without writing any parameter or buffer, every value stays, and the
    block still runs."""
    seed = 17
    entry = generated(seed)
    blk = loaded_block(seed)
    bad = cheap_weights()
    if case == "late_expert_shape":
        bad["mlp.experts.255.up_proj.weight"] = bad["mlp.experts.255.up_proj.weight"].t()
    elif case == "q_norm_fp32":
        bad["self_attn.q_norm.weight"] = torch.zeros((), device="cuda").expand(128)
    elif case == "router_weight_bf16":
        bad["mlp.gate.weight"] = bad["mlp.gate.weight"].to(DTYPE)
    names = (
        "self_attn.q_norm.weight",
        "post_feedforward_layernorm.weight",
        "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias",
        "mlp.experts.255.up_proj.weight",
    )
    snapshots = {name: blk.checkpoint_tensors()[name].clone() for name in names}
    versions = tensor_versions(blk)
    try:
        with no_dispatch(), pytest.raises(error, match=match):
            blk.load_weights(bad)
    finally:
        written = sorted(n for n, v in tensor_versions(blk).items() if v != versions[n])
        if written:
            entry["loaded"] = False  # reload the generated state before the block runs again
    current = blk.checkpoint_tensors()
    changed = [name for name, value in snapshots.items() if not torch.equal(current[name], value)]
    assert not written, f"{len(written)} tensors written by a rejected load: {written[:8]}"
    assert not changed, changed
    x, positions = case_inputs(seed, 1)
    with fresh_context(1) as md:
        out = blk(x, positions, md)
    assert out.shape == (1, 1, 5120) and bool(torch.isfinite(out).all())


_BAD_INPUTS = [
    "fp16_hidden",
    "fp32_hidden",
    "hidden_5120x2",
    "batch_2",
    "too_long",
    "empty",
    "non_contiguous",
    "positions_offset",
    "float_positions",
    "positions_shape",
    "metadata_type",
    "cached_prefix",
    "two_requests",
    "context_length_mismatch",
    "kv_cache_heads",
    "kv_cache_dtype",
    "layer0_cache",
]


@pytest.mark.parametrize("case", _BAD_INPUTS)
def test_rejects_unsupported_inputs_before_dispatch(case) -> None:
    """Each declared restriction raises ValueError on the loaded block, and no op runs
    first -- including a cache manager that does not hold logical layer 3 (a default
    one-layer manager holds layer 0)."""
    blk = loaded_block(17)
    length = 17
    x = torch.randn(1, length, 5120, device="cuda").to(DTYPE)
    positions = torch.arange(length, device="cuda").view(1, length)
    context = dict(num_tokens=length)
    if case == "fp16_hidden":
        x = x.half()
    elif case == "fp32_hidden":
        x = x.float()
    elif case == "hidden_5120x2":
        x = torch.cat((x, x), dim=-1)
    elif case == "batch_2":
        x = torch.cat((x, x), dim=0)
    elif case == "too_long":
        length = block_module.MAX_TOKENS + 1
        x = torch.randn(1, length, 5120, device="cuda").to(DTYPE)
        positions = torch.arange(length, device="cuda").view(1, length)
        context = dict(num_tokens=17)
    elif case == "empty":
        x = x[:, :0]
        positions = positions[:, :0]
    elif case == "non_contiguous":
        x = torch.randn(1, length, 2 * 5120, device="cuda").to(DTYPE)[..., ::2]
    elif case == "positions_offset":
        positions = positions + 1
    elif case == "float_positions":
        positions = positions.float()
    elif case == "positions_shape":
        positions = positions.view(length)
    elif case == "cached_prefix":
        context = dict(num_tokens=length, cached_tokens=8)
    elif case == "two_requests":
        context = dict(num_tokens=length, num_requests=2)
    elif case == "context_length_mismatch":
        context = dict(num_tokens=length - 1)
    elif case == "kv_cache_heads":
        context = dict(num_tokens=length, num_kv_heads=4)
    elif case == "kv_cache_dtype":
        context = dict(num_tokens=length, dtype=DataType.HALF)
    elif case == "layer0_cache":
        context = dict(num_tokens=length, layer_mask=None)
    with fresh_context(**context) as md:
        if case == "metadata_type":
            md = object()
        elif case == "layer0_cache":
            assert dict(md.kv_cache_manager.layer_offsets) == {0: 0}
        with no_dispatch(), pytest.raises(ValueError):
            blk(x, positions, md)


def test_cache_slot_is_the_managers_local_index(ieee_fp32) -> None:
    """Logical layer 3 lives in slot 0 of a manager whose layer mask selects it alone: the
    attention op is called with local index 0, and the K and V it appended are read back
    through the manager under the logical index 3."""
    seed, length = 17, 17
    blk = loaded_block(seed)
    x, positions = case_inputs(seed, length)
    with fresh_context(length) as md:
        mgr = md.kv_cache_manager
        layout = {
            "manager": type(mgr).__name__,
            "num_layers": mgr.num_layers,
            "pp_layers": list(mgr.pp_layers),
            "layer_offsets": dict(mgr.layer_offsets),
            "num_local_layers": mgr.num_local_layers,
        }
        step = blk.step_args(md)
        with recording(thop_attention) as calls:
            out = blk(x, positions, md)
            torch.cuda.synchronize()
        keys, values = cache_rows(md, length)
    _write_evidence(
        f"cache-slot-seed{seed}-T{length}",
        {
            "test": "test_cache_slot_is_the_managers_local_index",
            "layout": layout,
            "step_local_layer_idx": step["local_layer_idx"],
            "op_calls": loggable(calls),
            "environment": _environment(ieee_fp32),
        },
    )
    assert isinstance(mgr, KVCacheManagerV2)
    assert layout["num_layers"] == NUM_LAYERS and layout["pp_layers"] == [LAYER_IDX]
    assert layout["layer_offsets"] == {LAYER_IDX: 0} and layout["num_local_layers"] == 1
    assert step["local_layer_idx"] == 0
    (call,) = calls
    assert {
        k: call[k]
        for k in (
            "local_layer_idx",
            "num_heads",
            "num_kv_heads",
            "head_size",
            "attention_window_size",
            "mask_type",
            "is_fused_qkv",
        )
    } == {
        "local_layer_idx": 0,
        "num_heads": 48,
        "num_kv_heads": 8,
        "head_size": 128,
        "attention_window_size": 512,
        "mask_type": 1,
        "is_fused_qkv": True,
    }
    assert call["q_shape"] == [length, (48 + 2 * 8) * 128]
    assert out.shape == (1, length, 5120) and bool(torch.isfinite(out).all())
    # The op appended exactly this call's post-RoPE K and V to logical layer 3's slot.
    qkv = call["qkv"]
    k_in = qkv[:, 48 * 128 : 56 * 128].reshape(length, 8, 128)
    v_in = qkv[:, 56 * 128 :].reshape(length, 8, 128)
    assert torch.equal(keys, k_in) and torch.equal(values, v_in)


def test_cache_slot_in_a_manager_holding_two_layers(ieee_fp32) -> None:
    """In a manager that holds logical layers 2 and 3, layer 3 is local slot 1: the attention
    op is called with that slot, the K and V are appended there, and the block output is
    bitwise the output on a manager holding layer 3 alone."""
    seed, length = 17, 17
    blk = loaded_block(seed)
    x, positions = case_inputs(seed, length)
    with fresh_context(length) as md:
        alone = blk(x, positions, md)
    with fresh_context(
        length, num_layers=2, layer_mask=[i in (2, 3) for i in range(NUM_LAYERS)]
    ) as md:
        assert dict(md.kv_cache_manager.layer_offsets) == {2: 0, 3: 1}
        assert blk.step_args(md)["local_layer_idx"] == 1
        with recording(thop_attention) as calls:
            out = blk(x, positions, md)
            torch.cuda.synchronize()
        keys, values = cache_rows(md, length)
    (call,) = calls
    assert call["local_layer_idx"] == 1
    qkv = call["qkv"]
    assert torch.equal(keys, qkv[:, 48 * 128 : 56 * 128].reshape(length, 8, 128))
    assert torch.equal(values, qkv[:, 56 * 128 :].reshape(length, 8, 128))
    assert bitwise_equal(out, alone)


# -------------------------------------------------- tests: the six cases


@pytest.mark.parametrize("seed,length", CASES, ids=CASE_IDS)
def test_block_matches_reference(seed, length, ieee_fp32) -> None:
    """One case, routing judged apart from numerical error: exact routing on the block's own
    MoE input; against the whole reference block, routing differences only on near ties and
    the output within its rule on the tokens routed alike; the MoE output within its rule
    on the block's own MoE input; input and parameters untouched."""
    blk = loaded_block(seed)
    x, positions = case_inputs(seed, length)
    x_before = x.clone()
    versions = tensor_versions(blk)
    out, captured = run_block(blk, x, positions)
    after = tensor_versions(blk)
    ref = reference_record(seed, x, positions)
    results = block_results(seed, out, captured, ref)
    results.append(unchanged(x, x_before, versions, after))
    _write_evidence(
        f"block-seed{seed}-T{length}",
        {
            "test": "test_block_matches_reference",
            "case": {"seed": seed, "length": length},
            "rules": {k: list(v) for k, v in RULES.items()},
            "route_gap": ROUTE_GAP,
            "results": results,
            "input_sha256": tensor_sha256(x),
            "environment": _environment(ieee_fp32),
        },
    )
    assert_all_pass(results)


# -------------------------------------------------- tests: the component path


def test_forward_runs_the_component_path(ieee_fp32) -> None:
    """One case: the component's own composition ran, op by op, with the experts' calls
    nested inside BeamDispatchCombine, through thop attention on this GPU, with no SDPA, no
    built-in Beam module and every op's guard admitting its arguments."""
    seed, length = 17, 17
    blk = loaded_block(seed)
    x, positions = case_inputs(seed, length)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with fresh_context(length) as md:
        with builtin_beam_forbidden(), validating(*OPS), recording(*OPS, experts=True) as calls:
            with torch.profiler.profile(activities=activities) as prof:
                out = blk(x, positions, md)
                torch.cuda.synchronize()
        manager = md.kv_cache_manager
    events = prof.events()
    kernels = [e.name for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
    cpu_ops = sorted({e.name for e in events if e.device_type == torch.autograd.DeviceType.CPU})
    names = [c["op"] for c in calls]
    record = {
        "test": "test_forward_runs_the_component_path",
        "case": {"seed": seed, "length": length},
        "block": f"{type(blk).__module__}.{type(blk).__qualname__}",
        "op_calls": loggable(calls),
        "cuda_kernels": kernels,
        "attention_kernels": [k for k in kernels if _ATTENTION_KERNEL.search(k)],
        "sdpa_cpu_ops": [n for n in cpu_ops if _SDPA_OPS.search(n)],
        "kv_cache_manager": type(manager).__name__,
        "environment": _environment(ieee_fp32),
    }
    _write_evidence(f"path-seed{seed}-T{length}", record)

    assert type(blk).__module__.endswith("sparse_block_bf16__sm_103__tp1.block")
    top = [c for c in calls if c["depth"] == 0]
    assert tuple(c["op"] for c in top) == EXPECTED_CALLS, [c["op"] for c in top]
    # Every other call ran inside BeamDispatchCombine: right after it, one level down.
    position = names.index("BeamDispatchCombine")
    inner = len(calls) - len(top)
    depths = [c["depth"] for c in calls]
    assert depths == [0] * (position + 1) + [1] * inner + [0] * (len(calls) - position - 1 - inner)
    router = calls[names.index("BeamGroupedTopkRouter")]
    ids = router["output"]["ids"]
    selected = sorted(set(ids.reshape(-1).tolist()))
    nested = calls[position + 1 : position + 1 + inner]
    assert [c["op"] for c in nested] == list(EXPERT_CALLS) * len(selected)
    assert [c["expert_id"] for c in nested if c["op"] == "expert"] == selected
    assert sum(c["rows"].shape[0] for c in nested if c["op"] == "expert") == 8 * length
    assert inner == len(EXPERT_CALLS) * len(selected)
    (call,) = [c for c in calls if c["op"] == "ThopAttention"]
    assert call["local_layer_idx"] == 0 and call["num_contexts"] == 1
    assert call["num_ctx_tokens"] == length and call["use_paged_context_fmha"] is False
    assert call["quant_mode"] == 0 and call["position_embedding_type"] == 0
    assert isinstance(manager, KVCacheManagerV2)
    assert record["attention_kernels"], "no attention kernel ran on the GPU"
    assert not record["sdpa_cpu_ops"], record["sdpa_cpu_ops"]
    assert torch.cuda.get_device_capability(x.device) == SM
    assert out.shape == (1, length, 5120) and bool(torch.isfinite(out.float()).all())


# -------------------------------------------------- tests: fixtures


def test_causal_prefix_independence(ieee_fp32) -> None:
    """Two 129-token inputs sharing their first 17 rows: those rows reach the MoE bit for
    bit alike and are routed alike, and their outputs agree within the block rule (each
    expert's batch differs, so only the expert GEMMs' row counts change)."""
    seed, length, prefix = 17, 129, 17
    blk = loaded_block(seed)
    x_a, positions = case_inputs(seed, length)
    suffix = reference.generate_input(41)[:, prefix:length]
    x_b = torch.cat((x_a[:, :prefix], suffix), dim=1).contiguous()
    assert not torch.equal(x_b[:, prefix:], x_a[:, prefix:])
    out_a, captured_a = run_block(blk, x_a, positions)
    out_b, captured_b = run_block(blk, x_b, positions)
    ids_a, ids_b = captured_a["routing"]["ids"], captured_b["routing"]["ids"]
    results = [
        compare_exact("prefix_ids", ids_b[:prefix], ids_a[:prefix]),
        {
            "tensor": "prefix_moe_input",
            "rule": "bitwise",
            "passed": bitwise_equal(captured_b["pre_ff"][:prefix], captured_a["pre_ff"][:prefix]),
            "problems": [],
        },
        compare("prefix_block_output", out_b[0, :prefix], out_a[0, :prefix], "block_output"),
    ]
    if not results[1]["passed"]:
        results[1]["problems"].append("the shared rows' MoE inputs differ")
    _write_evidence(
        f"prefix-seed{seed}-T{length}",
        {
            "test": "test_causal_prefix_independence",
            "prefix": prefix,
            "results": results,
            "suffix_ids_differ": not torch.equal(ids_b[prefix:], ids_a[prefix:]),
            "environment": _environment(ieee_fp32),
        },
    )
    assert_all_pass(results)
    assert not torch.equal(out_b[0, prefix:], out_a[0, prefix:]), "the suffix changed nothing"


def test_residual_uses_block3_coefficients(ieee_fp32, monkeypatch) -> None:
    """The residual entry with the block's coefficients is block 3's add bit for bit and
    differs from block 0's; and the block comparison rejects a block run with block 0's
    coefficients, and accepts the unchanged and the restored block."""
    seed, length = 17, 17
    blk = loaded_block(seed)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(moe_reference.tensor_seed(seed, "residual.branch"))
    branch = torch.randn(length, 5120, generator=gen, device="cuda").to(DTYPE)
    gen.manual_seed(moe_reference.tensor_seed(seed, "residual.stream"))
    stream = torch.randn(length, 5120, generator=gen, device="cuda").to(DTYPE)
    assert blk.attention_scales == reference.ATTENTION_SCALES
    assert blk.feed_forward_scales == reference.FEED_FORWARD_SCALES
    for label, scales, is_attention in (
        ("attention", blk.attention_scales, True),
        ("feed_forward", blk.feed_forward_scales, False),
    ):
        got = beam_depth_residual(branch, stream, *scales)
        expected = reference.residual_add(branch, stream, reference.depth_coefs(3, is_attention))
        block0 = reference.residual_add(branch, stream, reference.depth_coefs(0, is_attention))
        assert bitwise_equal(got, expected), label
        assert not torch.equal(got, block0), label

    x, positions = case_inputs(seed, length)
    ref = reference_record(seed, x, positions)
    runs = {}
    for label in ("unchanged", "block0_coefficients", "restored"):
        if label == "block0_coefficients":
            monkeypatch.setattr(blk, "attention_scales", block_module.depth_scales(0, True))
            monkeypatch.setattr(blk, "feed_forward_scales", block_module.depth_scales(0, False))
        elif label == "restored":
            monkeypatch.undo()
        out, captured = run_block(blk, x, positions)
        runs[label] = block_results(seed, out, captured, ref)
    _write_evidence(
        f"fault-residual-seed{seed}-T{length}",
        {
            "test": "test_residual_uses_block3_coefficients",
            "fault": "block 0's residual coefficients in both adds, test-scoped",
            "runs": runs,
            "environment": _environment(ieee_fp32),
        },
    )
    assert_all_pass(runs["unchanged"])
    failed = {r["tensor"] for r in runs["block0_coefficients"] if r["passed"] is False}
    assert failed, "block 0's coefficients passed the block comparison"
    assert failed <= {"routing_vs_reference_block", "block_output_on_set_matched_tokens"}, failed
    assert blk.attention_scales == block_module.depth_scales(LAYER_IDX, True)
    assert blk.feed_forward_scales == block_module.depth_scales(LAYER_IDX, False)
    assert_all_pass(runs["restored"])


def test_changed_routing_between_block_calls(ieee_fp32) -> None:
    """Input A, then input B, then A again, each on a fresh context: B routes differently
    from A and meets the block rules itself, and the repeated A reproduces the first one bit
    for bit -- its output, MoE input and MoE output."""
    seed, length = 17, 17
    blk = loaded_block(seed)
    x_a, positions = case_inputs(seed, length)
    x_b = reference.generate_input(41)[:, :length].contiguous()
    runs = [run_block(blk, x, positions) for x in (x_a, x_b, x_a)]
    (out_a1, cap_a1), (out_b, cap_b), (out_a2, cap_a2) = runs
    results_b = block_results(seed, out_b, cap_b, reference_record(seed, x_b, positions))
    ids_a1, ids_b, ids_a2 = [c["routing"]["ids"] for c in (cap_a1, cap_b, cap_a2)]
    differing = [int(t) for t in torch.nonzero((ids_b != ids_a1).any(dim=-1)).flatten().cpu()]
    _write_evidence(
        f"changed-routing-seed{seed}-T{length}",
        {
            "test": "test_changed_routing_between_block_calls",
            "tokens_routed_differently_in_b": differing,
            "results_b": results_b,
            "environment": _environment(ieee_fp32),
        },
    )
    assert differing, "input B routed every token exactly as input A"
    assert_all_pass(results_b)
    assert torch.equal(ids_a2, ids_a1)
    assert bitwise_equal(out_a2, out_a1)
    assert bitwise_equal(cap_a2["pre_ff"], cap_a1["pre_ff"])
    assert bitwise_equal(cap_a2["moe_out"], cap_a1["moe_out"])
