# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the Beam dense block 0 component: bf16 / sm_103 / tp1.

Configuration under test, fixed below and nowhere else: block 0 (sliding attention,
dense MLP) at hidden 5120, intermediate 20480 and 48 q / 8 kv heads of 128; bf16
weights and activations with the source model's fp32 steps; one sm_103 GPU; eager,
batch 1, one fresh context per call; generated weights; seeds {17, 41} x lengths
{1, 17, 129}.

What is compared. ``_dense_block_reference.py`` restates the source block in plain
torch and generates its state and input deterministically on the CPU. For every case
the component's final output is compared with the reference's, and every operation of
the block is replayed on the reference's own input to that operation, so a local error
cannot hide behind the sandwich norms. Each compared tensor is held to one rule in
`RULES`.

Invocation (fp32 steps need the TF32 overrides removed before torch is imported)::

    env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE -u NVIDIA_TF32_OVERRIDE \\
        python -m pytest tests/unittest/_torch/modeling_v2/models/beam/dense_block_bf16__sm_103__tp1

Optional environment:

* ``BEAM_DENSE_BLOCK_SOURCE_CAPTURES``: a directory of ``seed<S>/T<T>.pt`` files, each a
  dict of tensors captured from the source block under the reference's record names
  (plus ``input`` and ``positions``). The final output and every replay are then also
  compared against the captures, and the generated input must equal theirs bit for bit.
* ``BEAM_DENSE_BLOCK_EVIDENCE``: a directory that receives one JSON record per test case:
  every comparison with its rule, the op calls and CUDA kernels of the forward, and the
  device and software it ran on; and ``fp32-session.json``, the fp32 GEMM checks made
  before and after the module's tests.

Dependencies: a TensorRT-LLM build for sm_103 with flashinfer importable. No model files.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

import tensorrt_llm
import tensorrt_llm._torch.models.modeling_beam as builtin_beam
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
    flashinfer_silu_and_mul,
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
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.residual.beam_depth_residual import (
    beam_depth_residual,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.dense_block_bf16__sm_103__tp1 import (  # noqa: E501
    block as block_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.dense_block_bf16__sm_103__tp1.block import (  # noqa: E501
    BeamDenseBlockBf16Sm103Tp1,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
    _attention,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

__extra_import_path__ = [".", "../../.."]
import _dense_block_reference as reference  # noqa: E402
from _validating import validating  # noqa: E402

# ----------------------------------------------------------------------------- the fixed configuration

#: The block as the checkpoint configures it. Only these fields are read.
BLOCK_CONFIG = types.MappingProxyType(
    dict(
        hidden_size=5120,
        intermediate_size=20480,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=128,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=512,
        layer_types=("sliding_attention",),
        mlp_layer_types=("dense",),
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": None,
        },
    )
)
LAYER_IDX = 0
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
#: The KV cache the attention op appends to: one bf16 layer, 32-token pages, 512 tokens per
#: sequence (the window), one request.
KV_CACHE = types.MappingProxyType(dict(tokens_per_block=32, max_seq_len=512, num_blocks=16))
REQUEST_ID = 0
ROPE_THETA = 10000.0

# ----------------------------------------------------------------------------- comparison rules

_RTOL = 2.0**-7

#: (atol, rtol, max fraction of unequal elements or None) per kind of compared tensor. An
#: element passes when |a - r| <= atol + rtol * |r| with both widened to fp64; every element
#: of every tensor of every case must pass, and no NaN or Inf is accepted anywhere.
#:
#: Derivation: the largest atol each kind needed at rtol 2^-7 across valid evaluations of
#: the source block on these exact cases (other GEMM kernels, reduction orders, attention
#: backends and Torch builds) was 0 for the norms, RoPE, gate, residual adds and SiLU
#: product (with at most 5.7e-5 of elements one ulp apart, hence the 0.1% unequal bound),
#: 5.0e-6 for a single projection, 1.6e-4 for the dense SwiGLU output, 7.8e-4 for the
#: attention module and 0.0375 for the whole block. The bands below sit 100x, 4.9x, 3.8x
#: and 4.0x above those last four, and a zeroed final residual coefficient needs an atol
#: of 4.8.
RULES = types.MappingProxyType(
    {
        "rmsnorm": (1e-6, _RTOL, 1e-3),
        "rope": (1e-6, _RTOL, 1e-3),
        "gate": (1e-6, _RTOL, 1e-3),
        "residual": (1e-6, _RTOL, 1e-3),
        "swiglu_activation": (1e-6, _RTOL, 1e-3),
        "projection": (5e-4, _RTOL, None),
        "swiglu_output": (8e-4, _RTOL, None),
        "attention_output": (3e-3, _RTOL, None),
        "block_output": (0.15, _RTOL, None),
    }
)

#: The forward's op calls, in order: the composition this component claims to run.
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
    "CublasMm",
    "CublasMm",
    "FlashinferSiluAndMul",
    "CublasMm",
    "FlashinferRmsnorm",
    "BeamDepthResidual",
)
OPS = (
    flashinfer_rmsnorm,
    cublas_mm,
    beam_neox_rope,
    thop_attention,
    beam_output_gate,
    beam_depth_residual,
    flashinfer_silu_and_mul,
)

_SDPA_OPS = re.compile(r"scaled_dot_product|_flash_attention|_efficient_attention|_cudnn_attention")
_ATTENTION_KERNEL = re.compile(r"(?i)fmha|attention")

assert torch.cuda.is_available(), "the Beam dense block requires a CUDA device"


def block_config(**overrides) -> types.SimpleNamespace:
    fields = dict(BLOCK_CONFIG)
    fields["layer_types"] = list(fields["layer_types"])
    fields["mlp_layer_types"] = list(fields["mlp_layer_types"])
    fields["rope_parameters"] = {
        k: (dict(v) if v else v) for k, v in fields["rope_parameters"].items()
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


# ----------------------------------------------------------------------------- precision and environment


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
    directory = os.environ.get("BEAM_DENSE_BLOCK_EVIDENCE")
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


# ----------------------------------------------------------------------------- comparison


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor, rule: str | None) -> dict:
    """Score one tensor against one rule; `rule=None` measures without judging."""
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
    a, r = actual.double(), expected.double()
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(r).all()):
        result.update(passed=False, problems=["non-finite values"])
        return result
    diff = (a - r).abs()
    numel = diff.numel()
    worst = int(diff.argmax())
    result.update(
        numel=numel,
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        worst_index=[int(i) for i in torch.unravel_index(torch.tensor(worst), diff.shape)],
        worst_actual=float(a.reshape(-1)[worst]),
        worst_expected=float(r.reshape(-1)[worst]),
        unequal=int((a != r).sum()),
    )
    if rule is None:
        result["passed"] = None
        return result
    atol, rtol, max_unequal = RULES[rule]
    violations = int((diff > atol + rtol * r.abs()).sum())
    result.update(atol=atol, rtol=rtol, violations=violations)
    if violations:
        problems.append(
            f"{violations} of {numel} elements outside atol {atol} + rtol {rtol} * |ref|"
        )
    if max_unequal is not None:
        allowed = math.floor(max_unequal * numel)
        result["max_unequal"] = allowed
        if result["unequal"] > allowed:
            problems.append(f"{result['unequal']} unequal elements, at most {allowed} allowed")
    result.update(passed=not problems, problems=problems)
    return result


def assert_all_pass(results: list[dict]) -> None:
    failed = [r for r in results if r["passed"] is False]
    assert not failed, "\n".join(
        f"{r['tensor']} [{r['rule']}]: {'; '.join(r['problems'])}" for r in failed
    )


# ----------------------------------------------------------------------------- state, inputs, fresh context

_STATE_CACHE: dict[int, dict] = {}


def generated(seed: int) -> dict:
    """The seed's bf16 state and [1, 129, hidden] input, generated once on the CPU."""
    if seed not in _STATE_CACHE:
        _STATE_CACHE.clear()  # one seed's 0.8 GB of weights at a time
        state = reference.generate_state(seed)
        assert not reference.state_problems(state), reference.state_problems(state)
        _STATE_CACHE[seed] = {
            "state_cpu": state,
            "state": {name: t.cuda() for name, t in state.items()},
            "input": reference.generate_input(seed).cuda(),
            "block": None,
        }
    return _STATE_CACHE[seed]


def loaded_block(seed: int) -> BeamDenseBlockBf16Sm103Tp1:
    entry = generated(seed)
    if entry["block"] is None:
        blk = BeamDenseBlockBf16Sm103Tp1(block_config(), LAYER_IDX, device="cuda")
        blk.load_weights(entry["state"])
        entry["block"] = blk
    return entry["block"]


@contextlib.contextmanager
def fresh_context(num_tokens: int, cached_tokens: int = 0, num_requests: int = 1, **mgr_overrides):
    """A single-layer KVCacheManagerV2 and prepared TrtllmAttentionMetadata for one new request."""
    mgr_args = dict(
        num_layers=1,
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


def reference_record(seed: int, length: int) -> dict:
    """The reference block's output and every named intermediate, on the GPU."""
    x, positions = case_inputs(seed, length)
    record = {"input": x, "positions": positions}

    def keep(name: str, value: torch.Tensor) -> None:
        record[name] = value

    with torch.no_grad():
        out = reference.block(x, positions, generated(seed)["state"], record=keep)
    assert torch.equal(out, record["block_output"])
    return record


def source_captures(seed: int, length: int) -> dict | None:
    directory = os.environ.get("BEAM_DENSE_BLOCK_SOURCE_CAPTURES")
    if not directory:
        return None
    captures = torch.load(Path(directory) / f"seed{seed}" / f"T{length}.pt", map_location="cuda")
    x, positions = case_inputs(seed, length)
    assert torch.equal(captures["input"], x), "the generated input differs from the captured one"
    assert torch.equal(
        captures["positions"].view(1, length), positions.to(captures["positions"].dtype)
    )
    return captures


# ----------------------------------------------------------------------------- replays


def replay(
    blk: BeamDenseBlockBf16Sm103Tp1, md: TrtllmAttentionMetadata, src: dict, label: str
) -> list[dict]:
    """Each operation of the block on `src`'s own input to it, against `src`'s output of it."""
    length = src["input"].shape[1]
    hidden = BLOCK_CONFIG["hidden_size"]
    heads, kv_heads = BLOCK_CONFIG["num_attention_heads"], BLOCK_CONFIG["num_key_value_heads"]
    d = BLOCK_CONFIG["head_dim"]
    eps = BLOCK_CONFIG["rms_norm_eps"]
    w = dict(blk.checkpoint_tensors())
    positions = src["positions"].view(length).cuda()
    step = blk.step_args(md)
    # Logical block 0 is the manager's local slot 0: the core call indexes the slot.
    assert step["local_layer_idx"] == md.kv_cache_manager.layer_offsets[LAYER_IDX] == 0

    def flat(name: str, width: int) -> torch.Tensor:
        # Dense row-major: the GEMM and norm entries read their input that way.
        return src[name].reshape(length, width).contiguous()

    def check(name: str, rule: str | None, actual: torch.Tensor, expected: str) -> dict:
        result = compare(name, actual.reshape(src[expected].shape), src[expected], rule)
        result.update(reference=label)
        return result

    x = flat("input", hidden)
    h = flat("input_layernorm", hidden)
    out = [
        check(
            "input_layernorm",
            "rmsnorm",
            flashinfer_rmsnorm(x, w["input_layernorm.weight"], eps),
            "input_layernorm",
        )
    ]
    for proj in ("q_proj", "k_proj", "v_proj", "gate_proj"):
        weight = w[f"self_attn.{proj}.weight"]
        out.append(check(proj, "projection", cublas_mm(h, weight.t()), proj))
    q_rows = flat("q_proj", heads * d).reshape(length * heads, d)
    k_rows = flat("k_proj", kv_heads * d).reshape(length * kv_heads, d)
    out.append(
        check(
            "q_norm",
            "rmsnorm",
            flashinfer_rmsnorm(q_rows, w["self_attn.q_norm.weight"], eps),
            "q_norm",
        )
    )
    out.append(
        check(
            "k_norm",
            "rmsnorm",
            flashinfer_rmsnorm(k_rows, w["self_attn.k_norm.weight"], eps),
            "k_norm",
        )
    )
    q_rope, k_rope = beam_neox_rope(
        src["q_norm"].reshape(length, heads, d),
        src["k_norm"].reshape(length, kv_heads, d),
        positions,
        ROPE_THETA,
        1.0,
    )
    out.append(check("q_rope", "rope", q_rope, "q_rope"))
    out.append(check("k_rope", "rope", k_rope, "k_rope"))
    qkv = torch.cat(
        (flat("q_rope", heads * d), flat("k_rope", kv_heads * d), flat("v_proj", kv_heads * d)),
        dim=-1,
    )
    core = torch.empty(length, heads * d, dtype=DTYPE, device="cuda")
    core = thop_attention(
        q=qkv,
        output=core,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_size=d,
        attention_window_size=BLOCK_CONFIG["sliding_window"],
        **step,
        **_attention.CALL_CONSTANTS,
    )
    # No rule of its own: the attention core is judged through the attention module below.
    out.append(check("attention_core", None, core, "attention_core"))
    gated = beam_output_gate(
        src["attention_core"].reshape(length, heads, d),
        src["gate_proj"].reshape(length, heads, d),
        w["self_attn.gate_bias"],
    )
    out.append(check("gated", "gate", gated, "gated"))
    out.append(
        check(
            "o_proj",
            "projection",
            cublas_mm(flat("gated", heads * d), w["self_attn.o_proj.weight"].t()),
            "attention_output",
        )
    )
    out.append(
        check(
            "attention_output",
            "attention_output",
            blk.attention(h, positions, step),
            "attention_output",
        )
    )
    out.append(
        check(
            "post_attention_layernorm",
            "rmsnorm",
            flashinfer_rmsnorm(
                flat("attention_output", hidden), w["post_attention_layernorm.weight"], eps
            ),
            "post_attention_layernorm",
        )
    )
    out.append(
        check(
            "attention_residual",
            "residual",
            beam_depth_residual(flat("post_attention_layernorm", hidden), x, *blk.attention_scales),
            "attention_residual",
        )
    )
    out.append(
        check(
            "pre_feedforward_layernorm",
            "rmsnorm",
            flashinfer_rmsnorm(
                flat("attention_residual", hidden), w["pre_feedforward_layernorm.weight"], eps
            ),
            "pre_feedforward_layernorm",
        )
    )
    h2 = flat("pre_feedforward_layernorm", hidden)
    inter = BLOCK_CONFIG["intermediate_size"]
    out.append(
        check(
            "mlp_gate",
            "projection",
            cublas_mm(h2, w["mlp.gate_proj.weight"].t()),
            "mlp_gate",
        )
    )
    out.append(check("mlp_up", "projection", cublas_mm(h2, w["mlp.up_proj.weight"].t()), "mlp_up"))
    act = flashinfer_silu_and_mul(
        torch.cat((flat("mlp_gate", inter), flat("mlp_up", inter)), dim=-1)
    )
    out.append(check("mlp_act", "swiglu_activation", act, "mlp_act"))
    out.append(
        check(
            "down_proj",
            "projection",
            cublas_mm(flat("mlp_act", inter), w["mlp.down_proj.weight"].t()),
            "mlp_output",
        )
    )
    out.append(check("mlp_output", "swiglu_output", blk.feed_forward(h2), "mlp_output"))
    out.append(
        check(
            "post_feedforward_layernorm",
            "rmsnorm",
            flashinfer_rmsnorm(
                flat("mlp_output", hidden), w["post_feedforward_layernorm.weight"], eps
            ),
            "post_feedforward_layernorm",
        )
    )
    out.append(
        check(
            "block_residual",
            "residual",
            beam_depth_residual(
                flat("post_feedforward_layernorm", hidden),
                flat("attention_residual", hidden),
                *blk.feed_forward_scales,
            ),
            "block_output",
        )
    )
    return out


def accounting(seed: int, blk: BeamDenseBlockBf16Sm103Tp1) -> dict:
    """The fifteen tensors: the same names, order, shapes, dtypes and values on both sides."""
    state = generated(seed)["state"]
    expected = reference.param_shapes()
    loaded = blk.checkpoint_tensors()
    assert list(loaded) == list(state) == list(expected), (list(loaded), list(state))
    for name, shape in expected.items():
        assert tuple(loaded[name].shape) == shape and loaded[name].dtype == DTYPE, name
        assert torch.equal(loaded[name], state[name]), f"{name} differs from the reference's"
    assert not reference.state_problems(generated(seed)["state_cpu"])
    record = {"tensors": len(loaded), "state_problems": []}
    if os.environ.get("BEAM_DENSE_BLOCK_EVIDENCE"):
        record["sha256"] = {
            name: tensor_sha256(t) for name, t in generated(seed)["state_cpu"].items()
        }
        record["input_sha256"] = tensor_sha256(generated(seed)["input"])
    return record


# ----------------------------------------------------------------------------- op-path instrumentation


@contextlib.contextmanager
def recording(*wrappers):
    """Log every call of these entries, in order, with the arguments that identify the path."""
    calls: list[dict] = []
    saved = []
    for w in wrappers:
        cls = type(w)
        original = cls.__call__
        saved.append((cls, original))

        def logged(self, *args, _original=original, _name=cls.__name__, **kwargs):
            entry = {"op": _name}
            tensors = [a for a in args if isinstance(a, torch.Tensor)]
            entry["shapes"] = [list(t.shape) for t in tensors]
            for key in (
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
            ):
                if key in kwargs:
                    entry[key] = kwargs[key]
            if "q" in kwargs:
                entry["q_shape"] = list(kwargs["q"].shape)
                entry["qkv"] = kwargs["q"]
            calls.append(entry)
            return _original(self, *args, **kwargs)

        cls.__call__ = logged
    try:
        yield calls
    finally:
        for cls, original in reversed(saved):
            cls.__call__ = original


@contextlib.contextmanager
def builtin_beam_forbidden():
    """The built-in Beam modeling must not run while the component does."""
    targets = [
        builtin_beam.BeamDecoderLayer,
        builtin_beam.BeamAttention,
        builtin_beam.BeamSwiGLU,
        builtin_beam.BeamRMSNorm,
    ]
    saved = [(cls, cls.forward) for cls in targets]

    def refuse(self, *args, **kwargs):
        raise AssertionError(f"the built-in {type(self).__name__} ran inside the component")

    for cls in targets:
        cls.forward = refuse
    try:
        yield
    finally:
        for cls, forward in saved:
            cls.forward = forward


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
    """This request's K and V rows read back from the pool, [T, Hkv, D] each."""
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


# ----------------------------------------------------------------------------- tests: configuration and construction


def test_configuration_is_the_fixed_scope() -> None:
    """One configuration, no open choices: what the component accepts is what is tested."""
    assert CASES == ((17, 1), (17, 17), (17, 129), (41, 1), (41, 17), (41, 129))
    assert block_module.SUPPORTED_LAYER_IDX == LAYER_IDX
    for field, value in block_module.SUPPORTED_CONFIG.items():
        assert BLOCK_CONFIG[field] == value, field
    assert BLOCK_CONFIG["rope_parameters"]["sliding_attention"] == block_module.SUPPORTED_ROPE
    assert block_module.MAX_TOKENS == 1024 >= max(LENGTHS)
    assert reference.BLOCK0.hidden == BLOCK_CONFIG["hidden_size"]
    assert reference.BLOCK0.intermediate == BLOCK_CONFIG["intermediate_size"]
    assert (reference.BLOCK0.heads, reference.BLOCK0.kv_heads, reference.BLOCK0.head_dim) == (
        48,
        8,
        128,
    )
    assert reference.BLOCK0.window == BLOCK_CONFIG["sliding_window"] == KV_CACHE["max_seq_len"]
    assert (reference.SEEDS, reference.LENGTHS) == (SEEDS, LENGTHS)
    assert torch.cuda.get_device_capability() == SM, "this suite certifies sm_103 only"


def test_block_constructs_alone(ieee_fp32) -> None:
    """The block is built directly from its configuration, without the model around it."""

    def refuse(self, *args, **kwargs):
        raise AssertionError(f"{type(self).__name__} constructed")

    with pytest.MonkeyPatch.context() as mp:
        for cls in (
            builtin_beam.BeamForCausalLM,
            builtin_beam.BeamModel,
            builtin_beam.BeamDecoderLayer,
        ):
            mp.setattr(cls, "__init__", refuse)
        blk = BeamDenseBlockBf16Sm103Tp1(block_config(), LAYER_IDX, device="cuda")
    assert isinstance(blk, torch.nn.Module) and not list(blk.children())
    assert list(blk.checkpoint_tensors()) == list(reference.param_shapes())
    for name, shape in reference.param_shapes().items():
        t = blk.checkpoint_tensors()[name]
        assert tuple(t.shape) == shape and t.dtype == DTYPE and t.is_cuda, name
    q_width = BLOCK_CONFIG["num_attention_heads"] * BLOCK_CONFIG["head_dim"]
    kv_width = BLOCK_CONFIG["num_key_value_heads"] * BLOCK_CONFIG["head_dim"]
    assert (
        blk.checkpoint_tensors()["self_attn.q_proj.weight"].shape == (q_width, 5120) == (6144, 5120)
    )
    assert blk.checkpoint_tensors()["self_attn.gate_proj.weight"].shape == (6144, 5120)
    assert (
        blk.checkpoint_tensors()["self_attn.k_proj.weight"].shape
        == (kv_width, 5120)
        == (1024, 5120)
    )
    assert blk.checkpoint_tensors()["self_attn.v_proj.weight"].shape == (1024, 5120)
    assert blk.attention_scales == (1 / math.sqrt(2), 1 / math.sqrt(2))
    assert blk.feed_forward_scales == (1 / math.sqrt(3), math.sqrt(2) / math.sqrt(3))
    assert "modeling_beam" not in Path(block_module.__file__).read_text()


# ----------------------------------------------------------------------------- tests: declared limits

_BAD_CONFIGS = [
    ("hidden_size", dict(hidden_size=4096)),
    ("intermediate_size", dict(intermediate_size=18432)),
    ("num_attention_heads", dict(num_attention_heads=32)),
    ("num_key_value_heads", dict(num_key_value_heads=4)),
    ("head_dim", dict(head_dim=64)),
    ("rms_norm_eps", dict(rms_norm_eps=1e-5)),
    ("hidden_act", dict(hidden_act="gelu")),
    ("attention_bias", dict(attention_bias=True)),
    ("sliding_window", dict(sliding_window=1024)),
    ("sliding_attention layer", dict(layer_types=["full_attention"])),
    ("dense MLP", dict(mlp_layer_types=["sparse"])),
    (
        "rope parameters",
        dict(rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 1e6}}),
    ),
    (
        "rope parameters",
        dict(
            rope_parameters={
                "sliding_attention": {"rope_type": "linear", "rope_theta": 10000.0, "factor": 2.0}
            }
        ),
    ),
]


@pytest.mark.parametrize(
    "match,overrides", _BAD_CONFIGS, ids=[f"{m}-{i}" for i, (m, _) in enumerate(_BAD_CONFIGS)]
)
def test_rejects_unsupported_configuration(match, overrides) -> None:
    with pytest.raises(ValueError, match=match):
        BeamDenseBlockBf16Sm103Tp1(block_config(**overrides), LAYER_IDX, device="cuda")


def test_rejects_other_blocks_and_devices() -> None:
    for layer_idx in (3, 5):
        with pytest.raises(ValueError, match="only dense blocks"):
            BeamDenseBlockBf16Sm103Tp1(block_config(), layer_idx, device="cuda")
    with pytest.raises(ValueError, match="sm_103"):
        BeamDenseBlockBf16Sm103Tp1(block_config(), LAYER_IDX, device="cpu")


def test_weights_must_be_exactly_the_blocks() -> None:
    blk = BeamDenseBlockBf16Sm103Tp1(block_config(), LAYER_IDX, device="cuda")
    good = {
        name: torch.zeros(shape, dtype=DTYPE) for name, shape in reference.param_shapes().items()
    }
    missing = dict(good)
    del missing["self_attn.gate_bias"]
    extra = dict(good, **{"mlp.experts.0.down_proj.weight": torch.zeros(4, dtype=DTYPE)})
    transposed = dict(good, **{"self_attn.q_proj.weight": torch.zeros(5120, 6144, dtype=DTYPE)})
    fp32 = dict(good, **{"self_attn.q_norm.weight": torch.zeros(128, dtype=torch.float32)})
    for weights, match in (
        (missing, "missing"),
        (extra, "unexpected"),
        (transposed, "q_proj"),
        (fp32, "q_norm"),
    ):
        with pytest.raises(ValueError, match=match):
            blk.load_weights(weights)


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
]


@pytest.mark.parametrize("case", _BAD_INPUTS)
def test_rejects_unsupported_inputs_before_dispatch(case) -> None:
    """Each declared restriction raises ValueError, and no op runs first."""
    blk = BeamDenseBlockBf16Sm103Tp1(block_config(), LAYER_IDX, device="cuda")
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
    with fresh_context(**context) as md:
        if case == "metadata_type":
            md = object()
        with no_dispatch(), pytest.raises(ValueError):
            blk(x, positions, md)


# ----------------------------------------------------------------------------- tests: the six cases


@pytest.mark.parametrize("seed,length", CASES, ids=CASE_IDS)
def test_block_matches_reference(seed, length, ieee_fp32) -> None:
    """One case: the whole block, then every operation on its own input."""
    blk = loaded_block(seed)
    accounting_record = accounting(seed, blk)
    ref = reference_record(seed, length)
    x, positions = case_inputs(seed, length)
    results = []
    with fresh_context(length) as md:
        out = blk(x, positions, md)
        torch.cuda.synchronize()
        assert out.shape == (1, length, 5120) and out.dtype == DTYPE
        final = compare("block_output", out, ref["block_output"], "block_output")
        final.update(reference="reference")
        results.append(final)
        results.extend(replay(blk, md, ref, "reference"))
        captures = source_captures(seed, length)
        if captures is not None:
            from_source = compare("block_output", out, captures["block_output"], "block_output")
            from_source.update(reference="source_captures")
            results.append(from_source)
            results.extend(replay(blk, md, captures, "source_captures"))
    _write_evidence(
        f"block-seed{seed}-T{length}",
        {
            "test": "test_block_matches_reference",
            "case": {"seed": seed, "length": length},
            "accounting": accounting_record,
            "rules": {k: list(v) for k, v in RULES.items()},
            "results": results,
            "environment": _environment(ieee_fp32),
        },
    )
    assert_all_pass(results)


@pytest.mark.parametrize("seed,length", CASES, ids=CASE_IDS)
def test_forward_runs_the_component_path(seed, length, ieee_fp32) -> None:
    """One case: the component's own composition ran, op by op, through thop attention on
    this GPU, with no SDPA and no built-in Beam module on the way."""
    blk = loaded_block(seed)
    x, positions = case_inputs(seed, length)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with fresh_context(length) as md:
        with builtin_beam_forbidden(), validating(*OPS), recording(*OPS) as calls:
            with torch.profiler.profile(activities=activities) as prof:
                out = blk(x, positions, md)
                torch.cuda.synchronize()
        keys, values = cache_rows(md, length)
        mgr = md.kv_cache_manager
        cache_mapping = {
            "layer_offsets": dict(mgr.layer_offsets),
            "num_local_layers": mgr.num_local_layers,
            "num_layers": mgr.num_layers,
        }
    events = prof.events()
    kernels = [e.name for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
    cpu_ops = sorted({e.name for e in events if e.device_type == torch.autograd.DeviceType.CPU})
    names = [c["op"] for c in calls]
    attention = [c for c in calls if c["op"] == "ThopAttention"]
    qkv = attention[0].pop("qkv") if attention else None
    for c in calls:
        c.pop("qkv", None)
    record = {
        "test": "test_forward_runs_the_component_path",
        "case": {"seed": seed, "length": length},
        "block": f"{type(blk).__module__}.{type(blk).__qualname__}",
        "op_calls": calls,
        "cuda_kernels": kernels,
        "attention_kernels": [k for k in kernels if _ATTENTION_KERNEL.search(k)],
        "sdpa_cpu_ops": [n for n in cpu_ops if _SDPA_OPS.search(n)],
        "kv_cache": dict(KV_CACHE, manager=type(md.kv_cache_manager).__name__, **cache_mapping),
        "environment": _environment(ieee_fp32),
    }
    _write_evidence(f"path-seed{seed}-T{length}", record)

    assert type(blk).__module__.endswith("dense_block_bf16__sm_103__tp1.block")
    assert tuple(names) == EXPECTED_CALLS, names
    (call,) = attention
    assert call["q_shape"] == [length, (48 + 2 * 8) * 128]
    assert {
        k: call[k] for k in ("num_heads", "num_kv_heads", "head_size", "attention_window_size")
    } == {
        "num_heads": 48,
        "num_kv_heads": 8,
        "head_size": 128,
        "attention_window_size": 512,
    }
    assert call["mask_type"] == 1 and call["is_fused_qkv"] and call["position_embedding_type"] == 0
    assert call["num_contexts"] == 1 and call["num_ctx_tokens"] == length
    assert call["use_paged_context_fmha"] is False and call["quant_mode"] == 0
    assert isinstance(md.kv_cache_manager, KVCacheManagerV2)
    # Logical block 0 in a one-layer manager: slot 0, which the core call indexes.
    assert cache_mapping == {"layer_offsets": {0: 0}, "num_local_layers": 1, "num_layers": 1}
    assert call["local_layer_idx"] == 0
    assert record["attention_kernels"], "no attention kernel ran on the GPU"
    assert not record["sdpa_cpu_ops"], record["sdpa_cpu_ops"]
    assert torch.cuda.get_device_capability(x.device) == SM
    assert out.shape == (1, length, 5120) and bool(torch.isfinite(out.float()).all())
    # The op appended exactly this call's post-RoPE K and V to the pool.
    k_in = qkv[:, 48 * 128 : 56 * 128].reshape(length, 8, 128)
    v_in = qkv[:, 56 * 128 :].reshape(length, 8, 128)
    assert torch.equal(keys, k_in) and torch.equal(values, v_in)


def test_detects_a_wrong_residual_coefficient(ieee_fp32, monkeypatch) -> None:
    """The complete-block comparison rejects a zeroed final residual coefficient and
    accepts the unchanged block, on the same case and the same comparison."""
    seed, length = 17, 17
    blk = loaded_block(seed)
    ref = reference_record(seed, length)
    x, positions = case_inputs(seed, length)
    runs = {}
    for label in ("unchanged", "faulted", "restored"):
        if label == "faulted":
            branch_scale, _ = blk.feed_forward_scales
            monkeypatch.setattr(blk, "feed_forward_scales", (branch_scale, 0.0))
        elif label == "restored":
            monkeypatch.undo()
        with fresh_context(length) as md:
            out = blk(x, positions, md)
        runs[label] = compare("block_output", out, ref["block_output"], "block_output")
        runs[label]["feed_forward_scales"] = list(blk.feed_forward_scales)
    _write_evidence(
        f"fault-seed{seed}-T{length}",
        {
            "test": "test_detects_a_wrong_residual_coefficient",
            "case": {"seed": seed, "length": length},
            "fault": "final residual coefficient sqrt(2)/sqrt(3) -> 0.0, fixture-scoped",
            "runs": runs,
            "environment": _environment(ieee_fp32),
        },
    )
    assert runs["unchanged"]["passed"] is True, runs["unchanged"]["problems"]
    assert runs["faulted"]["passed"] is False and runs["faulted"]["violations"] > 0
    assert runs["restored"][
        "passed"
    ] is True and blk.feed_forward_scales == block_module.depth_scales(0, False)
