# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test support shared by the Beam GPU tests under ``models/beam/``.

Only harness code lives here: the IEEE fp32 session guard, the optional evidence writer,
result bookkeeping that judges with the rules each test file passes in, op-path
instrumentation, the one-request KV cache context of the block tests, and the generated
configuration and state of the six-block small model. No reference math lives here; each
reference stays in its own ``_*_reference.py`` or ``_*_cases.py`` module, and every rule
and tolerance stays in the test file that applies it.

Imported through ``__extra_import_path__``. Deliberately not a conftest: the CPU tests in
this tree must not pick up the CUDA probe, and nothing here is autouse. The KV cache and
attention-metadata modules are imported where they are used, so a test that needs none of
them imports nothing more than before.
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
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.overrides import TorchFunctionMode, resolve_name

import tensorrt_llm

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

#: Environment variables that substitute TF32 for fp32 GEMMs whatever torch is told.
TF32_OVERRIDES = ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")

#: CPU op names of PyTorch's SDPA paths, and CUDA kernel names of attention kernels.
SDPA_OPS = re.compile(r"scaled_dot_product|_flash_attention|_efficient_attention|_cudnn_attention")
ATTENTION_KERNEL = re.compile(r"(?i)fmha|attention")

# -------------------------------------------------- precision and environment


def assert_no_tf32_override() -> None:
    for var in TF32_OVERRIDES:
        assert var not in os.environ, (
            f"{var} is set and substitutes TF32 for fp32 GEMMs; run with `env -u {var}`"
        )


def fp32_gemm_probe() -> float:
    """Worst relative error of an fp32 GEMM against fp64; TF32 substitution shows as ~1e-3."""
    gen = torch.Generator().manual_seed(1234)
    x = torch.randn(64, 5120, generator=gen).to(torch.bfloat16).float().cuda()
    w = (torch.randn(512, 5120, generator=gen) * 5120**-0.5).float().cuda()
    y = torch.nn.functional.linear(x, w)
    ref = x.double() @ w.double().t()
    return float((y.double() - ref).abs().max()) / float(ref.pow(2).mean().sqrt())


def math_state() -> dict:
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def ieee_fp32_session(write_evidence):
    """Body of a module-scoped ``ieee_fp32`` fixture: refuse a process that substitutes TF32,
    both before the module's tests and after them; yields the session's math state."""
    assert_no_tf32_override()
    saved = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    before = fp32_gemm_probe()
    assert before <= 1e-5, f"fp32 GEMM is not IEEE fp32 (relative error {before:.3g})"
    state = {"math": math_state(), "fp32_gemm_probe_before": before}
    yield state
    after = fp32_gemm_probe()
    state.update(math_after=math_state(), fp32_gemm_probe_after=after)
    write_evidence("fp32-session", {"fixture": "ieee_fp32", "math": state})
    precision, allow_matmul, allow_cudnn = saved
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = allow_matmul
    torch.backends.cudnn.allow_tf32 = allow_cudnn
    assert after <= 1e-5, f"fp32 GEMM stopped being IEEE fp32 (relative error {after:.3g})"


def git_revision() -> str | None:
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


def environment(ieee: dict, execution, **extra) -> dict:
    """The device and software a case ran on; ``extra`` entries follow ``execution``."""
    device = torch.cuda.current_device()
    return {
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "tensorrt_llm": tensorrt_llm.__version__,
        "tensorrt_llm_file": tensorrt_llm.__file__,
        "source_revision": git_revision(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "argv": sys.argv,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "execution": dict(execution),
        **extra,
        "math": ieee,
    }


def write_evidence(env_var: str, name: str, record: dict) -> None:
    """Write ``record`` as ``<name>.json`` into the directory named by ``env_var``, if set."""
    directory = os.environ.get(env_var)
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


# -------------------------------------------------- comparison bookkeeping


def compare(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    rule: str | None,
    tolerance: tuple[float, float] | None,
    dtype: torch.dtype,
) -> dict:
    """Score one tensor against one rule; ``rule=None`` measures without judging.

    ``tolerance`` is the rule's ``(atol, rtol)`` from the calling test file and ``dtype``
    the dtype both tensors must have. An element passes when |actual - expected| <= atol +
    rtol * |expected|, both widened to fp64; no NaN or Inf is accepted.
    """
    result = {
        "tensor": name,
        "rule": rule,
        "shape": list(expected.shape),
        "dtype": str(actual.dtype),
    }
    problems = []
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
    atol, rtol = tolerance
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


def output_contract(out, shape: tuple[int, ...], dtype: torch.dtype, sm: tuple[int, int]) -> dict:
    """The forward returned a finite ``shape`` bf16 tensor on the sm_103 device."""
    problems = []
    if not isinstance(out, torch.Tensor):
        problems.append(f"returned {type(out).__name__}")
    elif tuple(out.shape) != shape or out.dtype != dtype or not out.is_cuda:
        problems.append(f"returned {tuple(out.shape)} {out.dtype} on {out.device}")
    elif torch.cuda.get_device_capability(out.device) != sm:
        problems.append(f"on {torch.cuda.get_device_name(out.device)}")
    elif not bool(torch.isfinite(out).all()):
        problems.append("non-finite values")
    return {
        "tensor": "output_contract",
        "rule": f"{list(shape)} bf16 on the sm_103 device, finite",
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


# -------------------------------------------------- weights


def cheap_weights(param_shapes) -> "OrderedDict[str, torch.Tensor]":
    """A valid mapping for ``param_shapes`` (name -> (shape, dtype)) that allocates nothing:
    one expanded constant per dtype, so a load that copied any of it would change the
    generated values it replaced."""
    constants = {
        torch.float32: torch.full((), 0.5, dtype=torch.float32, device="cuda"),
        torch.bfloat16: torch.full((), 0.25, dtype=torch.bfloat16, device="cuda"),
    }
    return OrderedDict(
        (name, constants[dtype].expand(shape)) for name, (shape, dtype) in param_shapes.items()
    )


# -------------------------------------------------- op-path instrumentation


@contextlib.contextmanager
def recording(
    *wrappers,
    recorded_kwargs: tuple[str, ...],
    experts=None,
    data_ptrs: bool = False,
    qkv: bool = False,
):
    """Log every call of these entries, in order, with its nesting depth, the arguments that
    identify the path and its output. ``experts``: an MoE class whose ``_expert`` calls are
    also logged, with the expert asked for and the rows it is passed. ``data_ptrs``: the
    positional tensors' data pointers. ``qkv``: the shape and tensor of a ``q`` keyword."""
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
            if data_ptrs:
                entry["data_ptrs"] = [t.data_ptr() for t in tensors]
            entry.update({key: kwargs[key] for key in recorded_kwargs if key in kwargs})
            if qkv and "q" in kwargs:
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
    if experts is not None:
        original_expert = experts._expert
        saved.append((experts, "_expert", original_expert))

        def asked(self, expert_id, rows, _original=original_expert):
            calls.append({"op": "expert", "depth": depth[0], "expert_id": expert_id, "rows": rows})
            return _original(self, expert_id, rows)

        experts._expert = asked
    try:
        yield calls
    finally:
        for cls, name, original in reversed(saved):
            setattr(cls, name, original)


def loggable(calls: list[dict], keep: tuple[str, ...]) -> list[dict]:
    """The recorded calls without their tensors, for the evidence record."""
    out = []
    for call in calls:
        entry = {key: call[key] for key in keep if key in call}
        if "rows" in call:
            entry["rows"] = list(call["rows"].shape)
        out.append(entry)
    return out


@contextlib.contextmanager
def builtin_forbidden(targets):
    """The built-in Beam modeling must not run while the component does: every ``(class,
    method)`` in ``targets`` raises for the duration of the block."""
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
def no_dispatch(ops):
    """Fail on any call of ``ops``: rejected inputs must be rejected before a kernel runs."""
    saved = []
    for w in ops:
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


# -------------------------------------------------- one request on a fresh KV cache


@contextlib.contextmanager
def fresh_context(
    num_tokens: int,
    cached_tokens: int,
    num_requests: int,
    *,
    max_kv_tokens: int,
    max_tokens: int,
    request_id: int,
    mgr_args: dict,
):
    """A KVCacheManagerV2 built from ``mgr_args`` and prepared TrtllmAttentionMetadata for
    ``num_requests`` new requests of ``num_tokens`` after ``cached_tokens`` cached ones."""
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    mgr = KVCacheManagerV2(
        KvCacheConfig(max_tokens=max_kv_tokens, enable_block_reuse=False),
        CacheTypeCpp.SELF,
        **mgr_args,
    )
    try:
        request_ids = [request_id + i for i in range(num_requests)]
        mgr.add_dummy_requests(request_ids, token_nums=[cached_tokens + num_tokens] * num_requests)
        md = TrtllmAttentionMetadata(
            max_num_requests=num_requests,
            max_num_tokens=max_tokens * num_requests,
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


def cache_rows(
    md: TrtllmAttentionMetadata, length: int, layer_idx: int, tokens_per_block: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """This request's K and V rows read back from the pool, [T, Hkv, D] each. The manager's
    APIs take the logical layer and translate it to the local slot themselves."""
    mgr = md.kv_cache_manager
    view = mgr.get_buffers(layer_idx, kv_layout="HND")
    view = view[:: mgr.get_layer_page_index_scale(layer_idx) // 2]
    keys, values = [], []
    for page in range((length + tokens_per_block - 1) // tokens_per_block):
        k_off = int(md.kv_cache_block_offsets[0, 0, 0, page])
        rows = min(tokens_per_block, length - page * tokens_per_block)
        keys.append(view[k_off // 2, 0, :, :rows, :].permute(1, 0, 2))
        values.append(view[k_off // 2, 1, :, :rows, :].permute(1, 0, 2))
    return torch.cat(keys), torch.cat(values)


# -------------------------------------------------- unsupported block inputs

#: The message each hidden-state or position refusal of ``_attention.check_step`` raises.
#: Metadata refusals have no pattern here: they are checked by type alone.
_BAD_INPUT_MATCH = {
    "fp16_hidden": "hidden_states must be a torch.bfloat16 tensor",
    "fp32_hidden": "hidden_states must be a torch.bfloat16 tensor",
    "hidden_5120x2": r"hidden_states must be \[1, T, 5120\]",
    "batch_2": r"hidden_states must be \[1, T, 5120\]",
    "too_long": "supported token counts per call are",
    "empty": "supported token counts per call are",
    "non_contiguous": "hidden_states must be contiguous",
    "positions_offset": "its positions must be exactly those",
    "float_positions": "position_ids must be an integer",
    "positions_shape": "position_ids must be an integer",
}


def bad_block_input(case: str, max_tokens: int, dtype: torch.dtype):
    """One unsupported call of a block at 17 tokens: ``(hidden_states, position_ids,
    fresh_context kwargs, ValueError message pattern or None)``. ``metadata_type`` keeps
    the valid inputs; the caller replaces the metadata itself."""
    from tensorrt_llm._torch.pyexecutor.resource_manager import DataType

    length = 17
    x = torch.randn(1, length, 5120, device="cuda").to(dtype)
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
        length = max_tokens + 1
        x = torch.randn(1, length, 5120, device="cuda").to(dtype)
        positions = torch.arange(length, device="cuda").view(1, length)
        context = dict(num_tokens=17)
    elif case == "empty":
        x = x[:, :0]
        positions = positions[:, :0]
    elif case == "non_contiguous":
        x = torch.randn(1, length, 2 * 5120, device="cuda").to(dtype)[..., ::2]
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
    return x, positions, context, _BAD_INPUT_MATCH.get(case)


# -------------------------------------------------- the six-block small model

#: The checkpoint's configuration reduced to its first six layers; only these fields are read.
SMALL_MODEL_CONFIG = types.MappingProxyType(
    dict(
        num_hidden_layers=6,
        hidden_size=5120,
        intermediate_size=20480,
        moe_intermediate_size=2560,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=512,
        layer_types=("sliding_attention",) * 5 + ("full_attention",),
        mlp_layer_types=("dense",) * 3 + ("sparse",) * 3,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": None,
        },
        num_experts=256,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        tie_word_embeddings=False,
        final_logit_softcapping=30.0,
        max_position_embeddings=1048576,
    )
)


def small_model_config(**overrides) -> types.SimpleNamespace:
    fields = dict(SMALL_MODEL_CONFIG)
    fields["layer_types"] = list(fields["layer_types"])
    fields["mlp_layer_types"] = list(fields["mlp_layer_types"])
    fields["rope_parameters"] = {
        k: (dict(v) if v else v) for k, v in fields["rope_parameters"].items()
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _small_model_half_width(name: str, shape: tuple, embed_name: str) -> tuple[float, float]:
    """(half width, offset) of the uniform draw for one tensor: unit-variance projections,
    embeddings of std 1, norm weights 1 +- 0.4, gate bias std 0.75, correction bias std 0.05."""
    if name == embed_name:
        return math.sqrt(3.0), 0.0
    if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
        return 0.4, 1.0
    if name.endswith("gate_bias"):
        return 0.75 * math.sqrt(3.0), 0.0
    if name.endswith("e_score_correction_bias"):
        return 0.05 * math.sqrt(3.0), 0.0
    return math.sqrt(3.0 / shape[-1]), 0.0


def generate_small_model_state(seed: int, model_module) -> "OrderedDict[str, torch.Tensor]":
    """All 2394 tensors of ``model_module.weight_schema``, each drawn from its own generator
    keyed by seed and name, fp32 then rounded once to the stored dtype."""
    state = OrderedDict()
    schema = model_module.weight_schema(small_model_config())
    for name, (shape, dtype) in schema.items():
        g = torch.Generator(device="cuda").manual_seed(zlib.crc32(f"{seed}|{name}".encode()))
        half, offset = _small_model_half_width(name, shape, model_module.EMBED)
        t = torch.rand(shape, generator=g, device="cuda", dtype=torch.float32)
        state[name] = (t.mul_(2.0 * half).sub_(half).add_(offset)).to(dtype)
    return state
