# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the kernels of the drafter PDL chain (``pdl_conv``, ``pdl_merge``, ``pdl_silu``).

* ``drafter_silu_and_mul`` (``pdl_silu``): every cell bitwise ``torch.ops.trtllm.silu_and_mul``
  (the built-in ``swiglu`` the drafter's ``GatedMLP`` calls), eagerly and under
  CUDA-graph replay.
* The memory order of the three entry-triggering kernels -- the fused block
  convolution with ``entry_trigger``, the window-attention merge with
  ``merge_entry_trigger`` and the silu copy -- in their generated code:
  ``griddepcontrol.launch_dependents`` before ``griddepcontrol.wait`` before
  every global load in the PTX, and in the SASS (``_sass_wait_audit``) a
  wait on every path to an exit, one early trigger, no global access before
  the wait.
* Each in a delayed chain: a predecessor that releases its dependents at entry
  and writes the kernel's input late, the kernel under test, then the real
  ``drafter_fixed_gemms`` split-K GEMM that follows it in the drafter; every output bit equal
  to the same chain without programmatic dependent launch, eagerly and in a
  captured graph whose programmatic edges are checked.
* Feature off: with ``MODELING_V2_PDL_CHAIN_BASE_TREE`` naming the base tree's target
  folder (``.../muse_spark_open_816b_a42b_nvfp4__sm_103__tep``), the default
  builds of the convolution and the merge are compared with the base tree's:
  the PTX without line information, and the SASS.
"""

from __future__ import annotations

import importlib.util
import os
import re
import tempfile

import pytest
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.gemm.fixed_low_m_gemm import (  # noqa: E501
    fixed_low_m_gemm,
    tactic_for,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec import (  # noqa: E501
    dflash2_block_conv as conv_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec import (  # noqa: E501
    dflash_window_attention as attention_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec import (  # noqa: E501
    drafter_silu_and_mul as silu_module,
)
from tensorrt_llm._torch.modules.swiglu import swiglu

__extra_import_path__ = ["../../../../..", "../.."]
from _graph_edges import programmatic_edges  # noqa: E402
from _sass_wait_audit import audit, disassemble, parse  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "the drafter PDL kernels require a CUDA device"

op = silu_module.drafter_silu_and_mul
HIDDEN = 8192
DELAY_NS = 200_000


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def _assert_same_bits(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert got.shape == want.shape and got.dtype == want.dtype, what
    differ = int((_bits(got) != _bits(want)).sum())
    assert differ == 0, f"{what}: {differ} values differ"


# --------------------------------------------------------------------------
# silu copy: the built-in bits
# --------------------------------------------------------------------------
def _silu_input(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dtype = spec.get("dtype", torch.bfloat16)
    x = (3.0 * torch.randn(spec["rows"], 2 * spec["d"], generator=g, device="cuda")).to(dtype)
    if spec.get("specials") and x.numel():
        flat = x.view(-1)
        n = flat.numel()
        picks = torch.randperm(n, generator=g, device="cuda")[:64]
        values = torch.tensor(
            [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, 1e4, -1e4, 88.0],
            device="cuda",
        ).to(dtype)
        flat[picks] = values.repeat(8)
    return x


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
@torch.inference_mode()
def test_silu_cells_are_the_builtin_bits(cell, certifying):  # noqa: F811
    certifying(op)
    x = _silu_input(cell.spec, 7)
    before = x.clone()
    out = op(x, cell.spec["entry_trigger"])
    assert torch.equal(_bits(before), _bits(x)), "input mutated"
    want = swiglu(x)
    assert out.shape == want.shape and out.dtype == want.dtype
    assert torch.equal(out.isnan(), want.isnan())
    _assert_same_bits(
        torch.where(out.isnan(), 0, out), torch.where(want.isnan(), 0, want), "silu copy"
    )
    # The op's fp32 reference: the kernel's sigmoid goes through the fast exp, the
    # reference's through torch's, then one rounding each -- a few output ulp at
    # most, against each row's largest magnitude; compared where both are finite.
    ref = op.reference(x, cell.spec["entry_trigger"])
    finite = torch.isfinite(out) & torch.isfinite(ref)
    if finite.any():
        assert_within_ulp(out[finite].view(1, -1), ref[finite].view(1, -1), 4.0, 2.0)


@torch.inference_mode()
def test_silu_graph_replay_follows_new_inputs():
    x = _silu_input(dict(rows=8, d=3584), 1)
    op(x, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x, True)
    for seed in range(2, 8):
        x.copy_(_silu_input(dict(rows=8, d=3584), seed))
        graph.replay()
        torch.cuda.synchronize()
        _assert_same_bits(out, swiglu(x), f"replay seed {seed}")


# --------------------------------------------------------------------------
# Generated code: the memory order
# --------------------------------------------------------------------------
def _conv_operands(rows=8, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    projection = (0.05 * torch.randn(rows, 2, 2, HIDDEN // 16, generator=g, device="cuda")).to(
        torch.bfloat16
    )
    base = (0.3 * torch.randn(2, HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    return x, projection[:, 0], base


def _compiled_conv(entry: bool, module=conv_module):
    x, delta, base = _conv_operands()
    out = torch.empty_like(x)
    return module._dflash2_block_conv_kernel[(x.shape[0], triton.cdiv(HIDDEN, 1024))](
        x,
        delta,
        base,
        out,
        HIDDEN,
        8,
        x.stride(0),
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        base.stride(0),
        out.stride(0),
        TAPS=2,
        GROUP_SIZE=16,
        ROUNDING=1,
        BLOCK=1024,
        **({"ENTRY_TRIGGER": True} if entry else {}),
        num_warps=4,
        enable_fp_fusion=False,
        launch_pdl=entry,
    )


def _attention_operands(seed=0, ctx=100_770, capacity=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    capacity = capacity or ctx + 64
    k = torch.randn(2, capacity, 1, 128, generator=g, device="cuda").to(torch.bfloat16)
    v = torch.randn(2, capacity, 1, 128, generator=g, device="cuda").to(torch.bfloat16)
    q = torch.randn(1, 8, 8, 128, generator=g, device="cuda").to(torch.bfloat16)
    seqlens = torch.tensor([ctx + 8], dtype=torch.int32, device="cuda")
    slots = torch.tensor([1], dtype=torch.int32, device="cuda")
    return q, k, v, seqlens, slots


def _compiled_merge(entry: bool, module=attention_module):
    """The merge kernel as the served call compiles it (65 splits of 32 keys, 64 rows)."""
    num_splits, tile_rows, d = 65, 64, 128
    o_part = torch.zeros(num_splits, 1, tile_rows, d, dtype=torch.float32, device="cuda")
    lse_part = torch.zeros(num_splits, 1, tile_rows, dtype=torch.float32, device="cuda")
    seqlens = torch.tensor([100_778], dtype=torch.int32, device="cuda")
    out = torch.empty(1, 8, 8, d, dtype=torch.bfloat16, device="cuda")
    return module._window_attention_merge_kernel[(1, tile_rows)](
        o_part,
        lse_part,
        seqlens,
        out,
        1,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        M=8,
        GROUP=8,
        HKV=1,
        D=d,
        SPLITS_P2=128,
        SPLIT_KEYS=32,
        WINDOW_LEFT=2047,
        LAUNCH_WITH_PDL=True,
        **({"ENTRY_TRIGGER": True} if entry else {}),
        num_warps=4,
        launch_pdl=True,
    )


def _compiled_silu(entry: bool):
    x = _silu_input(dict(rows=8, d=3584), 0)
    out = torch.empty(8, 3584, dtype=x.dtype, device="cuda")
    return silu_module._silu_and_mul_pdl_kernel[(8, 4)](
        out,
        out.stride(0),
        x,
        x.stride(0),
        3584,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        BLOCK_SIZE=1024,
        ENTRY_TRIGGER=entry,
        num_warps=4,
        launch_pdl=entry,
    )


_ENTRY_KERNELS = {
    "conv": lambda: _compiled_conv(True),
    "merge": lambda: _compiled_merge(True),
    "silu": lambda: _compiled_silu(True),
}


@pytest.mark.parametrize("name", sorted(_ENTRY_KERNELS))
def test_ptx_triggers_then_waits_before_every_global_access(name):
    kernel = _ENTRY_KERNELS[name]()
    torch.cuda.synchronize()
    lines = [line.strip() for line in kernel.asm["ptx"].splitlines()]
    trig = [i for i, s in enumerate(lines) if s.startswith("griddepcontrol.launch_dependents")]
    wait = [i for i, s in enumerate(lines) if s.startswith("griddepcontrol.wait")]
    loads = [i for i, s in enumerate(lines) if re.match(r"(@!?%p\d+\s+)?ld\.global", s)]
    stores = [i for i, s in enumerate(lines) if re.match(r"(@!?%p\d+\s+)?st\.global", s)]
    assert len(trig) == 1 and len(wait) == 1 and loads and stores, (trig, wait)
    assert trig[0] < wait[0] < min(loads) and wait[0] < min(stores)
    # Nothing but the trigger and register setup branches around the wait.
    branches = [i for i, s in enumerate(lines) if re.match(r"(@!?%p\d+\s+)?bra", s)]
    assert not [b for b in branches if b < wait[0]], f"{name}: a branch precedes the wait"
    print(f"\n[ptx {name}] trigger {trig[0]}, wait {wait[0]}, first load {min(loads)}")


def _sass_audit(kernel, name: str, threads: int):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"{name}.cubin")
        with open(path, "wb") as f:
            f.write(kernel.asm["cubin"])
        functions = parse(disassemble(path))
    ((instructions, labels),) = functions.values()
    return audit(instructions, labels, threads)


@pytest.mark.parametrize("name, threads", [("conv", 128), ("merge", 128), ("silu", 128)])
def test_sass_waits_before_every_exit_and_reads_nothing_before(name, threads):
    kernel = _ENTRY_KERNELS[name]()
    torch.cuda.synchronize()
    result = _sass_audit(kernel, name, threads)
    print(f"\n[sass {name}] {result}")
    assert result["ok"], result
    assert result["preexit"] == 1 and result["acqbulk"] == 1, result
    assert result["global_loads_before_wait"] == {}, result
    assert result["global_stores_before_wait"] == {}, result


def test_default_builds_have_no_early_trigger():
    conv = _compiled_conv(False)
    merge = _compiled_merge(False)
    torch.cuda.synchronize()
    assert "griddepcontrol" not in conv.asm["ptx"]
    ptx = merge.asm["ptx"]
    assert "griddepcontrol.wait" in ptx and "griddepcontrol.launch_dependents" not in ptx


# --------------------------------------------------------------------------
# Delayed chains with the real successor GEMM
# --------------------------------------------------------------------------
@triton.jit
def _globaltimer(dummy):
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l,r", [dummy], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _late_writer_kernel(src, dst, n, delay_ns, BLOCK: tl.constexpr):
    """A PDL predecessor that releases its dependents first and writes its output late."""
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    start = _globaltimer(0)
    now = start
    while now - start < delay_ns:
        now = _globaltimer(0)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(dst + offsets, tl.load(src + offsets, mask=mask), mask=mask)


def _late_write(src: torch.Tensor, dst: torch.Tensor, delay_ns: int = DELAY_NS) -> None:
    n = src.numel()
    _late_writer_kernel[(triton.cdiv(n, 1024),)](
        src, dst, n, delay_ns, BLOCK=1024, num_warps=4, launch_pdl=True
    )


def _weight(n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(n, k, generator=g, device="cuda") / k**0.5).to(torch.bfloat16)


class _ConvChain:
    """late writer (x) -> conv prepare (entry) -> qkv GEMM (fixed split-K)."""

    def __init__(self):
        self.x, self.delta, self.base = _conv_operands(seed=3)
        self.src = self.x.clone()
        self.weight = _weight(1280, HIDDEN, 4)
        self.tactic = tactic_for(8, HIDDEN, 1280)

    def run(self, pdl: bool):
        if pdl:
            _late_write(self.src, self.x)
        else:
            self.x.copy_(self.src)
        conv = conv_module.dflash2_block_conv(self.x, self.delta, self.base, 8, 16, pdl)
        return fixed_low_m_gemm(conv, self.weight, self.tactic, pdl)

    def stale(self):
        self.x.fill_(float("nan"))


class _MergeChain:
    """late writer (q) -> split -> merge (entry) -> o_proj GEMM (fixed split-K)."""

    def __init__(self):
        self.q, self.k, self.v, self.seqlens, self.slots = _attention_operands(seed=5)
        self.src = self.q.clone()
        self.weight = _weight(HIDDEN, 1024, 6)
        self.tactic = tactic_for(8, 1024, HIDDEN)

    def run(self, pdl: bool):
        if pdl:
            _late_write(self.src, self.q)
        else:
            self.q.copy_(self.src)
        out = attention_module.dflash_window_attention(
            self.q,
            self.k,
            self.v,
            self.seqlens,
            self.slots,
            (2047, 2047),
            None,
            pdl,
            merge_entry_trigger=pdl,
        )
        return fixed_low_m_gemm(out.reshape(8, 1024), self.weight, self.tactic, pdl)

    def stale(self):
        self.q.fill_(float("nan"))


class _SiluChain:
    """late writer (gate_up output) -> silu copy (entry) -> down GEMM (fixed split-K)."""

    def __init__(self):
        self.h = _silu_input(dict(rows=8, d=3584), 9)
        self.src = self.h.clone()
        self.weight = _weight(HIDDEN, 3584, 10)
        self.tactic = tactic_for(8, 3584, HIDDEN)

    def run(self, pdl: bool):
        if pdl:
            _late_write(self.src, self.h)
        else:
            self.h.copy_(self.src)
        act = op(self.h, True) if pdl else swiglu(self.h)
        return fixed_low_m_gemm(act, self.weight, self.tactic, pdl)

    def stale(self):
        self.h.fill_(float("nan"))


_CHAINS = {"conv": _ConvChain, "merge": _MergeChain, "silu": _SiluChain}


@pytest.mark.parametrize("name", sorted(_CHAINS))
@torch.inference_mode()
def test_a_late_producer_is_waited_for_eagerly_and_in_a_graph(name):
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    if not get_env_enable_pdl():
        pytest.skip("set TRTLLM_ENABLE_PDL=1")
    chain = _CHAINS[name]()
    want = chain.run(False).clone()
    for _ in range(3):
        chain.stale()
        got = chain.run(True)
        torch.cuda.synchronize()
        _assert_same_bits(got, want, f"{name} eager")
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    chain.stale()
    with torch.cuda.graph(graph):
        captured = chain.run(True)
    # Read before the first replay instantiates the graph.
    edges = programmatic_edges(graph)
    for replay in range(5):
        chain.stale()
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _assert_same_bits(captured, want, f"{name} replay {replay}")
    kernel = {"conv": "dflash2_block_conv", "merge": "merge", "silu": "silu_and_mul_pdl"}[name]
    into_gemm = {
        key: count for key, count in edges.items() if kernel in key[0] and "plitK" in key[1]
    }
    print(f"\n[edges {name}] {into_gemm}")
    assert into_gemm, f"{name}: no programmatic edge into the GEMM: {list(edges)[:8]}"


# --------------------------------------------------------------------------
# Feature off: the base tree's machine code
# --------------------------------------------------------------------------
_BASE_TREE = os.environ.get("MODELING_V2_PDL_CHAIN_BASE_TREE")


def _load_base(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_BASE_TREE, relative))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strip_ptx(ptx: str) -> str:
    """The PTX without line information, debug sections and comments."""
    body = ptx.split("\t.section\t.debug", 1)[0]
    kept = []
    for line in body.splitlines():
        s = line.strip()
        if not s or s.startswith(".loc") or s.startswith(".file") or s.startswith("//"):
            continue
        kept.append(s)
    return "\n".join(kept)


def _sass(kernel, name: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"{name}.cubin")
        with open(path, "wb") as f:
            f.write(kernel.asm["cubin"])
        text = disassemble(path)
    return "\n".join(
        line.split(";")[0].strip()
        for line in text.splitlines()
        if re.match(r"\s*/\*[0-9a-f]{4}\*/", line)
    )


@pytest.mark.skipif(
    not _BASE_TREE, reason="set MODELING_V2_PDL_CHAIN_BASE_TREE to the base tree's target"
)
def test_feature_off_builds_are_the_base_trees_machine_code():
    base_conv = _load_base("ops/spec/dflash2_block_conv.py", "_drafter_pdl_base_conv")
    base_attention = _load_base(
        "ops/spec/dflash_window_attention.py", "_drafter_pdl_base_attention"
    )
    pairs = {
        "conv": (_compiled_conv(False), _compiled_conv(False, base_conv)),
        "merge": (_compiled_merge(False), _compiled_merge(False, base_attention)),
    }
    torch.cuda.synchronize()
    for name, (ours, base) in pairs.items():
        assert _strip_ptx(ours.asm["ptx"]) == _strip_ptx(base.asm["ptx"]), f"{name}: PTX differs"
        ours_sass, base_sass = _sass(ours, name), _sass(base, name)
        assert ours_sass and ours_sass == base_sass, f"{name}: SASS differs"
        print(
            f"\n[feature off {name}] PTX and SASS identical ({len(ours_sass.splitlines())} lines)"
        )
