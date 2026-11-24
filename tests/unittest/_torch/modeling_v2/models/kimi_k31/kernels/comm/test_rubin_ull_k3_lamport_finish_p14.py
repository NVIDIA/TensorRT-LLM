# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The finish kernel of ``kernels/comm/rubin_ull_k3_lamport_ar_pdlfix.py``
(``K3LamportFinishKernelP14``) stores into ``out`` only after ``griddepcontrol.wait`` when it runs
with PDL and without ``wait_first``, and otherwise does what ``rubin_ull_k3_lamport_ar``'s
``K3LamportFinishKernel`` does. Both kernels' Python bodies run as the CuTe DSL traces them, on
stand-ins for the PTX helpers that record each operation in program order per CTA and thread.

CPU only; the SM10x reproducer is ``test_rubin_ull_k3_lamport_finish_p14_sm10x.py``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

SHARED, P14 = "rubin_ull_k3_lamport_ar", "rubin_ull_k3_lamport_ar_pdlfix"


def _comm_dir() -> Path:
    spec = importlib.util.find_spec("tensorrt_llm")
    root = Path(spec.submodule_search_locations[0]).resolve()
    return root / "_torch/_experimental/modeling_v2/models/kimi_k31/kernels/comm"


UC, MC, FLAGS, OUT = (k << 40 for k in (1, 2, 3, 4))
NBYTES = 4 << 20


def _region(addr: int) -> str:
    return {1: "uc", 2: "mc", 3: "flags", 4: "out"}[addr >> 40]


def _fake_dsl(ctx) -> dict:
    """The CuTe DSL and CUDA bindings stood in: compile-time branches and loops are Python's."""

    def decorator(fn=None, **kwargs):
        return fn if fn is not None else (lambda f: f)

    def crange(start, stop=None, step=1, **kwargs):
        return range(start) if stop is None else range(start, stop, step)

    cutlass = types.ModuleType("cutlass")
    cutlass.__path__ = []
    cutlass.Int32 = cutlass.Int64 = cutlass.Uint32 = cutlass.Uint64 = int
    cutlass.Float32 = float
    cutlass.const_expr = bool
    cutlass.range_constexpr = crange
    cutlass.range = crange
    cute = types.ModuleType("cutlass.cute")
    cute.jit = cute.kernel = decorator
    cute.arch = SimpleNamespace(
        thread_idx=lambda: (ctx.tid, 0, 0),
        block_idx=lambda: (ctx.bid, 0, 0),
        barrier=lambda: ctx.events.append(("barrier",)),
    )
    cute.make_rmem_tensor = lambda shape, dtype: [0] * shape[0]
    cutlass.cute = cute
    dsl = types.ModuleType("cutlass.cutlass_dsl")
    dsl.T, dsl.dsl_user_op = SimpleNamespace(), decorator
    mlir = types.ModuleType("cutlass._mlir")
    mlir.__path__ = []
    dialects = types.ModuleType("cutlass._mlir.dialects")
    dialects.__path__ = []
    dialects.llvm = types.ModuleType("cutlass._mlir.dialects.llvm")
    cuda = types.ModuleType("cuda")
    cuda.__path__ = []
    bindings = types.ModuleType("cuda.bindings")
    bindings.__path__ = []
    bindings.driver = types.ModuleType("cuda.bindings.driver")
    bindings.driver.CUstream = int
    cuda.bindings = bindings
    return {
        "cutlass": cutlass,
        "cutlass.cute": cute,
        "cutlass.cutlass_dsl": dsl,
        "cutlass._mlir": mlir,
        "cutlass._mlir.dialects": dialects,
        "cutlass._mlir.dialects.llvm": dialects.llvm,
        "cuda": cuda,
        "cuda.bindings": bindings,
        "cuda.bindings.driver": bindings.driver,
    }


def _primitives(ctx) -> dict:
    """The PTX helpers stood in, each recording what it does where into ``ctx.events``."""

    def rec(event):
        ctx.events.append(event)

    def ld_volatile_v4(addr):
        if addr == FLAGS:
            return (0, 2, NBYTES, 1)  # current 0, dirty 2, bytes per buffer, one dirty stage
        if addr == FLAGS + 16:
            return (64, 0, 0, 0)  # bytes to re-arm per stage
        rec(("load", _region(addr)))
        return (1, 2, 3, 4)

    return dict(
        ld_volatile_v4=ld_volatile_v4,
        clean4=lambda *w: 1,
        st_v4=lambda addr, *v: rec(("store", _region(addr))),
        multimem_st_v4=lambda addr, *v: rec(("multicast", _region(addr))),
        st_sentinel_v4=lambda addr: rec(("rearm", _region(addr))),
        st_u32=lambda addr, v: rec(("store", _region(addr))),
        red_release_add=lambda addr, v: rec(("count",)),
        ld_acquire_u32=lambda addr: 1 << 30,
        griddepcontrol_wait=lambda: rec(("wait",)),
        griddepcontrol_launch_dependents=lambda: rec(("release",)),
        acc_bf16x2=lambda word, lo, hi: (lo + 1.0, hi + 1.0),
        add_f32=lambda a, b: a + b,
        pack_bf16x2=lambda lo, hi: 7,
        sanitize_bf16x2=lambda w: w,
        ag_add_bf16x2=lambda lo, hi, y: 7,
    )


@pytest.fixture(scope="module")
def traced():
    """Both modules loaded on the stand-ins, under a test-only package (the pdlfix module imports
    SHARED relatively)."""
    ctx = SimpleNamespace(tid=0, bid=0, events=[])
    package = "_k31_p14_under_test"
    with pytest.MonkeyPatch.context() as mp:
        for name, module in _fake_dsl(ctx).items():
            mp.setitem(sys.modules, name, module)
        root = types.ModuleType(package)
        root.__path__ = [str(_comm_dir())]
        mp.setitem(sys.modules, package, root)
        modules = {}
        for name in (SHARED, P14):
            spec = importlib.util.spec_from_file_location(
                f"{package}.{name}", _comm_dir() / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            mp.setitem(sys.modules, spec.name, module)
            spec.loader.exec_module(module)
            modules[name] = module
        for module in modules.values():
            for name, fn in _primitives(ctx).items():
                if hasattr(module, name):
                    mp.setattr(module, name, fn)
        yield ctx, modules


def _trace(traced, which: str, bid: int, tid: int, m: int, rank: int, **config):
    ctx, modules = traced
    module = modules[which]
    cls = module.K3LamportFinishKernel if which == SHARED else module.K3LamportFinishKernelP14
    kernel = cls(**config)
    ctx.bid, ctx.tid, ctx.events = bid, tid, []
    kernel.kernel(UC, MC, FLAGS, OUT, m, rank)
    return ctx.events


# (world, hidden, one_shot, ag, gather), rows, rank: the drafter's and L0's one-shot calls (16
# ranks), the all-gather variant, the two-shot gather on the owner of the traced row (1, the last
# of two rows), the in-place two-shot, and a row with a thread past its vectors.
CONFIGS = [
    (dict(world=16, hidden=7168, one_shot=True), 2, 0),
    (dict(world=16, hidden=7168, one_shot=True, ag=True), 2, 0),
    (dict(world=16, hidden=7168, one_shot=False), 2, 1),
    (dict(world=16, hidden=7168, one_shot=False, gather=False), 2, 1),
    (dict(world=2, hidden=1032, one_shot=True), 3, 1),
]


def _threads(config, m):
    """(bid, tid) of: the first and the last worker vector of the last row, a thread past the
    row's vectors where there is one, and the updater CTA."""
    vecs = config["hidden"] // 8
    cols = -(-vecs // 128)
    workers = m * cols
    picks = [(workers - cols, 0), (workers - 1, (vecs - 1) % 128)]
    if vecs % 128:
        picks.append((workers - 1, 127))
    return picks + [(workers, 0)]


@pytest.mark.parametrize("use_pdl, wait_first", [(True, False), (True, True), (False, False)])
@pytest.mark.parametrize("config, m, rank", CONFIGS)
def test_every_store_into_out_follows_a_wait(traced, config, m, rank, use_pdl, wait_first):
    """Per CTA and thread: with PDL every store into ``out`` follows a wait (the shared kernel
    stores first without ``wait_first``); only that one wait is added; without PDL none."""
    full = dict(config, use_pdl=use_pdl, wait_first=wait_first)
    stores = 0
    for bid, tid in _threads(config, m):
        shared = _trace(traced, SHARED, bid, tid, m, rank, **full)
        fixed = _trace(traced, P14, bid, tid, m, rank, **full)
        assert [e for e in fixed if e != ("wait",)] == [e for e in shared if e != ("wait",)]
        out = [i for i, e in enumerate(fixed) if e == ("store", "out")]
        stores += len(out)
        if not use_pdl:
            assert ("wait",) not in fixed and ("release",) not in fixed
            continue
        for i in out:
            assert ("wait",) in fixed[:i], (bid, tid, fixed)
        added = fixed.count(("wait",)) - shared.count(("wait",))
        assert added == (1 if out and not wait_first else 0), (bid, tid, fixed)
        if out and not wait_first:
            first = shared.index(("store", "out"))
            assert ("wait",) not in shared[:first], "the control: the shared kernel stores first"
    expect_out = config.get("gather", True) or config["one_shot"]
    assert (stores > 0) == expect_out
