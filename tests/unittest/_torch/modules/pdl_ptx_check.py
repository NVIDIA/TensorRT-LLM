# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PDL ordering check on compiled PTX.

Rule: a kernel that a producer may release early (any PDL dependent of an entry / mid-kernel trigger) must not touch
global memory before ``griddepcontrol.wait``: a pre-wait write into a fresh ``torch.empty`` buffer can land in a block
the caching allocator just recycled from the producer's input (freed in stream order) while the producer still reads
it, and a pre-wait read of non-static data is stale. Reading static data (weights) before the wait is fine; pass
``allow_loads_before=True`` for kernels that prefetch weights on purpose. A trigger
(``griddepcontrol.launch_dependents``) is safe late: after a CTA barrier that follows the kernel's last global
access.

Usage from a test (Triton kernels; ``ck`` = the CompiledKernel a launch returns, ``kernel[grid](...)``)::

    import importlib.util, os

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pdl_ptx_check.py"
    )  # next to the test
    spec = importlib.util.spec_from_file_location("pdl_ptx_check", path)
    pdl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pdl)
    pdl.assert_wait_first(ck, "my_kernel")  # dependents: nothing global before the wait
    pdl.assert_wait_first(
        ck, "router", allow_loads_before=True
    )  # weight prefetch allowed, no stores / atomics
    pdl.assert_trigger_last(
        ck, "my_kernel"
    )  # if it triggers: after the last global access + barrier

CLI: ``python3 pdl_ptx_check.py <file.ptx | dir with *.ptx (e.g. a TRITON_CACHE_DIR)> [--allow-loads-before]``.

Limits: the check is on PTX text order, not control flow. A kernel whose branches each wait separately (e.g. GEMM CTAs
and quant CTAs of one launch) is checked against its FIRST wait; split such kernels per variant (constexpr flags) or
inspect the report. CUDA C++ / CuTe DSL kernels can be checked the same way from their PTX (``cuobjdump -ptx``, or the
DSL's dumped PTX).
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, List

# global-memory instructions (generic ld/st are excluded: they cannot be classified without address-space analysis;
# compilers emit ld.global / st.global for global pointers)
_LOAD = r"ld\.(?:global|volatile\.global|relaxed\.\w+\.global|acquire\.\w+\.global)\S*"
_STORE = r"st\.(?:global|volatile\.global|relaxed\.\w+\.global|release\.\w+\.global)\S*"
_ATOMIC = r"(?:atom|red)\.(?!shared)\S*"
_ASYNC_LOAD = r"(?:cp\.async(?!\.bulk\.(?:global\.shared|tensor\.\dd\.global\.shared))\S*|prefetch\.global\S*)"
_ASYNC_STORE = (
    r"(?:cp\.reduce\.async\S*|cp\.async\.bulk\.(?:global\.shared|tensor\.\dd\.global\.shared)\S*)"
)
_OPS = re.compile(
    rf"^\s*(?:@!?%\w+\s+)?({_LOAD}|{_STORE}|{_ATOMIC}|{_ASYNC_STORE}|{_ASYNC_LOAD})", re.M
)


def _is_load(op: str) -> bool:
    """Reads only (allowed before the wait for static data): global loads, global -> shared async copies, prefetch."""
    return (
        op.startswith("ld.")
        or (op.startswith("cp.async") and not re.match(_ASYNC_STORE, op))
        or op.startswith("prefetch.")
    )


_WAIT = re.compile(r"griddepcontrol\.wait")
_TRIGGER = re.compile(r"griddepcontrol\.launch_dependents")
_BARRIER = re.compile(
    r"^\s*(?:@!?%\w+\s+)?(?:bar\.sync|barrier\.sync|bar\.warp\.sync|barrier\.cta\.sync)\S*", re.M
)


def _ptx_of(kernel_or_ptx) -> str:
    if isinstance(kernel_or_ptx, str):
        return kernel_or_ptx
    asm = getattr(kernel_or_ptx, "asm", None)
    if asm is not None and "ptx" in asm:
        return asm["ptx"]
    raise TypeError("expected PTX text or a compiled Triton kernel (with .asm['ptx'])")


def _line(ptx: str, pos: int) -> str:
    start = ptx.rfind("\n", 0, pos) + 1
    end = ptx.find("\n", pos)
    return ptx[start : end if end >= 0 else None].strip()


def ptx_report(kernel_or_ptx) -> Dict[str, object]:
    """Positions (PTX text order) of the first griddepcontrol.wait, the global-memory instructions before it, and of
    each launch_dependents relative to the last global access and the last barrier before it."""
    ptx = _ptx_of(kernel_or_ptx)
    waits = [m.start() for m in _WAIT.finditer(ptx)]
    ops = [(m.start(1), m.group(1)) for m in _OPS.finditer(ptx)]
    first_wait = waits[0] if waits else -1
    before = [(p, op) for p, op in ops if first_wait < 0 or p < first_wait]
    triggers = []
    last_op = max((p for p, _ in ops), default=-1)
    bars = [m.start() for m in _BARRIER.finditer(ptx)]
    for t in (m.start() for m in _TRIGGER.finditer(ptx)):
        ops_after = [op for p, op in ops if p > t]
        prev_ops = [p for p, _ in ops if p < t]
        bar_between = any(b < t and (not prev_ops or b > max(prev_ops)) for b in bars)
        triggers.append(
            dict(
                pos=t,
                global_ops_after=len(ops_after),
                barrier_after_last_access=bar_between,
                after_last_global_op=t > last_op,
            )
        )
    return dict(
        waits=len(waits),
        first_wait=first_wait,
        global_ops=len(ops),
        before_wait=[(op, _line(ptx, p)) for p, op in before],
        triggers=triggers,
    )


def assert_wait_first(
    kernel_or_ptx, name: str = "kernel", allow_loads_before: bool = False
) -> Dict[str, object]:
    """Fail unless the kernel waits (griddepcontrol.wait) before any global store / atomic / async copy, and (unless
    ``allow_loads_before``: static-data prefetch) before any global load."""
    rep = ptx_report(kernel_or_ptx)
    if rep["first_wait"] < 0:
        raise AssertionError(f"{name}: no griddepcontrol.wait in the PTX (compiled without PDL?)")
    bad = [
        (op, line) for op, line in rep["before_wait"] if not (allow_loads_before and _is_load(op))
    ]
    if bad:
        shown = "; ".join(line for _, line in bad[:6])
        raise AssertionError(
            f"{name}: {len(bad)} global memory instruction(s) before griddepcontrol.wait: {shown}"
        )
    return rep


def assert_trigger_last(
    kernel_or_ptx, name: str = "kernel", strict: bool = False
) -> Dict[str, object]:
    """Fail if a griddepcontrol.launch_dependents has no CTA barrier between the last global access before it (text
    order) and itself. With ``strict`` also fail when global accesses follow it in text order (they may belong to
    another branch, e.g. when the compiler duplicates the epilogue per branch: inspect ``global_ops_after``).
    Entry / mid-kernel triggers need every possible dependent to pass assert_wait_first."""
    rep = ptx_report(kernel_or_ptx)
    for t in rep["triggers"]:
        if not t["barrier_after_last_access"] or (strict and not t["after_last_global_op"]):
            raise AssertionError(
                f"{name}: launch_dependents at PTX offset {t['pos']} is not after the last global "
                f"access + a CTA barrier ({t['global_ops_after']} global ops follow it in text order)"
            )
    return rep


def _main(argv: List[str]) -> int:
    allow = "--allow-loads-before" in argv
    paths = [a for a in argv if not a.startswith("--")]
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, n) for n in names if n.endswith(".ptx")]
        else:
            files.append(p)
    rc = 0
    for f in sorted(files):
        rep = ptx_report(open(f).read())
        bad = [x for x in rep["before_wait"] if not (allow and _is_load(x[0]))]
        late = all(
            t["after_last_global_op"] and t["barrier_after_last_access"] for t in rep["triggers"]
        )
        status = (
            "no-wait"
            if rep["first_wait"] < 0
            else ("OK" if not bad else f"FAIL({len(bad)} before wait)")
        )
        print(f"{status:>18s}  triggers={len(rep['triggers'])}{'' if late else ' (early)'}  {f}")
        for _, line in bad[:4]:
            print(f"{'':>20s}{line}")
        rc |= int(bool(bad))
    return rc


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
