# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PDL ordering check on compiled PTX.

A kernel its producer may release early must not touch global memory before ``griddepcontrol.wait``: a pre-wait
store can hit a block the caching allocator just recycled from the producer's still-read input, and a pre-wait
load of non-static data is stale. Static loads (weights) are allowed with ``allow_loads_before=True``. A
``griddepcontrol.launch_dependents`` must follow a CTA barrier that follows the last global access.

The check is on PTX text order, not control flow: a kernel whose branches wait separately is checked against its
FIRST wait. CLI: ``python3 pdl_ptx_check.py <file.ptx | dir> [--allow-loads-before]``.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, List

# Generic ld/st are not matched: classifying them needs address-space analysis.
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
    """Global loads, global -> shared async copies, prefetch."""
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
    """Fail if a griddepcontrol.launch_dependents has no CTA barrier between the last global access before it
    (text order) and itself. ``strict`` also fails when global accesses follow it in text order, which may be
    another branch's epilogue."""
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
