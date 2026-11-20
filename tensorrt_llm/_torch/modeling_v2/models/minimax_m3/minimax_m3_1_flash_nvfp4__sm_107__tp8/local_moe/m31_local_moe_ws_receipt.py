# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""INFO-only per-rank workspace identity receipt for the TP8 m3_moe engine states (for the TP8 serving hook).
Default ON; TRTLLM_M31_DISABLE_LOCAL_MOE_WS_RECEIPT=1 turns it off. Logging only:
no allocation, no placement change, no kernel or numerics change.

receipt(state, rank, locality=None) -> list of lines, one per hot per-call buffer of an M3MoeState / LocalMoeState:
  [LOCAL_MOE_WS] rank=<r> gen=<g> state=<cls> tokens=<m> buf=<name> ptr=0x<va> bytes=<n> off2m=<va % 2 MiB>
                 page2m=0x<va // 2 MiB> dom=<locality domain | -1 striped | -99 query failed | na>
gen = this process's state build ordinal (1, 2, ..), so a served placement is identifiable after the fact; the
2 MiB block ids make the hot set's page co-location readable (absolute addresses of every hot buffer, lgbuf / cnt
included)."""

import itertools
import sys

from ..m31_switches import m31_disabled

HOT = (
    "hbuf32",
    "lgbuf",
    "epochs",
    "cnt",
    "partials",
    "plan",
    "img32",
    "hx",
    "dnx",
    "route_dbg",
    "smap",
)
SLOT = 2 << 20
_GEN = itertools.count(1)


def enabled():
    return not m31_disabled("LOCAL_MOE_WS_RECEIPT")


def _dom(locality, ptr):
    if locality is None:
        return "na"
    try:
        return str(locality.domain_of(ptr))
    except Exception:  # noqa: BLE001 -- INFO only: never fail the caller
        return "-99"


def receipt(state, rank, locality=None, out=None):
    if not enabled():
        return []
    gen = next(_GEN)
    lines = []
    for name in HOT:
        t = getattr(state, name, None)
        if t is None or not hasattr(t, "data_ptr"):
            continue
        ptr = int(t.data_ptr())
        nbytes = int(t.numel() * t.element_size())
        lines.append(
            f"[LOCAL_MOE_WS] rank={rank} gen={gen} state={type(state).__name__} tokens={getattr(state, 'm', '?')} "
            f"buf={name} ptr=0x{ptr:x} bytes={nbytes} off2m={ptr % SLOT} page2m=0x{ptr // SLOT:x} "
            f"dom={_dom(locality, ptr)}"
        )
    stream = out if out is not None else sys.stderr
    for ln in lines:
        print(ln, file=stream, flush=True)
    return lines
