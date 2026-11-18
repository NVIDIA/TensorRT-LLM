# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measurement-only recorder of the decode steps' expert routing (s316 lane, hot-expert study).

Off unless ``MUSESPARK_OPEN_S316_ROUTE_REC=<path>`` is set; then rank 0 records, on every call of the
NVFP4 ring-buddy layers' routing (``ring_buddy._patch_routing``) of at most 128 rows, the call's global
top-8 expert ids into a device ring of slots: one small Triton launch per call, captured into the decode
graph like the call itself. A call counter (device atomic) numbers the calls; only every
``MUSESPARK_OPEN_S316_ROUTE_REC_EVERY``-th group of ``MUSESPARK_OPEN_S316_ROUTE_REC_LAYERS`` (60)
consecutive calls is kept, until the ring is full. A host thread copies the ring to ``<path>.npz`` every
30 s (ids int16 [slots, 128, 8], -1 past the call's rows; rows; call index). The recorder reads the ids
and writes only its own buffers: the step's values are unchanged, its timing is not (one launch a layer):
never use it on a timing arm.
"""

from __future__ import annotations

import os
import threading
import time

import torch
import triton
import triton.language as tl

_PATH = os.environ.get("MUSESPARK_OPEN_S316_ROUTE_REC", "")
_EVERY = int(os.environ.get("MUSESPARK_OPEN_S316_ROUTE_REC_EVERY", "8"))
_LAYERS = int(os.environ.get("MUSESPARK_OPEN_S316_ROUTE_REC_LAYERS", "60"))
_SLOTS = int(os.environ.get("MUSESPARK_OPEN_S316_ROUTE_REC_SLOTS", "196608"))
_R = 128
_state = {}


def enabled(rank: int) -> bool:
    return bool(_PATH) and rank == 0


@triton.jit
def _record_kernel(
    sel,
    ids,
    rows_buf,
    call_buf,
    ctr,
    rows,
    EVERY: tl.constexpr,
    LAYERS: tl.constexpr,
    SLOTS: tl.constexpr,
    R8: tl.constexpr,
):
    c = tl.atomic_add(ctr, 1)  # the call index
    grp = c // LAYERS
    keep = (grp % EVERY) == 0
    slot = (grp // EVERY) * LAYERS + (c % LAYERS)
    keep = keep & (slot < SLOTS)
    offs = tl.arange(0, R8)
    v = tl.load(sel + offs, mask=offs < rows * 8, other=-1)
    tl.store(ids + slot.to(tl.int64) * R8 + offs, v.to(tl.int16), mask=keep & (offs < R8))
    tl.store(rows_buf + slot, rows, mask=keep)
    tl.store(call_buf + slot, c, mask=keep)


def _dump_loop():
    s = torch.cuda.Stream(device=_state["ids"].device)
    while True:
        time.sleep(30)
        try:
            with torch.cuda.stream(s):
                ids = _state["ids"].cpu().numpy()
                rows = _state["rows"].cpu().numpy()
                calls = _state["calls"].cpu().numpy()
                ctr = _state["ctr"].cpu().numpy()
            import numpy as np

            tmp = _PATH + ".tmp.npz"
            np.savez(tmp, ids=ids, rows=rows, calls=calls, ctr=ctr, every=_EVERY, layers=_LAYERS)
            os.replace(tmp, _PATH + ".npz")
        except Exception as e:  # noqa: BLE001 - measurement only
            print(f"[s316 route_recorder] dump failed: {e}", flush=True)


def record(selected: torch.Tensor) -> None:
    """Record one call's global top-8 ids (int32 [rows, 8], rows <= 128) on rank 0."""
    rows = int(selected.shape[0])
    if rows > _R:
        return
    if not _state:
        dev = selected.device
        _state["ids"] = torch.full((_SLOTS, _R * 8), -1, dtype=torch.int16, device=dev)
        _state["rows"] = torch.zeros(_SLOTS, dtype=torch.int32, device=dev)
        _state["calls"] = torch.full((_SLOTS,), -1, dtype=torch.int64, device=dev)
        _state["ctr"] = torch.zeros(1, dtype=torch.int64, device=dev)
        threading.Thread(target=_dump_loop, daemon=True).start()
        print(
            f"[s316 route_recorder] recording to {_PATH}.npz: {_SLOTS} slots, every {_EVERY} x {_LAYERS} calls",
            flush=True,
        )
    _record_kernel[(1,)](
        selected,
        _state["ids"],
        _state["rows"],
        _state["calls"],
        _state["ctr"],
        rows,
        EVERY=_EVERY,
        LAYERS=_LAYERS,
        SLOTS=_SLOTS,
        R8=_R * 8,
        num_warps=4,
    )
