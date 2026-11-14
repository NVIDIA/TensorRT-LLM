# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TAILGLUE_B: the DSpark step's next input tokens and the forward-tail spec fold in one CuTe DSL launch.

A decode-only DSpark step ends with two launches: the catalog's ``next_new_tokens`` (N3 A2,
``ops/dspark/strict_accept_kernel.py``) builds ``out[i] = [accepted[batch_idx[i], num_accepted[i] - 1],
*next_draft[i]]``, then, after the wait on the sampler's store copies, ``m3_fold_tail`` (this target's C++ op
``minimaxM3Sm107Tp4/foldTail.cu``, or its Triton twin) stores the step's outputs into the spec sampler's per-slot
store and stages the next step's inputs for the armed rows. This kernel does both: CTA ``i`` (lane ``c`` = column)
computes row ``i`` of ``out`` as ``next_new_tokens_kernel`` does and, when ``i < min(ctrl[0], fold_rows)``, writes with
``s = slots[i]`` exactly what ``foldTailKernel`` writes for row ``i`` from (``accepted``, ``out``, ``next_draft``,
``num_accepted``):

* ``dst_new[s, c] = accepted[i, c]`` (0 past ``accepted``'s width) for ``c <`` the store width; the same for
  ``dst_next`` from ``out`` and ``dst_draft`` from ``next_draft``; ``dst_lens[s] = num_accepted[i]``;
* ``ids[i T + c] = out[i, c]`` (0 past the store's next-token width), ``draft_ids[i D + c] = next_draft[i, c]`` (0
  past the store's draft width), ``pos_off[i T + c] = num_accepted[i]``, ``kv_off[i] = num_accepted[i] - T``, with
  ``T = D + 1``.

Integer copies only, so the outputs are bitwise the pair's. ``ctrl`` and ``slots`` are read on the device (a captured
graph whose step was not armed writes no store). Every width must fit one 32-lane CTA (foldTail.cu's bound).

A plain launch (no programmatic dependency, no griddepcontrol), as the fold tail is: it runs right after the store
wait (an event node under capture), where no kernel edge could be programmatic, and it starts after its predecessor
completed.
"""

import threading
from typing import Dict, Optional

import torch

LANES = 32  # capacity: fixed: one lane per column (foldTail.cu's 32-lane CTA)
MAX_ROWS = (
    64  # capacity: fixed: catalog strict_accept MAX_REQUESTS (next_new_tokens' one-CTA bound)
)

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}
_kernel: Dict[str, object] = {}


def _build():
    """The CuTe DSL kernel and its launcher, defined on first use (they pull in cutlass)."""
    mod = _kernel.get("mod")
    if mod is not None:
        return mod
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute

    @cute.kernel
    def nnt_fold_kernel(
        accepted: cute.Tensor,  # int32, flat: row i at i * acc_s0 (columns contiguous)
        num_accepted: cute.Tensor,  # int32 [>= n]
        batch_idx: cute.Tensor,  # int32 or int64 [>= n]
        next_draft: cute.Tensor,  # int32, flat: row i at i * nd_s0 (columns contiguous)
        out: cute.Tensor,  # int32 [n * T], out
        ctrl: cute.Tensor,  # int32 [1]: armed rows
        slots: cute.Tensor,  # int64 [>= fold_rows]
        dst_new: cute.Tensor,  # int32, flat: [s, c] at s * dn0 + c * dn1
        dst_next: cute.Tensor,
        dst_draft: cute.Tensor,
        dst_lens: cute.Tensor,  # int32 [slots]
        ids: cute.Tensor,  # int32 [>= fold_rows * T]
        draft_ids: cute.Tensor,  # int32 [>= fold_rows * D]
        pos_off: cute.Tensor,  # int32 [>= fold_rows * T]
        kv_off: cute.Tensor,  # int32 [>= fold_rows]
        acc_s0: cutlass.Int64,
        nd_s0: cutlass.Int64,
        sn_w: cutlass.Int32,  # accepted's width (the fold's source new-token width)
        dn0: cutlass.Int64,
        dn1: cutlass.Int64,
        dn_w: cutlass.Int32,
        dx0: cutlass.Int64,
        dx1: cutlass.Int64,
        dx_w: cutlass.Int32,
        dd0: cutlass.Int64,
        dd1: cutlass.Int64,
        dd_w: cutlass.Int32,
        draft_len: cutlass.Int32,  # D (next_draft's width); T = D + 1
        fold_rows: cutlass.Int32,
    ):
        c, _, _ = cute.arch.thread_idx()
        i, _, _ = cute.arch.block_idx()
        zero = cutlass.Int32(0)
        t = draft_len + cutlass.Int32(1)
        i64 = cutlass.Int64(i)
        # Every lane loads in bounds (clamped columns) and selects, so no value crosses a divergent branch.
        # next_new_tokens_kernel's row i: the token at the accepted count of row batch_idx[i], then the drafts.
        first = accepted[
            cutlass.Int64(batch_idx[i]) * acc_s0 + cutlass.Int64(num_accepted[i]) - cutlass.Int64(1)
        ]
        is_draft = (c >= cutlass.Int32(1)) & (c < t)
        jd = cutlass.Int32(cutlass.select_(is_draft, c - cutlass.Int32(1), zero))
        drafted = next_draft[i64 * nd_s0 + cutlass.Int64(jd)]
        nxt = cutlass.Int32(
            cutlass.select_(c == zero, first, cutlass.select_(is_draft, drafted, zero))
        )
        if c < t:
            out[i * t + c] = nxt
        # foldTailKernel's row i (src_new = accepted row i, src_next = out row i, src_draft = next_draft row i).
        if i < fold_rows:
            if i < ctrl[0]:
                s = cutlass.Int64(slots[i])
                cn = cutlass.Int32(cutlass.select_(c < sn_w, c, zero))
                nw = cutlass.Int32(
                    cutlass.select_(c < sn_w, accepted[i64 * acc_s0 + cutlass.Int64(cn)], zero)
                )
                if c < dn_w:
                    dst_new[s * dn0 + cutlass.Int64(c) * dn1] = nw
                if c < dx_w:
                    dst_next[s * dx0 + cutlass.Int64(c) * dx1] = nxt
                cd = cutlass.Int32(cutlass.select_(c < draft_len, c, zero))
                drf = cutlass.Int32(
                    cutlass.select_(
                        c < draft_len, next_draft[i64 * nd_s0 + cutlass.Int64(cd)], zero
                    )
                )
                if c < dd_w:
                    dst_draft[s * dd0 + cutlass.Int64(c) * dd1] = drf
                ln = num_accepted[i]
                if c == zero:
                    dst_lens[s] = ln
                    kv_off[i] = ln - t
                if c < t:
                    ids[i * t + c] = cutlass.Int32(cutlass.select_(c < dx_w, nxt, zero))
                    pos_off[i * t + c] = ln
                if c < draft_len:
                    draft_ids[i * draft_len + c] = cutlass.Int32(
                        cutlass.select_(c < dd_w, drf, zero)
                    )

    @cute.jit
    def nnt_fold(
        accepted: cute.Tensor,
        num_accepted: cute.Tensor,
        batch_idx: cute.Tensor,
        next_draft: cute.Tensor,
        out: cute.Tensor,
        ctrl: cute.Tensor,
        slots: cute.Tensor,
        dst_new: cute.Tensor,
        dst_next: cute.Tensor,
        dst_draft: cute.Tensor,
        dst_lens: cute.Tensor,
        ids: cute.Tensor,
        draft_ids: cute.Tensor,
        pos_off: cute.Tensor,
        kv_off: cute.Tensor,
        acc_s0: cutlass.Int64,
        nd_s0: cutlass.Int64,
        sn_w: cutlass.Int32,
        dn0: cutlass.Int64,
        dn1: cutlass.Int64,
        dn_w: cutlass.Int32,
        dx0: cutlass.Int64,
        dx1: cutlass.Int64,
        dx_w: cutlass.Int32,
        dd0: cutlass.Int64,
        dd1: cutlass.Int64,
        dd_w: cutlass.Int32,
        draft_len: cutlass.Int32,
        fold_rows: cutlass.Int32,
        n_req: cutlass.Int32,
        stream: cuda_driver.CUstream,
    ) -> None:
        nnt_fold_kernel(
            accepted, num_accepted, batch_idx, next_draft, out, ctrl, slots, dst_new, dst_next, dst_draft, dst_lens,
            ids, draft_ids, pos_off, kv_off, acc_s0, nd_s0, sn_w, dn0, dn1, dn_w, dx0, dx1, dx_w, dd0, dd1, dd_w,
            draft_len, fold_rows,
        ).launch(grid=[n_req, 1, 1], block=[LANES, 1, 1], stream=stream)  # fmt: skip

    _kernel["mod"] = nnt_fold
    return nnt_fold


def _flat(t: torch.Tensor) -> torch.Tensor:
    """A 1-D stride-1 view over every element ``t`` addresses (its storage from its first element)."""
    if t.dim() == 1:
        return t
    span = 1 + sum((s - 1) * st for s, st in zip(t.shape, t.stride()))
    return t.as_strided((span,), (1,))


def _arg(t: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=t.element_size()).mark_layout_dynamic(
        leading_dim=0
    )


def supports(
    accepted: torch.Tensor,
    num_accepted: torch.Tensor,
    batch_idx: torch.Tensor,
    next_draft: torch.Tensor,
    ctrl: torch.Tensor,
    slots: torch.Tensor,
    dst_new: torch.Tensor,
    dst_next: torch.Tensor,
    dst_draft: torch.Tensor,
    dst_lens: torch.Tensor,
    ids: torch.Tensor,
    draft_ids: torch.Tensor,
    pos_off: torch.Tensor,
    kv_off: torch.Tensor,
) -> bool:
    """Whether one launch reproduces the pair: catalog ``spec_next_new_tokens``'s operand rules with int32 accepted
    tokens and drafts (an int32 ``out``, as the fold needs), and ``m3_fold_tail.fold_tail``'s (every operand int32
    except int64 slots, 2-D stores, 1-D stride-1 rows), every width within one 32-lane CTA, at most MAX_ROWS rows."""
    two = (accepted, next_draft, dst_new, dst_next, dst_draft)
    one = (num_accepted, batch_idx, dst_lens, ids, draft_ids, pos_off, kv_off, ctrl)
    if any(not t.is_cuda for t in two + one + (slots,)):
        return False
    if any(t.dim() != 2 for t in two) or any(t.dim() != 1 or t.stride(0) != 1 for t in one):
        return False
    if (
        any(t.dtype != torch.int32 for t in two + one if t is not batch_idx)
        or slots.dtype != torch.long
    ):
        return False
    if batch_idx.dtype not in (torch.int32, torch.int64):
        return False
    devices = {t.device for t in two + one + (slots,)}
    if len(devices) != 1:
        return False
    n, d = next_draft.shape
    t = d + 1
    rows = min(n, slots.numel())
    if not (0 < n <= MAX_ROWS and d > 0 and accepted.shape[0] == n):
        return False
    if accepted.stride(1) != 1 or next_draft.stride(1) != 1:
        return False
    if num_accepted.numel() < n or batch_idx.numel() < n:
        return False
    widths = (accepted.shape[1], dst_new.shape[1], t, dst_next.shape[1], d, dst_draft.shape[1])
    if max(widths) > LANES:
        return False
    return (
        ids.numel() >= rows * t
        and pos_off.numel() >= rows * t
        and draft_ids.numel() >= rows * d
        and kv_off.numel() >= rows
        and ctrl.numel() >= 1
    )


def _key(batch_idx: torch.Tensor) -> tuple:
    return ("nnt_fold", batch_idx.dtype)


def is_compiled(batch_idx: torch.Tensor) -> bool:
    return _key(batch_idx) in _compiled


def nnt_fold(
    accepted: torch.Tensor,
    num_accepted: torch.Tensor,
    batch_idx: torch.Tensor,
    next_draft: torch.Tensor,
    ctrl: torch.Tensor,
    slots: torch.Tensor,
    dst_new: torch.Tensor,
    dst_next: torch.Tensor,
    dst_draft: torch.Tensor,
    dst_lens: torch.Tensor,
    ids: torch.Tensor,
    draft_ids: torch.Tensor,
    pos_off: torch.Tensor,
    kv_off: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Launch over the ``next_draft.shape[0]`` rows; returns ``out`` [n, D + 1] int32 (next_new_tokens). The fold
    writes the first ``min(ctrl[0], n, slots.numel())`` rows. Check ``supports`` first; the first call per batch_idx
    dtype compiles and must run outside CUDA-graph capture."""
    import cuda.bindings.driver as cuda_driver
    import cutlass.cute as cute

    n, d = next_draft.shape
    t = d + 1
    if out is None:
        out = torch.empty(n, t, dtype=torch.int32, device=next_draft.device)
    rows = min(n, slots.numel())
    tensors = (
        _arg(_flat(accepted)),
        _arg(num_accepted),
        _arg(batch_idx),
        _arg(_flat(next_draft)),
        _arg(out.view(-1)),
        _arg(ctrl),
        _arg(slots),
        _arg(_flat(dst_new)),
        _arg(_flat(dst_next)),
        _arg(_flat(dst_draft)),
        _arg(dst_lens),
        _arg(ids),
        _arg(draft_ids),
        _arg(pos_off),
        _arg(kv_off),
    )
    scalars = (
        int(accepted.stride(0)),
        int(next_draft.stride(0)),
        int(accepted.shape[1]),
        int(dst_new.stride(0)),
        int(dst_new.stride(1)),
        int(dst_new.shape[1]),
        int(dst_next.stride(0)),
        int(dst_next.stride(1)),
        int(dst_next.shape[1]),
        int(dst_draft.stride(0)),
        int(dst_draft.stride(1)),
        int(dst_draft.shape[1]),
        int(d),
        int(rows),
        int(n),
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(next_draft.device).cuda_stream)
    key = _key(batch_idx)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "m3_nnt_fold must run once per batch_idx dtype outside CUDA-graph capture first"
            )
        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(_build(), *tensors, *scalars, stream)
    fn(*tensors, *scalars, stream)
    return out


def precompile(device, batch_idx_dtypes=(torch.int32,)) -> None:
    """Compile (and run once, unarmed) the launch per batch_idx dtype outside capture."""
    i32 = dict(dtype=torch.int32, device=device)
    for dt in batch_idx_dtypes:
        acc = torch.zeros(1, 2, **i32)
        num = torch.ones(1, **i32)
        bidx = torch.zeros(1, dtype=dt, device=device)
        nd = torch.zeros(1, 1, **i32)
        ctrl = torch.zeros(1, **i32)  # unarmed: no store write
        slots = torch.zeros(1, dtype=torch.long, device=device)
        store = torch.zeros(1, 2, **i32)
        engine = (
            torch.zeros(2, **i32),
            torch.zeros(1, **i32),
            torch.zeros(2, **i32),
            torch.zeros(1, **i32),
        )
        lens = torch.zeros(1, **i32)
        nnt_fold(acc, num, bidx, nd, ctrl, slots, store, store, store[:, :1], lens, *engine)
    torch.cuda.synchronize(device)
