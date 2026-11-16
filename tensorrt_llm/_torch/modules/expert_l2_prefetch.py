# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-step L2 prefetch of the routed experts a MoE layer is likely to use at the next decode step.

At small decode batches DRAM idles through most of a layer's dependency chain, while the experts a layer touches
change slowly between steps; so their FC1 weights are read into L2 while the chain runs, on a side stream forked from
the current one (the main chain's kernels and PDL edges are untouched):

- record: after the decode front has built a layer's routing tables, a 1-CTA kernel stores the ``num_experts`` local
  experts with the most token tiles, or in a budgeted forward the most routed rows (ties: lower id), for the next step;
- prefetch: early in the layer, ``num_ctas`` 1-warp CTAs issue ``cp.async.bulk.prefetch.L2`` of those experts'
  ``w3_w1_weight`` / ``w3_w1_weight_scale`` in 64 KiB pieces, in rank order; in a budgeted forward only for the
  layer's issue budget after the CTA starts;
- join: the model's forward joins the side stream once, after its last decoder layer.

The selection only steers prefetches, so outputs are unchanged by construction. Forwards outside
``min_rows``..``max_rows`` rows neither record nor prefetch (decided per captured batch size).
"""

from types import MappingProxyType
from typing import Mapping, Optional, Tuple

import torch
import triton
import triton.language as tl

PIECE = 65536  # bytes per bulk prefetch
NUM_LOCAL_MAX = 128


@triton.jit
def _record_kernel(
    batch_ptr,
    nct_ptr,
    limit_ptr,
    sel_ptr,
    K: tl.constexpr,
    NL: tl.constexpr,
    BLOCK: tl.constexpr,
    BY_ROWS: tl.constexpr,
    TILE: tl.constexpr,
):
    """sel[0:K] = the K local experts with the most token tiles in this call, or (BY_ROWS) the most routed rows:
    row r of the padded permuted order belongs to tile r // TILE and counts while r < that tile's row limit; -1
    where fewer are touched."""
    n = tl.load(nct_ptr)
    if BY_ROWS:
        r = tl.arange(0, BLOCK)
        t = r // TILE
        e = tl.load(batch_ptr + t, mask=t < n, other=NL)
        lim = tl.load(limit_ptr + t, mask=t < n, other=0)
        e = tl.where((r < lim) & (e >= 0) & (e < NL), e, NL)
    else:
        offs = tl.arange(0, BLOCK)
        e = tl.load(batch_ptr + offs, mask=offs < n, other=NL)
        e = tl.where((e >= 0) & (e < NL), e, NL)
    cnt = tl.histogram(e, 2 * NL)  # bin NL holds the padding
    ids = tl.arange(0, 2 * NL)
    # key: count high, then the lower id first; experts with no count and the padding bins get key 0
    key = tl.where((ids < NL) & (cnt > 0), cnt * 1024 + (1023 - ids), 0)
    key = tl.sort(key, descending=True)
    sel = tl.where(key > 0, 1023 - (key % 1024), -1)
    tl.store(sel_ptr + ids, sel, mask=ids < K)


@triton.jit
def _globaltimer():
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _issue(
    w_ptr, sf_ptr, sel_ptr, per_w, per_sf, piece, k, lane, NPW: tl.constexpr, NPP: tl.constexpr
):
    """Item k = (selection slot k // NPP, piece k % NPP) of that expert's [FC1 weight (NPW pieces), FC1 block scales
    (NPP - NPW pieces)]: one fire-and-forget bulk L2 prefetch of ``piece`` bytes from lane 0."""
    i = k // NPP
    p = k % NPP
    e = tl.load(sel_ptr + i)
    e64 = tl.maximum(e, 0).to(tl.int64)
    pw = tl.minimum(p, NPW - 1).to(tl.int64)
    ps = tl.maximum(p - NPW, 0).to(tl.int64)
    aw = w_ptr + (e64 * per_w + pw * piece) + lane * 0
    asf = sf_ptr + (e64 * per_sf + ps * piece) + lane * 0
    go = (lane == 0) & (e >= 0)
    sz = lane * 0 + piece
    rw = tl.inline_asm_elementwise(
        "{ .reg .pred q; setp.ne.b32 q, $2, 0; @q cp.async.bulk.prefetch.L2.global [$1], $3; mov.u32 $0, 0; }",
        "=r,l,r,r",
        [aw, (go & (p < NPW)).to(tl.int32), sz],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    rs = tl.inline_asm_elementwise(
        "{ .reg .pred q; setp.ne.b32 q, $2, 0; @q cp.async.bulk.prefetch.L2.global [$1], $3; mov.u32 $0, 0; }",
        "=r,l,r,r",
        [asf, (go & (p >= NPW)).to(tl.int32), sz],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    return rw + rs


@triton.jit(do_not_specialize=["budget_ns"])
def _prefetch_kernel(
    w_ptr,
    sf_ptr,
    sel_ptr,
    sink_ptr,
    per_w,
    per_sf,
    piece,
    n_items,
    budget_ns,
    NPW: tl.constexpr,
    NPP: tl.constexpr,
    G: tl.constexpr,
    TIMED: tl.constexpr,
):
    """CTA c issues items c, c + G, ... in rank order (``_issue``) from thread 0 (one thread per CTA in a loop issues
    faster than several lanes of one instruction). TIMED: a CTA stops issuing once ``budget_ns`` have passed since
    it started (%globaltimer); the prefetches already issued still land."""
    c = tl.program_id(0)
    lane = tl.arange(0, 32)
    acc = tl.zeros([32], tl.int32)
    if TIMED:
        t0 = _globaltimer()
        t = t0
        k = c
        while (k < n_items) & (t - t0 < budget_ns):
            acc += _issue(w_ptr, sf_ptr, sel_ptr, per_w, per_sf, piece, k, lane, NPW, NPP)
            k += G
            t = _globaltimer()
        acc += (t - t0).to(tl.int32)
    else:
        for k in range(c, n_items, G):
            acc += _issue(w_ptr, sf_ptr, sel_ptr, per_w, per_sf, piece, k, lane, NPW, NPP)
    tl.store(sink_ptr + c * 32 + lane, acc)


class _DeviceState:
    def __init__(self, device: torch.device, num_layers: int, num_experts: int) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the expert prefetch state must exist before CUDA-graph capture (run one eager "
                "forward first)"
            )
        self.sel = torch.full((num_layers, num_experts), -1, dtype=torch.int32, device=device)
        self.sink = torch.zeros(1 << 12, dtype=torch.int32, device=device)


class ExpertL2Prefetch:
    """Cross-step expert prefetch of one model (module docstring), for forwards of ``min_rows``..``max_rows`` rows
    of a router picking ``top_k`` experts per row, on the model's side ``stream``. Budgeted forwards, of more than
    ``budget_rows[0]`` and at most ``budget_rows[1]`` rows, record the experts with the most routed rows instead of
    token tiles and stop each layer's prefetch after its issue budget.

    Layers ``register`` the MoE backend that holds their routed FC1 weights (``w3_w1_weight``,
    ``w3_w1_weight_scale``: [local experts, ...], contiguous), and the ns after which their prefetch CTAs stop
    issuing (0: no limit)."""

    # Each layer prefetches the experts of this many rows' picks, from one CTA per two experts: a slower, gentler
    # burst than the whole grid, so the prefetch competes less with the latency-bound chain for DRAM.
    PICK_ROWS = 4

    _state: Optional[_DeviceState] = None
    _forked = False  # this forward forked the side stream; join() clears it
    budget_rows = (0, 0)
    num_layers = 0
    min_rows = 0
    max_rows = 0
    num_experts = 0
    num_ctas = 0
    stream: Optional[torch.cuda.Stream] = None
    _backends: Mapping[int, torch.nn.Module] = MappingProxyType({})
    _budget_ns: Mapping[int, int] = MappingProxyType({})

    def __init__(
        self,
        num_layers: int,
        min_rows: int,
        max_rows: int,
        top_k: int,
        stream: torch.cuda.Stream,
        budget_rows: Tuple[int, int] = (0, 0),
    ) -> None:
        self.num_layers = num_layers
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.budget_rows = budget_rows
        self.num_experts = min(self.PICK_ROWS * top_k, NUM_LOCAL_MAX)
        self.num_ctas = -(-self.num_experts // 2)
        self.stream = stream
        self._backends = {}
        self._budget_ns = {}

    def register(self, layer_idx: int, backend: torch.nn.Module, issue_budget_ns: int = 0) -> None:
        if 0 <= layer_idx < self.num_layers:
            self._backends[layer_idx] = backend
            self._budget_ns[layer_idx] = issue_budget_ns

    def unregister(self, layer_idx: int) -> None:
        """Neither record nor prefetch for ``layer_idx`` any more: for a layer whose decode front never records
        its experts, a prefetch would only launch on an empty or stale selection.

        Args:
            layer_idx: The layer registered with :meth:`register`.
        """
        self._backends.pop(layer_idx, None)
        self._budget_ns.pop(layer_idx, None)

    def registered(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` records and prefetches (registered and not unregistered)."""
        return layer_idx in self._backends

    def takes_rows(self, rows: int) -> bool:
        """Whether a forward of ``rows`` rows records and prefetches. Side-stream forks and Triton launches: not
        under torch.compile, which is tested first so that a symbolic row count is not compared (dynamo would
        guard on the row bounds and recompile when a count crosses one)."""
        return not torch.compiler.is_compiling() and self.min_rows <= rows <= self.max_rows

    def budgeted(self, rows: int) -> bool:
        lo, hi = self.budget_rows
        return lo < rows <= hi

    def _device_state(self, device: torch.device) -> _DeviceState:
        if self._state is None:
            self._state = _DeviceState(device, self.num_layers, self.num_experts)
        return self._state

    def _fork(self) -> None:
        self.stream.wait_stream(torch.cuda.current_stream())
        self._forked = True

    def prefetch(self, layer_idx: int, rows: int) -> None:
        """Issue the FC1 prefetch of the experts recorded for ``layer_idx`` at the previous step (on the side
        stream), for a forward of ``rows`` rows."""
        backend = self._backends.get(layer_idx)
        # NVFP4 experts only: the pieces are FC1 weights and their block scales.
        if backend is None or not self.takes_rows(rows) or not backend.has_nvfp4:
            return
        w, sf = backend.w3_w1_weight, backend.w3_w1_weight_scale
        if (
            w.dim() != 3
            or w.shape[0] > NUM_LOCAL_MAX
            or not w.is_contiguous()
            or not sf.is_contiguous()
        ):
            return
        wb, sb = (
            w.view(torch.uint8).view(w.shape[0], -1),
            sf.view(torch.uint8).view(sf.shape[0], -1),
        )
        per_w, per_sf = wb.shape[1], sb.shape[1]
        if per_w % PIECE or per_sf % PIECE:
            return
        state = self._device_state(w.device)
        npw, nps = per_w // PIECE, per_sf // PIECE
        budget_ns = self._budget_ns[layer_idx] if self.budgeted(rows) else 0
        self._fork()
        with torch.cuda.stream(self.stream):
            _prefetch_kernel[(self.num_ctas,)](
                wb,
                sb,
                state.sel[layer_idx],
                state.sink,
                per_w,
                per_sf,
                PIECE,
                self.num_experts * (npw + nps),
                budget_ns,
                NPW=npw,
                NPP=npw + nps,
                G=self.num_ctas,
                TIMED=budget_ns > 0,
                num_warps=1,
            )

    def record(
        self,
        layer_idx: int,
        cta_batch: torch.Tensor,
        cta_limit: torch.Tensor,
        num_ctas: torch.Tensor,
        tile: int,
        num_local: int,
        keep: torch.Tensor,
        rows: int,
    ) -> None:
        """Store the top experts of this call's routing tables for ``tile`` (``cta_batch``: local expert per token
        tile, ``cta_limit``: its row limit in the padded permuted order, ``num_ctas``: tile count) for ``layer_idx``'s
        next prefetch. ``keep``: the tensor that owns the tables, held for the side stream's read."""
        if (
            layer_idx not in self._backends
            or not self.takes_rows(rows)
            or num_local > NUM_LOCAL_MAX
        ):
            return
        state = self._device_state(cta_batch.device)
        by_rows = self.budgeted(rows)
        entries = cta_batch.numel() * tile if by_rows else cta_batch.numel()
        block = max(256, triton.next_power_of_2(entries))
        self._fork()
        with torch.cuda.stream(self.stream):
            _record_kernel[(1,)](
                cta_batch,
                num_ctas,
                cta_limit,
                state.sel[layer_idx],
                K=self.num_experts,
                NL=NUM_LOCAL_MAX,
                BLOCK=block,
                BY_ROWS=by_rows,
                TILE=tile,
                num_warps=4,
            )
        keep.record_stream(self.stream)

    def join(self) -> None:
        """Join the side stream into the current stream (once per forward, after the last decoder layer). Only a
        stream this forward forked: a CUDA-graph capture may not wait on a stream it never forked."""
        if self._forked:
            torch.cuda.current_stream().wait_stream(self.stream)
            self._forked = False
