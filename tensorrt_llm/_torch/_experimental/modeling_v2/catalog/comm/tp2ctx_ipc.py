# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_SP / AR1 workspace on MPI IPC: two-rank prefill boundary collectives without torch.distributed.

The VR tp2ctx CTX workers are MPI-launched and have no torch process group, so the peer buffers come from TRT's
``IpcMemory`` (cudaIpc handles exchanged over the TP communicator) and the cross-rank sync is a one-CTA device-flag
barrier (``torch.ops.trtllm.minimax_m31_tp2ctx_ipc_barrier``, ~4.8 us on VR). Microbench (MPI only, T x 6144 bf16):
the sequence-parallel boundary (reduce-scatter of the owned rows + all-gather of the packed MXFP8 rows) is 91.9 vs
127.5 us at 8K and 163.2 vs 233.7 us at 16K against the NCCL ring all-reduce, bit for bit (the two-operand bf16 sum
is commutative).

Row ownership is fixed for a step: rank 0 owns rows [0, S), rank 1 [S, T), where S is the 128-row boundary nearest
T / 2 (S = T / 2 whenever T % 256 == 0). Rank 0's rows are then whole 128-row MXFP8 scale-factor tiles and rank 1's
block starts on a tile, so each rank's swizzled scale bytes are one contiguous range of the full step's
(``own_sf_slice``); rank 1's last tile carries the same zero pad rows as the full step's quantize. Per boundary:
the producer writes its partial into ``input_slot(T)``; ``reduce_scatter_owned`` (barrier, owned-row peer add); the
caller writes its owned outputs into ``gather_view(kind, T)`` at ``own_rows(T)`` (the scale factors at
``own_sf_slice(T)``); ``push_owned(kinds, T)`` pushes them into the peer's views and barriers.
Two barriers per boundary make single-buffered slots safe: a peer's read of buffer n finishes before
either rank passes the next barrier, and every overwrite of buffer n + 1 comes after it. Eager prefill only. Kill
switch: TRTLLM_M31_DISABLE_TP2CTX_SP=1.
"""

from __future__ import annotations

import os

import torch

_DEFAULT_FUSION_WORKSPACE_BYTES = 64 * 1024 * 1024
_SF_VEC = 32  # MXFP8 scale-factor block
_SF_TILE_ROWS = 128  # swizzled scale-factor tile: 128 rows x 4 scale columns


def enabled() -> bool:
    """Default ON; TRTLLM_M31_DISABLE_TP2CTX_SP=1 restores the all-reduce boundary."""
    return os.environ.get("TRTLLM_M31_DISABLE_TP2CTX_SP", "0") != "1"


# TP2CTX_ROUTER_ZC (prefill): the pre-MoE boundary writes its fp32 router rows straight into this
# rank's rows of the router's full-step buffer (``router_owned_input``), and ``route_owned_full_shape`` no longer
# zeroes the [T, K] buffer or copies the owned rows in per call (a 165K-token prefill: 570 x 201 MB copies + 570
# x 402 MB zero fills per request). The rows outside this rank's own hold earlier router inputs (or the zeros of
# the allocation): each output row of the router GEMM depends only on its own input row, and the call's shape and
# algorithm are unchanged, so the owned logits stay bit for bit. Default ON; TRTLLM_M31_DISABLE_TP2CTX_ROUTER_ZC=1
# restores the per-call zero + copy. Read once at import.
_ROUTER_ZC = os.environ.get("TRTLLM_M31_DISABLE_TP2CTX_ROUTER_ZC", "0") != "1"


def router_zc_enabled() -> bool:
    return _ROUTER_ZC


def _nccl_fallback_bytes() -> int:
    env = os.environ.get("TRTLLM_ALLREDUCE_FUSION_WORKSPACE_SIZE")
    return int(env) if env else _DEFAULT_FUSION_WORKSPACE_BYTES


def _view(ptr: int, shape, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """A tensor over raw device memory owned by an IpcMemory (no copy, no ownership)."""
    nbytes = 1
    for d in shape:
        nbytes *= d
    nbytes *= torch.empty((), dtype=dtype).element_size()

    class _Iface:
        __cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }

    return torch.as_tensor(_Iface(), device=device).view(dtype).view(shape)


# kind -> (bytes per row, layout); every layout is row-major in whole rows except the swizzled MXFP8 scale
# factors, whose 128-row tiles are contiguous blocks (owned-half quantize == the full step's block:
# UT test_owned_half_quantize_...).
_KINDS = {
    "mxfp8": "hidden",  # e4m3 data [T, hidden]
    "mxfp8_sf": "sf_swizzled",  # UE8M0, swizzled 128x4, K32
    "nvfp4": "hidden_half",  # packed e2m1 [T, hidden / 2]
    "nvfp4_sf": "nvfp4_sf",  # E4M3, linear [T, hidden / 16]
    "topk_ids": "topk",  # int32 [T, 4]
    "topk_w": "topk",  # fp32 [T, 4]
    "bf16": "bf16",  # [T, hidden]
    "logits": "logits",  # fp32 [T, 128]: router logits (TP2AGG_SP, whose trtllm-gen experts route in-kernel)
}
_LOGITS_COLS = 128


class Tp2ctxIpcWorkspace:
    """Per-process two-rank IPC workspace; build it collectively (both TP ranks) at model init, outside capture."""

    def __init__(self, mapping, max_tokens: int, hidden: int, device: torch.device):
        from tensorrt_llm._ipc_utils import IpcMemory

        if mapping.tp_size != 2:
            raise ValueError(f"Tp2ctxIpcWorkspace needs TP2, got {mapping.tp_size}")
        self.rank = mapping.tp_rank
        self.peer = 1 - self.rank
        self.max_tokens, self.hidden, self.device = max_tokens, hidden, device
        sf_cols_pad = (hidden // _SF_VEC + 3) // 4 * 4
        self._row_bytes = {
            "hidden": hidden,
            "sf_swizzled": sf_cols_pad,
            "hidden_half": hidden // 2,
            "nvfp4_sf": hidden // 16,
            "topk": 16,
            "bf16": hidden * 2,
            "logits": _LOGITS_COLS * 4,
        }
        rows_pad = (max_tokens + _SF_TILE_ROWS - 1) // _SF_TILE_ROWS * _SF_TILE_ROWS
        sizes = {"slot": max_tokens * hidden * 2}
        for kind, layout in _KINDS.items():
            sizes[kind] = (rows_pad if layout == "sf_swizzled" else max_tokens) * self._row_bytes[
                layout
            ]
        self._mem = {k: IpcMemory(mapping, v) for k, v in sizes.items()}
        self._flags = IpcMemory(mapping, 4096)  # zeroed by IpcMemory
        self._state = torch.zeros(4, dtype=torch.int32, device=device)
        self._router_buf = None
        self._side = None
        self._pending_push = None

    def _ptr(self, kind: str, peer: bool = False) -> int:
        m = self._mem[kind]
        return m.peer_ptrs[self.peer] if peer else m.local_ptr

    def supports(self, num_tokens: int, *, nccl_only: bool = True) -> bool:
        """Storage capacity, optionally restricted to rows above the fused all-reduce crossover."""
        return 2 * _SF_TILE_ROWS <= num_tokens <= self.max_tokens and (
            not nccl_only or num_tokens * self.hidden * 2 > _nccl_fallback_bytes()
        )

    @staticmethod
    def split_row(num_tokens: int) -> int:
        """The first row of rank 1: the 128-row boundary nearest T / 2 (exactly T / 2 when T % 256 == 0)."""
        split = (num_tokens // 2 + _SF_TILE_ROWS // 2) // _SF_TILE_ROWS * _SF_TILE_ROWS
        return min(max(split, _SF_TILE_ROWS), num_tokens - 1)

    def own_rows(self, num_tokens: int):
        """(first owned row, owned row count): rank 0 owns [0, S), rank 1 [S, T) (S = ``split_row(T)``)."""
        split = self.split_row(num_tokens)
        return (0, split) if self.rank == 0 else (split, num_tokens - split)

    def own_sf_slice(self, num_tokens: int) -> slice:
        """This rank's byte range of the swizzled MXFP8 scale factors in ``gather_view("mxfp8_sf", T)``: the 128-row
        tiles of its rows (rank 1's last tile includes the pad rows up to the full step's 128-row multiple), i.e.
        exactly the bytes its own rows' quantize writes."""
        per = self._row_bytes["sf_swizzled"]
        split = self.split_row(num_tokens)
        rows_pad = (num_tokens + _SF_TILE_ROWS - 1) // _SF_TILE_ROWS * _SF_TILE_ROWS
        return slice(0, split * per) if self.rank == 0 else slice(split * per, rows_pad * per)

    def barrier(self) -> None:
        self.wait_pending_push()
        torch.ops.trtllm.minimax_m31_tp2ctx_ipc_barrier(
            self._flags.local_ptr, self._flags.peer_ptrs[self.peer], self._state, self.rank
        )

    def wait_pending_push(self) -> None:
        """Make the current stream wait for an outstanding ``push_owned_async`` (its pushes and barrier)."""
        event, self._pending_push = self._pending_push, None
        if event is not None:
            torch.cuda.current_stream(self.device).wait_event(event)

    def peer_sf_slice(self, num_tokens: int) -> slice:
        """The peer's byte range of the swizzled MXFP8 scale factors (the complement of ``own_sf_slice``)."""
        own = self.own_sf_slice(num_tokens)
        rows_pad = (num_tokens + _SF_TILE_ROWS - 1) // _SF_TILE_ROWS * _SF_TILE_ROWS
        full = rows_pad * self._row_bytes["sf_swizzled"]
        return slice(own.stop, full) if own.start == 0 else slice(0, own.start)

    def push_owned_async(
        self, kinds, num_tokens: int, engine: str = "ce", barrier: bool = True
    ) -> torch.cuda.Event:
        """``push_owned`` off the current stream: after the current stream's work so far, a side stream copies this
        rank's owned rows of each kind into the peer's views, by copy engine (``engine="ce"``, no SMs) or by the peer
        push kernel (``"sm"``), and runs the barrier unless ``barrier=False`` (then a later ``push_owned`` on the
        current stream ends the boundary with it). Returns the event a consumer of the gathered rows waits on; the next
        ``barrier`` / ``push_owned`` on the current stream waits on it too, so the barrier epochs stay in order."""
        if engine not in ("ce", "sm"):
            raise ValueError(f"push_owned_async: engine must be 'ce' or 'sm', got {engine!r}")
        from cuda.bindings import driver as cuda_driver

        self.wait_pending_push()
        main = torch.cuda.current_stream(self.device)
        if self._side is None:
            self._side = torch.cuda.Stream(device=self.device)
        side = self._side
        side.wait_stream(main)
        first, count = self.own_rows(num_tokens)
        for kind in kinds:
            if _KINDS[kind] == "sf_swizzled":
                sl = self.own_sf_slice(num_tokens)
                off, nbytes = sl.start, sl.stop - sl.start
            else:
                per_row = self._row_bytes[_KINDS[kind]]
                off, nbytes = first * per_row, count * per_row
            if engine == "sm":
                with torch.cuda.stream(side):
                    src = _view(self._ptr(kind) + off, (nbytes,), torch.uint8, self.device)
                    torch.ops.trtllm.minimax_m31_tp2ctx_peer_push(
                        src, self._ptr(kind, peer=True) + off
                    )
                continue
            (err,) = cuda_driver.cuMemcpyAsync(
                self._ptr(kind, peer=True) + off, self._ptr(kind) + off, nbytes, side.cuda_stream
            )
            if err != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"Tp2ctxIpcWorkspace.push_owned_async: cuMemcpyAsync({kind}) failed: {err}"
                )
        if barrier:
            with torch.cuda.stream(side):
                torch.ops.trtllm.minimax_m31_tp2ctx_ipc_barrier(
                    self._flags.local_ptr, self._flags.peer_ptrs[self.peer], self._state, self.rank
                )
        event = torch.cuda.Event()
        event.record(side)
        self._pending_push = event
        return event

    def input_slot(self, num_tokens: int) -> torch.Tensor:
        """The [T, hidden] bf16 buffer the producer writes its partial into (zero-copy)."""
        return _view(
            self._ptr("slot"), (self.max_tokens, self.hidden), torch.bfloat16, self.device
        )[:num_tokens]

    def reduce_scatter_owned(self, slot: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        """The owned rows of the two-rank sum of ``slot`` (bitwise the all-reduce's rows), [T / 2, hidden] bf16."""
        first, count = self.own_rows(slot.shape[0])
        off = first * self.hidden * 2
        if out is None:
            out = torch.empty((count, self.hidden), dtype=torch.bfloat16, device=self.device)
        self.barrier()
        torch.ops.trtllm.minimax_m31_tp2ctx_peer_add(
            self._ptr("slot") + off, self._ptr("slot", peer=True) + off, out, count * self.hidden
        )
        return out

    def owned_slot_ptrs(self, num_tokens: int):
        """(this rank's, the peer's) device addresses of the owned rows of the input slot: the operands of
        ``reduce_scatter_owned``'s peer add, for an op that fuses the add into its consumer (call ``barrier()``
        first, as ``reduce_scatter_owned`` does)."""
        first, _ = self.own_rows(num_tokens)
        off = first * self.hidden * 2
        return self._ptr("slot") + off, self._ptr("slot", peer=True) + off

    def gather_view(self, kind: str, num_tokens: int) -> torch.Tensor:
        """The full-T local view of ``kind`` (IPC memory): producers write their owned rows into it, consumers read
        all of it after ``push_owned``."""
        layout = _KINDS[kind]
        ptr = self._ptr(kind)
        if layout == "sf_swizzled":
            rows_pad = (num_tokens + _SF_TILE_ROWS - 1) // _SF_TILE_ROWS * _SF_TILE_ROWS
            return _view(ptr, (rows_pad * self._row_bytes[layout],), torch.uint8, self.device)
        if kind == "mxfp8":
            return _view(ptr, (self.max_tokens, self.hidden), torch.uint8, self.device)[:num_tokens]
        if kind == "nvfp4":
            return _view(ptr, (self.max_tokens, self.hidden // 2), torch.uint8, self.device)[
                :num_tokens
            ]
        if kind == "nvfp4_sf":
            return _view(ptr, (self.max_tokens, self.hidden // 16), torch.uint8, self.device)[
                :num_tokens
            ]
        if kind == "topk_ids":
            return _view(ptr, (self.max_tokens, 4), torch.int32, self.device)[:num_tokens]
        if kind == "topk_w":
            return _view(ptr, (self.max_tokens, 4), torch.float32, self.device)[:num_tokens]
        if kind == "logits":
            return _view(ptr, (self.max_tokens, _LOGITS_COLS), torch.float32, self.device)[
                :num_tokens
            ]
        return _view(ptr, (self.max_tokens, self.hidden), torch.bfloat16, self.device)[:num_tokens]

    def push_owned(self, kinds, num_tokens: int) -> None:
        """Copy this rank's owned rows of each kind into the peer's view, then one barrier."""
        self.wait_pending_push()
        first, count = self.own_rows(num_tokens)
        for kind in kinds:
            if _KINDS[kind] == "sf_swizzled":
                sl = self.own_sf_slice(num_tokens)
                off, nbytes = sl.start, sl.stop - sl.start
            else:
                per_row = self._row_bytes[_KINDS[kind]]
                off, nbytes = first * per_row, count * per_row
            src = _view(self._ptr(kind) + off, (nbytes,), torch.uint8, self.device)
            torch.ops.trtllm.minimax_m31_tp2ctx_peer_push(src, self._ptr(kind, peer=True) + off)
        self.barrier()

    def _router_full(self, num_tokens: int, k: int) -> torch.Tensor:
        """The router's full-step [T, K] fp32 input buffer (allocated zeroed, reused across calls)."""
        buf = self._router_buf
        if buf is None or buf.shape[1] != k or buf.shape[0] < num_tokens:
            buf = self._router_buf = torch.zeros(
                (self.max_tokens, k), dtype=torch.float32, device=self.device
            )
        return buf[:num_tokens]

    def router_owned_input(self, num_tokens: int, k: int):
        """TP2CTX_ROUTER_ZC: this rank's rows of the router's full-step input buffer ([own count, K] fp32,
        contiguous), for the boundary to write its fp32 router rows into; None when the item is off."""
        if not _ROUTER_ZC:
            return None
        first, count = self.own_rows(num_tokens)
        return self._router_full(num_tokens, k)[first : first + count]

    def route_owned_full_shape(
        self, x_owned: torch.Tensor, weight: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Router logits of the owned rows, bitwise those of today's full-step router call.

        The router is a cuBLAS fp32 / TF32 GEMM whose algorithm depends on M (split-K at small M), so running it at
        M = T / 2 changes the K-reduction order. Instead the owned rows go to their own row offsets in a [T, K]
        buffer and the SAME call as today's (F.linear at M = T) runs: identical shape, identical algorithm, identical
        per-row result. The cost is today's full-step router GEMM. With TP2CTX_ROUTER_ZC an ``x_owned`` that already
        is ``router_owned_input(T, K)`` is used in place, and the other rows are not re-zeroed (see _ROUTER_ZC)."""
        first, count = self.own_rows(num_tokens)
        full = self._router_full(num_tokens, x_owned.shape[1])
        owned = full[first : first + count]
        if not _ROUTER_ZC:
            full.zero_()
        if x_owned.data_ptr() != owned.data_ptr():
            owned.copy_(x_owned)
        return torch.nn.functional.linear(full, weight)[first : first + count]


_WS: dict = {}


def get_workspace(
    mapping, max_tokens: int, hidden: int, device: torch.device
) -> Tp2ctxIpcWorkspace:
    """Process-wide workspace (created collectively on first use, outside CUDA-graph capture)."""
    key = (mapping.tp_rank, max_tokens, hidden, device.index)
    ws = _WS.get(key)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Tp2ctxIpcWorkspace: create the workspace outside CUDA-graph capture"
            )
        ws = _WS[key] = Tp2ctxIpcWorkspace(mapping, max_tokens, hidden, device)
    return ws
