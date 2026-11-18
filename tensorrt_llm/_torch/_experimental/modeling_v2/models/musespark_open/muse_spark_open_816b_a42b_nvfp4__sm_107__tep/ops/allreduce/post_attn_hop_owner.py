# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The wide hop's scatter shot and join on the row's owner only (``row_owner_moe``).

``post_attn_hop_wide`` runs the o_proj all-reduce as a two-shot (row ``t`` owned by rank
``t % ranks``: scatter to the owner, owner sums, gather back to every rank) and then the
gated residual join + pre-FFN norm on every row of every rank. The row-owner MoE island
keeps the per-row work of a decoder layer on the row's owner between the phases of the
exchanges, so this entry stops after the first shot: a non-owner publishes its partial
into the owner's scatter slab and is done; the owner sums the ranks' partials (the wide
hop's arithmetic: BF16 ``-0`` cleared, FP32 from ``+0`` in rank order, one RN round) and
runs the wide hop's join on that row, statement for statement, reading the row's FP32
residual at its compact row ``t // ranks`` and storing the updated residual and the
normalized BF16 row at the same compact row. Every value the owner stores is therefore
bitwise what the wide hop stores for row ``t`` (same kernel body, same thread map, same
cluster reduction order); the gather shot and its ingress (``(ranks - 1) / ranks`` of the
rows at 16 KiB each) are gone, and so is the join on the ``(ranks - 1) / ranks`` rows a
rank does not own.

Transport: the wide hop's two-shot scatter, unchanged (Lamport sentinel ``0x80000000``,
stale-only polling, reset by the reader, buffer = the call's parity from the arrival
counter). Buffer reuse without the gather back-edge: every call has at least ``ranks``
rows, so every rank owns a row and its call completes only after every peer's scatter of
that call arrived; a peer issues its scatter of call ``n + 2`` only after its call
``n + 1`` completed, hence after this rank's scatter of call ``n + 1``, which this rank
issues only after its call ``n`` (with the resets of its scatter slots) completed. So a
scatter slot is never rewritten while it holds unread data. Calls must be issued in the
same order with the same row counts on every rank, serialized on one stream.

Workspace per rank: the scatter area ``2 * ranks * (max_rows / ranks) * 16 KiB`` (16 MiB at
1024 rows), its own fabric allocation and counter, never shared with the wide hop.
"""

from __future__ import annotations

import functools
from typing import Optional, Sequence, Tuple

import torch

from tensorrt_llm.mapping import Mapping

from . import post_attn_hop as _hop
from . import post_attn_hop_wide as _wide

HIDDEN = _wide.HIDDEN
ROW_BYTES = _wide.ROW_BYTES
BUFFERS = _wide.BUFFERS
CLUSTER = _wide.CLUSTER
#: Rows of a call: at least one row per rank (the buffer-reuse argument); the row-owner
#: island serves decode steps of 584..768 rows (extensible to 1024 rows, 128 requests).
MIN_ROWS = 64
MAX_ROWS = _wide.WIDE_MAX_ROWS
HAS_CUTE_DSL = _wide.HAS_CUTE_DSL


def scatter_bytes(ranks: int, max_rows: int) -> int:
    """Bytes of one rank's receive memory: the two-shot scatter area only."""
    return BUFFERS * ranks * (max_rows // ranks) * ROW_BYTES


if HAS_CUTE_DSL:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as cutlass_utils
    from cutlass import Float32, Int32, Int64, Uint32
    from cutlass.cute.runtime import make_ptr

    from .post_attn_hop import (
        _arrive,
        _load_counter,
        _load_x4,
        _poll_x4,
        _reset_x4,
        _sanitize_bf16x2,
        _stale_bit,
        _store_peer_x4,
        _unpack_bf16x2,
    )
    from .post_attn_hop_wide import _pack_bf16x2

    class _OwnerHopKernel(_wide._WideHopKernel):
        """The wide hop's one-row kernel, two-shot scatter + owner join only.

        Grid ``rows * cluster`` CTAs as the wide hop (the non-owner CTAs publish only).
        ``residual``, ``output`` and ``next_output`` are the owner's compact rows.
        """

        def __init__(
            self,
            ranks: int,
            post_eps: float,
            pre_eps: float,
            norm_fp32: bool,
            pdl: bool,
            max_rows: int,
            cluster: int = CLUSTER,
        ):
            super().__init__(
                ranks,
                False,
                post_eps,
                pre_eps,
                norm_fp32,
                pdl,
                True,
                0,
                False,
                max_rows,
                0,
                cluster,
            )
            # The scatter area starts the receive memory (no one-shot slabs, no gather).
            assert self.scatter_base == 0

        @cute.kernel
        def kernel(
            self,
            partial: cute.Pointer,
            residual: cute.Pointer,
            alpha: cute.Pointer,
            beta: cute.Pointer,
            weight: cute.Pointer,
            output: cute.Pointer,
            next_output: cute.Pointer,
            control: cute.Pointer,
            receive: cute.Pointer,
            mcast: cute.Pointer,
            peer0: cute.Pointer,
            peer1: cute.Pointer,
            peer2: cute.Pointer,
            peer3: cute.Pointer,
            peer4: cute.Pointer,
            peer5: cute.Pointer,
            peer6: cute.Pointer,
            peer7: cute.Pointer,
            rows: Int32,
            rank: Int32,
            dry: Int32,
        ):
            peers = (peer0, peer1, peer2, peer3, peer4, peer5, peer6, peer7)
            tid = cute.arch.thread_idx()[0]
            crank = cute.arch.block_idx()[0] % self.cluster
            row = cute.arch.block_idx()[0] // self.cluster
            offset = cute.assume((crank * self.threads + tid) * 8, divby=8)
            smem = cutlass_utils.SmemAllocator()
            partials = smem.allocate_tensor(
                Float32, cute.make_layout((self.slots * 4 * 2,)), byte_alignment=16
            )
            bars = smem.allocate_tensor(cutlass.Uint64, cute.make_layout((2,)), byte_alignment=8)
            if dry == 0:
                control_address = control.toint()
                if tid < 2:
                    cute.arch.mbarrier_init(bars.iterator + tid, 1)
                    cute.arch.mbarrier_arrive_and_expect_tx(bars.iterator + tid, self.slots * 16)
                cute.arch.mbarrier_init_fence()
                cute.arch.barrier()
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_wait()

                counter = _load_counter(control_address + 4)
                buffer = Int64((counter >> self.grid_shift) & Uint32(1))
                slot = (Int64(row) * HIDDEN + Int64(offset)) * 2
                words = _load_x4(partial.toint() + slot)
                own = (
                    _sanitize_bf16x2(words[0]),
                    _sanitize_bf16x2(words[1]),
                    _sanitize_bf16x2(words[2]),
                    _sanitize_bf16x2(words[3]),
                )
                receive_address = receive.toint()
                owner = row % self.ranks
                compact = row // self.ranks
                source_bytes = self.scatter_rows * ROW_BYTES
                scatter = (
                    buffer * (self.ranks * source_bytes)
                    + Int64(compact) * ROW_BYTES
                    + Int64(offset) * 2
                )
                if owner != rank:
                    # Publish this rank's partial into its slab on the owner; nothing else.
                    for p in cutlass.range_constexpr(self.ranks):
                        if owner == p:
                            _store_peer_x4(
                                peers[p].toint() + scatter + Int64(rank) * source_bytes, *own
                            )
                    if cutlass.const_expr(self.pdl):
                        cute.arch.griddepcontrol_launch_dependents()
                else:
                    a = self.load(alpha + offset, Float32)
                    b = self.load(beta + offset, Float32)
                    w = self.load(weight + offset, cutlass.BFloat16)
                    r = self.load(residual + compact * HIDDEN + offset, Float32)
                    values = cute.make_rmem_tensor(cute.make_layout((self.ranks, 4)), Uint32)
                    for p in cutlass.range_constexpr(self.ranks):
                        v0, v1, v2, v3 = _poll_x4(
                            receive_address + scatter + p * source_bytes,
                            Uint32(rank != p),
                            own[0],
                            own[1],
                            own[2],
                            own[3],
                        )
                        values[p, 0] = v0
                        values[p, 1] = v1
                        values[p, 2] = v2
                        values[p, 3] = v3
                    stale = Uint32(0)
                    for p in cutlass.range_constexpr(self.ranks):
                        stale = stale | _stale_bit(
                            values[p, 0],
                            values[p, 1],
                            values[p, 2],
                            values[p, 3],
                            Uint32(1 << p),
                        )
                    while stale != Uint32(0):
                        polled = stale
                        for p in cutlass.range_constexpr(self.ranks):
                            v0, v1, v2, v3 = _poll_x4(
                                receive_address + scatter + p * source_bytes,
                                (polled >> Uint32(p)) & Uint32(1),
                                values[p, 0],
                                values[p, 1],
                                values[p, 2],
                                values[p, 3],
                            )
                            values[p, 0] = v0
                            values[p, 1] = v1
                            values[p, 2] = v2
                            values[p, 3] = v3
                        stale = Uint32(0)
                        for p in cutlass.range_constexpr(self.ranks):
                            stale = stale | _stale_bit(
                                values[p, 0],
                                values[p, 1],
                                values[p, 2],
                                values[p, 3],
                                Uint32(1 << p),
                            )
                    for p in cutlass.range_constexpr(self.ranks):
                        _reset_x4(receive_address + scatter + p * source_bytes, Uint32(rank != p))
                    if cutlass.const_expr(self.pdl):
                        cute.arch.griddepcontrol_launch_dependents()
                    # The wide hop's sum and RN round (packed, then unpacked: the same bits).
                    sums = [Float32(0.0)] * 8
                    for p in cutlass.range_constexpr(self.ranks):
                        for k in cutlass.range_constexpr(4):
                            lo, hi = _unpack_bf16x2(values[p, k])
                            sums[2 * k] = sums[2 * k] + lo
                            sums[2 * k + 1] = sums[2 * k + 1] + hi
                    x = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), cutlass.BFloat16)
                    for k in cutlass.range_constexpr(4):
                        packed = _pack_bf16x2(sums[2 * k], sums[2 * k + 1])
                        low, high = _unpack_bf16x2(packed)
                        x[2 * k, 0] = cutlass.BFloat16(low)
                        x[2 * k + 1, 0] = cutlass.BFloat16(high)

                    # The wide hop's join, statement for statement (gain off).
                    scaled = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), Float32)
                    updated = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), Float32)
                    normed = cute.make_rmem_tensor(
                        cute.make_layout((8, self.iters)), cutlass.BFloat16
                    )
                    total = Float32(0)
                    for p in cutlass.range_constexpr(self.iters):
                        for v in cutlass.range_constexpr(8):
                            value = Float32(x[v, p])
                            scaled[v, p] = value
                            total += value * value
                    total = self.reduce(total, partials, bars, 0)
                    inv = cute.math.rsqrt(
                        total / Float32(8192) + Float32(self.post_eps), fastmath=True
                    )
                    total = Float32(0)
                    for p in cutlass.range_constexpr(self.iters):
                        for v in cutlass.range_constexpr(8):
                            value = Float32(cutlass.BFloat16(scaled[v, p] * inv))
                            u = a[v, p] * r[v, p] + b[v, p] * value
                            updated[v, p] = u
                            if cutlass.const_expr(self.norm_fp32):
                                scaled[v, p] = u
                                total += u * u
                            else:
                                rounded = Float32(cutlass.BFloat16(u))
                                scaled[v, p] = rounded
                                total += rounded * rounded
                    total = self.reduce(total, partials, bars, 1)
                    inv = cute.math.rsqrt(
                        total / Float32(8192) + Float32(self.pre_eps), fastmath=True
                    )
                    for p in cutlass.range_constexpr(self.iters):
                        for v in cutlass.range_constexpr(8):
                            normed[v, p] = cutlass.BFloat16(
                                scaled[v, p] * inv * (Float32(w[v, p]) + Float32(1))
                            )
                    layout = cute.make_layout(
                        (8, self.iters), stride=(1, self.threads * self.cluster * 8)
                    )
                    atom32 = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
                    )
                    atom16 = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
                    )
                    cute.copy(
                        atom32,
                        updated,
                        cute.make_tensor((output + compact * HIDDEN + offset).align(16), layout),
                    )
                    cute.copy(
                        atom16,
                        normed,
                        cute.make_tensor(
                            (next_output + compact * HIDDEN + offset).align(16), layout
                        ),
                    )
                cute.arch.barrier()
                if tid == 0:
                    block = cute.arch.block_idx()[0]
                    blocks = cute.arch.grid_dim()[0]
                    amount = Uint32(1)
                    if block == 0:
                        amount = Uint32(self.grid_step + 1 - blocks)
                    _arrive(control_address + 4, amount)

    @functools.lru_cache(maxsize=16)
    def _compiled_owner(
        device_index: int,
        ranks: int,
        post_eps: float,
        pre_eps: float,
        norm_fp32: bool,
        pdl: bool,
        max_rows: int,
        cluster: int,
    ):
        kernel = _OwnerHopKernel(ranks, post_eps, pre_eps, norm_fp32, pdl, max_rows, cluster)
        with torch.cuda.device(device_index):
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            return cute.compile(
                kernel,
                *[
                    make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
                    for dtype in _wide._POINTER_DTYPES
                ],
                Int32(MIN_ROWS),
                Int32(0),
                Int32(1),
                stream,
                options="--enable-tvm-ffi",
            )


class OwnerHopWorkspace:
    """Transport memory, counter and kernels of the owner hop for one TP group.

    Construction is collective (fabric allocation, sentinel fill, compilation and a dry
    launch per variant) and succeeds or raises on every rank alike.
    """

    def __init__(
        self,
        mapping: Mapping,
        *,
        variants: Sequence[Tuple[float, float, bool]] = (),
        max_rows: int = MAX_ROWS,
        cluster: int = CLUSTER,
    ):
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
            _mnnvl_workspace_all_succeeded,
        )

        self.ranks = mapping.tp_size
        self.rank = mapping.tp_rank
        if self.ranks not in (4, _hop.MAX_RANKS):
            raise ValueError(f"the owner hop serves 4 or 8 ranks, got {self.ranks}")
        if max_rows not in _wide.CAPACITIES or max_rows % self.ranks:
            raise ValueError(f"max_rows must be one of {_wide.CAPACITIES}, got {max_rows}")
        self.max_rows = int(max_rows)
        self.cluster = int(cluster)
        self.pdl = _wide._LAUNCH_PDL
        self.workspace_bytes = scatter_bytes(self.ranks, self.max_rows)
        self.device = torch.device("cuda", _mnnvl_device_index(mapping))
        self._kernels = {}
        self._comm = _get_mnnvl_workspace_comm(mapping)
        error: Optional[BaseException] = None
        try:
            if not HAS_CUTE_DSL:
                raise RuntimeError("CuTe DSL is unavailable")
            if torch.cuda.get_device_capability(self.device) != (10, 7):
                raise RuntimeError("the owner hop runs on SM107 only")
            self._handle = _make_mnnvl_mcast_buffer(
                self._comm, self.workspace_bytes, mapping, mapping.is_multi_node()
            )
            words = self.workspace_bytes // 4
            self._uc = [
                self._handle.get_uc_buffer(peer, (words,), torch.int32, 0)
                for peer in range(self.ranks)
            ]
            with torch.cuda.device(self.device):
                self._uc[self.rank].fill_(-(2**31))
                self.control = torch.zeros(
                    _wide.CONTROL_WORDS, dtype=torch.int32, device=self.device
                )
                torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "owner hop workspace failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error
        receive = self._uc[self.rank].data_ptr()
        peers = [t.data_ptr() for t in self._uc] + [receive] * (_hop.MAX_RANKS - self.ranks)
        self._fixed = (self.control.data_ptr(), receive, receive, *peers)
        error = None
        try:
            for post_eps, pre_eps, norm_fp32 in variants:
                self._kernel(post_eps, pre_eps, norm_fp32)
            torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "owner hop kernel compilation failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error

    def _kernel(self, post_eps: float, pre_eps: float, norm_fp32: bool):
        key = (float(post_eps), float(pre_eps), bool(norm_fp32))
        compiled = self._kernels.get(key)
        if compiled is not None:
            return compiled
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"owner hop variant {key} was not built before graph capture")
        compiled = _compiled_owner(
            self.device.index, self.ranks, *key, self.pdl, self.max_rows, self.cluster
        )
        dummy = self.control.data_ptr()
        compiled(
            *([dummy] * 7),
            *self._fixed,
            MIN_ROWS,
            self.rank,
            1,
            cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream),
        )
        self._kernels[key] = compiled
        return compiled

    def launch(
        self,
        partial: torch.Tensor,
        residual_own: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        next_weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
        norm_fp32: bool,
        updated: Optional[torch.Tensor] = None,
        normalized: Optional[torch.Tensor] = None,
    ):
        """``partial`` [R, 8192] BF16 (this rank's o_proj partial, every row); ``residual_own``
        [R / ranks, 8192] FP32 (the owner's rows ``t = ranks * i + rank``). Returns the owner's
        updated FP32 residual rows and normalized BF16 rows, both [R / ranks, 8192]."""
        rows = partial.shape[0]
        if not (MIN_ROWS <= rows <= self.max_rows) or rows % self.ranks:
            raise ValueError(
                f"owner hop: {rows} rows (capacity {self.max_rows}, multiple of {self.ranks})"
            )
        own = rows // self.ranks
        if residual_own.shape[0] < own:
            raise ValueError("residual_own holds fewer than rows / ranks rows")
        compiled = self._kernel(post_eps, pre_eps, norm_fp32)
        if updated is None:
            updated = torch.empty((own, HIDDEN), dtype=torch.float32, device=self.device)
        if normalized is None:
            normalized = torch.empty((own, HIDDEN), dtype=torch.bfloat16, device=self.device)
        compiled(
            partial.data_ptr(),
            residual_own.data_ptr(),
            alpha.data_ptr(),
            beta.data_ptr(),
            next_weight.data_ptr(),
            updated.data_ptr(),
            normalized.data_ptr(),
            *self._fixed,
            rows,
            self.rank,
            0,
            cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream),
        )
        return updated, normalized
