# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused Kimi K3.1 latent-MoE tail for tensor-parallel decode.

Serves TP 4 (one GB300 tray), 8, 16 and 32.

Replaces, for ``M <= max_m`` tokens,

    latent = RMSNorm(AllReduce(routed_partial))          # [M, latent]
    out = latent @ up_weight.T + AllReduce(shared_partial)  # [M, hidden]

with three PDL-chained kernels:

1. ``CollectiveKernel``: one-shot multicast AllReduce of the routed latent
   partials with the RMSNorm fused in, plus a one-shot ReduceScatter of the
   shared-expert partials (rank ``r`` keeps hidden columns
   ``[r * S, (r + 1) * S)``, ``S = hidden / tp``).
2. The up-projection (``AdaptiveUpProjectionKernel`` at one token, the split-K
   ``SplitKMulticastUpProjection`` above): the rank's ``S`` rows of it
   (``S x latent`` weight bytes instead of ``hidden x latent``) with the shared
   shard added in the epilogue and the result stored through the multicast
   mapping of a symmetric mailbox, i.e. the all-gather is the GEMM store.
3. ``LamportCopyKernel``: waits for every shard of the local mailbox copy,
   copies it out and re-arms the mailbox.

The symmetric memory is one ``McastGPUBuffer`` (the MNNVL all-reduce
allocator), so the op works across NVLink-connected trays.

The collective also takes the routed experts' unfinalized output
(``UnfinalizedRoutedOutput``: the FC2 rows in permuted order, the routing
weights and the (token, slot) -> row map) and does the top-k weighted sum
while loading this rank's routed row, so the routed experts skip their
finalize kernel; the sum has the finalize's numerics (FP32 in slot order, one
BF16 rounding).

Numerics: every rank reduces in rank order in FP32 and rounds once to BF16,
like the MNNVL one-shot all-reduce; the RMSNorm squares the BF16-rounded sum
(as the MNNVL RMSNorm fusion does); the up-projection is rounded to BF16
before the BF16 shared shard is added in FP32. Only the up-projection's
accumulation order differs from an unsharded GEMM. With ``mnnvl_norm`` (on for
``MNNVL_NORM_TP_SIZES``) the rank sum, the sum of squares and the normalized
product follow the MNNVL all-reduce RMSNorm fusion's orders for the row count
(``mnnvl_norm``), so the normalized latent is bitwise the unfused path's.

At TP 32 the default tail is one-shot only (up to 32 rows). A second instance (``twoshot_only``) serves the
rows above it with the two-shot collective and ``mnnvl_norm``: the routed rows and the shared slices reduce in
the MNNVL two-shot all-reduce's rank chunks and the up-projection splits K as the unfused path's cuBLAS GEMM
does at those rows, so its output is bitwise the unfused path's.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import ClassVar, Dict, NamedTuple, Optional

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .moe_tail_routed_output import UnfinalizedRoutedOutput

# Workspace capacity in tokens: the largest generation batch of the Kimi K3.1
# TEP gens (192 tokens per step).
MAX_NUM_TOKENS = 192
# Largest token count the model dispatches to the fused tail, per TP size: the
# measured crossover with the unfused tail (TP16: every step of the 192-token
# gen batches).
_DISPATCH_MAX_NUM_TOKENS = {4: 128, 8: 128, 16: 192, 32: 32}
_SKINNY_MAX_NUM_TOKENS = 1
# Token tiles of the split-K up-projection (one compiled tactic per bucket).
_SPLITK_TOKEN_BUCKETS = (8, 16, 32, 64, 128)
# Routed experts per token: the top-k sum the finalize-on-load collectives fold.
_FINALIZE_TOP_K = 16
# Token counts up to here use the one-shot collective (every rank reduces every
# routed row: one cross-rank trip, ingress ~ tp * M * latent); larger ones the
# two-shot collective (each row reduced on one owner rank, then multicast: two
# trips, ingress ~ 2 * M * latent).
_ONESHOT_MAX_NUM_TOKENS = {4: 32, 8: 32, 16: 16, 32: 32}
# TP sizes whose tail is one-shot only: the workspace capacity in tokens.
_ONESHOT_ONLY_CAPACITY = {32: 32}
# TP sizes with a second, two-shot-only instance (``twoshot_only``) above the one-shot tail's rows: the
# (first, last) token counts it serves.
_TWOSHOT_ONLY_ROWS = {32: (33, 192)}
# Up-projection tactic buckets of the two-shot-only instance (one compiled tactic per bucket; the last serves any M
# above): the row bounds of ``splitk_multicast_gemm._WHOLE_K_TILES``. At its rows the unfused path's cuBLAS latent up
# projection accumulates the whole K in one pass (GB300, 33..192 rows), so the instance's GEMM does too.
_TWOSHOT_ONLY_SPLITK_BUCKETS = (192,)
# Enough CTAs for one 16-byte fragment per thread up to 128 tokens: the copy
# polls with volatile loads, so it is latency-bound on a few threads.
_LAMPORT_COPY_CTAS = 1024
_LAMPORT_COPY_THREADS = 128
_REGION_ALIGN = 4096


def _splitk_bucket(m: int, buckets: tuple = _SPLITK_TOKEN_BUCKETS) -> int:
    """Split-K tactic bucket of a token count (the largest serves any M above)."""
    return next((b for b in buckets if b >= m), buckets[-1])


def _align(value: int, alignment: int = _REGION_ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class LatentMoETailKey:
    mapping: Mapping
    hidden_size: int
    latent_size: int
    max_num_tokens: int
    rms_eps: float
    mnnvl_norm: bool = False
    twoshot_only: bool = False


# TP sizes whose unfused path reduces and normalizes the routed latent in the MNNVL all-reduce
# RMSNorm fusion, and whose tail reproduces that fusion's numerics.
MNNVL_NORM_TP_SIZES = frozenset({8, 16})


def fused_tail_max_tokens() -> int:
    """Workspace capacity in tokens."""
    return MAX_NUM_TOKENS


def fused_tail_dispatch_max_tokens(tp_size: int) -> int:
    """Largest token count the model sends to the fused tail."""
    return _DISPATCH_MAX_NUM_TOKENS.get(tp_size, 0)


def fused_tail_twoshot_only_rows(tp_size: int) -> Optional[tuple[int, int]]:
    """``(first, last)`` token counts of the two-shot-only tail instance at ``tp_size``, or None where the default
    tail serves every row it takes."""
    return _TWOSHOT_ONLY_ROWS.get(tp_size)


class FusedTailRows(NamedTuple):
    """The token counts of one tail instance."""

    min_tokens: int  # fewest tokens a call takes
    capacity: int  # most tokens a call takes (the workspace's rows)
    dispatch_max: int  # most tokens the model sends to it
    oneshot_max: int  # calls up to here take the one-shot collective, larger ones the two-shot


def fused_tail_rows(tp_size: int, max_num_tokens: int, twoshot_only: bool = False) -> FusedTailRows:
    """The token counts of the instance a ``LatentMoETailKey`` builds."""
    if twoshot_only:
        rows = fused_tail_twoshot_only_rows(tp_size)
        if rows is None or not rows[0] <= max_num_tokens:
            raise ValueError(f"no two-shot-only fused MoE tail at tp_size {tp_size}")
        return FusedTailRows(rows[0], max_num_tokens, min(rows[1], max_num_tokens), 0)
    # 32 ranks: the one-shot collective only (no two-shot kernels), so the capacity is its row limit.
    capacity = min(max_num_tokens, _ONESHOT_ONLY_CAPACITY.get(tp_size, max_num_tokens))
    return FusedTailRows(
        1,
        capacity,
        min(fused_tail_dispatch_max_tokens(tp_size), capacity),
        min(_ONESHOT_MAX_NUM_TOKENS.get(tp_size, 16), capacity),
    )


def fused_tail_mnnvl_norm(tp_size: int, twoshot_only: bool = False) -> bool:
    """Whether an instance reproduces the MNNVL all-reduce RMSNorm fusion's numerics by default."""
    return tp_size in MNNVL_NORM_TP_SIZES or twoshot_only


class KimiK3LatentMoETail:
    """Process-wide fused latent-MoE tail; one instance per TP group and shape."""

    _instances: ClassVar[Dict[LatentMoETailKey, "KimiK3LatentMoETail"]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get(
        cls,
        mapping: Mapping,
        *,
        hidden_size: int,
        latent_size: int,
        rms_eps: float,
        max_num_tokens: Optional[int] = None,
        mnnvl_norm: Optional[bool] = None,
        twoshot_only: bool = False,
    ) -> "KimiK3LatentMoETail":
        """Collective on the TP group the first time a key is seen.

        ``mnnvl_norm`` (default: on for ``MNNVL_NORM_TP_SIZES`` and ``twoshot_only``) makes the routed
        sum and RMSNorm reproduce the MNNVL all-reduce RMSNorm fusion's numerics bit for bit.
        ``twoshot_only``: the instance above a one-shot-only tail's rows
        (``fused_tail_twoshot_only_rows``), the two-shot collective alone."""
        if mnnvl_norm is None:
            mnnvl_norm = fused_tail_mnnvl_norm(mapping.tp_size, twoshot_only)
        key = LatentMoETailKey(
            mapping=mapping,
            hidden_size=hidden_size,
            latent_size=latent_size,
            max_num_tokens=max_num_tokens or fused_tail_max_tokens(),
            rms_eps=float(rms_eps),
            mnnvl_norm=bool(mnnvl_norm),
            twoshot_only=bool(twoshot_only),
        )
        with cls._lock:
            op = cls._instances.get(key)
            if op is None:
                op = cls(key)
                cls._instances[key] = op
        return op

    def __init__(self, key: LatentMoETailKey) -> None:
        # Deferred: the CuTe DSL modules pull in the cutlass DSL, which only
        # SM10x deployments of this op need.
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
            _mnnvl_workspace_all_succeeded,
        )

        from .moe_tail_collective import CollectiveKernel
        from .moe_tail_lamport_copy import LamportCopyKernel
        from .moe_tail_mnnvl_norm import mnnvl_norm_plan
        from .moe_tail_multicast_gemm import AdaptiveUpProjectionKernel
        from .moe_tail_primitives import NUM_LAMPORT_BUFFERS
        from .moe_tail_twoshot_collective import TwoShotCollectiveKernel

        mapping = key.mapping
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Kimi K3 fused MoE tail must be created outside CUDA graph capture")
        self.key = key
        self.tp_size = tp = mapping.tp_size
        self.rank = rank = mapping.tp_rank
        self.hidden_size = hidden = key.hidden_size
        self.latent_size = latent = key.latent_size
        self.shard_size = hidden // tp
        # The two-shot-only instance (above a one-shot-only tail's rows) has no one-shot regions or kernels.
        rows = fused_tail_rows(tp, key.max_num_tokens, key.twoshot_only)
        self.min_num_tokens = rows.min_tokens
        self.max_num_tokens = max_m = rows.capacity
        self.dispatch_max_tokens = rows.dispatch_max
        self._oneshot_max = oneshot_max = rows.oneshot_max
        oneshot = not key.twoshot_only
        device = torch.device("cuda", _mnnvl_device_index(mapping))

        # One-shot regions are sized for at least one token so the kernel compiles.
        oneshot_rows = max(oneshot_max, 1)
        routed_buffer_bytes = shared_buffer_bytes = 0
        if oneshot:
            routed_buffer_bytes = CollectiveKernel.routed_bytes_per_buffer(
                max_m=oneshot_rows, tp_size=tp, latent_dim=latent
            )
            shared_buffer_bytes = CollectiveKernel.shared_bytes_per_buffer(
                max_m=oneshot_rows, tp_size=tp, hidden_dim=hidden
            )
        mailbox_bytes = max_m * hidden * 2
        routed_off = 0
        shared_off = _align(routed_off + NUM_LAMPORT_BUFFERS * routed_buffer_bytes)
        mailbox_off = _align(shared_off + NUM_LAMPORT_BUFFERS * shared_buffer_bytes)
        twoshot = TwoShotCollectiveKernel.region_bytes(
            max_m=max_m, tp_size=tp, latent_dim=latent, hidden_dim=hidden
        )
        rs_off = _align(mailbox_off + mailbox_bytes)
        ss_off = _align(rs_off + twoshot["rs"])
        lat_off = _align(ss_off + twoshot["ss"])
        total_bytes = _align(lat_off + twoshot["lat"])

        self._comm = _get_mnnvl_workspace_comm(mapping)
        error: Optional[Exception] = None
        try:
            self._buffer = _make_mnnvl_mcast_buffer(
                self._comm, total_bytes, mapping, mapping.is_multi_node()
            )
        except Exception as exc:  # converged below so no rank is left waiting
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "Kimi K3 fused MoE tail workspace allocation failed on at least one rank"
            ) from error

        buf = self._buffer
        # Unicast views (this rank) and addresses (peers / multicast).
        routed_mc_ptr = None
        shared_peer_ptrs = []
        if oneshot:
            self._routed_workspace = buf.get_uc_buffer(
                rank,
                [NUM_LAMPORT_BUFFERS * routed_buffer_bytes // 4],
                torch.float32,
                routed_off // 4,
            )
            routed_mc_ptr = buf.get_mc_buffer([1], torch.float32, routed_off // 4).data_ptr()
            shared_shape = [NUM_LAMPORT_BUFFERS, oneshot_rows, tp, self.shard_size]
            self._shared_workspace = buf.get_uc_buffer(
                rank, shared_shape, torch.bfloat16, shared_off // 2
            )
            shared_peer_ptrs = [
                buf.get_uc_buffer(peer, shared_shape, torch.bfloat16, shared_off // 2).data_ptr()
                for peer in range(tp)
            ]
        self._mailbox = buf.get_uc_buffer(
            rank, [1, max_m, hidden], torch.bfloat16, mailbox_off // 2
        )
        mailbox_mc = buf.get_mc_buffer([1], torch.bfloat16, mailbox_off // 2)
        # Two-shot regions (sentinel-filled as one span: rs, ss and lat are adjacent).
        self._twoshot_region = buf.get_uc_buffer(
            rank, [(total_bytes - rs_off) // 4], torch.int32, rs_off // 4
        )
        rs_peer_ptrs = [
            buf.get_uc_buffer(peer, [1], torch.uint8, rs_off).data_ptr() for peer in range(tp)
        ]
        ss_peer_ptrs = [
            buf.get_uc_buffer(peer, [1], torch.uint8, ss_off).data_ptr() for peer in range(tp)
        ]
        lat_uc = buf.get_uc_buffer(rank, [1], torch.uint8, lat_off).data_ptr()
        lat_mc = buf.get_mc_buffer([1], torch.uint8, lat_off).data_ptr()
        self._twoshot_rs_peers = torch.tensor(rs_peer_ptrs, dtype=torch.int64, device=device)
        self._twoshot_ss_peers = torch.tensor(ss_peer_ptrs, dtype=torch.int64, device=device)
        self._twoshot_ctrl = torch.zeros(3, dtype=torch.uint32, device=device)
        if (
            (oneshot and (routed_mc_ptr == 0 or 0 in shared_peer_ptrs))
            or mailbox_mc.data_ptr() == 0
            or 0 in rs_peer_ptrs
            or 0 in ss_peer_ptrs
            or lat_mc == 0
        ):
            raise RuntimeError("Kimi K3 fused MoE tail: multicast/peer mapping unavailable")
        if oneshot:
            self._shared_peer_ptrs = torch.tensor(
                shared_peer_ptrs, dtype=torch.int64, device=device
            )
            self._routed_flags = torch.zeros(9, dtype=torch.uint32, device=device)
            self._shared_flags = torch.zeros(12, dtype=torch.int32, device=device)

        with torch.inference_mode():
            if oneshot:
                CollectiveKernel.init_routed_protocol(
                    self._routed_workspace, self._routed_flags, routed_buffer_bytes
                )
                CollectiveKernel.init_shared_protocol(
                    self._shared_workspace, self._shared_flags, shared_buffer_bytes
                )
            AdaptiveUpProjectionKernel.init_mailbox(self._mailbox)
            self._twoshot_region.fill_(-0x80000000)
        torch.cuda.synchronize(device)
        # Fence: no rank may publish into a peer that has not cleared its buffers.
        if not _mnnvl_workspace_all_succeeded(self._comm, True):
            raise RuntimeError("Kimi K3 fused MoE tail protocol reset failed")

        oneshot_norm_plan = twoshot_norm_plan = None
        if key.mnnvl_norm:
            if oneshot:
                oneshot_norm_plan = mnnvl_norm_plan(
                    tp_size=tp, dim=latent, max_rows=oneshot_rows, device=device
                )
            twoshot_norm_plan = mnnvl_norm_plan(
                tp_size=tp, dim=latent, max_rows=max_m, device=device
            )
        # The two-shot-only instance's collectives know its fewest rows (above 16 ranks it fixes their sum orders).
        twoshot_rows = {"min_m": self.min_num_tokens} if key.twoshot_only else {}
        error = None
        try:
            self._collective = (
                CollectiveKernel(
                    rank=rank,
                    tp_size=tp,
                    latent_dim=latent,
                    hidden_dim=hidden,
                    max_m=oneshot_rows,
                    rms_eps=key.rms_eps,
                    routed_workspace=self._routed_workspace,
                    routed_flags=self._routed_flags,
                    routed_multicast_ptr=routed_mc_ptr,
                    shared_workspace=self._shared_workspace,
                    shared_flags=self._shared_flags,
                    shared_peer_ptrs=self._shared_peer_ptrs,
                    output_rows=max_m,
                    norm_plan=oneshot_norm_plan,
                )
                if oneshot
                else None
            )
            self._twoshot_collective = (
                TwoShotCollectiveKernel(
                    rank=rank,
                    tp_size=tp,
                    latent_dim=latent,
                    hidden_dim=hidden,
                    max_m=max_m,
                    rms_eps=key.rms_eps,
                    rs_peers=self._twoshot_rs_peers,
                    ss_peers=self._twoshot_ss_peers,
                    lat_uc=lat_uc,
                    lat_mc=lat_mc,
                    ctrl=self._twoshot_ctrl,
                    norm_plan=twoshot_norm_plan,
                    **twoshot_rows,
                )
                if max_m > oneshot_max
                else None
            )
            # Finalize-on-load variants over the same buffers and flags (the
            # protocol does not depend on how the routed row is produced).
            self._collective_topk = (
                CollectiveKernel(
                    rank=rank,
                    tp_size=tp,
                    latent_dim=latent,
                    hidden_dim=hidden,
                    max_m=oneshot_rows,
                    rms_eps=key.rms_eps,
                    routed_workspace=self._routed_workspace,
                    routed_flags=self._routed_flags,
                    routed_multicast_ptr=routed_mc_ptr,
                    shared_workspace=self._shared_workspace,
                    shared_flags=self._shared_flags,
                    shared_peer_ptrs=self._shared_peer_ptrs,
                    top_k=_FINALIZE_TOP_K,
                    output_rows=max_m,
                    norm_plan=oneshot_norm_plan,
                )
                if oneshot
                else None
            )
            self._twoshot_collective_topk = (
                TwoShotCollectiveKernel(
                    rank=rank,
                    tp_size=tp,
                    latent_dim=latent,
                    hidden_dim=hidden,
                    max_m=max_m,
                    rms_eps=key.rms_eps,
                    rs_peers=self._twoshot_rs_peers,
                    ss_peers=self._twoshot_ss_peers,
                    lat_uc=lat_uc,
                    lat_mc=lat_mc,
                    ctrl=self._twoshot_ctrl,
                    top_k=_FINALIZE_TOP_K,
                    norm_plan=twoshot_norm_plan,
                    **twoshot_rows,
                )
                if max_m > oneshot_max
                else None
            )
            self._up_projection = AdaptiveUpProjectionKernel(
                rank=rank,
                tp_size=tp,
                latent_dim=latent,
                hidden_dim=hidden,
                max_m=max_m,
                # The two-shot-only instance's rows are all above the SIMT range.
                skinny_max_m=0 if key.twoshot_only else min(_SKINNY_MAX_NUM_TOKENS, max_m),
                mailbox=self._mailbox,
                mailbox_multicast_ptr=mailbox_mc.data_ptr(),
            )
            for m in range(1, self._up_projection.skinny_max_m + 1):
                self._up_projection.compile_skinny(m)
            # Above the SIMT range: the split-K tcgen05 GEMM.
            skinny_max = self._up_projection.skinny_max_m
            splitk_counts = list(range(max(skinny_max + 1, self.min_num_tokens), max_m + 1))
            self._splitk_buckets = (
                _TWOSHOT_ONLY_SPLITK_BUCKETS if key.twoshot_only else _SPLITK_TOKEN_BUCKETS
            )
            self._splitk = None
            self._splitk_by_bucket: Dict[int, object] = {}
            if splitk_counts:
                from .moe_tail_splitk_multicast_gemm import SplitKMulticastUpProjection

                self._splitk = SplitKMulticastUpProjection(
                    latent_dim=latent, hidden_dim=hidden, shard_dim=self.shard_size, max_m=max_m
                )
                for bucket in sorted(
                    {_splitk_bucket(m, self._splitk_buckets) for m in splitk_counts}
                ):
                    self._splitk_by_bucket[bucket] = self._splitk.compile_for(
                        min(bucket, max_m), device, whole_k=key.twoshot_only
                    )
                self._mailbox_shard_mc = mailbox_mc.data_ptr() + rank * self.shard_size * 2
            self._lamport_copy = LamportCopyKernel(
                hidden_dim=hidden,
                max_m=max_m,
                ctas=_LAMPORT_COPY_CTAS,
                threads=_LAMPORT_COPY_THREADS,
            )
        except Exception as exc:
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError("Kimi K3 fused MoE tail kernel compilation failed") from error
        logger.info_once(
            f"Kimi K3 fused latent-MoE tail ready: tp={tp}, tokens "
            f"{self.min_num_tokens}..{self.dispatch_max_tokens} "
            f"(capacity {max_m}, one-shot <= {oneshot_max}), "
            f"workspace {total_bytes / 2**20:.1f} MiB",
            key=f"kimi_k3_fused_moe_tail_ready_{key.twoshot_only}",
        )

    def pushes_shared_early(self, num_tokens: int) -> bool:
        """Whether a call of ``num_tokens`` rows takes the two-shot collective, whose shared reduce-scatter
        :meth:`push_shared` can store ahead of the call (the one-shot collective's shared push is a small part of a
        latency-bound call: GB300 W = 16 / 32 measured no gain from moving it)."""
        return (
            self._twoshot_collective is not None
            and self._oneshot_max < num_tokens
            and self.min_num_tokens <= num_tokens <= self.max_num_tokens
        )

    def push_shared(self, shared_partial: torch.Tensor) -> None:
        """Store this rank's ``[M, hidden]`` shared-expert partial rows into the two-shot collective's slots of the
        next call (``pushes_shared_early(M)``), on the stream that produced them while the routed experts run (the
        shared expert's); the next call must pass ``shared_pushed=True``. The same stores and sums as the call's own
        push, so the same bits."""
        self._twoshot_collective.push_shared(shared_partial.contiguous())

    def __call__(
        self,
        routed_partial: torch.Tensor | UnfinalizedRoutedOutput,
        shared_partial: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
        shared_pushed: bool = False,
    ) -> torch.Tensor:
        """``routed_partial`` ``[M, latent]`` and ``shared_partial`` ``[M, hidden]``
        are this rank's unreduced BF16 partial sums; ``up_weight`` is the full
        replicated ``[hidden, latent]`` weight. Returns the reduced
        ``[M, hidden]`` MoE output (routed + shared) on every rank.

        ``routed_partial`` may instead be the routed experts'
        ``UnfinalizedRoutedOutput`` (``expanded_idx_to_permuted_idx`` and
        ``expert_weights`` ``[M, top_k]``). ``shared_pushed``: :meth:`push_shared` already stored this call's
        shared rows."""
        return self._forward(
            routed_partial,
            shared_partial,
            rms_weight,
            up_weight,
            stages=3,
            shared_pushed=shared_pushed,
        )

    def forward_to_mailbox(
        self,
        routed_partial: torch.Tensor | UnfinalizedRoutedOutput,
        shared_partial: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
        shared_pushed: bool = False,
    ) -> torch.Tensor:
        """``__call__`` without the copy kernel: returns the mailbox (``[1, capacity, hidden]``)
        whose first ``M`` rows receive the reduced MoE output on every rank. The caller must
        consume those rows exactly once before the next call on this op: :meth:`copy_out`, or a
        kernel that polls them and re-arms the mailbox as the copy kernel does
        (``trtllm::k3_attn_res_mailbox_add_rmsnorm``)."""
        return self._forward(
            routed_partial,
            shared_partial,
            rms_weight,
            up_weight,
            stages=2,
            shared_pushed=shared_pushed,
        )

    def copy_out(self, mailbox: torch.Tensor, m: int) -> torch.Tensor:
        """The ``[m, hidden]`` output a :meth:`forward_to_mailbox` call left in ``mailbox`` (the
        tail's copy kernel, which re-arms the mailbox)."""
        return self._lamport_copy(mailbox, m=m).view(m, self.hidden_size)

    def _forward(
        self,
        routed_partial,
        shared_partial,
        rms_weight,
        up_weight,
        *,
        stages: int,
        shared_pushed: bool = False,
    ):
        """Run the first ``stages`` kernels (2: collective and up-projection, 3: + Lamport
        copy)."""
        unfinalized = isinstance(routed_partial, UnfinalizedRoutedOutput)
        if unfinalized:
            m = routed_partial.expanded_idx_to_permuted_idx.shape[0]
        else:
            m = routed_partial.shape[0]
            routed_partial = routed_partial.contiguous()
        if not self.min_num_tokens <= m <= self.max_num_tokens:
            raise ValueError(
                f"fused MoE tail supports {self.min_num_tokens}..{self.max_num_tokens} tokens, got {m}"
            )
        if m <= self._oneshot_max:
            if shared_pushed:
                raise ValueError("the one-shot collective pushes its shared rows itself")
            collective = self._collective_topk if unfinalized else self._collective
            latent, shared_shard = collective(
                routed_partial, shared_partial.contiguous(), rms_weight
            )
        else:
            collective = self._twoshot_collective_topk if unfinalized else self._twoshot_collective
            latent, shared_shard = collective(
                routed_partial, shared_partial.contiguous(), rms_weight, shared_pushed=shared_pushed
            )
        local_up_weight = up_weight.narrow(0, self.rank * self.shard_size, self.shard_size)
        if m > self._up_projection.skinny_max_m:
            bucket = _splitk_bucket(m, self._splitk_buckets)
            self._splitk(
                self._splitk_by_bucket[bucket],
                local_up_weight,
                latent,
                shared_shard,
                self._mailbox_shard_mc,
            )
            mailbox = self._mailbox
        else:
            mailbox = self._up_projection(latent, local_up_weight, shared_shard)
        if stages == 2:
            return mailbox
        return self._lamport_copy(mailbox, m=m).view(m, self.hidden_size)

    def can_run(self, num_tokens: int) -> bool:
        """Whether the model should take the fused tail for this token count."""
        return self.min_num_tokens <= num_tokens <= self.dispatch_max_tokens


__all__ = [
    "FusedTailRows",
    "KimiK3LatentMoETail",
    "MAX_NUM_TOKENS",
    "UnfinalizedRoutedOutput",
    "fused_tail_max_tokens",
    "fused_tail_mnnvl_norm",
    "fused_tail_rows",
    "fused_tail_twoshot_only_rows",
]
