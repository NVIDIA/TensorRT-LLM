# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused Kimi K3 latent-MoE tail for tensor-parallel decode.

Replaces, for ``M <= max_m`` tokens,

    latent = RMSNorm(AllReduce(routed_partial))          # [M, latent]
    out = latent @ up_weight.T + AllReduce(shared_partial)  # [M, hidden]

with three PDL-chained kernels:

1. ``CollectiveKernel``: one-shot multicast AllReduce of the routed latent
   partials with the RMSNorm fused in, plus a one-shot ReduceScatter of the
   shared-expert partials (rank ``r`` keeps hidden columns
   ``[r * S, (r + 1) * S)``, ``S = hidden / tp``).
2. ``AdaptiveUpProjectionKernel``: the rank's ``S`` rows of the up-projection
   (``S x latent`` weight bytes instead of ``hidden x latent``) with the shared
   shard added in the epilogue and the result stored through the multicast
   mapping of a symmetric mailbox, i.e. the all-gather is the GEMM store.
3. ``LamportCopyKernel``: waits for every shard of the local mailbox copy,
   copies it out and re-arms the mailbox.

The symmetric memory is one ``McastGPUBuffer`` (the MNNVL all-reduce
allocator), so the op works across NVLink-connected trays.

Unless ``KIMI_K3_FUSED_MOE_TAIL_FINALIZE=0``, the collective also takes the routed
experts' unfinalized output (``UnfinalizedRoutedOutput``: the FC2 rows in
permuted order, the routing weights and the (token, slot) -> row map) and does
the top-k weighted sum while loading this rank's routed row, so the routed
experts skip their finalize kernel; the sum has the finalize's numerics (FP32
in slot order, one BF16 rounding).

Numerics: every rank reduces in rank order in FP32 and rounds once to BF16,
like the MNNVL one-shot all-reduce; the RMSNorm squares the BF16-rounded sum
(as the MNNVL RMSNorm fusion does); the up-projection is rounded to BF16
before the BF16 shared shard is added in FP32. Only the up-projection's
accumulation order differs from an unsharded GEMM.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .routed_output import UnfinalizedRoutedOutput

# Workspace capacity in tokens: the largest generation batch of the Kimi K3.1
# TEP gens (192 tokens per step).
MAX_NUM_TOKENS = 192
# Largest token count the model dispatches to the fused tail, per TP size: the
# measured crossover with the unfused tail (TP16: every step of the 192-token
# gen batches).
_DISPATCH_MAX_NUM_TOKENS = {8: 128, 16: 192, 32: 32}
_SKINNY_MAX_NUM_TOKENS = 1
# Token tiles of the split-K up-projection (one compiled tactic per bucket).
_SPLITK_TOKEN_BUCKETS = (8, 16, 32, 64, 128)
# Largest token count of the (opt-in) persistent tcgen05 up-projection.
_PERSISTENT_MAX_NUM_TOKENS = 128
_MMA_TILER_MN = (64, 32)
_GEMM_CLUSTER_MN = (1, 8)
_B_PRIME_STAGES = 2
_COLLECTIVE_TOKEN_CTAS = 32
# Token counts up to here use the one-shot collective (every rank reduces every
# routed row: one cross-rank trip, ingress ~ tp * M * latent); larger ones the
# two-shot collective (each row reduced on one owner rank, then multicast: two
# trips, ingress ~ 2 * M * latent).
_ONESHOT_MAX_NUM_TOKENS = {8: 32, 16: 16, 32: 32}
# TP sizes whose tail is one-shot only: the workspace capacity in tokens.
_ONESHOT_ONLY_CAPACITY = {32: 32}
# Enough CTAs for one 16-byte fragment per thread up to 128 tokens: the copy
# polls with volatile loads, so it is latency-bound on a few threads.
_LAMPORT_COPY_CTAS = 1024
_LAMPORT_COPY_THREADS = 128
SUPPORTED_TP_SIZES = (8, 16, 32)
SUPPORTED_SHAPE = (7168, 3584)  # (hidden, latent)
_REGION_ALIGN = 4096


def _splitk_bucket(m: int) -> int:
    """Split-K tactic bucket of a token count (the largest serves any M above)."""
    return next((b for b in _SPLITK_TOKEN_BUCKETS if b >= m), _SPLITK_TOKEN_BUCKETS[-1])


def _align(value: int, alignment: int = _REGION_ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class LatentMoETailKey:
    mapping: Mapping
    hidden_size: int
    latent_size: int
    max_num_tokens: int
    rms_eps: float


def fused_tail_max_tokens() -> int:
    """Workspace capacity in tokens."""
    return MAX_NUM_TOKENS


def fused_tail_finalize_requested() -> bool:
    """Whether the tail folds the routed experts' finalize (``KIMI_K3_FUSED_MOE_TAIL_FINALIZE``,
    default on: the folded sum is bit-identical; ``0`` keeps the routed experts' finalize kernel)."""
    return os.environ.get("KIMI_K3_FUSED_MOE_TAIL_FINALIZE", "1") == "1"


def fused_tail_dispatch_max_tokens(tp_size: int) -> int:
    """Largest token count the model sends to the fused tail."""
    return _DISPATCH_MAX_NUM_TOKENS.get(tp_size, 0)


def fused_tail_unsupported_reason(
    mapping: Mapping, hidden_size: int, latent_size: int, device: torch.device
) -> Optional[str]:
    """Why the fused tail cannot run for this layer, or ``None`` if it can."""
    if mapping.tp_size not in SUPPORTED_TP_SIZES:
        return f"tp_size {mapping.tp_size} not in {SUPPORTED_TP_SIZES}"
    if (hidden_size, latent_size) != SUPPORTED_SHAPE:
        return f"(hidden, latent) {(hidden_size, latent_size)} != {SUPPORTED_SHAPE}"
    if mapping.enable_attention_dp:
        return "attention DP"
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] != 10:
        return "requires an SM10x GPU"
    return None


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
    ) -> "KimiK3LatentMoETail":
        """Collective on the TP group the first time a key is seen."""
        key = LatentMoETailKey(
            mapping=mapping,
            hidden_size=hidden_size,
            latent_size=latent_size,
            max_num_tokens=max_num_tokens or fused_tail_max_tokens(),
            rms_eps=float(rms_eps),
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

        from .collective import CollectiveKernel
        from .lamport_copy import LamportCopyKernel
        from .multicast_gemm import AdaptiveUpProjectionKernel
        from .primitives import NUM_LAMPORT_BUFFERS
        from .twoshot_collective import TwoShotCollectiveKernel

        mapping = key.mapping
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Kimi K3 fused MoE tail must be created outside CUDA graph capture")
        self.key = key
        self.tp_size = tp = mapping.tp_size
        self.rank = rank = mapping.tp_rank
        self.hidden_size = hidden = key.hidden_size
        self.latent_size = latent = key.latent_size
        self.shard_size = hidden // tp
        # 32 ranks: the one-shot collective only (no two-shot kernels), so the capacity is its row limit.
        self.max_num_tokens = max_m = min(
            key.max_num_tokens, _ONESHOT_ONLY_CAPACITY.get(mapping.tp_size, key.max_num_tokens)
        )
        self.dispatch_max_tokens = min(fused_tail_dispatch_max_tokens(tp), max_m)
        device = torch.device("cuda", _mnnvl_device_index(mapping))

        self._oneshot_max = oneshot_max = min(
            int(
                os.environ.get(
                    "KIMI_K3_FUSED_MOE_TAIL_ONESHOT_MAX", _ONESHOT_MAX_NUM_TOKENS.get(tp, 16)
                )
            ),
            max_m,
        )
        # One-shot regions are sized for at least one token so the kernel compiles.
        oneshot_rows = max(oneshot_max, 1)
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
        use_fabric_handle = (
            os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or mapping.is_multi_node()
        )
        error: Optional[Exception] = None
        try:
            self._buffer = _make_mnnvl_mcast_buffer(
                self._comm, total_bytes, mapping, use_fabric_handle
            )
        except Exception as exc:  # converged below so no rank is left waiting
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "Kimi K3 fused MoE tail workspace allocation failed on at least one rank"
            ) from error

        buf = self._buffer
        # Unicast views (this rank) and addresses (peers / multicast).
        self._routed_workspace = buf.get_uc_buffer(
            rank, [NUM_LAMPORT_BUFFERS * routed_buffer_bytes // 4], torch.float32, routed_off // 4
        )
        routed_mc = buf.get_mc_buffer([1], torch.float32, routed_off // 4)
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
            routed_mc.data_ptr() == 0
            or mailbox_mc.data_ptr() == 0
            or 0 in shared_peer_ptrs
            or 0 in rs_peer_ptrs
            or 0 in ss_peer_ptrs
            or lat_mc == 0
        ):
            raise RuntimeError("Kimi K3 fused MoE tail: multicast/peer mapping unavailable")
        self._shared_peer_ptrs = torch.tensor(shared_peer_ptrs, dtype=torch.int64, device=device)
        self._routed_flags = torch.zeros(9, dtype=torch.uint32, device=device)
        self._shared_flags = torch.zeros(12, dtype=torch.int32, device=device)

        with torch.inference_mode():
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

        error = None
        try:
            self._collective = CollectiveKernel(
                rank=rank,
                tp_size=tp,
                latent_dim=latent,
                hidden_dim=hidden,
                max_m=oneshot_rows,
                max_token_ctas=min(
                    int(
                        os.environ.get("KIMI_K3_FUSED_MOE_TAIL_TOKEN_CTAS", _COLLECTIVE_TOKEN_CTAS)
                    ),
                    oneshot_rows,
                ),
                rms_eps=key.rms_eps,
                fp32_internal=False,
                routed_workspace=self._routed_workspace,
                routed_flags=self._routed_flags,
                routed_multicast_ptr=routed_mc.data_ptr(),
                shared_workspace=self._shared_workspace,
                shared_flags=self._shared_flags,
                shared_peer_ptrs=self._shared_peer_ptrs,
                output_rows=max_m,
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
                )
                if max_m > oneshot_max
                else None
            )
            # Finalize-on-load variants over the same buffers and flags (the
            # protocol does not depend on how the routed row is produced).
            self._finalize_top_k = 16 if fused_tail_finalize_requested() else 0
            self._collective_topk = None
            self._twoshot_collective_topk = None
            if self._finalize_top_k:
                self._collective_topk = CollectiveKernel(
                    rank=rank,
                    tp_size=tp,
                    latent_dim=latent,
                    hidden_dim=hidden,
                    max_m=oneshot_rows,
                    max_token_ctas=min(
                        int(
                            os.environ.get(
                                "KIMI_K3_FUSED_MOE_TAIL_TOKEN_CTAS", _COLLECTIVE_TOKEN_CTAS
                            )
                        ),
                        oneshot_rows,
                    ),
                    rms_eps=key.rms_eps,
                    fp32_internal=False,
                    routed_workspace=self._routed_workspace,
                    routed_flags=self._routed_flags,
                    routed_multicast_ptr=routed_mc.data_ptr(),
                    shared_workspace=self._shared_workspace,
                    shared_flags=self._shared_flags,
                    shared_peer_ptrs=self._shared_peer_ptrs,
                    top_k=self._finalize_top_k,
                    output_rows=max_m,
                )
                if max_m > oneshot_max:
                    self._twoshot_collective_topk = TwoShotCollectiveKernel(
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
                        top_k=self._finalize_top_k,
                    )
            self._up_projection = AdaptiveUpProjectionKernel(
                rank=rank,
                tp_size=tp,
                latent_dim=latent,
                hidden_dim=hidden,
                max_m=max_m,
                skinny_max_m=min(
                    int(
                        os.environ.get("KIMI_K3_FUSED_MOE_TAIL_SKINNY_MAX", _SKINNY_MAX_NUM_TOKENS)
                    ),
                    max_m,
                ),
                mma_tiler_mn=_MMA_TILER_MN,
                cluster_shape_mn=_GEMM_CLUSTER_MN,
                b_prime_stages=_B_PRIME_STAGES,
                mailbox=self._mailbox,
                mailbox_multicast_ptr=mailbox_mc.data_ptr(),
            )
            for m in range(1, self._up_projection.skinny_max_m + 1):
                self._up_projection.compile_skinny(m)
            # Above the SIMT range: the split-K tcgen05 GEMM;
            # KIMI_K3_FUSED_MOE_TAIL_GEMM=persistent selects the persistent
            # tcgen05 GEMM instead (one tile per CTA: <= 128 tokens).
            self._gemm_kind = os.environ.get("KIMI_K3_FUSED_MOE_TAIL_GEMM", "splitk")
            skinny_max = self._up_projection.skinny_max_m
            self._persistent_range = (
                (skinny_max + 1, min(max_m, _PERSISTENT_MAX_NUM_TOKENS))
                if self._gemm_kind == "persistent"
                else (1, 0)
            )
            lo, hi = self._persistent_range
            if lo <= hi:
                self._up_projection.compile_dynamic()
            splitk_counts = [m for m in range(skinny_max + 1, max_m + 1) if not lo <= m <= hi]
            self._splitk = None
            self._splitk_by_bucket: Dict[int, object] = {}
            if splitk_counts:
                from .splitk_multicast_gemm import SplitKMulticastUpProjection

                self._splitk = SplitKMulticastUpProjection(
                    latent_dim=latent, hidden_dim=hidden, shard_dim=self.shard_size, max_m=max_m
                )
                for bucket in sorted({_splitk_bucket(m) for m in splitk_counts}):
                    self._splitk_by_bucket[bucket] = self._splitk.compile_for(
                        min(bucket, max_m), device
                    )
                self._mailbox_shard_mc = mailbox_mc.data_ptr() + rank * self.shard_size * 2
            self._lamport_copy = LamportCopyKernel(
                hidden_dim=hidden,
                max_m=max_m,
                ctas=int(os.environ.get("KIMI_K3_FUSED_MOE_TAIL_COPY_CTAS", _LAMPORT_COPY_CTAS)),
                threads=_LAMPORT_COPY_THREADS,
            )
        except Exception as exc:
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError("Kimi K3 fused MoE tail kernel compilation failed") from error
        logger.info_once(
            f"Kimi K3 fused latent-MoE tail ready: tp={tp}, tokens <= {self.dispatch_max_tokens} "
            f"(capacity {max_m}, one-shot <= {oneshot_max}), "
            f"workspace {total_bytes / 2**20:.1f} MiB",
            key="kimi_k3_fused_moe_tail_ready",
        )

    @property
    def folds_finalize(self) -> bool:
        """Whether ``__call__`` accepts the routed experts' unfinalized output."""
        return self._finalize_top_k > 0

    def __call__(
        self,
        routed_partial: torch.Tensor | UnfinalizedRoutedOutput,
        shared_partial: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> torch.Tensor:
        """``routed_partial`` ``[M, latent]`` and ``shared_partial`` ``[M, hidden]``
        are this rank's unreduced BF16 partial sums; ``up_weight`` is the full
        replicated ``[hidden, latent]`` weight. Returns the reduced
        ``[M, hidden]`` MoE output (routed + shared) on every rank.

        With ``folds_finalize``, ``routed_partial`` may instead be the routed
        experts' ``UnfinalizedRoutedOutput`` (``expanded_idx_to_permuted_idx``
        and ``expert_weights`` ``[M, top_k]``)."""
        return self._forward(routed_partial, shared_partial, rms_weight, up_weight, stages=3)

    def forward_to_mailbox(
        self,
        routed_partial: torch.Tensor | UnfinalizedRoutedOutput,
        shared_partial: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> torch.Tensor:
        """``__call__`` without the copy kernel: returns the mailbox (``[1, capacity, hidden]``)
        whose first ``M`` rows receive the reduced MoE output on every rank. The caller must
        consume those rows exactly once before the next call on this op: :meth:`copy_out`, or a
        kernel that polls them and re-arms the mailbox as the copy kernel does
        (``trtllm::k3_attn_res_mailbox_add_rmsnorm``)."""
        return self._forward(routed_partial, shared_partial, rms_weight, up_weight, stages=2)

    def copy_out(self, mailbox: torch.Tensor, m: int) -> torch.Tensor:
        """The ``[m, hidden]`` output a :meth:`forward_to_mailbox` call left in ``mailbox`` (the
        tail's copy kernel, which re-arms the mailbox)."""
        return self._lamport_copy(mailbox, m=m).view(m, self.hidden_size)

    def _forward(self, routed_partial, shared_partial, rms_weight, up_weight, *, stages: int):
        """Run the first ``stages`` kernels (1: collective, 2: + up-projection,
        3: + Lamport copy); fewer than 3 is for kernel timing only."""
        unfinalized = isinstance(routed_partial, UnfinalizedRoutedOutput)
        if unfinalized:
            if not self.folds_finalize:
                raise ValueError("fused MoE tail was built without the finalize fold")
            m = routed_partial.expanded_idx_to_permuted_idx.shape[0]
        else:
            m = routed_partial.shape[0]
            routed_partial = routed_partial.contiguous()
        if not 1 <= m <= self.max_num_tokens:
            raise ValueError(f"fused MoE tail supports 1..{self.max_num_tokens} tokens, got {m}")
        if m <= self._oneshot_max:
            collective = self._collective_topk if unfinalized else self._collective
        else:
            collective = self._twoshot_collective_topk if unfinalized else self._twoshot_collective
        latent, shared_shard = collective(routed_partial, shared_partial.contiguous(), rms_weight)
        if stages == 1:
            return latent
        local_up_weight = up_weight.narrow(0, self.rank * self.shard_size, self.shard_size)
        lo, hi = self._persistent_range
        if m > self._up_projection.skinny_max_m and not lo <= m <= hi:
            bucket = _splitk_bucket(m)
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
        return 1 <= num_tokens <= self.dispatch_max_tokens


def math_reference(
    routed_partials: torch.Tensor,
    shared_partials: torch.Tensor,
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Unfused FP32 reference: ``[tp, M, latent]`` / ``[tp, M, hidden]`` partials -> ``[M, hidden]``."""
    routed = routed_partials.float().sum(0)
    var = routed.pow(2).mean(-1, keepdim=True)
    latent = (routed * torch.rsqrt(var + eps) * rms_weight.float()).to(torch.bfloat16)
    up = latent.float() @ up_weight.float().t()
    out = up + shared_partials.float().sum(0)
    return out


__all__ = [
    "KimiK3LatentMoETail",
    "MAX_NUM_TOKENS",
    "UnfinalizedRoutedOutput",
    "fused_tail_finalize_requested",
    "fused_tail_max_tokens",
    "fused_tail_unsupported_reason",
    "math_reference",
]
