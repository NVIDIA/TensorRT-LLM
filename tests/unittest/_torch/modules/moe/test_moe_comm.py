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
"""
Unified Communication Strategy Tests for MoE

Tests all Communication subclasses (AllGatherReduceScatter, DeepEP,
DeepEPLowLatency, NVLinkOneSided, NVLinkTwoSided, NVLinkTwoSidedFlashinfer) via the public
dispatch() and combine() interfaces defined in Communication base class.

Each test runs the full pipeline: dispatch -> verify dispatch -> simple_moe
-> combine -> verify combine.  Dispatch errors are caught early with clear
diagnostics before combine verification runs.

Dispatch verification:
  - AllGatherRS: verifies allgathered data is bitwise exact concatenation
  - AllToAll comms: encodes (rank_id, token_idx) in hidden_states bytes,
    then checks routing correctness, content integrity, and completeness

Combine verification:
  Uses simple_moe (weighted sum of hidden_states, no expert-specific
  computation).  Reference matches kernel behavior per comm type:
  - NVLinkOneSided lpc: fp8 quant/dequant simulation (moeA2ACombineKernel)
  - NVLinkTwoSided lpc: NVFP4 quant/dequant simulation
  - DeepEPLL: weighted reduction with real topk_weights
  - Default: float32 accumulation matching kernel registers
  This isolates purely the combine communication error.

Singleton safety:
  NVLinkOneSided._WORKSPACE is reset before each creation to avoid
  assertion failures from varying params across MPI process reuse.
  All tests use num_experts=32 to avoid DeepEP buffer_pool num_experts
  assertion failures.

Run with: mpirun -np 8 pytest test_moe_comm.py -x -v
"""

import atexit
import os
import pickle
import sys
import traceback
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm as tllm
from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._torch.modules.fused_moe.communication.allgather_reducescatter import (
    AllGatherReduceScatter,
)
from tensorrt_llm._torch.modules.fused_moe.communication.deep_ep import DeepEP
from tensorrt_llm._torch.modules.fused_moe.communication.deep_ep_low_latency import DeepEPLowLatency
from tensorrt_llm._torch.modules.fused_moe.communication.nccl_ep import NcclEP
from tensorrt_llm._torch.modules.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.modules.fused_moe.communication.nvlink_two_sided import NVLinkTwoSided
from tensorrt_llm._torch.modules.fused_moe.communication.nvlink_two_sided_flashinfer import (
    NVLinkTwoSidedFlashinfer,
)
from tensorrt_llm._torch.modules.fused_moe.deep_ep_utils import deep_ep_installed
from tensorrt_llm._torch.modules.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.modules.fused_moe.nccl_ep_utils import is_nccl_ep_installed
from tensorrt_llm.deep_ep.buffer import Buffer
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# ============================================================================
# Constants
# ============================================================================

COMM_ALLGATHER_RS = "AllGatherReduceScatter"
COMM_DEEP_EP = "DeepEP"
COMM_DEEP_EP_LL = "DeepEPLowLatency"
COMM_NVLINK_ONE_SIDED = "NVLinkOneSided"
COMM_NVLINK_TWO_SIDED = "NVLinkTwoSided"
COMM_NVLINK_TWO_SIDED_FLASHINFER = "NVLinkTwoSidedFlashinfer"
COMM_NCCL_EP = "NcclEP"

ALL_COMM_TYPES = [
    COMM_ALLGATHER_RS,
    COMM_DEEP_EP,
    COMM_DEEP_EP_LL,
    COMM_NVLINK_ONE_SIDED,
    COMM_NVLINK_TWO_SIDED,
    COMM_NVLINK_TWO_SIDED_FLASHINFER,
    COMM_NCCL_EP,
]

# Must be in DeepEPLowLatency.SUPPORTED_HIDDEN_SIZES
DEFAULT_HIDDEN_SIZE = 4096

# Fixed across all tests to avoid DeepEP buffer_pool singleton conflicts
# (VariableLengthLowLatencyBuffer.reserve asserts num_experts is consistent).
FIXED_NUM_EXPERTS = 32

# Force consistent NVLinkOneSided workspace size across varying top_k
# to avoid _WORKSPACE singleton assertion failures.
NVLINK_WORKSPACE_MB = "512"

# ============================================================================
# Test Configuration
# ============================================================================


@dataclass
class CommTestConfig:
    """Configuration for a single comm test case."""

    comm_type: str
    ep_size: int
    num_experts: int
    top_k: int
    hidden_size: int
    all_num_tokens: List[int]
    quant_mode: str = "none"  # "none" | "fp8" | "nvfp4" | "w4afp8"
    use_low_precision_combine: bool = False

    def __str__(self) -> str:
        tokens_str = "x".join(str(t) for t in self.all_num_tokens)
        s = (
            f"{self.comm_type}_ep{self.ep_size}_e{self.num_experts}"
            f"_k{self.top_k}_h{self.hidden_size}_t{tokens_str}"
        )
        if self.quant_mode != "none":
            s += f"_q{self.quant_mode}"
        if self.use_low_precision_combine:
            s += "_lpcombine"
        return s


@dataclass
class WorkerInputs:
    """Per-rank inputs prepared before dispatch."""

    hs: torch.Tensor
    hidden_states_sf: Optional[torch.Tensor]
    global_scale: Optional[torch.Tensor]
    slots: torch.Tensor
    scales: torch.Tensor
    dispatch_kwargs: Dict[str, torch.Tensor]
    original_fp8: Optional[torch.Tensor] = None
    w4afp8_roundtrip_ok: Optional[bool] = None


@dataclass
class DispatchOutputs:
    """Per-rank tensors returned by dispatch()."""

    recv_hs: torch.Tensor
    recv_sf: Optional[torch.Tensor]
    recv_slots: torch.Tensor
    recv_scales: Optional[torch.Tensor]


@dataclass
class CommTestGroup:
    """A group of configs sharing one MPI pool size and communication type."""

    configs: List[CommTestConfig]

    def __str__(self) -> str:
        first = self.configs[0]
        return f"{first.comm_type}_ep{first.ep_size}_n{len(self.configs)}"


@dataclass
class PendingWorkerResults:
    """Worker futures for one config submitted to the MPI pool."""

    config: CommTestConfig
    futures: List


# ============================================================================
# MPI Serialization Helpers
# ============================================================================

_FLOAT8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _safe_cpu(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Move tensor to CPU, converting float8 to uint8 for MPI serialization.

    PyTorch float8 tensors cannot be pickled reliably across all builds.
    Since float8 and uint8 have the same element size, view(uint8) preserves
    shape and content.  Downstream verification already works on the byte
    view, so this is a transparent change.
    """
    if t is None:
        return None
    if t.dtype in _FLOAT8_DTYPES:
        return t.view(torch.uint8).cpu()
    return t.cpu()


def _ep_mask_words(ep_size: int, dead_ranks: Set[int]) -> torch.Tensor:
    """Build the CPU active-rank mask tensor expected by moe_a2a ops."""
    health = EPGroupHealth(ep_size)
    for rank in dead_ranks:
        health.mark_failed(rank)
    return torch.tensor(health.get_mask_words(), dtype=torch.uint64, device="cpu")


def _make_rank_mask_payload(local_num_tokens: int, hidden_size: int, rank: int) -> torch.Tensor:
    """Make deterministic per-rank payloads for exact equality assertions."""
    base = torch.arange(local_num_tokens * hidden_size, dtype=torch.bfloat16, device="cuda").view(
        local_num_tokens, hidden_size
    )
    return base + (rank * 1000.0)


def _read_nvlink_topk_target_ranks(
    comm: NVLinkOneSided,
    max_num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    """Read topk_target_ranks[max_num_tokens, top_k] from NVLinkOneSided workspace."""
    from tensorrt_llm.bindings import internal as _tllm_internal

    offset_index = int(_tllm_internal.thop.MOE_A2A_TOPK_TARGET_RANKS_OFFSET_INDEX)
    offset = comm.moe_a2a_metainfo[offset_index].item()
    raw = comm.workspace[
        comm.ep_rank,
        offset : offset + max_num_tokens * top_k * 4,
    ]
    return raw.view(torch.int32).view(max_num_tokens, top_k).cpu()


def _read_nvlink_topk_send_indices(
    comm: NVLinkOneSided,
    max_num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    """Read topk_send_indices[max_num_tokens, top_k] from NVLinkOneSided workspace."""
    from tensorrt_llm.bindings import internal as _tllm_internal

    offset_index = int(_tllm_internal.thop.MOE_A2A_TOPK_SEND_INDICES_OFFSET_INDEX)
    offset = comm.moe_a2a_metainfo[offset_index].item()
    raw = comm.workspace[
        comm.ep_rank,
        offset : offset + max_num_tokens * top_k * 4,
    ]
    return raw.view(torch.int32).view(max_num_tokens, top_k).cpu()


def _run_nvlink_rank_mask_dispatch(
    comm: NVLinkOneSided,
    token_selected_experts: torch.Tensor,
    payload: torch.Tensor,
    runtime_max_tokens_per_rank: int,
    enable_rank_mask: bool,
    active_rank_mask: Optional[torch.Tensor],
) -> Tuple[List[torch.Tensor], int, torch.Tensor, torch.Tensor]:
    """Run raw NVLink one-sided dispatch with an optional active rank mask."""
    recv_tensors, combine_payload_offset, _ = torch.ops.trtllm.moe_a2a_dispatch(
        token_selected_experts,
        [payload],
        comm.workspace,
        comm.moe_a2a_metainfo,
        runtime_max_tokens_per_rank,
        comm.ep_rank,
        comm.ep_size,
        comm.top_k,
        comm.num_experts,
        None,  # eplb_local_stats
        enable_rank_mask,
        active_rank_mask,
    )

    topk_target_ranks = _read_nvlink_topk_target_ranks(
        comm,
        runtime_max_tokens_per_rank,
        comm.top_k,
    )
    topk_send_indices = _read_nvlink_topk_send_indices(
        comm,
        runtime_max_tokens_per_rank,
        comm.top_k,
    )
    return recv_tensors, int(combine_payload_offset), topk_target_ranks, topk_send_indices


def _run_nvlink_rank_mask_combine(
    comm: NVLinkOneSided,
    combine_payload: torch.Tensor,
    local_num_tokens: int,
    runtime_max_tokens_per_rank: int,
    combine_payload_offset: int,
    enable_rank_mask: bool,
    active_rank_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run raw NVLink one-sided combine with an optional active rank mask."""
    return torch.ops.trtllm.moe_a2a_combine(
        combine_payload,
        local_num_tokens,
        comm.workspace,
        comm.moe_a2a_metainfo,
        runtime_max_tokens_per_rank,
        comm.ep_rank,
        comm.ep_size,
        comm.top_k,
        combine_payload_offset,
        False,  # payload_in_workspace
        False,  # use_low_precision
        enable_rank_mask,
        active_rank_mask,
    )


def _run_nvlink_rank_mask_dispatch_combine(
    comm: NVLinkOneSided,
    token_selected_experts: torch.Tensor,
    payload: torch.Tensor,
    runtime_max_tokens_per_rank: int,
    enable_rank_mask: bool,
    active_rank_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run raw NVLink one-sided dispatch/combine with an optional active rank mask."""
    recv_tensors, combine_payload_offset, topk_target_ranks, topk_send_indices = (
        _run_nvlink_rank_mask_dispatch(
            comm,
            token_selected_experts,
            payload,
            runtime_max_tokens_per_rank,
            enable_rank_mask,
            active_rank_mask,
        )
    )
    combined = _run_nvlink_rank_mask_combine(
        comm,
        recv_tensors[0],
        token_selected_experts.size(0),
        runtime_max_tokens_per_rank,
        combine_payload_offset,
        enable_rank_mask,
        active_rank_mask,
    )
    return combined.cpu(), topk_target_ranks, topk_send_indices


def _expected_nvlink_rank_mask_combine_output(
    comm: NVLinkOneSided,
    payload: torch.Tensor,
    topk_target_ranks: torch.Tensor,
    topk_send_indices: torch.Tensor,
    local_num_tokens: int,
    runtime_max_tokens_per_rank: int,
) -> torch.Tensor:
    """Compute combine output from the routes recorded by dispatch."""
    from tensorrt_llm.bindings import internal as _tllm_internal

    hidden_size = payload.shape[-1]
    expected = torch.zeros(
        (local_num_tokens, hidden_size),
        dtype=torch.float32,
        device=payload.device,
    )
    payload_offset_index = int(_tllm_internal.thop.MOE_A2A_PAYLOAD_DATA_OFFSET_INDEX)
    payload_offset = comm.moe_a2a_metainfo[payload_offset_index].item()
    bytes_per_rank = (
        comm.ep_size * runtime_max_tokens_per_rank * hidden_size * payload.element_size()
    )

    for token_idx in range(local_num_tokens):
        for k in range(comm.top_k):
            target_rank = int(topk_target_ranks[token_idx, k].item())
            dst_idx = int(topk_send_indices[token_idx, k].item())
            if dst_idx < 0:
                continue
            raw = comm.workspace[target_rank, payload_offset : payload_offset + bytes_per_rank]
            recv_payload = raw.view(payload.dtype).view(
                comm.ep_size,
                runtime_max_tokens_per_rank,
                hidden_size,
            )
            expected[token_idx] += recv_payload[comm.ep_rank, dst_idx].float()
    return expected.to(payload.dtype).cpu()


# ============================================================================
# Source Encoding Utilities
# ============================================================================


def encode_source_info(
    hidden_states: torch.Tensor,
    rank_id: int,
) -> torch.Tensor:
    """Encode (rank_id, token_idx) into the last 4 bytes of each row.

    Format: [rank_id_u16, token_idx_u16] in big-endian.
    Works regardless of dtype because we operate on the raw byte view.
    """
    hs = hidden_states.clone()
    row_bytes = hidden_states.shape[1] * hidden_states.element_size()
    num_tokens = hidden_states.shape[0]

    if num_tokens == 0:
        return hs
    if not (0 <= rank_id <= 0xFFFF) or num_tokens > 0x10000:
        raise ValueError(
            f"Source encoding requires uint16 rank/token, got rank_id={rank_id}, "
            f"num_tokens={num_tokens}"
        )

    row_bytes_view = hs.view(torch.uint8).reshape(num_tokens, row_bytes)
    tail = row_bytes_view[:, -4:]
    token_idx = torch.arange(num_tokens, device=hidden_states.device, dtype=torch.int32)

    tail[:, 0] = rank_id >> 8
    tail[:, 1] = rank_id & 0xFF
    tail[:, 2] = (token_idx >> 8).to(torch.uint8)
    tail[:, 3] = (token_idx & 0xFF).to(torch.uint8)

    return hs


def decode_source_info(
    hidden_states: torch.Tensor,
    dtype: torch.dtype,
    hidden_size: int,
) -> torch.Tensor:
    """Decode (rank_id, token_idx) from the last 4 bytes of each row."""
    element_size = torch.tensor([], dtype=dtype).element_size()
    row_bytes = hidden_size * element_size
    flat_bytes = hidden_states.contiguous().view(torch.uint8).reshape(-1)
    num_rows = flat_bytes.numel() // row_bytes

    if num_rows == 0:
        return torch.empty(0, 2, dtype=torch.int64)

    tail = flat_bytes.reshape(num_rows, row_bytes)[:, -4:].cpu().to(torch.int32)
    rank_ids = (tail[:, 0] << 8) | tail[:, 1]
    token_idxs = (tail[:, 2] << 8) | tail[:, 3]

    return torch.stack((rank_ids, token_idxs), dim=1).to(torch.int64)


# ============================================================================
# Simple MoE Substitute (for combine verification)
# ============================================================================


def _compute_ep_partition(num_experts: int, ep_size: int, ep_rank: int) -> Tuple[int, int, int]:
    """Compute per-rank expert count and slot boundaries.

    Mirrors the kernel's compute_target_rank_id ceil/floor distribution:
    ranks [0, remainder) hold (base + 1) experts, and the remaining ranks
    hold base experts. Covers all experts even when num_experts % ep_size
    != 0 (non-divisible EP).

    Returns:
        (expert_size, slot_start, slot_end)
    """
    base = num_experts // ep_size
    remainder = num_experts % ep_size
    expert_size = base + (1 if ep_rank < remainder else 0)
    slot_start = ep_rank * base + min(ep_rank, remainder)
    return expert_size, slot_start, slot_start + expert_size


def _expert_id_to_rank(expert_id: int, num_experts: int, ep_size: int) -> int:
    """Inverse mapping of _compute_ep_partition. Mirrors kernel logic."""
    base = num_experts // ep_size
    remainder = num_experts % ep_size
    split = remainder * (base + 1)
    if expert_id < split:
        return expert_id // (base + 1)
    return remainder + (expert_id - split) // base


def simple_moe(
    hidden_states: torch.Tensor,
    token_selected_slots: torch.Tensor,
    token_final_scales: torch.Tensor,
    slot_start: int,
    slot_end: int,
) -> torch.Tensor:
    """Trivial MoE: weighted sum of hidden_states for local experts only.

    Each expert applies: output += hidden_states * scale.
    Dispatch verification already covers routing correctness, so experts
    do not need distinct computations.

    Uses dispatch-returned token_final_scales which handles the
    scale-application asymmetry:
    - AllGatherRS / NVLink / DeepEP: real scales (MoE applies them)
    - DeepEPLL: all-ones (combine applies real scales internally)
    """
    local_mask = (token_selected_slots >= slot_start) & (token_selected_slots < slot_end)
    local_scale_sum = (token_final_scales.float() * local_mask.float()).sum(dim=1)
    output = hidden_states.float() * local_scale_sum.unsqueeze(-1)

    return output.to(hidden_states.dtype)


# ============================================================================
# Communication Object Factory
# ============================================================================


def create_comm_object(
    comm_type: str,
    mapping: Mapping,
    config: CommTestConfig,
):
    """Create a Communication object for the given type and config."""
    num_experts = config.num_experts
    num_slots = num_experts
    max_num_tokens = max(config.all_num_tokens)

    # DeepEP / DeepEP LL need a mock quant_config for post-quant dispatch.
    # enable_postquant_alltoall is read from env var (default "1" = True),
    # NOT a constructor parameter -- do not pass it.
    qc = (
        _make_mock_quant_config(config.quant_mode)
        if config.quant_mode != "none" and comm_type in (COMM_DEEP_EP, COMM_DEEP_EP_LL)
        else None
    )

    if comm_type == COMM_ALLGATHER_RS:
        return AllGatherReduceScatter(mapping=mapping)

    elif comm_type == COMM_DEEP_EP:
        return DeepEP(
            mapping=mapping,
            num_slots=num_slots,
            hidden_size=config.hidden_size,
            weight_dtype=torch.bfloat16,
            quant_config=qc,
            expert_size_per_partition=num_experts // config.ep_size,
        )

    elif comm_type == COMM_DEEP_EP_LL:
        return DeepEPLowLatency(
            mapping=mapping,
            num_slots=num_slots,
            hidden_size=config.hidden_size,
            weight_dtype=torch.bfloat16,
            quant_config=qc,
            expert_size_per_partition=num_experts // config.ep_size,
            max_num_tokens=max_num_tokens,
            use_low_precision_combine=config.use_low_precision_combine,
            moe_max_num_tokens=max_num_tokens,
        )

    elif comm_type == COMM_NVLINK_ONE_SIDED:
        # Reset class-level singleton to avoid assertion failures when
        # test params change across MPI process reuse.
        NVLinkOneSided._WORKSPACE = None
        os.environ["TRTLLM_MOE_A2A_WORKSPACE_MB"] = NVLINK_WORKSPACE_MB

        return NVLinkOneSided(
            mapping=mapping,
            num_slots=num_slots,
            top_k=config.top_k,
            max_num_tokens_per_rank=max_num_tokens,
            hidden_size=config.hidden_size,
            dtype=torch.bfloat16,
            use_low_precision_combine=config.use_low_precision_combine,
        )

    elif comm_type == COMM_NVLINK_TWO_SIDED:
        # Keep combine() output reduced to [tokens, hidden] so it matches the
        # single-card reference and the production CommunicationFactory default.
        return NVLinkTwoSided(
            mapping=mapping,
            num_experts=num_experts,
            num_slots=num_slots,
            top_k=config.top_k,
            use_low_precision_combine=config.use_low_precision_combine,
            alltoall_result_do_sum=True,
        )

    elif comm_type == COMM_NVLINK_TWO_SIDED_FLASHINFER:
        # FlashInfer requires reduced combine output and does not support the
        # low-precision combine path.
        return NVLinkTwoSidedFlashinfer(
            mapping=mapping,
            num_experts=num_experts,
            num_slots=num_slots,
            top_k=config.top_k,
            alltoall_result_do_sum=True,
        )

    elif comm_type == COMM_NCCL_EP:
        return NcclEP(
            mapping=mapping,
            num_slots=num_slots,
            hidden_size=config.hidden_size,
            max_num_tokens=max_num_tokens,
            moe_max_num_tokens=max_num_tokens,
            top_k=config.top_k,
        )

    else:
        raise ValueError(f"Unknown comm type: {comm_type}")


_WORKER_COMM_KEY = None
_WORKER_COMM = None


def _comm_reuse_key(config: CommTestConfig) -> Tuple:
    """Return the constructor-affecting key used for worker-side comm reuse."""
    if config.comm_type == COMM_ALLGATHER_RS:
        return (config.comm_type, config.ep_size)

    if config.comm_type == COMM_DEEP_EP:
        return (
            config.comm_type,
            config.ep_size,
            config.num_experts,
            config.hidden_size,
            config.quant_mode,
        )

    if config.comm_type == COMM_DEEP_EP_LL:
        return (
            config.comm_type,
            config.ep_size,
            config.num_experts,
            config.hidden_size,
            config.quant_mode,
            max(config.all_num_tokens),
            config.use_low_precision_combine,
        )

    if config.comm_type == COMM_NVLINK_ONE_SIDED:
        return (
            config.comm_type,
            config.ep_size,
            config.num_experts,
            config.top_k,
            config.hidden_size,
            max(config.all_num_tokens),
            config.use_low_precision_combine,
        )

    if config.comm_type == COMM_NVLINK_TWO_SIDED:
        return (
            config.comm_type,
            config.ep_size,
            config.num_experts,
            config.top_k,
            config.use_low_precision_combine,
        )

    if config.comm_type == COMM_NVLINK_TWO_SIDED_FLASHINFER:
        return (
            config.comm_type,
            config.ep_size,
            config.num_experts,
            config.top_k,
        )

    return (config.comm_type,)


def _destroy_cached_worker_comm():
    """Destroy the cached worker comm object, if present."""
    global _WORKER_COMM_KEY, _WORKER_COMM
    if _WORKER_COMM is not None and hasattr(_WORKER_COMM, "destroy"):
        _WORKER_COMM.destroy()
    _WORKER_COMM_KEY = None
    _WORKER_COMM = None


def _get_worker_comm(mapping: Mapping, config: CommTestConfig):
    """Reuse the current worker comm object when the constructor key matches."""
    global _WORKER_COMM_KEY, _WORKER_COMM
    key = _comm_reuse_key(config)
    if _WORKER_COMM is not None and _WORKER_COMM_KEY != key:
        _destroy_cached_worker_comm()

    if _WORKER_COMM is None:
        _WORKER_COMM = create_comm_object(config.comm_type, mapping, config)
        _WORKER_COMM_KEY = key

    return _WORKER_COMM


atexit.register(_destroy_cached_worker_comm)


# ============================================================================
# Platform / Feasibility Checks
# ============================================================================


@lru_cache(maxsize=None)
def _check_mnnvl_support() -> Optional[str]:
    """Return skip reason if MNNVL is not supported, else None."""
    try:
        MnnvlMemory.initialize()
        if not MnnvlMemory.supports_mnnvl():
            return "MNNVL not supported"
    except Exception:
        return "MNNVL initialization failed"
    return None


@lru_cache(maxsize=None)
def _check_flashinfer_mnnvl_support() -> Optional[str]:
    """Return skip reason if FlashInfer MNNVL is not supported, else None."""
    try:
        if not NVLinkTwoSidedFlashinfer.is_platform_supported():
            return "FlashInfer MNNVL not supported"
    except Exception:
        return "FlashInfer MNNVL initialization failed"
    return None


@lru_cache(maxsize=None)
def check_platform_support(comm_type: str) -> Optional[str]:
    """Return skip reason string if comm type is unsupported, else None."""
    if comm_type == COMM_ALLGATHER_RS:
        return None

    if comm_type in (COMM_DEEP_EP, COMM_DEEP_EP_LL):
        if not deep_ep_installed:
            return "DeepEP library not installed"
        return _check_mnnvl_support()

    if comm_type == COMM_NVLINK_ONE_SIDED:
        return _check_mnnvl_support()

    if comm_type == COMM_NVLINK_TWO_SIDED:
        return _check_mnnvl_support()

    if comm_type == COMM_NVLINK_TWO_SIDED_FLASHINFER:
        return _check_flashinfer_mnnvl_support()

    if comm_type == COMM_NCCL_EP:
        if not is_nccl_ep_installed():
            return "NCCL EP not available (install the nccl4py wheel)"
        return None

    return f"Unknown comm type: {comm_type}"


def check_feasibility(comm_type: str, config: CommTestConfig) -> Optional[str]:
    """Return skip reason string if config is infeasible, else None."""
    if config.num_experts % config.ep_size != 0 and comm_type not in (
        COMM_NVLINK_ONE_SIDED,
        COMM_ALLGATHER_RS,
    ):
        # NVLinkOneSided supports non-divisible EP natively (ceil/floor
        # partitioning); AllGatherReduceScatter is the production fallback
        # selected by CommunicationFactory for the same case. Other comm
        # types still require num_experts divisible by ep_size.
        return (
            f"comm_type={comm_type} requires num_experts divisible by ep_size, "
            f"got num_experts={config.num_experts}, ep_size={config.ep_size}"
        )

    if config.use_low_precision_combine and not _supports_low_precision_combine(config):
        return (
            f"{comm_type} does not support use_low_precision_combine for "
            f"quant_mode={config.quant_mode}, hidden_size={config.hidden_size}"
        )

    if comm_type in (COMM_DEEP_EP, COMM_DEEP_EP_LL):
        if not DeepEP._is_deepep_feasible(config.ep_size):
            return f"DeepEP not feasible for ep_size={config.ep_size}"

    if comm_type == COMM_DEEP_EP_LL:
        qm = config.quant_mode
        if qm == "none":
            if config.hidden_size not in DeepEPLowLatency.SUPPORTED_HIDDEN_SIZES:
                return f"DeepEPLL does not support hidden_size={config.hidden_size}"
        elif qm == "nvfp4":
            if config.hidden_size not in DeepEPLowLatency.SUPPORTED_HIDDEN_SIZES_EXTENSION:
                return (
                    f"DeepEPLL nvfp4 requires hidden_size in "
                    f"SUPPORTED_HIDDEN_SIZES_EXTENSION, got {config.hidden_size}"
                )
        elif qm in ("fp8", "w4afp8"):
            if (config.hidden_size // 2) not in DeepEPLowLatency.SUPPORTED_HIDDEN_SIZES:
                return (
                    f"DeepEPLL {qm} requires hidden_size//2 in "
                    f"SUPPORTED_HIDDEN_SIZES, got {config.hidden_size}"
                )

    if comm_type == COMM_NVLINK_ONE_SIDED:
        if config.top_k > NVLinkOneSided.MAX_TOP_K:
            return f"NVLinkOneSided MAX_TOP_K={NVLinkOneSided.MAX_TOP_K}, got top_k={config.top_k}"

    if comm_type == COMM_NCCL_EP:
        if config.quant_mode != "none":
            return f"NcclEP does not support quant_mode={config.quant_mode}"

    if comm_type == COMM_NVLINK_TWO_SIDED_FLASHINFER:
        # FlashInfer alltoallv requires every 2D payload row to be 16-byte aligned.
        # This test dispatches both int32 slots [N, top_k] and bf16 scales
        # [N, top_k], so top_k must satisfy both alignments.
        if config.top_k % 8 != 0:
            return (
                "NVLinkTwoSidedFlashinfer requires top_k to be a multiple of 8 "
                f"for 16-byte row alignment, got top_k={config.top_k}"
            )

    # W4AFP8: encode_source_info writes token_idx into fp8 low byte.
    # token_idx >= 127 can create fp8 NaN which may not survive bf16 roundtrip.
    if config.quant_mode == "w4afp8":
        max_tokens = max(config.all_num_tokens)
        if max_tokens >= 127:
            return f"W4AFP8 requires max_tokens < 127, got {max_tokens}"

    return None


# ============================================================================
# Worker Function (runs on each MPI rank)
# ============================================================================


def _generate_test_data(
    rank: int,
    config: CommTestConfig,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate test data for a single rank.

    Returns (hidden_states, token_selected_slots, token_final_scales).
    hidden_states has source info encoded in last 4 bytes of each row.
    """
    num_tokens = config.all_num_tokens[rank]

    torch.manual_seed(seed + rank)

    hidden_states = torch.randn(num_tokens, config.hidden_size, dtype=torch.bfloat16, device="cuda")
    hidden_states = encode_source_info(hidden_states, rank)

    token_selected_slots = torch.randint(
        0, config.num_experts, (num_tokens, config.top_k), dtype=torch.int32, device="cuda"
    )

    token_final_scales = torch.rand(num_tokens, config.top_k, dtype=torch.bfloat16, device="cuda")
    # Avoid near-zero scales that amplify relative error in bf16 verification.
    token_final_scales = token_final_scales.clamp(min=0.1)

    return hidden_states, token_selected_slots, token_final_scales


# ============================================================================
# Post-Quant Helpers
# ============================================================================


def _make_mock_quant_config(quant_mode: str) -> MagicMock:
    """Build a mock QuantConfig with the correct nested attribute paths.

    DeepEP checks:  quant_config.layer_quant_mode.has_nvfp4()
    DeepEP LL:      quant_config.layer_quant_mode.has_fp8_qdq()
                    quant_config.layer_quant_mode.has_nvfp4()
                    quant_config.quant_mode.is_int4_weight_only_per_group()
    """
    mock = MagicMock()
    mock.layer_quant_mode.has_fp8_qdq.return_value = quant_mode == "fp8"
    mock.layer_quant_mode.has_nvfp4.return_value = quant_mode == "nvfp4"
    mock.quant_mode.is_int4_weight_only_per_group.return_value = quant_mode == "w4afp8"
    return mock


def _generate_postquant_data(
    rank: int,
    config: CommTestConfig,
    seed: int = 42,
) -> Tuple[
    torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor
]:
    """Generate quantized test data using real quant ops.

    Returns (hidden_states, hidden_states_sf, global_scale,
             token_selected_slots, token_final_scales).
    Source info is NOT encoded here -- the caller encodes after this returns.
    """
    num_tokens = config.all_num_tokens[rank]
    H = config.hidden_size

    torch.manual_seed(seed + rank)
    bf16_hs = torch.randn(num_tokens, H, dtype=torch.bfloat16, device="cuda")

    if config.quant_mode == "fp8":
        hs = bf16_hs.to(torch.float8_e4m3fn)
        sf = None
        global_scale = None

    elif config.quant_mode == "nvfp4":
        global_scale = torch.ones(num_tokens, 1, device="cuda", dtype=torch.float32)
        hs, sf = Buffer.quantize_bf16_to_nvfp4(bf16_hs, global_scale)

    elif config.quant_mode == "w4afp8":
        hs = bf16_hs.to(torch.float8_e4m3fn)
        sf = None
        global_scale = None

    else:
        raise ValueError(f"Unknown quant_mode: {config.quant_mode}")

    token_selected_slots = torch.randint(
        0,
        config.num_experts,
        (num_tokens, config.top_k),
        dtype=torch.int32,
        device="cuda",
    )
    token_final_scales = torch.rand(
        num_tokens,
        config.top_k,
        dtype=torch.bfloat16,
        device="cuda",
    ).clamp(min=0.1)

    return hs, sf, global_scale, token_selected_slots, token_final_scales


def _to_bf16(
    hs: torch.Tensor,
    sf: Optional[torch.Tensor],
    global_scale: Optional[torch.Tensor],
    quant_mode: str,
) -> torch.Tensor:
    """Convert hidden states to bf16 for simple_moe.

    For quant_mode 'none', hs is already bf16 from dispatch; return as-is.
    FP8 / w4afp8 use PyTorch dtype conversion. NVFP4 uses Buffer CUDA dequant.
    Must run on CUDA for nvfp4.
    """
    if quant_mode == "none":
        return hs
    if quant_mode in ("fp8", "w4afp8"):
        return hs.to(torch.bfloat16)
    elif quant_mode == "nvfp4":
        # global_scale must be (N, 1) where N == hs.size(0).  After dispatch
        # the received token count differs from the original, so recreate.
        if global_scale.size(0) != hs.size(0):
            global_scale = torch.ones(hs.size(0), 1, device=hs.device, dtype=torch.float32)
        return Buffer.dequantize_nvfp4_to_bf16(hs, global_scale, sf)
    raise ValueError(f"Unknown quant_mode: {quant_mode}")


def _prepare_worker_inputs(rank: int, config: CommTestConfig) -> WorkerInputs:
    """Generate and encode per-rank inputs before dispatch."""
    if config.quant_mode == "none":
        hs, slots, scales = _generate_test_data(rank, config)
        return WorkerInputs(
            hs=hs,
            hidden_states_sf=None,
            global_scale=None,
            slots=slots,
            scales=scales,
            dispatch_kwargs={},
        )

    hs, hidden_states_sf, global_scale, slots, scales = _generate_postquant_data(rank, config)

    if config.quant_mode == "w4afp8":
        # Encode (rank_id, token_idx) in the FP8 payload so dispatch verification can
        # match received rows to sources. DeepEPLowLatency w4afp8 expects bf16
        # hidden_states at dispatch() and re-quantizes inside dispatch via
        # (hs * pre_quant_scale).to(float8).view(bf16), so we must round-trip
        # through bf16 before the kernel sees the tensor.
        hs = encode_source_info(hs, rank)
        original_fp8 = hs.clone()
        # Preflight: if FP8 -> bf16 -> FP8 changes any element, byte-level
        # dispatch checks against original_fp8 would be misleading (encoded
        # trailing bytes may not survive conversion). verify_dispatch_alltoall
        # skips per-row content equality when any rank sets this to False.
        roundtrip_fp8 = hs.to(torch.bfloat16).to(torch.float8_e4m3fn)
        w4afp8_roundtrip_ok = torch.equal(hs, roundtrip_fp8)
        hs = hs.to(torch.bfloat16)
        pre_quant_scale = torch.ones(
            1,
            config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        return WorkerInputs(
            hs=hs,
            hidden_states_sf=hidden_states_sf,
            global_scale=global_scale,
            slots=slots,
            scales=scales,
            dispatch_kwargs={"pre_quant_scale": pre_quant_scale},
            original_fp8=original_fp8,
            w4afp8_roundtrip_ok=w4afp8_roundtrip_ok,
        )

    return WorkerInputs(
        hs=encode_source_info(hs, rank),
        hidden_states_sf=hidden_states_sf,
        global_scale=global_scale,
        slots=slots,
        scales=scales,
        dispatch_kwargs={},
    )


def _run_worker_dispatch(
    comm,
    worker_inputs: WorkerInputs,
    config: CommTestConfig,
) -> DispatchOutputs:
    """Prepare dispatch metadata if needed, then run dispatch()."""
    if config.comm_type in (COMM_NVLINK_TWO_SIDED, COMM_NVLINK_TWO_SIDED_FLASHINFER):
        comm.prepare_dispatch(worker_inputs.slots, config.all_num_tokens)

    recv_hs, recv_sf, recv_slots, recv_scales = comm.dispatch(
        worker_inputs.hs,
        worker_inputs.hidden_states_sf,
        worker_inputs.slots,
        worker_inputs.scales,
        config.all_num_tokens,
        # Normalize invalid / non-local expert ids across backends so
        # dispatch verification can use a consistent slot contract.
        enable_sanitize_expert_ids=True,
        **worker_inputs.dispatch_kwargs,
    )

    return DispatchOutputs(
        recv_hs=recv_hs,
        recv_sf=recv_sf,
        recv_slots=recv_slots,
        recv_scales=recv_scales,
    )


def _build_worker_result(
    rank: int,
    worker_inputs: WorkerInputs,
    dispatch_outputs: DispatchOutputs,
    combined: torch.Tensor,
    moe_output: torch.Tensor,
    recv_source_info: torch.Tensor,
    config: CommTestConfig,
    recv_hs_bf16: Optional[torch.Tensor] = None,
    moe_output_for_ref: Optional[torch.Tensor] = None,
) -> dict:
    """Collect tensors needed by host-side verification."""
    result = {
        "rank": rank,
        "original_hs": _safe_cpu(worker_inputs.hs),
        "original_hs_sf": _safe_cpu(worker_inputs.hidden_states_sf),
        "original_slots": worker_inputs.slots.cpu(),
        "original_scales": worker_inputs.scales.cpu(),
        "recv_hs": _safe_cpu(dispatch_outputs.recv_hs),
        "recv_sf": _safe_cpu(dispatch_outputs.recv_sf),
        "recv_slots": dispatch_outputs.recv_slots.cpu(),
        "recv_scales": (
            dispatch_outputs.recv_scales.cpu() if dispatch_outputs.recv_scales is not None else None
        ),
        "combined": combined.cpu(),
        "recv_source_info": recv_source_info,
    }
    if moe_output_for_ref is not None:
        result["moe_output_for_ref"] = moe_output_for_ref.cpu()
    else:
        result["moe_output"] = moe_output.cpu()

    if recv_hs_bf16 is not None:
        result["recv_hs_bf16"] = recv_hs_bf16.cpu()

    if config.quant_mode == "w4afp8":
        result["original_fp8"] = _safe_cpu(worker_inputs.original_fp8)
        result["w4afp8_roundtrip_ok"] = worker_inputs.w4afp8_roundtrip_ok

    return result


def _prepare_moe_output_for_combine_reference(
    moe_output: torch.Tensor,
    config: CommTestConfig,
) -> Optional[torch.Tensor]:
    """Precompute low-precision combine reference payload on the worker GPU."""
    if not config.use_low_precision_combine:
        return None

    if config.comm_type == COMM_NVLINK_ONE_SIDED:
        return moe_output.to(torch.float8_e4m3fn).to(torch.bfloat16)

    if config.comm_type == COMM_NVLINK_TWO_SIDED:
        return _simulate_nvfp4_round_trip(moe_output)

    return None


def _worker_full_pipeline(config: CommTestConfig) -> dict:
    """Run dispatch -> simple_moe -> combine on a single MPI rank.

    Returns both dispatch intermediate results (for dispatch verification)
    and final combine output (for combine verification).

    Post-quant flow (quant_mode != "none"):
      generate quantized data -> encode source info in quantized domain
      -> dispatch -> dequant -> simple_moe -> combine.
    W4AFP8 special: encode on fp8, convert to bf16 for dispatch with
      pre_quant_scale=ones, preflight-check roundtrip fidelity.
    """
    rank = tllm.mpi_rank()
    torch.cuda.set_device(rank)

    try:
        mapping = Mapping(
            rank=rank,
            tp_size=config.ep_size,
            moe_ep_size=config.ep_size,
            world_size=config.ep_size,
        )

        comm = _get_worker_comm(mapping, config)

        worker_inputs = _prepare_worker_inputs(rank, config)
        dispatch_outputs = _run_worker_dispatch(comm, worker_inputs, config)

        # ----- decode routing + dequant + MoE + combine -----
        # Decode source info from raw dispatch output (before dequant) so we
        # know which original (src_rank, token_idx) each dispatched row maps to.
        # Using recv_hs.dtype and recv_hs.shape[1] ensures correct byte layout
        # regardless of quant format (bf16, fp8, nvfp4, w4afp8-as-bf16).
        recv_source_info = decode_source_info(
            dispatch_outputs.recv_hs,
            dispatch_outputs.recv_hs.dtype,
            dispatch_outputs.recv_hs.shape[1],
        )

        recv_hs_bf16 = _to_bf16(
            dispatch_outputs.recv_hs,
            dispatch_outputs.recv_sf,
            worker_inputs.global_scale,
            config.quant_mode,
        )

        # Use ceil/floor partitioning so the slot range is correct for both
        # uniform and non-divisible EP (num_experts % ep_size != 0).
        _, local_slot_start, local_slot_end = _compute_ep_partition(
            config.num_experts, config.ep_size, rank
        )
        # Run a minimal local-expert compute step before combine(). This keeps
        # the test aligned with the real MoE pipeline, where combine() consumes
        # per-rank expert outputs rather than raw dispatch payloads.
        moe_output = simple_moe(
            recv_hs_bf16,
            dispatch_outputs.recv_slots,
            dispatch_outputs.recv_scales,
            local_slot_start,
            local_slot_end,
        )

        combined = comm.combine(
            moe_output,
            all_rank_max_num_tokens=max(config.all_num_tokens),
        )

        # Save recv_hs_bf16 for DeepEPLL combine reference (weighted reduce
        # needs raw dispatched hidden_states, not accumulated moe_output).
        saved_recv_hs_bf16 = recv_hs_bf16.clone() if config.comm_type == COMM_DEEP_EP_LL else None
        moe_output_for_ref = _prepare_moe_output_for_combine_reference(moe_output, config)

        return _build_worker_result(
            rank,
            worker_inputs,
            dispatch_outputs,
            combined,
            moe_output,
            recv_source_info,
            config,
            recv_hs_bf16=saved_recv_hs_bf16,
            moe_output_for_ref=moe_output_for_ref,
        )
    except Exception:
        _destroy_cached_worker_comm()
        traceback.print_exc()
        raise


def _nccl_ep_replay_slots(
    *,
    target_rank: int,
    num_tokens: int,
    experts_per_rank: int,
) -> torch.Tensor:
    """Route every local token to one EP rank, using distinct local experts."""
    local_experts = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
    local_experts %= experts_per_rank
    return (target_rank * experts_per_rank + local_experts).view(num_tokens, 1)


def _worker_nccl_ep_cuda_graph_replay(config: CommTestConfig) -> dict:
    """Capture LL dispatch, change routing, and verify the replay sees the change.

    ``NcclEP.dispatch`` converts the stable input routing tensor to the dtype
    expected by nccl-ep inside the graph. The captured handle therefore must
    consume the updated device buffer on each replay, rather than reusing the
    routes present while the graph was captured.
    """
    rank = tllm.mpi_rank()
    torch.cuda.set_device(rank)
    comm = None
    try:
        mapping = Mapping(
            rank=rank,
            tp_size=config.ep_size,
            moe_ep_size=config.ep_size,
            world_size=config.ep_size,
        )
        comm = create_comm_object(COMM_NCCL_EP, mapping, config)
        num_tokens = config.all_num_tokens[rank]
        experts_per_rank = config.num_experts // config.ep_size
        all_rank_num_tokens = config.all_num_tokens

        # A rank-tagged payload makes a routing change visible without relying
        # on private nccl-ep state: local routing receives rank + 1, while the
        # second replay must receive the peer rank tag.
        hidden_states = torch.full(
            (num_tokens, config.hidden_size),
            float(rank + 1),
            dtype=torch.bfloat16,
            device="cuda",
        )
        weights = torch.ones(num_tokens, 1, dtype=torch.float32, device="cuda")
        local_routes = _nccl_ep_replay_slots(
            target_rank=rank,
            num_tokens=num_tokens,
            experts_per_rank=experts_per_rank,
        )
        peer_rank = (rank + 1) % config.ep_size
        peer_routes = _nccl_ep_replay_slots(
            target_rank=peer_rank,
            num_tokens=num_tokens,
            experts_per_rank=experts_per_rank,
        )

        # Initialize the context and handle eagerly. Capture is intentionally
        # rejected before this point, so this mirrors production graph setup.
        comm.dispatch(
            hidden_states,
            None,
            local_routes,
            weights,
            all_rank_num_tokens,
        )
        torch.cuda.synchronize()

        static_routes = local_routes.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            recv_hs, _, recv_slots, _ = comm.dispatch(
                hidden_states,
                None,
                static_routes,
                weights,
                all_rank_num_tokens,
            )
        torch.cuda.synchronize()

        def replay_and_check(expected_sender: int) -> dict:
            graph.replay()
            torch.cuda.synchronize()
            valid = recv_slots[:, 0] >= 0
            received = recv_hs[valid, 0].to(torch.float32).round().to(torch.int64)
            return {
                "valid_count": int(valid.sum().item()),
                "sender_matches": bool(torch.all(received == expected_sender + 1).item()),
            }

        static_routes.copy_(local_routes)
        local_result = replay_and_check(rank)
        static_routes.copy_(peer_routes)
        peer_result = replay_and_check(peer_rank)
        return {"rank": rank, "local": local_result, "peer": peer_result}
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if comm is not None:
            comm.destroy()


def _make_rank_mask_config(
    ep_size: int,
    local_num_tokens: int,
    top_k: int,
) -> CommTestConfig:
    """Build the small NVLinkOneSided config used by active-rank-mask tests."""
    return CommTestConfig(
        comm_type=COMM_NVLINK_ONE_SIDED,
        ep_size=ep_size,
        num_experts=FIXED_NUM_EXPERTS,
        top_k=top_k,
        hidden_size=1024,
        all_num_tokens=[local_num_tokens] * ep_size,
    )


def _worker_rank_mask_all_active_matches_no_mask(config: CommTestConfig) -> dict:
    """Check that all-active active_rank_mask is bit-identical to no mask."""
    rank = tllm.mpi_rank()
    torch.cuda.set_device(rank)

    comm = None
    try:
        mapping = Mapping(
            rank=rank,
            tp_size=config.ep_size,
            moe_ep_size=config.ep_size,
            world_size=config.ep_size,
        )
        comm = create_comm_object(config.comm_type, mapping, config)

        local_num_tokens = config.all_num_tokens[rank]
        torch.manual_seed(0xA2A + rank)
        token_selected_experts = torch.randint(
            0,
            config.num_experts,
            (local_num_tokens, config.top_k),
            dtype=torch.int32,
            device="cuda",
        )
        payload = _make_rank_mask_payload(local_num_tokens, config.hidden_size, rank)
        all_active_mask = _ep_mask_words(config.ep_size, dead_ranks=set())

        with pytest.raises(RuntimeError, match="requires enable_rank_mask"):
            _run_nvlink_rank_mask_dispatch(
                comm,
                token_selected_experts,
                payload,
                local_num_tokens,
                enable_rank_mask=False,
                active_rank_mask=all_active_mask,
            )
        with pytest.raises(RuntimeError, match="active_rank_mask must be defined"):
            _run_nvlink_rank_mask_dispatch(
                comm,
                token_selected_experts,
                payload,
                local_num_tokens,
                enable_rank_mask=True,
                active_rank_mask=None,
            )

        out_no_mask, topk_no_mask, _ = _run_nvlink_rank_mask_dispatch_combine(
            comm,
            token_selected_experts,
            payload,
            local_num_tokens,
            enable_rank_mask=False,
            active_rank_mask=None,
        )
        out_all_active, topk_all_active, _ = _run_nvlink_rank_mask_dispatch_combine(
            comm,
            token_selected_experts,
            payload,
            local_num_tokens,
            enable_rank_mask=True,
            active_rank_mask=all_active_mask,
        )

        return {
            "rank": rank,
            "output_eq": torch.equal(out_no_mask, out_all_active),
            "topk_eq": torch.equal(topk_no_mask, topk_all_active),
        }
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if comm is not None and hasattr(comm, "destroy"):
            comm.destroy()


def _expected_target_ranks(
    token_selected_experts: torch.Tensor,
    num_experts: int,
    ep_size: int,
) -> torch.Tensor:
    """Map each selected expert to its target EP rank using the kernel partition rule."""
    token_selected_experts_cpu = token_selected_experts.cpu()
    expected = torch.empty_like(token_selected_experts_cpu)
    for token_idx in range(token_selected_experts_cpu.shape[0]):
        for k in range(token_selected_experts_cpu.shape[1]):
            expert_id = int(token_selected_experts_cpu[token_idx, k].item())
            expected[token_idx, k] = _expert_id_to_rank(expert_id, num_experts, ep_size)
    return expected


def _worker_rank_mask_one_rank_masked(
    config: CommTestConfig,
    dead_rank: int,
) -> dict:
    """Verify masked-route rejection, then run with survivor-only routing."""
    rank = tllm.mpi_rank()
    torch.cuda.set_device(rank)

    comm = None
    try:
        mapping = Mapping(
            rank=rank,
            tp_size=config.ep_size,
            moe_ep_size=config.ep_size,
            world_size=config.ep_size,
        )
        # All ranks must initialize the symmetric workspace before the dead rank
        # stops participating in dispatch/combine.
        comm = create_comm_object(config.comm_type, mapping, config)

        if rank == dead_rank:
            MPI.COMM_WORLD.barrier()
            return {"rank": rank, "status": "dead"}

        local_num_tokens = config.all_num_tokens[rank]
        torch.manual_seed(0xA2A + rank)
        mask = _ep_mask_words(config.ep_size, dead_ranks={dead_rank})
        dead_expert_id = next(
            expert_id
            for expert_id in range(config.num_experts)
            if _expert_id_to_rank(expert_id, config.num_experts, config.ep_size) == dead_rank
        )
        masked_routes = torch.full(
            (local_num_tokens, config.top_k),
            dead_expert_id,
            dtype=torch.int32,
            device="cuda",
        )
        _, _, masked_target_ranks, masked_send_indices = _run_nvlink_rank_mask_dispatch(
            comm,
            masked_routes,
            _make_rank_mask_payload(local_num_tokens, config.hidden_size, rank),
            local_num_tokens,
            enable_rank_mask=True,
            active_rank_mask=mask,
        )

        live_expert_ids = torch.tensor(
            [
                expert_id
                for expert_id in range(config.num_experts)
                if _expert_id_to_rank(expert_id, config.num_experts, config.ep_size) != dead_rank
            ],
            dtype=torch.int32,
            device="cuda",
        )
        live_expert_indices = torch.randint(
            0,
            live_expert_ids.numel(),
            (local_num_tokens, config.top_k),
            dtype=torch.int64,
            device="cuda",
        )
        token_selected_experts = live_expert_ids[live_expert_indices]
        payload = _make_rank_mask_payload(local_num_tokens, config.hidden_size, rank)

        combined, topk_target_ranks, topk_send_indices = _run_nvlink_rank_mask_dispatch_combine(
            comm,
            token_selected_experts,
            payload,
            local_num_tokens,
            enable_rank_mask=True,
            active_rank_mask=mask,
        )
        expected_target_ranks = _expected_target_ranks(
            token_selected_experts,
            config.num_experts,
            config.ep_size,
        )
        expected = _expected_nvlink_rank_mask_combine_output(
            comm,
            payload,
            topk_target_ranks,
            topk_send_indices,
            local_num_tokens,
            local_num_tokens,
        )

        MPI.COMM_WORLD.barrier()
        return {
            "rank": rank,
            "status": "alive",
            "masked_target_ranks": masked_target_ranks[:local_num_tokens],
            "masked_send_indices": masked_send_indices[:local_num_tokens],
            "combined": combined,
            "expected": expected,
            "topk_target_ranks": topk_target_ranks,
            "expected_target_ranks": expected_target_ranks,
        }
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if comm is not None and hasattr(comm, "destroy"):
            comm.destroy()


# ============================================================================
# Verification Functions
# ============================================================================


def verify_dispatch_allgather_rs(
    all_results: List[dict],
    config: CommTestConfig,
):
    """Verify AllGatherRS: allgathered data is bitwise-exact concatenation.

    For nvfp4 mode, also verifies hidden_states_sf is allgathered correctly.
    """
    total_tokens = sum(config.all_num_tokens)

    for result in all_results:
        recv_hs = result["recv_hs"]
        assert recv_hs.shape[0] == total_tokens, (
            f"Rank {result['rank']}: expected {total_tokens} tokens, got {recv_hs.shape[0]}"
        )

        offset = 0
        for src_rank in range(config.ep_size):
            src_hs = all_results[src_rank]["original_hs"]
            n = config.all_num_tokens[src_rank]
            assert torch.equal(recv_hs[offset : offset + n], src_hs), (
                f"Rank {result['rank']}: chunk for source rank {src_rank} mismatch"
            )
            offset += n

        # For nvfp4, verify hidden_states_sf (scale factors) are gathered too.
        if config.quant_mode == "nvfp4":
            recv_sf = result.get("recv_sf")
            if recv_sf is not None:
                sf_offset = 0
                for src_rank in range(config.ep_size):
                    src_sf = all_results[src_rank].get("original_hs_sf")
                    n = config.all_num_tokens[src_rank]
                    if src_sf is not None:
                        assert torch.equal(recv_sf[sf_offset : sf_offset + n], src_sf), (
                            f"Rank {result['rank']}: sf chunk for source rank {src_rank} mismatch"
                        )
                    sf_offset += n


def _compute_expected_tokens_per_rank(
    all_results: List[dict],
    config: CommTestConfig,
) -> Dict[int, Set[Tuple[int, int]]]:
    """Compute the set of (src_rank, token_idx) each rank should receive.

    A token is expected at dest_rank if at least one of its top_k expert
    selections falls in dest_rank's expert range. Uses the same ceil/floor
    partitioning as the kernel so this works for non-divisible EP as well.
    """
    expected: Dict[int, Set[Tuple[int, int]]] = {r: set() for r in range(config.ep_size)}
    for result in all_results:
        src_rank = result["rank"]
        slots = result["original_slots"]

        for i in range(slots.shape[0]):
            for k in range(slots.shape[1]):
                eid = slots[i, k].item()
                if 0 <= eid < config.num_experts:
                    target_rank = _expert_id_to_rank(eid, config.num_experts, config.ep_size)
                    expected[target_rank].add((src_rank, i))

    return expected


def _get_dispatch_decode_spec(config: CommTestConfig) -> Tuple[torch.dtype, int]:
    """Return decode dtype and hidden size used by decode_source_info()."""
    qm = config.quant_mode
    if qm == "fp8":
        return torch.float8_e4m3fn, config.hidden_size
    if qm == "nvfp4":
        return torch.uint8, config.hidden_size // 2
    if qm == "w4afp8":
        return torch.float8_e4m3fn, config.hidden_size
    return torch.bfloat16, config.hidden_size


def _build_original_data_lookup(
    all_results: List[dict],
    quant_mode: str,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Build a (src_rank, token_idx) -> original row lookup for content checks."""
    original_data: Dict[Tuple[int, int], torch.Tensor] = {}
    for result in all_results:
        src_rank = result["rank"]
        orig = result["original_fp8"] if quant_mode == "w4afp8" else result["original_hs"]
        for i in range(orig.shape[0]):
            original_data[(src_rank, i)] = orig[i]
    return original_data


def _validate_dispatch_row_slots(
    recv_slots_row: torch.Tensor,
    token_idx: int,
    recv_rank: int,
    slot_start: int,
    slot_end: int,
    num_experts: int,
    comm_type: str,
) -> bool:
    """Validate one received row of slots and return whether any slot is local."""
    has_valid = False
    for k in range(recv_slots_row.shape[0]):
        slot = recv_slots_row[k].item()
        if comm_type == COMM_DEEP_EP_LL:
            is_local = slot_start <= slot < slot_end
            is_invalid = slot == num_experts
            assert is_local or is_invalid, (
                f"Rank {recv_rank}, token {token_idx}, k={k}: slot={slot} "
                f"not local [{slot_start},{slot_end}) and "
                f"not invalid marker ({num_experts})"
            )
        else:
            assert -1 <= slot < num_experts, (
                f"Rank {recv_rank}, token {token_idx}, k={k}: slot={slot} "
                f"out of valid range [-1, {num_experts})"
            )
            is_local = slot_start <= slot < slot_end
        has_valid = has_valid or is_local
    return has_valid


def _record_received_token(
    recv_hs: torch.Tensor,
    decoded: torch.Tensor,
    token_idx: int,
    recv_rank: int,
    original_data: Dict[Tuple[int, int], torch.Tensor],
    skip_content_check: bool,
    actually_received: Set[Tuple[int, int]],
) -> None:
    """Record one received token and optionally verify row content."""
    src_rank, src_idx = decoded[token_idx].tolist()
    key = (src_rank, src_idx)
    actually_received.add(key)
    if key in original_data and not skip_content_check:
        assert torch.equal(recv_hs[token_idx], original_data[key]), (
            f"Rank {recv_rank}, token {token_idx}: content mismatch. "
            f"Source: rank={src_rank}, idx={src_idx}"
        )


def _check_dispatch_completeness(
    recv_rank: int,
    actually_received: Set[Tuple[int, int]],
    expected_per_rank: Dict[int, Set[Tuple[int, int]]],
    comm_type: str,
) -> None:
    """Check that a rank received all tokens expected by routing."""
    # DeepEPLL rewrites output slots in _modify_output_to_adapt_fused_moe,
    # so returned tensors no longer preserve source-level completeness.
    if comm_type == COMM_DEEP_EP_LL:
        return

    expected_tokens = expected_per_rank[recv_rank]
    missing = expected_tokens - actually_received
    assert not missing, (
        f"Rank {recv_rank}: {len(missing)} expected tokens not received. "
        f"Expected {len(expected_tokens)}, got {len(actually_received)}. "
        f"Missing (first 5): {list(missing)[:5]}"
    )


def verify_dispatch_alltoall(
    all_results: List[dict],
    config: CommTestConfig,
):
    """Verify AllToAll dispatch: slot validity, content integrity, completeness.

    Slot semantics differ by communication backend:
      - NVLinkOneSided / NVLinkTwoSided / NVLinkTwoSidedFlashinfer:
        valid tokens keep global expert IDs
        [0, num_experts), padding tokens are set to -1 by sanitize kernel.
      - DeepEP: valid tokens keep global expert IDs [0, num_experts), empty-
        tensor padding uses num_experts as invalid marker.
      - DeepEPLowLatency: _modify_output_to_adapt_fused_moe creates local
        expert IDs [slot_start, slot_end), padding uses num_experts.

    Checks:
    1. Per-comm-type slot range validation (see above)
    2. Non-padding tokens must have at least one LOCAL slot
    3. Encoded (rank_id, token_idx) in received data matches original content
    4. All tokens that should be routed to this rank are present (completeness)
    """
    num_experts = config.num_experts

    qm = config.quant_mode

    # Global lookup: (src_rank, token_idx) -> original hidden_states row.
    # For w4afp8, use original_fp8 (fp8 domain) for content comparison.
    original_data = _build_original_data_lookup(all_results, qm)
    expected_per_rank = _compute_expected_tokens_per_rank(all_results, config)

    # For w4afp8: if any rank's preflight roundtrip failed, skip content check
    w4afp8_skip_content = False
    if qm == "w4afp8":
        w4afp8_skip_content = not all(r.get("w4afp8_roundtrip_ok", True) for r in all_results)

    for result in all_results:
        recv_rank = result["rank"]
        recv_hs = result["recv_hs"]
        recv_slots = result["recv_slots"]

        # Per-rank slot range — handles non-divisible EP.
        _, slot_start, slot_end = _compute_ep_partition(num_experts, config.ep_size, recv_rank)

        decoded = result["recv_source_info"]
        actually_received: Set[Tuple[int, int]] = set()

        for i in range(recv_hs.shape[0]):
            has_valid = _validate_dispatch_row_slots(
                recv_slots[i],
                i,
                recv_rank,
                slot_start,
                slot_end,
                num_experts,
                config.comm_type,
            )
            if has_valid:
                _record_received_token(
                    recv_hs,
                    decoded,
                    i,
                    recv_rank,
                    original_data,
                    w4afp8_skip_content,
                    actually_received,
                )

        _check_dispatch_completeness(
            recv_rank,
            actually_received,
            expected_per_rank,
            config.comm_type,
        )


def verify_dispatch_results(
    all_results: List[dict],
    config: CommTestConfig,
):
    """Route to the appropriate dispatch verification based on comm type."""
    if config.comm_type == COMM_ALLGATHER_RS:
        verify_dispatch_allgather_rs(all_results, config)
    else:
        verify_dispatch_alltoall(all_results, config)


def _simulate_nvfp4_round_trip(tensor: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    """Simulate NVFP4 E2M1 quantize-dequantize round-trip per row.

    Matches the two-level scaling scheme in fusedMoeCommKernels.cu:
      quantize_nvfp4_sharedmem / dequantize_nvfp4_sharedmem.

    Level 1 (per-group fp8 scale): Each group of ``group_size`` elements
    gets an fp8_e4m3 scale factor derived from the group's absmax.
    Level 2 (per-row global fp32 scale): SFScaleVal = 448*6 / row_amax.

    Quantize path (per row):
      SFScaleVal = 448 * 6 / global_amax
      sf8 = fp8(SFScaleVal * group_max / 6)     # per-group fp8 scale
      output_scale = SFScaleVal / float(sf8)
      e2m1_val = round_to_e2m1(val * output_scale)

    Dequantize path (per row):
      dequant_scale = float(sf8) / SFScaleVal
      result = e2m1_val * dequant_scale

    E2M1 representable values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
    """
    if tensor.numel() == 0:
        return tensor

    x = tensor.float()
    N, H = x.shape
    assert H % group_size == 0

    # E2M1 lookup tables (same as torch_quant.py)
    e2m1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    e2m1_pos_vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)

    # Per-row global absmax (kernel computes global_amax per send-field)
    row_amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)  # [N, 1]

    # Global scale per row: SFScaleVal = (448 * 6) / row_amax
    sf_scale_val = (448.0 * 6.0) / row_amax  # [N, 1]

    # Reshape into groups
    x_grouped = x.reshape(N, H // group_size, group_size)

    # Per-group vecMax
    vec_max = x_grouped.abs().amax(dim=-1)  # [N, H//group_size]

    # Per-group fp8 scale: sf8 = fp8(SFScaleVal * vecMax / 6)
    sf_value = sf_scale_val * vec_max / 6.0
    sf8 = sf_value.to(torch.float8_e4m3fn)
    sf_narrow = sf8.float()

    # output_scale = SFScaleVal / float(sf8)
    # Kernel uses reciprocal_approximate_ftz (~1 ULP); we use exact division.
    # If the per-group fp8 scale underflows to zero, that group dequantizes to
    # zero. Avoid computing 0 * inf for zero elements in such groups.
    valid_scale = (vec_max > 0) & (sf_narrow > 0)
    safe_sf_narrow = torch.where(valid_scale, sf_narrow, torch.ones_like(sf_narrow))
    output_scale = torch.where(
        valid_scale,
        sf_scale_val / safe_sf_narrow,
        torch.zeros_like(sf_narrow),
    )

    # Scale each element and quantize to E2M1
    scaled = x_grouped * output_scale.unsqueeze(-1)
    sign = scaled.sign()
    abs_scaled = scaled.abs()
    idx = torch.searchsorted(e2m1_bounds, abs_scaled).clamp(max=7)
    e2m1_quantized = e2m1_pos_vals[idx] * sign

    # Dequantize: result = e2m1_val * float(sf8) / SFScaleVal
    dequant_scale = torch.where(
        sf_narrow > 0,
        sf_narrow / sf_scale_val,
        torch.zeros_like(sf_narrow),
    )
    result = e2m1_quantized * dequant_scale.unsqueeze(-1)

    return result.reshape(N, H).to(torch.bfloat16)


def _valid_recv_row_mask(proc_result: dict, config: CommTestConfig) -> torch.Tensor:
    """Return rows that correspond to real dispatched tokens, not padding."""
    recv_slots = proc_result["recv_slots"]
    if config.comm_type == COMM_ALLGATHER_RS:
        return torch.ones(recv_slots.shape[0], dtype=torch.bool)

    proc_rank = proc_result["rank"]
    _, slot_start, slot_end = _compute_ep_partition(config.num_experts, config.ep_size, proc_rank)
    return ((recv_slots >= slot_start) & (recv_slots < slot_end)).any(dim=1)


def _target_source_mask(
    proc_result: dict,
    target_rank: int,
    num_tokens: int,
    config: CommTestConfig,
) -> torch.Tensor:
    """Return valid rows in proc_result that contribute to target_rank."""
    source_info = proc_result["recv_source_info"]
    valid_rows = _valid_recv_row_mask(proc_result, config)
    return valid_rows & (source_info[:, 0] == target_rank) & (source_info[:, 1] < num_tokens)


def _build_combine_reference(
    all_results: List[dict],
    target_rank: int,
    config: CommTestConfig,
) -> torch.Tensor:
    """Build combine reference from actual per-rank data + routing info.

    Reference paths matched to kernel behavior:

    1a. NVLinkOneSided + low_precision_combine: Simulates fp8 quant/dequant
        (moeA2APrepareCombineKernel/moeA2ACombineKernel use bf16->fp8->float32
        accumulation).

    1b. NVLinkTwoSided + low_precision_combine: Simulates NVFP4 E2M1
        quant/dequant (fusedMoeCommKernels.cu quantize_nvfp4_sharedmem /
        dequantize_nvfp4_sharedmem) with two-level scaling + float32
        accumulation.

    2.  DeepEP low-latency: Reconstructs weighted reduction using real
        topk_weights (original_scales), since DeepEPLL's combine internally
        applies weights rather than receiving pre-scaled moe_output.

    3.  Default: Accumulates moe_output in float32 to match the kernel's
        use of float32 registers for cross-rank reduction.

    All paths produce bf16 output to match the final kernel output dtype.
    """
    num_tokens = config.all_num_tokens[target_rank]
    hidden_size = config.hidden_size

    # Use float32 accumulator for all paths (matches kernel behavior)
    ref = torch.zeros(num_tokens, hidden_size, dtype=torch.float32)

    if config.use_low_precision_combine and config.comm_type == COMM_NVLINK_ONE_SIDED:
        # Path 1a: NVLinkOneSided fp8 quantize/dequantize + float32 accumulation.
        # The fp8 round-trip is precomputed on the worker GPU to keep parent-side
        # verification from repeating per-row quantization work on CPU tensors.
        for proc_result in all_results:
            moe_out = proc_result["moe_output_for_ref"]
            source_info = proc_result["recv_source_info"]
            target_mask = _target_source_mask(proc_result, target_rank, num_tokens, config)
            if target_mask.any():
                ref.index_add_(0, source_info[target_mask, 1], moe_out[target_mask].float())

    elif config.use_low_precision_combine and config.comm_type == COMM_NVLINK_TWO_SIDED:
        # Path 1b: NVLinkTwoSided NVFP4 simulation + float32 accumulation.
        # fusedMoeCommKernels.cu quantize_nvfp4_sharedmem uses two-level
        # scaling: per-row global fp32 scale + per-group-of-16 fp8 scale,
        # with E2M1 quantization. After NVLink transfer,
        # dequantize_nvfp4_sharedmem reverses the process. The top_k
        # reduction is then done in bf16 by torch.sum in _mnnvl_utils.py. The
        # NVFP4 round-trip is precomputed on the worker GPU.
        for proc_result in all_results:
            nvfp4_out = proc_result["moe_output_for_ref"]
            source_info = proc_result["recv_source_info"]
            target_mask = _target_source_mask(proc_result, target_rank, num_tokens, config)
            if target_mask.any():
                ref.index_add_(0, source_info[target_mask, 1], nvfp4_out[target_mask].float())

    elif config.comm_type == COMM_DEEP_EP_LL:
        # Path 2: DeepEPLL weighted reduction.
        # DeepEPLL's dispatch returns ones as recv_scales, so simple_moe
        # produces unweighted output. The combine kernel (low_latency_combine)
        # internally applies real topk_weights during weighted reduction.
        # Reconstruct by weighting recv_hs_bf16 per local expert with the
        # real weights from original_scales on the source rank.
        target_original_scales = all_results[target_rank]["original_scales"]

        for proc_result in all_results:
            proc_rank = proc_result["rank"]
            recv_hs_bf16 = proc_result["recv_hs_bf16"]
            recv_slots = proc_result["recv_slots"]
            source_info = proc_result["recv_source_info"]
            _, slot_start, slot_end = _compute_ep_partition(
                config.num_experts, config.ep_size, proc_rank
            )

            target_mask = _target_source_mask(proc_result, target_rank, num_tokens, config)
            if target_mask.any():
                target_source_info = source_info[target_mask]
                local_slot_mask = (recv_slots[target_mask] >= slot_start) & (
                    recv_slots[target_mask] < slot_end
                )
                weights = target_original_scales[target_source_info[:, 1]].float()
                local_scale_sum = (weights * local_slot_mask.float()).sum(dim=1)
                contributions = recv_hs_bf16[target_mask].float() * local_scale_sum.unsqueeze(-1)
                ref.index_add_(0, target_source_info[:, 1], contributions)

    else:
        # Path 3: Default — float32 accumulation.
        # The combine kernel (vectorized_combine_impl) uses float32
        # registers for cross-rank reduction, then casts to bf16.
        for proc_result in all_results:
            moe_out = proc_result["moe_output"]
            source_info = proc_result["recv_source_info"]
            target_mask = _target_source_mask(proc_result, target_rank, num_tokens, config)
            if target_mask.any():
                ref.index_add_(0, source_info[target_mask, 1], moe_out[target_mask].float())

    return ref.to(torch.bfloat16)


def verify_combine_results(
    all_results: List[dict],
    config: CommTestConfig,
    rtol: float = 0.05,
    atol: float = 0.1,
):
    """Verify combine results against kernel-behavior-matched reference.

    Builds the reference using _build_combine_reference which has three paths
    matching actual kernel behavior:
    - lpcombine: fp8 quant/dequant + float32 accumulation
    - DeepEPLL: weighted reduction with real topk_weights
    - Default: float32 accumulation
    All paths isolate ONLY the combine communication error.
    """
    for result in all_results:
        rank = result["rank"]
        combined = result["combined"]
        num_tokens = config.all_num_tokens[rank]

        if num_tokens == 0:
            continue

        ref = _build_combine_reference(all_results, rank, config)

        combined_tokens = combined[:num_tokens]
        ref_tokens = ref[:num_tokens]

        assert combined_tokens.shape == ref_tokens.shape, (
            f"Rank {rank}: shape mismatch. combined={combined_tokens.shape}, ref={ref_tokens.shape}"
        )

        try:
            torch.testing.assert_close(
                combined_tokens.float(),
                ref_tokens.float(),
                rtol=rtol,
                atol=atol,
            )
        except AssertionError as e:
            diff = (combined_tokens.float() - ref_tokens.float()).abs()
            max_idx = diff.argmax().item()
            token_idx = max_idx // combined_tokens.shape[1]
            elem_idx = max_idx % combined_tokens.shape[1]
            raise AssertionError(
                f"Rank {rank}: combine mismatch at token={token_idx}, elem={elem_idx}. "
                f"combined={combined_tokens[token_idx, elem_idx]:.6f}, "
                f"ref={ref_tokens[token_idx, elem_idx]:.6f}, "
                f"max_abs_diff={diff.max():.6f}\n"
                f"Original error: {e}"
            ) from e


# ============================================================================
# Test Parameter Generation
# ============================================================================


POSTQUANT_COMM_MAP: Dict[str, List[str]] = {
    "fp8": [
        COMM_NVLINK_ONE_SIDED,
        COMM_NVLINK_TWO_SIDED,
        COMM_NVLINK_TWO_SIDED_FLASHINFER,
        COMM_DEEP_EP_LL,
        COMM_ALLGATHER_RS,
    ],
    "nvfp4": [
        COMM_NVLINK_ONE_SIDED,
        COMM_NVLINK_TWO_SIDED,
        COMM_NVLINK_TWO_SIDED_FLASHINFER,
        COMM_DEEP_EP,
        COMM_DEEP_EP_LL,
        COMM_ALLGATHER_RS,
    ],
    "w4afp8": [COMM_DEEP_EP_LL],
}
"""Only valid (quant_mode, comm_type) combinations for post-quant tests.

 - fp8: NVLink (payload-agnostic, including FlashInfer variant) + DeepEP LL
   (fp8 branch) + AllGatherRS.
  DeepEP normal only supports nvfp4 post-quant, NOT fp8.
 - nvfp4: All tested COMM types except w4afp8-only branches support nvfp4
   post-quant.
- w4afp8: Only DeepEP LL has the w4afp8 dispatch branch.
"""


def _supports_low_precision_combine(config: CommTestConfig) -> bool:
    """Return whether this config can exercise the low-precision combine path."""
    # NVLink backends accept low-precision combine for both the regular and
    # post-quant test matrices. DeepEPLowLatency only supports it on the
    # quantized branches guarded below.
    if config.comm_type in (COMM_NVLINK_ONE_SIDED, COMM_NVLINK_TWO_SIDED):
        return True

    if config.comm_type == COMM_DEEP_EP_LL:
        if config.hidden_size not in DeepEPLowLatency.SUPPORTED_HIDDEN_SIZES_EXTENSION:
            return False
        return config.quant_mode in ("fp8", "nvfp4", "w4afp8")

    return False


def _is_static_feasible(config: CommTestConfig) -> bool:
    """Return whether a generated config can run before platform checks."""
    return check_feasibility(config.comm_type, config) is None


def _make_workloads(ep_size: int) -> List[List[int]]:
    """Generate token distributions: uniform, non-uniform, minimal."""
    workloads = [[32] * ep_size]

    if ep_size == 2:
        workloads.append([16, 48])
    elif ep_size == 4:
        workloads.append([16, 32, 48, 64])

    workloads.append([1] * ep_size)
    return workloads


def _make_grouped_params(configs: List[CommTestConfig]) -> List:
    """Group configs so each pytest item can pipeline same-comm workloads."""
    grouped: Dict[Tuple[int, str], List[CommTestConfig]] = {}
    for config in configs:
        grouped.setdefault((config.ep_size, config.comm_type), []).append(config)

    # Sort each group by the worker-side comm reuse key so configs sharing a
    # constructor key run back-to-back and actually hit the single-slot cache
    # (e.g. postquant NVLinkOneSided variants differing only in quant_mode).
    return [
        pytest.param(
            ep_size,
            CommTestGroup(sorted(group_configs, key=_comm_reuse_key)),
            id=str(CommTestGroup(group_configs)),
        )
        for (ep_size, _), group_configs in grouped.items()
    ]


def _make_test_params():
    """Generate grouped full-pipeline test parameters.

    Each entry is (ep_size, group). ep_size is passed to mpi_pool_executor
    via indirect parametrization; group is passed directly to the test.
    """
    configs = []
    # Keep equal-sized MPI pools adjacent to reduce MPIPoolExecutor churn.
    for ep_size in [2, 4]:
        for comm_type in ALL_COMM_TYPES:
            for top_k in [2, 4, 8]:
                for workload in _make_workloads(ep_size):
                    for use_low_precision_combine in [False, True]:
                        config = CommTestConfig(
                            comm_type=comm_type,
                            ep_size=ep_size,
                            num_experts=FIXED_NUM_EXPERTS,
                            top_k=top_k,
                            hidden_size=DEFAULT_HIDDEN_SIZE,
                            all_num_tokens=workload,
                            use_low_precision_combine=use_low_precision_combine,
                        )
                        if not _is_static_feasible(config):
                            continue
                        configs.append(config)
    return _make_grouped_params(configs)


def _make_boundary_test_params():
    """Generate boundary / edge-case test parameters."""
    configs = []
    for target_ep_size in [2, 4]:
        for comm_type in ALL_COMM_TYPES:
            aligned_top_k = 8 if comm_type == COMM_NVLINK_TWO_SIDED_FLASHINFER else 2
            boundary_cases = []
            if comm_type != COMM_NVLINK_TWO_SIDED_FLASHINFER:
                boundary_cases.append(
                    (
                        2,
                        CommTestConfig(
                            comm_type=comm_type,
                            ep_size=2,
                            num_experts=FIXED_NUM_EXPERTS,
                            top_k=1,
                            hidden_size=DEFAULT_HIDDEN_SIZE,
                            all_num_tokens=[8, 8],
                        ),
                        f"{comm_type}_topk1",
                    )
                )
            boundary_cases.extend(
                [
                    (
                        2,
                        CommTestConfig(
                            comm_type=comm_type,
                            ep_size=2,
                            num_experts=FIXED_NUM_EXPERTS,
                            top_k=aligned_top_k,
                            hidden_size=2048,
                            all_num_tokens=[16, 16],
                        ),
                        f"{comm_type}_h2048",
                    ),
                    (
                        4,
                        CommTestConfig(
                            comm_type=comm_type,
                            ep_size=4,
                            num_experts=FIXED_NUM_EXPERTS,
                            top_k=aligned_top_k,
                            hidden_size=DEFAULT_HIDDEN_SIZE,
                            all_num_tokens=[1, 1, 1, 1],
                        ),
                        f"{comm_type}_single_token",
                    ),
                ]
            )

            # Zero tokens on some ranks (DeepEPLL kernel does not support this).
            if comm_type != COMM_DEEP_EP_LL:
                boundary_cases.append(
                    (
                        4,
                        CommTestConfig(
                            comm_type=comm_type,
                            ep_size=4,
                            num_experts=FIXED_NUM_EXPERTS,
                            top_k=aligned_top_k,
                            hidden_size=DEFAULT_HIDDEN_SIZE,
                            all_num_tokens=[32, 0, 16, 0],
                        ),
                        f"{comm_type}_zero_tokens",
                    )
                )

            for ep_size, base_config, case_id in boundary_cases:
                if ep_size != target_ep_size:
                    continue
                for use_low_precision_combine in [False, True]:
                    config = CommTestConfig(
                        comm_type=base_config.comm_type,
                        ep_size=base_config.ep_size,
                        num_experts=base_config.num_experts,
                        top_k=base_config.top_k,
                        hidden_size=base_config.hidden_size,
                        all_num_tokens=base_config.all_num_tokens,
                        quant_mode=base_config.quant_mode,
                        use_low_precision_combine=use_low_precision_combine,
                    )
                    if not _is_static_feasible(config):
                        continue
                    configs.append(config)

    return _make_grouped_params(configs)


def _make_non_divisible_ep_test_params():
    """Generate non-divisible EP test parameters for NVLinkOneSided.

    Exercises ceil/floor expert partitioning where num_experts % ep_size != 0:
        base      = num_experts // ep_size
        remainder = num_experts %  ep_size
        Ranks [0, remainder)        own (base + 1) experts each
        Ranks [remainder, ep_size)  own  base      experts each

    Limited to NVLinkOneSided because other comm backends still require
    num_experts divisible by ep_size (enforced by check_feasibility).
    """
    configs = []
    # (ep_size, num_experts, top_k, all_num_tokens) — picked to cover both
    # uneven ratios (5 / 2 = 2 + 1) and a larger remainder (17 / 4).
    cases = [
        (2, 5, 2, [16, 16]),  # base=2, remainder=1: rank0=3, rank1=2
        (4, 17, 2, [16, 16, 16, 16]),  # base=4, remainder=1: rank0=5, others=4
        (4, 22, 4, [8, 8, 8, 8]),  # base=5, remainder=2: rank0..1=6, rank2..3=5
    ]
    for ep_size, num_experts, top_k, all_num_tokens in cases:
        for use_low_precision_combine in [False, True]:
            config = CommTestConfig(
                comm_type=COMM_NVLINK_ONE_SIDED,
                ep_size=ep_size,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=DEFAULT_HIDDEN_SIZE,
                all_num_tokens=all_num_tokens,
                use_low_precision_combine=use_low_precision_combine,
            )
            if not _is_static_feasible(config):
                continue
            configs.append(config)
    return _make_grouped_params(configs)


def _make_postquant_test_params():
    """Generate post-quant test parameters using POSTQUANT_COMM_MAP.

    Uses simplified workloads (ep_size=2, top_k=2, small tokens) to keep
    the matrix manageable while covering all valid (comm_type, quant_mode)
    combinations.
    """
    configs = []
    for quant_mode, comm_types in POSTQUANT_COMM_MAP.items():
        for comm_type in comm_types:
            top_k = 8 if comm_type == COMM_NVLINK_TWO_SIDED_FLASHINFER else 2
            for use_low_precision_combine in [False, True]:
                config = CommTestConfig(
                    comm_type=comm_type,
                    ep_size=2,
                    num_experts=FIXED_NUM_EXPERTS,
                    top_k=top_k,
                    hidden_size=DEFAULT_HIDDEN_SIZE,
                    all_num_tokens=[16, 16],
                    quant_mode=quant_mode,
                    use_low_precision_combine=use_low_precision_combine,
                )
                if not _is_static_feasible(config):
                    continue
                configs.append(config)
    return _make_grouped_params(configs)


# ============================================================================
# Pytest Fixtures & Test Runner
# ============================================================================


@pytest.fixture(autouse=True)
def setup_test() -> None:
    torch.manual_seed(0x1234)
    tllm.logger.set_level("error")


def _get_skip_reason(config: CommTestConfig) -> Optional[str]:
    """Return skip reason for a config, or None if it should run."""
    skip_reason = check_platform_support(config.comm_type)
    if skip_reason:
        return skip_reason

    skip_reason = check_feasibility(config.comm_type, config)
    if skip_reason:
        return skip_reason

    if config.ep_size > torch.cuda.device_count():
        return f"Need {config.ep_size} GPUs but only {torch.cuda.device_count()} available"

    return None


def _submit_worker_pipeline(mpi_pool_executor, config: CommTestConfig) -> PendingWorkerResults:
    """Submit one config to all MPI workers without waiting for verification."""
    futures = [
        mpi_pool_executor.submit(_worker_full_pipeline, config) for _ in range(config.ep_size)
    ]
    return PendingWorkerResults(config=config, futures=futures)


def _collect_worker_results(pending: PendingWorkerResults) -> List[dict]:
    """Collect worker results in rank order for host-side verification."""
    return sorted((future.result() for future in pending.futures), key=lambda r: r["rank"])


def _drain_pending_results(pending: PendingWorkerResults):
    """Wait for already-submitted work before propagating a verification error."""
    for future in pending.futures:
        try:
            future.result()
        except Exception:
            pass


def _verify_full_test_results(all_results: List[dict], config: CommTestConfig):
    """Run host-side dispatch and combine verification for one config."""
    try:
        verify_dispatch_results(all_results, config)

        # Reference matches kernel behavior per comm type:
        # - NVLinkOneSided lpc: fp8 simulation (matches moeA2ACombineKernel)
        # - NVLinkTwoSided lpc: NVFP4 simulation (matches fusedMoeCommKernels.cu)
        # - DeepEPLL: weighted reduce with real topk_weights
        # - Default: float32 accumulation (matches kernel registers)
        if config.use_low_precision_combine:
            if config.comm_type == COMM_NVLINK_ONE_SIDED:
                # FP8 simulation closely matches kernel — tight tolerance
                verify_combine_results(all_results, config, rtol=0.02, atol=0.15)
            elif config.comm_type == COMM_NVLINK_TWO_SIDED:
                # NVFP4 simulation matches kernel's two-level scaling.
                # Residual error from rcp.approx.ftz.f32 vs exact division
                # causes ~0.35/element boundary effects at E2M1 midpoints
                # (fp8 scale rounding difference pushes elements across
                # quantization boundaries). Scales linearly with top_k.
                nvfp4_atol = 0.4 * config.top_k
                verify_combine_results(all_results, config, rtol=0.1, atol=nvfp4_atol)
            else:
                # DeepEPLL low-precision combine transfers in fp8 while the
                # reference reduction stays unquantized, so it needs a bound
                # looser than the non-lpc DeepEPLL branch below. Keep the
                # previous (conservative) top_k-scaled tolerance.
                verify_combine_results(all_results, config, rtol=0.1, atol=0.4 * config.top_k)
        elif config.comm_type == COMM_DEEP_EP_LL:
            verify_combine_results(all_results, config, rtol=0.05, atol=0.3)
        else:
            verify_combine_results(all_results, config, rtol=0.02, atol=0.15)
    except Exception as e:
        raise AssertionError(f"{config}: {e}") from e


def _run_full_test(mpi_pool_executor, config: CommTestConfig):
    """Run dispatch -> verify dispatch -> simple_moe -> combine -> verify combine."""
    skip_reason = _get_skip_reason(config)
    if skip_reason:
        pytest.skip(skip_reason)

    pending = _submit_worker_pipeline(mpi_pool_executor, config)
    all_results = _collect_worker_results(pending)
    _verify_full_test_results(all_results, config)


def _run_full_test_group(mpi_pool_executor, group: CommTestGroup):
    """Run a same-comm group, overlapping parent verification with next dispatch."""
    runnable_configs = []
    skip_reasons = []
    for config in group.configs:
        skip_reason = _get_skip_reason(config)
        if skip_reason:
            skip_reasons.append(f"{config}: {skip_reason}")
        else:
            runnable_configs.append(config)

    if not runnable_configs:
        pytest.skip("; ".join(skip_reasons[:3]))

    pending = _submit_worker_pipeline(mpi_pool_executor, runnable_configs[0])

    for next_config in runnable_configs[1:]:
        current_pending = pending
        all_results = _collect_worker_results(current_pending)
        pending = _submit_worker_pipeline(mpi_pool_executor, next_config)
        try:
            _verify_full_test_results(all_results, current_pending.config)
        except Exception:
            _drain_pending_results(pending)
            raise

    all_results = _collect_worker_results(pending)
    _verify_full_test_results(all_results, pending.config)


def _run_nccl_ep_cuda_graph_replay_test(mpi_pool_executor) -> None:
    """Verify graph replay observes routing changed between replays."""
    ep_size = mpi_pool_executor.num_workers
    config = CommTestConfig(
        comm_type=COMM_NCCL_EP,
        ep_size=ep_size,
        num_experts=FIXED_NUM_EXPERTS,
        top_k=1,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        all_num_tokens=[16] * ep_size,
    )
    skip_reason = _get_skip_reason(config)
    if skip_reason:
        pytest.skip(skip_reason)

    futures = [
        mpi_pool_executor.submit(_worker_nccl_ep_cuda_graph_replay, config)
        for _ in range(config.ep_size)
    ]
    results = sorted((future.result() for future in futures), key=lambda result: result["rank"])
    for result in results:
        rank = result["rank"]
        for replay_name in ("local", "peer"):
            replay = result[replay_name]
            valid_count = replay["valid_count"]
            assert valid_count == 16, (
                f"rank {rank}: {replay_name} replay received {valid_count} valid rows, expected 16"
            )
            assert replay["sender_matches"], (
                f"rank {rank}: {replay_name} replay did not observe the expected routing buffer"
            )


def _skip_if_rank_mask_config_unsupported(config: CommTestConfig) -> None:
    """Skip active-rank-mask tests when NVLinkOneSided cannot run locally."""
    skip_reason = check_platform_support(config.comm_type)
    if skip_reason:
        pytest.skip(skip_reason)

    skip_reason = check_feasibility(config.comm_type, config)
    if skip_reason:
        pytest.skip(skip_reason)

    if config.ep_size > torch.cuda.device_count():
        pytest.skip(f"Need {config.ep_size} GPUs but only {torch.cuda.device_count()} available")


def _run_rank_mask_all_active_test(
    mpi_pool_executor,
    local_num_tokens: int,
    top_k: int,
) -> None:
    ep_size = mpi_pool_executor.num_workers
    config = _make_rank_mask_config(ep_size, local_num_tokens, top_k)
    _skip_if_rank_mask_config_unsupported(config)

    results = list(
        mpi_pool_executor.map(
            _worker_rank_mask_all_active_matches_no_mask,
            *zip(*[(config,)] * config.ep_size),
        )
    )

    for result in results:
        rank = result["rank"]
        assert result["output_eq"], (
            f"rank {rank}: combine output differs between no-mask and all-active mask"
        )
        assert result["topk_eq"], (
            f"rank {rank}: topk_target_ranks differ between no-mask and all-active mask"
        )


def _run_rank_mask_one_rank_masked_test(
    mpi_pool_executor,
    dead_rank: int,
    local_num_tokens: int,
    top_k: int,
) -> None:
    ep_size = mpi_pool_executor.num_workers
    config = _make_rank_mask_config(ep_size, local_num_tokens, top_k)
    _skip_if_rank_mask_config_unsupported(config)
    assert 0 <= dead_rank < ep_size

    worker_args = [(config, dead_rank)] * config.ep_size
    results = list(
        mpi_pool_executor.map(
            _worker_rank_mask_one_rank_masked,
            *zip(*worker_args),
        )
    )

    saw_dead = False
    for result in results:
        rank = result["rank"]
        if result["status"] == "dead":
            assert rank == dead_rank
            saw_dead = True
            continue

        assert result["status"] == "alive"
        assert torch.all(result["masked_target_ranks"] == -1), (
            f"rank {rank}: dispatch retained a route to masked rank {dead_rank}"
        )
        assert torch.all(result["masked_send_indices"] == -1), (
            f"rank {rank}: dispatch allocated a send slot for masked rank {dead_rank}"
        )
        combined = result["combined"]
        expected = result["expected"]
        topk_target_ranks = result["topk_target_ranks"]
        expected_target_ranks = result["expected_target_ranks"]

        assert combined.shape == (local_num_tokens, config.hidden_size)
        assert torch.equal(combined, expected), (
            f"rank {rank}: combine output does not match the pre-routed live payloads"
        )

        live_topk = topk_target_ranks[:local_num_tokens]
        live_expected = expected_target_ranks[:local_num_tokens]
        for token_idx in range(local_num_tokens):
            seen_ranks: Set[int] = set()
            for k in range(top_k):
                expected = int(live_expected[token_idx, k].item())
                got = int(live_topk[token_idx, k].item())
                assert expected != dead_rank
                if expected in seen_ranks:
                    assert got == -1
                else:
                    assert got == expected, (
                        f"rank {rank} token {token_idx} k={k}: target rank mismatch "
                        f"(expected={expected}, got={got})"
                    )
                    seen_ranks.add(expected)

    assert saw_dead, f"dead rank {dead_rank} did not appear in results"


# ============================================================================
# Test Class
# ============================================================================


class TestMoEComm:
    """Full-pipeline tests for all MoE Communication types.

    Each test: dispatch -> verify dispatch -> simple_moe -> combine
    -> verify combine.  Dispatch errors are caught early with clear
    diagnostics before combine verification runs.
    """

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize(
        "mpi_pool_executor,group",
        _make_test_params(),
        indirect=["mpi_pool_executor"],
    )
    def test_moe_comm(self, mpi_pool_executor, group: CommTestGroup):
        """Verify full dispatch -> compute -> combine pipeline."""
        _run_full_test_group(mpi_pool_executor, group)

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize(
        "mpi_pool_executor,group",
        _make_boundary_test_params(),
        indirect=["mpi_pool_executor"],
    )
    def test_moe_comm_boundary(self, mpi_pool_executor, group: CommTestGroup):
        """Test full pipeline with boundary / edge-case parameters."""
        _run_full_test_group(mpi_pool_executor, group)

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize(
        "mpi_pool_executor,group",
        _make_postquant_test_params(),
        indirect=["mpi_pool_executor"],
    )
    def test_moe_comm_postquant(self, mpi_pool_executor, group: CommTestGroup):
        """Verify post-quant dispatch -> dequant -> MoE -> combine pipeline."""
        _run_full_test_group(mpi_pool_executor, group)

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize(
        "mpi_pool_executor,group",
        _make_non_divisible_ep_test_params(),
        indirect=["mpi_pool_executor"],
    )
    def test_moe_comm_non_divisible_ep(self, mpi_pool_executor, group: CommTestGroup):
        """Verify NVLinkOneSided with non-divisible EP (num_experts % ep_size != 0)."""
        _run_full_test_group(mpi_pool_executor, group)

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
    def test_nccl_ep_cuda_graph_replay_uses_updated_routing(self, mpi_pool_executor) -> None:
        """Verify LL CUDA graph replay reads routing written after capture."""
        _run_nccl_ep_cuda_graph_replay_test(mpi_pool_executor)

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize(
        "mpi_pool_executor,local_num_tokens,top_k",
        [
            (4, 16, 2),
            (4, 32, 4),
        ],
        indirect=["mpi_pool_executor"],
    )
    def test_moe_comm_rank_mask_all_active_matches_no_mask(
        self,
        mpi_pool_executor,
        local_num_tokens: int,
        top_k: int,
    ) -> None:
        """Verify all-active active_rank_mask matches omitted mask for NVLinkOneSided."""
        _run_rank_mask_all_active_test(mpi_pool_executor, local_num_tokens, top_k)

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize(
        "mpi_pool_executor,dead_rank,local_num_tokens,top_k",
        [
            (4, 2, 16, 2),
            (4, 0, 16, 4),
            (4, 3, 32, 4),
        ],
        indirect=["mpi_pool_executor"],
    )
    def test_moe_comm_rank_mask_one_rank_masked_completes(
        self,
        mpi_pool_executor,
        dead_rank: int,
        local_num_tokens: int,
        top_k: int,
    ) -> None:
        """Verify peer synchronization skips a masked rank after routing excludes it."""
        _run_rank_mask_one_rank_masked_test(
            mpi_pool_executor,
            dead_rank,
            local_num_tokens,
            top_k,
        )
