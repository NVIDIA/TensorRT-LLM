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
  - NVLinkTwoSided lpc: float32 accum (NVFP4 too complex to simulate)
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

import os
import pickle
import struct
import sys
import traceback
from dataclasses import dataclass
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
from tensorrt_llm._torch.modules.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.modules.fused_moe.communication.nvlink_two_sided import NVLinkTwoSided
from tensorrt_llm._torch.modules.fused_moe.communication.nvlink_two_sided_flashinfer import (
    NVLinkTwoSidedFlashinfer,
)
from tensorrt_llm._torch.modules.fused_moe.deep_ep_utils import deep_ep_installed
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

ALL_COMM_TYPES = [
    COMM_ALLGATHER_RS,
    COMM_DEEP_EP,
    COMM_DEEP_EP_LL,
    COMM_NVLINK_ONE_SIDED,
    COMM_NVLINK_TWO_SIDED,
    COMM_NVLINK_TWO_SIDED_FLASHINFER,
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
    flat_bytes = hs.view(torch.uint8).reshape(-1)
    row_bytes = hidden_states.shape[1] * hidden_states.element_size()
    num_tokens = hidden_states.shape[0]

    for i in range(num_tokens):
        offset = i * row_bytes + (row_bytes - 4)
        packed = struct.pack(">HH", rank_id, i)
        for j, b in enumerate(packed):
            flat_bytes[offset + j] = b

    return hs


def decode_source_info(
    hidden_states: torch.Tensor,
    dtype: torch.dtype,
    hidden_size: int,
) -> List[Tuple[int, int]]:
    """Decode (rank_id, token_idx) from the last 4 bytes of each row."""
    element_size = torch.tensor([], dtype=dtype).element_size()
    row_bytes = hidden_size * element_size
    flat_bytes = hidden_states.contiguous().view(torch.uint8).reshape(-1)
    num_rows = flat_bytes.numel() // row_bytes
    results = []

    for i in range(num_rows):
        offset = i * row_bytes + (row_bytes - 4)
        raw = bytes(flat_bytes[offset : offset + 4].cpu().tolist())
        rank_id, token_idx = struct.unpack(">HH", raw)
        results.append((rank_id, token_idx))

    return results


# ============================================================================
# Simple MoE Substitute (for combine verification)
# ============================================================================


def simple_moe(
    hidden_states: torch.Tensor,
    token_selected_slots: torch.Tensor,
    token_final_scales: torch.Tensor,
    ep_rank: int,
    experts_per_rank: int,
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
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    slot_start = ep_rank * experts_per_rank
    slot_end = slot_start + experts_per_rank

    for i in range(hidden_states.shape[0]):
        for k in range(token_selected_slots.shape[1]):
            eid = token_selected_slots[i, k].item()
            if not (slot_start <= eid < slot_end):
                continue
            output[i] += hidden_states[i].float() * token_final_scales[i, k].float()

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

    else:
        raise ValueError(f"Unknown comm type: {comm_type}")


# ============================================================================
# Platform / Feasibility Checks
# ============================================================================


def _check_mnnvl_support() -> Optional[str]:
    """Return skip reason if MNNVL is not supported, else None."""
    try:
        MnnvlMemory.initialize()
        if not MnnvlMemory.supports_mnnvl():
            return "MNNVL not supported"
    except Exception:
        return "MNNVL initialization failed"
    return None


def _check_flashinfer_mnnvl_support() -> Optional[str]:
    """Return skip reason if FlashInfer MNNVL is not supported, else None."""
    try:
        if not NVLinkTwoSidedFlashinfer.is_platform_supported():
            return "FlashInfer MNNVL not supported"
    except Exception:
        return "FlashInfer MNNVL initialization failed"
    return None


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

    return f"Unknown comm type: {comm_type}"


def check_feasibility(comm_type: str, config: CommTestConfig) -> Optional[str]:
    """Return skip reason string if config is infeasible, else None."""
    if config.num_experts % config.ep_size != 0:
        return f"num_experts={config.num_experts} not divisible by ep_size={config.ep_size}"

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
    recv_source_info: List[Tuple[int, int]],
    config: CommTestConfig,
    recv_hs_bf16: Optional[torch.Tensor] = None,
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
        "moe_output": moe_output.cpu(),
        "recv_source_info": recv_source_info,
    }
    if recv_hs_bf16 is not None:
        result["recv_hs_bf16"] = recv_hs_bf16.cpu()

    if config.quant_mode == "w4afp8":
        result["original_fp8"] = _safe_cpu(worker_inputs.original_fp8)
        result["w4afp8_roundtrip_ok"] = worker_inputs.w4afp8_roundtrip_ok

    return result


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

    comm = None
    try:
        mapping = Mapping(
            rank=rank,
            tp_size=config.ep_size,
            moe_ep_size=config.ep_size,
            world_size=config.ep_size,
        )

        comm = create_comm_object(config.comm_type, mapping, config)

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

        experts_per_rank = config.num_experts // config.ep_size
        # Run a minimal local-expert compute step before combine(). This keeps
        # the test aligned with the real MoE pipeline, where combine() consumes
        # per-rank expert outputs rather than raw dispatch payloads.
        moe_output = simple_moe(
            recv_hs_bf16,
            dispatch_outputs.recv_slots,
            dispatch_outputs.recv_scales,
            rank,
            experts_per_rank,
        )

        combined = comm.combine(
            moe_output,
            all_rank_max_num_tokens=max(config.all_num_tokens),
        )

        # Save recv_hs_bf16 for DeepEPLL combine reference (weighted reduce
        # needs raw dispatched hidden_states, not accumulated moe_output).
        saved_recv_hs_bf16 = recv_hs_bf16.clone() if config.comm_type == COMM_DEEP_EP_LL else None

        return _build_worker_result(
            rank,
            worker_inputs,
            dispatch_outputs,
            combined,
            moe_output,
            recv_source_info,
            config,
            recv_hs_bf16=saved_recv_hs_bf16,
        )
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
    selections falls in dest_rank's expert range.
    """
    experts_per_rank = config.num_experts // config.ep_size

    expected: Dict[int, Set[Tuple[int, int]]] = {r: set() for r in range(config.ep_size)}
    for result in all_results:
        src_rank = result["rank"]
        slots = result["original_slots"]

        for i in range(slots.shape[0]):
            for k in range(slots.shape[1]):
                eid = slots[i, k].item()
                if 0 <= eid < config.num_experts:
                    expected[eid // experts_per_rank].add((src_rank, i))

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
    decoded: List[Tuple[int, int]],
    token_idx: int,
    recv_rank: int,
    original_data: Dict[Tuple[int, int], torch.Tensor],
    skip_content_check: bool,
    actually_received: Set[Tuple[int, int]],
) -> None:
    """Record one received token and optionally verify row content."""
    src_rank, src_idx = decoded[token_idx]
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
    experts_per_rank = num_experts // config.ep_size

    # For fp8/w4afp8, data is transported as fp8 viewed as bf16 (half width).
    # For nvfp4, data is packed uint8 (half width).
    qm = config.quant_mode
    decode_dtype, decode_hidden = _get_dispatch_decode_spec(config)

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

        slot_start = recv_rank * experts_per_rank
        slot_end = slot_start + experts_per_rank

        decoded = decode_source_info(recv_hs, decode_dtype, decode_hidden)
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
    output_scale = torch.where(
        vec_max > 0,
        sf_scale_val / sf_narrow,
        torch.zeros_like(sf_narrow),
    )

    # Scale each element and quantize to E2M1
    scaled = x_grouped * output_scale.unsqueeze(-1)
    sign = scaled.sign()
    abs_scaled = scaled.abs()
    idx = torch.searchsorted(e2m1_bounds, abs_scaled).clamp(max=7)
    e2m1_quantized = e2m1_pos_vals[idx] * sign

    # Dequantize: result = e2m1_val * float(sf8) / SFScaleVal
    dequant_scale = sf_narrow / sf_scale_val
    result = e2m1_quantized * dequant_scale.unsqueeze(-1)

    return result.reshape(N, H).to(torch.bfloat16)


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
    experts_per_rank = config.num_experts // config.ep_size

    # Use float32 accumulator for all paths (matches kernel behavior)
    ref = torch.zeros(num_tokens, hidden_size, dtype=torch.float32)

    if config.use_low_precision_combine and config.comm_type == COMM_NVLINK_ONE_SIDED:
        # Path 1a: NVLinkOneSided fp8 quantize/dequantize + float32 accumulation.
        # moeA2APrepareCombineKernel does vectorized_quant bf16->fp8
        # (per-element cast, no scale). moeA2ACombineKernel loads fp8,
        # casts to float32, accumulates in float32, then casts to bf16.
        for proc_result in all_results:
            moe_out = proc_result["moe_output"]
            source_info = proc_result["recv_source_info"]
            for i, (src_rank, token_idx) in enumerate(source_info):
                if src_rank == target_rank and token_idx < num_tokens:
                    fp8_val = moe_out[i].to(torch.float8_e4m3fn)
                    ref[token_idx] += fp8_val.float()

    elif config.use_low_precision_combine and config.comm_type == COMM_NVLINK_TWO_SIDED:
        # Path 1b: NVLinkTwoSided NVFP4 simulation + float32 accumulation.
        # fusedMoeCommKernels.cu quantize_nvfp4_sharedmem uses two-level
        # scaling: per-row global fp32 scale + per-group-of-16 fp8 scale,
        # with E2M1 quantization. After NVLink transfer,
        # dequantize_nvfp4_sharedmem reverses the process. The top_k
        # reduction is then done in bf16 by torch.sum in _mnnvl_utils.py.
        for proc_result in all_results:
            moe_out = proc_result["moe_output"]
            nvfp4_out = _simulate_nvfp4_round_trip(moe_out)
            source_info = proc_result["recv_source_info"]
            for i, (src_rank, token_idx) in enumerate(source_info):
                if src_rank == target_rank and token_idx < num_tokens:
                    ref[token_idx] += nvfp4_out[i].float()

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
            slot_start = proc_rank * experts_per_rank
            slot_end = slot_start + experts_per_rank

            for i, (src_rank, token_idx) in enumerate(source_info):
                if src_rank == target_rank and token_idx < num_tokens:
                    for k in range(config.top_k):
                        eid = recv_slots[i, k].item()
                        if slot_start <= eid < slot_end:
                            weight = target_original_scales[token_idx, k].float()
                            ref[token_idx] += recv_hs_bf16[i].float() * weight

    else:
        # Path 3: Default — float32 accumulation.
        # The combine kernel (vectorized_combine_impl) uses float32
        # registers for cross-rank reduction, then casts to bf16.
        for proc_result in all_results:
            moe_out = proc_result["moe_output"]
            source_info = proc_result["recv_source_info"]
            for i, (src_rank, token_idx) in enumerate(source_info):
                if src_rank == target_rank and token_idx < num_tokens:
                    ref[token_idx] += moe_out[i].float()

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


def _make_workloads(ep_size: int) -> List[List[int]]:
    """Generate token distributions: uniform, non-uniform, minimal."""
    workloads = [[32] * ep_size]

    if ep_size == 2:
        workloads.append([16, 48])
    elif ep_size == 4:
        workloads.append([16, 32, 48, 64])

    workloads.append([1] * ep_size)
    return workloads


def _make_test_params():
    """Generate full-pipeline test parameters.

    Each entry is (ep_size, config).  ep_size is passed to mpi_pool_executor
    via indirect parametrization; config is passed directly to the test.
    """
    params = []
    for comm_type in ALL_COMM_TYPES:
        for ep_size in [2, 4]:
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
                        if use_low_precision_combine and not _supports_low_precision_combine(
                            config
                        ):
                            continue
                        params.append(pytest.param(ep_size, config, id=str(config)))
    return params


def _make_boundary_test_params():
    """Generate boundary / edge-case test parameters."""
    params = []
    for comm_type in ALL_COMM_TYPES:
        boundary_cases = [
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
            ),
            (
                2,
                CommTestConfig(
                    comm_type=comm_type,
                    ep_size=2,
                    num_experts=FIXED_NUM_EXPERTS,
                    top_k=2,
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
                    top_k=2,
                    hidden_size=DEFAULT_HIDDEN_SIZE,
                    all_num_tokens=[1, 1, 1, 1],
                ),
                f"{comm_type}_single_token",
            ),
        ]

        # Zero tokens on some ranks (DeepEPLL kernel does not support this).
        if comm_type != COMM_DEEP_EP_LL:
            boundary_cases.append(
                (
                    4,
                    CommTestConfig(
                        comm_type=comm_type,
                        ep_size=4,
                        num_experts=FIXED_NUM_EXPERTS,
                        top_k=2,
                        hidden_size=DEFAULT_HIDDEN_SIZE,
                        all_num_tokens=[32, 0, 16, 0],
                    ),
                    f"{comm_type}_zero_tokens",
                )
            )

        for ep_size, base_config, case_id in boundary_cases:
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
                if use_low_precision_combine and not _supports_low_precision_combine(config):
                    continue
                param_id = case_id
                if use_low_precision_combine:
                    param_id += "_lpcombine"
                params.append(pytest.param(ep_size, config, id=param_id))

    return params


def _make_postquant_test_params():
    """Generate post-quant test parameters using POSTQUANT_COMM_MAP.

    Uses simplified workloads (ep_size=2, top_k=2, small tokens) to keep
    the matrix manageable while covering all valid (comm_type, quant_mode)
    combinations.
    """
    params = []
    for quant_mode, comm_types in POSTQUANT_COMM_MAP.items():
        for comm_type in comm_types:
            for use_low_precision_combine in [False, True]:
                config = CommTestConfig(
                    comm_type=comm_type,
                    ep_size=2,
                    num_experts=FIXED_NUM_EXPERTS,
                    top_k=2,
                    hidden_size=DEFAULT_HIDDEN_SIZE,
                    all_num_tokens=[16, 16],
                    quant_mode=quant_mode,
                    use_low_precision_combine=use_low_precision_combine,
                )
                if use_low_precision_combine and not _supports_low_precision_combine(config):
                    continue
                params.append(pytest.param(2, config, id=str(config)))
    return params


# ============================================================================
# Pytest Fixtures & Test Runner
# ============================================================================


@pytest.fixture(autouse=True)
def setup_test():
    torch.manual_seed(0x1234)
    tllm.logger.set_level("error")


def _run_full_test(mpi_pool_executor, config: CommTestConfig):
    """Run dispatch -> verify dispatch -> simple_moe -> combine -> verify combine."""
    skip_reason = check_platform_support(config.comm_type)
    if skip_reason:
        pytest.skip(skip_reason)

    skip_reason = check_feasibility(config.comm_type, config)
    if skip_reason:
        pytest.skip(skip_reason)

    if config.ep_size > torch.cuda.device_count():
        pytest.skip(f"Need {config.ep_size} GPUs but only {torch.cuda.device_count()} available")

    results = mpi_pool_executor.map(
        _worker_full_pipeline,
        *zip(*[(config,)] * config.ep_size),
    )
    all_results = list(results)

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
        else:
            # NVFP4 simulation matches kernel's two-level scaling.
            # Residual error from rcp.approx.ftz.f32 vs exact division
            # causes ~0.35/element boundary effects at E2M1 midpoints
            # (fp8 scale rounding difference pushes elements across
            # quantization boundaries). Scales linearly with top_k.
            nvfp4_atol = 0.4 * config.top_k
            verify_combine_results(all_results, config, rtol=0.1, atol=nvfp4_atol)
    elif config.comm_type == COMM_DEEP_EP_LL:
        verify_combine_results(all_results, config, rtol=0.05, atol=0.3)
    else:
        verify_combine_results(all_results, config, rtol=0.02, atol=0.15)


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
        "mpi_pool_executor,config",
        _make_test_params(),
        indirect=["mpi_pool_executor"],
    )
    def test_moe_comm(self, mpi_pool_executor, config: CommTestConfig):
        """Verify full dispatch -> compute -> combine pipeline."""
        _run_full_test(mpi_pool_executor, config)

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize(
        "mpi_pool_executor,config",
        _make_boundary_test_params(),
        indirect=["mpi_pool_executor"],
    )
    def test_moe_comm_boundary(self, mpi_pool_executor, config: CommTestConfig):
        """Test full pipeline with boundary / edge-case parameters."""
        _run_full_test(mpi_pool_executor, config)

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize(
        "mpi_pool_executor,config",
        _make_postquant_test_params(),
        indirect=["mpi_pool_executor"],
    )
    def test_moe_comm_postquant(self, mpi_pool_executor, config: CommTestConfig):
        """Verify post-quant dispatch -> dequant -> MoE -> combine pipeline."""
        _run_full_test(mpi_pool_executor, config)
