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

import os
import pickle
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm as tllm
from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._torch.distributed import MoeAlltoAll
from tensorrt_llm._torch.distributed.moe_alltoall import (
    _CFT_MAX_BATCH_FOR_COMBINE_ENV, _CFT_MAX_BATCH_FOR_DISPATCH_ENV)
from tensorrt_llm._torch.modules.fused_moe.communication.nvlink_one_sided import (
    FORCE_CFT_ENV, QUANTIZE_ON_SEND_ENV, NVLinkOneSided, _dispatch_block_size,
    quantize_on_send_launchable)
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

_CFT_COUNTER_STRIDE_BYTES = 256
_CFT_COUNTER_STRIDE_U64 = _CFT_COUNTER_STRIDE_BYTES // 8
_NEEDS_8_GPUS = pytest.mark.skipif(torch.cuda.device_count() < 8,
                                   reason="needs at least 8 GPUs")

# TLLM_MOE_A2A_FUSED_ROUND=1 makes the op launch the fused-round kernels (the fence
# dispatch prepares its own round and leaves its counters zeroed); unset or 0 the
# legacy kernels. The C++ op reads it once per process; the pool workers
# inherit it from the test process, so one pytest run checks one mode.
_FUSED_ROUND_ENV = "TLLM_MOE_A2A_FUSED_ROUND"


def fused_round_kernels() -> bool:
    return os.environ.get(_FUSED_ROUND_ENV) == "1"


# The fused-round fence dispatch reads TLLM_MOE_A2A_DISPATCH_PRELOAD once per process
# (unset or a nonzero integer: payload preload; 0: the per-payload loop), and the
# module-scoped pool reuses its workers, so a pytest run checks one setting.
_DISPATCH_PRELOAD_ENV = "TLLM_MOE_A2A_DISPATCH_PRELOAD"


def dispatch_preload_enabled() -> bool:
    value = os.environ.get(_DISPATCH_PRELOAD_ENV)
    if value is None:
        return True
    digits = value.strip()
    sign = digits[:1] if digits[:1] in "+-" else ""
    digits = digits[len(sign):]
    digits = digits[:len(digits) - len(digits.lstrip("0123456789"))]
    return digits != "" and int(digits) != 0  # std::atoi(value) != 0


@pytest.fixture(autouse=True)
def setup_test():
    torch.manual_seed(0x1234)
    tllm.logger.set_level('error')


def compute_target_rank_id(expert_id, num_experts_per_rank):
    """Compute the rank that owns a given expert using contiguous partitioning.
    Experts are divided evenly across ranks:
      - Rank 0: experts [0, num_experts_per_rank)
      - Rank 1: experts [num_experts_per_rank, 2 * num_experts_per_rank)
      - ...
    For example, with 32 experts and 4 ranks (8 experts per rank):
      - Rank 0: experts 0-7
      - Rank 1: experts 8-15
      - Rank 2: experts 16-23
      - Rank 3: experts 24-31
    """
    return expert_id // num_experts_per_rank


def generate_token_selected_experts(local_num_tokens: int, num_experts: int,
                                    top_k: int) -> torch.Tensor:
    """Generate global expert IDs tensor, aligned with single-GPU test semantics."""
    return torch.randint(
        0,
        num_experts,
        (local_num_tokens, top_k),
        dtype=torch.int32,
        device='cuda',
    )


def create_experts_per_rank(num_experts_per_rank,
                            hidden_size,
                            ep_rank,
                            device,
                            dtype=torch.bfloat16):
    """
    Create a 3D tensor of expert weights for a given rank.

    Args:
        num_experts_per_rank: Number of experts on this rank
        hidden_size: Hidden dimension size
        ep_rank: EP rank ID
        device: Device to create experts on

    Returns:
        experts: Tensor of shape [num_experts_per_rank, hidden_size, hidden_size]
    """
    # For reproducibility, set the seed based on rank
    experts = torch.empty((num_experts_per_rank, hidden_size, hidden_size),
                          dtype=dtype,
                          device=device)
    for i in range(num_experts_per_rank):
        torch.manual_seed(ep_rank * 1000 + i)
        # Xavier uniform initialization for each expert
        torch.nn.init.xavier_uniform_(experts[i])
    return experts


def fake_moe(hidden_states,
             token_selected_experts,
             token_final_scales,
             experts,
             is_ep=False,
             ep_rank=None,
             num_experts_per_rank=None):
    """
    Emulate MoE computation by scaling tokens based on which experts belong to this rank.

    Args:
        hidden_states: [num_tokens, hidden_size] - input hidden states
        token_selected_experts: [num_tokens, top_k] - selected expert indices
        token_final_scales: [num_tokens, top_k] - scaling factors for each expert
        experts: [num_experts_per_rank, hidden_size, hidden_size] if is_ep, otherwise [num_experts, hidden_size, hidden_size] - expert weights
        is_ep: If true, emulate MoE on a EP rank; otherwise, emulate MoE with all experts
        ep_rank: EP rank ID
        num_experts_per_rank: Number of experts per rank

    Returns:
        processed_states: [num_tokens, hidden_size] - processed hidden states
    """
    num_tokens, _ = hidden_states.shape
    _, top_k = token_selected_experts.shape

    if is_ep:
        assert ep_rank is not None and num_experts_per_rank is not None

    # Initialize output
    processed_states = torch.zeros_like(hidden_states)

    # Process each token
    for token_idx in range(num_tokens):
        # For each expert selected for this token/
        for k in range(top_k):
            expert_id = token_selected_experts[token_idx, k].item()
            if is_ep:
                if not (expert_id >= ep_rank * num_experts_per_rank
                        and expert_id < (ep_rank + 1) * num_experts_per_rank):
                    continue
                # Convert global expert ID to local expert ID for this rank
                local_expert_id = expert_id - ep_rank * num_experts_per_rank
                expert = experts[local_expert_id]
            else:
                expert = experts[expert_id]

            scale = token_final_scales[token_idx, k]
            processed_states[
                token_idx] += hidden_states[token_idx] @ expert * scale

    return processed_states


def make_nvfp4_payloads(
        local_num_tokens: int, hidden_size: int, top_k: int, rank: int,
        token_selected_experts: torch.Tensor) -> tuple[list, int]:
    """Create the four NV FP4 payloads exactly as in single-GPU test."""
    payloads = []
    # Payload 0: Packed FP4 tokens (uint8)
    packed_hidden_size = hidden_size // 2
    packed_hidden_states = torch.randint(0,
                                         256,
                                         (local_num_tokens, packed_hidden_size),
                                         dtype=torch.uint8,
                                         device='cuda')
    payloads.append(packed_hidden_states)

    # Payload 1: Scaling factors (fp8)
    num_elts_per_sf = 16
    num_scaling_factors = hidden_size // num_elts_per_sf
    scaling_factors = torch.randn(
        local_num_tokens,
        num_scaling_factors,
        dtype=torch.float32,
        device='cuda')  #  .to(torch.float8_e4m3fn) TODO: Test failed.
    scaling_factors += rank
    payloads.append(scaling_factors)

    # Payload 2: token_selected_experts
    payloads.append(token_selected_experts)

    # Payload 3: token_final_scales (bfloat16)
    token_final_scales = torch.rand(local_num_tokens,
                                    top_k,
                                    dtype=torch.bfloat16,
                                    device='cuda')

    # Construct the data to contain info about send rank and local_token_idx, which is used for debugging
    # token_final_scales[:, 0] = rank
    # token_final_scales[:, 1] = torch.linspace(0, local_num_tokens - 1, local_num_tokens, dtype=torch.bfloat16, device='cuda')

    payloads.append(token_final_scales)
    return payloads, 2


def make_floating_point_payloads(
        local_num_tokens: int, hidden_size: int, top_k: int, rank: int,
        token_selected_experts: torch.Tensor,
        payload_dtype: torch.dtype) -> tuple[list, int]:
    """Create floating-point test payloads matching NVFP4 structure without scaling factors."""
    payloads = []

    # Payload 0: hidden states
    hidden_states = torch.randn(local_num_tokens,
                                hidden_size,
                                dtype=payload_dtype,
                                device='cuda')
    # Add rank-specific pattern for verification
    hidden_states += rank
    payloads.append(hidden_states)

    # Payload 1: token_selected_experts
    payloads.append(token_selected_experts)

    # Payload 2: token_final_scales
    token_final_scales = torch.rand(local_num_tokens,
                                    top_k,
                                    dtype=payload_dtype,
                                    device='cuda')

    # Optional: Construct the data that is easier to debug
    # token_final_scales[:, 0] = rank
    # token_final_scales[:, 1] = torch.linspace(0, local_num_tokens - 1, local_num_tokens, dtype=torch.bfloat16, device='cuda')

    payloads.append(token_final_scales)

    return payloads, 1


def read_workspace_int32(moe_a2a, rank: int, offset_name: str,
                         count: int) -> torch.Tensor:
    """Read the first `count` int32 values of a workspace field (CPU)."""
    start = moe_a2a.metainfo[MoeAlltoAll._METAINFO_INDEX[offset_name]].item()
    return moe_a2a.workspace[rank,
                             start:start + count * 4].view(torch.int32).cpu()


def check_dispatch_counters_reset(moe_a2a, rank: int, ep_size: int):
    """With the fused-round kernels both dispatch kernels leave send_counters and
    local_token_counter zeroed for the next round (the fence dispatch has no prepare
    kernel). The legacy kernels leave that to the next prepare kernel."""
    if not fused_round_kernels():
        return
    send_counters = read_workspace_int32(moe_a2a, rank,
                                         "SEND_COUNTERS_OFFSET_INDEX", ep_size)
    local_token_counter = read_workspace_int32(
        moe_a2a, rank, "LOCAL_TOKEN_COUNTER_OFFSET_INDEX", 1)
    assert torch.all(send_counters == 0), (
        f"Rank {rank} send_counters not reset after dispatch: {send_counters}")
    assert local_token_counter.item() == 0, (
        f"Rank {rank} local_token_counter not reset after dispatch: "
        f"{local_token_counter.item()}")


def read_cft_dispatch_counters(moe_a2a: MoeAlltoAll, rank: int,
                               ep_size: int) -> torch.Tensor:
    counter_offset = moe_a2a.metainfo[MoeAlltoAll._METAINFO_INDEX[
        "DISPATCH_COUNTED_WRITE_COUNTERS_OFFSET_INDEX"]].item()
    counters = moe_a2a.workspace[rank, counter_offset:counter_offset +
                                 ep_size * _CFT_COUNTER_STRIDE_BYTES].view(
                                     torch.int64)
    return counters[::_CFT_COUNTER_STRIDE_U64][:ep_size].cpu()


def run_moe_a2a_dispatch_single_rank(ep_size,
                                     all_num_tokens,
                                     top_k,
                                     workspace_size_per_rank,
                                     num_experts,
                                     hidden_size,
                                     invalid_token_expert_id,
                                     enable_eplb,
                                     can_use_cft_counted_writes=False):
    """Worker function for MPIPoolExecutor."""
    rank = tllm.mpi_rank()
    torch.cuda.set_device(rank)

    try:
        mapping = Mapping(
            rank=rank,
            tp_size=ep_size,
            moe_ep_size=ep_size,
            world_size=ep_size,
        )

        # Create MoeAlltoAll manager
        max_num_tokens = max(all_num_tokens)

        eplb_stats_num_experts = (
            num_experts // 2 if enable_eplb else None
        )  # Use half of the experts for testing EPLB stats
        moe_a2a = MoeAlltoAll(
            mapping=mapping,
            max_num_tokens=max_num_tokens,
            top_k=top_k,
            num_slots=num_experts,
            workspace_size_per_rank=workspace_size_per_rank,
            num_experts=eplb_stats_num_experts,
            can_use_cft_counted_writes=can_use_cft_counted_writes,
        )

        # Get the number of tokens for this specific rank (same as single-GPU)
        rank_local_tokens = all_num_tokens[rank]

        # Generate data using helper functions
        token_selected_experts = generate_token_selected_experts(
            rank_local_tokens, num_experts, top_k)
        payloads, expert_id_payload_index = make_nvfp4_payloads(
            rank_local_tokens, hidden_size, top_k, rank, token_selected_experts)
        payload_bytes_per_token = [
            payload.shape[1] * payload.element_size() for payload in payloads
        ]
        actual_cft_dispatch = (
            can_use_cft_counted_writes
            and moe_a2a.use_cft_for_dispatch(max_num_tokens)
            and all(bytes_per_token % 16 == 0
                    for bytes_per_token in payload_bytes_per_token))

        eplb_local_stats = None
        if enable_eplb:
            eplb_local_stats = (torch.arange(
                eplb_stats_num_experts, dtype=torch.int32, device="cuda") +
                                rank * 1000)

        # CFT counted writes require LE initialization before first dispatch
        if can_use_cft_counted_writes:
            moe_a2a.cft_initialize()
        cft_counters_before = (read_cft_dispatch_counters(
            moe_a2a, rank, ep_size) if actual_cft_dispatch else None)
        if actual_cft_dispatch:
            # All ranks must snapshot their baseline before any peer can issue counted writes.
            tllm.mpi_barrier()

        recv_tensors = moe_a2a.dispatch(
            token_selected_experts,
            payloads,
            max_num_tokens,
            invalid_token_expert_id=invalid_token_expert_id,
            expert_id_payload_index=expert_id_payload_index,
            eplb_local_stats=eplb_local_stats)

        # Verify completion flags after dispatch unless the counted-write path actually ran.
        if not actual_cft_dispatch:
            completion_flags_offset = moe_a2a.metainfo[
                MoeAlltoAll._METAINFO_INDEX[
                    "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX"]].item()
            completion_flags = moe_a2a.workspace[
                rank, completion_flags_offset:completion_flags_offset +
                ep_size * 4].view(torch.int32).cpu()
            flag_val_offset = moe_a2a.metainfo[
                MoeAlltoAll._METAINFO_INDEX["FLAG_VAL_OFFSET_INDEX"]].item()
            expected_flag_val = moe_a2a.workspace[
                rank,
                flag_val_offset:flag_val_offset + 4].view(torch.int32).cpu()

            assert torch.all(completion_flags == expected_flag_val), (
                f"Rank {rank} completion flags: {completion_flags}, expected flag val: {expected_flag_val}"
            )

        check_dispatch_counters_reset(moe_a2a, rank, ep_size)

        # Read counters and compact routing tensors from workspace
        recv_counters_offset = moe_a2a.metainfo[
            MoeAlltoAll._METAINFO_INDEX["RECV_COUNTERS_OFFSET_INDEX"]].item()
        topk_target_ranks_offset = moe_a2a.metainfo[MoeAlltoAll._METAINFO_INDEX[
            "TOPK_TARGET_RANKS_OFFSET_INDEX"]].item()
        topk_send_indices_offset = moe_a2a.metainfo[MoeAlltoAll._METAINFO_INDEX[
            "TOPK_SEND_INDICES_OFFSET_INDEX"]].item()

        flag_val_offset = moe_a2a.metainfo[
            MoeAlltoAll._METAINFO_INDEX["FLAG_VAL_OFFSET_INDEX"]].item()
        dispatch_flag = moe_a2a.workspace[rank,
                                          flag_val_offset:flag_val_offset +
                                          4].view(torch.int32).item()
        recv_counters_parity = (dispatch_flag >> 1) & 1
        recv_counters_offset += recv_counters_parity * ep_size * 4
        recv_counters = moe_a2a.workspace[
            rank, recv_counters_offset:recv_counters_offset + ep_size * 4].view(
                torch.int32).cpu()
        topk_target_ranks = moe_a2a.workspace[
            rank, topk_target_ranks_offset:topk_target_ranks_offset +
            max_num_tokens * top_k * 4].view(torch.int32).view(
                max_num_tokens, top_k).cpu()
        topk_send_indices = moe_a2a.workspace[
            rank, topk_send_indices_offset:topk_send_indices_offset +
            max_num_tokens * top_k * 4].view(torch.int32).view(
                max_num_tokens, top_k).cpu()
        assert torch.all(recv_counters >= 0), (
            f"Rank {rank} recv_counters contain pending sentinel values: "
            f"{recv_counters}")
        if actual_cft_dispatch:
            cft_counters_after = read_cft_dispatch_counters(
                moe_a2a, rank, ep_size)
            cft_counter_delta = cft_counters_after - cft_counters_before
            expected_payload_bytes_per_token = sum(payload_bytes_per_token)
            for peer_rank in range(ep_size):
                if peer_rank == rank:
                    continue
                expected_bytes = (recv_counters[peer_rank].item() *
                                  expected_payload_bytes_per_token)
                actual_bytes = cft_counter_delta[peer_rank].item()
                assert actual_bytes >= expected_bytes, (
                    f"Rank {rank} CFT dispatch counter from rank {peer_rank}: "
                    f"delta={actual_bytes}, expected at least {expected_bytes}")

        # Return results to be collected (move to CPU for MPI transfer)
        eplb_gathered_stats = moe_a2a._state.eplb_gathered_stats
        if eplb_gathered_stats is not None:
            eplb_gathered_stats = eplb_gathered_stats.cpu()
        if eplb_local_stats is not None:
            eplb_local_stats = eplb_local_stats.cpu()

        return (token_selected_experts.cpu(), [p.cpu() for p in payloads],
                [rt.cpu() for rt in recv_tensors
                 ], topk_send_indices, topk_target_ranks, recv_counters,
                expert_id_payload_index, eplb_local_stats, eplb_gathered_stats)
    except Exception:
        traceback.print_exc()
        raise


def verify_dispatch(all_token_selected_experts, all_payloads, all_recv_tensors,
                    all_topk_send_indices, all_topk_target_ranks,
                    all_recv_counters, ep_size, all_num_tokens, top_k,
                    num_experts, expert_id_payload_index,
                    invalid_token_expert_id):
    """Verify dispatch results including actual content verification"""

    max_num_tokens = max(all_num_tokens)
    num_experts_per_rank = num_experts // ep_size
    # Verify dimensions and dtypes
    for send_rank in range(ep_size):
        local_num_tokens = all_num_tokens[send_rank]

        token_selected_experts = all_token_selected_experts[send_rank]
        assert len(token_selected_experts.shape
                   ) == 2, "token_selected_experts should be a 2D tensor"
        assert token_selected_experts.dtype == torch.int32, "token_selected_experts should be a 32-bit integer tensor"
        assert token_selected_experts.shape[
            0] == local_num_tokens, "token_selected_experts.shape[0] should be local_num_tokens"
        assert token_selected_experts.shape[
            1] == top_k, "token_selected_experts.shape[1] should be top_k"

        payloads = all_payloads[send_rank]
        recv_tensors = all_recv_tensors[send_rank]
        num_payloads = len(payloads)
        assert len(
            recv_tensors
        ) == num_payloads, "recv_tensors should have the same number of payloads as payloads"
        for i in range(num_payloads):
            payload = payloads[i]
            assert len(payload.shape) == 2, "payload should be a 2D tensor"
            assert payload.shape[
                0] == local_num_tokens, "payload.shape[0] should be local_num_tokens"

            recv_tensor = recv_tensors[i]
            assert len(
                recv_tensor.shape) == 3, "recv_tensor should be a 3D tensor"
            assert recv_tensor.shape[
                0] == ep_size, "recv_tensor.shape[0] should be ep_size"
            assert recv_tensor.shape[
                1] == max_num_tokens, "recv_tensor.shape[1] should be max_num_tokens"
            assert recv_tensor.shape[2] == payload.shape[
                1], "recv_tensor.shape[2] should be payload.shape[1]"
            assert recv_tensor.dtype == payload.dtype, "recv_tensor.dtype should be payload.dtype"

        # Verify counters and compact routing tensors
        recv_counters = all_recv_counters[send_rank]
        assert len(
            recv_counters.shape) == 1, "recv_counters should be a 1D tensor"
        assert recv_counters.shape[0] == ep_size
        assert recv_counters.dtype == torch.int32

        topk_send_indices = all_topk_send_indices[send_rank]
        topk_target_ranks = all_topk_target_ranks[send_rank]
        assert topk_send_indices.shape == (max_num_tokens,
                                           top_k), "topk_send_indices shape"
        assert topk_target_ranks.shape == (max_num_tokens,
                                           top_k), "topk_target_ranks shape"
        assert topk_send_indices.dtype == torch.int32
        assert topk_target_ranks.dtype == torch.int32

    # Verify the token count of every (send_rank -> target_rank) pair as the target
    # rank received it
    for send_rank in range(ep_size):
        expected_sends = {}
        token_experts = all_token_selected_experts[send_rank]
        sent_to_rank = set()

        for token_idx in range(token_experts.shape[0]):
            experts = token_experts[token_idx]
            target_ranks = compute_target_rank_id(experts, num_experts_per_rank)
            sent_to_rank.clear()

            for target_rank in target_ranks.tolist():
                if target_rank not in sent_to_rank:
                    if target_rank not in expected_sends:
                        expected_sends[target_rank] = 0
                    expected_sends[target_rank] += 1
                    sent_to_rank.add(target_rank)

        for target_rank in range(ep_size):
            expected_to_rank = expected_sends.get(target_rank, 0)
            actual_to_rank = all_recv_counters[target_rank][send_rank].item()
            assert actual_to_rank == expected_to_rank, (
                f"Rank {target_rank} received {actual_to_rank} tokens from rank {send_rank}, expected {expected_to_rank}"
            )

    # Verify payload content using topk_send_indices and topk_target_ranks
    for send_rank in range(ep_size):
        token_selected_experts = all_token_selected_experts[send_rank]
        payloads = all_payloads[send_rank]
        topk_send_indices = all_topk_send_indices[send_rank]
        topk_target_ranks = all_topk_target_ranks[send_rank]
        local_num_tokens = all_num_tokens[send_rank]

        for token_idx in range(local_num_tokens):
            experts = token_selected_experts[token_idx]
            target_ranks = compute_target_rank_id(experts, num_experts_per_rank)
            # Deduplicate target ranks per token
            topk_target_ranks_ref = target_ranks.clone()
            seen = set()
            for kk in range(top_k):
                tr = int(topk_target_ranks_ref[kk].item())
                if tr in seen:
                    topk_target_ranks_ref[kk] = -1
                else:
                    seen.add(tr)

            assert topk_target_ranks[
                token_idx, :].tolist() == topk_target_ranks_ref.tolist()

            for k in range(top_k):
                dst_pos = topk_send_indices[token_idx, k].item()
                target_rank = topk_target_ranks[token_idx, k].item()
                if dst_pos == -1:
                    assert target_rank == -1
                    continue
                recv_tensors = all_recv_tensors[target_rank]
                for payload_idx, payload in enumerate(payloads):
                    recv_tensor = recv_tensors[payload_idx]
                    source_data = payload[token_idx]
                    received_data = recv_tensor[send_rank, dst_pos]
                    torch.testing.assert_close(received_data,
                                               source_data,
                                               atol=0,
                                               rtol=0)

    # Verify token_selected_experts of invalid tokens are correctly sanitized
    for recv_rank in range(ep_size):
        expert_ids_recv = all_recv_tensors[recv_rank][expert_id_payload_index]
        for source_rank in range(ep_size):
            valid = int(all_recv_counters[recv_rank][source_rank].item())
            for token_idx in range(max_num_tokens):
                token_expert_ids = expert_ids_recv[source_rank, token_idx]
                if token_idx >= valid:
                    assert torch.all(
                        token_expert_ids == invalid_token_expert_id)


def make_quantizable_hidden(local_num_tokens: int, hidden_size: int, rank: int,
                            seed: int) -> torch.Tensor:
    """BF16 hidden states whose NVFP4 quantization exercises the whole range of the scales: rows and 16-element
    blocks spanning 2^-16..2^20, all-zero blocks and rows, rows that saturate the E4M3 scale and the E2M1 codes,
    and rows of tiny values. Every value is finite."""
    generator = torch.Generator(device='cuda')
    generator.manual_seed(1000 * rank + seed)
    blocks = hidden_size // 16
    x = torch.randn(local_num_tokens,
                    blocks,
                    16,
                    generator=generator,
                    device='cuda')
    row_exp = torch.randint(-10,
                            13, (local_num_tokens, 1, 1),
                            generator=generator,
                            device='cuda')
    block_exp = torch.randint(-6,
                              8, (local_num_tokens, blocks, 1),
                              generator=generator,
                              device='cuda')
    x = x * torch.exp2((row_exp + block_exp).float())
    x[:, ::7, :] = 0
    if local_num_tokens > 0:
        x[0] = 0
    if local_num_tokens > 1:
        x[1] = x[1].sign() * 1e30  # saturates the scale (448) and the codes (6)
    if local_num_tokens > 2:
        x[2] = x[2].sign() * 2.0**-100
    return x.view(local_num_tokens, hidden_size).to(torch.bfloat16)


def read_recv_counters(moe_a2a, rank: int, ep_size: int) -> torch.Tensor:
    """The per-source token counts of the round that the last dispatch started (CPU)."""
    recv_counters_offset = moe_a2a.metainfo[
        MoeAlltoAll._METAINFO_INDEX["RECV_COUNTERS_OFFSET_INDEX"]].item()
    flag_val_offset = moe_a2a.metainfo[
        MoeAlltoAll._METAINFO_INDEX["FLAG_VAL_OFFSET_INDEX"]].item()
    dispatch_flag = moe_a2a.workspace[rank, flag_val_offset:flag_val_offset +
                                      4].view(torch.int32).item()
    recv_counters_offset += ((dispatch_flag >> 1) & 1) * ep_size * 4
    return moe_a2a.workspace[rank, recv_counters_offset:recv_counters_offset +
                             ep_size * 4].view(torch.int32).cpu()


def run_moe_a2a_quantize_on_send_single_rank(ep_size, rounds_num_tokens, top_k,
                                             workspace_size_per_rank,
                                             num_experts, hidden_size,
                                             global_scale, abort_rounds):
    """Worker function for MPIPoolExecutor: dispatch rounds that quantize on send.

    Every round dispatches BF16 hidden states with quantize_global_scale and returns what verify_dispatch needs,
    with the NVFP4 quantization of the input by fp4_quantize_kf and by fp4_quantize as the payloads that must
    arrive. The round ends with a combine, or with MoeAlltoAll.reset_state() for the rounds in abort_rounds.
    """
    rank = tllm.mpi_rank()
    torch.cuda.set_device(rank)

    try:
        mapping = Mapping(rank=rank,
                          tp_size=ep_size,
                          moe_ep_size=ep_size,
                          world_size=ep_size)
        max_num_tokens = max(max(n) for n in rounds_num_tokens)
        moe_a2a = MoeAlltoAll(
            mapping=mapping,
            max_num_tokens=max_num_tokens,
            top_k=top_k,
            num_slots=num_experts,
            workspace_size_per_rank=workspace_size_per_rank,
        )
        scale = torch.tensor([global_scale], dtype=torch.float32, device='cuda')
        rounds = []
        for round_idx, all_num_tokens in enumerate(rounds_num_tokens):
            num_tokens = all_num_tokens[rank]
            runtime_max_tokens = max(all_num_tokens)
            token_selected_experts = generate_token_selected_experts(
                num_tokens, num_experts, top_k)
            token_final_scales = torch.rand(num_tokens,
                                            top_k,
                                            dtype=torch.bfloat16,
                                            device='cuda')
            x = make_quantizable_hidden(num_tokens, hidden_size, rank,
                                        round_idx)

            # cloudpickle serializes this module by value, and a function that names torch.ops itself cannot be
            # pickled: the ops are called from a nested function, as in the combine worker below.
            def quantize_reference(x, scale):
                kf_codes, kf_sf = torch.ops.trtllm.fp4_quantize_kf(x, scale)
                prod_codes, prod_sf = torch.ops.trtllm.fp4_quantize(
                    x, scale, 16, False, False)
                return kf_codes, kf_sf, prod_codes, prod_sf

            if num_tokens > 0:
                codes, sf, prod_codes, prod_sf = quantize_reference(x, scale)
                sf = sf.view(num_tokens, hidden_size // 16)
                # The production quantization kernel, which the kernel above equals on finite inputs.
                assert torch.equal(prod_codes, codes)
                assert torch.equal(prod_sf.view(num_tokens, -1), sf)
            else:  # a rank without tokens sends nothing
                codes = torch.empty((0, hidden_size // 2),
                                    dtype=torch.uint8,
                                    device='cuda')
                sf = torch.empty((0, hidden_size // 16),
                                 dtype=torch.uint8,
                                 device='cuda')

            recv_tensors = moe_a2a.dispatch(
                token_selected_experts,
                [x, token_selected_experts, token_final_scales],
                runtime_max_tokens,
                invalid_token_expert_id=-1,
                expert_id_payload_index=1,
                quantize_global_scale=scale)
            assert len(recv_tensors) == 4
            assert [t.dtype for t in recv_tensors] == [
                torch.uint8, torch.uint8, torch.int32, torch.bfloat16
            ]
            recv_cpu = [t.cpu() for t in recv_tensors]
            topk_target_ranks, topk_send_indices = read_dispatch_routing(
                moe_a2a, rank, num_tokens, top_k)
            # verify_dispatch indexes the routing tensors by max_num_tokens rows.
            pad = runtime_max_tokens - num_tokens
            topk_target_ranks = torch.nn.functional.pad(topk_target_ranks,
                                                        (0, 0, 0, pad),
                                                        value=-1)
            topk_send_indices = torch.nn.functional.pad(topk_send_indices,
                                                        (0, 0, 0, pad),
                                                        value=-1)
            recv_counters = read_recv_counters(moe_a2a, rank, ep_size)
            if round_idx in abort_rounds:
                moe_a2a.reset_state()
            else:
                moe_a2a.combine(
                    torch.zeros((ep_size, runtime_max_tokens, 64),
                                dtype=torch.bfloat16,
                                device='cuda'), runtime_max_tokens)
            rounds.append((token_selected_experts.cpu(), [
                codes.cpu(),
                sf.cpu(),
                token_selected_experts.cpu(),
                token_final_scales.cpu()
            ], recv_cpu, topk_send_indices, topk_target_ranks, recv_counters))
        return rounds
    except Exception:
        traceback.print_exc()
        raise


def run_nvlink_one_sided_quantize_on_send_ab_single_rank(
        ep_size, rounds_num_tokens, top_k, num_experts, hidden_size,
        global_scale, with_final_scales):
    """Worker function for MPIPoolExecutor: NVLinkOneSided.dispatch with and without quantize on send.

    Every round dispatches the same BF16 hidden states twice, each dispatch followed by its combine: as a MoE layer does
    without quantize on send (fp4_quantize in front of the dispatch, the codes and the scales sent as payloads), and
    with quantize_on_send_scale (the BF16 hidden states sent as payload 0, the dispatch kernel quantizing them). Both
    return what verify_dispatch needs, with the NVFP4 quantization of the input as the payloads that must arrive, so the
    mapping of the returned tensors is checked on the way. The gate that decides which of the two a layer runs is asked
    with the switch on and off.
    """
    rank = tllm.mpi_rank()
    torch.cuda.set_device(rank)

    comm = None
    try:
        mapping = Mapping(rank=rank,
                          tp_size=ep_size,
                          moe_ep_size=ep_size,
                          world_size=ep_size)
        max_num_tokens = max(max(n) for n in rounds_num_tokens)
        comm = NVLinkOneSided(mapping=mapping,
                              num_slots=num_experts,
                              top_k=top_k,
                              max_num_tokens_per_rank=max_num_tokens,
                              hidden_size=hidden_size,
                              dtype=torch.bfloat16,
                              can_use_cft_counted_writes=False)
        # read_dispatch_routing and read_recv_counters take what MoeAlltoAll calls its workspace and metainfo, and index
        # the metainfo by MoeAlltoAll's constants, which only a MoeAlltoAll fills in.
        MoeAlltoAll._init_constants()
        a2a = SimpleNamespace(workspace=comm.workspace,
                              metainfo=comm.moe_a2a_metainfo)
        scale = torch.tensor([global_scale], dtype=torch.float32, device='cuda')
        recv_dtypes = [torch.uint8, torch.uint8, torch.int32]
        if with_final_scales:
            recv_dtypes.append(torch.float32)

        # cloudpickle serializes this module by value, and a function that names torch.ops itself cannot be
        # pickled: the op is called from a nested function.
        def quantize_reference(x, scale):
            return torch.ops.trtllm.fp4_quantize(x, scale, 16, False, False)

        rounds = []
        for round_idx, all_num_tokens in enumerate(rounds_num_tokens):
            num_tokens = all_num_tokens[rank]
            runtime_max_tokens = max(all_num_tokens)
            slots = generate_token_selected_experts(num_tokens, num_experts,
                                                    top_k)
            final_scales = None
            if with_final_scales:
                final_scales = torch.rand(num_tokens,
                                          top_k,
                                          dtype=torch.float32,
                                          device='cuda')
            x = make_quantizable_hidden(num_tokens, hidden_size, rank,
                                        round_idx)
            x_input = x.clone()
            if num_tokens > 0:
                codes, sf = quantize_reference(x, scale)
                sf = sf.view(num_tokens, hidden_size // 16)
            else:  # a rank without tokens sends nothing
                codes = torch.empty((0, hidden_size // 2),
                                    dtype=torch.uint8,
                                    device='cuda')
                sf = torch.empty((0, hidden_size // 16),
                                 dtype=torch.uint8,
                                 device='cuda')
            expected_payloads = [codes.cpu(), sf.cpu(), slots.cpu()]
            if with_final_scales:
                expected_payloads.append(final_scales.cpu())

            arms = []
            for quantize_on_send in (False, True):
                if quantize_on_send:
                    with patch.dict(os.environ, {QUANTIZE_ON_SEND_ENV: "1"}):
                        assert comm.can_quantize_on_send(
                            x, scale, slots, final_scales, all_num_tokens)
                    returned = comm.dispatch(x,
                                             None,
                                             slots,
                                             final_scales,
                                             all_num_tokens,
                                             quantize_on_send_scale=scale)
                    assert torch.equal(x, x_input)  # the input is not touched
                else:
                    returned = comm.dispatch(codes, sf, slots, final_scales,
                                             all_num_tokens)
                # (hidden states, their scales, expert ids, final scales), the sources flattened into the rows.
                assert len(returned) == 4
                assert (returned[3] is not None) == with_final_scales
                received = [t for t in returned if t is not None]
                assert [t.dtype for t in received] == recv_dtypes
                recv_cpu = [
                    t.view(ep_size, runtime_max_tokens, -1).cpu()
                    for t in received
                ]
                topk_target_ranks, topk_send_indices = read_dispatch_routing(
                    a2a, rank, num_tokens, top_k)
                # verify_dispatch indexes the routing tensors by max_num_tokens rows.
                pad = runtime_max_tokens - num_tokens
                topk_target_ranks = torch.nn.functional.pad(topk_target_ranks,
                                                            (0, 0, 0, pad),
                                                            value=-1)
                topk_send_indices = torch.nn.functional.pad(topk_send_indices,
                                                            (0, 0, 0, pad),
                                                            value=-1)
                recv_counters = read_recv_counters(a2a, rank, ep_size)
                comm.combine(
                    torch.zeros((ep_size, runtime_max_tokens, 64),
                                dtype=torch.bfloat16,
                                device='cuda'))
                arms.append(
                    (slots.cpu(), expected_payloads, recv_cpu,
                     topk_send_indices, topk_target_ranks, recv_counters))
            rounds.append(arms)

        # The gate behind the scheduler's choice, with the switch unset (the default: off), off and on.
        x = make_quantizable_hidden(8, hidden_size, rank, 99)
        slots = generate_token_selected_experts(8, num_experts, top_k)
        final_scales = torch.rand(8, top_k, dtype=torch.float32, device='cuda')
        gate = []
        for switch in (None, "0", "1"):
            with patch.dict(os.environ):
                if switch is None:
                    os.environ.pop(QUANTIZE_ON_SEND_ENV, None)
                else:
                    os.environ[QUANTIZE_ON_SEND_ENV] = switch
                gate.append(
                    comm.can_quantize_on_send(x, scale, slots, final_scales,
                                              [8] * ep_size))
        return rounds, gate
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if comm is not None:
            comm.destroy()


def run_moe_a2a_quantize_on_send_boundaries_single_rank(ep_size, shapes,
                                                        num_tokens, top_k,
                                                        workspace_size_per_rank,
                                                        num_experts,
                                                        global_scale):
    """Worker function for MPIPoolExecutor: dispatches that quantize on send, one per (hidden size, extra bytes).

    A shape whose launch checks reject the dispatch gives ("rejected", the error); one that passes gives
    ("accepted", what verify_dispatch needs). The payloads are the BF16 hidden state, int32 expert ids and bfloat16
    final scales; the extra bytes of a shape widen the final scales (the op sends at most four payloads, and the
    quantized hidden state counts as two, so a fourth input payload would be rejected for its number and not for its
    bytes). Whatever happens on one shape, the rank goes on to the next one (a rank that left early would leave its
    peers waiting in the next dispatch).
    """
    rank = tllm.mpi_rank()
    torch.cuda.set_device(rank)

    try:
        mapping = Mapping(rank=rank,
                          tp_size=ep_size,
                          moe_ep_size=ep_size,
                          world_size=ep_size)
        moe_a2a = MoeAlltoAll(
            mapping=mapping,
            max_num_tokens=num_tokens,
            top_k=top_k,
            num_slots=num_experts,
            workspace_size_per_rank=workspace_size_per_rank,
        )
        scale = torch.tensor([global_scale], dtype=torch.float32, device='cuda')

        def quantize_reference(x, scale):
            return torch.ops.trtllm.fp4_quantize(x, scale, 16, False, False)

        results = []
        for shape_idx, (hidden_size, extra_bytes) in enumerate(shapes):
            tag = f"[quantize-on-send boundaries rank {rank}] shape {shape_idx} (hidden {hidden_size}, {extra_bytes} extra bytes)"
            slots = generate_token_selected_experts(num_tokens, num_experts,
                                                    top_k)
            assert extra_bytes % 2 == 0
            final_scales = torch.rand(num_tokens,
                                      top_k + extra_bytes // 2,
                                      dtype=torch.bfloat16,
                                      device='cuda')
            x = make_quantizable_hidden(num_tokens, hidden_size, rank,
                                        shape_idx)
            print(f"{tag}: dispatch", flush=True)
            try:
                recv_tensors = moe_a2a.dispatch(slots, [x, slots, final_scales],
                                                num_tokens,
                                                invalid_token_expert_id=-1,
                                                expert_id_payload_index=1,
                                                quantize_global_scale=scale)
            except RuntimeError as error:
                # The checks run before anything is launched: nothing has changed, and the next shape dispatches.
                print(f"{tag}: rejected ({str(error)[:100]})", flush=True)
                results.append(("rejected", str(error), None))
                continue
            try:
                codes, sf = quantize_reference(x, scale)
                sf = sf.view(num_tokens, hidden_size // 16)
                recv_cpu = [t.cpu() for t in recv_tensors]
                topk_target_ranks, topk_send_indices = read_dispatch_routing(
                    moe_a2a, rank, num_tokens, top_k)
                recv_counters = read_recv_counters(moe_a2a, rank, ep_size)
                moe_a2a.combine(
                    torch.zeros((ep_size, num_tokens, 64),
                                dtype=torch.bfloat16,
                                device='cuda'), num_tokens)
                results.append(("accepted", "", (slots.cpu(), [
                    codes.cpu(),
                    sf.cpu(),
                    slots.cpu(),
                    final_scales.cpu(),
                ], recv_cpu, topk_send_indices, topk_target_ranks,
                                                 recv_counters)))
                print(f"{tag}: accepted", flush=True)
            except Exception as error:
                traceback.print_exc()
                moe_a2a.reset_state()
                results.append(("error", repr(error), None))
        return results
    except Exception:
        traceback.print_exc()
        raise


class TestMoEAlltoAll:

    @pytest.mark.threadleak(
        enabled=False
    )  # MPI pool executors have known thread cleanup timing issues
    @pytest.mark.parametrize(
        "mpi_pool_executor,all_num_tokens,top_k,enable_eplb,hidden_size,can_use_cft_counted_writes",
        [
            (4, [32, 32, 32, 32], 2, False, 1024, False),
            (4, [16, 32, 64, 48], 2, False, 1024, False),
            (2, [100, 50], 2, False, 1024, False),
            pytest.param(8, [10, 20, 30, 40, 50, 60, 70, 80],
                         2,
                         False,
                         1024,
                         False,
                         marks=_NEEDS_8_GPUS),
            (4, [32, 32, 32, 32], 4, False, 1024, False),
            (4, [32, 32, 32, 32], 8, False, 1024, False),
            (4, [1, 1, 1, 1], 2, False, 1024, False),
            (4, [32, 32, 32, 32], 2, True, 1024, False),  # EPLB
            (4, [2, 2, 2, 2], 2, False, 7168, False),  # DSR1-like hidden
            (4, [64, 64, 64, 64], 2, False, 7168, False),  # larger batch + DSR1
            pytest.param(8, [64, 64, 64, 64, 64, 64, 64, 64],
                         2,
                         False,
                         7168,
                         False,
                         marks=_NEEDS_8_GPUS),  # 8-GPU
            (4, [32, 32, 32, 32
                 ], 2, False, 1024, True),  # CFT request with fallback
            (4, [32, 32, 32, 32], 8, False, 1024, True),  # CFT counted writes
            (4, [1, 1, 1, 1
                 ], 8, False, 1024, True),  # CFT counted writes, small batch
            # Expert-id receive buffer above kFusedSanitizeMaxBytes (4 x 1100 x 16
            # int32): the padding rows are filled by the sanitize kernel after the
            # dispatch instead of inside it.
            (4, [1100, 100, 0, 1100], 16, False, 512, False),
        ],
        indirect=["mpi_pool_executor"])
    def test_dispatch(self, mpi_pool_executor, all_num_tokens, top_k,
                      enable_eplb, hidden_size, can_use_cft_counted_writes):
        """Test MoE A2A dispatch with MNNVL across multiple GPUs."""

        try:
            MnnvlMemory.initialize()
            assert MnnvlMemory.supports_mnnvl()
        except Exception:
            pytest.skip("MNNVL not supported on this system")

        ep_size = mpi_pool_executor.num_workers
        assert ep_size == len(
            all_num_tokens), "ep_size does not match all_num_tokens"

        if ep_size > torch.cuda.device_count():
            pytest.skip(
                f"Need at least {ep_size} GPUs, found {torch.cuda.device_count()}"
            )

        num_experts = 32
        workspace_size_per_rank = 512 * 1024 * 1024
        invalid_token_expert_id = -1

        results = mpi_pool_executor.map(
            run_moe_a2a_dispatch_single_rank,
            *zip(*[(ep_size, all_num_tokens, top_k, workspace_size_per_rank,
                    num_experts, hidden_size, invalid_token_expert_id,
                    enable_eplb, can_use_cft_counted_writes)] * ep_size),
        )

        all_results = list(results)
        all_token_selected_experts = [r[0] for r in all_results]
        all_payloads = [r[1] for r in all_results]
        all_recv_tensors = [r[2] for r in all_results]
        all_topk_send_indices = [r[3] for r in all_results]
        all_topk_target_ranks = [r[4] for r in all_results]
        all_recv_counters = [r[5] for r in all_results]
        all_expert_id_payload_index = [r[6] for r in all_results]
        expert_id_payload_index = all_expert_id_payload_index[0]
        all_eplb_local_stats = [r[7] for r in all_results]
        all_eplb_gathered_stats = [r[8] for r in all_results]

        assert all(i == expert_id_payload_index
                   for i in all_expert_id_payload_index
                   ), "all_expert_id_payload_index should be the same"

        verify_dispatch(all_token_selected_experts, all_payloads,
                        all_recv_tensors, all_topk_send_indices,
                        all_topk_target_ranks, all_recv_counters, ep_size,
                        all_num_tokens, top_k, num_experts,
                        expert_id_payload_index, invalid_token_expert_id)

        if enable_eplb:
            expected_stats = torch.stack(all_eplb_local_stats, dim=0)
            for rank in range(ep_size):
                gathered_stats = all_eplb_gathered_stats[rank]
                assert gathered_stats is not None
                assert torch.equal(
                    gathered_stats,
                    expected_stats), (f"Rank {rank} gathered_stats mismatch")

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize(
        "mpi_pool_executor,rounds_num_tokens,hidden_size,global_scale",
        [
            # Kimi K3: the 3584-wide latent at decode batch sizes, FP4 scale 1.0 as the checkpoint ships it.
            (4, [[48, 48, 48, 48]], 3584, 1.0),
            (4, [[16, 32, 64, 48], [1, 1, 1, 1]], 3584, 0.37),
            (4, [[256, 256, 256, 256]], 3584, 1.0),
            (2, [[100, 50]], 1024, 1.0),
            (4, [[64, 64, 64, 64]], 7168, 1.0),
            # Changing counts, ranks without tokens, and (the fourth round) a dispatch whose round is aborted before
            # its combine, as after a failed forward; the sixth round is above kFusedSanitizeMaxBytes of expert ids
            # (4 x 1100 x 16 int32), so the sanitize kernel fills the padding rows after the dispatch.
            (4, [[32, 32, 32, 32], [5, 0, 7, 2], [40, 1, 33, 2], [1, 1, 1, 1],
                 [3, 0, 0, 8], [1100, 100, 0, 1100]], 512, 1.0),
        ],
        indirect=["mpi_pool_executor"])
    def test_dispatch_quantize_on_send(self, mpi_pool_executor,
                                       rounds_num_tokens, hidden_size,
                                       global_scale):
        """The dispatch quantizes BF16 hidden states while it sends them (MoeAlltoAll.dispatch with
        quantize_global_scale): every received row equals, bit for bit, the NVFP4 codes and E4M3 scales that
        fp4_quantize_kf and fp4_quantize give for the sender's input, and the rest of the dispatch (counts, routing,
        expert ids and scales, padding rows) and the rounds that follow behave as without it."""
        if not fused_round_kernels():
            pytest.skip("quantize on send needs TLLM_MOE_A2A_FUSED_ROUND=1")
        if not (10, 0) <= torch.cuda.get_device_capability() < (11, 0):
            pytest.skip("quantize on send needs an SM10x GPU")
        try:
            MnnvlMemory.initialize()
            assert MnnvlMemory.supports_mnnvl()
        except Exception:
            pytest.skip("MNNVL not supported on this system")

        ep_size = mpi_pool_executor.num_workers
        assert all(len(n) == ep_size for n in rounds_num_tokens)
        if ep_size > torch.cuda.device_count():
            pytest.skip(
                f"Need at least {ep_size} GPUs, found {torch.cuda.device_count()}"
            )

        top_k = 16
        num_experts = 32
        # The fourth round of the multi-round case ends without a combine.
        abort_rounds = (3, ) if len(rounds_num_tokens) > 3 else ()
        results = mpi_pool_executor.map(
            run_moe_a2a_quantize_on_send_single_rank,
            *zip(*[(ep_size, rounds_num_tokens, top_k, 512 * 1024 * 1024,
                    num_experts, hidden_size, global_scale, abort_rounds)] *
                 ep_size),
        )
        all_rounds = [list(r) for r in results]  # [rank][round]
        for round_idx, all_num_tokens in enumerate(rounds_num_tokens):
            per_rank = [rounds[round_idx] for rounds in all_rounds]
            verify_dispatch([r[0] for r in per_rank], [r[1] for r in per_rank],
                            [r[2] for r in per_rank], [r[3] for r in per_rank],
                            [r[4] for r in per_rank], [r[5] for r in per_rank],
                            ep_size, all_num_tokens, top_k, num_experts, 2, -1)

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize(
        "mpi_pool_executor,rounds_num_tokens,hidden_size,with_final_scales",
        [
            # Kimi K3: the 3584-wide latent at decode batch sizes, with the float32 final scales of the router.
            (4, [[48, 48, 48, 48]], 3584, True),
            # Ranks with other row counts than the largest, and without rows; the dispatch is one for all of them.
            (4, [[16, 32, 64, 48], [1, 1, 1, 1], [100, 0, 7, 33]], 3584, True),
            (4, [[100, 0, 7, 33], [3, 0, 0, 8]], 3584, False),
            (2, [[100, 50]], 1024, True),
        ],
        indirect=["mpi_pool_executor"])
    def test_nvlink_one_sided_quantize_on_send_ab(self, mpi_pool_executor,
                                                  rounds_num_tokens,
                                                  hidden_size,
                                                  with_final_scales):
        """NVLinkOneSided.dispatch with and without quantize_on_send_scale hands the layer the same thing: the codes,
        scales, expert ids and final scales that arrive, and the routing and counts of the dispatch, equal what the
        separate fp4_quantize in front of the dispatch gives, bit for bit, in both and in the rounds that follow each
        other on one workspace. The scheduler's gate (can_quantize_on_send) follows the switch: off unless it is set."""
        if not fused_round_kernels():
            pytest.skip("quantize on send needs TLLM_MOE_A2A_FUSED_ROUND=1")
        if not (10, 0) <= torch.cuda.get_device_capability() < (11, 0):
            pytest.skip("quantize on send needs an SM10x GPU")
        try:
            MnnvlMemory.initialize()
            assert MnnvlMemory.supports_mnnvl()
        except Exception:
            pytest.skip("MNNVL not supported on this system")

        ep_size = mpi_pool_executor.num_workers
        assert all(len(n) == ep_size for n in rounds_num_tokens)
        if ep_size > torch.cuda.device_count():
            pytest.skip(
                f"Need at least {ep_size} GPUs, found {torch.cuda.device_count()}"
            )

        top_k = 16
        num_experts = 32
        results = mpi_pool_executor.map(
            run_nvlink_one_sided_quantize_on_send_ab_single_rank,
            *zip(*[(ep_size, rounds_num_tokens, top_k, num_experts, hidden_size,
                    0.37, with_final_scales)] * ep_size),
        )
        per_rank_results = list(results)
        assert [gate for _, gate in per_rank_results
                ] == [[False, False, True]] * ep_size  # unset, 0, 1
        all_rounds = [rounds for rounds, _ in per_rank_results]  # [rank][round]
        for round_idx, all_num_tokens in enumerate(rounds_num_tokens):
            for arm in range(2):  # separate quantization, quantize on send
                per_rank = [rounds[round_idx][arm] for rounds in all_rounds]
                verify_dispatch([r[0]
                                 for r in per_rank], [r[1] for r in per_rank],
                                [r[2]
                                 for r in per_rank], [r[3] for r in per_rank],
                                [r[4] for r in per_rank],
                                [r[5] for r in per_rank], ep_size,
                                all_num_tokens, top_k, num_experts, 2, -1)

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize("mpi_pool_executor,shape_group", [(2, "small"),
                                                               (2, "largest"),
                                                               (2, "payloads")],
                             indirect=["mpi_pool_executor"])
    def test_quantize_on_send_launch_boundaries(self, mpi_pool_executor,
                                                shape_group):
        """The shapes that NVLinkOneSided.can_quantize_on_send accepts (quantize_on_send_launchable) are the shapes
        whose quantize-on-send dispatch the launch accepts and sends correctly, in the process's
        TLLM_MOE_A2A_DISPATCH_BLOCK_SIZE; every other shape raises on all ranks before anything is launched, and the
        dispatches after it work. The groups: a K3 and a smaller hidden size and one that is not a multiple of 256;
        the largest hidden size the gate accepts and the steps around it; the bytes of other payloads (16 vectors
        beside the codes and scales is the most that fit). Run once per block size (a pytest run checks the one of
        its environment): at 128 and 512 the bound is the vectors a thread stages, at 1024 the 48 KB of shared memory.
        A rejected shape needs C++ exceptions to propagate (see test_k3_moe_sort_kf.py: LD_PRELOAD=libstdc++.so.6 in an
        image where libcurand.so.11 interposes the exception personality routine)."""
        if not fused_round_kernels():
            pytest.skip("quantize on send needs TLLM_MOE_A2A_FUSED_ROUND=1")
        if not (10, 0) <= torch.cuda.get_device_capability() < (11, 0):
            pytest.skip("quantize on send needs an SM10x GPU")
        try:
            MnnvlMemory.initialize()
            assert MnnvlMemory.supports_mnnvl()
        except Exception:
            pytest.skip("MNNVL not supported on this system")

        ep_size = mpi_pool_executor.num_workers
        if ep_size > torch.cuda.device_count():
            pytest.skip(
                f"Need at least {ep_size} GPUs, found {torch.cuda.device_count()}"
            )

        top_k = 16
        num_experts = 32
        num_tokens = 8
        block_size = _dispatch_block_size()
        # int32 expert ids and bfloat16 final scales go as they are, besides the extra bytes of the shapes below.
        other_row_bytes = top_k * 4 + top_k * 2

        def launchable(hidden_size, extra_bytes):
            return quantize_on_send_launchable(hidden_size, top_k,
                                               other_row_bytes + extra_bytes,
                                               block_size)

        # The largest hidden size that the gate accepts in this process, and the steps around it.
        accepted_hidden = [
            h for h in range(256, 100_000, 256) if launchable(h, 0)
        ]
        edge = accepted_hidden[-1] if accepted_hidden else 0
        shapes = {
            "small": [(512, 0), (3584, 0), (3584 + 128, 0)],  # 3712: not 256 n
            "largest": [(h, 0)
                        for h in (edge - 256, edge, edge + 256, edge + 512)
                        if h >= 256],
            # 16 vectors outside the codes and the scales is the most that fit: 96 + 160 bytes, but not 96 + 176.
            "payloads": [(3584, 160), (3584, 176)],
        }[shape_group] + [(512, 0)]  # the dispatch after the others works

        results = mpi_pool_executor.map(
            run_moe_a2a_quantize_on_send_boundaries_single_rank,
            *zip(*[(ep_size, shapes, num_tokens, top_k, 512 * 1024 * 1024,
                    num_experts, 1.0)] * ep_size),
        )
        all_results = [list(r) for r in results]  # [rank][shape]
        for shape_idx, (hidden_size, extra_bytes) in enumerate(shapes):
            per_rank = [rank_results[shape_idx] for rank_results in all_results]
            expect_accepted = launchable(hidden_size, extra_bytes)
            verdicts = [verdict for verdict, _, _ in per_rank]
            assert verdicts == [
                "accepted" if expect_accepted else "rejected"
            ] * ep_size, (f"hidden {hidden_size} + {extra_bytes} "
                          f"extra bytes at block size {block_size}: "
                          f"{verdicts}, {[e for _, e, _ in per_rank]}")
            if expect_accepted:
                rounds = [data for _, _, data in per_rank]
                verify_dispatch([r[0] for r in rounds], [r[1] for r in rounds],
                                [r[2] for r in rounds], [r[3] for r in rounds],
                                [r[4] for r in rounds], [r[5] for r in rounds],
                                ep_size, [num_tokens] * ep_size, top_k,
                                num_experts, 2, -1)
            else:
                for _, error, _ in per_rank:
                    assert "quantize on send" in error, error

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize("can_use_cft_counted_writes", [False, True])
    @pytest.mark.parametrize(
        "mpi_pool_executor,all_num_tokens,top_k,payload_in_workspace,use_fp8_combine",
        [
            # (num_workers, all_num_tokens, top_k, payload_in_workspace, use_fp8_combine)
            (4, [32, 32, 32, 32], 2, False, False),
            (4, [16, 32, 64, 48], 2, False, False),
            (2, [100, 50], 2, False, False),
            (4, [32, 32, 32, 32], 4, False, False),
            (4, [32, 32, 32, 32
                 ], 10, False, False),  # top_k=10 used by Qwen3-next
            (4, [1, 1, 1, 1], 2, False, False),
            (8, [640, 640, 640, 640, 640, 640, 640, 640], 4, False, False),
            (4, [32, 0, 16, 0], 2, False, False),
            # payload_in_workspace=True
            (4, [32, 32, 32, 32], 4, True, False),
            (4, [32, 0, 16, 0], 4, True, False),
            (4, [16, 32, 64, 48], 4, True, False),  # non-uniform tokens
            (4, [32, 32, 32, 32], 10, True, False),
            # use_fp8_combine=True: staged quantization (external payload)
            (4, [32, 32, 32, 32], 4, False, True),
            (4, [32, 0, 16, 0], 4, False, True),
            (4, [16, 32, 64, 48], 4, False, True),  # non-uniform tokens
            (4, [32, 32, 32, 32], 10, False, True),
            # use_fp8_combine=True, payload_in_workspace=True: in-place quantization
            (4, [32, 32, 32, 32], 4, True, True),
            (4, [32, 0, 16, 0], 4, True, True),
            (4, [16, 32, 64, 48], 4, True, True),  # non-uniform tokens
            (4, [32, 32, 32, 32], 10, True, True),
            # top_k=16 with FP8 combine (Kimi K3) takes the register-lean gather
            # (vectorized_combine_impl_lowreg); repeated target ranks exercise the
            # skipped (+0.0) contributions.
            (4, [32, 32, 32, 32], 16, False, True),
            (4, [16, 32, 64, 48], 16, True, True),
        ],
        indirect=["mpi_pool_executor"])
    def test_combine(self, mpi_pool_executor, all_num_tokens, top_k,
                     payload_in_workspace, use_fp8_combine,
                     can_use_cft_counted_writes):
        """Test MoE A2A combine with MNNVL across multiple GPUs.

        When use_fp8_combine=True, checks the combine bit for bit against every target
        rank's E4M3 wire values summed in the kernel's order (plus the dispatch routing
        and received payloads). When False, verifies against a ground-truth fake-MoE
        computation.
        """
        try:
            MnnvlMemory.initialize()
            assert MnnvlMemory.supports_mnnvl()
        except Exception:
            pytest.skip("MNNVL not supported on this system")

        ep_size = mpi_pool_executor.num_workers
        if ep_size > torch.cuda.device_count():
            pytest.skip(
                f"Need at least {ep_size} GPUs to run this test, but only {torch.cuda.device_count()} are available"
            )
        assert ep_size == len(
            all_num_tokens), "ep_size does not match all_num_tokens"

        # gpt-oss-20b
        hidden_size = 2880
        num_experts = 32

        # Large enough workspace
        workspace_size_per_rank = 512 * 1024 * 1024

        invalid_token_expert_id = -1
        results = mpi_pool_executor.map(
            run_moe_a2a_dispatch_moe_combine_single_rank,
            *zip(*[(ep_size, all_num_tokens, top_k, workspace_size_per_rank,
                    num_experts, hidden_size, invalid_token_expert_id,
                    payload_in_workspace, use_fp8_combine,
                    can_use_cft_counted_writes)] * ep_size),
        )

        try:
            all_results = list(results)
        except Exception:
            traceback.print_exc()
            raise

        if use_fp8_combine:
            verify_low_precision_combine_exact(all_results, ep_size, top_k,
                                               num_experts)
        else:
            verify_combine(all_results, ep_size, rtol=0.1, atol=0.5)

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize(
        "mpi_pool_executor,hidden_size,payload_dtype,expected_cft_combine,check_zero_runtime",
        [
            (4, 8, torch.bfloat16, False, True),
            (4, 8, torch.float16, False, False),
            (4, 16, torch.float16, True, False),
            (4, 16, torch.float32, True, False),
        ],
        indirect=["mpi_pool_executor"],
    )
    def test_low_precision_combine_wire_alignment_and_output_dtype(
            self, mpi_pool_executor, hidden_size, payload_dtype,
            expected_cft_combine, check_zero_runtime):
        """Verify FP8 wire alignment and payload-dtype output for fence and CFT combine."""
        try:
            MnnvlMemory.initialize()
            assert MnnvlMemory.supports_mnnvl()
        except Exception:
            pytest.skip("MNNVL not supported on this system")

        ep_size = mpi_pool_executor.num_workers
        if ep_size > torch.cuda.device_count():
            pytest.skip(
                f"Need at least {ep_size} GPUs to run this test, but only {torch.cuda.device_count()} are available"
            )

        all_num_tokens = [16] * ep_size
        worker_args = (
            ep_size,
            all_num_tokens,
            2,
            512 * 1024 * 1024,
            32,
            hidden_size,
            -1,
            False,
            True,
            True,
            payload_dtype,
            expected_cft_combine,
            check_zero_runtime,
        )
        results = mpi_pool_executor.map(
            run_moe_a2a_dispatch_moe_combine_single_rank,
            *zip(*[worker_args] * ep_size),
        )
        all_results = list(results)
        verify_combine(all_results, ep_size, rtol=0.13, atol=1.0)

    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize("dispatch_preload", [True, False])
    @pytest.mark.parametrize("cft_max_batch_for_dispatch", [None, 8])
    @pytest.mark.parametrize(
        "mpi_pool_executor,top_k,use_fp8_combine",
        [
            (4, 4, False),
            (4, 16, True),  # Kimi K3: top-16 with FP8 combine
        ],
        indirect=["mpi_pool_executor"])
    def test_multi_round(self, mpi_pool_executor, top_k, use_fp8_combine,
                         cft_max_batch_for_dispatch, dispatch_preload):
        """Several dispatch + combine rounds on one workspace.

        The token counts change every round, with ranks that have no tokens, so the
        receive layout changes too. With cft_max_batch_for_dispatch=8 the rounds with
        at most 8 tokens per rank use the counted-write dispatch and the others the
        fence dispatch, so consecutive rounds also switch between the two kernels
        (checked through the completion flags, which only the fence dispatch
        writes). The rounds are enqueued without host synchronization, except
        that with the fused-round kernels (TLLM_MOE_A2A_FUSED_ROUND=1) round 2 is
        aborted after its dispatch (MoeAlltoAll.reset_state, no combine), as after a
        failed forward. Every round checks the flag sequence, the received payloads
        and counts, the sanitized padding rows and the combined output bit for bit,
        and with the fused-round kernels that dispatch left its counters zeroed
        (with the legacy kernels, that it kept the send counts).
        dispatch_preload is the setting the shared pool workers run: the fused-round
        fence dispatch's payload preload (TLLM_MOE_A2A_DISPATCH_PRELOAD, read once
        per process) or its per-payload loop; the legacy kernels have no
        preload. The case for the other setting is skipped.
        """
        try:
            MnnvlMemory.initialize()
            assert MnnvlMemory.supports_mnnvl()
        except Exception:
            pytest.skip("MNNVL not supported on this system")

        ep_size = mpi_pool_executor.num_workers
        if ep_size > torch.cuda.device_count():
            pytest.skip(
                f"Need at least {ep_size} GPUs, found {torch.cuda.device_count()}"
            )
        # The pool's workers keep the preload setting of this process for their whole
        # life (and the legacy kernels have no preload), so only the case that
        # matches it can run here; TLLM_MOE_A2A_DISPATCH_PRELOAD=0 in a separate pytest
        # run checks the per-payload loop.
        if dispatch_preload != (dispatch_preload_enabled()
                                and fused_round_kernels()):
            pytest.skip("dispatch_preload differs from what the workers run "
                        "(TLLM_MOE_A2A_DISPATCH_PRELOAD / "
                        "TLLM_MOE_A2A_FUSED_ROUND of this pytest process)")

        rounds_num_tokens = [
            [32, 32, 32, 32],
            [5, 0, 7, 2],
            [40, 1, 33, 2],
            [1, 1, 1, 1],
            [3, 0, 0, 8],
            [17, 9, 0, 4],
        ]
        assert all(len(n) == ep_size for n in rounds_num_tokens)
        # The legacy kernels (TLLM_MOE_A2A_FUSED_ROUND unset) do not survive a
        # dispatch without combine: the next dispatch takes an even flag and its
        # combine reads the other counter bank. The fused-round kernels take the next
        # odd flag, so only they run the aborted round.
        abort_rounds = (2, ) if fused_round_kernels() else ()
        num_experts = 64
        hidden_size = 256
        results = mpi_pool_executor.map(
            run_moe_a2a_multi_round_single_rank,
            *zip(*[(ep_size, rounds_num_tokens, abort_rounds, top_k,
                    num_experts, hidden_size, use_fp8_combine,
                    cft_max_batch_for_dispatch, dispatch_preload)] * ep_size),
        )
        verify_multi_round(list(results), rounds_num_tokens, abort_rounds,
                           ep_size, top_k, num_experts, use_fp8_combine,
                           cft_max_batch_for_dispatch)


def run_moe_a2a_multi_round_single_rank(ep_size, rounds_num_tokens,
                                        abort_rounds, top_k, num_experts,
                                        hidden_size, use_low_precision_combine,
                                        cft_max_batch_for_dispatch,
                                        dispatch_preload):
    """Worker for test_multi_round: enqueues every round on one MoeAlltoAll without
    host synchronization (device-side copies of what verify_multi_round checks),
    so each dispatch launches right behind the previous combine, then synchronizes
    once and returns the per-round data."""
    from tensorrt_llm._torch.distributed import MoeAlltoAll as WorkerMoeAlltoAll

    rank = tllm.mpi_rank()
    torch.cuda.set_device(rank)
    # The C++ op reads both switches once per process and the pool's workers are
    # shared by the module's tests, so they come from the test process's environment.
    assert dispatch_preload == (dispatch_preload_enabled()
                                and fused_round_kernels())
    # The dispatch path must follow cft_max_batch_for_dispatch only. The combine
    # stays on the fence path: the counted-write combine has no barrier across
    # ranks, so without host syncs a peer could start the next dispatch into rows
    # this rank still has to copy.
    os.environ.pop(FORCE_CFT_ENV, None)
    try:
        mapping = Mapping(rank=rank,
                          tp_size=ep_size,
                          moe_ep_size=ep_size,
                          world_size=ep_size)
        use_cft = cft_max_batch_for_dispatch is not None
        env = {_CFT_MAX_BATCH_FOR_COMBINE_ENV: "0"}
        if use_cft:
            env[_CFT_MAX_BATCH_FOR_DISPATCH_ENV] = str(
                cft_max_batch_for_dispatch)
        with patch.dict(os.environ, env):
            moe_a2a = WorkerMoeAlltoAll(
                mapping=mapping,
                max_num_tokens=max(max(n) for n in rounds_num_tokens),
                top_k=top_k,
                num_slots=num_experts,
                workspace_size_per_rank=512 * 1024 * 1024,
                can_use_cft_counted_writes=use_cft,
            )
        index = WorkerMoeAlltoAll._METAINFO_INDEX

        def field(name, count):
            """Device copy of `count` int32 values of a workspace field."""
            start = moe_a2a.metainfo[index[name]].item()
            return moe_a2a.workspace[rank, start:start + count * 4].view(
                torch.int32).clone()

        first_flag = field("FLAG_VAL_OFFSET_INDEX", 1).item()
        rounds = []
        for round_idx, all_num_tokens in enumerate(rounds_num_tokens):
            num_tokens = all_num_tokens[rank]
            runtime_max_tokens = max(all_num_tokens)
            generator = torch.Generator(device="cuda")
            generator.manual_seed(1000 * round_idx + rank)
            experts = torch.randint(0,
                                    num_experts, (num_tokens, top_k),
                                    dtype=torch.int32,
                                    device="cuda",
                                    generator=generator)
            # Small integers keep every product and sum exact, so the combine check
            # is bitwise and does not depend on the reduction order.
            hidden = torch.randint(-8,
                                   9, (num_tokens, hidden_size),
                                   device="cuda",
                                   generator=generator).to(torch.bfloat16)
            scales = torch.rand(num_tokens,
                                top_k,
                                device="cuda",
                                generator=generator)
            flags_before = field("DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX",
                                 ep_size)
            recv_hidden, recv_ids, recv_scales = moe_a2a.dispatch(
                experts, [hidden, experts, scales],
                runtime_max_tokens,
                invalid_token_expert_id=-1,
                expert_id_payload_index=1)
            round_data = dict(
                experts=experts,
                hidden=hidden,
                scales=scales,
                recv_hidden=recv_hidden.clone(),
                recv_ids=recv_ids.clone(),
                recv_scales=recv_scales.clone(),
                flag_after_dispatch=field("FLAG_VAL_OFFSET_INDEX", 1),
                send_counters=field("SEND_COUNTERS_OFFSET_INDEX", ep_size),
                local_token_counter=field("LOCAL_TOKEN_COUNTER_OFFSET_INDEX",
                                          1),
                recv_counters=field("RECV_COUNTERS_OFFSET_INDEX",
                                    2 * ep_size).view(2, ep_size),
                flags_before=flags_before,
                flags_after=field("DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX",
                                  ep_size),
                target_ranks=field("TOPK_TARGET_RANKS_OFFSET_INDEX",
                                   num_tokens * top_k).view(num_tokens, top_k),
                send_indices=field("TOPK_SEND_INDICES_OFFSET_INDEX",
                                   num_tokens * top_k).view(num_tokens, top_k),
                cft_dispatch=moe_a2a.use_cft_for_dispatch(runtime_max_tokens),
                fused_round=fused_round_kernels(),
            )
            rounds.append(round_data)
            if round_idx in abort_rounds:
                # A forward that fails after its dispatch: no combine. Without the
                # combine's barrier the next dispatch could overwrite rows this
                # rank has not copied yet, hence the host barrier.
                moe_a2a.reset_state()
                torch.cuda.synchronize()
                tllm.mpi_barrier()
                continue
            # Stand-in for the experts: every received row times (rank + 1).
            round_data["combined"] = moe_a2a.combine(
                recv_hidden * (rank + 1),
                runtime_max_tokens,
                use_low_precision_combine=use_low_precision_combine)
            round_data["flag_after_combine"] = field("FLAG_VAL_OFFSET_INDEX", 1)
        torch.cuda.synchronize()
        return first_flag, [{
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in r.items()
        } for r in rounds]
    except Exception:
        traceback.print_exc()
        raise


def verify_multi_round(all_results, rounds_num_tokens, abort_rounds, ep_size,
                       top_k, num_experts, use_low_precision_combine,
                       cft_max_batch_for_dispatch):
    """Check every round of test_multi_round across all ranks."""
    num_experts_per_rank = num_experts // ep_size
    # A dispatch takes the next odd flag value, its combine the one after it.
    flag = all_results[0][0]
    assert all(r[0] == flag for r in all_results)
    # Every worker runs the kernels the test process selected.
    assert all(d["fused_round"] == fused_round_kernels() for r in all_results
               for d in r[1])
    for round_idx, all_num_tokens in enumerate(rounds_num_tokens):
        aborted = round_idx in abort_rounds
        dispatch_flag = (flag + 1) | 1
        flag = dispatch_flag if aborted else dispatch_flag + 1
        data = []
        for rank in range(ep_size):
            d = all_results[rank][1][round_idx]
            where = f"round {round_idx} rank {rank}"
            assert d["flag_after_dispatch"].item() == dispatch_flag, where
            if not aborted:
                assert d["flag_after_combine"].item(
                ) == dispatch_flag + 1, where
            if d["fused_round"]:
                # Both dispatch kernels leave their counters zeroed for the next round.
                assert torch.all(d["send_counters"] == 0), where
                assert d["local_token_counter"].item() == 0, where
            else:
                # The legacy kernels leave the round's send counts to the next
                # prepare kernel, which also shows that the op launched them.
                assert (int(d["send_counters"].sum())
                        > 0) == (all_num_tokens[rank] > 0), where
            # Which kernel ran: only the fence dispatch writes completion flags.
            expect_cft = (cft_max_batch_for_dispatch is not None
                          and max(all_num_tokens) <= cft_max_batch_for_dispatch)
            assert d["cft_dispatch"] == expect_cft, where
            if expect_cft:
                assert torch.equal(d["flags_after"], d["flags_before"]), where
            else:
                assert torch.all(d["flags_after"] == dispatch_flag), where
            parity = ((dispatch_flag - 1) >> 1) & 1
            d["recv_counts"] = d["recv_counters"][parity]
            data.append(d)
        # slots[target][source]: receive slots the source's tokens took on target.
        slots = [[[] for _ in range(ep_size)] for _ in range(ep_size)]
        for src in range(ep_size):
            src_data = data[src]
            for t in range(all_num_tokens[src]):
                seen = set()
                total = torch.zeros(src_data["hidden"].shape[-1],
                                    dtype=torch.float32)
                for k in range(top_k):
                    where = f"round {round_idx} rank {src} token {t} slot {k}"
                    owner = compute_target_rank_id(
                        int(src_data["experts"][t, k]), num_experts_per_rank)
                    keep = owner not in seen
                    seen.add(owner)
                    slot = int(src_data["send_indices"][t, k])
                    assert int(src_data["target_ranks"][t,
                                                        k]) == (owner if keep
                                                                else -1), where
                    assert (slot >= 0) == keep, where
                    if not keep:
                        continue
                    slots[owner][src].append(slot)
                    dst_data = data[owner]
                    assert torch.equal(dst_data["recv_hidden"][src, slot],
                                       src_data["hidden"][t]), where
                    assert torch.equal(dst_data["recv_ids"][src, slot],
                                       src_data["experts"][t]), where
                    assert torch.equal(dst_data["recv_scales"][src, slot],
                                       src_data["scales"][t]), where
                    contribution = src_data["hidden"][t].float() * (owner + 1)
                    if use_low_precision_combine:
                        contribution = fp8_e4m3_wire(contribution)
                    total += contribution
                if not aborted:
                    assert torch.equal(
                        src_data["combined"][t],
                        total.to(src_data["combined"].dtype)), (
                            f"round {round_idx} rank {src} token "
                            f"{t}: combined output differs")
        for dst in range(ep_size):
            for src in range(ep_size):
                count = len(slots[dst][src])
                where = f"round {round_idx}: rank {dst} from rank {src}"
                assert sorted(slots[dst][src]) == list(range(count)), where
                assert data[dst]["recv_counts"][src].item() == count, where
                assert torch.all(data[dst]["recv_ids"][src,
                                                       count:] == -1), where


def run_moe_a2a_dispatch_moe_combine_single_rank(
        ep_size,
        all_num_tokens,
        top_k,
        workspace_size_per_rank,
        num_experts,
        hidden_size,
        invalid_token_expert_id,
        payload_in_workspace=False,
        use_low_precision_combine=False,
        can_use_cft_counted_writes=False,
        payload_dtype=torch.bfloat16,
        expected_cft_combine=None,
        check_zero_runtime=False):
    """Worker function for dispatch and combine test.

    Runs one dispatch+combine round and returns
    (token_selected_experts, payloads, combined_output, rank_experts) for
    ground-truth verification via verify_combine.
    """
    from tensorrt_llm._torch.distributed import MoeAlltoAll as WorkerMoeAlltoAll

    rank = tllm.mpi_rank()
    torch.cuda.set_device(rank)
    device = torch.cuda.current_device()
    max_num_tokens = max(all_num_tokens)
    rank_local_tokens = all_num_tokens[rank]

    try:
        mapping = Mapping(rank=rank,
                          tp_size=ep_size,
                          moe_ep_size=ep_size,
                          world_size=ep_size)

        def create_moe_a2a():
            return WorkerMoeAlltoAll(
                mapping=mapping,
                max_num_tokens=max_num_tokens,
                top_k=top_k,
                num_slots=num_experts,
                workspace_size_per_rank=workspace_size_per_rank,
                can_use_cft_counted_writes=can_use_cft_counted_writes,
            )

        if expected_cft_combine is not None:
            with patch.dict(os.environ, {FORCE_CFT_ENV: "1"}):
                moe_a2a = create_moe_a2a()
        else:
            moe_a2a = create_moe_a2a()

        token_selected_experts = generate_token_selected_experts(
            rank_local_tokens, num_experts, top_k)
        payloads, expert_id_payload_index = make_floating_point_payloads(
            rank_local_tokens, hidden_size, top_k, rank, token_selected_experts,
            payload_dtype)

        num_experts_per_rank = num_experts // ep_size
        rank_experts = create_experts_per_rank(num_experts_per_rank,
                                               hidden_size,
                                               rank,
                                               device,
                                               dtype=payload_dtype)

        def dispatch_and_fake_moe():
            """Run one dispatch round; return the fake-MoE output and the received hidden
            states, both [ep_size, max_tokens, hidden]."""
            recv_tensors = moe_a2a.dispatch(
                token_selected_experts,
                payloads,
                max_num_tokens,
                invalid_token_expert_id=invalid_token_expert_id,
                expert_id_payload_index=expert_id_payload_index)
            hs, tse, tfs = recv_tensors[0], recv_tensors[1], recv_tensors[2]
            moe_out = fake_moe(
                hs.view(ep_size * max_num_tokens, hs.shape[-1]),
                tse.view(ep_size * max_num_tokens, tse.shape[-1]),
                tfs.view(ep_size * max_num_tokens, tfs.shape[-1]),
                rank_experts,
                is_ep=True,
                ep_rank=rank,
                num_experts_per_rank=num_experts_per_rank,
            )
            return moe_out.view(ep_size, max_num_tokens, hs.shape[-1]), hs

        def _combine(moe_out, use_low_precision):
            """Call combine, optionally staging moe_out via workspace buffer."""
            if payload_in_workspace:
                ws = moe_a2a.get_combine_payload_tensor_in_workspace(
                    max_num_tokens, hidden_size, payload_dtype)
                ws.copy_(moe_out.view(-1, hidden_size))
                return moe_a2a.combine(
                    ws.view(ep_size, max_num_tokens, hidden_size),
                    max_num_tokens,
                    use_low_precision_combine=use_low_precision,
                )
            return moe_a2a.combine(moe_out,
                                   max_num_tokens,
                                   use_low_precision_combine=use_low_precision)

        moe_out, recv_hidden = dispatch_and_fake_moe()
        topk_target_ranks, topk_send_indices = read_dispatch_routing(
            moe_a2a, rank, rank_local_tokens, top_k)
        if check_zero_runtime:
            combine_payload_offset = moe_a2a._state.combine_payload_offset
            assert combine_payload_offset is not None
            zero_runtime_payload = moe_out.new_empty((ep_size, 0, hidden_size))
            with pytest.raises(
                    RuntimeError,
                    match="runtimeMaxTokensPerRank must be positive"):

                def combine(*args, **kwargs):
                    return torch.ops.trtllm.moe_a2a_combine(*args, **kwargs)

                combine(
                    zero_runtime_payload,
                    0,
                    moe_a2a.workspace,
                    moe_a2a.metainfo,
                    0,
                    rank,
                    ep_size,
                    top_k,
                    int(combine_payload_offset),
                    use_low_precision=False,
                    use_cft_counted_writes=False,
                    enable_rank_mask=False,
                    active_rank_mask=None,
                )
        completion_flags_offset = moe_a2a.metainfo[
            WorkerMoeAlltoAll.
            _METAINFO_INDEX["COMBINE_COMPLETION_FLAGS_OFFSET_INDEX"]].item()
        completion_flags_before = moe_a2a.workspace[
            rank, completion_flags_offset:completion_flags_offset +
            ep_size * 4].view(torch.int32).cpu().clone()

        combined_output = _combine(moe_out,
                                   use_low_precision=use_low_precision_combine)
        assert combined_output.dtype == payload_dtype

        requested_cft_combine = moe_a2a.use_cft_for_combine(max_num_tokens)
        payload_element_size = torch.empty((),
                                           dtype=payload_dtype).element_size()
        wire_bytes_per_token = hidden_size * (1 if use_low_precision_combine
                                              else payload_element_size)
        cft_eligible = requested_cft_combine and wire_bytes_per_token % 16 == 0
        if expected_cft_combine is not None:
            assert requested_cft_combine
            assert cft_eligible is expected_cft_combine

        completion_flags_after = moe_a2a.workspace[
            rank, completion_flags_offset:completion_flags_offset +
            ep_size * 4].view(torch.int32).cpu()
        flag_val_offset = moe_a2a.metainfo[
            WorkerMoeAlltoAll._METAINFO_INDEX["FLAG_VAL_OFFSET_INDEX"]].item()
        expected_flag_val = moe_a2a.workspace[rank,
                                              flag_val_offset:flag_val_offset +
                                              4].view(torch.int32).cpu()

        # Completion flags are written only by fence combine. Their unchanged
        # value positively distinguishes an eligible CFT launch from fallback.
        if cft_eligible:
            assert torch.equal(
                completion_flags_after, completion_flags_before
            ), (f"Rank {rank} CFT combine changed completion flags: "
                f"before={completion_flags_before}, after={completion_flags_after}"
                )
            assert not torch.all(completion_flags_before == expected_flag_val)
        else:
            assert torch.all(completion_flags_after == expected_flag_val), (
                f"Rank {rank} completion flags: {completion_flags_after}, expected flag val: {expected_flag_val}"
            )

        return (
            token_selected_experts.cpu(),
            [p.cpu() for p in payloads],
            combined_output.cpu(),
            rank_experts.cpu(),
            moe_out.cpu(),
            recv_hidden.cpu(),
            topk_target_ranks,
            topk_send_indices,
        )
    except Exception:
        traceback.print_exc()
        raise


def read_dispatch_routing(moe_a2a, rank, num_tokens, top_k):
    """Routing written by dispatch: the target rank and receive slot of every top-k
    slot, -1 where a slot repeats a target rank (both [num_tokens, top_k], CPU)."""
    offsets = {
        name: moe_a2a.metainfo[MoeAlltoAll._METAINFO_INDEX[name]].item()
        for name in ("TOPK_TARGET_RANKS_OFFSET_INDEX",
                     "TOPK_SEND_INDICES_OFFSET_INDEX")
    }

    def read(name):
        start = offsets[name]
        return moe_a2a.workspace[rank,
                                 start:start + num_tokens * top_k * 4].view(
                                     torch.int32).view(num_tokens, top_k).cpu()

    return (read("TOPK_TARGET_RANKS_OFFSET_INDEX"),
            read("TOPK_SEND_INDICES_OFFSET_INDEX"))


def combine_tree_pairs(top_k):
    """The (dst, src) FP32 additions of combine_tree_sum<TOP_K> (moeAlltoAllKernels.cu)
    in order: pairwise trees for the TOP_K values it lists, else sequential into 0."""
    if top_k not in (2, 4, 6, 8, 10, 16, 22):
        return [(0, k) for k in range(1, top_k)]
    pairs = []
    stride = 1
    while stride < top_k:
        pairs += [(i, i + stride) for i in range(0, top_k, 2 * stride)
                  if i + stride < top_k]
        stride *= 2
    return pairs


def fp8_e4m3_wire(x):
    """BF16/FP16 -> E4M3 -> FP32, rounding to nearest even and saturating at 448
    like the kernel's cvt.rn.satfinite."""
    return x.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()


def verify_low_precision_combine_exact(all_results, ep_size, top_k,
                                       num_experts):
    """Check the FP8 combine bit for bit.

    For every token: the dispatch routing (first slot of each target rank kept, later
    repeats -1), the payload each target rank received, and the combined output against
    a reference built from each target rank's fake-MoE row cast to E4M3 (the wire format),
    summed in FP32 in the kernel's reduction order with +0.0 for repeated slots, then
    rounded to the output dtype.
    """
    num_experts_per_rank = num_experts // ep_size
    all_moe_out = [r[4] for r in all_results]
    all_recv_hidden = [r[5] for r in all_results]
    pairs = combine_tree_pairs(top_k)
    for rank in range(ep_size):
        (token_selected_experts, payloads, combined_output, _, _, _,
         target_ranks, send_indices) = all_results[rank]
        hidden_states = payloads[0]
        expected = torch.empty_like(combined_output)
        for t in range(token_selected_experts.shape[0]):
            seen = set()
            parts = []
            for k in range(top_k):
                owner = compute_target_rank_id(
                    int(token_selected_experts[t, k]), num_experts_per_rank)
                keep = owner not in seen
                seen.add(owner)
                target = int(target_ranks[t, k])
                slot = int(send_indices[t, k])
                expected_target = owner if keep else -1
                assert target == expected_target and (slot >= 0) == keep, (
                    f"Rank {rank} token {t} slot {k}: routed to rank {target} "
                    f"slot {slot}, expected rank {expected_target}")
                if not keep:
                    parts.append(
                        torch.zeros(hidden_states.shape[-1],
                                    dtype=torch.float32))
                    continue
                received = all_recv_hidden[target][rank, slot]
                assert torch.equal(received, hidden_states[t]), (
                    f"Rank {rank} token {t}: wrong payload in rank {target} "
                    f"slot {slot}")
                parts.append(fp8_e4m3_wire(all_moe_out[target][rank, slot]))
            for dst, src in pairs:
                parts[dst] = parts[dst] + parts[src]
            expected[t] = parts[0].to(combined_output.dtype)
        mismatch = combined_output.view(torch.int16) != expected.view(
            torch.int16)
        assert not mismatch.any(), (
            f"Rank {rank}: {int(mismatch.sum())} of {mismatch.numel()} combined "
            f"elements differ bitwise from the E4M3 wire reference; first at "
            f"{torch.nonzero(mismatch)[0].tolist()}")


def verify_combine(all_results, ep_size, rtol, atol):
    """Verify that combine correctly sums the dispatched tokens."""

    # Extract results
    all_token_selected_experts = [r[0] for r in all_results]
    all_original_payloads = [r[1] for r in all_results]
    all_combined_outputs = [r[2] for r in all_results]
    all_rank_experts = [r[3]
                        for r in all_results]  # Extract experts from each rank

    # For each rank, verify the combined output
    for rank in range(ep_size):
        # print("### Verify rank %d ###" % rank)
        token_selected_experts = all_token_selected_experts[rank]
        original_payloads = all_original_payloads[rank]
        hidden_states = original_payloads[0]
        token_final_scales = original_payloads[2]

        combined_output = all_combined_outputs[rank]

        # Check the following are equal:
        # expected: Directly emulate MoE with all experts as if EP is not used.
        # actual: Tokens are dispatched to target ranks, MoE is performed on target ranks, and then the results from all target ranks are summed up (combine).

        # Gather all experts from all ranks for non-EP emulation
        all_experts = torch.cat(all_rank_experts, dim=0)
        expected_combined_output = fake_moe(hidden_states,
                                            token_selected_experts,
                                            token_final_scales,
                                            all_experts,
                                            is_ep=False)

        # Custom assertion with detailed error message
        try:
            torch.testing.assert_close(combined_output,
                                       expected_combined_output,
                                       rtol=rtol,
                                       atol=atol)
        except AssertionError as e:
            # Find the first mismatch location
            abs_diff = (combined_output - expected_combined_output).abs()
            rel_diff = abs_diff / (expected_combined_output.abs() + 1e-8)

            # Check both absolute and relative tolerance
            mask = (abs_diff > atol) & (rel_diff > rtol)
            if mask.any():
                # Get the first mismatch
                mismatch_indices = torch.nonzero(mask)[0].tolist()
                token_idx, elem_idx = mismatch_indices

                # Build context visualization
                context_values_expected = []
                context_values_actual = []

                for offset in [-2, -1, 0, 1, 2]:
                    idx = elem_idx + offset
                    if 0 <= idx < combined_output.shape[1]:
                        context_values_expected.append(
                            f"{expected_combined_output[token_idx, idx].item():.4f}"
                        )
                        context_values_actual.append(
                            f"{combined_output[token_idx, idx].item():.4f}")
                    else:
                        context_values_expected.append("-")
                        context_values_actual.append("-")

                # Add ... to indicate continuation
                expected_str = ' '.join(context_values_expected)
                actual_str = ' '.join(context_values_actual)

                # Add ... on left if not at beginning
                if elem_idx > 2:
                    expected_str = "... " + expected_str
                    actual_str = "... " + actual_str

                # Add ... on right if not at end
                if elem_idx < combined_output.shape[1] - 3:
                    expected_str = expected_str + " ..."
                    actual_str = actual_str + " ..."

                error_msg = f"\nexpected: [{expected_str}]\n"
                error_msg += f"actual:   [{actual_str}]\n"
                error_msg += f"\n{str(e)}"

                raise AssertionError(error_msg)
