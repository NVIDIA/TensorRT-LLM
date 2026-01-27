# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm as tllm
from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._torch.distributed import MoeAlltoAll
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


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


def make_bfloat16_payloads(
        local_num_tokens: int, hidden_size: int, top_k: int, rank: int,
        token_selected_experts: torch.Tensor) -> tuple[list, int]:
    """Create bfloat16 test payloads matching nvfp4 structure but without scaling factors."""
    payloads = []

    # Payload 0: Hidden states (bfloat16)
    hidden_states = torch.randn(local_num_tokens,
                                hidden_size,
                                dtype=torch.bfloat16,
                                device='cuda')
    # Add rank-specific pattern for verification
    hidden_states += rank
    payloads.append(hidden_states)

    # Payload 1: token_selected_experts
    payloads.append(token_selected_experts)

    # Payload 2: token_final_scales (bfloat16) - similar to nvfp4's payload 4
    token_final_scales = torch.rand(local_num_tokens,
                                    top_k,
                                    dtype=torch.bfloat16,
                                    device='cuda')

    # Optional: Construct the data that is easier to debug
    # token_final_scales[:, 0] = rank
    # token_final_scales[:, 1] = torch.linspace(0, local_num_tokens - 1, local_num_tokens, dtype=torch.bfloat16, device='cuda')

    payloads.append(token_final_scales)

    return payloads, 1


def run_moe_a2a_dispatch_single_rank(ep_size, all_num_tokens, top_k,
                                     workspace_size_per_rank, num_experts,
                                     hidden_size, invalid_token_expert_id,
                                     enable_eplb):
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
        )

        # Get the number of tokens for this specific rank (same as single-GPU)
        rank_local_tokens = all_num_tokens[rank]

        # Generate data using helper functions
        token_selected_experts = generate_token_selected_experts(
            rank_local_tokens, num_experts, top_k)
        payloads, expert_id_payload_index = make_nvfp4_payloads(
            rank_local_tokens, hidden_size, top_k, rank, token_selected_experts)

        eplb_local_stats = None
        if enable_eplb:
            eplb_local_stats = (torch.arange(
                eplb_stats_num_experts, dtype=torch.int32, device="cuda") +
                                rank * 1000)

        recv_tensors = moe_a2a.dispatch(
            token_selected_experts,
            payloads,
            max_num_tokens,
            invalid_token_expert_id=invalid_token_expert_id,
            expert_id_payload_index=expert_id_payload_index,
            eplb_local_stats=eplb_local_stats)

        # Verify completion flags after dispatch
        completion_flags_offset = moe_a2a.metainfo[MoeAlltoAll._METAINFO_INDEX[
            "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX"]].item()
        completion_flags = moe_a2a.workspace[
            rank, completion_flags_offset:completion_flags_offset +
            ep_size * 4].view(torch.int32).cpu()
        flag_val_offset = moe_a2a.metainfo[
            MoeAlltoAll._METAINFO_INDEX["FLAG_VAL_OFFSET_INDEX"]].item()
        expected_flag_val = moe_a2a.workspace[rank,
                                              flag_val_offset:flag_val_offset +
                                              4].view(torch.int32).cpu()

        assert torch.all(completion_flags == expected_flag_val), (
            f"Rank {rank} completion flags: {completion_flags}, expected flag val: {expected_flag_val}"
        )

        # Read counters and compact routing tensors from workspace
        send_counters_offset = moe_a2a.metainfo[
            MoeAlltoAll._METAINFO_INDEX["SEND_COUNTERS_OFFSET_INDEX"]].item()
        recv_counters_offset = moe_a2a.metainfo[
            MoeAlltoAll._METAINFO_INDEX["RECV_COUNTERS_OFFSET_INDEX"]].item()
        topk_target_ranks_offset = moe_a2a.metainfo[MoeAlltoAll._METAINFO_INDEX[
            "TOPK_TARGET_RANKS_OFFSET_INDEX"]].item()
        topk_send_indices_offset = moe_a2a.metainfo[MoeAlltoAll._METAINFO_INDEX[
            "TOPK_SEND_INDICES_OFFSET_INDEX"]].item()

        send_counters = moe_a2a.workspace[
            rank, send_counters_offset:send_counters_offset + ep_size * 4].view(
                torch.int32).cpu()
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

        # Return results to be collected (move to CPU for MPI transfer)
        eplb_gathered_stats = moe_a2a._state.eplb_gathered_stats
        if eplb_gathered_stats is not None:
            eplb_gathered_stats = eplb_gathered_stats.cpu()
        if eplb_local_stats is not None:
            eplb_local_stats = eplb_local_stats.cpu()

        return (token_selected_experts.cpu(), [p.cpu() for p in payloads],
                [rt.cpu() for rt in recv_tensors], send_counters,
                topk_send_indices, topk_target_ranks, recv_counters,
                expert_id_payload_index, eplb_local_stats, eplb_gathered_stats)
    except Exception:
        traceback.print_exc()
        raise


def verify_dispatch(all_token_selected_experts, all_payloads, all_recv_tensors,
                    all_send_counters, all_topk_send_indices,
                    all_topk_target_ranks, all_recv_counters, ep_size,
                    all_num_tokens, top_k, num_experts, expert_id_payload_index,
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
        send_counters = all_send_counters[send_rank]
        assert len(
            send_counters.shape) == 1, "send_counters should be a 1D tensor"
        assert send_counters.shape[0] == ep_size
        assert send_counters.dtype == torch.int32

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

    # Verify send_counters per (send_rank -> target_rank)
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
            actual_to_rank = all_send_counters[send_rank][target_rank].item()
            assert actual_to_rank == expected_to_rank, (
                f"Rank {send_rank} sent {actual_to_rank} tokens to rank {target_rank}, expected {expected_to_rank}"
            )

    # Verify recv_counters match send_counters
    for recv_rank in range(ep_size):
        for send_rank in range(ep_size):
            expected_recv = all_send_counters[send_rank][recv_rank].item()
            actual_recv = all_recv_counters[recv_rank][send_rank].item()
            assert actual_recv == expected_recv, (
                f"Rank {recv_rank} received {actual_recv} tokens from rank {send_rank}, expected {expected_recv}"
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


class TestMoEAlltoAll:

    @pytest.mark.skipif(torch.cuda.device_count() < 8,
                        reason='needs at least 8 GPUs to run multi-GPU test')
    @pytest.mark.threadleak(
        enabled=False
    )  # MPI pool executors have known thread cleanup timing issues
    @pytest.mark.parametrize(
        "mpi_pool_executor,all_num_tokens,top_k,enable_eplb",
        [
            # (num_workers, all_num_tokens, top_k)
            # Basic configurations
            (4, [32, 32, 32, 32], 2, False
             ),  # Four ranks with uniform distribution
            (4, [16, 32, 64, 48
                 ], 2, False),  # Four ranks with non-uniform distribution
            (2, [100, 50], 2, False),  # Two ranks with different loads
            (8, [10, 20, 30, 40, 50, 60, 70, 80
                 ], 2, False),  # Eight ranks with increasing load

            # Different top_k values
            (4, [32, 32, 32, 32], 4, False),  # Four ranks with top_k = 4
            (4, [32, 32, 32, 32], 8, False),  # Four ranks with top_k = 8

            # Edge cases
            (4, [1, 1, 1, 1], 2, False
             ),  # Four ranks with single token per rank

            # EPLB stats path
            (4, [32, 32, 32, 32], 2, True),
        ],
        indirect=["mpi_pool_executor"])
    def test_dispatch(self, mpi_pool_executor, all_num_tokens, top_k,
                      enable_eplb):
        """Test MoE A2A dispatch with MNNVL across multiple GPUs"""

        try:
            MnnvlMemory.initialize()
            assert MnnvlMemory.supports_mnnvl()
        except Exception:
            pytest.skip("MNNVL not supported on this system")

        ep_size = mpi_pool_executor.num_workers
        assert ep_size == len(
            all_num_tokens), "ep_size does not match all_num_tokens"

        assert torch.cuda.device_count(
        ) >= ep_size, f"Need at least {ep_size} GPUs, found {torch.cuda.device_count()}"

        hidden_size = 1024
        num_experts = 32

        # Large enough workspace
        workspace_size_per_rank = 512 * 1024 * 1024

        invalid_token_expert_id = -1

        # Run dispatch on workers - each worker executes the same logic as single-GPU
        # but on separate GPUs with MNNVL memory instead of regular CUDA memory
        results = mpi_pool_executor.map(
            run_moe_a2a_dispatch_single_rank,
            *zip(*[(ep_size, all_num_tokens, top_k, workspace_size_per_rank,
                    num_experts, hidden_size, invalid_token_expert_id,
                    enable_eplb)] * ep_size),
        )

        # Collect results from all ranks (same as single-GPU collecting from emulated ranks)
        all_results = list(results)

        # Extract results in same format as single-GPU test
        all_token_selected_experts = [r[0] for r in all_results]
        all_payloads = [r[1] for r in all_results]
        all_recv_tensors = [r[2] for r in all_results]
        all_send_counters = [r[3] for r in all_results]
        all_topk_send_indices = [r[4] for r in all_results]
        all_topk_target_ranks = [r[5] for r in all_results]
        all_recv_counters = [r[6] for r in all_results]
        all_expert_id_payload_index = [r[7] for r in all_results]
        expert_id_payload_index = all_expert_id_payload_index[0]
        all_eplb_local_stats = [r[8] for r in all_results]
        all_eplb_gathered_stats = [r[9] for r in all_results]

        assert all(i == expert_id_payload_index
                   for i in all_expert_id_payload_index
                   ), "all_expert_id_payload_index should be the same"

        # Verify dispatch results with content verification
        verify_dispatch(all_token_selected_experts, all_payloads,
                        all_recv_tensors, all_send_counters,
                        all_topk_send_indices, all_topk_target_ranks,
                        all_recv_counters, ep_size, all_num_tokens, top_k,
                        num_experts, expert_id_payload_index,
                        invalid_token_expert_id)

        if enable_eplb:
            expected_stats = torch.stack(all_eplb_local_stats, dim=0)
            for rank in range(ep_size):
                gathered_stats = all_eplb_gathered_stats[rank]
                assert gathered_stats is not None
                assert torch.equal(
                    gathered_stats,
                    expected_stats), (f"Rank {rank} gathered_stats mismatch")

    @pytest.mark.skipif(torch.cuda.device_count() < 8,
                        reason='needs at least 8 GPUs to run multi-GPU test')
    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize(
        "mpi_pool_executor,all_num_tokens,top_k",
        [
            # (num_workers, all_num_tokens, top_k)
            (4, [32, 32, 32, 32], 2),
            (4, [16, 32, 64, 48], 2),
            (2, [100, 50], 2),
            (4, [32, 32, 32, 32], 4),
            (4, [32, 32, 32, 32], 10),  # (top_k=10 is used by Qwen3-next)
            (4, [32, 32, 32, 32], 22),
            (4, [1, 1, 1, 1], 2),
            (8, [640, 640, 640, 640, 640, 640, 640, 640], 4),
            (4, [32, 0, 16, 0], 2),
        ],
        indirect=["mpi_pool_executor"])
    def test_combine(self, mpi_pool_executor, all_num_tokens, top_k):
        """Test MoE A2A combine with MNNVL across multiple GPUs"""

        try:
            MnnvlMemory.initialize()
            assert MnnvlMemory.supports_mnnvl()
        except Exception:
            pytest.skip("MNNVL not supported on this system")

        ep_size = mpi_pool_executor.num_workers
        assert ep_size == len(
            all_num_tokens), "ep_size does not match all_num_tokens"

        assert torch.cuda.device_count(
        ) >= ep_size, f"Need at least {ep_size} GPUs, found {torch.cuda.device_count()}"

        # gpt-oss-20b
        hidden_size = 2880
        num_experts = 32

        # Large enough workspace
        workspace_size_per_rank = 512 * 1024 * 1024

        # Run dispatch and combine on workers
        print("Starting dispatch and combine on workers...")
        invalid_token_expert_id = -1
        results = mpi_pool_executor.map(
            run_moe_a2a_dispatch_moe_combine_single_rank,
            *zip(*[(ep_size, all_num_tokens, top_k, workspace_size_per_rank,
                    num_experts, hidden_size, invalid_token_expert_id)] *
                 ep_size),
        )

        # Collect results
        print("Collecting results from workers...")
        try:
            all_results = list(results)
            print(
                f"Successfully collected results from {len(all_results)} workers"
            )
        except Exception as e:
            print(f"Error collecting results: {e}")
            traceback.print_exc()
            raise

        # Verify combine results
        print("Starting verification...")
        verify_combine(all_results, ep_size)


def run_moe_a2a_dispatch_moe_combine_single_rank(ep_size, all_num_tokens, top_k,
                                                 workspace_size_per_rank,
                                                 num_experts, hidden_size,
                                                 invalid_token_expert_id):
    """Worker function for dispatch and combine test."""
    rank = tllm.mpi_rank()
    torch.cuda.set_device(rank)
    device = torch.cuda.current_device()
    max_num_tokens = max(all_num_tokens)

    try:
        mapping = Mapping(rank=rank,
                          tp_size=ep_size,
                          moe_ep_size=ep_size,
                          world_size=ep_size)

        # Create MoeAlltoAll manager
        moe_a2a = MoeAlltoAll(
            mapping=mapping,
            max_num_tokens=max_num_tokens,
            top_k=top_k,
            num_slots=num_experts,
            workspace_size_per_rank=workspace_size_per_rank,
        )

        rank_local_tokens = all_num_tokens[rank]

        # Generate test data - use simpler payload for combine test
        token_selected_experts = generate_token_selected_experts(
            rank_local_tokens, num_experts, top_k)

        payloads, expert_id_payload_index = make_bfloat16_payloads(
            rank_local_tokens, hidden_size, top_k, rank, token_selected_experts)

        # Run dispatch
        with torch.cuda.profiler.profile():
            recv_tensors = moe_a2a.dispatch(
                token_selected_experts,
                payloads,
                max_num_tokens,
                invalid_token_expert_id=invalid_token_expert_id,
                expert_id_payload_index=expert_id_payload_index)

        hidden_states_recv = recv_tensors[
            0]  # [ep_size, max_num_tokens, hidden_size]
        token_selected_experts_recv = recv_tensors[
            1]  # [ep_size, max_num_tokens, top_k]
        token_final_scales_recv = recv_tensors[
            2]  # [ep_size, max_num_tokens, top_k]

        # emulate MoE computation on the received data
        # Create experts for this rank
        num_experts_per_rank = num_experts // ep_size
        rank_experts = create_experts_per_rank(num_experts_per_rank,
                                               hidden_size,
                                               rank,
                                               device,
                                               dtype=torch.bfloat16)

        hidden_states_recv = fake_moe(
            hidden_states_recv.view(ep_size * max_num_tokens,
                                    hidden_states_recv.shape[-1]),
            token_selected_experts_recv.view(
                ep_size * max_num_tokens,
                token_selected_experts_recv.shape[-1]),
            token_final_scales_recv.view(ep_size * max_num_tokens,
                                         token_final_scales_recv.shape[-1]),
            rank_experts,  # experts for current rank
            is_ep=True,
            ep_rank=rank,
            num_experts_per_rank=num_experts_per_rank).view(
                ep_size, max_num_tokens, hidden_states_recv.shape[-1])

        with torch.cuda.profiler.profile():
            combined_output = moe_a2a.combine(hidden_states_recv,
                                              max_num_tokens)

        # Verify completion flags after combine
        completion_flags_offset = moe_a2a.metainfo[MoeAlltoAll._METAINFO_INDEX[
            "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX"]].item()
        completion_flags_ptr = moe_a2a.workspace[
            rank, completion_flags_offset:completion_flags_offset + ep_size * 4]
        completion_flags = completion_flags_ptr.view(torch.int32).cpu()
        flag_val_offset = moe_a2a.metainfo[
            MoeAlltoAll._METAINFO_INDEX["FLAG_VAL_OFFSET_INDEX"]].item()
        expected_flag_val = moe_a2a.workspace[rank,
                                              flag_val_offset:flag_val_offset +
                                              4].view(torch.int32).cpu()
        assert torch.all(completion_flags == expected_flag_val), (
            f"Rank {rank} completion flags: {completion_flags}, expected flag val: {expected_flag_val}"
        )

        # Return results for verification
        return (
            token_selected_experts.cpu(),
            [p.cpu() for p in payloads],  # Return actual payloads used
            combined_output.cpu(),
            rank_experts.cpu()  # Return the experts used on this rank
        )
    except Exception:
        traceback.print_exc()
        raise


def verify_combine(all_results, ep_size):
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
                                       rtol=1e-1,
                                       atol=5e-1)
        except AssertionError as e:
            # Find the first mismatch location
            abs_diff = (combined_output - expected_combined_output).abs()
            rel_diff = abs_diff / (expected_combined_output.abs() + 1e-8)

            # Check both absolute and relative tolerance
            mask = (abs_diff > 5e-1) & (rel_diff > 1e-1)
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
