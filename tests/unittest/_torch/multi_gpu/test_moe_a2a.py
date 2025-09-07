# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# def compute_nvfp4_workspace_size(ep_size: int, max_tokens_per_rank: int,
#                                  hidden_size: int, top_k: int) -> int:
#     """Compute total workspace size per rank matching NV FP4 layout used in single-GPU test."""
#     packed_hidden_size = hidden_size // 2  # 2 FP4 per uint8
#     num_elts_per_sf = 16
#     num_scaling_factors = hidden_size // num_elts_per_sf
#     packed_tokens_size = ep_size * max_tokens_per_rank * packed_hidden_size * 1  # uint8
#     sf_size = ep_size * max_tokens_per_rank * num_scaling_factors * 4  # float32
#     experts_size = ep_size * max_tokens_per_rank * top_k * 4  # int32
#     final_scales_size = ep_size * max_tokens_per_rank * top_k * 4  # float32
#     completion_flags_size = ep_size * 4  # int32, one flag per rank
#     return packed_tokens_size + sf_size + experts_size + final_scales_size + completion_flags_size

# Before we correctly implement sync in kernel, we have to manually sync (GPU + all hosts MPI)
def sync():
    torch.cuda.synchronize()

    comm = MPI.COMM_WORLD
    comm.Barrier()



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

def generate_token_selected_experts(local_num_tokens: int, ep_size: int,
                                    num_experts_per_rank: int,
                                    top_k: int) -> torch.Tensor:
    """Generate global expert IDs tensor, aligned with single-GPU test semantics."""
    return torch.randint(
        0,
        ep_size * num_experts_per_rank,
        (local_num_tokens, top_k),
        dtype=torch.int32,
        device='cuda',
    )
    
def create_experts(num_experts_per_rank, hidden_size, ep_rank, device, dtype=torch.bfloat16):
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
    experts = torch.empty(
        (num_experts_per_rank, hidden_size, hidden_size),
        dtype=dtype,
        device=device
    )
    for i in range(num_experts_per_rank):
        torch.manual_seed(ep_rank * 1000 + i)
        # Xavier uniform initialization for each expert
        torch.nn.init.xavier_uniform_(experts[i])
    return experts


def fake_moe(hidden_states, token_selected_experts, token_final_scales, experts, is_ep=False, ep_rank=None, num_experts_per_rank=None):
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
                if not (expert_id >= ep_rank * num_experts_per_rank and expert_id < (ep_rank + 1) * num_experts_per_rank):
                    continue
                # Convert global expert ID to local expert ID for this rank
                local_expert_id = expert_id - ep_rank * num_experts_per_rank
                expert = experts[local_expert_id]
            else:
                expert = experts[expert_id]
            
            scale = token_final_scales[token_idx, k]
            processed_states[token_idx] += hidden_states[token_idx] @ expert * scale

    return processed_states


def make_nvfp4_payloads(local_num_tokens: int, hidden_size: int, top_k: int,
                        rank: int,
                        token_selected_experts: torch.Tensor) -> list:
    """Create the four NV FP4 payloads exactly as in single-GPU test."""
    payloads = []
    # Payload 0: Packed FP4 tokens (uint8)
    packed_hidden_size = hidden_size // 2
    packed_hidden_states = torch.randint(0,
                                  256, (local_num_tokens, packed_hidden_size),
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
    return payloads


def make_bfloat16_payloads(local_num_tokens: int, hidden_size: int, top_k: int,
                           rank: int,
                           token_selected_experts: torch.Tensor) -> list:
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

    # Optional: Add debug info like in nvfp4
    # token_final_scales[:, 0] = rank
    # token_final_scales[:, 1] = torch.linspace(0, local_num_tokens - 1, local_num_tokens, dtype=torch.bfloat16, device='cuda')

    payloads.append(token_final_scales)

    return payloads


def run_moe_a2a_dispatch_single_rank(ep_size, all_num_tokens, top_k,
                                     workspace_size_per_rank,
                                     num_experts_per_rank, hidden_size,
                                     max_tokens_per_rank):
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
        moe_a2a = MoeAlltoAll(mapping, max_tokens_per_rank, workspace_size_per_rank)

        # Get the number of tokens for this specific rank (same as single-GPU)
        rank_local_tokens = all_num_tokens[rank]

        # Generate data using helper functions
        token_selected_experts = generate_token_selected_experts(
            rank_local_tokens, ep_size, num_experts_per_rank, top_k)
        payloads = make_nvfp4_payloads(rank_local_tokens, hidden_size, top_k,
                                       rank, token_selected_experts)

        # Execute dispatch using wrapper to avoid pickle issues
        num_experts = ep_size * num_experts_per_rank
        recv_buffers, send_counters, send_indices = moe_a2a.dispatch(
            token_selected_experts, payloads, max_tokens_per_rank,
            top_k, num_experts)

        # TODO: remove this after synchronization is okay.
        # sync()

        # Verify completion flags after dispatch
        completion_flags_offset = moe_a2a.moe_a2a_metainfo[MoeAlltoAll.COMPLETION_FLAGS_OFFSET_INDEX].item()
        completion_flags = moe_a2a.workspace[
            rank, completion_flags_offset:completion_flags_offset + ep_size * 4].view(torch.int32).cpu()
        flag_val_offset = moe_a2a.moe_a2a_metainfo[MoeAlltoAll.FLAG_VAL_OFFSET_INDEX].item()
        expected_flag_val = moe_a2a.workspace[
            rank, flag_val_offset:flag_val_offset + 4].view(torch.int32).cpu() - 1


        print("completion_flags_ptr (hex):", hex(moe_a2a.workspace[
            rank, completion_flags_offset:completion_flags_offset + ep_size * 4].data_ptr()))

        assert torch.all(completion_flags == expected_flag_val), (
            f"Rank {rank} completion flags: {completion_flags}, expected flag val: {expected_flag_val}"
        )

        # Return results to be collected (move to CPU for MPI transfer)
        return (token_selected_experts.cpu(), [p.cpu() for p in payloads],
                [rb.cpu() for rb in recv_buffers], send_counters.cpu(),
                send_indices.cpu())
    except Exception:
        traceback.print_exc()
        raise


def verify_dispatch(all_token_selected_experts, all_payloads, all_recv_buffers,
                    all_send_counters, all_send_indices, ep_size,
                    all_num_tokens, top_k, max_tokens_per_rank, num_experts_per_rank):
    """Verify dispatch results including actual content verification"""

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
        recv_buffers = all_recv_buffers[send_rank]
        num_payloads = len(payloads)
        assert len(
            recv_buffers
        ) == num_payloads, "recv_buffers should have the same number of payloads as payloads"
        for i in range(num_payloads):
            payload = payloads[i]
            assert len(payload.shape) == 2, "payload should be a 2D tensor"
            assert payload.shape[
                0] == local_num_tokens, "payload.shape[0] should be local_num_tokens"

            recv_buffer = recv_buffers[i]
            assert len(
                recv_buffer.shape) == 3, "recv_buffer should be a 3D tensor"
            assert recv_buffer.shape[
                0] == ep_size, "recv_buffer.shape[0] should be ep_size"
            assert recv_buffer.shape[
                1] == max_tokens_per_rank, "recv_buffer.shape[1] should be max_tokens_per_rank"
            assert recv_buffer.shape[2] == payload.shape[
                1], "recv_buffer.shape[2] should be payload.shape[1]"
            assert recv_buffer.dtype == payload.dtype, "recv_buffer.dtype should be payload.dtype"

        send_counters = all_send_counters[send_rank]
        assert len(
            send_counters.shape) == 1, "send_counters should be a 1D tensor"
        assert send_counters.shape[
            0] == ep_size, "send_counters.shape[0] should be ep_size"
        assert send_counters.dtype == torch.int32, "send_counters.dtype should be torch.int32"

        send_indices = all_send_indices[send_rank]
        assert len(
            send_indices.shape) == 2, "send_indices should be a 2D tensor"
        assert send_indices.shape[
            0] == local_num_tokens, "send_indices.shape[0] should be local_num_tokens"
        assert send_indices.shape[
            1] == ep_size, "send_indices.shape[1] should be ep_size"
        assert send_indices.dtype == torch.int32, "send_indices.dtype should be torch.int32"

    # Verify send_counters
    for send_rank in range(ep_size):
        send_counters = all_send_counters[send_rank]

        # Count expected sends to each target
        expected_sends = {}
        token_experts = all_token_selected_experts[send_rank]
        sent_to_rank = set()

        for token_idx in range(token_experts.shape[0]):
            experts = token_experts[token_idx]
            target_ranks = compute_target_rank_id(experts, num_experts_per_rank)
            sent_to_rank.clear()

            # Due to deduplication, each token is sent to each unique target rank only once
            for target_rank in target_ranks.tolist():
                if target_rank not in sent_to_rank:
                    if target_rank not in expected_sends:
                        expected_sends[target_rank] = 0
                    expected_sends[target_rank] += 1
                    sent_to_rank.add(target_rank)

        # Verify send counters for each target rank
        for target_rank in range(ep_size):
            expected_to_rank = expected_sends.get(target_rank, 0)
            actual_to_rank = send_counters[target_rank].item()
            assert actual_to_rank == expected_to_rank, \
                f"Rank {send_rank} sent {actual_to_rank} tokens to rank {target_rank}, " \
                f"expected {expected_to_rank}"

    # Verify payloads using send_indices
    for send_rank in range(ep_size):
        send_indices = all_send_indices[
            send_rank]  # [local_num_tokens, ep_size]
        token_selected_experts = all_token_selected_experts[send_rank]
        payloads = all_payloads[send_rank]

        local_num_tokens = all_num_tokens[send_rank]

        # For each source token on this send rank
        for token_idx in range(local_num_tokens):
            experts = token_selected_experts[token_idx]
            target_ranks = compute_target_rank_id(experts, num_experts_per_rank)
            unique_targets = set(target_ranks.tolist())

            # Verify send_indices records correct destinations
            for target_rank in range(ep_size):
                dst_pos = send_indices[token_idx, target_rank].item()

                if target_rank in unique_targets:
                    # This token should have been sent to target_rank
                    assert dst_pos >= 0, \
                        f"Send rank {send_rank} token {token_idx} should be sent to rank {target_rank} but dst_pos={dst_pos}"
                    assert dst_pos < max_tokens_per_rank, \
                        f"Send rank {send_rank} token {token_idx} dst_pos={dst_pos} exceeds max_tokens_per_rank={max_tokens_per_rank}"

                    # Verify actual payload content was copied correctly
                    recv_buffers = all_recv_buffers[target_rank]
                    for payload_idx, payload in enumerate(payloads):
                        recv_buffer = recv_buffers[payload_idx]

                        source_data = payload[token_idx]
                        received_data = recv_buffer[send_rank, dst_pos]
                        # Compare source and received data
                        torch.testing.assert_close(
                            received_data,
                            source_data,
                            atol=0, # Dispatch is pure copy, should expact exactly the same
                            rtol=0,
                            msg=
                            f"Content mismatch: received_data={received_data} source_data={source_data} send_rank={send_rank} token_idx={token_idx} experts={experts.tolist()} target_rank={target_rank}, send_indices[token_idx]={send_indices[token_idx].tolist()}"
                        )
                else:
                    # This token should NOT have been sent to target_rank
                    assert dst_pos == -1, \
                        f"dst_pos should be -1: send_rank={send_rank} token_idx={token_idx} experts={experts.tolist()} target_rank={target_rank}, send_indices[token_idx]={send_indices[token_idx].tolist()}"


class TestMoEAlltoAll:

    @pytest.mark.skipif(torch.cuda.device_count() < 2,
                        reason='needs at least 2 GPUs to run multi-GPU test')
    @pytest.mark.threadleak(
        enabled=False
    )  # MPI pool executors have known thread cleanup timing issues
    @pytest.mark.parametrize(
        "mpi_pool_executor,all_num_tokens,top_k",
        [
            # (num_workers, all_num_tokens, top_k)
            # Basic configurations
            (4, [32, 32, 32, 32], 2),  # Four ranks with uniform distribution
            (4, [16, 32, 64, 48
                 ], 2),  # Four ranks with non-uniform distribution
            (2, [100, 50], 2),  # Two ranks with different loads
            (8, [10, 20, 30, 40, 50, 60, 70, 80
                 ], 2),  # Eight ranks with increasing load

            # Different top_k values
            (4, [32, 32, 32, 32], 4),  # Four ranks with top_k = 4
            (4, [32, 32, 32, 32], 8),  # Four ranks with top_k = 8

            # Edge cases
            (4, [1, 1, 1, 1], 2),  # Four ranks with single token per rank
        ],
        indirect=["mpi_pool_executor"])
    def test_dispatch(self, mpi_pool_executor, all_num_tokens, top_k):
        """Test MoE A2A dispatch with MNNVL across multiple GPUs"""

        try:
            MnnvlMemory.initialize()
            assert MnnvlMemory.supports_mnnvl()
        except Exception:
            pytest.skip("MNNVL not supported on this system")

        ep_size = mpi_pool_executor.num_workers
        assert ep_size == len(
            all_num_tokens), "ep_size does not match all_num_tokens"

        # Skip if not enough GPUs
        assert torch.cuda.device_count(
        ) >= ep_size, f"Need at least {ep_size} GPUs, found {torch.cuda.device_count()}"

        hidden_size = 1024
        num_experts_per_rank = 8
        max_tokens_per_rank = max(all_num_tokens)

        # Calculate workspace size for all payloads
        # workspace_size_per_rank = compute_nvfp4_workspace_size(
        #     ep_size, max_tokens_per_rank, hidden_size, top_k)
        workspace_size_per_rank = 512 * 1024 * 1024  # Large enough workspace

        # Run dispatch on workers - each worker executes the same logic as single-GPU
        # but on separate GPUs with MNNVL memory instead of regular CUDA memory
        results = mpi_pool_executor.map(
            run_moe_a2a_dispatch_single_rank,
            *zip(*[(ep_size, all_num_tokens, top_k, workspace_size_per_rank,
                    num_experts_per_rank, hidden_size, max_tokens_per_rank)] *
                 ep_size),
        )

        # Collect results from all ranks (same as single-GPU collecting from emulated ranks)
        all_results = list(results)

        # Extract results in same format as single-GPU test
        all_token_selected_experts = [r[0] for r in all_results]
        all_payloads = [r[1] for r in all_results]
        all_recv_buffers = [r[2] for r in all_results]
        all_send_counters = [r[3] for r in all_results]
        all_send_indices = [r[4] for r in all_results]

        # Verify dispatch results with content verification
        verify_dispatch(all_token_selected_experts, all_payloads,
                        all_recv_buffers, all_send_counters, all_send_indices,
                        ep_size, all_num_tokens, top_k, max_tokens_per_rank, num_experts_per_rank)

    @pytest.mark.skipif(torch.cuda.device_count() < 2,
                        reason='needs at least 2 GPUs to run multi-GPU test')
    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize(
        "mpi_pool_executor,all_num_tokens,top_k,dtype",
        [
            # (num_workers, all_num_tokens, top_k, dtype)
            (4, [32, 32, 32, 32], 2, torch.float32),
            (4, [16, 32, 64, 48], 2, torch.float32),
            (2, [100, 50], 2, torch.float16),
            (4, [32, 32, 32, 32], 4, torch.float32),
            (4, [1, 1, 1, 1], 2, torch.float32),
        ],
        indirect=["mpi_pool_executor"])
    def test_combine(self, mpi_pool_executor, all_num_tokens, top_k, dtype):
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
        ) >= ep_size, f"Need at least {ep_size} GPUs"

        hidden_size = 256  # Smaller for combine test
        num_experts_per_rank = 8
        max_tokens_per_rank = max(all_num_tokens)

        # Calculate workspace size
        # workspace_size_per_rank = compute_nvfp4_workspace_size(
        #     ep_size, max_tokens_per_rank, hidden_size, top_k)
        workspace_size_per_rank = 512 * 1024 * 1024  # Large enough workspace

        # Run dispatch and combine on workers
        print("Starting dispatch and combine on workers...")
        results = mpi_pool_executor.map(
            run_moe_a2a_dispatch_moe_combine_single_rank,
            *zip(*[(ep_size, all_num_tokens, top_k, workspace_size_per_rank,
                    num_experts_per_rank, hidden_size, max_tokens_per_rank,
                    dtype)] * ep_size),
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
        verify_combine_results(all_results, ep_size, all_num_tokens, top_k,
                               hidden_size, num_experts_per_rank, max_tokens_per_rank)


def run_moe_a2a_dispatch_moe_combine_single_rank(ep_size, all_num_tokens, top_k,
                                                 workspace_size_per_rank,
                                                 num_experts_per_rank,
                                                 hidden_size,
                                                 max_tokens_per_rank, dtype):
    """Worker function for dispatch and combine test."""
    rank = tllm.mpi_rank()
    torch.cuda.set_device(rank)
    device = torch.cuda.current_device()

    try:
        mapping = Mapping(rank=rank,
                          tp_size=ep_size,
                          moe_ep_size=ep_size,
                          world_size=ep_size)

        # Create MoeAlltoAll manager
        moe_a2a = MoeAlltoAll(mapping, max_tokens_per_rank, workspace_size_per_rank)

        rank_local_tokens = all_num_tokens[rank]

        # Generate test data - use simpler payload for combine test
        token_selected_experts = generate_token_selected_experts(
            rank_local_tokens, ep_size, num_experts_per_rank, top_k)

        payloads = make_bfloat16_payloads(rank_local_tokens, hidden_size, top_k,
                                          rank, token_selected_experts)

        # Run dispatch
        num_experts = ep_size * num_experts_per_rank
        recv_buffers, send_counters, send_indices = moe_a2a.dispatch(
            token_selected_experts, payloads, max_tokens_per_rank,
            top_k, num_experts)

        # TODO: remove this
        sync()


        hidden_states_recv = recv_buffers[
            0]  # [ep_size, max_tokens_per_rank, hidden_size]
        token_selected_experts_recv = recv_buffers[
            1]  # [ep_size, max_tokens_per_rank, top_k]
        token_final_scales_recv = recv_buffers[
            2]  # [ep_size, max_tokens_per_rank, top_k]

        ep_size = hidden_states_recv.shape[0]
        max_tokens_per_rank = hidden_states_recv.shape[1]

        # emulate MoE computation on the received data
        # Create experts for this rank
        rank_experts = create_experts(num_experts_per_rank, hidden_size, rank, device, dtype=torch.bfloat16)
        
        hidden_states_recv = fake_moe(
            hidden_states_recv.view(ep_size * max_tokens_per_rank, hidden_states_recv.shape[-1]),
            token_selected_experts_recv.view(ep_size * max_tokens_per_rank, token_selected_experts_recv.shape[-1]),
            token_final_scales_recv.view(ep_size * max_tokens_per_rank, token_final_scales_recv.shape[-1]),
            rank_experts,  # experts for current rank
            is_ep=True,
            ep_rank=rank,
            num_experts_per_rank=num_experts_per_rank).view(ep_size, max_tokens_per_rank, hidden_states_recv.shape[-1])

        # TODO: remove this
        sync()

        # Run combine on the received data
        combined_output = moe_a2a.combine(send_indices,
                                          hidden_states_recv,
                                          max_tokens_per_rank,
                                          top_k)

        # TODO: remove this
        sync()

        # Verify completion flags after combine
        completion_flags_offset = moe_a2a.moe_a2a_metainfo[MoeAlltoAll.COMPLETION_FLAGS_OFFSET_INDEX].item()
        completion_flags_ptr = moe_a2a.workspace[
            rank, completion_flags_offset:completion_flags_offset + ep_size * 4]
        completion_flags = completion_flags_ptr.view(torch.int32).cpu()
        flag_val_offset = moe_a2a.moe_a2a_metainfo[MoeAlltoAll.FLAG_VAL_OFFSET_INDEX].item()
        expected_flag_val = moe_a2a.workspace[
            rank, flag_val_offset:flag_val_offset + 4].view(torch.int32).cpu() - 1
        assert torch.all(completion_flags == expected_flag_val), (
            f"Rank {rank} completion flags: {completion_flags}, expected flag val: {expected_flag_val}"
        )

        # Return results for verification
        return (
            token_selected_experts.cpu(),
            [p.cpu() for p in payloads],  # Return actual payloads used
            send_indices.cpu(),
            combined_output.cpu(),
            rank_experts.cpu()  # Return the experts used on this rank
        )
    except Exception:
        traceback.print_exc()
        raise


def verify_combine_results(all_results, ep_size, all_num_tokens, top_k,
                           hidden_size, num_experts_per_rank, max_tokens_per_rank):
    """Verify that combine correctly sums the dispatched tokens."""

    # Extract results
    all_token_selected_experts = [r[0] for r in all_results]
    all_original_payloads = [r[1] for r in all_results]
    all_send_indices = [r[2] for r in all_results]
    all_combined_outputs = [r[3] for r in all_results]
    all_rank_experts = [r[4] for r in all_results]  # Extract experts from each rank

    # For each rank, verify the combined output
    for rank in range(ep_size):
        # print("### Verify rank %d ###" % rank)
        token_selected_experts = all_token_selected_experts[rank]
        original_payloads = all_original_payloads[rank]
        hidden_states = original_payloads[0]
        token_final_scales = original_payloads[2]

        send_indices = all_send_indices[rank]
        combined_output = all_combined_outputs[rank]

        local_num_tokens = all_num_tokens[rank]

        # Check send_indices
        for token_idx in range(local_num_tokens):
            unique_targets = set()
            for k in range(top_k):
                expert_id = token_selected_experts[token_idx, k].item()
                target_rank = compute_target_rank_id(expert_id, num_experts_per_rank)
                unique_targets.add(target_rank)
                assert send_indices[token_idx, target_rank].item() >= 0 and send_indices[token_idx, target_rank].item() < max_tokens_per_rank, "send_indices should be >= 0 and < max_tokens_per_rank"

            for target_rank in range(ep_size):
                if target_rank not in unique_targets:
                    assert send_indices[token_idx, target_rank].item() == -1, "send_indices should be -1"

        # Check the following are equal:
        # expected: Directly emulate MoE with all experts as if EP is not used.
        # actual: Tokens are dispatched to target ranks, MoE is performed on target ranks, and then the results from all target ranks are summed up (combine).
        
        # Gather all experts from all ranks for non-EP emulation
        all_experts = torch.cat(all_rank_experts, dim=0)        
        expected_combined_output = fake_moe(
            hidden_states, 
            token_selected_experts, 
            token_final_scales, 
            all_experts,
            is_ep=False)
        
        # Custom assertion with detailed error message
        try:
            torch.testing.assert_close(combined_output, expected_combined_output, rtol=5e-2, atol=5e-2)
        except AssertionError as e:
            # Find the first mismatch location
            abs_diff = (combined_output - expected_combined_output).abs()
            rel_diff = abs_diff / (expected_combined_output.abs() + 1e-8)
            
            # Check both absolute and relative tolerance
            mask = (abs_diff > 1e-2) & (rel_diff > 5e-2)
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
                        context_values_expected.append(f"{expected_combined_output[token_idx, idx].item():.4f}")
                        context_values_actual.append(f"{combined_output[token_idx, idx].item():.4f}")
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