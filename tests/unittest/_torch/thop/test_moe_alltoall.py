# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import unittest

import pytest
import torch
from parameterized import parameterized

import tensorrt_llm as tllm

has_setup_max_sm_count = False


class TestMoeAlltoAllSingleGPU(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0x1234)
        tllm.logger.set_level('error')

    @parameterized.expand([
        (902, 701, 32768, 100, torch.float16),
        (902, 701, 32768, 100, torch.bfloat16),
        (902, 701, 32768, 100, torch.float),
        (902, 701, 7168, 100, torch.float16),
        (902, 701, 7168, 100, torch.bfloat16),
        (902, 701, 7168, 100, torch.float),
        (101, 75, 288, 10, torch.float16),
        (101, 75, 288, 10, torch.bfloat16),
        (101, 75, 288, 10, torch.float),
        (10, 5, 8, 1, torch.float16),
        (10, 5, 8, 1, torch.bfloat16),
        (10, 5, 8, 1, torch.float),
    ])
    def test_moe_alltoall_single_gpu(self, input_entry_count,
                                     output_entry_count, vector_dim,
                                     send_recv_count, dtype):
        torch.cuda.set_device(0)
        # Create a random input tensor
        input_tensor = torch.randn(input_entry_count,
                                   vector_dim,
                                   dtype=dtype,
                                   device=torch.device('cuda'))
        output_tensor = torch.zeros(output_entry_count,
                                    vector_dim,
                                    dtype=dtype,
                                    device=torch.device('cuda'))

        send_cumsum = torch.ones(
            (1, ), dtype=torch.int32,
            device=torch.device('cuda')) * send_recv_count
        recv_cumsum = torch.ones(
            (1, ), dtype=torch.int32,
            device=torch.device('cuda')) * send_recv_count
        send_indices = torch.randperm(
            input_entry_count, dtype=torch.int32,
            device=torch.device('cuda'))[:send_recv_count]
        recv_indices = torch.randperm(
            output_entry_count, dtype=torch.int32,
            device=torch.device('cuda'))[:send_recv_count]

        ref_output_tensor = torch.zeros(output_entry_count,
                                        vector_dim,
                                        dtype=dtype,
                                        device=torch.device('cuda'))
        ref_output_tensor[recv_indices] = input_tensor[send_indices]

        workspace_size = torch.ops.trtllm.get_moe_commworkspace_size_per_rank(1)
        all_workspaces = torch.zeros(1,
                                     workspace_size // 8,
                                     dtype=torch.uint64,
                                     device=torch.device('cuda'))
        torch.ops.trtllm.moe_initialize_workspace(all_workspaces, 0, 1)

        torch.ops.trtllm.moe_comm([input_tensor], send_cumsum, send_indices,
                                  [output_tensor], recv_cumsum, recv_indices,
                                  all_workspaces, 0, 1)

        torch.testing.assert_close(output_tensor,
                                   ref_output_tensor,
                                   atol=1e-5,
                                   rtol=1e-5)

    def do_warmup(self):
        torch.cuda.synchronize()
        input_tensor = torch.randn(1,
                                   8,
                                   dtype=torch.float16,
                                   device=torch.device('cuda'))
        send_cumsum = torch.ones(1,
                                 dtype=torch.int32,
                                 device=torch.device('cuda'))
        send_indices = torch.zeros(1,
                                   dtype=torch.int32,
                                   device=torch.device('cuda'))
        output_tensor = torch.zeros(1,
                                    8,
                                    dtype=torch.float16,
                                    device=torch.device('cuda'))
        recv_cumsum = torch.ones(1,
                                 dtype=torch.int32,
                                 device=torch.device('cuda'))
        recv_indices = torch.zeros(1,
                                   dtype=torch.int32,
                                   device=torch.device('cuda'))
        input_tensors = [input_tensor]
        output_tensors = [output_tensor]
        workspace_size = torch.ops.trtllm.get_moe_commworkspace_size_per_rank(1)
        all_workspaces = torch.zeros(1,
                                     workspace_size // 8,
                                     dtype=torch.uint64,
                                     device=torch.device('cuda'))
        torch.ops.trtllm.moe_comm(input_tensors, send_cumsum, send_indices,
                                  output_tensors, recv_cumsum, recv_indices,
                                  all_workspaces, 0, 1)
        torch.cuda.synchronize()

    @parameterized.expand([
        (2, 5, [4, 4], torch.float16),  # small input as smoke test
        (2, 1, [8], torch.float16),  # some ranks have no data to send/recv
        (4, 5, [8], torch.float16),  # small input with larger world size
        (4, 901, [1472, 46, 4,
                  4], torch.float16),  # large input that reuses workspace
        (4, 5, [2944], torch.bfloat16),  # large input that reuses workspace
        (8, 901, [
            32768,
        ],
         torch.float16),  # large input that reuses workspace, larger world size
        (
            8, 16384, [
                128,
            ], torch.float16
        ),  # large input count with small vector dim that requires more indices per fifo
    ])
    def test_moe_alltoall_multi_rank_single_gpu(self, world_size,
                                                input_entry_per_rank,
                                                vector_dims, dtype):
        torch.cuda.set_device(0)
        max_world_size = 8
        assert world_size <= max_world_size, f"should run with world_size at most {max_world_size}"

        global has_setup_max_sm_count
        if not has_setup_max_sm_count:
            sm_count = torch.cuda.get_device_properties(0).multi_processor_count
            max_sm_count = sm_count // max_world_size  # we use single gpu to test multiple gpu communication
            torch.ops.trtllm.set_moe_max_usable_sm_count(max_sm_count)
            has_setup_max_sm_count = True

        tensor_count = len(vector_dims)
        input_tensors = []
        output_tensors = []
        ref_output_tensors = []
        for vector_dim in vector_dims:
            input_tensors.append(
                torch.randn(input_entry_per_rank * world_size,
                            vector_dim,
                            dtype=dtype,
                            device=torch.device('cuda')))
            output_tensors.append(
                torch.zeros(input_entry_per_rank * world_size,
                            vector_dim,
                            dtype=dtype,
                            device=torch.device('cuda')))
            ref_output_tensors.append(
                torch.zeros(input_entry_per_rank * world_size,
                            vector_dim,
                            dtype=dtype,
                            device=torch.device('cuda')))

        target_rank_ids = torch.randint(0,
                                        world_size,
                                        (input_entry_per_rank * world_size, ),
                                        dtype=torch.int32,
                                        device=torch.device('cuda'))

        input_tensors_all_ranks = []
        for i in range(tensor_count):
            input_tensors_all_ranks.append(
                list(torch.split(input_tensors[i], input_entry_per_rank)))

        target_rank_ids_all_ranks = list(
            torch.split(target_rank_ids, input_entry_per_rank))

        send_ids_all_ranks = []
        send_counts_all_ranks = []
        send_cumsum_all_ranks = []
        send_start_end_all_ranks = []

        # each rank do its own local compute to get how to send data to other ranks.
        for rank in range(world_size):
            send_start_end = []
            local_target_rank_ids = target_rank_ids_all_ranks[rank]
            sorted_local_target_rank_ids, local_send_id = torch.sort(
                local_target_rank_ids)
            local_send_id = local_send_id.to(torch.int32)
            padded_sorted_local_target_rank_ids = torch.cat(
                (sorted_local_target_rank_ids,
                 torch.arange(world_size,
                              dtype=torch.int32,
                              device=torch.device('cuda'))))
            unique_target_rank_ids, local_send_counts = torch.unique(
                padded_sorted_local_target_rank_ids, return_counts=True)
            local_send_counts = local_send_counts.to(torch.int32)
            assert unique_target_rank_ids.numel(
            ) == world_size, "unique_target_rank_ids must be equal to world_size"
            local_send_counts -= 1  # remove padding
            local_send_cumsum = torch.cumsum(local_send_counts,
                                             dim=0).to(torch.int32)
            send_ids_all_ranks.append(local_send_id)
            send_counts_all_ranks.append(local_send_counts)
            send_cumsum_all_ranks.append(local_send_cumsum)
            local_send_cumsum_cpu = local_send_cumsum.cpu().tolist()
            for i in range(len(local_send_cumsum_cpu)):
                send_start_end.append(
                    (local_send_cumsum_cpu[i - 1] if i > 0 else 0,
                     local_send_cumsum_cpu[i]))
            send_start_end_all_ranks.append(send_start_end)

        recv_ids_all_ranks = []
        recv_cumsum_all_ranks = []

        # Initialize output_tensors_all_ranks as a list of lists
        output_tensors_all_ranks = [[] for _ in range(tensor_count)]

        total_recv_all_ranks_cpu = []
        output_indice_offset = 0

        output_start_current_rank = 0
        # each rank do compute based on other ranks' send counts to get how to receive data from other ranks.
        for rank in range(world_size):
            local_recv_counts = torch.zeros(world_size,
                                            dtype=torch.int32,
                                            device=torch.device('cuda'))
            for other_rank in range(world_size):
                local_recv_counts[other_rank] = send_counts_all_ranks[
                    other_rank][rank]
                local_recv_count_pair = local_recv_counts[other_rank].cpu(
                ).item()
                send_rank_start_end = send_start_end_all_ranks[other_rank][rank]
                for i in range(tensor_count):
                    ref_output_tensors[i][output_indice_offset:output_indice_offset + local_recv_count_pair] = \
                        input_tensors_all_ranks[i][other_rank][send_ids_all_ranks[other_rank][send_rank_start_end[0]:send_rank_start_end[1]]]
                output_indice_offset += local_recv_count_pair
            local_recv_cumsum = torch.cumsum(local_recv_counts,
                                             dim=0).to(torch.int32)
            recv_cumsum_all_ranks.append(local_recv_cumsum)
            total_recv_count = local_recv_cumsum[-1].cpu()
            total_recv_all_ranks_cpu.append(total_recv_count)
            # Correctly append output tensors for each tensor index
            for i in range(tensor_count):
                output_tensors_all_ranks[i].append(
                    output_tensors[i]
                    [output_start_current_rank:output_start_current_rank +
                     total_recv_count])
            output_start_current_rank += total_recv_count
            local_recv_ids = torch.arange(total_recv_count,
                                          dtype=torch.int32,
                                          device=torch.device('cuda'))
            recv_ids_all_ranks.append(local_recv_ids)

        cuda_streams_all_ranks = [
            torch.cuda.Stream() for _ in range(world_size)
        ]

        workspace_size = torch.ops.trtllm.get_moe_commworkspace_size_per_rank(
            world_size)
        all_workspaces = torch.zeros(world_size,
                                     workspace_size // 8,
                                     dtype=torch.uint64,
                                     device=torch.device('cuda'))
        for i in range(world_size):
            torch.ops.trtllm.moe_initialize_workspace(all_workspaces, i,
                                                      world_size)

        # do one warmup for each rank to avoid possible synchronization at first launch.
        for rank in range(world_size):
            with torch.cuda.stream(cuda_streams_all_ranks[rank]):
                self.do_warmup()

        torch.cuda.synchronize()

        # do alltoall in parallel
        for rank in range(world_size):
            input_tensors_this_rank = [
                input_tensors_all_ranks[i][rank] for i in range(tensor_count)
            ]
            output_tensors_this_rank = [
                output_tensors_all_ranks[i][rank] for i in range(tensor_count)
            ]
            with torch.cuda.stream(cuda_streams_all_ranks[rank]):
                torch.ops.trtllm.moe_comm(
                    input_tensors_this_rank, send_cumsum_all_ranks[rank],
                    send_ids_all_ranks[rank], output_tensors_this_rank,
                    recv_cumsum_all_ranks[rank], recv_ids_all_ranks[rank],
                    all_workspaces, rank, world_size)
        for rank in range(world_size):
            cuda_streams_all_ranks[rank].synchronize()

        for i in range(tensor_count):
            # Concatenate output tensors from all ranks for comparison
            actual_output = torch.cat(output_tensors_all_ranks[i], dim=0)
            torch.testing.assert_close(actual_output,
                                       ref_output_tensors[i],
                                       atol=1e-5,
                                       rtol=1e-5)


class TestMoeAlltoAllFP8SingleGPU(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0x1234)
        tllm.logger.set_level('error')

    def test_moe_alltoall_fp8_with_indices(self):
        """Test fp8 alltoall with properly constructed indices"""
        torch.cuda.set_device(0)

        # Match dimensions from the error
        input_entry_count = 16384
        output_entry_count = 16384
        vector_dim = 2944
        sf_vector_dim = 92  # Scaling factor dimension from error
        send_recv_count = 1000  # Number of entries to send/receive

        # Create input tensors - first as float16, then convert
        input_tensor_fp16 = torch.randn(input_entry_count,
                                        vector_dim,
                                        dtype=torch.float16,
                                        device='cuda')
        input_tensor_fp8 = input_tensor_fp16.to(torch.float8_e4m3fn)

        # Scaling factor tensor
        input_sf_tensor = torch.randint(1,
                                        255, (input_entry_count, sf_vector_dim),
                                        dtype=torch.uint8,
                                        device='cuda')

        # Expert selection tensors
        input_experts = torch.randint(0,
                                      64, (input_entry_count, 4),
                                      dtype=torch.int32,
                                      device='cuda')
        input_scales = torch.rand(input_entry_count,
                                  4,
                                  dtype=torch.float32,
                                  device='cuda')

        # Output tensors
        output_tensor_fp8 = torch.zeros(output_entry_count,
                                        vector_dim,
                                        dtype=torch.float8_e4m3fn,
                                        device='cuda')
        output_sf_tensor = torch.zeros(output_entry_count,
                                       sf_vector_dim,
                                       dtype=torch.uint8,
                                       device='cuda')
        output_experts = torch.zeros(output_entry_count,
                                     4,
                                     dtype=torch.int32,
                                     device='cuda')
        output_scales = torch.zeros(output_entry_count,
                                    4,
                                    dtype=torch.float32,
                                    device='cuda')

        # Construct send/recv indices
        send_cumsum = torch.tensor([send_recv_count],
                                   dtype=torch.int32,
                                   device='cuda')
        recv_cumsum = torch.tensor([send_recv_count],
                                   dtype=torch.int32,
                                   device='cuda')

        # Random indices for sending
        send_indices = torch.randperm(input_entry_count,
                                      dtype=torch.int32,
                                      device='cuda')[:send_recv_count]
        recv_indices = torch.randperm(output_entry_count,
                                      dtype=torch.int32,
                                      device='cuda')[:send_recv_count]

        # Create workspace
        workspace_size = torch.ops.trtllm.get_moe_commworkspace_size_per_rank(1)
        all_workspaces = torch.zeros(1,
                                     workspace_size // 8,
                                     dtype=torch.uint64,
                                     device='cuda')
        torch.ops.trtllm.moe_initialize_workspace(all_workspaces, 0, 1)

        print(f"Test configuration:")
        print(f"  Input entries: {input_entry_count}")
        print(f"  Vector dim: {vector_dim}")
        print(f"  SF vector dim: {sf_vector_dim}")
        print(f"  Send/recv count: {send_recv_count}")
        print(f"  FP8 tensor shape: {input_tensor_fp8.shape}")
        print(f"  SF tensor shape: {input_sf_tensor.shape}")

        try:
            # Test with all 4 tensors
            torch.ops.trtllm.moe_comm([
                input_tensor_fp8, input_sf_tensor, input_experts, input_scales
            ], send_cumsum, send_indices, [
                output_tensor_fp8, output_sf_tensor, output_experts,
                output_scales
            ], recv_cumsum, recv_indices, all_workspaces, 0, 1)

            torch.cuda.synchronize()
            print("FP8 alltoall test PASSED!")

            # Verify outputs
            print(f"\nOutput verification:")
            print(f"  Output FP8 shape: {output_tensor_fp8.shape}")
            print(f"  Output SF shape: {output_sf_tensor.shape}")
            print(
                f"  Non-zero FP8 elements: {(output_tensor_fp8 != 0).sum().item()}"
            )
            print(
                f"  Non-zero SF elements: {(output_sf_tensor != 0).sum().item()}"
            )

        except Exception as e:
            print(f"FP8 alltoall test FAILED: {e}")
            print(f"Error type: {type(e)}")
            raise

    @parameterized.expand([
        (0, 8, 256, 4, 3, False),
        (0, 8, 256, 4, 3, True),
        (1, 8, 256, 4, 3, False),
        (1, 8, 256, 4, 3, True),
        (1, 4, 256, 8, 3, False),
        (1, 4, 256, 8, 3, True),
        (7, 8, 256, 8, 1025, False),
        (7, 8, 256, 8, 1025, True),
        (7, 64, 1024, 32, 1029, False),
        (7, 64, 1024, 32, 1029, True),
    ])
    def test_moe_alltoall_prepare_indices(
            self, ep_rank: int, ep_size: int, expert_count: int, top_k: int,
            max_token_count_per_rank: int,
            use_real_rank_token_count_cumsum: bool):
        torch.cuda.set_device(0)
        gathered_target_rank_ids = torch.randint(
            0,
            ep_size, (ep_size * max_token_count_per_rank, top_k),
            dtype=torch.int32,
            device=torch.device('cuda'))
        real_rank_token_count_cumsum = None
        if use_real_rank_token_count_cumsum:
            real_rank_token_count_cumsum = torch.randint(
                0,
                max_token_count_per_rank + 1, (ep_size, ),
                dtype=torch.int32,
                device=torch.device('cuda'))
            real_rank_token_count_cumsum = torch.cumsum(
                real_rank_token_count_cumsum, dim=0).to(torch.int32)

        def generate_references():
            gathered_target_rank_ids_cpu_lists = gathered_target_rank_ids.cpu(
            ).tolist()
            if use_real_rank_token_count_cumsum:
                real_rank_token_count_cumsum_cpu_lists = real_rank_token_count_cumsum.cpu(
                ).tolist()
            else:
                real_rank_token_count_cumsum_cpu_lists = [
                    (i + 1) * max_token_count_per_rank for i in range(ep_size)
                ]
            rank_token_start = 0
            ref_local_gather_indices_cpu_lists = []
            ref_recv_rank_count_cumsum_cpu_lists = [0] * ep_size
            ref_recv_rank_local_indices_cpu_lists = []
            ref_send_rank_count_cumsum_cpu_lists = [0] * ep_size
            ref_send_rank_local_indices_cpu_lists = []
            ref_backward_recv_rank_local_indices_cpu_lists = []
            total_recv_count = 0
            for rank in range(ep_size):
                rank_token_end = real_rank_token_count_cumsum_cpu_lists[rank]
                for token_id in range(rank_token_start, rank_token_end):
                    if ep_rank in gathered_target_rank_ids_cpu_lists[token_id]:
                        ref_local_gather_indices_cpu_lists.append(token_id)
                        ref_recv_rank_local_indices_cpu_lists.append(
                            total_recv_count)
                        total_recv_count += 1
                ref_recv_rank_count_cumsum_cpu_lists[rank] = total_recv_count
                if rank == ep_rank:
                    total_send_count = 0
                    for target_rank in range(ep_size):
                        for token_id in range(rank_token_start, rank_token_end):
                            local_token_id = token_id - rank_token_start
                            if target_rank in gathered_target_rank_ids_cpu_lists[
                                    token_id]:
                                pos = gathered_target_rank_ids_cpu_lists[
                                    token_id].index(target_rank)
                                ref_send_rank_local_indices_cpu_lists.append(
                                    local_token_id)
                                ref_backward_recv_rank_local_indices_cpu_lists.append(
                                    local_token_id * top_k + pos)
                                total_send_count += 1
                        ref_send_rank_count_cumsum_cpu_lists[
                            target_rank] = total_send_count
                rank_token_start = rank_token_end
            ref_local_gather_indices = torch.IntTensor(
                ref_local_gather_indices_cpu_lists).cuda()
            ref_send_rank_count_cumsum = torch.IntTensor(
                ref_send_rank_count_cumsum_cpu_lists).cuda()
            ref_send_rank_local_indices = torch.IntTensor(
                ref_send_rank_local_indices_cpu_lists).cuda()
            ref_recv_rank_count_cumsum = torch.IntTensor(
                ref_recv_rank_count_cumsum_cpu_lists).cuda()
            ref_recv_rank_local_indices = torch.IntTensor(
                ref_recv_rank_local_indices_cpu_lists).cuda()
            ref_backward_recv_rank_local_indices = torch.IntTensor(
                ref_backward_recv_rank_local_indices_cpu_lists).cuda()
            return ref_local_gather_indices, ref_send_rank_count_cumsum, ref_send_rank_local_indices, ref_recv_rank_count_cumsum, ref_recv_rank_local_indices, ref_backward_recv_rank_local_indices

        ref_local_gather_indices, ref_send_rank_count_cumsum, ref_send_rank_local_indices, ref_recv_rank_count_cumsum, ref_recv_rank_local_indices, ref_backward_recv_rank_local_indices = generate_references(
        )

        local_gather_indices, send_rank_count_cumsum, send_rank_local_indices, recv_rank_count_cumsum, recv_rank_local_indices, backward_recv_rank_local_indices = \
            torch.ops.trtllm.moe_comm_prepare_indices(gathered_target_rank_ids, real_rank_token_count_cumsum, max_token_count_per_rank, expert_count, top_k, ep_rank, ep_size)

        assert torch.equal(
            local_gather_indices[:torch.numel(ref_local_gather_indices)],
            ref_local_gather_indices)
        assert torch.equal(
            send_rank_count_cumsum[:torch.numel(ref_send_rank_count_cumsum)],
            ref_send_rank_count_cumsum)
        assert torch.equal(
            send_rank_local_indices[:torch.numel(ref_send_rank_local_indices)],
            ref_send_rank_local_indices)
        assert torch.equal(
            recv_rank_count_cumsum[:torch.numel(ref_recv_rank_count_cumsum)],
            ref_recv_rank_count_cumsum)
        assert torch.equal(
            recv_rank_local_indices[:torch.numel(ref_recv_rank_local_indices)],
            ref_recv_rank_local_indices)
        assert torch.equal(
            backward_recv_rank_local_indices[:torch.numel(
                ref_backward_recv_rank_local_indices)],
            ref_backward_recv_rank_local_indices)

    @parameterized.expand([
        (0, 8, 256, 4, 3),
        (1, 8, 256, 4, 3),
        (7, 8, 256, 4, 3),
        (7, 8, 256, 8, 32),
        (7, 8, 256, 32, 10),
        (7, 8, 1024, 32, 127),
        (7, 64, 1024, 32, 1029),
        (9, 64, 1024, 3, 1029),
    ])
    def test_moe_local_gather(self, ep_rank: int, ep_size: int,
                              expert_count: int, top_k: int,
                              max_token_count_per_rank: int):
        torch.cuda.set_device(0)
        rank_token_count_cumsum = torch.randint(0,
                                                max_token_count_per_rank + 1,
                                                (ep_size, ),
                                                dtype=torch.int32,
                                                device=torch.device('cuda'))
        rank_token_count_cumsum = torch.cumsum(rank_token_count_cumsum,
                                               dim=0).to(torch.int32)
        local_token_count = rank_token_count_cumsum[ep_size - 1].cpu().item()
        local_max_token_count = max_token_count_per_rank * ep_size
        local_gather_indices = torch.randint(0,
                                             max_token_count_per_rank * ep_size,
                                             (local_max_token_count, ),
                                             dtype=torch.int32,
                                             device=torch.device('cuda'))

        gathered_expert_ids = torch.randint(
            0,
            expert_count, (max_token_count_per_rank * ep_size, top_k),
            dtype=torch.int32,
            device=torch.device('cuda'))
        gathered_scales = torch.rand(
            (max_token_count_per_rank * ep_size, top_k),
            dtype=torch.float32,
            device=torch.device('cuda'))

        ref_local_expert_ids = torch.zeros(local_max_token_count,
                                           top_k,
                                           dtype=torch.int32,
                                           device=torch.device('cuda'))
        ref_local_scales = torch.zeros(local_max_token_count,
                                       top_k,
                                       dtype=torch.float32,
                                       device=torch.device('cuda'))

        # compute reference
        ref_local_expert_ids += expert_count
        valid_local_gather_indices = local_gather_indices[:local_token_count]
        ref_local_expert_ids[:local_token_count] = gathered_expert_ids[
            valid_local_gather_indices]
        ref_local_scales[:local_token_count] = gathered_scales[
            valid_local_gather_indices]

        local_expert_ids = torch.empty(local_max_token_count,
                                       top_k,
                                       dtype=torch.int32,
                                       device=torch.device('cuda'))
        local_scales = torch.empty(local_max_token_count,
                                   top_k,
                                   dtype=torch.float32,
                                   device=torch.device('cuda'))

        torch.ops.trtllm.moe_local_gather(rank_token_count_cumsum,
                                          local_gather_indices,
                                          gathered_expert_ids, gathered_scales,
                                          local_expert_ids, local_scales,
                                          max_token_count_per_rank,
                                          expert_count, top_k, ep_rank, ep_size)

        assert torch.equal(local_expert_ids, ref_local_expert_ids)
        assert torch.equal(local_scales, ref_local_scales)

    @parameterized.expand([
        (0, 2, 16, 20, 8, 512),
        (0, 2, 16, 16, 3, 300),
        (0, 4, 20, 24, 8, 4000),
        (0, 8, 96, 96, 8, 1000),
        (3, 8, 128, 128, 8, 1000),
        (3, 8, 128, 144, 8, 1),
        (0, 4, 72, 80, 4, 2256),
        (0, 4, 72, 80, 6, 3333),
        # Hang with stream count > 8
        #(0, 9, 90, 8, 100),
    ])
    @pytest.mark.no_xdist
    def test_moe_alltoall_prepare(self, ep_rank: int, ep_size: int,
                                  expert_count: int, slot_count: int,
                                  top_k: int, max_token_count_per_rank: int):
        torch.cuda.set_device(0)

        cpu_expert_ids_all_ranks_lists = []
        cpu_token_count_lists = []
        cpu_scales_all_ranks_lists = []
        for _ in range(ep_size):
            token_count = torch.randint(max_token_count_per_rank // 2,
                                        max_token_count_per_rank + 1, (1, ),
                                        dtype=torch.int32,
                                        device=torch.device('cpu'))
            token_count = 1 if token_count == 0 else token_count

            token_count = max_token_count_per_rank

            cpu_expert_ids_all_ranks_lists.append(
                torch.randint(0,
                              slot_count, (token_count, top_k),
                              dtype=torch.int32,
                              device=torch.device('cpu')))

            cpu_scales_all_ranks_lists.append(
                torch.zeros(token_count,
                            top_k,
                            dtype=torch.float32,
                            device=torch.device('cpu')) + 0.5)

            cpu_token_count_lists.append(token_count)

        def compute_target_rank(expert_id):
            ep_per_rank = slot_count // ep_size
            return expert_id // ep_per_rank

        def generate_references():
            ref_prepared_local_expert_ids = []
            ref_prepared_local_scales = []
            ref_local_send_rank_count_cumsum = [0] * ep_size
            ref_local_recv_rank_count_cumsum = [0] * ep_size
            ref_local_recv_rank_indices = []

            local_token_count = cpu_token_count_lists[ep_rank]
            send_token_count_to_ranks = [0] * ep_size

            # send part
            for token_id in range(local_token_count):
                target_set = set()
                for pos in range(top_k):
                    expert_id = int(
                        cpu_expert_ids_all_ranks_lists[ep_rank][token_id][pos])
                    target_rank_id = compute_target_rank(expert_id)
                    target_set.add(target_rank_id)

                for target_rank_id in target_set:
                    send_token_count_to_ranks[target_rank_id] += 1

            total_send_token_count = 0
            for rank in range(ep_size):
                #print(f'rank: {rank}, send_token_count_to_ranks[rank]: {send_token_count_to_ranks[rank]}')
                base = ref_local_send_rank_count_cumsum[rank -
                                                        1] if rank > 0 else 0
                ref_local_send_rank_count_cumsum[
                    rank] = send_token_count_to_ranks[rank] + base
                total_send_token_count += send_token_count_to_ranks[rank]

            ref_local_backward_send_rank_indices = [0
                                                    ] * (total_send_token_count)
            ref_local_send_rank_indices = [0] * (total_send_token_count)

            current_send_token_ids = [0] * ep_size
            for token_id in range(local_token_count):
                target_set = set()
                for pos in range(top_k):
                    expert_id = int(
                        cpu_expert_ids_all_ranks_lists[ep_rank][token_id][pos])
                    target_rank_id = compute_target_rank(expert_id)
                    if target_rank_id not in target_set:
                        cumsum_before = 0 if target_rank_id == 0 else ref_local_send_rank_count_cumsum[
                            target_rank_id - 1]
                        send_index = cumsum_before + current_send_token_ids[
                            target_rank_id]
                        ref_local_send_rank_indices[send_index] = token_id
                        ref_local_backward_send_rank_indices[
                            send_index] = token_id * top_k + pos
                        current_send_token_ids[target_rank_id] += 1
                        target_set.add(target_rank_id)

            # receive part
            total_recv_token_count = 0
            for rank in range(ep_size):
                token_count = cpu_token_count_lists[rank]
                current_recv_token_count = 0
                for token_id in range(token_count):
                    token_is_received = False
                    for pos in range(top_k):
                        expert_id = int(
                            cpu_expert_ids_all_ranks_lists[rank][token_id][pos])
                        sf = cpu_scales_all_ranks_lists[rank][token_id][pos]
                        target_rank_id = compute_target_rank(expert_id)
                        if target_rank_id == ep_rank:
                            if not token_is_received:
                                token_is_received = True
                                ref_prepared_local_expert_ids.append(
                                    [slot_count] * top_k)
                                ref_prepared_local_scales.append([0.0] * top_k)
                            ref_prepared_local_expert_ids[-1][pos] = expert_id
                            ref_prepared_local_scales[-1][pos] = sf
                    if token_is_received:
                        ref_local_recv_rank_indices.append(
                            total_recv_token_count)
                        total_recv_token_count += 1
                        current_recv_token_count += 1
                ref_local_recv_rank_count_cumsum[
                    rank] = current_recv_token_count if rank == 0 else ref_local_recv_rank_count_cumsum[
                        rank - 1] + current_recv_token_count

            return ref_prepared_local_expert_ids, ref_prepared_local_scales, ref_local_send_rank_count_cumsum, ref_local_send_rank_indices, ref_local_recv_rank_count_cumsum, ref_local_recv_rank_indices, ref_local_backward_send_rank_indices, total_recv_token_count

        ref_prepared_local_expert_ids, ref_prepared_local_scales, ref_local_send_rank_count_cumsum, ref_local_send_rank_indices, ref_local_recv_rank_count_cumsum, ref_local_recv_rank_indices, ref_local_backward_send_rank_indices, total_recv_token_count = generate_references(
        )

        cpu_experter_count_lists = []
        for rank in range(ep_size):
            local_expert_count = []
            for i in range(expert_count):
                local_expert_count.append(rank * expert_count + i)
            cpu_experter_count_lists.append(torch.IntTensor(local_expert_count))

        #expert_ids_all_ranks = torch.tensor(cpu_expert_ids_all_ranks_lists).cuda()
        expert_ids_all_ranks = [
            cpu_expert_ids_all_ranks_lists[i].cuda() for i in range(ep_size)
        ]
        #scales_all_ranks = torch.FloatTensor(cpu_scales_all_ranks_lists).cuda()
        scales_all_ranks = [
            cpu_scales_all_ranks_lists[i].cuda() for i in range(ep_size)
        ]

        experter_count_lists = [
            cpu_experter_count_lists[i].cuda() for i in range(ep_size)
        ]

        cuda_streams_all_ranks = [torch.cuda.Stream() for _ in range(ep_size)]

        workspace_size = torch.ops.trtllm.get_moe_prepare_workspace_size_per_rank(
            ep_size)

        all_workspaces = torch.zeros(ep_size,
                                     workspace_size,
                                     dtype=torch.uint64,
                                     device=torch.device('cuda'))

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.ops.trtllm.mnnvl_moe_alltoallv_prepare_without_allgather(
                expert_ids_all_ranks[0], scales_all_ranks[0],
                experter_count_lists[0], all_workspaces,
                max_token_count_per_rank, 0, 1, expert_count, slot_count, top_k)
        stream.wait_stream(torch.cuda.current_stream())

        # Make torch alloc tensor to avoid cuda sync
        prepared_local_experts = []
        prepared_local_scales = []
        local_send_rank_count_cumsum = []
        local_send_rank_indices = []
        local_recv_rank_count_cumsum = []
        local_recv_rank_indices = []
        backward_local_recv_rank_indices = []
        for _ in range(ep_size):
            prepared_local_experts.append(
                torch.empty(max_token_count_per_rank * ep_size,
                            top_k,
                            dtype=torch.int32,
                            device=torch.device('cuda')))
            prepared_local_scales.append(
                torch.empty(max_token_count_per_rank * ep_size,
                            top_k,
                            dtype=torch.float32,
                            device=torch.device('cuda')))
            local_send_rank_count_cumsum.append(
                torch.empty(ep_size,
                            dtype=torch.int32,
                            device=torch.device('cuda')))
            local_send_rank_indices.append(
                torch.empty(max_token_count_per_rank * ep_size,
                            dtype=torch.int32,
                            device=torch.device('cuda')))
            local_recv_rank_count_cumsum.append(
                torch.empty(0, dtype=torch.int32, device=torch.device('cuda')))
            local_recv_rank_indices.append(
                torch.empty(0, dtype=torch.int32, device=torch.device('cuda')))
            backward_local_recv_rank_indices.append(
                torch.empty(0, dtype=torch.int32, device=torch.device('cuda')))

        prepared_local_experts = []
        prepared_local_scales = []
        local_send_rank_count_cumsum = []
        local_send_rank_indices = []
        local_recv_rank_count_cumsum = []
        local_recv_rank_indices = []
        backward_local_recv_rank_indices = []

        # reset the workspace
        all_workspaces = torch.zeros(ep_size,
                                     workspace_size,
                                     dtype=torch.uint64,
                                     device=torch.device('cuda'))

        # do prepare in parallel
        for rank in range(ep_size):
            with torch.cuda.stream(cuda_streams_all_ranks[rank]):
                if rank == ep_rank:
                    prepared_local_experts, prepared_local_scales, local_send_rank_count_cumsum, \
                    local_send_rank_indices, local_recv_rank_count_cumsum, local_recv_rank_indices, \
                    backward_local_recv_rank_indices, gathered_expert_statics\
                        = torch.ops.trtllm.mnnvl_moe_alltoallv_prepare_without_allgather(expert_ids_all_ranks[rank], scales_all_ranks[rank], experter_count_lists[rank], all_workspaces, max_token_count_per_rank,
                                                                                         rank, ep_size, expert_count, slot_count, top_k)
                else:
                    torch.ops.trtllm.mnnvl_moe_alltoallv_prepare_without_allgather(
                        expert_ids_all_ranks[rank], scales_all_ranks[rank],
                        experter_count_lists[rank], all_workspaces,
                        max_token_count_per_rank, rank, ep_size, expert_count,
                        slot_count, top_k)
        for rank in range(ep_size):
            cuda_streams_all_ranks[rank].synchronize()

        prepared_local_experts_cpu = prepared_local_experts[:
                                                            total_recv_token_count].cpu(
                                                            )
        prepared_local_scales_cpu = prepared_local_scales[:
                                                          total_recv_token_count].cpu(
                                                          )
        for i in range(total_recv_token_count):
            for j in range(top_k):
                expert_id = int(prepared_local_experts_cpu[i][j])
                assert 0 <= expert_id and expert_id <= slot_count
                if expert_id < slot_count:
                    assert compute_target_rank(expert_id) == ep_rank
                    scale = float(prepared_local_scales_cpu[i][j])
                    assert scale > 1e-6

        gathered_expert_statics_cpu = gathered_expert_statics.cpu()
        for rank in range(ep_size):
            for i in range(expert_count):
                assert int(gathered_expert_statics_cpu[rank]
                           [i]) == rank * expert_count + i

        ref_local_send_rank_count_cumsum = torch.IntTensor(
            ref_local_send_rank_count_cumsum)
        assert torch.equal(local_send_rank_count_cumsum.cpu(),
                           ref_local_send_rank_count_cumsum)

        local_send_rank_indices = local_send_rank_indices.cpu()
        backward_local_recv_rank_indices = backward_local_recv_rank_indices.cpu(
        )
        for i in range(ep_size):
            base = 0 if i == 0 else ref_local_send_rank_count_cumsum[i - 1]
            for j in range(base, ref_local_send_rank_count_cumsum[i]):
                token_id = local_send_rank_indices[j]
                lane_id = backward_local_recv_rank_indices[j] - token_id * top_k
                expert_id = int(
                    cpu_expert_ids_all_ranks_lists[ep_rank][token_id][lane_id])
                assert compute_target_rank(expert_id) == i

        ref_local_recv_rank_count_cumsum = torch.IntTensor(
            ref_local_recv_rank_count_cumsum)
        assert torch.equal(
            local_recv_rank_count_cumsum[:ref_local_recv_rank_count_cumsum.
                                         size(0)].cpu(),
            ref_local_recv_rank_count_cumsum)

        ref_local_recv_rank_indices = torch.IntTensor(
            ref_local_recv_rank_indices)
        assert torch.equal(
            local_recv_rank_indices[:ref_local_recv_rank_indices.size(0)].cpu(),
            ref_local_recv_rank_indices)


if __name__ == "__main__":
    unittest.main()
