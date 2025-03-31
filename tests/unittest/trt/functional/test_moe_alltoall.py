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

# isort: off
import torch
# isort: on

from parameterized import parameterized

import tensorrt_llm as tllm


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
                                     workspace_size,
                                     dtype=torch.uint64,
                                     device=torch.device('cuda'))

        torch.ops.trtllm.moe_comm(input_tensor, send_cumsum, send_indices,
                                  output_tensor, recv_cumsum, recv_indices,
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
        workspace_size = torch.ops.trtllm.get_moe_commworkspace_size_per_rank(1)
        all_workspaces = torch.zeros(1,
                                     workspace_size,
                                     dtype=torch.uint64,
                                     device=torch.device('cuda'))
        torch.ops.trtllm.moe_comm(input_tensor, send_cumsum, send_indices,
                                  output_tensor, recv_cumsum, recv_indices,
                                  all_workspaces, 0, 1)
        torch.cuda.synchronize()

    @parameterized.expand([
        (2, 5, 8, torch.float16),  # small input as smoke test
        (2, 1, 8, torch.float16),  # some ranks have no data to send/recv
        (4, 5, 8, torch.float16),  # small input with larger world size
        (4, 901, 32768, torch.bfloat16),  # large input that reuses workspace
        (8, 901, 32768,
         torch.float16),  # large input that reuses workspace, larger world size
        (
            8, 16384, 128, torch.float16
        ),  # large input count with small vector dim that requires more indices per fifo
    ])
    def test_moe_alltoall_multi_rank_single_gpu(self, world_size,
                                                input_entry_per_rank,
                                                vector_dim, dtype):
        torch.cuda.set_device(0)
        # Create a random input tensor
        input_tensor = torch.randn(input_entry_per_rank * world_size,
                                   vector_dim,
                                   dtype=dtype,
                                   device=torch.device('cuda'))
        output_tensor = torch.zeros(input_entry_per_rank * world_size,
                                    vector_dim,
                                    dtype=dtype,
                                    device=torch.device('cuda'))
        ref_output_tensor = torch.zeros(input_entry_per_rank * world_size,
                                        vector_dim,
                                        dtype=dtype,
                                        device=torch.device('cuda'))
        target_rank_ids = torch.randint(0,
                                        world_size,
                                        (input_entry_per_rank * world_size, ),
                                        dtype=torch.int32,
                                        device=torch.device('cuda'))

        input_tensors_all_ranks = list(
            torch.split(input_tensor, input_entry_per_rank))
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

        output_tensors_all_ranks = []

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
                ref_output_tensor[output_indice_offset:output_indice_offset + local_recv_count_pair] = \
                    input_tensors_all_ranks[other_rank][send_ids_all_ranks[other_rank][send_rank_start_end[0]:send_rank_start_end[1]]]
                output_indice_offset += local_recv_count_pair
            local_recv_cumsum = torch.cumsum(local_recv_counts,
                                             dim=0).to(torch.int32)
            recv_cumsum_all_ranks.append(local_recv_cumsum)
            total_recv_count = local_recv_cumsum[-1].cpu()
            total_recv_all_ranks_cpu.append(total_recv_count)
            output_tensors_all_ranks.append(output_tensor[
                output_start_current_rank:output_start_current_rank +
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
                                     workspace_size,
                                     dtype=torch.uint64,
                                     device=torch.device('cuda'))

        # do one warmup for each rank to avoid possible synchronization at first launch.
        for rank in range(world_size):
            with torch.cuda.stream(cuda_streams_all_ranks[rank]):
                self.do_warmup()

        torch.cuda.synchronize()

        # do alltoall in parallel
        for rank in range(world_size):
            with torch.cuda.stream(cuda_streams_all_ranks[rank]):
                torch.ops.trtllm.moe_comm(
                    input_tensors_all_ranks[rank], send_cumsum_all_ranks[rank],
                    send_ids_all_ranks[rank], output_tensors_all_ranks[rank],
                    recv_cumsum_all_ranks[rank], recv_ids_all_ranks[rank],
                    all_workspaces, rank, world_size)
        for rank in range(world_size):
            cuda_streams_all_ranks[rank].synchronize()

        torch.testing.assert_close(output_tensor,
                                   ref_output_tensor,
                                   atol=1e-5,
                                   rtol=1e-5)

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


if __name__ == "__main__":
    unittest.main()
