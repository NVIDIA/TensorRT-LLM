# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Four-rank regression for alternating quantized/unquantized A2A payloads."""

import pytest
import torch
from mpi4py import MPI

import tensorrt_llm as tllm
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm.mapping import Mapping


@pytest.mark.parametrize("use_cft", [False, True])
@pytest.mark.parametrize("low_precision", [False, True])
@pytest.mark.parametrize(
    "capture,in_workspace", [(False, False), (False, True), (True, False), (True, True)]
)
def test_mixed_dispatch_layout_preserves_previous_combine(
    capture, in_workspace, use_cft, low_precision
):
    rank = tllm.mpi_rank()
    assert tllm.mpi_world_size() == 4
    torch.cuda.set_device(rank)
    hidden, capacity = 6144, 32768
    mapping = Mapping(
        world_size=4, rank=rank, tp_size=4, moe_ep_size=4, gpus_per_node=4, enable_attention_dp=True
    )
    first = NVLinkOneSided(
        mapping,
        256,
        8,
        capacity,
        payload_in_workspace=in_workspace,
        hidden_size=hidden,
        dtype=torch.bfloat16,
        can_use_cft_counted_writes=use_cft,
        use_low_precision_combine=low_precision,
    )
    second = NVLinkOneSided(
        mapping,
        256,
        8,
        capacity,
        payload_in_workspace=in_workspace,
        hidden_size=hidden,
        dtype=torch.bfloat16,
        can_use_cft_counted_writes=use_cft,
        use_low_precision_combine=low_precision,
    )
    assert first.workspace.data_ptr() == second.workspace.data_ptr()
    assert first.use_cft_for_dispatch(32) == use_cft
    assert first.use_cft_for_combine(32) == use_cft
    ids = torch.tensor([0, 64, 128, 192, 1, 65, 129, 193], dtype=torch.int32, device="cuda").repeat(
        capacity, 1
    )
    scales = torch.full((capacity, 8), 0.125, device="cuda")
    small = torch.full((capacity, hidden // 2), 85, dtype=torch.uint8, device="cuda")
    sf = torch.ones((capacity, hidden // 16), dtype=torch.uint8, device="cuda")
    big = torch.full((capacity, hidden), float("nan"), dtype=torch.bfloat16, device="cuda")
    expert = torch.full(
        (4 * capacity, hidden), float(rank + 1), dtype=torch.bfloat16, device="cuda"
    )
    lengths_list = (
        [266, 9, 19034, 9],
        [266, 9, 19034, 9],
        [0, 1, 32, 0],
        [9, 0, capacity, 1],
        [1, 9, 64, 0],
        [1, 9, 64, 0],
    )

    def rounds():
        checks = []
        for i, lengths in enumerate(lengths_list):
            count, maximum = lengths[rank], max(lengths)
            comm = first if i % 2 == 0 else second
            comm.dispatch(
                small[:count] if i % 2 == 0 else big[:count],
                sf[:count] if i % 2 == 0 else None,
                ids[:count],
                scales[:count],
                lengths,
            )
            if in_workspace:
                payload = comm.get_combine_payload_tensor_in_workspace(
                    maximum, hidden, torch.bfloat16
                )
                payload.fill_(rank + 1)
            else:
                payload = expert[: 4 * maximum]
            if rank == 2:
                torch.cuda._sleep(200000)
            output = comm.combine(payload)
            checks.append(output.eq(10).all())
        return torch.stack(checks)

    # Initialize kernels before capture; no host synchronization inside rounds.
    rounds()
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    records = []
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            checks = rounds()
        for _ in range(20):
            graph.replay()
            records.append(checks.clone())
    else:
        for _ in range(20):
            records.append(rounds())
    valid = torch.stack(records).cpu()
    failures = (~valid).nonzero().tolist()
    MPI.COMM_WORLD.Barrier()
    first.destroy()
    second.destroy()
    assert not failures, (rank, capture, in_workspace, use_cft, low_precision, failures)
