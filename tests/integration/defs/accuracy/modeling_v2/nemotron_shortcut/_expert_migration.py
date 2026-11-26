# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify physical weight migration and routing on a real expert-parallel group."""

import uuid

import torch
from mpi4py import MPI

from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.expert_placement import (
    ExpertPlacement,
)
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    MoeLoadBalancer,
    MoeLoadBalancerIterContext,
)
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig


def verify_hot_expert_migration():
    comm = MPI.COMM_WORLD
    assert comm.size == 4 and torch.cuda.device_count() >= comm.size
    rank = comm.rank
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (10, 3)
    initial = [0, 1, 0, 2, 3, 1, 4, 5, 2, 6, 7, 3]
    config = MoeLoadBalancerConfig(
        num_slots=len(initial), layer_updates_per_iter=1, initial_global_assignments={0: initial}
    )
    config.setup(ep_rank=rank, ep_size=comm.size)
    name = comm.bcast("shortcut_eplb_test_" + uuid.uuid4().hex if rank == 0 else None, root=0)
    manager = MoeLoadBalancer(rank, comm.size, 1, shared_memory_base_name=name)
    try:
        placement = ExpertPlacement(config, manager, 0, 8, 1)
        # Both planes encode the logical expert independently of its physical slot.
        source = torch.arange(8, dtype=torch.bfloat16).view(-1, 1, 1)
        source = source + torch.arange(128, dtype=torch.bfloat16).view(1, -1, 1)
        source = source.expand(-1, -1, 256)
        initial_ids = placement.initial_local_expert_ids
        banks = torch.nn.ParameterDict()
        host = {}
        for plane, offset in (("up", 1), ("down", 17)):
            # A last-dimension TP window has a pitch larger than its logical row width.
            values = (source + offset)[:, :, 64:192]
            banks["experts_" + plane] = torch.nn.Parameter(
                values[initial_ids].cuda(), requires_grad=False
            )
            host[plane] = values
        # Mixed precision stores one calibration scalar per logical expert.
        # These individual checkpoint tensors must migrate with the GEMM matrices.
        scale = torch.arange(8, dtype=torch.float32) + 0.25
        banks["experts_up_input_scale"] = torch.nn.Parameter(
            scale[initial_ids].cuda(), requires_grad=False
        )
        host["up_input_scale"] = list(scale.unbind())
        placement.bind_weights(banks, "", host)
        manager.register_weight_slots_after_to_cuda()
        manager.finalize_model()
        ids = torch.full((128, 1), 7, dtype=torch.int32, device="cuda")
        correct = torch.ones((), dtype=torch.bool, device="cuda")
        used = torch.zeros(len(initial) + 1, dtype=torch.bool, device="cuda")
        for _ in range(16):
            with MoeLoadBalancerIterContext(manager, enable_statistic=True, enable_updates=True):
                slots = placement.route(ids).flatten().long()
                local = slots - config.slot_start
                owns = (local >= 0) & (local < config.num_local_slots)
                index = local.clamp(0, config.num_local_slots - 1)
                correct &= (
                    ~owns
                    | (
                        (banks["experts_up"][index, 1, 0] == 9)
                        & (banks["experts_down"][index, 1, 0] == 25)
                        & (banks["experts_up_input_scale"][index] == 7.25)
                    )
                ).all()
                used.scatter_(0, torch.where(owns, slots, len(initial)), True)
                placement.finish()
        assert comm.allreduce(bool(correct.item()), op=MPI.LAND)
        seen = set().union(*comm.allgather(set(used[:-1].nonzero().flatten().tolist())))
        assert len(seen) > initial.count(7), f"No new physical slot served expert 7: {seen}"
    finally:
        manager.shutdown()


if __name__ == "__main__":
    verify_hot_expert_migration()
