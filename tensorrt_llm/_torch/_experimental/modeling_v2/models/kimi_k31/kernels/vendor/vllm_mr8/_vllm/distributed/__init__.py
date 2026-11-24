# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vllm.distributed stand-in: the TP group the replica runs on (set by ``set_tp_group``)."""
import torch.distributed as dist


class GroupCoordinator:
    def __init__(self, group, device):
        self.device_group = group
        self.world_size = dist.get_world_size(group)
        self.rank_in_group = dist.get_rank(group)
        self.rank = dist.get_rank()
        self.device = device
        self.unique_name = "tp:0"


_TP = None


def set_tp_group(group, device):
    global _TP
    _TP = GroupCoordinator(group, device)
    return _TP


def get_tp_group():
    if _TP is None:
        raise RuntimeError("replica: TP group not set (replica_chain.install)")
    return _TP


def get_dcp_group():
    return get_tp_group()


def tensor_model_parallel_all_reduce(x):
    dist.all_reduce(x, group=get_tp_group().device_group)
    return x
