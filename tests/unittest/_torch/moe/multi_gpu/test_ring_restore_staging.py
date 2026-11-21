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
"""Experts restored from another node's host weights arrive bit-exactly through the workspace.

Four processes share a real MNNVL MoE All-to-All workspace, and each counts as a node of its own.
The last rank fails. The node before it, rank 2, keeps its experts' host weights, so ranks 0 and 1
load them through the workspace rows that the survivors remapped with fresh pages, in chunks of
half a row. In ``writing`` mode the failed rank keeps dispatching into the survivors' rows.
"""

import functools
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from mnnvl_workers_test_utils import (
    EP_SIZE,
    TIMEOUT,
    VICTIM,
    StoreComm,
    bound_state,
    map_workspace,
    run_victim,
    run_workers,
)

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.moe.fused_moe.eplb_expert_restore import (
    DeploymentHostWeights,
    EplbExpertRestore,
    staging_pieces,
)
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import SingleLayerMoeLoadBalancer

_LAYERS = 3
# Two experts per rank and two spare slots.
_PLACEMENT = [[2 * rank, 2 * rank + 1, -1, -1] for rank in range(EP_SIZE)]
# Per slot: w1 32 MiB of bytes and w2 8 MiB of bfloat16, so that each target takes several chunks.
_WEIGHTS = {"w1": (32 << 20, torch.uint8), "w2": (4 << 20, torch.bfloat16)}


def _host_weight(layer_idx: int, expert: int, name: str) -> torch.Tensor:
    count, dtype = _WEIGHTS[name]
    generator = torch.Generator().manual_seed(layer_idx * 1000 + expert * 10 + len(name))
    data = torch.randint(0, 256, (count * dtype.itemsize,), dtype=torch.uint8, generator=generator)
    return data.view(dtype)


def _expert_restore(rank: int) -> EplbExpertRestore:
    """Return the expert restore of ``rank``, with its slots and the host weights its node keeps."""
    balancer = SimpleNamespace(
        ep_rank=rank,
        ep_size=EP_SIZE,
        layer_updates_per_iter=0,
        mapping=SimpleNamespace(get_node_rank=lambda r: r),
        in_iter=False,
        single_layer_load_balancers=[],
        load_balancer_impl=Mock(),
    )
    restore = EplbExpertRestore(balancer)
    for layer_idx in range(_LAYERS):
        impl = Mock(
            get_layer_id=Mock(return_value=layer_idx),
            get_old_rank_expert_ids=Mock(return_value=_PLACEMENT),
        )
        layer = SingleLayerMoeLoadBalancer(
            impl,
            Mock(Get_rank=Mock(return_value=0), Get_size=Mock(return_value=1)),
            2 * EP_SIZE,
            updates_enabled=False,
            total_slot_count_per_rank=len(_PLACEMENT[0]),
            spare_slot_count=2,
            host_tensor_sharer_cls=functools.partial(
                DeploymentHostWeights,
                replacement=False,
                next_node_ranks=restore.next_node_ranks(),
            ),
        )
        layer.set_initial_weight_assignments([expert for row in _PLACEMENT for expert in row])
        sharer = layer.host_tensor_sharer
        sharer.keep_next_node_experts(np.array(_PLACEMENT))
        for expert in sharer.expert_ids:
            for name in _WEIGHTS:
                sharer.node_host_weights[(expert, name)] = _host_weight(layer_idx, expert, name)
        for name, (count, dtype) in _WEIGHTS.items():
            layer.slot_tensors[name] = torch.zeros(
                (len(_PLACEMENT[0]), count), dtype=dtype, device="cuda"
            )
        balancer.single_layer_load_balancers.append(layer)
    return restore


def _survivor(store, memory, workspace, meta, rank: int) -> None:
    """Remap the workspace with the other survivors and restore the failed rank's experts."""
    state = bound_state(memory, workspace, meta, rank)
    restore = _expert_restore(rank)
    store.set(f"captured/{rank}", "1")
    store.wait(["proceed"])

    gather = StoreComm(store, rank, "survivors", range(VICTIM))
    state.ep_group_health.mark_failed(VICTIM)
    state.remap_workspaces(gather, frozenset(range(EP_SIZE)))
    active = [r != VICTIM for r in range(EP_SIZE)]
    plan = restore.plan_expert_restore(active, [np.array(_PLACEMENT)] * _LAYERS)
    # Ranks 0 and 1 stage experts 6 and 7, whose host weights only rank 2 keeps.
    staged = {(target, source) for _, target, _, _, source in plan.staged.tolist()}
    assert staged == {(0, 2), (1, 2)}
    staging = state.staging_workspace()
    weight_bytes = [[count * dtype.itemsize for count, dtype in _WEIGHTS.values()]] * _LAYERS
    # Several chunks, so the copies alternate between the halves of a row.
    assert staging_pieces(plan.staged, weight_bytes, staging.shape[1])[-1].chunk >= 1
    restore.restore_lost_experts(plan, staging, gather)
    torch.cuda.synchronize()
    for layer_idx, layer in enumerate(restore.balancer.single_layer_load_balancers):
        for slot, expert in enumerate(plan.placements[layer_idx][rank].tolist()):
            if expert < 0 or _PLACEMENT[rank][slot] == expert:
                continue
            for name, tensor in layer.slot_tensors.items():
                loaded = tensor[slot].view(torch.uint8).cpu()
                expected = _host_weight(layer_idx, expert, name).view(torch.uint8)
                assert torch.equal(loaded, expected), f"rank {rank} layer {layer_idx} slot {slot}"
    state.rearm()
    gather(None)


def _worker(rank: int, port: int, mode: str) -> None:
    store, memory, workspace, meta = map_workspace(rank, port)
    if rank != VICTIM:
        _survivor(store, memory, workspace, meta, rank)
    elif mode == "writing":
        run_victim(store, workspace, meta)
    else:
        store.wait([f"captured/{survivor}" for survivor in range(VICTIM)])
        store.set("running", "1")
        store.wait(["done"])


@pytest.mark.parametrize("mode", ["idle", "writing"])
def test_staged_restore_is_bit_exact(mode: str) -> None:
    """Check that the restored slots hold their experts' host weights byte for byte."""
    with run_workers(__file__, mode) as (store, workers):
        *survivors, victim = workers
        store.set("proceed", "1")
        for survivor in survivors:
            assert survivor.wait(timeout=TIMEOUT.total_seconds()) == 0
        assert victim.poll() is None, "the victim exited before the survivors finished"


if __name__ == "__main__":
    _worker(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
