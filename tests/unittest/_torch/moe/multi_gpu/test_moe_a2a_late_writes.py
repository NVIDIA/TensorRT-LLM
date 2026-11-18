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
"""A failed EP peer's late writes land only on pages the survivors no longer map.

Four processes share a real MNNVL MoE All-to-All workspace. The victim, the last rank, keeps
dispatching into the survivors' rows. The survivors remap the workspace with fresh pages at the same
addresses, rearm, and replay CUDA graphs captured before the failure.
"""

import sys
import time

import pytest
import torch
from mnnvl_workers_test_utils import (
    CAPACITY,
    EP_SIZE,
    HIDDEN,
    TIMEOUT,
    VICTIM,
    StoreComm,
    bound_state,
    controls,
    dispatch,
    map_workspace,
    mask_words,
    run_victim,
    run_workers,
    victim_steps,
)
from torch.distributed import TCPStore

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.distributed.membership_mnnvl_memory import MembershipMnnvlMemory

_GRACE_S = 2.0
# The negative control polls the active rows this often, at most this long, for the victim's writes.
_POLL_S, _CONTROL_DEADLINE_S = 0.05, 10.0


def _rows(source: int) -> torch.Tensor:
    """Return distinct [CAPACITY, HIDDEN] bf16 rows of ``source``.

    Columns 0-2 encode the token and the source exactly, so a misplaced row reads as wrong values.
    """
    token = torch.arange(CAPACITY, device="cuda")[:, None]
    values = (token + torch.arange(HIDDEN, device="cuda")) % 128
    values[:, 0:1] = token % 128
    values[:, 1:2] = token // 128
    values[:, 2] = source
    return values.bfloat16()


def _decode_tokens(rows: torch.Tensor, step: int) -> torch.Tensor:
    """Return the tokens of received ``_rows(source) + step`` rows, read from their columns 0-1."""
    return ((rows[:, 0].float() - step) + 128 * (rows[:, 1].float() - step)).long()


def _combine(received, offset, workspace, meta, rank, mask, abort):
    args = (received, CAPACITY, workspace, meta, CAPACITY, rank, EP_SIZE, 1)
    return torch.ops.trtllm.moe_a2a_combine(*args, offset, False, **controls(mask, abort))


def _check_late_writes(store: TCPStore, row: torch.Tensor, active: torch.Tensor, mode: str) -> None:
    """Check that no byte of this survivor's ``row`` changes for _GRACE_S while the victim writes.

    In ``no-remap`` mode, check instead that the victim's writes reach ``active``, which is source
    0's rows of ``row``.
    """
    torch.cuda.synchronize()
    before = row.clone()
    steps = victim_steps(store)
    # The victim alternates two patterns, so a single late look could miss a rewritten byte.
    if mode == "no-remap":
        start = active.data_ptr() - row.data_ptr()
        end = start + active.numel() * active.element_size()
        deadline = time.monotonic() + _CONTROL_DEADLINE_S
        while not bool((row[start:end] != before[start:end]).any()):
            assert time.monotonic() < deadline, (
                "the victim's writes missed the survivor's active rows without the remap"
            )
            time.sleep(_POLL_S)
        return
    deadline = time.monotonic() + _GRACE_S
    while True:
        changed = (row != before).nonzero().flatten()
        assert changed.numel() == 0, f"bytes of the remapped row changed: {changed[:16]}"
        if time.monotonic() >= deadline:
            break
        time.sleep(_POLL_S)
    if mode == "writing":
        assert victim_steps(store) > steps, "the victim stopped writing"


def _survivor(
    store: TCPStore,
    memory: MembershipMnnvlMemory,
    workspace: torch.Tensor,
    meta: torch.Tensor,
    rank: int,
    mode: str,
) -> None:
    """Remap with the other survivors and replay a graph whose tokens go to every survivor."""
    row = workspace[rank]
    state = bound_state(memory, workspace, meta, rank)
    mask, abort = state.active_rank_mask, state.abort
    token = torch.arange(CAPACITY, device="cuda")
    # Every source sends token t to survivor t % VICTIM.
    experts = (token % VICTIM).int()[:, None]
    payload = torch.empty((CAPACITY, HIDDEN), dtype=torch.bfloat16, device="cuda")

    # Capture with full membership and replay only after the victim failed.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        received, offset, _ = dispatch(
            workspace, meta, rank, experts, [payload], CAPACITY, mask, abort
        )
        output = _combine(received[0], offset, workspace, meta, rank, mask, abort)
    torch.cuda.synchronize()
    store.set(f"captured/{rank}", "1")
    store.wait(["proceed"])

    gather = StoreComm(store, rank, "survivors", range(VICTIM))
    state.ep_group_health.mark_failed(VICTIM)
    if mode != "no-remap":
        state.remap_workspaces(gather, frozenset(range(EP_SIZE)))
        assert memory.mapped
    state.rearm()
    state.install_mask()
    assert mask.cpu().tolist() == mask_words(range(VICTIM))
    # No survivor writes into a peer's row before that peer has rearmed or while it checks its row.
    gather(None)
    _check_late_writes(store, row, received[0][0], mode)
    gather(None)
    if mode == "no-remap":
        return

    mine = token[token % VICTIM == rank]
    steps = victim_steps(store)
    for step in range(4):
        payload.copy_(_rows(rank) + step)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
        # Each survivor's tokens arrive once each in this rank's fresh pages, in dispatch order.
        for source in range(VICTIM):
            rows = received[0][source][: mine.numel()]
            tokens = _decode_tokens(rows, step)
            assert torch.equal(tokens.sort().values, mine), f"tokens from source {source}"
            assert torch.equal(rows, (_rows(source) + step)[tokens]), f"rows from source {source}"
        # The failed victim's rows keep the zeros of the reset.
        assert torch.count_nonzero(received[0][VICTIM]) == 0
        assert abort[:2].tolist() == [0, 0]
        # No survivor dispatches the next round into this rank's rows while it reads them.
        gather(None)
    if mode == "writing":
        assert victim_steps(store) > steps, "the victim stopped writing during the replays"


def _worker(rank: int, port: int, mode: str) -> None:
    store, memory, workspace, meta = map_workspace(rank, port)
    if rank == VICTIM:
        run_victim(store, workspace, meta)
    else:
        _survivor(store, memory, workspace, meta, rank, mode)


@pytest.mark.parametrize(
    "mode, kill_delay_s",
    [("killed", 0.003), ("writing", None), ("no-remap", None)],
    ids=["killed-victim", "writing-victim", "no-remap-control"],
)
def test_late_writes_land_on_orphaned_pages(mode: str, kill_delay_s: float | None) -> None:
    """Check that the survivors' results stay exact and none of their bytes change after the remap.

    The victim is killed mid-dispatch (``killed``) or keeps writing (``writing``). In the
    ``no-remap`` control, the victim's writes must reach the survivors' active rows.
    """
    with run_workers(__file__, mode) as (store, workers):
        *survivors, victim = workers
        if kill_delay_s is not None:
            time.sleep(kill_delay_s)
            victim.kill()
            victim.wait()
        store.set("proceed", "1")
        for survivor in survivors:
            assert survivor.wait(timeout=TIMEOUT.total_seconds()) == 0


if __name__ == "__main__":
    _worker(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
