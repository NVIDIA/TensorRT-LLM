# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eight MPI ranks: TP8 SA-1R and two-round sampling must be bit-identical."""

import gc
import json

import pytest
import torch
from _torch.modeling_v2.models.minimax_m3.comm._sampled_accept_tp8 import (
    _CALLS,
    _CASES,
    _DRAFT,
    _VS,
    _assert_collective,
    _assert_same,
    _logits,
    _snapshot,
    sa,
)

from tensorrt_llm._utils import mpi_comm, mpi_rank
from tensorrt_llm.mapping import Mapping


def test_comparison_failure_is_collective():
    left = torch.tensor([1], device="cuda")
    right = torch.tensor([2 if mpi_rank() == 3 else 1], device="cuda")
    with pytest.raises(AssertionError, match="exchange variants differ"):
        _assert_same((left,), (right,))
    mpi_comm().Barrier()


@pytest.mark.parametrize("rows,mode,top_p", _CASES)
def test_one_round_exchange_matches_two_rounds(monkeypatch, rows, mode, top_p):
    assert "one_round_k" in sa._op_args() and "decided" in sa._op_args()
    rank = mpi_rank()
    comm = mpi_comm()
    device = torch.device("cuda", torch.cuda.current_device())
    mapping = Mapping(world_size=8, rank=rank, gpus_per_node=torch.cuda.device_count(), tp_size=8)
    logits = _logits(rows, mode, rank, device)
    nreq = rows // (_DRAFT + 1)
    temperatures = torch.linspace(0.8, 1.2, rows, device=device)
    top_ks = torch.zeros(rows, dtype=torch.int32, device=device)
    top_ps = torch.full((rows,), top_p, device=device)
    seeds = torch.arange(rows, device=device, dtype=torch.int64) * 17 + 20261101
    base_offsets = torch.arange(rows, device=device, dtype=torch.int64)
    offsets = base_offsets.clone()
    draft = torch.zeros(nreq, _DRAFT, dtype=torch.int32, device=device)
    batch = torch.arange(nreq, dtype=torch.int32, device=device)
    masses = comm.allgather(torch.logsumexp(logits[0].float() / temperatures[0], 0).item())
    assert max(masses) - min(masses) > 0.5, "the test needs unequal shard masses"
    workspaces = {}
    try:
        for variant in (0, 82):
            monkeypatch.setattr(sa, "ONE_ROUND_K", variant)
            workspaces[variant] = sa.SampledAcceptWorkspace.create(mapping, device)
        warm = []
        for variant in (0, 82):
            monkeypatch.setattr(sa, "ONE_ROUND_K", variant)
            warm.append(
                _snapshot(
                    logits,
                    mapping,
                    draft,
                    batch,
                    temperatures,
                    top_ks,
                    top_ps,
                    seeds,
                    offsets,
                    workspaces[variant],
                )
            )
        torch.cuda.synchronize()
        _assert_same(*warm)
        # Reuse the exact Philox inputs, but exercise every strict-accept prefix.
        sampled = warm[0][-1].reshape(nreq, _DRAFT + 1)
        draft.copy_(sampled[:, :_DRAFT])
        prefixes = torch.arange(nreq, device=device) % (_DRAFT + 1)
        for request in range(nreq):
            prefix = request % (_DRAFT + 1)
            if prefix < _DRAFT:
                draft[request, prefix] = (draft[request, prefix] + 1) % (_VS * 8)
        observations = []
        for call in range(_CALLS):
            offsets.copy_(base_offsets + call * 1000)
            pair = []
            for variant in (0, 82):
                monkeypatch.setattr(sa, "ONE_ROUND_K", variant)
                pair.append(
                    _snapshot(
                        logits,
                        mapping,
                        draft,
                        batch,
                        temperatures,
                        top_ks,
                        top_ps,
                        seeds,
                        offsets,
                        workspaces[variant],
                    )
                )
            observations.append(pair)
        torch.cuda.synchronize()
        for call, pair in enumerate(observations):
            _assert_same(*pair)
            if call == 0:
                _assert_collective(
                    torch.equal(pair[0][1], (prefixes + 1).to(torch.int32)),
                    "strict-accept prefixes differ",
                )
            record = [tensor.cpu().tolist() for tensor in pair[0]]
            peers = comm.allgather(record)
            assert all(peer == peers[0] for peer in peers), "ranks disagree on sampled outputs"
        for ws in workspaces.values():
            _assert_collective(
                ws.counters.cpu().tolist() == [0, _CALLS + 1],
                "workspace counter did not exercise both halves",
            )
        decided = workspaces[82].decided.item()
        if mode == "spiked":
            _assert_collective(decided > 0, "the spiked case must exercise SA-1R decisions")
        if rank == 0:
            print(
                json.dumps(
                    dict(
                        rows=rows,
                        mode=mode,
                        top_p=top_p,
                        world=8,
                        calls_per_variant=_CALLS + 1,
                        both_halves=True,
                        unequal_log_masses=masses,
                        decided=decided,
                        all_outputs_bitwise=True,
                    )
                ),
                flush=True,
            )
    finally:
        torch.cuda.synchronize()
        comm.Barrier()
        # Release multicast memory while each workspace communicator is still live.
        for ws in workspaces.values():
            ws.uc = None
            ws.handle = None
        gc.collect()
        comm.Barrier()
        for ws in workspaces.values():
            if ws.comm is not None:
                ws.comm.Free()
