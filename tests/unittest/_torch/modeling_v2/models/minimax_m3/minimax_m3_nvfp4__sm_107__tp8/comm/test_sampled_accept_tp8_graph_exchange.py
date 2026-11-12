# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare eight-rank captured SA-1R with captured and eager two-round sampling."""

import gc
import json

import pytest
import torch
from _torch.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp8.comm._sampled_accept_tp8 import (
    _CASES,
    _DRAFT,
    _assert_collective,
    _assert_same,
    _logits,
    _snapshot,
    sa,
)

from tensorrt_llm._utils import mpi_comm, mpi_rank
from tensorrt_llm.mapping import Mapping


@pytest.mark.parametrize("rows,mode,top_p", _CASES)
def test_graph_exchange_matches_eager_two_rounds(monkeypatch, rows, mode, top_p):
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
    workspaces = {}
    graphs = {}
    try:
        for key, variant in (("reference", 0), (0, 0), (82, 82)):
            monkeypatch.setattr(sa, "ONE_ROUND_K", variant)
            ws = workspaces[key] = sa.SampledAcceptWorkspace.create(mapping, device)
            _snapshot(
                logits, mapping, draft, batch, temperatures, top_ks, top_ps, seeds, offsets, ws
            )
        torch.cuda.synchronize()
        for variant in (0, 82):
            monkeypatch.setattr(sa, "ONE_ROUND_K", variant)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
                captured = _snapshot(
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
            torch.cuda.current_stream().wait_stream(stream)
            graphs[variant] = (graph, captured)
        torch.cuda.synchronize()
        comm.Barrier()
        observations = []
        decisions = []
        for replay in range(4):
            offsets.copy_(base_offsets + (replay + 20) * 1000)
            monkeypatch.setattr(sa, "ONE_ROUND_K", 0)
            reference = _snapshot(
                logits,
                mapping,
                draft,
                batch,
                temperatures,
                top_ks,
                top_ps,
                seeds,
                offsets,
                workspaces["reference"],
            )
            captured_outputs = []
            for variant in (0, 82):
                graph, captured = graphs[variant]
                graph.replay()
                captured_outputs.append(tuple(t.clone() for t in captured))
                if variant == 82:
                    decisions.append(workspaces[82].decided.clone())
            observations.append((reference, *captured_outputs))
        torch.cuda.synchronize()
        for reference, captured_zero, captured_one in observations:
            _assert_same(reference, captured_zero)
            _assert_same(reference, captured_one)
            peers = comm.allgather([t.cpu().tolist() for t in captured_one])
            assert all(peer == peers[0] for peer in peers), "captured ranks disagree"
        for ws in workspaces.values():
            _assert_collective(
                ws.counters.cpu().tolist() == [0, 5], "graph workspace counter mismatch"
            )
        decided = sum(t.item() for t in decisions)
        if mode == "spiked":
            _assert_collective(decided > 0, "captured SA-1R must make round-zero decisions")
        if rank == 0:
            print(
                json.dumps(
                    dict(
                        rows=rows,
                        mode=mode,
                        top_p=top_p,
                        world=8,
                        graph_replays=4,
                        both_halves=True,
                        decided=decided,
                        eager_vs_graph_bitwise=True,
                    )
                ),
                flush=True,
            )
    finally:
        torch.cuda.synchronize()
        comm.Barrier()
        graphs.clear()
        gc.collect()
        for ws in workspaces.values():
            ws.uc = None
            ws.handle = None
        gc.collect()
        comm.Barrier()
        for ws in workspaces.values():
            if ws.comm is not None:
                ws.comm.Free()
