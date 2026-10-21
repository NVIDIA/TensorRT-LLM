# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the fc fork's counter wait (modeling's DI_FC_FLAG, ops/m3_fc_wait): the one-thread wait kernel
queued on the main stream releases its successor only after the fork's bump on another stream (so the successor reads
what the fork wrote, with no stream join before it), eagerly and on CUDA-graph replays; it counts an overrun instead
of hanging when no bump comes."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp8.ops import (  # noqa: E501
    m3_fc_wait,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

N = 1 << 16


def _wait(counts, spins=m3_fc_wait.MAX_SPINS):
    m3_fc_wait.fc_wait_kernel[(1,)](counts, MAX_SPINS=spins, num_warps=1, launch_pdl=True)


def _step(counts, aux, src, dst, value, delay_cycles):
    """One fork / wait / consume: the fork (after a delay) writes ``value`` into src and bumps the counter; the main
    stream waits on the counter, copies src to dst, then joins the fork (deferred, as the worker does)."""
    main = torch.cuda.current_stream()
    aux.wait_stream(main)
    with torch.cuda.stream(aux):
        torch.cuda._sleep(delay_cycles)
        src.fill_(value)
        counts[:1].add_(1)
    _wait(counts)
    dst.copy_(src)
    main.wait_stream(aux)


@pytest.mark.parametrize("delay_cycles", [0, 200_000, 2_000_000])
def test_wait_releases_after_the_fork(delay_cycles) -> None:
    counts = torch.zeros(3, dtype=torch.int64, device="cuda")
    src = torch.zeros(N, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    aux = torch.cuda.Stream()
    for it in range(1, 9):
        _step(counts, aux, src, dst, float(it), delay_cycles)
        torch.cuda.synchronize()
        assert bool((dst == float(it)).all()), f"iteration {it}: read before the fork finished"
    assert counts.tolist() == [8, 8, 0]


def test_wait_releases_on_graph_replays() -> None:
    counts = torch.zeros(3, dtype=torch.int64, device="cuda")
    src = torch.zeros(N, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    val = torch.zeros(1, dtype=torch.float32, device="cuda")
    aux = torch.cuda.Stream()
    cap = torch.cuda.Stream()

    def body():
        main = torch.cuda.current_stream()
        aux.wait_stream(main)
        with torch.cuda.stream(aux):
            torch.cuda._sleep(1_000_000)
            src.copy_(val.expand(N))
            counts[:1].add_(1)
        _wait(counts)
        dst.copy_(src)
        main.wait_stream(aux)

    with torch.cuda.stream(cap):
        body()  # eager warm-up (compiles the wait kernel outside capture)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(cap):
        with torch.cuda.graph(graph, stream=cap):
            body()
    torch.cuda.synchronize()
    base = counts.tolist()
    for it in range(1, 11):
        val.fill_(float(it))
        graph.replay()
        torch.cuda.synchronize()
        assert bool((dst == float(it)).all()), f"replay {it}: read before the fork finished"
    assert counts.tolist() == [base[0] + 10, base[1] + 10, 0]


def test_wait_without_a_fork_counts_an_overrun() -> None:
    counts = torch.zeros(3, dtype=torch.int64, device="cuda")
    _wait(counts, spins=1000)
    torch.cuda.synchronize()
    assert counts.tolist() == [0, 1, 1]
