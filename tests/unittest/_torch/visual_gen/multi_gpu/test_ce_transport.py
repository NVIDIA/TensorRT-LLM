# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from ._worker_pool import run_distributed


def _exchange_and_recover(rank: int, world_size: int) -> None:
    from tensorrt_llm._torch.visual_gen.parallel import _wait_many, ce_transport
    from tensorrt_llm._torch.visual_gen.parallel.ce_transport import CeTransport

    device = torch.device("cuda", rank)
    with patch.object(
        ce_transport, "_peer_atomics_error", return_value="unavailable" if rank == 0 else ""
    ):
        assert CeTransport.create(dist.group.WORLD, device, min_channels=3) is None
    # Exercise both ranks after a warmup failure on rank 0, then repeat with fused waits.
    for fail_warmup, nflags in ((True, 4), (False, 34)):
        with (
            patch.object(_wait_many, "warm_wait_many", side_effect=RuntimeError("setup failed"))
            if fail_warmup and rank == 0
            else nullcontext()
        ):
            transport = CeTransport.create(dist.group.WORLD, device, min_channels=nflags + 1)
        assert transport is not None
        _exercise_transport(transport, nflags, expect_fused=not fail_warmup)


def _exercise_transport(transport, nflags: int, *, expect_fused: bool) -> None:
    from tensorrt_llm._torch.visual_gen.parallel import _wait_many

    pool = transport.pool
    rank, device = pool.rank, pool.device
    peer = 1 - rank
    graph = None
    try:
        pool.reserve(pool.world_size * pool.align, slots=1, min_channels=nflags + 1)
        incoming = pool.region(0, peer, 0, nflags)
        outgoing = pool.peer_region(peer, 0, 0, nflags)
        values = torch.arange(nflags, device=device, dtype=torch.uint8) + rank * 10
        result = torch.empty_like(values)
        signals = [(peer, channel) for channel in range(nflags)]

        def exchange(first: int, fused_expected: bool) -> None:
            with transport.boundary(expected=signals):
                for channel in range(nflags):
                    transport.push(
                        channel,
                        lambda dst, i=channel: outgoing[i : i + 1].copy_(values[i : i + 1]),
                    )
                transport.wait_ready(peer, first)
                with (
                    patch.object(
                        _wait_many, "wait_signal_words", wraps=_wait_many.wait_signal_words
                    ) as fused,
                    patch.object(pool, "wait", wraps=pool.wait) as native,
                ):
                    transport.wait_many(signals)
                assert fused.call_count == int(fused_expected)
                assert native.call_count == (0 if fused_expected else nflags - 1)
                result.copy_(incoming)

        if not expect_fused:
            exchange(0, False)
            assert result.tolist() == [peer * 10 + i for i in range(nflags)]
            return

        # Only rank 0 leaves a posted signal. The next exchange reuses that same flag.
        with pytest.raises(RuntimeError, match="interrupted"):
            with transport.boundary(expected=[(peer, 0)]):
                if rank == 0:
                    transport.push(0, lambda dst: outgoing[:1].fill_(255))
                raise RuntimeError("interrupted")
        assert transport.ready()

        def fail_after_copy(dst: int) -> None:
            outgoing[:1].fill_(240 + rank)
            raise RuntimeError("callback interrupted")

        with pytest.raises(RuntimeError, match="callback interrupted"):
            with transport.boundary(expected=[(peer, 0)]):
                transport.push(0, fail_after_copy)
        assert transport.ready()
        sentinel = incoming[0].item()
        dist.barrier()  # Both ranks must sample the sentinel before either overwrites it.
        assert sentinel == 240 + peer

        # Leave a flag unread on both ranks, then immediately reuse it in the exchange.
        with transport.boundary(expected=[(peer, 0)]):
            transport.push(0, lambda dst: outgoing[:1].fill_(254))
        exchange(0, True)
        assert result.tolist() == [peer * 10 + i for i in range(nflags)]

        snapshots = torch.empty((2, nflags), dtype=values.dtype, device=device)
        # Only the set excluding channel 0 was warmed. Excluding channel 1 is cold.
        for first in (0, 1):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                exchange(first, fused_expected=first == 0)
            for snapshot in snapshots.unbind():
                values.add_(1)
                graph.replay()
                snapshot.copy_(result)
            assert snapshots.tolist() == [
                [peer * 10 + i + 2 * first + step for i in range(nflags)] for step in (1, 2)
            ]
    finally:
        del graph
        transport.close()


def test_ce_transport() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two GPUs")
    from cuda.bindings import runtime as cudart

    for src, dst in ((0, 1), (1, 0)):
        status, supported = cudart.cudaDeviceGetP2PAttribute(
            cudart.cudaDeviceP2PAttr.cudaDevP2PAttrNativeAtomicSupported, src, dst
        )
        assert int(status) == 0, status
        if not supported:
            pytest.skip("Requires native peer atomics in both directions")
    run_distributed(2, _exchange_and_recover, dist_backend="nccl", preload=("tensorrt_llm",))
