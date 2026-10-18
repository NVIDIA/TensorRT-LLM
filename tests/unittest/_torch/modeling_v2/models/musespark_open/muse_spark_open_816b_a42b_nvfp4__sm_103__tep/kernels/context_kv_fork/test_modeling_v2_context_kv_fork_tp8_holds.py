# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eight-rank test for ``context_kv_fork`` under long cross-rank skew.

``test_modeling_v2_context_kv_fork_tp8.py`` holds one rank's main stream back
~0.1 ms before the fork. Here: (a) a ~5 ms hold of one rank's main stream
before the fork, and (b) a ~2 ms hold *inside* one rank's side chain, before
its sharded-fc fabric gather, so the other ranks' side gathers spin for milliseconds while their
main streams run the NCCL all-gather, a second fabric gather and a GEMM and
then wait at the join. Eager and under CUDA-graph replay; bitwise vs the
serial built-in per rank, arenas equal across ranks, no hang.
"""

import os

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    spec_worker as drafter_worker,
)

__extra_import_path__ = ["."]
from test_modeling_v2_context_kv_fork import (  # noqa: E402
    SLOT,
    K,
    _assert_states,
    _copy_state,
    _decode,
    _fork_worker,
    _Inputs,
    _worker,
    drafter,  # noqa: E402,F401
)
from test_modeling_v2_context_kv_fork_tp8 import (  # noqa: E402
    _comm,
    _MainStreamCollectives,
    tp8,  # noqa: E402,F401
)

_DRAFTER_DIR = os.environ.get("MUSESPARK_REFERENCE_DRAFTER")
pytestmark = pytest.mark.skipif(
    not _DRAFTER_DIR or _comm() is None,
    reason="needs MUSESPARK_REFERENCE_DRAFTER and eight MPI ranks",
)

MAIN_HOLD = 10_000_000  # cycles, ~5 ms at 2 GHz
SIDE_HOLD = 4_000_000  # cycles, ~2 ms


@pytest.mark.parametrize("where", ["main", "side"])
@torch.inference_mode()
def test_fork_tp8_long_holds(tp8, drafter, where):  # noqa: F811
    comm, rank, mapping, head_gather = tp8
    if drafter_worker.install_sharded_fc(drafter, mapping, True) is not None:
        pytest.fail("drafter_fc_shard sharded fc did not install")
    base = _worker(drafter, 1_700)
    forked = _fork_worker(drafter, 1_700)
    vocab = drafter.draft_model_full.model.embed_tokens.num_embeddings
    inp = _Inputs(drafter.config.hidden_size)
    main_work = _MainStreamCollectives(mapping, head_gather, rank)
    fork = forked._kv_fork
    state = {"step": 0}
    original = drafter.project_target_hidden

    def held_projection(hidden):
        # Runs on the side stream (inside the side chain): hold this rank's side before its gather.
        if where == "side" and rank == (3 * state["step"] + 1) % 8:
            torch.cuda._sleep(SIDE_HOLD)
        return original(hidden)

    drafter.project_target_hidden = held_projection
    try:

        def forked_step(step):
            state["step"] = step
            fork.arm(inp.spec, inp.attn)
            if where == "main" and rank == (5 * step + 2) % 8:
                torch.cuda._sleep(MAIN_HOLD)
            fork.fire("head")
            main_work.run()
            out = inp.prepare(forked, drafter)
            return out, _decode(drafter, out, forked)

        for step in range(K + 1):
            inp.load(700 + step, vocab)
            # The reference must not take the held projection's sleep as part of its result: it
            # only delays, it does not change the projection.
            ref = inp.prepare(base, drafter)
            ref_out = _decode(drafter, ref, base)
            got, got_out = forked_step(step)
            torch.cuda.synchronize()
            _assert_states(forked, base, got, ref, f"{where} rank {rank} eager step {step}")
            assert torch.equal(got_out.view(torch.int16), ref_out.view(torch.int16))
            written = forked._ctx_k_buf[SLOT, :, : int(forked._ctx_len[SLOT])]
            arena = comm.allgather(written.view(torch.int16).sum().item())
            assert len(set(arena)) == 1, f"ranks disagree on the arena: {arena}"
            comm.Barrier()

        # Graph: the hold is captured per rank for the step index used at capture; replays then
        # carry that rank's hold every time (a fixed delayed rank).
        snapshot = _worker(drafter, 1_700)
        _copy_state(snapshot, forked)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forked_step(0)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        fork.arm(inp.spec, inp.attn)
        state["step"] = 3
        with torch.cuda.graph(graph):
            if where == "main" and rank == (5 * 3 + 2) % 8:
                torch.cuda._sleep(MAIN_HOLD)
            fork.fire("head")
            main_work.run()
            static_inputs = inp.prepare(forked, drafter)
            static_out = _decode(drafter, static_inputs, forked)
        _copy_state(forked, snapshot)
        _copy_state(base, snapshot)
        torch.cuda.synchronize()
        comm.Barrier()
        for step in range(K + 1):
            inp.load(900 + step, vocab)
            ref = inp.prepare(base, drafter)
            ref_out = _decode(drafter, ref, base)
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            _assert_states(forked, base, static_inputs, ref, f"{where} rank {rank} replay {step}")
            assert torch.equal(static_out.view(torch.int16), ref_out.view(torch.int16))
            comm.Barrier()
        del graph
        comm.Barrier()
        print(
            f"context_kv_fork tp8 {where}-hold rank {rank}: {K + 1} eager steps + {K + 1} replays bitwise"
        )
    finally:
        drafter.project_target_hidden = original
