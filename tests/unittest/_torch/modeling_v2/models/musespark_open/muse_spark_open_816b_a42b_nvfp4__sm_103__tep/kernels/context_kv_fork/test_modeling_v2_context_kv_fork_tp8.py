# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eight-rank GPU test for ``context_kv_fork`` with ``drafter_fc_shard``: the side chain's fabric gather
beside main-stream collectives.

Eight MPI ranks on two NVLink-connected four-GPU GB300 nodes, the published
drafter on every rank with the sharded ``fc`` (``drafter_fc_shard``, its own projection-gather
workspace). Each step the forked worker launches its side chain -- whose
``fc`` gather spin-polls the other ranks -- while the main stream runs what
a serving step runs there: an NCCL all-gather of LM-head-sized logits, a
second fabric gather on a workspace of its own (a stand-in for a head gather
on the same transport) and a GEMM; one rank, chosen per step, is held back
before the fork. The arena, the lengths and the drafter inputs must equal a
serial built-in worker's on every rank and agree across ranks, eagerly and
under CUDA-graph replay.

Needs ``MUSESPARK_REFERENCE_DRAFTER`` and eight MPI ranks; skipped otherwise.
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

assert torch.cuda.is_available(), "the context-K/V fork requires a CUDA device"

_DRAFTER_DIR = os.environ.get("MUSESPARK_REFERENCE_DRAFTER")

#: Vocab shard of the target LM head per rank (202048 / 8).
HEAD_SHARD = 25256


def _comm():
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    comm = MPI.COMM_WORLD
    return comm if comm.Get_size() == 8 else None


pytestmark = pytest.mark.skipif(
    not _DRAFTER_DIR or _comm() is None,
    reason="needs MUSESPARK_REFERENCE_DRAFTER and eight MPI ranks on two NVLink-connected nodes",
)


@pytest.fixture(scope="module")
def tp8():
    from tensorrt_llm._torch.models.musespark_open_projection_gather import (
        ProjectionGatherWorkspace,
    )
    from tensorrt_llm.mapping import Mapping

    comm = _comm()
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Requires GB300")
    mapping = Mapping(world_size=8, tp_size=8, rank=rank, gpus_per_node=4)
    return comm, rank, mapping, ProjectionGatherWorkspace(mapping)


class _MainStreamCollectives:
    """What the main stream runs between the fork and the join in the served step, by stand-in."""

    def __init__(self, mapping, gather_workspace, rank):
        from tensorrt_llm._torch.distributed import allgather

        self.allgather = allgather
        self.mapping = mapping
        self.gather = gather_workspace
        g = torch.Generator(device="cuda").manual_seed(rank)
        self.logits = torch.randn(K + 1, HEAD_SHARD, generator=g, device="cuda").to(torch.bfloat16)
        self.local = torch.randn(K + 1, 1024, generator=g, device="cuda").to(torch.bfloat16)
        self.a = torch.randn(4096, 4096, generator=g, device="cuda").to(torch.bfloat16)

    def run(self):
        full = self.allgather(self.logits, self.mapping, dim=-1)
        gathered = self.gather.gather(self.local)
        b = torch.nn.functional.linear(self.a, self.a)
        return full, gathered, b


def _hold_back(rank, step):
    """Delay one rank's main stream before the fork, a different rank per step."""
    if rank == step % 8:
        torch.cuda._sleep(200_000)


@torch.inference_mode()
def test_fork_tp8_beside_main_stream_collectives(tp8, drafter):  # noqa: F811
    comm, rank, mapping, head_gather = tp8
    assert drafter_worker.install_sharded_fc(drafter, mapping, True) is None
    base = _worker(drafter, 1_500)
    forked = _fork_worker(drafter, 1_500)
    vocab = drafter.draft_model_full.model.embed_tokens.num_embeddings
    inp = _Inputs(drafter.config.hidden_size)
    main_work = _MainStreamCollectives(mapping, head_gather, rank)
    fork = forked._kv_fork

    def forked_step(step):
        fork.arm(inp.spec, inp.attn)
        _hold_back(rank, step)
        fork.fire("head")
        main_work.run()
        out = inp.prepare(forked, drafter)
        return out, _decode(drafter, out, forked)

    for step in range(2 * (K + 1)):
        inp.load(step, vocab)  # same seed on every rank: identical inputs
        ref = inp.prepare(base, drafter)
        ref_out = _decode(drafter, ref, base)
        got, got_out = forked_step(step)
        torch.cuda.synchronize()
        _assert_states(forked, base, got, ref, f"rank {rank} eager step {step}")
        assert torch.equal(got_out.view(torch.int16), ref_out.view(torch.int16))
        written = forked._ctx_k_buf[SLOT, :, : int(forked._ctx_len[SLOT])]
        arena = comm.allgather(written.view(torch.int16).sum().item())
        assert len(set(arena)) == 1, f"ranks disagree on the arena: {arena}"
        comm.Barrier()

    snapshot = _worker(drafter, 1_500)
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
    with torch.cuda.graph(graph):
        fork.fire("head")
        main_work.run()
        static_inputs = inp.prepare(forked, drafter)
        static_out = _decode(drafter, static_inputs, forked)
    _copy_state(forked, snapshot)
    _copy_state(base, snapshot)
    torch.cuda.synchronize()
    comm.Barrier()
    for step in range(2 * (K + 1)):
        inp.load(500 + step, vocab)
        ref = inp.prepare(base, drafter)
        ref_out = _decode(drafter, ref, base)
        torch.cuda.synchronize()
        comm.Barrier()
        _hold_back(rank, step)
        graph.replay()
        torch.cuda.synchronize()
        _assert_states(forked, base, static_inputs, ref, f"rank {rank} replay {step}")
        assert torch.equal(static_out.view(torch.int16), ref_out.view(torch.int16))
        comm.Barrier()
    del graph
    comm.Barrier()
