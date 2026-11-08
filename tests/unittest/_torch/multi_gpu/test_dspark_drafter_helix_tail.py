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
"""The helix MLA DSpark drafter's tail on a ``HelixTail``: every head's latent
partials and softmax stats of this rank's pages, V up-projected, exchanged and combined in one
launch, agree to BF16 rounding with the up-projection followed by ``_helix_post_process``; with
the target's head count launched on the same receive buffer in between, eagerly and in a CUDA
graph replayed with new inputs."""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from utils.util import skip_pre_blackwell

from tensorrt_llm.mapping import CpType, Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)

DRAFT_HEADS = 64  # every head of the drafter, decoded on every rank
TARGET_HEADS = 96
LATENT = 512
V_DIM = 128


def _case(num_tokens, heads, seed, rank, world):
    """One rank's partials, stats and zero-KV mask. Token ``t``'s KV is never empty on rank
    ``t % world``: in the drafter every block token sees the block positions, which some rank
    holds, so no token is masked on every rank (that would be 0 / 0)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows = torch.randn((num_tokens, heads, LATENT), device="cuda", generator=gen).bfloat16()
    stats = torch.stack(
        [
            torch.randn((num_tokens, heads), device="cuda", generator=gen) * 4,
            torch.rand((num_tokens, heads), device="cuda", generator=gen) * 8 + 0.5,
        ],
        dim=-1,
    )
    mask = torch.rand(num_tokens, device="cuda", generator=gen) < 0.25
    mask &= torch.arange(num_tokens, device="cuda") % world != rank
    return rows, stats, mask


def _run_rank(world_size, token_counts):
    from tensorrt_llm._torch.attention import attention
    from tensorrt_llm._torch.distributed.mnnvl_memory import init_helix_cp_comm
    from tensorrt_llm._torch.models.modeling_dspark import _helix_tail_attn_out
    from tensorrt_llm._torch.modules.helix_tail import HelixTail

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            cp_size=world_size,
            cp_config={"cp_type": CpType.HELIX, "use_nccl_for_alltoall": False, "fifo_version": 2},
        )
        init_helix_cp_comm(mapping)
        assert HelixTail.max_tokens(mapping, DRAFT_HEADS, LATENT, V_DIM) >= max(token_counts)
        tail = HelixTail(mapping)
        local = DRAFT_HEADS // world_size
        gen = torch.Generator(device="cuda").manual_seed(123)  # the same W_UV on every rank
        w_uv = (torch.randn((DRAFT_HEADS, V_DIM, LATENT), device="cuda", generator=gen) * 0.05).to(
            torch.bfloat16
        )
        w_target = (
            torch.randn((TARGET_HEADS, V_DIM, LATENT), device="cuda", generator=gen) * 0.05
        ).to(torch.bfloat16)
        bad = []

        def reference(rows, stats, mask):
            num_tokens = rows.shape[0]
            v = torch.empty((num_tokens, DRAFT_HEADS, V_DIM), dtype=torch.bfloat16, device="cuda")
            torch.ops.trtllm.bmm_out(rows.transpose(0, 1), w_uv.transpose(1, 2), v.transpose(0, 1))
            return attention._helix_post_process(
                v.view(num_tokens, DRAFT_HEADS * V_DIM),
                stats.contiguous(),
                mapping,
                local,
                V_DIM,
                zero_kv_mask=mask,
            ).view(num_tokens, local, V_DIM)

        def check(tag, got, ref):
            got, ref = got.float(), ref.float()
            scale = ref.abs().amax(-1, keepdim=True)
            if not bool(((got - ref).abs() <= scale * 2.0**-6 + 1e-6).all()):
                bad.append(f"{tag}: {float((got - ref).abs().max())}")

        # The target-shaped launches' inputs, built outside any graph capture (a generator
        # cannot be created while capturing).
        target_cases = {n: _case(n, TARGET_HEADS, 7 + rank, rank, world_size) for n in token_counts}

        def target_launch(n):
            rows, stats, mask = target_cases[n]
            out = torch.empty((n, TARGET_HEADS // world_size * V_DIM), **_bf16())
            tail(rows.view(n, -1), stats, w_target, mask, fold=None, out=out)

        for n in token_counts:
            rows, stats, mask = _case(n, DRAFT_HEADS, 31 * n + rank, rank, world_size)
            for m in (mask, None):
                got = _helix_tail_attn_out(tail, rows, stats, w_uv, m, local)
                target_launch(n)
                torch.cuda.synchronize()
                check(f"eager n={n} mask={m is not None}", got, reference(rows, stats, m))

        # One graph with every size and a target-shaped launch in between, replayed with new inputs.
        cases = [_case(n, DRAFT_HEADS, 1000 + n + rank, rank, world_size) for n in token_counts]
        outs = []
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for rows, stats, mask in cases:
                outs.append(_helix_tail_attn_out(tail, rows, stats, w_uv, mask, local))
                target_launch(rows.shape[0])
        for rep in range(2):
            for i, (case, n) in enumerate(zip(cases, token_counts)):
                for dst, src in zip(
                    case, _case(n, DRAFT_HEADS, 77 * rep + i + rank, rank, world_size)
                ):
                    dst.copy_(src)
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            for case, out, n in zip(cases, outs, token_counts):
                check(f"graph rep {rep} n={n}", out, reference(*case))
        del graph
        failures = comm.allgather(bad)
        assert not any(failures), f"rank {rank}: {failures}"
    except Exception:
        traceback.print_exc()
        raise
    return True


def _bf16():
    return {"dtype": torch.bfloat16, "device": "cuda"}


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_dspark_drafter_helix_tail(mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    # Draft tokens of 1, 2, 4 and 8 requests at block 7.
    counts = [7, 14, 28, 56]
    results = mpi_pool_executor.map(_run_rank, *zip(*[(world_size, counts)] * world_size))
    for r in results:
        assert r is True
