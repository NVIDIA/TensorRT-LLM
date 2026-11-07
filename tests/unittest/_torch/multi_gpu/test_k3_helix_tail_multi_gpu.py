# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 fused Helix tail across CP ranks (``HelixTail``, MNNVL receive buffers).

With a ``W_UV`` that selects 128 latent columns the V up-projection is exact, so each rank's
output is bitwise the production combine (``_helix_post_process``: FIFO all-to-all +
``helix_post_process_native``) of the packed split reduction's rows and statistics, with and
without a zero-KV mask. With a random ``W_UV`` it agrees with the production path (reduction,
BMM, all-to-all, combine) to BF16 rounding. Eagerly and in a CUDA graph replayed with new inputs.
"""

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

NUM_HEADS = 96
SM_SCALE = 0.1
OUTPUT_SCALE = 0.25


def _split_case(reduce_mod, kv_tiles, splits, query_len, seed):
    """One rank's random decode split state (see test_k3_helix_tail.py) and zero-KV mask."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    requests = len(kv_tiles)
    valid_m = NUM_HEADS * query_len
    physical_m = (valid_m + 127) // 128 * 128
    tasks = splits * requests
    kv_lens = torch.tensor([t * 256 - 3 for t in kv_tiles], dtype=torch.int32, device="cuda")
    plan = torch.empty(
        reduce_mod.packed_plan_length(requests, tasks), dtype=torch.int32, device="cuda"
    )
    reduce_mod.launch_packed_plan(
        kv_lens,
        plan,
        num_requests=requests,
        num_splits=splits,
        wave_tasks=106,
        max_tiles_per_task=max(kv_tiles),
        query_len=query_len,
    )
    begins = plan[2 : 3 + requests].tolist()
    partial = torch.randn((tasks, physical_m, 512), device="cuda", generator=gen)
    maxima = torch.randn((tasks, physical_m), device="cuda", generator=gen) * 12
    sums = torch.rand((tasks, physical_m), device="cuda", generator=gen) * 16 + 0.5
    for begin, end in zip(begins, begins[1:]):
        if end - begin > 1:
            sums[begin] = 0
    maxima.masked_fill_(sums == 0, -torch.inf)
    partial.masked_fill_((sums == 0).unsqueeze(-1), torch.nan)
    rows = torch.randn((requests * valid_m, 512), device="cuda", generator=gen).to(torch.bfloat16)
    mask = torch.rand(requests * query_len, device="cuda", generator=gen) < 0.25
    return dict(
        partial=partial,
        maxima=maxima,
        sums=sums,
        plan=plan,
        q_scale=torch.tensor(1.25, device="cuda"),
        kv_scale=torch.tensor(0.75, device="cuda"),
        rows=rows,
        mask=mask,
        requests=requests,
        splits=splits,
        valid_m=valid_m,
        physical_m=physical_m,
    )


def _run_rank(world_size, configs):
    from tensorrt_llm._torch.attention import attention
    from tensorrt_llm._torch.attention.backends.fp4_mla import fp4_mla_packed_reduce as reduce_mod
    from tensorrt_llm._torch.distributed.mnnvl_memory import init_helix_cp_comm
    from tensorrt_llm._torch.modules.helix_tail import HelixTail

    rank = MPI.COMM_WORLD.Get_rank()
    torch.cuda.set_device(rank)
    try:
        comm = MPI.COMM_WORLD
        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            cp_size=world_size,
            cp_config={"cp_type": CpType.HELIX, "use_nccl_for_alltoall": False, "fifo_version": 2},
        )
        init_helix_cp_comm(mapping)
        assert HelixTail.max_tokens(mapping, NUM_HEADS, 512, 128) >= 128
        tail = HelixTail(mapping)
        heads_local = NUM_HEADS // world_size
        num_sms = torch.cuda.get_device_properties(rank).multi_processor_count
        w_select = torch.zeros((NUM_HEADS, 128, 512), dtype=torch.bfloat16, device="cuda")
        w_select[:, torch.arange(128), 128 + torch.arange(128)] = 1.0
        gen = torch.Generator(device="cuda")
        gen.manual_seed(123)  # the same W_UV on every rank
        w_uv = (torch.randn((NUM_HEADS, 128, 512), device="cuda", generator=gen) * 0.05).to(
            torch.bfloat16
        )
        bad = []

        def reduce(case):
            rows = case["rows"].clone()
            stats = torch.empty((rows.shape[0], 2), device="cuda")
            reduce_mod.reduce_packed_splits(
                case["partial"], case["maxima"], case["sums"], rows, case["q_scale"],
                case["kv_scale"], num_requests=case["requests"], num_splits=case["splits"],
                valid_m=case["valid_m"], physical_m=case["physical_m"], sm_scale=SM_SCALE,
                output_scale=OUTPUT_SCALE, plan=case["plan"], stats=stats, sm_count=num_sms,
            )  # fmt: skip
            return rows, stats

        def partials_of(case):
            return reduce_mod.PackedSplitPartials(
                case["partial"], case["maxima"], case["sums"], case["plan"], case["q_scale"],
                case["kv_scale"], num_requests=case["requests"], valid_m=case["valid_m"],
                physical_m=case["physical_m"], sm_scale=SM_SCALE, output_scale=OUTPUT_SCALE,
            )  # fmt: skip

        def launch(case, w, out, mask):
            num_tokens = out.shape[0]
            stats = torch.empty((num_tokens, NUM_HEADS, 2), device="cuda")
            tail(case["rows"].view(num_tokens, -1), stats, w, mask, fold=partials_of(case), out=out)

        def combine(values, stats, mask):
            num_tokens = values.shape[0] // NUM_HEADS
            return attention._helix_post_process(
                values.reshape(num_tokens, NUM_HEADS * 128),
                stats.view(num_tokens, NUM_HEADS, 2),
                mapping,
                heads_local,
                128,
                zero_kv_mask=mask,
            )

        def bmm(latent):
            num_tokens = latent.shape[0] // NUM_HEADS
            out = torch.empty((num_tokens, NUM_HEADS, 128), dtype=torch.bfloat16, device="cuda")
            torch.ops.trtllm.bmm_out(
                latent.view(num_tokens, NUM_HEADS, 512).transpose(0, 1),
                w_uv.transpose(1, 2),
                out.transpose(0, 1),
            )
            return out.view(-1, 128)

        def check(tag, case, out_select, out_random, mask):
            # Every rank's combine is a collective: all ranks run the same checks in order.
            latent_ref, stats_ref = reduce(case)
            num_tokens = out_select.shape[0]
            ref = combine(latent_ref[:, 128:256].contiguous(), stats_ref, mask)
            if not torch.equal(ref.view(torch.int16), out_select.view(torch.int16)):
                bad.append(f"{tag}: output differs from the reduction + _helix_post_process")
            prod = combine(bmm(latent_ref), stats_ref, mask).float()
            prod = prod.view(num_tokens, heads_local, 128)
            got = out_random.float().view(num_tokens, heads_local, 128)
            finite = torch.isfinite(prod) & torch.isfinite(got)
            scale = torch.where(finite, prod, 0.0).abs().amax(-1, keepdim=True)
            err = torch.where(finite, (got - prod).abs(), 0.0)
            if not bool((err <= scale * 2.0**-6 + 1e-6).all()):
                bad.append(f"{tag}: output beyond the production path ({float(err.max())})")

        def outputs(num_tokens):
            return [
                torch.empty((num_tokens, heads_local * 128), dtype=torch.bfloat16, device="cuda")
                for _ in range(2)
            ]

        # Eager, with and without the zero-KV mask.
        for i, (kv_tiles, splits, query_len) in enumerate(configs):
            case = _split_case(reduce_mod, kv_tiles, splits, query_len, seed=17 * i + rank)
            num_tokens = case["requests"] * query_len
            for mask in (case["mask"], None):
                out_select, out_random = outputs(num_tokens)
                launch(case, w_select, out_select, mask)
                launch(case, w_uv, out_random, mask)
                torch.cuda.synchronize()
                check(f"eager {kv_tiles} mask={mask is not None}", case, out_select, out_random,
                      mask)  # fmt: skip

        # One CUDA graph with every configuration, replayed with new inputs.
        cases = [
            _split_case(reduce_mod, kv_tiles, splits, query_len, seed=99 + rank)
            for kv_tiles, splits, query_len in configs
        ]
        outs = [outputs(c["requests"] * q) for c, (_, _, q) in zip(cases, configs)]

        def launch_all():
            for case, (out_select, out_random) in zip(cases, outs):
                launch(case, w_select, out_select, case["mask"])
                launch(case, w_uv, out_random, case["mask"])

        launch_all()
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch_all()
        for rep in range(3):
            for i, (case, (kv_tiles, splits, query_len)) in enumerate(zip(cases, configs)):
                new = _split_case(reduce_mod, kv_tiles, splits, query_len,
                                  seed=1000 * rep + 7 * i + rank)  # fmt: skip
                for key in ("partial", "maxima", "sums", "rows", "mask"):
                    case[key].copy_(new[key])
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            for case, (out_select, out_random), cfg in zip(cases, outs, configs):
                check(f"graph rep {rep} {cfg[0]}", case, out_select, out_random, case["mask"])
        del graph
        failures = comm.allgather(bad)
        assert not any(failures), f"rank {rank}: {failures}"
    except Exception:
        traceback.print_exc()
        raise
    return True


CONFIGS = [
    # (kv tiles per request, splits, query length)
    ([86], 8, 8),  # n1
    ([86, 60], 8, 8),  # n2
    ([1, 79, 3, 176], 6, 8),  # skewed, single-task requests
    ([40] * 8, 8, 8),  # n8
    ([5, 9, 3], 4, 5),  # tokens not a multiple of 16
]


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_k3_helix_tail(mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    results = mpi_pool_executor.map(_run_rank, *zip(*[(world_size, CONFIGS)] * world_size))
    for r in results:
        assert r is True
