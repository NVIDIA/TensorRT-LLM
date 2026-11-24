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
"""The cep target's balanced indexer split across a CP group of 4: moving only the rows whose rank
changes (``_cp_move_rows``, pushed over the CP group's MNNVL workspace) gives every rank the Top-K
the all-gather of every row gave.

The scoring is a row-wise stand-in for the indexer (each row's Top-K of its FP8 query against the
step's keys, causal at the row's KV length), so it depends on the row's inputs and its step row
only, as the indexer does. Needs one node with 4 GPUs. ``python3 test_rubin_cep_idx_move.py`` under
``mpirun -n 4`` runs the same body without the pool executor.
"""

import importlib
import pickle
import random
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
pytestmark = pytest.mark.threadleak(enabled=False)

WORLD = 4
_FAM = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_MODELING = f"{_FAM}.kimi_k31_pro_nvfp4__sm_107__cep.modeling"
HEADS, DIM, TOPK = 32, 128, 64
# A small receive region, so that the max-token move below runs as several pushes.
REGION = 64 << 20


def _pushes(push_mod, rows_mod, src, dst, rank, width, ws):
    """The pushes of moving rows from split ``src`` to split ``dst`` (``_CpMla.idx_*_push``)."""
    plans = push_mod.push_chunks(
        *rows_mod.row_move_blocks(src, dst, rank), rank, ws.region // width
    )
    return tuple(
        (torch.tensor(table, dtype=torch.int32, device="cuda"), most, got)
        for table, most, got in plans
    )


def _inputs(count: int, keys: int, seed: int):
    """Every step row's FP8 query and FP32 weights, and the step's FP8 keys (the same on every
    rank)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(count, HEADS, DIM, device="cuda", generator=gen).to(torch.float8_e4m3fn)
    w = torch.rand(count, HEADS, device="cuda", generator=gen)
    k = torch.randn(keys, DIM, device="cuda", generator=gen).to(torch.float8_e4m3fn)
    return q, w, k


def _score(q8, w, k8, kv_len):
    """Row-wise Top-K: each row's indices of its ``TOPK`` best keys among its first ``kv_len``
    (-1 past them), ``[rows, TOPK]`` int32, in index order. The logits are FP64 weighted dot
    products without a ReLU, so ties (whose order ``topk`` leaves open) do not occur."""
    logits = torch.einsum("rhd,kd->rhk", q8.double(), k8.double())
    logits = (logits * w.double().unsqueeze(-1)).sum(1)
    cols = torch.arange(k8.shape[0], device="cuda")
    logits = logits.masked_fill(cols >= kv_len.unsqueeze(1), float("-inf"))
    vals, idx = logits.topk(TOPK, dim=1)
    idx = torch.where(vals.isfinite(), idx, -1)
    return idx.sort(dim=1).values.to(torch.int32)


def _gathered_back(bounds, begin, end, stride):
    """The all-gather path's row index: where rows [begin, end) sit in the padded gather."""
    index, rank = [], 0
    for row in range(begin, end):
        while row >= bounds[rank + 1]:
            rank += 1
        index.append(rank * stride + row - bounds[rank])
    return index


def _body() -> int:
    from tensorrt_llm.mapping import CpType, Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    modeling = importlib.import_module(_MODELING)
    rows_mod = importlib.import_module(f"{_FAM}.ops.attention.rubin_cp_mla_rows")
    comm_mod = importlib.import_module(f"{_FAM}.ops.comm.rubin_cp_rows")
    push_mod = importlib.import_module(f"{_FAM}.ops.comm.rubin_cp_push")
    mapping = Mapping(
        world_size=WORLD,
        rank=rank,
        gpus_per_node=torch.cuda.device_count(),
        tp_size=1,
        cp_size=WORLD,
        cp_config={"cp_type": CpType.ULYSSES},
    )
    ws = push_mod.rubin_cp_push_workspace(mapping, REGION)
    out_w, back_w = HEADS * (DIM + 4), TOPK * 4
    rng = random.Random(11)
    steps = [
        ([1365] * 12, [148_000] * 12),
        ([2048] * 8, [0, 200_000] * 4),
        ([7000, 500, 500, 500], [0, 150_000, 150_000, 150_000]),
        ([3, 1, 2], [500_000, 0, 7]),
    ] + [
        ([rng.randint(1, 3000) for _ in range(k)], [rng.choice([0, 90_000]) for _ in range(k)])
        for k in (1, 3, 5, 9)
    ]
    checked = 0
    for seed, (new, cached) in enumerate(steps):
        count = sum(new)
        cp = comm_mod.cp_rows(count, WORLD, rank)
        bounds = rows_mod.balanced_query_bounds(new, cached, WORLD)
        equal = tuple(min(r * cp.rows, count) for r in range(WORLD)) + (count,)
        if bounds == equal:
            continue
        # A row's KV length, as a key count of the stand-in's key table.
        keys = 1024
        kv_len = torch.tensor(
            [(c + i + 1) % keys or keys for n, c in zip(new, cached) for i in range(n)],
            device="cuda",
        )
        q_all, w_all, k8 = _inputs(count, keys, seed)
        q8 = torch.zeros(cp.rows, HEADS, DIM, device="cuda").to(torch.float8_e4m3fn)
        w = torch.zeros(cp.rows, HEADS, device="cuda")
        q8[: cp.live].copy_(q_all[cp.begin : cp.end])
        w[: cp.live].copy_(w_all[cp.begin : cp.end])
        b, e = bounds[rank], bounds[rank + 1]

        # The all-gather path (every row out, the padded Top-K back).
        qg = comm_mod.rubin_cp_allgather(q8.view(torch.uint8), mapping)[:count].view(q8.dtype)
        wg = comm_mod.rubin_cp_allgather(w, mapping)[:count]
        scored = _score(qg[b:e], wg[b:e], k8, kv_len[b:e])
        stride = max(hi - lo for lo, hi in zip(bounds, bounds[1:]))
        send = scored.new_full((stride, TOPK), -1)
        send[: e - b].copy_(scored)
        back = torch.tensor(_gathered_back(bounds, cp.begin, cp.end, stride), device="cuda")
        old = comm_mod.rubin_cp_allgather(send, mapping).index_select(0, back)

        # The moved rows.
        to = rows_mod.row_move(equal, bounds, rank)
        back_move = rows_mod.row_move(bounds, equal, rank)
        to_p = _pushes(push_mod, rows_mod, equal, bounds, rank, out_w, ws)
        back_p = _pushes(push_mod, rows_mod, bounds, equal, rank, back_w, ws)
        q8_b, w_b = modeling._cp_move_rows([q8, w], to, to_p, ws)
        assert torch.equal(q8_b.view(torch.uint8), q_all[b:e].view(torch.uint8)), (rank, seed)
        assert torch.equal(w_b, w_all[b:e]), (rank, seed)
        new_topk = modeling._cp_move_rows(
            [_score(q8_b, w_b, k8, kv_len[b:e])], back_move, back_p, ws
        )
        new_topk = new_topk[0]

        own = _score(q8[: cp.live], w[: cp.live], k8, kv_len[cp.begin : cp.end])
        assert torch.equal(new_topk, old), (rank, seed, new, cached)
        assert torch.equal(new_topk, own), (rank, seed)
        checked += 1

    # A max-token step under memory pressure: the caching allocator holds all but 64 MiB of the
    # free memory, as in the memory-profiling warmup. The workspace exists since construction, so
    # the move allocates nothing outside PyTorch (its rows exceed a region: several pushes).
    count = 131072
    cp = comm_mod.cp_rows(count, WORLD, rank)
    equal = tuple(min(r * cp.rows, count) for r in range(WORLD)) + (count,)
    bounds = rows_mod.balanced_query_bounds([count], [0], WORLD)
    q8 = torch.zeros(cp.rows, HEADS, DIM, device="cuda").to(torch.float8_e4m3fn)
    w = torch.arange(cp.rows * HEADS, device="cuda", dtype=torch.float32).view(cp.rows, HEADS)
    w = w + cp.begin * HEADS
    free, _ = torch.cuda.mem_get_info()
    hog = torch.empty(free - (64 << 20), dtype=torch.uint8, device="cuda")
    del hog
    to_p = _pushes(push_mod, rows_mod, equal, bounds, rank, out_w, ws)
    assert len(to_p) > 1, "the max-token move fits one region"
    _, w_b = modeling._cp_move_rows([q8, w], rows_mod.row_move(equal, bounds, rank), to_p, ws)
    b, e = bounds[rank], bounds[rank + 1]
    want = torch.arange(b * HEADS, e * HEADS, device="cuda", dtype=torch.float32)
    assert torch.equal(w_b.flatten(), want), (rank, "max-token move under memory pressure")
    torch.cuda.empty_cache()
    comm.Barrier()
    return checked


def _run() -> bool:
    try:
        checked = MPI.COMM_WORLD.allgather(_body())
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"cep indexer row move: {checked[0]} balanced steps, Top-K equal", flush=True)
        assert checked[0] > 0
    except Exception:
        traceback.print_exc()
        raise
    return True


def test_moved_rows_give_the_all_gathered_top_k() -> None:
    from mpi4py.futures import MPIPoolExecutor

    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"needs {WORLD} GPUs, has {torch.cuda.device_count()}")
    with MPIPoolExecutor(max_workers=WORLD) as executor:
        assert all(executor.map(lambda _: _run(), range(WORLD)))


if __name__ == "__main__":
    _run()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("cep indexer row move test passed", flush=True)
