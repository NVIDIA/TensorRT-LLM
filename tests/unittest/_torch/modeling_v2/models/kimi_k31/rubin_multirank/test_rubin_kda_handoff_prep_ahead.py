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
"""The cep hand-off with the leads' chunk factors computed before the rounds (``prep_ahead``).

``kda_handoff_layer`` runs a lead (the rows of a request that continues from the previous rank)
from the FP32 state the hand-off round brings. Its prefill is two kernels: the chunk factors, which
read no state, and the state recursion. ``prep_ahead`` issues the first before the rounds, so the
serial chain across the ranks holds only the recursions. Across a CP group of 4 or 8 ranks, per
case: the layer with and without it, from the same inputs, gives the same output, KDA state rows
(end states and snapshots) and convolution windows bit for bit on every rank. Cases: one request
over every rank at 16,384 rows per rank from a cached state and from zero (the cep8 context step of
one request), one at 2,048 rows per rank (the shortest lead it splits), three requests with
block-reuse snapshots, and a mix of short requests (leads it runs whole).

``python3 test_rubin_kda_handoff_prep_ahead.py`` under ``mpirun -n 4`` (one node) or ``srun -N 2
--ntasks-per-node 4 --mpi pmix`` (8 ranks) also times the layer per case: both settings
alternately, CUDA events around the layer on every rank after a group barrier, the slowest rank of
each repetition, the median over repetitions; warm (back to back) and cold (L2 overwritten first).
"""

import importlib
import json
import os
import pickle
import statistics
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
pytestmark = pytest.mark.threadleak(enabled=False)

_FAM = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
REGION = 192 << 20
HEADS, D, W = 96, 128, 4
ROWS_FULL = 16384


def _cases(world: int) -> dict:
    """name -> (request lengths, has a cached state, snapshot offsets (0: none))."""
    return {
        "one_request_cached": ((ROWS_FULL * world,), (True,), (0,)),
        "one_request_fresh": ((ROWS_FULL * world,), (False,), (0,)),
        "one_request_2048_rows": ((2048 * world,), (True,), (0,)),
        "three_requests_snapshots": (
            (4096 * world - 700, 300, 400),
            (True, False, True),
            (8256, 0, 256),
        ),
        "short_mix": (
            (300, 900, 5, 700, 40) * (world // 4),
            (False, True, True, False, True) * (world // 4),
            (),
        ),
    }


def _layer_inputs(lengths, has_init, snaps, world: int, rank: int, seed: int) -> dict:
    """This rank's rows and its head shard of the pools (random, the same on every call)."""
    gen = torch.Generator(device="cuda").manual_seed(1000 * seed + rank)
    tokens, num = sum(lengths), len(lengths)
    dim = HEADS * D
    rows = -(-tokens // world)
    live = max(0, min(rows, tokens - rank * rows))
    y = torch.zeros(rows, 3 * dim + HEADS, dtype=torch.bfloat16, device="cuda")
    g = torch.zeros(rows, dim, dtype=torch.bfloat16, device="cuda")
    y[:live] = torch.randn(live, 3 * dim + HEADS, generator=gen, device="cuda").to(torch.bfloat16)
    g[:live] = torch.randn(live, dim, generator=gen, device="cuda").to(torch.bfloat16)
    wgen = torch.Generator(device="cuda").manual_seed(seed)  # the layer's weights: one per group
    weight = (torch.randn(3 * dim, W, generator=wgen, device="cuda") * 0.3).to(torch.bfloat16)
    a_log = torch.empty(HEADS, device="cuda").uniform_(1, 16, generator=wgen).log()
    dt_bias = torch.empty(dim, device="cuda").uniform_(-6.9, -2.3, generator=wgen)
    hs = HEADS // world
    nslots = 2 * num + 2
    state = torch.randn(nslots, hs, D, D, generator=gen, device="cuda") * 0.05
    conv = (torch.randn(nslots, 3 * hs * D, W - 1, generator=gen, device="cuda") * 0.5).to(
        torch.bfloat16
    )
    for b, init in enumerate(has_init):
        if not init:
            state[b].zero_()
    snapshots = [(b, num + b, off) for b, off in enumerate(snaps) if off]
    return dict(
        y=y,
        g=g,
        weight=weight,
        a_log=a_log,
        dt_bias=dt_bias,
        state=state,
        conv=conv,
        slots=torch.arange(num, dtype=torch.int32, device="cuda"),
        snapshots=snapshots,
        snap_slots=torch.tensor([s for _, s, _ in snapshots], dtype=torch.int64, device="cuda")
        if snapshots
        else None,
    )


def _run_layer(ho, ws, plan, inp: dict, state, conv, prep_ahead: bool) -> torch.Tensor:
    dim = HEADS * D
    return ho.kda_handoff_layer(
        plan,
        ws,
        inp["y"][:, : 3 * dim],
        inp["g"],
        inp["y"][:, 3 * dim :],
        inp["weight"],
        inp["a_log"],
        inp["dt_bias"],
        state,
        conv,
        D**-0.5,
        -5.0,
        "bt16",
        prep_ahead=prep_ahead,
    )


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _check_and_time(ws, world: int, rank: int, reps: int) -> dict:
    ho = importlib.import_module(f"{_FAM}.ops.ssm.rubin_kda_handoff")
    comm = MPI.COMM_WORLD
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    out = {}
    for seed, (name, (lengths, has_init, snaps)) in enumerate(sorted(_cases(world).items())):
        inp = _layer_inputs(lengths, has_init, snaps, world, rank, seed + 1)
        plan = ho.kda_handoff_plan(
            lengths,
            has_init,
            [(b, off) for b, _, off in inp["snapshots"]],
            world,
            rank,
            HEADS,
            inp["slots"],
            ws.region,
            inp["snap_slots"],
        )
        results = {}
        for prep_ahead in (False, True):
            state, conv = inp["state"].clone(), inp["conv"].clone()
            o = _run_layer(ho, ws, plan, inp, state, conv, prep_ahead)
            torch.cuda.synchronize()
            results[prep_ahead] = (o.clone(), state, conv)
        (o0, s0, c0), (o1, s1, c1) = results[False], results[True]
        same = dict(out=_same_bits(o0, o1), state=_same_bits(s0, s1), conv=_same_bits(c0, c1))
        assert all(same.values()), (rank, name, same)
        times = {}
        if reps:
            for cold in (False, True):
                ms = {False: [], True: []}
                for r in range(reps):
                    for prep_ahead in (False, True) if r % 2 == 0 else (True, False):
                        state, conv = inp["state"].clone(), inp["conv"].clone()
                        if cold:
                            flush.fill_(r & 0xFF)
                        torch.cuda.synchronize()
                        comm.Barrier()
                        a = torch.cuda.Event(enable_timing=True)
                        b = torch.cuda.Event(enable_timing=True)
                        a.record()
                        _run_layer(ho, ws, plan, inp, state, conv, prep_ahead)
                        b.record()
                        torch.cuda.synchronize()
                        ms[prep_ahead].append(a.elapsed_time(b))
                times["cold" if cold else "warm"] = ms
        out[name] = dict(
            rows=-(-sum(lengths) // world), requests=len(lengths), same=same, times=times
        )
    return out


def _body(reps: int) -> dict:
    from tensorrt_llm.mapping import CpType, Mapping

    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    mapping = Mapping(
        world_size=world,
        rank=rank,
        gpus_per_node=torch.cuda.device_count(),
        tp_size=1,
        cp_size=world,
        cp_config={"cp_type": CpType.ULYSSES},
    )
    push = importlib.import_module(f"{_FAM}.ops.comm.rubin_cp_push")
    ws = push.rubin_cp_push_workspace(mapping, REGION)
    with torch.inference_mode():
        res = _check_and_time(ws, world, rank, reps)
    comm.Barrier()
    return res


def _summary(per_rank: list) -> dict:
    """Per case and setting: the median over repetitions of the slowest rank's layer time (ms)."""
    world = len(per_rank)
    out = {}
    for name in per_rank[0]:
        row = dict(rows=per_rank[0][name]["rows"], requests=per_rank[0][name]["requests"])
        for kind in per_rank[0][name]["times"]:
            for prep_ahead in (False, True):
                series = [per_rank[r][name]["times"][kind][prep_ahead] for r in range(world)]
                slowest = [max(s[i] for s in series) for i in range(len(series[0]))]
                row[f"{kind}_{'ahead' if prep_ahead else 'base'}_ms"] = statistics.median(slowest)
            row[f"{kind}_saving_ms"] = row[f"{kind}_base_ms"] - row[f"{kind}_ahead_ms"]
        out[name] = row
    return out


def _run(reps: int = 0) -> bool:
    try:
        per_rank = MPI.COMM_WORLD.allgather(_body(reps))
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(
                f"hand-off prep_ahead, {len(per_rank)} ranks: "
                f"{len(per_rank[0])} cases bit-identical on every rank (output, states, windows)",
                flush=True,
            )
            if reps:
                print("PREP_AHEAD_TIMES " + json.dumps(_summary(per_rank)), flush=True)
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.parametrize("world", [4])
def test_prep_ahead_is_bit_identical(world) -> None:
    from mpi4py.futures import MPIPoolExecutor

    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs, has {torch.cuda.device_count()}")
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the cep hand-off runs on sm_107")
    with MPIPoolExecutor(max_workers=world) as executor:
        assert all(executor.map(lambda _: _run(), range(world)))


if __name__ == "__main__":
    _run(int(os.environ.get("PREP_AHEAD_REPS", "10")))
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("hand-off prep_ahead test passed", flush=True)
