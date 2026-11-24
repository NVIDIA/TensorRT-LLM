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
"""The cep group's one-sided MNNVL push (``rubin_cp_push``) across a CP group of 4 or 8 ranks.

Each cell of ``rubin_cp_push.CELLS`` whose world is the group's runs its call site's blocks (the
indexer row moves, the KDA Ulysses pushes out and back, a hand-off state) on the workspace that
``rubin_cp_push_workspace`` builds over fabric memory; every rank's receive region must equal the
entry's reference bit for bit (a push is a copy), over many pushes alternating the two regions.
Then the sequence-split KDA with the serial hand-off (``kda_handoff_layer``) of a step whose
requests span ranks, against one rank running the whole batch. Needs one node with 4 GPUs
(``mpirun -n 4 python3 test_rubin_cp_push.py``), or two for 8 ranks (``srun -N 2
--ntasks-per-node 4 --mpi pmix python3 test_rubin_cp_push.py``).
"""

import importlib
import os
import pickle
import sys
import time
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


def _sources(spec: dict, rank: int, world: int, step: int) -> dict:
    """Rank ``rank``'s blocks and table of a cell's push (regenerable on every rank)."""
    mod = importlib.import_module(f"{_FAM}.ops.comm.rubin_cp_push")
    Block = mod.CpPushBlock
    gen = torch.Generator(device="cuda").manual_seed(1000 * step + rank)
    rows = spec["rows"]
    blocks = spec["blocks"]
    if blocks == "kda1":
        hc = 96 // world
        pc, p = hc * 128, 96 * 128
        ncol = -(-(4 * p + 128 + 96) // 16) * 16
        beta = 3 * pc * 2
        fa = beta + -(-hc * 2 // 16) * 16
        w1 = fa + 256
        y = torch.randn(rows, ncol, generator=gen, device="cuda").to(torch.bfloat16)
        pitch = ncol * 2
        out = [
            Block(y[:, s * p :], pitch, pc * 2, pc * 2, s * pc * 2, w1, 0, rows) for s in range(3)
        ]
        out.append(Block(y[:, 4 * p + 128 :], pitch, hc * 2, hc * 2, beta, w1, 0, rows))
        out.append(Block(y[:, 4 * p :], pitch, 0, 256, fa, w1, 0, rows))
        table = [(0, rank * rows, rows)] * world
        return dict(blocks=out, table=table, keep=y)
    if blocks == "kda2":
        pc = 96 // world * 128
        o = torch.randn(world * rows, pc, generator=gen, device="cuda").to(torch.bfloat16)
        out = [Block(o, pc * 2, 0, pc * 2, 0, 96 * 128 * 2, pc * 2, rows)]
        table = [(d * rows, 0, rows) for d in range(world)]
        return dict(blocks=out, table=table, keep=o)
    if blocks == ((3072,),):
        # a hand-off round: this rank's state to the next rank
        state = torch.randn(rows, 768, generator=gen, device="cuda")
        table = [(0, 0, rows if d == (rank + 1) % world else 0) for d in range(world)]
        out = [Block(state, 3072, 0, 3072, 0, 3072, 0, rows)]
        return dict(blocks=out, table=table, keep=state)
    # a row move between two splits of world * rows rows; one plane per width
    rows_mod = importlib.import_module(f"{_FAM}.ops.attention.rubin_cp_mla_rows")
    total = world * rows
    cut = torch.Generator().manual_seed(step)
    cuts = sorted(torch.randint(0, total + 1, (world - 1,), generator=cut).tolist())
    src = tuple(r * rows for r in range(world)) + (total,)
    dst = (0, *cuts, total)
    table, recv = rows_mod.row_move_blocks(src, dst, rank)
    widths = [w for (w,) in blocks]
    cap = max(recv)
    keep, out, plane = [], [], 0
    for w in widths:
        t = torch.randint(0, 256, (rows, w), generator=gen, device="cuda", dtype=torch.uint8)
        keep.append(t)
        out.append(Block(t, w, 0, w, plane, w, 0, max(n for _, _, n in table)))
        plane += cap * w
    return dict(blocks=out, table=list(table), keep=keep)


def _check_cells(ws, world: int, rank: int) -> int:
    mod = importlib.import_module(f"{_FAM}.ops.comm.rubin_cp_push")
    cells = [c for c in mod.rubin_cp_push.CELLS if c.spec["world"] == world]
    checked = 0
    for step, cell in enumerate(cells * 3):
        for rows in cell.spec["rows"]:
            spec = dict(cell.spec, rows=rows)
            mine = _sources(spec, rank, world, step)
            table = mod.push_table(mine["table"], "cuda")
            before = ws.segment[((ws.pushes + 1) % 2) * ws.region :][: ws.region].clone()
            region = mod.rubin_cp_push(ws, mine["blocks"], table)
            torch.cuda.synchronize()
            every = [_sources(spec, r, world, step) for r in range(world)]
            tables = [mod.push_table(e["table"], "cuda") for e in every]
            want = mod.rubin_cp_push.reference([e["blocks"] for e in every], tables, rank, before)
            assert torch.equal(region, want), (rank, cell.why, rows)
            checked += 1
    return checked


def _latency(ws, world: int, rank: int) -> dict:
    """Mean time of a push of ``rows`` rows of 9504 bytes to every rank (the cep8 KDA row)."""
    mod = importlib.import_module(f"{_FAM}.ops.comm.rubin_cp_push")
    out = {}
    for rows in (16, 1024, 2048):
        src = torch.empty(rows, 9504, dtype=torch.uint8, device="cuda")
        table = mod.push_table([(0, rank * rows, rows)] * world, "cuda")
        block = mod.CpPushBlock(src, 9504, 0, 9504, 0, 9504, 0, rows)
        for _ in range(5):
            mod.rubin_cp_push(ws, (block,), table)
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            mod.rubin_cp_push(ws, (block,), table)
        end.record()
        torch.cuda.synchronize()
        out[rows] = start.elapsed_time(end) / 20 * 1e3
    return out


def _handoff_layer(ws, world: int, rank: int) -> int:
    """Sequence-split KDA with the serial hand-off across the group vs one rank."""
    ho = importlib.import_module(f"{_FAM}.ops.ssm.rubin_kda_handoff")
    pre = importlib.import_module(f"{_FAM}.ops.ssm.rubin_kda_prefill")
    conv_mod = importlib.import_module(f"{_FAM}.ops.ssm.rubin_kda_prefill_conv")
    heads, D, W = 96, 128, 4
    dim = heads * D
    cases = [
        ((4096 * world - 700, 300, 400), (True, False, True), (8256, 0, 256)),
        (
            (300, 900, 5, 700, 40) * (world // 4),
            (False, True, True, False, True) * (world // 4),
            (),
        ),
    ]
    for lengths, has_init, snaps in cases:
        gen = torch.Generator(device="cuda").manual_seed(sum(lengths))
        tokens, num = sum(lengths), len(lengths)
        y = torch.randn(tokens, 3 * dim + heads, generator=gen, device="cuda").to(torch.bfloat16)
        g = torch.randn(tokens, dim, generator=gen, device="cuda").to(torch.bfloat16)
        weight = (torch.randn(3 * dim, W, generator=gen, device="cuda") * 0.3).to(torch.bfloat16)
        a_log = torch.empty(heads, device="cuda").uniform_(1, 16, generator=gen).log()
        dt_bias = torch.empty(dim, device="cuda").uniform_(-6.9, -2.3, generator=gen)
        nslots = 2 * num + 2
        state = torch.randn(nslots, heads, D, D, generator=gen, device="cuda") * 0.05
        conv = (torch.randn(nslots, 3 * dim, W - 1, generator=gen, device="cuda") * 0.5).to(
            torch.bfloat16
        )
        slots = torch.arange(num, dtype=torch.int32, device="cuda")
        snapshots = [(b, num + b, off) for b, off in enumerate(snaps) if off]
        for b, init in enumerate(has_init):
            if not init:
                state[b].zero_()
        # one rank, the whole batch
        ref_state, ref_conv = state.clone(), conv.clone()
        starts = [0]
        for n in lengths:
            starts.append(starts[-1] + n)
        for b, slot, off in snapshots:
            ref_conv[slot] = y[starts[b] + off - 3 : starts[b] + off, : 3 * dim].t()
        cu = torch.tensor(starts, dtype=torch.int32, device="cuda")
        q, k, v = conv_mod.rubin_kda_prefill_conv(
            y, weight, ref_conv, cu, slots, torch.tensor(has_init, device="cuda"), max(lengths)
        )
        rows_s = [-1] * num
        chunks = [0] * num
        for b, slot, off in snapshots:
            rows_s[b], chunks[b] = slot, off // 64
        ref_out = pre.rubin_kda_prefill(
            q,
            k,
            v,
            g.view(1, tokens, heads, D),
            y[:, 3 * dim :].unsqueeze(0),
            ref_state,
            slots,
            cu.long(),
            a_log,
            dt_bias,
            D**-0.5,
            -5.0,
            "bt16",
            snapshot_rows=torch.tensor(rows_s, dtype=torch.int32, device="cuda")
            if snapshots
            else None,
            snapshot_chunks=torch.tensor(chunks, dtype=torch.int32, device="cuda")
            if snapshots
            else None,
        )[0]
        # this rank
        rows = -(-tokens // world)
        live = max(0, min(rows, tokens - rank * rows))
        y_r = y.new_zeros(rows, y.shape[1])
        g_r = g.new_zeros(rows, dim)
        y_r[:live] = y[rank * rows : rank * rows + live]
        g_r[:live] = g[rank * rows : rank * rows + live]
        hs = heads // world
        st_r = state[:, rank * hs : (rank + 1) * hs].contiguous()
        cv_r = (
            conv.view(nslots, 3, world, hs * D, W - 1)[:, :, rank]
            .reshape(nslots, 3 * hs * D, W - 1)
            .contiguous()
        )
        snap_slots = (
            torch.tensor([s for _, s, _ in snapshots], dtype=torch.int64, device="cuda")
            if snapshots
            else None
        )
        plan = ho.kda_handoff_plan(
            lengths,
            has_init,
            [(b, off) for b, _, off in snapshots],
            world,
            rank,
            heads,
            slots,
            ws.region,
            snap_slots,
        )
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        t0 = time.perf_counter()
        o = ho.kda_handoff_layer(
            plan,
            ws,
            y_r[:, : 3 * dim],
            g_r,
            y_r[:, 3 * dim :],
            weight,
            a_log,
            dt_bias,
            st_r,
            cv_r,
            D**-0.5,
            -5.0,
            "bt16",
        )
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        want = ref_out[rank * rows : rank * rows + live]
        scale = want.float().abs().max().clamp_min(1e-6)
        err = ((o.float() - want.float()).abs().max() / scale).item()
        assert err < 2e-2, (rank, "output", err)
        for slot in list(range(num)) + [s for _, s, _ in snapshots]:
            ref = ref_state[slot, rank * hs : (rank + 1) * hs]
            e = ((st_r[slot] - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
            assert e < 1e-2, (rank, "state", slot, e)
            ref_c = ref_conv.view(nslots, 3, world, hs * D, W - 1)[slot, :, rank].reshape(
                3 * hs * D, W - 1
            )
            assert torch.equal(cv_r[slot], ref_c), (rank, "window", slot)
        if rank == 0:
            print(
                f"hand-off layer, {world} ranks, {num} requests ({tokens} rows): output rel {err:.2e}, "
                f"{ms:.2f} ms (first call, compile included)",
                flush=True,
            )
    return len(cases)


def _body() -> dict:
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
    mod = importlib.import_module(f"{_FAM}.ops.comm.rubin_cp_push")
    free0 = torch.cuda.mem_get_info()[0]
    ws = mod.rubin_cp_push_workspace(mapping, REGION)
    free1 = torch.cuda.mem_get_info()[0]
    checked = _check_cells(ws, world, rank)
    lat = _latency(ws, world, rank)
    layers = _handoff_layer(ws, world, rank)
    comm.Barrier()
    return dict(checked=checked, lat=lat, layers=layers, fabric=(free0 - free1) >> 20)


def _run() -> bool:
    try:
        out = MPI.COMM_WORLD.allgather(_body())
        if MPI.COMM_WORLD.Get_rank() == 0:
            lat = {k: max(o["lat"][k] for o in out) for k in out[0]["lat"]}
            print(
                f"cep push, {len(out)} ranks: {out[0]['checked']} cell pushes bit-exact, "
                f"{out[0]['layers']} hand-off layers; push of 9504-byte rows to every rank "
                + ", ".join(f"{k} rows {v:.1f} us" for k, v in lat.items())
                + f"; workspace {out[0]['fabric']} MiB per rank",
                flush=True,
            )
        assert all(o["checked"] > 0 for o in out)
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.parametrize("world", [4])
def test_push_cells_and_handoff(world) -> None:
    from mpi4py.futures import MPIPoolExecutor

    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs, has {torch.cuda.device_count()}")
    with MPIPoolExecutor(max_workers=world) as executor:
        assert all(executor.map(lambda _: _run(), range(world)))


if __name__ == "__main__":
    _run()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("cep push test passed", flush=True)
