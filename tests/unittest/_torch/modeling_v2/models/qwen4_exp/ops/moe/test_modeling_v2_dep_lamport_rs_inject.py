# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`dep_lamport_rs_inject` on 4 and 8 ranks, bit for bit against the two launches it replaces.

One process a GPU, the Lamport workspace of `dep_lamport_round_trip` over torch symmetric memory.
For each cell's world size, row count, slot protocol and input form, eagerly and replayed from one
CUDA graph:

* the fused launch over every rank's partials equals `dep_lamport_round_trip(partial, trip, True)`
  followed by `qwen4_exp_hc_inject_norm(hc, routed, inject, norm, eps, 4, shared, gate)`, both
  `hc_out` and `normed`, bitwise;
* the finalize form over a synthetic unfinalized runner output (permuted FC2 rows, routing
  weights, an index map with -1 picks) equals the same two launches over its finalize
  (`finalize_reference`: slot-order fp32 fma from +0, one rounding);
* a rank that runs the plain reduce-scatter beside peers running the fused launch gets the plain
  reduce-scatter's bits (the same bytes travel).

The host-index cells interleave the fused launch with the workspace's own gathers and
reduce-scatters inside one region (the fused launch records its count, the self-cleaning launches
0) and replay the graph `QWEN4_EXP_LAMPORT_REPLAYS` times (default 200), so the re-arm pass of
every slot runs after a fused launch. The inputs carry -0.0 lanes (the transport's marker).
"""

import glob
import os
import socket
import subprocess
import sys

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_inject_norm import (
    qwen4_exp_hc_inject_norm,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import dep_lamport_round_trip as rt
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import dep_lamport_rs_inject as op

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _sass_pdl import nc_loads, pre_wait_violations  # noqa: E402

HIDDEN, HC, TOPK = 2560, 4, 10
REPLAYS = int(os.environ.get("QWEN4_EXP_LAMPORT_REPLAYS", "200"))
CELLS = op.dep_lamport_rs_inject.CELLS
GROUPS = sorted({(c.spec["world"], c.spec["step"]) for c in CELLS})

pytestmark = [pytest.mark.no_xdist]


def _skip(world: int):
    return pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.device_count() < world
        or torch.cuda.get_device_capability() != (10, 3),
        reason=f"the fused reduce-scatter runs on {world} SM103 GPUs of one NVLink domain here",
    )


def _bf16(gen, shape, device, scale=1.0) -> torch.Tensor:
    return (scale * torch.randn(shape, generator=gen, device=device)).to(torch.bfloat16)


def _with_neg_zero(gen, x: torch.Tensor) -> torch.Tensor:
    zero = torch.rand(x.shape, generator=gen, device=x.device) < 0.01
    return torch.where(zero, torch.full_like(x, -0.0), x)


def _site(gen, rows: int, device):
    """`(hc, inject, norm, shared, gate)` in the decode step's layouts (the gate a column)."""
    hc = _bf16(gen, [rows, HC * HIDDEN], device)
    inject = _bf16(gen, [rows, HC], device)
    norm = _bf16(gen, [HC * HIDDEN], device, 0.1)
    shared = _bf16(gen, [rows, HIDDEN], device)
    gate = _bf16(gen, [rows, 3], device)[:, 1:2]
    return hc, inject, norm, shared, gate


def _unfinalized(gen, gathered: int, device):
    """A runner's output without its finalize: `(g2, w, e2p)`, about a third of the picks -1."""
    picks = gathered * TOPK
    perm = torch.randperm(picks, generator=gen, device=device).to(torch.int32)
    drop = torch.rand([picks], generator=gen, device=device) < 0.3
    e2p = torch.where(drop, torch.full_like(perm, -1), perm).view(gathered, TOPK).contiguous()
    g2 = _with_neg_zero(gen, _bf16(gen, [picks, HIDDEN], device))
    w = torch.rand([gathered, TOPK], generator=gen, device=device).to(torch.bfloat16)
    return g2, w, e2p


def _check(name: str, got: torch.Tensor, want: torch.Tensor) -> None:
    same = got.view(torch.int16) == want.view(torch.int16)
    assert bool(same.all()), f"{name}: {int((~same).sum())} of {same.numel()} lanes differ"


def _plain(trip, partial, site):
    hc, inject, norm, shared, gate = site
    routed = rt.dep_lamport_round_trip(partial, trip, True)
    return qwen4_exp_hc_inject_norm(hc, routed, inject, norm, 1e-6, HC, shared, gate)


def _rank_main(rank: int, port: int, world: int, step: bool) -> None:
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), NCCL_ALGO="Ring")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        device = torch.device("cuda", rank)
        steps = rt.LamportSteps() if step else None
        trip = rt.build(HIDDEN, rt.MAX_ROWS, rank, world, steps=steps)
        rs = op.build(trip, HC, op.LAYOUTS[world], fin=True, topk=TOPK, pdl=True)
        gen = torch.Generator(device=device).manual_seed(20261126 + rank)
        cells = [c.spec for c in CELLS if (c.spec["world"], c.spec["step"]) == (world, step)]
        region = steps.region if step else _nothing
        for spec in cells:
            rows, fin = spec["rows"], spec["fin"]
            tag = f"{world} ranks, {rows} rows, fin {fin}"
            for k in range(3):
                site = _site(gen, rows, device)
                if fin:
                    unfin = _unfinalized(gen, world * rows, device)
                    partial = op.finalize_reference(*unfin)
                else:
                    partial = _with_neg_zero(gen, _bf16(gen, [world * rows, HIDDEN], device))
                with region():
                    # A gather before and after (the workspace's self-cleaning launches).
                    x = _bf16(gen, [rows, HIDDEN], device)
                    gathered = rt.dep_lamport_round_trip(x, trip, False)
                    want = _plain(trip, partial, site)
                    if fin:
                        got = op.dep_lamport_rs_inject(None, rs, *site[:3], 1e-6, *site[3:], unfin)
                    elif k == 2 and rank == 0:
                        # This rank runs the plain launch while its peers run the fused one.
                        got = _plain(trip, partial, site)
                    else:
                        got = op.dep_lamport_rs_inject(partial, rs, *site[:3], 1e-6, *site[3:])
                    again = rt.dep_lamport_round_trip(x, trip, False)
                _check(f"hc_out ({tag}, round {k})", got[0], want[0])
                _check(f"normed ({tag}, round {k})", got[1], want[1])
                _check(f"gather around it ({tag})", again, gathered)

            # Replayed from one graph: fresh inputs copied into its static buffers.
            site = _site(gen, rows, device)
            if fin:
                static = _unfinalized(gen, world * rows, device)
            else:
                static = (torch.zeros([world * rows, HIDDEN], dtype=torch.bfloat16, device=device),)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    with region():
                        if fin:
                            g_out = op.dep_lamport_rs_inject(
                                None, rs, *site[:3], 1e-6, *site[3:], static
                            )
                        else:
                            g_out = op.dep_lamport_rs_inject(
                                static[0], rs, *site[:3], 1e-6, *site[3:]
                            )
                        g_plain = rt.dep_lamport_round_trip(
                            op.finalize_reference(*static) if fin else static[0], trip, True
                        )
            torch.cuda.current_stream().wait_stream(stream)
            for i in range(REPLAYS):
                if fin:
                    fresh = _unfinalized(gen, world * rows, device)
                    for dst, src in zip(static, fresh):
                        dst.copy_(src)
                    partial = op.finalize_reference(*static)
                else:
                    static[0].copy_(_with_neg_zero(gen, _bf16(gen, [world * rows, HIDDEN], device)))
                    partial = static[0]
                graph.replay()
                hc, inject, norm, shared, gate = site
                want = qwen4_exp_hc_inject_norm(hc, g_plain, inject, norm, 1e-6, HC, shared, gate)
                _check(f"replay {i}: hc_out ({tag})", g_out[0], want[0])
                _check(f"replay {i}: normed ({tag})", g_out[1], want[1])
                ref = rt.dep_lamport_round_trip(partial, trip, True)
                _check(f"replay {i}: the reduce-scatter in the graph ({tag})", g_plain, ref)
        torch.cuda.synchronize(device)
    finally:
        dist.destroy_process_group()


class _nothing:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize(
    "world, step",
    [
        pytest.param(w, s, marks=_skip(w), id=f"w{w}-{'step' if s else 'counter'}")
        for w, s in GROUPS
    ],
)
def test_rs_inject_matches_reduce_scatter_then_inject_norm(world, step):
    import torch.multiprocessing as mp

    mp.spawn(_rank_main, args=(_free_port(), world, step), nprocs=world, join=True)


@pytest.mark.cpu_only
def test_layouts_pick_a_resident_grid():
    """Host only: the grid stays under each layout's CTA limit, and dep8's table keeps four CTAs
    a row up to 76 rows (the measured high-cap rows), dep4's up to 48."""
    for world, four_up_to in ((4, 48), (8, 76)):
        layouts = op.LAYOUTS[world]
        assert op.streams_per_cta(four_up_to, HC, layouts) == 1
        assert op.streams_per_cta(four_up_to + 1, HC, layouts) == 2
        for rows in range(1, op.MAX_ROWS + 1):
            spp = op.streams_per_cta(rows, HC, layouts)
            limit = dict(layouts).get(spp)
            assert limit is None or rows * (HC // spp) <= limit
    # The finalize form needs its whole grid resident (2 CTAs an SM, GB300's 152 SMs): every
    # front row count of dep8 (up to 112 rows a rank), and dep4 up to 152 (F1's 160..208 keep the
    # runner's finalize).
    for world, ok, past in ((8, (12, 72, 76, 112, 152), 153), (4, (12, 48, 128, 152), 160)):
        layouts = op.LAYOUTS[world]
        assert all(op.fin_resident(r, HC, layouts, 152) for r in ok)
        assert not op.fin_resident(past, HC, layouts, 152)
    for c in CELLS:
        if c.spec["fin"]:
            assert op.fin_resident(c.spec["rows"], HC, op.LAYOUTS[c.spec["world"]], 152), c.why
    assert op.parse_layouts("1:304,2:608") == ((1, 304), (2, 608))
    for bad in ("", "3:100", "1:x", "1"):
        with pytest.raises(ValueError):
            op.parse_layouts(bad)


_COMPILE = """
import sys
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import dep_lamport_rs_inject as op
world, rank, spp, step = (int(v) for v in sys.argv[1:5])
order = tuple((rank + 1 + k) % world for k in range(world))
op._compile((rank, world, 0, 2560, 4, order, spp, bool(step), True, 10, True))
"""


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="compiles for the running SM103 GPU",
)
@pytest.mark.parametrize("world, rank", [(4, 0), (4, 3), (8, 0), (8, 7)])
@pytest.mark.parametrize("spp", [1, 2, 4])
@pytest.mark.parametrize("step", [False, True])
def test_finalize_pdl_reads_nothing_before_its_wait(tmp_path, world, rank, spp, step):
    """The finalize form launched with PDL: in its SASS exactly one `ACQBULK`
    (`griddepcontrol.wait`) and no global access, branch or label before it, in its PTX no
    global load above the wait, and no `ld.global.nc` anywhere (ptxas may move one above the
    wait; the runner's output is still being written when the grid starts)."""
    env = dict(
        os.environ,
        CUTE_DSL_KEEP="ptx,cubin,sass",
        CUTE_DSL_DUMP_DIR=str(tmp_path),
        CUTE_DSL_CACHE_DIR=str(tmp_path / "cache"),
        CUTE_DSL_DISABLE_FILE_CACHING="1",
    )
    args = [str(world), str(rank), str(spp), str(int(step))]
    subprocess.run([sys.executable, "-c", _COMPILE, *args], env=env, check=True, timeout=1800)
    sass = glob.glob(str(tmp_path / "*rs_inject*.sass"))
    ptx = glob.glob(str(tmp_path / "*rs_inject*.ptx"))
    assert len(sass) == 1 and len(ptx) == 1, (sass, ptx)
    sass_text, ptx_text = open(sass[0]).read(), open(ptx[0]).read()
    assert not pre_wait_violations(sass_text, ptx_text)
    assert nc_loads(sass_text) == 0
