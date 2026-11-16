# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The 65..128-row (9..16 requests at K = 7) TEP8 band extensions on one SM107 GPU: device bits and addresses.

Each extension is a lever parameter whose default is the band served before (``test_modeling_v2_levers``
pins the tables); this file pins what the extended band computes, on one device, before any 8-rank run:

* W1 ``tep8_ring_buddy_wide__MAX_ROWS=128`` + ``prerouted_experts__TEP8_MAX_ROWS=128``: ``ring_buddy_assign``'s
  one program at 65..128 rows (1024 slots) equals its Python reference on every rank of 8 (local ids, BF16 weights,
  stats), and its prerouted tables (16-token tiles above 64 rows, ``ring_buddy._prerouted_tile``) are consistent
  (counts, CTA tables, every slot's permuted row maps back to its token); the prerouted expert op on those tables
  equals the stock FP4 block-scale MoE op at the stock tile / default config, slot by slot (skipped on a build
  without the prerouted op).
* W2 ``tep8_direct_hop__MAX_ROWS=128``: o_proj ``[R, 1024] x [8192, 1024]^T`` on the split_k-1 scatter producer
  (``chunks``, ``grid`` and ``tile128``) lands every token / feature exactly where the direct consumer polls it on a
  16-slot workspace (owner t % 8, slot t // 8, feature block f), nothing outside those slots, and every value is bit
  for bit the built-in ``F.linear`` (cuBLAS) of the whole call: the claim that makes the 65..128-row direct hop
  bitwise against the cuBLAS o_proj + two-shot hop it replaces (true on GB300; this asserts it on Rubin). The 33..64-row
  calls on the 16-slot workspace (the served kernel, its feature-block stride moved) too, and a CUDA graph replay.
* W3 ``tep8_publish_splitk__MAX_ROWS=128``: the one-launch split-K-4 publication of a 65..128-row pool / unpool slice
  equals, bit for bit, the served 33..64-row launch on each 64-row chunk and the same tactic on each 8-row request
  (each row's one-request bits), writes every element of its slot and nothing past it.
* W4 ``swa_one_cluster_block__TEP8_MAX_REQUESTS=16`` (+ ``__TEP8_MIN_REQUESTS=2``): the TEP8 one-cluster SWA block
  (CL8: 2 KV / 16 query heads per rank, 8 rows per request, 32-key FP8 pages) on a step of 9..16 requests, and of
  2..4 requests: every request's output rows and the K/V bytes the call appends equal, bit for bit, the same
  request's inside the certified 5..8-request calls (calls of 8 in order, calls of 5 in reverse order, padded with
  filler requests) and inside one shuffled call of up to 16 (2..4: also calls of 2); the call appends its step rows
  and writes nothing else, and reads no poisoned page or slot; one CUDA-graph capture of the step replays the eager
  bits. 2..4-request steps launch CL8 only with TEP8_MIN_REQUESTS at or below their count (and one2), else the split
  kernels, whose appended K/V bytes CL8's equal. (CL8's TEP8 build takes no 1-request call.) On the host: the caller
  routes 9..16 only under one2 and while CL8 launches them, the prep-append stands down for every step it routes,
  and the op refuses 9+ sequences without CL8.
* W5 ``tep8_publish_splitk__MIN_ROWS`` / ``tep8_direct_hop__MIN_ROWS`` / ``tep8_wide64_splitk__MIN_ROWS`` (9..33): the
  2..4-request steps (9..32 rows) on the 5..8-request family. The pool / unpool publication of a 9..32-row slice is the
  split-K producer's (its key at 9 rows the certified one), writes every element of its slot once and nothing past it,
  and each row equals, bit for bit, the same row inside a 48-row publication, its 8-row request's call and the 9..32-row
  wide table's kernel of the gathers path it replaces; a CUDA graph replays it. o_proj of a 9..32-row step on the served
  wide64 scatter tactic lands every token where the direct consumer polls it, nothing else, with the bits of the same
  rows inside a 48-row scatter call and of the wide table's split_k-1 o_proj; a graph replays it. The dispatcher's
  ``wide64`` routes the armed o_proj below 33 rows to the scatter and, with the wide64 floor moved, QKV / o_proj to the
  wide64 tactics with the wide table's bits.
* W6 ``tep8_wide64_splitk__MAX_ROWS=128`` (+ ``__WIDE_TILE=64x64|128x64|64x128``): a 65..128-row TP8 QKV call in ONE
  launch of the served wide64 tactic (``splitk_gemm.run_splitk_dense_tiles``, token tiles on the grid; or on the tile
  WIDE_TILE names) writes every element of its output and nothing past it, and each row equals, bit for bit, the same
  row inside the served 64-row launches (64-row chunks) and inside the 8-row cell's calls (each request's one-request
  bits); the dispatcher's ``wide64``
  routes 65..MAX_ROWS-row QKV there (o_proj and rows above MAX_ROWS keep their paths); a CUDA graph replays it. A
  timing print (us/call, 62-layer chain of distinct weights) against cuBLAS ``F.linear`` at 64 / 72 / 96 / 128 rows.
* W7 ``tep8_direct_hop__PREFETCH_MIN_ROWS``: the QKV dispatcher of an armed 65..128-row step forks the L2 prefetch of
  the layer's o_proj weight on the plan's side stream (eager and inside a CUDA graph capture, joined by the next
  ``plan.join``), and none of an unarmed step; a timing print of the 65..128-row scatter producer (grid / tile128 /
  chunks, the PRODUCER knob) with its weight cold in L2 vs prefetched.
* W9 ``tep8_router_push__RB_CUDA=1``: the TP8 ring-buddy placement + prerouted tables on one CUDA CTA
  (``ring_buddy_cuda.rb_tables``) equal the Triton program's (ids, BF16 weights, stats, nnc / tnp, c2b / c2m) on every
  rank at 9..128 rows, and its e2p / p2t the Python model's (an expert's rows in token order); a CUDA graph replays it.
* W8 ``tep8_publish_splitk__WIDE_POOL_TILE=64x32``: the one-launch pool publication of a 65..128-row slice on
  (64, 32, 4, 8) equals, bit for bit, the served (64, 16, 4, 10) one-launch publication and the 8-row cells' calls,
  writes every element of its slot and nothing past it; ``_wide_pool_tactic`` accepts it (a certified key) on this
  device and ``publish_projection`` routes the pool slice (not the unpool) there; a CUDA graph replays it; a timing
  print of both tiles (62-layer chain of distinct weight slices) at 72 / 96 / 128 rows.
* W9 ``swa_one_cluster_block__TEP8_WIDE_MIN_REQUESTS=12``: swa_cl8w, the TEP8 one-cluster SWA block with ONE 8-CTA
  cluster per request over both KV heads (one wave up to 22 requests): a step of 12 / 13 / 16 requests launches it (11
  keeps swa_cl8), and every request's output rows and appended K/V bytes equal, bit for bit, swa_cl8's n-request call
  and the certified calls of 8; one CUDA-graph capture replays the eager bits (int32 and int64 positions, checkpoint
  and permuted projections); a timing print of swa_cl8 (two waves) vs swa_cl8w (one wave) at 12 / 13 / 16 requests.
* W10 ``tep8_drafter_wide_gemms__MAX_ROWS=128`` (+ ``__WIDE_TILE``, ``__SITES``): a 65..128-row TP8 drafter GEMM
  (convolution coefficients, qkv / context K/V, o_proj, down, gate_up) in ONE launch of its 64-row split-K tactic
  writes every element of its output and nothing past it, and each row equals, bit for bit, the same row inside the
  64-row tactic's launches and inside the 8-row cell's calls (each request's one-request bits); ``spec_worker._gemm``
  and ``FixedTacticLinearMethod`` route only SITES' 65..MAX_ROWS-row calls there (the fc shard and other sites keep
  ``F.linear``); a CUDA graph replays it. A timing print (us/call over 5 distinct weights, the drafter's layers)
  against cuBLAS ``F.linear`` per site and tile at 64 / 72 / 96 / 128 rows, with the step delta. With
  ``__WINDOW_TILES=2..4``: the drafter window attention of a 9..16-request TEP8 call (8 query / 1 KV head, 2,047-key
  window) on multi-tile splits stays as close to the FP64 attention as the one-tile geometry, appends the same K/V
  bytes, applies only from WINDOW_MIN_ROWS rows, replays in a CUDA graph; a timing print per tile count.
  With ``__PREFETCH=1..3`` (kernels/drafter_l2_prefetch): a drafter-shaped GEMM chain forking the prefetches where the
  block decode does keeps its bits, eagerly and in a CUDA graph (one join rejoins every fork), MIN_ROWS gates it; a
  timing print of the 5-layer chain per level at 64 / 96 / 128 rows with the device's L2 size.
  With ``__FINISH_NORM=1``: a non-fused hop's tail (``ops/spec/drafter_finish_norm``) leaves the output and the in-place
  residual bit for bit those of ``dflash2_block_conv`` (finish side) + the built-in RMSNorm's fused add at 8 / 72 / 96 /
  128 rows; a CUDA graph replays it; a timing print vs the two launches.
  With ``__QK_FOLD=1``: the window attention on raw q / k_new with dr3_folds' QK fold (no drafter_attn_pdl) returns and
  appends bit for bit what it does after the built-in ``fused_qk_norm_rope`` launch, at 2 / 8 / 12 / 16 requests, 1 and
  3 tiles per split.
* W10 the pool / unpool publication census at 65..128 rows (``tep8_publish_splitk__MAX_ROWS=128``; pool on
  ``__WIDE_POOL_TILE`` 64x16 and 64x32, unpool on ``__WIDE_UNPOOL_TILE`` 64x32 and 64x64): a concurrent poller (a
  Triton kernel spinning on the slot from before the producer launches) sees every element of the slot only as the
  sentinel or its final value, and every element as the sentinel at least once (the observation window covers its
  store); the final slot equals, bit for bit, each row's 33..64-row tactic (64-row chunks) and the 8-row cells;
  nothing past the slot is written; the producer key of every served tile and of pool 64x32 is certified at this SM
  count, so ``_splitk_publisher`` takes 64x32; unpool 64x64's key is printed (``W10_CENSUS_KEY``: add an entry naming
  it once this test passes) and the gate refuses it while no entry names it, taking it once one does. Two trials per
  row count.
  The census property is per producer key (the device code and the launch options), not per peer: one GPU drives it.
* W11 ``tep8_wide64_splitk__WIDE_TILE`` / ``__WIDE_SPLIT_K`` (QKV at 65..128 rows) and ``tep8_publish_splitk__
  WIDE_UNPOOL_TILE``: each QKV one-launch candidate (64x64 / 64x128 / 128x128 at split_k 4, 128x128 / 128x64 at split_k
  8) compiles, equals bit for bit its split_k's 64-row launches (and at split_k 4 the 8-row cells), and is close to
  cuBLAS; a timing print per candidate and cuBLAS ``F.linear`` (62-layer chain of distinct weights, PDL) at 72 / 96 /
  128 rows; the unpool slice's 64x32 vs 64x64 one launch timed the same way. A build failure FAILS its case.
* W12 ``tep8_direct_hop__PRODUCER`` / ``__PRODUCER_STAGES``: the 65..128-row o_proj scatter producer (grid at 3..5
  stages, tile128 at 3..4, chunks) lands every token bit for bit as ``F.linear`` at each stage count; a timing print of
  each (62-layer chain of distinct o_proj weights in one CUDA graph, PDL as served) at 72 / 96 / 128 rows.
* W13 (a bench, no assertion beyond running): the QKV GEMM at 72 / 96 / 128 rows through cuBLAS variants -- F.linear
  (the served path), addmm with a zero bias (cuBLASLt's epilogue path), both under
  ``preferred_blas_library("cublaslt")``, and with ``CUBLASLT_WORKSPACE_SIZE`` 0 / 65536 KiB (subprocesses) -- us/call
  (62-layer graph of distinct weights) and the kernels each launches (torch.profiler): whether one avoids nvjet
  split-K's separate splitKreduce launch.

The 8-rank checks (finalize transport at 128 tokens == the built-in TP8 kernel; the direct hop's consumer at 65..128
rows == cuBLAS o_proj + the served two-shot hop; for W5's band, lanes/gen/wide128/vr_low17_ranks.py: the direct hop and
the publication's consumers at 9..32 rows) need two nodes: GEN_WIDE128_PORT.md says how.
* W14 ``ops/gemm/splitk_gemm_2cta`` (QKV at 65..128 rows, not served yet): the 2-CTA (``cta_group::2``) split-K
  kernel with its in-kernel K-slice reduction, at split_k 4 (with and without the pre-wait L2 prefetch of its weight
  slice) and 2, writes every element of its output and nothing past it, and each row equals, bit for bit, its
  split_k's 64-row launches (at split_k 4 also the 8-row cells); close to cuBLAS. A timing print per arm against
  cuBLAS ``F.linear`` (62-layer chain of distinct weights, PDL) at 72 / 96 / 128 rows. A build failure FAILS its case.
"""

import importlib
import math
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_107__tep"
)


def _supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(not _supported(), reason="the sm107 TEP8 kernels run on SM107")

EXPERTS, TOP_K, WINDOW = 256, 8, 64
HIDDEN, K_OPROJ = 8192, 1024


def _module(name: str):
    return importlib.import_module(f"{_TARGET}.{name}")


def _routing(rows: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    sel = torch.stack(
        [torch.randperm(EXPERTS, device="cuda", generator=g)[:TOP_K] for _ in range(rows)]
    ).to(torch.int32)
    w = torch.softmax(torch.randn(rows, TOP_K, device="cuda", generator=g), -1).float()
    return sel.contiguous(), w.contiguous()


# --------------------------------------------------------------------------------------------------
# W1: ring-buddy placement and prerouted tables at 65..128 rows
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [40, 64, 65, 72, 88, 96, 120, 128])
def test_w1_ring_buddy_assign_wide_equals_reference(rows):
    rba = _module("ops.moe.ring_buddy_assign")
    rb = _module("kernels.replicated_experts.ring_buddy")
    assert rba.MAX_SLOTS >= 128 * TOP_K
    tile = rb._prerouted_tile(rows)
    assert tile == (8 if rows <= 64 else 16)
    sel, w = _routing(rows, 300 + rows)
    for rank in range(8):
        lid, wb, st, meta = rba.ring_buddy_assign(sel, w, rank, True, None, ranks=8, meta_tile=tile)
        rl, rw, rs = rba.ring_buddy_assign.reference(sel, w, rank, True, None, ranks=8)
        torch.cuda.synchronize()
        assert torch.equal(lid, rl), (rows, rank)
        assert torch.equal(wb.view(torch.int16), rw.view(torch.int16)), (rows, rank)
        assert torch.equal(st, rs), (rows, rank)
        # The tables of the two-window call: counts and CTA tables are functions of the local ids; a slot's
        # position within its expert is arrival order, so check it through the permuted -> token map.
        local = lid.reshape(-1)
        mine = local < WINDOW
        counts = torch.bincount(local[mine].long(), minlength=WINDOW)
        tiles = (counts + tile - 1) // tile
        nnc = int(tiles.sum())
        assert int(meta["nnc"][0]) == nnc and int(meta["tnp"][0]) == nnc * tile, (rows, rank)
        offs = torch.cumsum(tiles, 0) - tiles
        c2b = torch.repeat_interleave(torch.arange(WINDOW, device="cuda"), tiles)
        assert torch.equal(meta["c2b"][:nnc].long(), c2b), (rows, rank)
        e2p = meta["e2p"].reshape(-1)
        assert torch.equal(e2p >= 0, mine), (rows, rank)
        slots = torch.nonzero(mine).flatten()
        assert torch.equal(meta["p2t"][e2p[slots].long()].long(), slots // TOP_K), (rows, rank)
        # Every expert's rows sit in its own tiles.
        expert_of_row = local[slots].long()
        assert bool(((e2p[slots] // tile) >= offs[expert_of_row]).all()), (rows, rank)
        assert bool(((e2p[slots] // tile) < offs[expert_of_row] + tiles[expert_of_row]).all())


def _upstream():
    """The upstream FP4 MoE tests' helpers (``tests/unittest/_torch/modeling_v2/moe``)."""
    root = Path(__file__).resolve().parents[3]
    for path in (str(root / "moe"), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module("test_modeling_v2_fp4_block_scale_moe_runner")


@pytest.mark.skipif(
    not hasattr(torch.ops.trtllm, "fp4_block_scale_moe_prerouted"),
    reason="this build registers no trtllm::fp4_block_scale_moe_prerouted",
)
@pytest.mark.parametrize("rows", [64, 65, 72, 96, 128])
def test_w1_prerouted_on_assign_tables_equals_stock(rows):
    rba = _module("ops.moe.ring_buddy_assign")
    rb = _module("kernels.replicated_experts.ring_buddy")
    up = _upstream()
    hidden, inter = 2048, 1024
    args, ref = up._build(WINDOW, hidden, inter, seed=rows)
    data, sf, _ = up._rand_activation(rows, hidden, 1.0, ref["gen"])
    sf = sf.view(torch.float8_e4m3fn)
    o1, o1g, o2 = up._scalars(WINDOW, 1.0, 1.0)
    sel, w = _routing(rows, 900 + rows)
    tile = rb._prerouted_tile(rows)
    for rank in (0, 3, 7):
        lid, wb, _, meta = rba.ring_buddy_assign(sel, w, rank, True, None, ranks=8, meta_tile=tile)
        config = torch.ops.trtllm.fp4_block_scale_moe_default_config(
            tile, TOP_K, hidden, inter, WINDOW, rows, 0
        )
        stock = torch.classes.trtllm.FP4BlockScaleMoERunner(0).run_moe(
            None, None, data, sf, args["gemm1_weights"], args["gemm1_weights_scale"], None, None,
            None, None, args["gemm2_weights"], args["gemm2_weights_scale"], None, o1, o1g, o2,
            EXPERTS, TOP_K, None, None, inter, 0, WINDOW, None, 1, False, [tile, config], wb, lid,
            None,
        )  # fmt: skip
        zeros = torch.zeros(rb.TEP8_PREROUTED_MAX_ROWS * TOP_K, dtype=torch.int32, device="cuda")
        pre = torch.ops.trtllm.fp4_block_scale_moe_prerouted(
            data, sf, args["gemm1_weights"], args["gemm1_weights_scale"], args["gemm2_weights"],
            args["gemm2_weights_scale"], o1, o1g, o2, lid, wb, meta["e2p"], meta["p2t"],
            meta["c2b"], meta["c2m"], meta["nnc"], meta["tnp"], zeros, EXPERTS, TOP_K, inter, 0,
            WINDOW, tile, config, 0,
        )  # fmt: skip
        torch.cuda.synchronize()
        s_out, s_e2p = stock[0], stock[2].reshape(-1)
        p_out, p_e2p = pre[0], pre[2].reshape(-1)
        assert torch.equal(s_e2p >= 0, p_e2p >= 0), (rows, rank)
        local = (s_e2p >= 0).nonzero().flatten()
        assert torch.equal(
            s_out[s_e2p[local].long()].view(torch.int16),
            p_out[p_e2p[local].long()].view(torch.int16),
        ), (rows, rank)


# --------------------------------------------------------------------------------------------------
# W9 tep8_router_push__RB_CUDA: the TP8 ring-buddy placement + prerouted tables as one CUDA CTA
# (ops/moe/ring_buddy_cuda.rb_tables) instead of the Triton program
# --------------------------------------------------------------------------------------------------
def _rb_tables_both(rba, sel, w, rank, tile, tilew=False):
    band = dict(tile_placement=tilew, tile_min_rows=41, tile_max_rows=64)  # tep8_tile_placement as served at bs6-8
    try:
        rba.configure(cuda_tables=False, **band)
        a = rba.ring_buddy_assign(sel, w, rank, True, None, ranks=8, meta_tile=tile)
        rba.configure(cuda_tables=True, **band)
        b = rba.ring_buddy_assign(sel, w, rank, True, None, ranks=8, meta_tile=tile)
    finally:
        rba.configure()
    torch.cuda.synchronize()
    return a, b


@pytest.mark.parametrize("tilew", [False, True])
@pytest.mark.parametrize("rows", [9, 16, 33, 41, 48, 64, 65, 72, 96, 120, 128])
def test_w9_rb_tables_equal_the_triton_program(rows, tilew):
    """ids, BF16 weights, stats, nnc / tnp and the nnc CTAs' c2b / c2m bit for bit the Triton program's on every
    rank (count rule; ``tilew``: also ``tep8_tile_placement``'s rule at 41..64 rows); e2p / p2t the same placement
    with an expert's rows dense and in token order, exactly the Python model's."""
    rba = _module("ops.moe.ring_buddy_assign")
    rb = _module("kernels.replicated_experts.ring_buddy")
    rbc = _module("ops.moe.ring_buddy_cuda")
    tile = rb._prerouted_tile(rows)
    for seed in (1, 2):
        sel, w = _routing(rows, 4000 + 10 * rows + seed)
        flat = sel.reshape(-1).tolist()
        for rank in range(8):
            (la, wa, sa, ma), (lb, wb, sb, mb) = _rb_tables_both(rba, sel, w, rank, tile, tilew)
            assert torch.cuda.current_device() in rbc._KERNELS  # the CUDA program served the second call
            assert torch.equal(la, lb) and torch.equal(sa, sb), (rows, rank)
            assert torch.equal(wa.view(torch.int16), wb.view(torch.int16)), (rows, rank)
            nnc = int(ma["nnc"][0])
            assert int(mb["nnc"][0]) == nnc and torch.equal(ma["tnp"], mb["tnp"]), (rows, rank)
            assert torch.equal(ma["c2b"][:nnc], mb["c2b"][:nnc]), (rows, rank)
            assert torch.equal(ma["c2m"][:nnc], mb["c2m"][:nnc]), (rows, rank)
            ea, eb = ma["e2p"].reshape(-1), mb["e2p"].reshape(-1)
            assert torch.equal(ea >= 0, eb >= 0), (rows, rank)
            model = rbc.model(flat, rank, tile, tilew=tilew and 41 <= rows <= 64, cost=rba._TILE_COST)
            assert eb.tolist() == model["e2p"], (rows, rank)
            p2t = mb["p2t"].tolist()
            assert all(p2t[p] == t for p, t in model["p2t"].items()), (rows, rank)


_EARLY_PRODUCER_SRC = r"""
extern "C" __global__ void produce(const int* ss, const float* sw, int* sel, float* w, int n, unsigned long long ns) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // dependents launch now (the push's EARLY_TRIGGER)
  unsigned long long t0, t1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1)); } while (t1 - t0 < ns);
  for (int i = threadIdx.x; i < n; i += blockDim.x) { sel[i] = ss[i]; w[i] = sw[i]; }
}
"""


def _early_producer():
    """A PDL primary that triggers its dependents first and copies the ids / weights ``ns`` later: the router push's
    EARLY_TRIGGER build on one GPU (the routing written long after the dependent is resident)."""
    import ctypes

    from cuda.bindings import driver as cu
    from cuda.bindings import nvrtc

    major, minor = torch.cuda.get_device_capability()
    prog = nvrtc.nvrtcCreateProgram(_EARLY_PRODUCER_SRC.encode(), b"produce.cu", 0, [], [])[1]
    opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"-default-device"]
    assert int(nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]) == 0
    cubin = b" " * nvrtc.nvrtcGetCUBINSize(prog)[1]
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    module = cu.cuModuleLoadData(cubin)[1]
    fn = cu.cuModuleGetFunction(module, b"produce")[1]

    def launch(ss, sw, sel, w, ns):
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = 1, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 256, 1, 1
        cfg.sharedMemBytes = 0
        cfg.hStream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        cfg.attrs = []
        cfg.numAttrs = 0
        vp = ctypes.c_void_p
        args = (
            (ss.data_ptr(), sw.data_ptr(), sel.data_ptr(), w.data_ptr(), ss.numel(), int(ns)),
            (vp, vp, vp, vp, ctypes.c_int, ctypes.c_ulonglong),
        )
        assert int(cu.cuLaunchKernelEx(cfg, fn, args, 0)[0]) == 0

    return launch, module


@pytest.mark.parametrize("rows", [9, 41, 64, 72, 128])
def test_w9_rb_tables_wait_for_an_early_triggering_producer(rows):
    """rb_tables as the PDL dependent of a producer that triggers it first and writes the ids / weights 200 us later
    into poisoned buffers (expert 255 everywhere, NaN weights): its tables are those of the produced ids, i.e. it reads
    them after its griddepcontrol.wait. (The 8-rank gate 1026468's RBC mismatches: rb_tables read the push's ids before
    the push wrote them; it then took them through const __restrict__ pointers.)"""
    rba = _module("ops.moe.ring_buddy_assign")
    rb = _module("kernels.replicated_experts.ring_buddy")
    tile = rb._prerouted_tile(rows)
    src_sel, src_w = _routing(rows, 5000 + rows)
    launch, _keep = _early_producer()
    for rank in range(8):
        tilew = rank % 2 == 1
        exp = _rb_tables_both(rba, src_sel, src_w, rank, tile, tilew)[0]  # the Triton program on the produced ids
        sel = torch.full_like(src_sel, 255)
        w = torch.full_like(src_w, float("nan"))
        rba.configure(cuda_tables=True, tile_placement=tilew, tile_min_rows=41, tile_max_rows=64)
        try:
            rba.ring_buddy_assign(src_sel, src_w, rank, True, None, ranks=8, meta_tile=tile)  # rb_tables built (eager)
            torch.cuda.synchronize()
            launch(src_sel, src_w, sel, w, 200_000)
            got = rba.ring_buddy_assign(sel, w, rank, True, None, ranks=8, meta_tile=tile)
            torch.cuda.synchronize()
        finally:
            rba.configure()
        assert torch.equal(sel, src_sel)  # the producer ran
        (le, we, se, me), (lg, wg, sg, mg) = exp, got
        assert torch.equal(le, lg) and torch.equal(se, sg), (rows, rank)
        assert torch.equal(we.view(torch.int16), wg.view(torch.int16)), (rows, rank)
        nnc = int(me["nnc"][0])
        assert int(mg["nnc"][0]) == nnc and torch.equal(me["c2b"][:nnc], mg["c2b"][:nnc]), (rows, rank)
        assert torch.equal(me["c2m"][:nnc], mg["c2m"][:nnc]), (rows, rank)


def test_w9_rb_tables_graph_replays():
    """The CUDA program captured in a CUDA graph (a PDL launch on the capturing stream) replays the eager bits on new
    ids written in place."""
    rba = _module("ops.moe.ring_buddy_assign")
    rows, tile, rank = 96, 16, 5
    sel, w = _routing(rows, 77)
    rba.configure(cuda_tables=True)
    try:
        rba.ring_buddy_assign(sel, w, rank, True, None, ranks=8, meta_tile=tile)  # compile (eager)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = rba.ring_buddy_assign(sel, w, rank, True, None, ranks=8, meta_tile=tile)
        sel2, w2 = _routing(rows, 78)
        sel.copy_(sel2)
        w.copy_(w2)
        g.replay()
        ref = rba.ring_buddy_assign(sel2, w2, rank, True, None, ranks=8, meta_tile=tile)
        torch.cuda.synchronize()
    finally:
        rba.configure()
    assert torch.equal(out[0], ref[0]) and torch.equal(out[2], ref[2])
    assert torch.equal(out[1].view(torch.int16), ref[1].view(torch.int16))
    nnc = int(ref[3]["nnc"][0])
    assert int(out[3]["nnc"][0]) == nnc
    for name in ("e2p", "c2b", "c2m"):
        n = None if name == "e2p" else nnc
        assert torch.equal(out[3][name].reshape(-1)[:n], ref[3][name].reshape(-1)[:n]), name


# --------------------------------------------------------------------------------------------------
# W2: the 65..128-row direct-hop producer
# --------------------------------------------------------------------------------------------------
@pytest.fixture
def hop128():
    sk = _module("ops.gemm.splitk_gemm")
    wide = _module("ops.allreduce.post_attn_hop_wide")
    if not wide.HAS_SCATTER_GEMM:
        pytest.skip("no scatter producer (CuTe DSL / flashinfer split-K unavailable)")
    sk.configure(wide64=True)
    wide.configure(direct=True, direct_max_rows=128)
    try:
        yield wide
    finally:
        wide.configure()
        sk.configure()


class _FakeWindow:
    """Eight sources' scatter areas on this GPU: source p's area at p * stride (int32 words)."""

    SENT = -(2**16) - 2  # 0xFFFEFFFE, the direct sentinel

    def __init__(self, max_rows: int):
        self.max_rows = max_rows
        self.slots = max_rows // 8
        self.src_el = self.slots * HIDDEN
        self.area_el = 2 * 8 * self.src_el  # [2 parities][8 sources][slots][8192]
        self.buf = torch.full((8 * self.area_el // 2,), self.SENT, dtype=torch.int32, device="cuda")
        self.stride_b = self.area_el * 2

    def view(self, parity: int, rank: int) -> torch.Tensor:
        flat = self.buf.view(torch.bfloat16)
        off = (parity * 8 + rank) * self.src_el
        return flat[off : off + self.max_rows * HIDDEN].view(self.max_rows, HIDDEN)

    def reset(self) -> None:
        self.buf.fill_(self.SENT)

    def polled(self, parity: int, rank: int, rows: int) -> torch.Tensor:
        """The consumer's read of source ``rank``: token t at owner t % 8, slot t // 8 (fb layout)."""
        flat = self.buf.view(torch.bfloat16)
        out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
        for t in range(rows):
            base = (t % 8) * self.area_el + (parity * 8 + rank) * self.src_el
            area = flat[base : base + self.src_el].view(HIDDEN // 64, self.slots, 64)
            out[t] = area[:, t // 8, :].reshape(HIDDEN)
        return out

    def untouched_elsewhere(self, parities, rank: int, rows: int) -> bool:
        words = self.buf.view(8, 2, 8, HIDDEN // 64, self.slots, 32)
        mask = torch.ones_like(words, dtype=torch.bool)
        for q in parities:
            for t in range(rows):
                mask[t % 8, q, rank, :, t // 8, :] = False
        return bool((words[mask] == self.SENT).all())


class _Publication:
    """The production ``chunk_view`` / ``target_slots`` on a fake publication."""

    def __init__(self, wide, window: _FakeWindow):
        self.workspace = type(
            "W", (), {"max_rows": window.max_rows, "fb_layout": True, "ranks": 8}
        )()
        self.chunk_view = wide.DirectPublication.chunk_view.__get__(self)
        self.target_slots = wide.DirectPublication.target_slots.__get__(self)


def _produce(wide, mode, a, w, window, parity, rank):
    """o_proj of ``a`` into ``window`` as the dispatcher's ``_direct128`` / ``wide64`` would."""
    disp = _module("kernels.dense_gemms.dispatcher")
    pub = _Publication(wide, window)
    rows = a.shape[0]
    view = window.view(parity, rank)[:rows]
    fields = disp.parse_kernel(disp.SM107_TP8_WIDE64_KERNELS["o_proj"])[1]
    if mode == "chunks" or rows <= 64:
        for row0 in range(0, rows, 64):
            n = min(64, rows - row0)
            wide.run_splitk_scatter(
                a[row0 : row0 + n], w.t(), pub.chunk_view(view, row0, n), True,
                wide.SplitKTactic(*fields), window.stride_b, True, pub.target_slots(),
            )  # fmt: skip
        return
    stages = wide.DIRECT128_PRODUCER_STAGES[mode]
    wide.run_splitk_scatter_wide(
        a, w.t(), pub.chunk_view(view, 0, rows), True, wide.hop128_tactic(mode, stages),
        window.stride_b, pub.target_slots(),
    )  # fmt: skip


@pytest.mark.parametrize("mode", ["chunks", "grid", "tile128"])
@pytest.mark.parametrize("rows", [33, 40, 64, 65, 72, 96, 104, 127, 128])
def test_w2_direct_hop_producer_bits_and_addresses(hop128, mode, rows):
    wide = hop128
    if mode != "chunks" and rows <= 64:
        pytest.skip("a 33..64-row call has one launch in every mode")
    g = torch.Generator(device="cuda").manual_seed(rows)
    window = _FakeWindow(128)
    for trial in range(2):
        a = torch.randn(rows, K_OPROJ, device="cuda", generator=g) * (0.5 + trial)
        a = a.to(torch.bfloat16)
        w = (torch.randn(HIDDEN, K_OPROJ, device="cuda", generator=g) * 0.03).to(torch.bfloat16)
        if trial:
            a[::7] = 0
            a[1::11] = -0.0
        parity, rank = (rows + trial) % 2, (3 * rows + trial) % 8
        window.reset()
        _produce(wide, mode, a, w, window, parity, rank)
        torch.cuda.synchronize()
        got = window.polled(parity, rank, rows)
        want = F.linear(a, w).view(torch.int16)
        assert torch.equal(got.view(torch.int16), want), (mode, rows, trial)
        assert window.untouched_elsewhere((parity,), rank, rows), (mode, rows, trial)


def _randn(g: torch.Generator, rows: int, cols: int, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(rows, cols, device="cuda", generator=g) * scale).to(torch.bfloat16)


@pytest.mark.parametrize("mode", ["chunks", "grid"])
def test_w2_direct_hop_producer_graph_replays(hop128, mode):
    wide = hop128
    g = torch.Generator(device="cuda").manual_seed(7)
    window = _FakeWindow(128)
    for rows in (64, 96, 128):
        a = [_randn(g, rows, K_OPROJ) for _ in range(2)]
        w = [_randn(g, HIDDEN, K_OPROJ, 0.03) for _ in range(2)]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for p in range(2):
                _produce(wide, mode, a[p], w[p], window, p, 5)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for p in range(2):
                _produce(wide, mode, a[p], w[p], window, p, 5)
        for rep in range(4):
            for p in range(2):
                a[p].copy_(_randn(g, rows, K_OPROJ, 1 + rep))
            window.reset()
            graph.replay()
            torch.cuda.synchronize()
            for p in range(2):
                got = window.polled(p, 5, rows).view(torch.int16)
                want = F.linear(a[p], w[p]).view(torch.int16)
                assert torch.equal(got, want), (mode, rows, rep, p)
            assert window.untouched_elsewhere((0, 1), 5, rows), (mode, rows, rep)


# --------------------------------------------------------------------------------------------------
# W3: the one-launch split-K publication of 65..128-row slices
# --------------------------------------------------------------------------------------------------
@pytest.fixture
def publish128():
    sk = _module("ops.gemm.splitk_gemm")
    pp = _module("ops.comm.projection_publish")
    sk.configure(wide64=True)
    pp.configure(tep8_publish_splitk=True, tep8_publish_splitk_max_rows=128)
    try:
        # A served forward runs without grad, and _splitk_tactic declines under grad mode.
        with torch.no_grad():
            yield sk, pp
    finally:
        pp.configure()
        sk.configure()


@pytest.mark.parametrize("site", ["pool", "unpool"])
@pytest.mark.parametrize("rows", [65, 72, 96, 120, 128])
def test_w3_one_launch_publication_bits(publish128, site, rows):
    sk, pp = publish128
    if site == "pool":
        k, n = pp.UNPOOL_HIDDEN, pp.POOL_HIDDEN // 8
    else:
        k, n = pp.POOL_HIDDEN, pp.UNPOOL_HIDDEN // 8
    tactic = sk.SplitKTactic(*pp.TP8_PUBLISH_SPLITK_TACTICS[(k, n)])
    g = torch.Generator(device="cuda").manual_seed(rows + k)
    hidden = torch.randn(rows, k, device="cuda", generator=g).to(torch.bfloat16)
    hidden[::9] = 0
    weight = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    sentinel = pp._SENTINEL_I16
    buf = torch.full((rows + 8, n), sentinel, dtype=torch.int16, device="cuda")
    slot = buf[:rows].view(torch.bfloat16)
    assert pp._splitk_tactic(hidden, weight, slot) is not None or pp._splitk_publisher() is None
    pp._run_splitk_wide(hidden, weight, slot, tactic)
    torch.cuda.synchronize()
    assert not bool((buf[:rows] == sentinel).any()), "an element of the slot was not published"
    assert bool((buf[rows:] == sentinel).all()), "the publication wrote past its slot"
    chunks = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
    for row0 in range(0, rows, 64):
        row1 = min(rows, row0 + 64)
        sk.run_splitk_dense(hidden[row0:row1], weight.t(), None, chunks[row0:row1], True, tactic)
    requests = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
    for row0 in range(0, rows, 8):
        row1 = min(rows, row0 + 8)
        sk.run_splitk_dense(hidden[row0:row1], weight.t(), None, requests[row0:row1], True, tactic)
    torch.cuda.synchronize()
    assert torch.equal(slot.view(torch.int16), chunks.view(torch.int16))
    assert torch.equal(slot.view(torch.int16), requests.view(torch.int16))


# --------------------------------------------------------------------------------------------------
# W4: the TEP8 one-cluster SWA block (CL8) on 2..4- and 9..16-request steps
# --------------------------------------------------------------------------------------------------
SWA_HQ, SWA_HKV, SWA_D = 16, 2, 64  # a TEP8 rank's sliding-window layer: 16 query, 2 KV heads of 64
SWA_WINDOW, SWA_PAGE, SWA_ROWS = 2048, 32, 8  # 32-key FP8 pages; K = 7 verify: 8 rows per request
SWA_ROPE_POSITIONS = 8192
#: Each real request's length L (its keys after the step, the step's 8 rows included). Any prefix of
#: 9 mixes contexts beyond the window with shorter ones; the edges: L = 8 (the step alone), 9, 2047,
#: 2048 (the step ends a page), 2055 (past = window - 1: key 0 still attended), 2056 (key 0 dropped;
#: the step starts a page), and 65, 1031, 2087, 4133 (the step straddles two pages).
SWA_REAL_LENS = (2100, 40, 3001, 8, 2048, 777, 4133, 2047, 2056, 300, 2055, 9, 4096, 1031, 2087, 65)
#: The filler requests that pad the reference calls to their size (their rows are not compared).
SWA_FILLER_LENS = (2500, 16, 1500, 3333, 100, 2049, 640)
#: An E4M3 NaN byte: never in the cache CL8 writes (its conversion saturates), so a read of a slot
#: holding it shows in the output, and a step slot still holding it was not appended.
_E4M3_NAN = 0x7F


class _SwaStep:
    """A TEP8 sliding-window decode step: ``n`` real requests then the fillers, request-major.

    Request ``b`` has ``lens[b]`` keys after the step and its own K and V pages (ids shuffled).
    Page 0 (every table entry past a request's last page), the pages wholly before a request's
    window, its step rows' slots and the slots after its last key hold ``_E4M3_NAN``; every
    other cache byte is a random finite E4M3 value. Pages ``[0, real_pages)`` are page 0 and the
    real requests', the fillers' come after.
    """

    def __init__(self, n: int, permuted: bool, pos_dtype: torch.dtype, seed: int):
        cpu = torch.Generator().manual_seed(seed)
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.n, self.permuted = n, permuted
        self.lens = SWA_REAL_LENS[:n] + SWA_FILLER_LENS
        pages = [-(-length // SWA_PAGE) for length in self.lens]
        self.real_pages = 1 + 2 * sum(pages[:n])
        total = self.real_pages + 2 * sum(pages[n:])
        free = [
            (1 + torch.randperm(self.real_pages - 1, generator=cpu)).tolist(),
            (self.real_pages + torch.randperm(total - self.real_pages, generator=cpu)).tolist(),
        ]
        bt = torch.zeros(len(self.lens), 2, max(pages), dtype=torch.int32)
        step = torch.zeros(total, SWA_PAGE, dtype=torch.bool)
        poison = torch.zeros(total, SWA_PAGE, dtype=torch.bool)
        poison[0] = True
        for b, length in enumerate(self.lens):
            past = length - SWA_ROWS
            first = max(0, past - (SWA_WINDOW - 1)) // SWA_PAGE
            for kv in range(2):
                for p in range(pages[b]):
                    bt[b, kv, p] = free[b >= n].pop()
                for pos in range(past, length):
                    step[int(bt[b, kv, pos // SWA_PAGE]), pos % SWA_PAGE] = True
                poison[int(bt[b, kv, pages[b] - 1]), (length - 1) % SWA_PAGE + 1 :] = True
                poison[bt[b, kv, :first].long()] = True
        real = step.clone()
        real[self.real_pages :] = False
        #: The real requests' step-row slots: what an n-request call appends.
        self.step_real = real.cuda()
        cache = torch.randn(total, SWA_HKV, SWA_PAGE, SWA_D, device="cuda", generator=g) * 0.5
        self.pool = cache.to(torch.float8_e4m3fn)
        nan = (step | poison).cuda()[:, None, :, None]
        self.pool.view(torch.uint8).masked_fill_(nan, _E4M3_NAN)
        self.bt = bt.cuda()
        self.seq_lens = torch.tensor(self.lens, dtype=torch.int32, device="cuda")
        width = 2 * (SWA_HQ + SWA_HKV) * SWA_D  # the raw projection: q, gate, k, v
        qkv = torch.randn(len(self.lens) * SWA_ROWS, width, device="cuda", generator=g)
        self.qkv = qkv.to(torch.bfloat16)
        self.positions = torch.tensor(
            [length - SWA_ROWS + t for length in self.lens for t in range(SWA_ROWS)],
            dtype=pos_dtype,
            device="cuda",
        )
        inv = 1.0 / 10000.0 ** (torch.arange(0, SWA_D, 2, dtype=torch.float64) / SWA_D)
        angle = torch.arange(SWA_ROPE_POSITIONS, dtype=torch.float64)[:, None] * inv
        self.cos_sin = torch.stack([angle.cos(), angle.sin()], 1).float().contiguous().cuda()
        self.kv_scale = torch.tensor([0.75], dtype=torch.float32, device="cuda")

    def inputs(self, ids) -> dict:
        """The fresh, contiguous tensors of one call of requests ``ids`` (in that order)."""
        index = torch.tensor(ids, device="cuda")
        rows = (index[:, None] * SWA_ROWS + torch.arange(SWA_ROWS, device="cuda")).reshape(-1)
        return {
            "qkv": self.qkv.index_select(0, rows),
            "positions": self.positions.index_select(0, rows),
            "seq_lens": self.seq_lens.index_select(0, index),
            "bt": self.bt.index_select(0, index),
        }


def _swa_gate(qkv: torch.Tensor, permuted: bool) -> torch.Tensor:
    """The output gate where the projection put it (``_SwaBlock._raw_gate``)."""
    rows = qkv.shape[0]
    if permuted:
        return qkv[:, (SWA_HQ + 2 * SWA_HKV) * SWA_D :].view(rows, SWA_HQ, SWA_D)
    return qkv[:, : 2 * SWA_HQ * SWA_D].view(rows, SWA_HQ, 2 * SWA_D)[..., SWA_D:]


def _swa_call(swa, step: _SwaStep, x: dict, pool: torch.Tensor) -> torch.Tensor:
    """``swa_block_multi_attention`` of one call, as ``_SwaBlock._swa_block_call`` makes it on a
    TEP8 sliding-window layer under swa_block's plan (fused prep and norm-gate, q_fp8, PDL)."""
    splits = _module("ops.attention.paged_decode_attention").window_splits(SWA_WINDOW)
    return swa.swa_block_multi_attention(
        x["qkv"], pool, x["bt"], x["seq_lens"], step.kv_scale, SWA_HQ, SWA_HKV, SWA_WINDOW, splits,
        True, fused_prep=True, cos_sin=step.cos_sin, position_ids=x["positions"], prep_eps=1e-6,
        q_scale=1.0, qkv_permuted=step.permuted, gate=_swa_gate(x["qkv"], step.permuted),
        gate_eps=1e-6, enable_pdl=True,
    )  # fmt: skip


def _swa_layout(swa, step: _SwaStep, calls):
    """Run ``calls`` (the request ids of each) in turn on a fresh copy of the cache; returns each
    request's output rows and the cache. Each call starts on an idle device, as in a decode step
    (the block reads the tables and the older cache bytes before its dependency wait)."""
    pool = step.pool.clone()
    rows = {}
    for ids in calls:
        x = step.inputs(ids)
        torch.cuda.synchronize()
        out = _swa_call(swa, step, x, pool)
        for k, b in enumerate(ids):
            rows[b] = out[k * SWA_ROWS : (k + 1) * SWA_ROWS]
    torch.cuda.synchronize()
    return rows, pool


def _cl8_max_active_clusters(swa, requests: int):
    """Co-resident clusters of the TEP8 CL8 build on this device (informational), or None."""
    try:
        from cuda.bindings import driver as cu

        fn = swa._cl8_function(torch.device("cuda", torch.cuda.current_device()), False, 2)
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = 8, SWA_HKV, requests
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 320, 1, 1
        cfg.sharedMemBytes = swa._CL8_SMEM
        cluster = cu.CUlaunchAttribute()
        cluster.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        cluster.value.clusterDim.x, cluster.value.clusterDim.y, cluster.value.clusterDim.z = 8, 1, 1
        cfg.attrs = [cluster]
        cfg.numAttrs = 1
        err, clusters = cu.cuOccupancyMaxActiveClusters(fn[0], cfg)
        return int(clusters) if int(err) == 0 else None
    except Exception:  # noqa: BLE001 - informational only
        return None


#: The fixture's TEP8 CL8 band: swa_one_cluster_block__TEP8_MIN_REQUESTS=2 and __TEP8_MAX_REQUESTS=16.
CL8_BAND = dict(cl8_tep8=True, cl8_tep8_min_requests=2, cl8_tep8_max_requests=16, one2=True)


@pytest.fixture
def cl8_tep8(monkeypatch):
    """swa_block with CL8 on TEP8 steps of 2..16 requests (and one2, whose routing CL8-TEP8
    extends) and the 2-KV-head build compiled; yields the module and the request count of every
    CL8 launch from then on."""
    swa = _module("ops.attention.swa_block")
    try:
        from cuda.bindings import driver, nvrtc  # noqa: F401
    except ImportError as error:
        pytest.skip(f"the CL8 build needs cuda-python's NVRTC and driver bindings: {error}")
    swa.configure(**CL8_BAND)
    # The CL8 build loads its cubin through the driver API, which needs the device's primary context
    # current on this thread: a fresh pytest process has none until torch touches the device
    # (VR run: CUDA_ERROR_INVALID_CONTEXT, every W4 case skipped). The server always has one.
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    try:
        try:
            swa._cl8_function(torch.device("cuda", torch.cuda.current_device()), False, 2)
        except RuntimeError as error:
            pytest.skip(f"no TEP8 CL8 build on this device: {error}")
        launches = []
        launch = swa._cl8_launch

        def counted(fn, qkv, kv_pool, bt, seq_lens, *args):
            launches.append(seq_lens.numel())
            return launch(fn, qkv, kv_pool, bt, seq_lens, *args)

        monkeypatch.setattr(swa, "_cl8_launch", counted)
        yield swa, launches
    finally:
        swa.configure()


@pytest.mark.parametrize("variant", ["checkpoint-int32", "permuted-int64"])
@pytest.mark.parametrize("n", [2, 3, 4, 9, 12, 13, 16])
def test_w4_cl8_tep8_wide_step_is_batch_invariant(cl8_tep8, n, variant):
    swa, launches = cl8_tep8
    permuted = variant.startswith("permuted")
    pos_dtype = torch.int64 if variant.endswith("int64") else torch.int32
    step = _SwaStep(n, permuted, pos_dtype, seed=400 + n + 100 * permuted)
    reals = list(range(n))
    fillers = list(range(n, n + len(SWA_FILLER_LENS)))

    # The n-request call: CL8 launches it; it reads no poisoned byte, appends every step row of
    # every request and writes no other cache byte.
    rows_a, pool_a = _swa_layout(swa, step, [reals])
    assert launches == [n], "the n-request call did not launch CL8"
    out_a = torch.cat([rows_a[b] for b in reals])
    assert bool(torch.isfinite(out_a.float()).all()), "the call read a poisoned cache byte"
    raw0, raw_a = step.pool.view(torch.uint8), pool_a.view(torch.uint8)
    mine = step.step_real[:, None, :, None].expand(raw0.shape)
    assert not bool(((raw_a[mine] & 0x7F) == 0x7F).any()), "a step row's K/V byte was not appended"
    assert torch.equal(raw_a[~mine], raw0[~mine]), "the call wrote outside its step rows"

    # (a) + (b): each request's rows, and the K/V bytes appended for it, inside certified
    # 5..8-request calls (and one shuffled call of up to 16) are the n-request call's.
    def padded(ids, size):
        return ids + fillers[: size - len(ids)]

    # 16 at 9..16 requests; n + the 7 fillers (9..11) at 2..4.
    size = min(16, n + len(fillers))
    members = reals + fillers[: size - n]
    shuffle = torch.randperm(size, generator=torch.Generator().manual_seed(n)).tolist()
    layouts = {
        "calls of 8 in order": [padded(reals[i : i + 8], 8) for i in range(0, n, 8)],
        "calls of 5 reversed": [padded(reals[::-1][i : i + 5], 5) for i in range(0, n, 5)],
        f"one shuffled call of {size}": [[members[i] for i in shuffle]],
    }
    if n <= swa.MULTI_MAX_REQUESTS:
        # The new band's smallest call: pairs, in reverse order.
        layouts["calls of 2 reversed"] = [padded(reals[::-1][i : i + 2], 2) for i in range(0, n, 2)]
    for name, calls in layouts.items():
        launches.clear()
        rows_x, pool_x = _swa_layout(swa, step, calls)
        assert launches == [len(ids) for ids in calls], f"{name}: a call did not launch CL8"
        for b in reals:
            assert torch.equal(rows_x[b].view(torch.int16), rows_a[b].view(torch.int16)), (
                f"{name}: request {b} (L = {step.lens[b]}) differs from the {n}-request call"
            )
        assert torch.equal(pool_x.view(torch.uint8)[: step.real_pages], raw_a[: step.real_pages]), (
            f"{name}: the appended K/V bytes differ from the {n}-request call's"
        )

    # (c): one CUDA-graph capture of the n-request call replays the eager bits, then reads new
    # projections as an eager call does.
    x = step.inputs(reals)
    pool_g = step.pool.clone()
    launches.clear()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _swa_call(swa, step, x, pool_g)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ready = swa.cl8_tep8_ready(torch.cuda.current_device())
        out_g = _swa_call(swa, step, x, pool_g)
    assert ready, "cl8_tep8_ready is False inside a capture with the TEP8 build compiled"
    assert launches == [n, n], "the captured call did not launch CL8"
    g = torch.Generator(device="cuda").manual_seed(n)
    for rep in range(3):
        if rep == 2:
            fresh = torch.randn(x["qkv"].shape, device="cuda", generator=g)
            x["qkv"].copy_(fresh.to(torch.bfloat16))
        pool_g.copy_(step.pool)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        if rep < 2:
            want_out, want_pool = out_a, raw_a
        else:
            pool_e = step.pool.clone()
            torch.cuda.synchronize()
            want_out = _swa_call(swa, step, x, pool_e)
            torch.cuda.synchronize()
            want_pool = pool_e.view(torch.uint8)
        assert torch.equal(out_g.view(torch.int16), want_out.view(torch.int16)), f"replay {rep}"
        assert torch.equal(pool_g.view(torch.uint8), want_pool), f"replay {rep}: cache bytes"

    if variant == "checkpoint-int32":
        clusters = _cl8_max_active_clusters(swa, n)
        waves = "" if clusters is None else (" (one wave)" if 2 * n <= clusters else " (2+ waves)")
        print(
            f"[W4] CL8 TEP8 at {n} requests: {2 * n} clusters of 8 CTAs; co-resident clusters on "
            f"this device: {clusters}{waves}"
        )


@pytest.mark.parametrize("n", [2, 3, 4])
def test_w4_cl8_tep8_few_request_steps_launch_cl8_only_in_its_band(cl8_tep8, n):
    """A 2..4-request TEP8 step launches CL8 exactly when TEP8_MIN_REQUESTS <= n (under one2), with
    the bits of the n-request CL8 call; at a higher minimum (5 is the default), without one2, or
    with CL8 off it runs the split kernels, as before the parameter. Either path appends the same
    K/V bytes (CL8's append is the stock's)."""
    swa, launches = cl8_tep8
    step = _SwaStep(n, False, torch.int32, seed=700 + n)
    reals = list(range(n))
    real = step.real_pages
    rows_c, pool_c = _swa_layout(swa, step, [reals])
    assert launches == [n], "the fixture's band (2..16) did not launch CL8"
    raw_c = pool_c.view(torch.uint8)[:real]
    cases = [
        (f"TEP8_MIN_REQUESTS={low}", dict(CL8_BAND, cl8_tep8_min_requests=low))
        for low in range(2, 6)
    ]
    cases += [
        ("one2 off", dict(CL8_BAND, one2=False)),
        ("cl8_tep8 off", dict(CL8_BAND, cl8_tep8=False)),
    ]
    for name, case in cases:
        swa.configure(**case)
        launches.clear()
        rows_x, pool_x = _swa_layout(swa, step, [reals])
        cl8 = case["cl8_tep8"] and case["one2"] and case["cl8_tep8_min_requests"] <= n
        assert launches == ([n] if cl8 else []), f"{name}: CL8 launches {launches} at {n} requests"
        if cl8:
            for b in reals:
                assert torch.equal(rows_x[b].view(torch.int16), rows_c[b].view(torch.int16)), (
                    f"{name}: request {b} differs from the first CL8 call"
                )
        assert torch.equal(pool_x.view(torch.uint8)[:real], raw_c), (
            f"{name}: the appended K/V bytes differ from CL8's ({'CL8' if cl8 else 'split kernels'})"
        )


def test_w4_cl8_tep8_wide_steps_route_only_through_cl8(monkeypatch):
    """Host side: the caller sends a 9..16-request TEP8 step to the block only under one2 and only
    while CL8 launches it (built, or no capture running); multi_request_prep_append stands down for
    every step the caller sends; ``swa_block_multi`` refuses 9+ sequences without CL8. The band's
    first count (TEP8_MIN_REQUESTS) moves none of that: 2..4-request steps reach the block in every
    case, and a 2..4-request step CL8 cannot launch (a capture before the build) runs the split
    kernels."""
    swa = _module("ops.attention.swa_block")
    block = _module("kernels.swa_block.attention")

    class _Layer(block._SwaBlock):
        is_sliding = True
        num_key_value_heads = 2
        # multi_request_prep_append's fewest sequences under wide_swa_block (layers.py).
        _mr_append_min_seqs = 1 + swa.MULTI_MAX_REQUESTS

    def routed(layer):
        sent = []
        for n in range(2, 18):
            md = types.SimpleNamespace(
                num_seqs=n,
                num_tokens=n * SWA_ROWS,
                num_generations=n,
                seq_lens=torch.full((n,), SWA_ROWS, dtype=torch.int32),
            )
            if layer._step_requests(md) == n:
                sent.append(n)
        return sent

    built = {}
    capturing = [False]
    monkeypatch.setattr(swa, "_CL8_FN", built)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    try:
        for one2, cl8, most, least in (
            (True, True, 16, 5),
            (True, True, 13, 5),
            (True, True, 8, 5),
            (True, False, 16, 5),
            (False, True, 16, 5),
            (False, True, 8, 5),
            (True, True, 16, 2),
            (True, True, 8, 2),
            (False, True, 8, 2),
        ):
            case = dict(
                one2=one2, cl8_tep8=cl8, cl8_tep8_max_requests=most, cl8_tep8_min_requests=least
            )
            swa.configure(**case)
            layer = _Layer()
            model = types.SimpleNamespace(
                layers=[types.SimpleNamespace(self_attn=layer, layer_idx=0)]
            )
            block.install_multi(model)
            top = (most if cl8 else 8) if one2 else swa.MULTI_MAX_REQUESTS
            assert swa.cl8_tep8_requests() == top, case
            capturing[0] = False  # an eager step: the call compiles CL8
            assert routed(layer) == list(range(2, top + 1)), case
            assert top < layer._mr_append_min_seqs, case
            capturing[0] = True  # a capture before the build: the steps above 8 decline
            assert routed(layer) == list(range(2, min(top, 8) + 1)), case
            built[(torch.cuda.current_device(), False, 2)] = None  # built: a capture takes them
            assert routed(layer) == list(range(2, top + 1)), case
            built.clear()
            capturing[0] = False

        step = _SwaStep(9, False, torch.int32, seed=49)
        x = step.inputs(list(range(9)))
        # As inside a capture before the build: the op launches nothing and refuses.
        monkeypatch.setattr(swa, "_cl8_function", lambda *args: None)
        swa.configure(**CL8_BAND)
        with pytest.raises(ValueError, match="only through CL8"):
            _swa_call(swa, step, x, step.pool.clone())
        # A 3-request step in the band: the split kernels take it instead.
        few = _SwaStep(3, False, torch.int32, seed=53)
        out = _swa_call(swa, few, few.inputs([0, 1, 2]), few.pool.clone())
        torch.cuda.synchronize()
        assert tuple(out.shape) == (3 * SWA_ROWS, SWA_HQ * SWA_D)
        for one2, cl8, most in ((False, True, 16), (True, True, 8), (True, False, 16)):
            swa.configure(cl8_tep8=cl8, cl8_tep8_max_requests=most, one2=one2)
            with pytest.raises(ValueError, match="swa_block_multi takes"):
                _swa_call(swa, step, x, step.pool.clone())
    finally:
        swa.configure()


# --------------------------------------------------------------------------------------------------
# W5: the 2..4-request band (9..32 rows) of the 5..8-request family (``tep8_publish_splitk__MIN_ROWS``,
# ``tep8_direct_hop__MIN_ROWS``, ``tep8_wide64_splitk__MIN_ROWS``)
# --------------------------------------------------------------------------------------------------
#: Row counts of the band: 2..4 requests at K = 7 (16, 24, 32), their first rows, and odd counts.
LOW_ROWS = [9, 16, 17, 24, 25, 31, 32]
#: Where a low-band step's rows sit inside the 33..64-row reference call (a 48-row call).
LOW_OFFSET, LOW_BIG = 16, 48


def _publish_sites(pp):
    """(site, K, N) of the two TP8 slice GEMMs."""
    return [
        ("pool", pp.UNPOOL_HIDDEN, pp.POOL_HIDDEN // 8),
        ("unpool", pp.POOL_HIDDEN, pp.UNPOOL_HIDDEN // 8),
    ]


@pytest.fixture
def publish_low():
    sk = _module("ops.gemm.splitk_gemm")
    pp = _module("ops.comm.projection_publish")
    sk.configure(wide64=True)
    pp.configure(tep8_publish_splitk=True, tep8_publish_splitk_min_rows=9)
    try:
        if pp._splitk_publisher() is None:
            pytest.skip(
                "the split-K producer does not publish here (its census key is not certified on this "
                "device / image): projection_publish logs why"
            )
        # The band below 33 is served only with the certified key at its first row count.
        keys = [
            pp._splitk_site_key(sk, pp.TP8_PUBLISH_SPLITK_TACTICS[(k, n)], (9,))
            for _, k, n in _publish_sites(pp)
        ]
        assert pp._splitk_state.get("low"), (
            f"the split-K producer key at 9 rows is not certified: {keys}"
        )
        assert pp.publish_min_rows() == 9 and pp.splitk_min_rows() == 9
        # A served forward runs without grad, and _splitk_tactic declines under grad mode.
        with torch.no_grad():
            yield sk, pp
    finally:
        pp.configure()
        sk.configure()


def _publish_into_slot(pp, hidden, weight, extra: int = 8):
    """``publish_projection`` into a sentinel-filled slot with ``extra`` sentinel rows past it."""
    rows, n = hidden.shape[0], weight.shape[0]
    buf = torch.full((rows + extra, n), pp._SENTINEL_I16, dtype=torch.int16, device="cuda")
    slot = buf[:rows].view(torch.bfloat16)
    pp.publish_projection(hidden, weight, slot)
    return buf, slot


@pytest.mark.parametrize("rows", LOW_ROWS)
def test_w5_publication_low_rows_bits(publish_low, rows):
    """A 9..32-row slice publishes from the split-K producer (never cuBLAS): every element of its slot
    once, nothing past it, and each row is bit for bit the same row inside a 48-row (33..64-row band)
    publication, its one-request (8-row) call, and the 9..32-row wide table's kernel of the gathers path
    it replaces."""
    sk, pp = publish_low
    disp = _module("kernels.dense_gemms.dispatcher")
    dk = _module("ops.gemm.dense_gemm")
    wide_table = disp.wide_kernels(8)
    for site, k, n in _publish_sites(pp):
        tactic = sk.SplitKTactic(*pp.TP8_PUBLISH_SPLITK_TACTICS[(k, n)])
        g = torch.Generator(device="cuda").manual_seed(rows * 31 + k)
        hidden = _randn(g, rows, k)
        hidden[::7] = 0
        hidden[1::5] = -0.0
        weight = _randn(g, n, k, 0.02)
        assert pp._splitk_tactic(hidden, weight, hidden.new_empty(rows, n)) is not None, site
        buf, slot = _publish_into_slot(pp, hidden, weight)
        # The 33..64-row path: the same rows inside a 48-row call.
        big = torch.cat(
            [_randn(g, LOW_OFFSET, k), hidden, _randn(g, LOW_BIG - LOW_OFFSET - rows, k)]
        )
        _, big_slot = _publish_into_slot(pp, big, weight)
        # Each request (8 rows) alone, and the gathers path's producer at this row count.
        requests = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
        for row0 in range(0, rows, 8):
            row1 = min(rows, row0 + 8)
            sk.run_splitk_dense(
                hidden[row0:row1], weight.t(), None, requests[row0:row1], True, tactic
            )
        gathers = dk.dense_gemm(hidden, weight, wide_table[site][rows - 9], True)
        torch.cuda.synchronize()
        sentinel = pp._SENTINEL_I16
        assert not bool((buf[:rows] == sentinel).any()), (
            site,
            rows,
            "an element was not published",
        )
        assert bool((buf[rows:] == sentinel).all()), (
            site,
            rows,
            "the publication wrote past its slot",
        )
        got = slot.view(torch.int16)
        assert torch.equal(got, big_slot[LOW_OFFSET : LOW_OFFSET + rows].view(torch.int16)), (
            site,
            rows,
        )
        assert torch.equal(got, requests.view(torch.int16)), (site, rows)
        assert torch.equal(got, gathers.view(torch.int16)), (site, rows, wide_table[site][rows - 9])


def test_w5_publication_low_rows_never_falls_back_to_cublas(publish_low):
    """Below 33 rows a slice ``_splitk_tactic`` declines for a rank-local reason (grad mode, an
    activation that is not 32-byte aligned) still publishes from the split-K producer (never cuBLAS,
    never a raise), with the bits the accepted call publishes."""
    sk, pp = publish_low
    g = torch.Generator(device="cuda").manual_seed(29)
    for site, k, n in _publish_sites(pp):
        hidden, weight = _randn(g, 24, k), _randn(g, n, k, 0.02)
        _, want = _publish_into_slot(pp, hidden, weight)
        with torch.enable_grad():
            assert pp._splitk_tactic(hidden, weight, hidden.new_empty(24, n)) is None
            buf, slot = _publish_into_slot(pp, hidden, weight)
        torch.cuda.synchronize()
        assert torch.equal(slot.view(torch.int16), want.view(torch.int16)), site
        assert bool((buf[24:] == pp._SENTINEL_I16).all()), site
        # An activation 16 bytes off its alignment: published from an aligned copy.
        storage = torch.empty(24 * k + 8, dtype=torch.bfloat16, device="cuda")
        shifted = storage[8:].view(24, k)
        shifted.copy_(hidden)
        assert shifted.data_ptr() % 32 == 16
        buf, slot = _publish_into_slot(pp, shifted, weight)
        torch.cuda.synchronize()
        assert torch.equal(slot.view(torch.int16), want.view(torch.int16)), site
    assert any(key[0] == "copy" for key in pp._LOW_ROWS_COPIES)


def test_w5_publication_low_rows_graph_replays(publish_low):
    """Both sites' 24- and 32-row publications captured in one CUDA graph replay the eager bits."""
    _, pp = publish_low
    g = torch.Generator(device="cuda").manual_seed(5)
    calls = []
    for rows in (24, 32):
        for _, k, n in _publish_sites(pp):
            hidden, weight = _randn(g, rows, k), _randn(g, n, k, 0.02)
            buf = torch.full((rows + 8, n), pp._SENTINEL_I16, dtype=torch.int16, device="cuda")
            calls.append((hidden, weight, buf, buf[:rows].view(torch.bfloat16)))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for hidden, weight, _, slot in calls:
            pp.publish_projection(hidden, weight, slot)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for hidden, weight, _, slot in calls:
            pp.publish_projection(hidden, weight, slot)
    for rep in range(4):
        for hidden, _, buf, _ in calls:
            hidden.copy_(_randn(g, *hidden.shape, scale=1 + rep))
            buf.fill_(pp._SENTINEL_I16)
        graph.replay()
        torch.cuda.synchronize()
        for hidden, weight, buf, slot in calls:
            want = torch.empty_like(slot)
            pp.publish_projection(hidden, weight, want)
            torch.cuda.synchronize()
            assert torch.equal(slot.view(torch.int16), want.view(torch.int16)), (hidden.shape, rep)
            assert bool((buf[hidden.shape[0] :] == pp._SENTINEL_I16).all()), (hidden.shape, rep)


@pytest.fixture
def hop_low():
    sk = _module("ops.gemm.splitk_gemm")
    wide = _module("ops.allreduce.post_attn_hop_wide")
    disp = _module("kernels.dense_gemms.dispatcher")
    if not wide.HAS_SCATTER_GEMM:
        pytest.skip("no scatter producer (CuTe DSL / flashinfer split-K unavailable)")
    sk.configure(wide64=True)
    disp.configure(tep8_wide64=True)
    wide.configure(direct=True, direct_min_rows=9)
    try:
        assert wide.SERVED_DIRECT_ROWS == (9, 64)
        # A served forward runs without grad, and the dispatcher's wide64 declines under grad mode.
        with torch.no_grad():
            yield wide, disp
    finally:
        wide._DIRECT_PUBLICATION = None
        wide.configure()
        disp.configure()
        sk.configure()


@pytest.mark.parametrize("rows", LOW_ROWS)
def test_w5_direct_hop_producer_low_rows(hop_low, rows):
    """o_proj of a 9..32-row step on the served wide64 scatter tactic (the 64-row workspace's 8 slots)
    lands every token / feature where the direct consumer polls it and nothing else, and each row is bit
    for bit the same row inside a 48-row (33..64-row band) scatter call and the 9..32-row wide table's
    split_k-1 o_proj (today's partial at these rows)."""
    wide, disp = hop_low
    dk = _module("ops.gemm.dense_gemm")
    kernel = disp.wide_kernels(8)["o_proj"][rows - 9]
    g = torch.Generator(device="cuda").manual_seed(500 + rows)
    window, big_window = _FakeWindow(64), _FakeWindow(64)
    for trial in range(2):
        a = _randn(g, rows, K_OPROJ, 0.5 + trial)
        if trial:
            a[::7] = 0
            a[1::11] = -0.0
        w = _randn(g, HIDDEN, K_OPROJ, 0.03)
        parity, rank = (rows + trial) % 2, (5 * rows + trial) % 8
        window.reset()
        big_window.reset()
        _produce(wide, "chunks", a, w, window, parity, rank)
        big = torch.cat(
            [_randn(g, LOW_OFFSET, K_OPROJ), a, _randn(g, LOW_BIG - LOW_OFFSET - rows, K_OPROJ)]
        )
        _produce(wide, "chunks", big, w, big_window, parity, rank)
        today = dk.dense_gemm(a, w, kernel, True)
        torch.cuda.synchronize()
        got = window.polled(parity, rank, rows).view(torch.int16)
        in_big = big_window.polled(parity, rank, LOW_BIG)[LOW_OFFSET : LOW_OFFSET + rows]
        assert torch.equal(got, in_big.view(torch.int16)), (rows, trial)
        assert torch.equal(got, today.view(torch.int16)), (rows, trial, kernel)
        assert window.untouched_elsewhere((parity,), rank, rows), (rows, trial)


def test_w5_direct_hop_producer_low_rows_graph_replays(hop_low):
    wide, disp = hop_low
    dk = _module("ops.gemm.dense_gemm")
    g = torch.Generator(device="cuda").manual_seed(17)
    window = _FakeWindow(64)
    for rows in (16, 24, 32):
        kernel = disp.wide_kernels(8)["o_proj"][rows - 9]
        a = [_randn(g, rows, K_OPROJ) for _ in range(2)]
        w = [_randn(g, HIDDEN, K_OPROJ, 0.03) for _ in range(2)]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for p in range(2):
                _produce(wide, "chunks", a[p], w[p], window, p, 6)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for p in range(2):
                _produce(wide, "chunks", a[p], w[p], window, p, 6)
        for rep in range(4):
            for p in range(2):
                a[p].copy_(_randn(g, rows, K_OPROJ, 1 + rep))
            window.reset()
            graph.replay()
            torch.cuda.synchronize()
            for p in range(2):
                want = dk.dense_gemm(a[p], w[p], kernel, True)
                torch.cuda.synchronize()
                got = window.polled(p, 6, rows).view(torch.int16)
                assert torch.equal(got, want.view(torch.int16)), (rows, rep, p)
            assert window.untouched_elsewhere((0, 1), 6, rows), (rows, rep)


class _LowPublication:
    """A direct publication armed for one step of ``rows`` rows on ``window`` (parity 1, rank 2)."""

    def __init__(self, window: _FakeWindow, dispatcher, rows) -> None:
        self.window, self.dispatcher, self.rows = window, dispatcher, rows

    def target(self, dispatcher, rows):
        if dispatcher is not self.dispatcher or rows != self.rows:
            return None
        return self.window.view(1, 2)[:rows], self.window.stride_b, True

    def target_slots(self) -> int:
        return self.window.max_rows // 8


def _fake_dispatcher(disp, role: str, shape):
    plan = types.SimpleNamespace(
        tp_size=8, enable_pdl=True, named_forks={}, join=lambda: None, join_role=lambda role: None
    )
    return disp.DenseGemmDispatcher(
        types.SimpleNamespace(role=role, plan=plan, shape=shape, layer_idx=0)
    )


def test_w5_dispatcher_routes_the_low_band(hop_low):
    """``DenseGemmDispatcher.wide64`` below 33 rows: the o_proj of a step the direct publication armed
    scatters (the served tactic) whatever the wide64 floor; every other call keeps the 9..32-row wide
    table (``None``) until ``tep8_wide64_min_rows`` moves the floor, after which QKV and o_proj run the
    wide64 tactics with the wide table's bits (each site's split_k)."""
    wide, disp = hop_low
    dk = _module("ops.gemm.dense_gemm")
    table = disp.wide_kernels(8)
    o_proj = _fake_dispatcher(disp, "o_proj", (K_OPROJ, HIDDEN))
    qkv = _fake_dispatcher(disp, "qkv", dk.QKV_SHAPE)
    g = torch.Generator(device="cuda").manual_seed(23)
    window = _FakeWindow(64)
    for rows in (16, 24, 32):
        a, w = _randn(g, rows, K_OPROJ), _randn(g, HIDDEN, K_OPROJ, 0.03)
        q_in = _randn(g, rows, dk.QKV_SHAPE[0])
        q_w = _randn(g, dk.QKV_SHAPE[1], dk.QKV_SHAPE[0], 0.02)
        want = dk.dense_gemm(a, w, table["o_proj"][rows - 9], True).view(torch.int16)
        q_want = dk.dense_gemm(q_in, q_w, table["qkv"][rows - 9], True).view(torch.int16)
        disp.configure(tep8_wide64=True)
        wide._DIRECT_PUBLICATION = None
        assert o_proj.wide64(a, w, None) is None and qkv.wide64(q_in, q_w, None) is None
        wide._DIRECT_PUBLICATION = _LowPublication(window, o_proj, rows)
        window.reset()
        out = o_proj.wide64(a, w, None)
        torch.cuda.synchronize()
        assert out is not None and out.data_ptr() == window.view(1, 2).data_ptr(), rows
        assert torch.equal(window.polled(1, 2, rows).view(torch.int16), want), rows
        assert window.untouched_elsewhere((1,), 2, rows), rows
        assert qkv.wide64(q_in, q_w, None) is None  # QKV keeps the wide table
        wide._DIRECT_PUBLICATION = None
        disp.configure(tep8_wide64=True, tep8_wide64_min_rows=rows)
        out = o_proj.wide64(a, w, None)
        q_out = qkv.wide64(q_in, q_w, None)
        torch.cuda.synchronize()
        assert out is not None and torch.equal(out.view(torch.int16), want), rows
        assert q_out is not None and torch.equal(q_out.view(torch.int16), q_want), rows


# --------------------------------------------------------------------------------------------------
# W6: 65..128-row TP8 QKV in one launch of the wide64 tactic (``tep8_wide64_splitk__MAX_ROWS``)
# --------------------------------------------------------------------------------------------------
W6_ROWS = [65, 72, 96, 104, 120, 128]


def _w6_configure(disp, max_rows: int = 128, tile: str = "64x64") -> None:
    """The dispatcher as ``lever_settings.apply`` configures it with the lever on and the knobs set."""
    disp.configure(
        tep8_wide64=True,
        tep8_wide64_kernels=dict(_module("lever_settings").TEP8_WIDE64_KERNELS),
        tep8_wide64_max_rows=max_rows,
        tep8_wide64_wide_tile=tile,
    )


@pytest.fixture
def qkv_wide128():
    sk = _module("ops.gemm.splitk_gemm")
    disp = _module("kernels.dense_gemms.dispatcher")
    dk = _module("ops.gemm.dense_gemm")
    sk.configure(wide64=True)
    _w6_configure(disp)
    try:
        # A served forward runs without grad, and the dispatcher's wide64 declines under grad mode.
        with torch.no_grad():
            yield sk, disp, dk
    finally:
        disp.configure()
        sk.configure()


def _w6_operands(dk, rows: int, seed: int):
    k, n = dk.QKV_SHAPE
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = _randn(g, rows, k)
    x[::9] = 0
    return x, _randn(g, n, k, 0.02)


def _w6_reference(sk, dk, x, w, kernel: str, step: int) -> torch.Tensor:
    """``x @ w.T`` as calls of at most ``step`` rows of ``kernel`` (``run_splitk_dense``)."""
    out = torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device="cuda")
    tactic = sk.SplitKTactic(*dk.parse_kernel(kernel)[1])
    for row0 in range(0, x.shape[0], step):
        row1 = min(x.shape[0], row0 + step)
        sk.run_splitk_dense(x[row0:row1], w.t(), None, out[row0:row1], True, tactic)
    return out


@pytest.mark.parametrize("tile", ["64x64", "128x64", "64x128"])
@pytest.mark.parametrize("rows", W6_ROWS)
def test_w6_qkv_one_launch_bits(qkv_wide128, rows, tile):
    sk, disp, dk = qkv_wide128
    _w6_configure(disp, 128, tile)
    k, n = dk.QKV_SHAPE
    served = disp.TEP8_WIDE64_KERNELS["qkv"]
    kernel = disp._wide_tiles_kernel(served, disp.TEP8_WIDE64_WIDE_TILES[tile], k, n)
    family, fields = dk.parse_kernel(kernel)
    assert family == "splitk" and fields[:2] == disp.TEP8_WIDE64_WIDE_TILES[tile]
    assert fields[2] == dk.parse_kernel(served)[1][2]  # the served split_k
    x, w = _w6_operands(dk, rows, 600 + rows)
    # The launch itself: every element of its output once, nothing past it.
    sentinel = 0x7FAA  # a BF16 NaN pattern: no product of finite operands takes it
    buf = torch.full((rows + 8, n), sentinel, dtype=torch.int16, device="cuda")
    slot = buf[:rows].view(torch.bfloat16)
    sk.run_splitk_dense_tiles(x, w.t(), slot, True, sk.SplitKTactic(*fields))
    # The dispatcher's route for this call.
    out = _fake_dispatcher(disp, "qkv", dk.QKV_SHAPE).wide64(x, w, None)
    chunks = _w6_reference(sk, dk, x, w, served, 64)  # the served 33..64-row launch
    eight = disp.SM107_KERNELS["qkv"][-1][1]  # the 8-row cell (split_k 4)
    requests = _w6_reference(sk, dk, x, w, eight, 8)
    torch.cuda.synchronize()
    assert not bool((buf[:rows] == sentinel).any()), "an element of the output was not stored"
    assert bool((buf[rows:] == sentinel).all()), "the launch wrote past its output"
    got = buf[:rows]
    assert out is not None and tuple(out.shape) == (rows, n)
    assert torch.equal(out.view(torch.int16), got)
    assert torch.equal(got, chunks.view(torch.int16)), "!= the served 64-row launches"
    assert torch.equal(got, requests.view(torch.int16)), "!= the 8-row cell's calls"
    # cuBLAS (the path it replaces) is a different summation: close, not bitwise.
    ref = F.linear(x, w).float()
    assert torch.allclose(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_w6_dispatcher_routes_only_qkv_up_to_max_rows(qkv_wide128):
    """MAX_ROWS 96: QKV of 65..96 rows takes the one launch, 97..128 keeps cuBLAS (``None``); o_proj
    above 64 rows never does (hop128 only on an armed step: ``None`` here); the default MAX_ROWS 64
    keeps every call above 64 rows on cuBLAS."""
    sk, disp, dk = qkv_wide128
    wide = _module("ops.allreduce.post_attn_hop_wide")
    wide._DIRECT_PUBLICATION = None
    qkv = _fake_dispatcher(disp, "qkv", dk.QKV_SHAPE)
    o_proj = _fake_dispatcher(disp, "o_proj", (K_OPROJ, HIDDEN))
    g = torch.Generator(device="cuda").manual_seed(66)
    for max_rows in (64, 96):
        _w6_configure(disp, max_rows)
        for rows in (72, 96, 97, 128):
            x, w = _w6_operands(dk, rows, rows)
            served = qkv.wide64(x, w, None)
            assert (served is not None) == (rows <= max_rows), (max_rows, rows)
            a, wo = _randn(g, rows, K_OPROJ), _randn(g, HIDDEN, K_OPROJ, 0.03)
            assert o_proj.wide64(a, wo, None) is None, (max_rows, rows)


def test_w6_qkv_one_launch_graph_replays(qkv_wide128):
    """72-, 96- and 128-row QKV calls through the dispatcher captured in one CUDA graph replay the
    eager bits."""
    sk, disp, dk = qkv_wide128
    qkv = _fake_dispatcher(disp, "qkv", dk.QKV_SHAPE)
    calls = [_w6_operands(dk, rows, 700 + rows) for rows in (72, 96, 128)]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for x, w in calls:
            assert qkv.wide64(x, w, None) is not None
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [qkv.wide64(x, w, None) for x, w in calls]
    g = torch.Generator(device="cuda").manual_seed(77)
    for rep in range(4):
        for x, _ in calls:
            x.copy_(_randn(g, *x.shape, scale=1 + rep))
        graph.replay()
        torch.cuda.synchronize()
        for (x, w), out in zip(calls, outs):
            want = qkv.wide64(x, w, None)
            torch.cuda.synchronize()
            assert torch.equal(out.view(torch.int16), want.view(torch.int16)), (x.shape, rep)


def test_w6_qkv_one_launch_timing(qkv_wide128, capsys):
    """A timing print, no assertion: us/call of the QKV call in a CUDA graph of 62 calls on distinct
    weights (cold in L2, as a decode step's 62 layers), the dispatcher's split-K path (PDL, as served)
    vs cuBLAS ``F.linear`` (the path 65..128 rows ran before), at 64 / 72 / 96 / 128 rows."""
    sk, disp, dk = qkv_wide128
    qkv = _fake_dispatcher(disp, "qkv", dk.QKV_SHAPE)
    k, n = dk.QKV_SHAPE
    layers = 62
    g = torch.Generator(device="cuda").manual_seed(88)
    weights = [_randn(g, n, k, 0.02) for _ in range(layers)]

    def per_call(run, x) -> float:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for w in weights[:2]:
                run(x, w)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for w in weights:
                run(x, w)
        graph.replay()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        reps = 20
        start.record()
        for _ in range(reps):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1e3 / (reps * layers)

    lines = []
    for tile in ("64x64", "128x64", "64x128"):
        _w6_configure(disp, 128, tile)
        for rows in (64, 72, 96, 128):
            if rows == 64 and tile != "64x64":
                continue  # 64 rows: the served 33..64-row tactic whatever WIDE_TILE
            x = _randn(g, rows, k)
            splitk = per_call(lambda a, w: qkv.wide64(a, w, None), x)
            cublas = per_call(lambda a, w: F.linear(a, w), x)
            lines.append(
                f"[W6] QKV {rows:3d} rows, WIDE_TILE {tile:6s}: split-K {splitk:6.2f} us/call, "
                f"cuBLAS {cublas:6.2f} us/call, x62 layers: {(splitk - cublas) * 62 / 1e3:+.3f} ms/step"
            )
    with capsys.disabled():
        print("\n" + "\n".join(lines), flush=True)


# --------------------------------------------------------------------------------------------------
# W7: the QKV call's o_proj L2 prefetch on armed 65..128-row direct-hop steps (``PREFETCH_MIN_ROWS``)
# --------------------------------------------------------------------------------------------------
class _Armed:
    """``post_attn_hop_wide._DIRECT_PUBLICATION`` as ``fork_direct_prefetch`` reads it."""

    def __init__(self, rows: int) -> None:
        self.rows = rows


@pytest.fixture
def direct_prefetch():
    disp = _module("kernels.dense_gemms.dispatcher")
    wide = _module("ops.allreduce.post_attn_hop_wide")
    dk = _module("ops.gemm.dense_gemm")
    disp.configure(tep8_wide64=True, tep8_direct_prefetch_min_rows=65)
    try:
        with torch.no_grad():
            yield disp, wide, dk
    finally:
        wide._DIRECT_PUBLICATION = None
        disp.configure()


def test_w7_qkv_forks_the_o_proj_prefetch_on_armed_steps(direct_prefetch):
    disp, wide, dk = direct_prefetch
    g = torch.Generator(device="cuda").manual_seed(7)
    o_weight = _randn(g, HIDDEN, K_OPROJ, 0.03)
    plan = disp.DenseGemmPlan({}, prefetch=True, enable_pdl=True)
    site = disp._Site(plan, "qkv", 0, dk.QKV_SHAPE, lambda: o_weight)
    qkv = disp.DenseGemmDispatcher(site)
    q_w = _randn(g, dk.QKV_SHAPE[1], dk.QKV_SHAPE[0], 0.02)
    for rows in (72, 128):
        x = _randn(g, rows, dk.QKV_SHAPE[0])
        wide._DIRECT_PUBLICATION = None
        assert not qkv.fork_direct_prefetch(x, q_w) and not plan.pending
        wide._DIRECT_PUBLICATION = _Armed(rows + 8)  # another step's rows
        assert not qkv.fork_direct_prefetch(x, q_w) and not plan.pending
        wide._DIRECT_PUBLICATION = _Armed(rows)
        assert qkv.fork_direct_prefetch(x, q_w) and plan.pending
        plan.join()
        torch.cuda.synchronize()
        assert not plan.pending
    # Inside a capture (the side stream exists from the eager forks): fork and join are graph nodes.
    x = _randn(g, 96, dk.QKV_SHAPE[0])
    wide._DIRECT_PUBLICATION = _Armed(96)
    graph = torch.cuda.CUDAGraph()
    out = torch.empty(96, dk.QKV_SHAPE[1], dtype=torch.bfloat16, device="cuda")
    with torch.cuda.graph(graph):
        torch.mm(x, q_w.t(), out=out)
        assert qkv.fork_direct_prefetch(x, q_w)
        plan.join()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, torch.mm(x, q_w.t())) and plan.dropped == 0


def test_w7_scatter_producer_cold_vs_prefetched_timing(hop128, capsys):
    """A timing print, no assertion: the 65..128-row o_proj scatter producer (PDL off, one call per
    measurement) with its weight evicted from L2 (a 512 MiB sweep) vs L2-prefetched first (the
    dispatcher's ``l2_prefetch``), per PRODUCER mode."""
    wide = hop128
    dk = _module("ops.gemm.dense_gemm")
    g = torch.Generator(device="cuda").manual_seed(70)
    window = _FakeWindow(128)
    w = _randn(g, HIDDEN, K_OPROJ, 0.03)
    sink = torch.zeros(1, dtype=torch.int32, device="cuda")
    sweep = torch.empty(512 * 2**20 // 4, dtype=torch.int32, device="cuda")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    lines = []
    for rows in (64, 72, 96, 128):
        a = _randn(g, rows, K_OPROJ)
        for mode in ("grid", "tile128", "chunks"):
            if rows <= 64 and mode != "grid":
                continue
            res = {}
            for warm in (False, True):
                times = []
                for rep in range(30):
                    sweep.add_(1)
                    if warm:
                        dk.l2_prefetch(w, sink)
                    torch.cuda.synchronize()
                    start.record()
                    _produce(wide, mode, a, w, window, rep % 2, 3)
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end) * 1e3)
                res[warm] = sorted(times)[len(times) // 2]
            lines.append(
                f"[W7] o_proj producer {rows:3d} rows {mode:7s}: cold {res[False]:6.2f} us, "
                f"L2-prefetched {res[True]:6.2f} us ({res[True] - res[False]:+.2f})"
            )
    window.reset()
    with capsys.disabled():
        print("\n" + "\n".join(lines), flush=True)


# --------------------------------------------------------------------------------------------------
# W8: the pool slice's one-launch tile at 65..128 rows (``tep8_publish_splitk__WIDE_POOL_TILE=64x32``)
# --------------------------------------------------------------------------------------------------
@pytest.fixture
def pool_tile():
    sk = _module("ops.gemm.splitk_gemm")
    pp = _module("ops.comm.projection_publish")
    sk.configure(wide64=True)
    pp.configure(
        tep8_publish_splitk=True,
        tep8_publish_splitk_max_rows=128,
        tep8_publish_splitk_wide_pool_tile="64x32",
    )
    try:
        with torch.no_grad():
            yield sk, pp
    finally:
        pp.configure()
        sk.configure()


def _pool_operands(pp, rows: int, seed: int):
    k, n = pp.POOL_SLICE
    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden = _randn(g, rows, k)
    hidden[::9] = 0
    return hidden, _randn(g, n, k, 0.02)


@pytest.mark.parametrize("rows", [65, 72, 96, 104, 120, 128])
def test_w8_pool_wide_tile_bits(pool_tile, rows):
    sk, pp = pool_tile
    k, n = pp.POOL_SLICE
    served = sk.SplitKTactic(*pp.TP8_PUBLISH_SPLITK_TACTICS[pp.POOL_SLICE])
    tile = sk.SplitKTactic(*pp.SPLITK_WIDE_POOL_TILES["64x32"])
    assert tile.split_k == served.split_k == 4
    hidden, weight = _pool_operands(pp, rows, 800 + rows)
    sentinel = pp._SENTINEL_I16
    buf = torch.full((rows + 8, n), sentinel, dtype=torch.int16, device="cuda")
    slot = buf[:rows].view(torch.bfloat16)
    pp._run_splitk_wide(hidden, weight, slot, tile)
    ref = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
    pp._run_splitk_wide(hidden, weight, ref, served)
    cells = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
    disp = _module("kernels.dense_gemms.dispatcher")
    eight = sk.SplitKTactic(*disp.parse_kernel(disp.SM107_KERNELS["pool"][-1][1])[1])  # 8-row cell
    assert eight.split_k == 4
    for row0 in range(0, rows, 8):
        row1 = min(rows, row0 + 8)
        sk.run_splitk_dense(hidden[row0:row1], weight.t(), None, cells[row0:row1], True, eight)
    torch.cuda.synchronize()
    assert not bool((buf[:rows] == sentinel).any()), "an element of the slot was not published"
    assert bool((buf[rows:] == sentinel).all()), "the publication wrote past its slot"
    assert torch.equal(slot.view(torch.int16), ref.view(torch.int16)), "!= the served tile"
    assert torch.equal(slot.view(torch.int16), cells.view(torch.int16)), "!= the 8-row cells"


def test_w8_pool_wide_tile_is_certified_and_routed(pool_tile):
    """On this device ``_splitk_publisher`` accepts 64x32 (its key is a certified one at this SM
    count; the failure names the key otherwise) and ``publish_projection`` launches it for a 96-row
    pool slice only; the unpool slice keeps (64, 32, 4, 8) as its own served tactic."""
    sk, pp = pool_tile
    assert pp._splitk_publisher() is not None, "the split-K publication is not certified here"
    assert pp._splitk_state.get("wide_pool") is not None, (
        "WIDE_POOL_TILE 64x32 not certified here: key "
        f"{pp._splitk_site_key(sk, pp.SPLITK_WIDE_POOL_TILES['64x32'], (64,))}"
    )
    assert tuple(pp._splitk_state["wide_pool"]) == (64, 32, 4, 8)
    seen = []
    real = pp._run_splitk_wide

    def spy(h, w, s, t):
        seen.append((h.shape[1], (t.mma_m, t.mma_n, t.split_k, t.ab_stages)))
        real(h, w, s, t)

    pp._run_splitk_wide = spy
    try:
        hidden, weight = _pool_operands(pp, 96, 9)
        slot = torch.empty(96, weight.shape[0], dtype=torch.bfloat16, device="cuda")
        pp.publish_projection(hidden, weight, slot)
        unpool_h = _randn(torch.Generator(device="cuda").manual_seed(10), 96, pp.POOL_HIDDEN)
        unpool_w = _randn(
            torch.Generator(device="cuda").manual_seed(11), 1024, pp.POOL_HIDDEN, 0.02
        )
        unpool_s = torch.empty(96, 1024, dtype=torch.bfloat16, device="cuda")
        pp.publish_projection(unpool_h, unpool_w, unpool_s)
    finally:
        pp._run_splitk_wide = real
    torch.cuda.synchronize()
    assert seen == [
        (pp.POOL_SLICE[0], (64, 32, 4, 8)),
        (pp.POOL_HIDDEN, tuple(pp.TP8_PUBLISH_SPLITK_TACTICS[(pp.POOL_HIDDEN, 1024)])),
    ]


def test_w8_pool_wide_tile_graph_replays(pool_tile):
    sk, pp = pool_tile
    tile = sk.SplitKTactic(*pp.SPLITK_WIDE_POOL_TILES["64x32"])
    calls = [_pool_operands(pp, rows, 900 + rows) for rows in (72, 128)]
    slots = [
        torch.empty(h.shape[0], w.shape[0], dtype=torch.bfloat16, device="cuda") for h, w in calls
    ]
    for (h, w), s in zip(calls, slots):
        pp._run_splitk_wide(h, w, s, tile)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for (h, w), s in zip(calls, slots):
            pp._run_splitk_wide(h, w, s, tile)
    g = torch.Generator(device="cuda").manual_seed(91)
    for rep in range(3):
        for h, _ in calls:
            h.copy_(_randn(g, *h.shape, scale=1 + rep))
        graph.replay()
        torch.cuda.synchronize()
        for (h, w), s in zip(calls, slots):
            want = torch.empty_like(s)
            pp._run_splitk_wide(h, w, want, tile)
            torch.cuda.synchronize()
            assert torch.equal(s.view(torch.int16), want.view(torch.int16)), (h.shape, rep)


def test_w8_pool_wide_tile_timing(pool_tile, capsys):
    """A timing print, no assertion: us/call of the one-launch pool publication in a CUDA graph of
    62 calls on distinct weight slices (cold in L2, PDL as served), 64x16 vs 64x32."""
    sk, pp = pool_tile
    k, n = pp.POOL_SLICE
    g = torch.Generator(device="cuda").manual_seed(92)
    weights = [_randn(g, n, k, 0.02) for _ in range(62)]
    lines = []
    for rows in (72, 96, 128):
        hidden = _randn(g, rows, k)
        slot = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
        res = {}
        for name in ("64x16", "64x32", "64x64"):
            tactic = sk.SplitKTactic(
                *(pp.SPLITK_WIDE_POOL_TILES[name] or pp.TP8_PUBLISH_SPLITK_TACTICS[pp.POOL_SLICE])
            )
            for w in weights[:2]:
                pp._run_splitk_wide(hidden, w, slot, tactic)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for w in weights:
                    pp._run_splitk_wide(hidden, w, slot, tactic)
            graph.replay()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            res[name] = start.elapsed_time(end) * 1e3 / (20 * len(weights))
        lines.append(
            f"[W8] pool slice {rows:3d} rows: 64x16 {res['64x16']:6.2f} us/call, 64x32 "
            f"{res['64x32']:6.2f}, 64x64 {res['64x64']:6.2f} us/call, x62 layers 64x32 "
            f"{(res['64x32'] - res['64x16']) * 62 / 1e3:+.3f} / 64x64 "
            f"{(res['64x64'] - res['64x16']) * 62 / 1e3:+.3f} ms/step"
        )
    with capsys.disabled():
        print("\n" + "\n".join(lines), flush=True)


# --------------------------------------------------------------------------------------------------
# W9: swa_cl8w, the TEP8 one-cluster SWA block with ONE cluster per request (both KV heads)
# --------------------------------------------------------------------------------------------------
#: The fixture's band: W4's CL8 band (2..16 requests) with swa_cl8w from 12 requests
#: (swa_one_cluster_block__TEP8_WIDE_MIN_REQUESTS=12).
W9_BAND = dict(CL8_BAND, cl8_tep8_wide_min_requests=12)


@pytest.fixture
def cl8_wide(monkeypatch):
    """swa_block with W9_BAND and both TEP8 builds compiled; yields the module and ``(requests,
    swa_cl8w)`` of every CL8 launch from then on."""
    swa = _module("ops.attention.swa_block")
    try:
        from cuda.bindings import driver, nvrtc  # noqa: F401
    except ImportError as error:
        pytest.skip(f"the CL8 build needs cuda-python's NVRTC and driver bindings: {error}")
    swa.configure(**W9_BAND)
    torch.zeros(1, device="cuda")  # the primary context, as W4's fixture
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    try:
        try:
            swa._cl8_function(torch.device("cuda", device), False, 2)
        except RuntimeError as error:
            if "swa_cl8w" in str(error):
                pytest.fail(f"swa_cl8w does not build: {error}")
            pytest.skip(f"no TEP8 CL8 build on this device: {error}")
        # swa_cl8w is what W9 tests: its build failure fails (with the NVRTC log), never skips. The
        # TEP8 build above was usually compiled already (W4, without the setting): this cache hit
        # must compile swa_cl8w too (the serving path's guarantee), checked first.
        assert (device, False, 2, "w") in swa._CL8_FN, (
            "a cache hit of the TEP8 build under cl8_tep8_wide_min_requests did not compile swa_cl8w"
        )
        try:
            swa._cl8_function(torch.device("cuda", device), True, 2, wide=True)
        except RuntimeError as error:
            pytest.fail(f"swa_cl8w (int64 positions) does not build: {error}")
        launches = []
        launch = swa._cl8_launch

        def counted(fn, qkv, kv_pool, bt, seq_lens, *args):
            launches.append((seq_lens.numel(), bool(args[11]) if len(args) > 11 else False))
            return launch(fn, qkv, kv_pool, bt, seq_lens, *args)

        monkeypatch.setattr(swa, "_cl8_launch", counted)
        yield swa, launches
    finally:
        swa.configure()


@pytest.mark.parametrize("variant", ["checkpoint-int32", "permuted-int64"])
@pytest.mark.parametrize("n", [11, 12, 13, 16])
def test_w9_cl8w_step_equals_swa_cl8(cl8_wide, n, variant):
    """A step of n requests launches swa_cl8w from 12 requests (swa_cl8 below: the 5..11 path
    unchanged), and every request's output rows and the K/V bytes appended equal, bit for bit, swa_cl8's
    n-request call and the certified calls of 8; one CUDA-graph capture replays the eager bits."""
    swa, launches = cl8_wide
    wide = n >= W9_BAND["cl8_tep8_wide_min_requests"]
    permuted = variant.startswith("permuted")
    pos_dtype = torch.int64 if variant.endswith("int64") else torch.int32
    step = _SwaStep(n, permuted, pos_dtype, seed=900 + n + 100 * permuted)
    reals = list(range(n))
    fillers = list(range(n, n + len(SWA_FILLER_LENS)))

    rows_w, pool_w = _swa_layout(swa, step, [reals])
    assert launches == [(n, wide)], launches
    out_w = torch.cat([rows_w[b] for b in reals])
    assert bool(torch.isfinite(out_w.float()).all()), "the call read a poisoned cache byte"
    raw0, raw_w = step.pool.view(torch.uint8), pool_w.view(torch.uint8)
    mine = step.step_real[:, None, :, None].expand(raw0.shape)
    assert not bool(((raw_w[mine] & 0x7F) == 0x7F).any()), "a step row's K/V byte was not appended"
    assert torch.equal(raw_w[~mine], raw0[~mine]), "the call wrote outside its step rows"

    def padded(ids, size):
        return ids + fillers[: size - len(ids)]

    references = {
        "calls of 8 in order (swa_cl8)": [padded(reals[i : i + 8], 8) for i in range(0, n, 8)]
    }
    for name, calls in references.items():
        launches.clear()
        rows_x, pool_x = _swa_layout(swa, step, calls)
        assert launches == [(len(ids), False) for ids in calls], launches
        for b in reals:
            assert torch.equal(rows_x[b].view(torch.int16), rows_w[b].view(torch.int16)), (name, b)
        assert torch.equal(pool_x.view(torch.uint8)[: step.real_pages], raw_w[: step.real_pages]), (
            name
        )
    # swa_cl8 on the same n-request call (wide build off).
    swa.configure(**CL8_BAND)
    launches.clear()
    rows_a, pool_a = _swa_layout(swa, step, [reals])
    assert launches == [(n, False)], launches
    for b in reals:
        assert torch.equal(rows_a[b].view(torch.int16), rows_w[b].view(torch.int16)), ("swa_cl8", b)
    assert torch.equal(pool_a.view(torch.uint8), raw_w), "swa_cl8's appended K/V bytes differ"
    swa.configure(**W9_BAND)

    x = step.inputs(reals)
    pool_g = step.pool.clone()
    launches.clear()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _swa_call(swa, step, x, pool_g)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_g = _swa_call(swa, step, x, pool_g)
    assert launches == [(n, wide), (n, wide)], launches
    g = torch.Generator(device="cuda").manual_seed(n)
    for rep in range(3):
        if rep == 2:
            x["qkv"].copy_(
                torch.randn(x["qkv"].shape, device="cuda", generator=g).to(torch.bfloat16)
            )
        pool_g.copy_(step.pool)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        if rep < 2:
            want_out, want_pool = out_w, raw_w
        else:
            pool_e = step.pool.clone()
            want_out = _swa_call(swa, step, x, pool_e)
            torch.cuda.synchronize()
            want_pool = pool_e.view(torch.uint8)
        assert torch.equal(out_g.view(torch.int16), want_out.view(torch.int16)), f"replay {rep}"
        assert torch.equal(pool_g.view(torch.uint8), want_pool), f"replay {rep}: cache bytes"


def test_w9_cl8w_timing(cl8_wide, capsys):
    """(info) one SWA block call at 12 / 13 / 16 requests: swa_cl8 (2 clusters per request: two waves
    from 12) vs swa_cl8w (one per request: one wave), each a 20-call CUDA graph."""
    swa, _ = cl8_wide
    lines = []
    for n in (12, 13, 16):
        step = _SwaStep(n, False, torch.int32, seed=950 + n)
        x = step.inputs(list(range(n)))
        res = {}
        for name, band in (("swa_cl8", CL8_BAND), ("swa_cl8w", W9_BAND)):
            swa.configure(**band)
            pool = step.pool.clone()
            _swa_call(swa, step, x, pool)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(20):
                    _swa_call(swa, step, x, pool)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            res[name] = start.elapsed_time(end) * 1e3 / 200
        swa.configure(**W9_BAND)
        clusters = _cl8_max_active_clusters(swa, n)
        waves = "one wave" if clusters and 2 * n <= clusters else "2+ waves"
        lines.append(
            f"[W9] {n} requests: swa_cl8 {2 * n} clusters ({waves}, "
            f"device holds {clusters}) {res['swa_cl8']:6.2f} us/call | swa_cl8w {n} clusters "
            f"{res['swa_cl8w']:6.2f} us/call ({res['swa_cl8w'] - res['swa_cl8']:+.2f}); x46 SWA layers: "
            f"{(res['swa_cl8w'] - res['swa_cl8']) * 46 / 1e3:+.3f} ms/step"
        )
# W10: 65..128-row TP8 drafter GEMMs in one launch of the 64-row tactic (``tep8_drafter_wide_gemms__MAX_ROWS``)
# --------------------------------------------------------------------------------------------------
W10_ROWS = [65, 72, 96, 104, 120, 128]
W10_SITES = ("conv_coef", "qkv", "o_proj", "down", "gate_up")


def _w10_configure(fl, max_rows: int = 128, tile: str = "64x64", sites=W10_SITES) -> None:
    """``fixed_low_m_gemm`` as ``lever_settings.apply`` configures it with ``tep8_wide64_splitk``,
    ``tep8_drafter_wide_gemms`` on and the knobs set."""
    fl.configure(
        tep8_wide64=True,
        tep8_drafter_wide=True,
        tep8_drafter_max_rows=max_rows,
        tep8_drafter_wide_tile=tile,
        tep8_drafter_wide_sites=tuple(sites),
    )


@pytest.fixture
def drafter_wide128():
    sk = _module("ops.gemm.splitk_gemm")
    fl = _module("ops.gemm.fixed_low_m_gemm")
    sw = _module("spec_worker")
    sk.configure(wide64=True)
    _w10_configure(fl)
    try:
        # A served drafter forward runs without grad; ``takes`` declines under grad mode.
        with torch.no_grad():
            yield sk, fl, sw
    finally:
        fl.configure()
        sk.configure()


def _w10_operands(fl, site: str, rows: int, seed: int):
    k, n = fl._TP8_DRAFTER_WIDE_SHAPES[site]
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = _randn(g, rows, k)
    x[::9] = 0
    return x, _randn(g, n, k, 0.02)


def _w10_chunks(sk, x, w, tactic, step: int) -> torch.Tensor:
    """``x @ w.T`` as ``run_splitk_dense`` calls of at most ``step`` rows at ``tactic``."""
    out = torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device="cuda")
    for row0 in range(0, x.shape[0], step):
        row1 = min(x.shape[0], row0 + step)
        sk.run_splitk_dense(x[row0:row1], w.t(), None, out[row0:row1], True, sk.SplitKTactic(*tactic))
    return out


@pytest.mark.parametrize("tile", ["64x64", "128x64", "64x128"])
@pytest.mark.parametrize("rows", W10_ROWS)
@pytest.mark.parametrize("site", W10_SITES)
def test_w10_drafter_one_launch_bits(drafter_wide128, site, rows, tile):
    """A 65..128-row drafter GEMM in one launch of its 64-row tactic (on WIDE_TILE's tile) writes every
    element of its output and nothing past it, and each row equals, bit for bit, the same row inside the
    64-row tactic's launches (64-row chunks) and inside the 8-row cell's calls (each request's one-request
    bits); ``fixed_low_m_gemm`` (what every drafter call site runs) returns those bits."""
    sk, fl, _ = drafter_wide128
    _w10_configure(fl, 128, tile)
    k, n = fl._TP8_DRAFTER_WIDE_SHAPES[site]
    tactic = fl.tactic_for(rows, k, n)
    served = fl._TP8_DRAFTER_WIDE_TILES[64][site]
    eight = fl._SM107_SPLITK_TACTICS_BASE[(8, k, n)]
    assert tactic is not None and tuple(tactic[:2]) == fl.TEP8_DRAFTER_WIDE_TILES[tile]
    assert tactic[2] == served[2] == eight[2]  # the 8-row cell's split_k
    launch = fl.one_launch_tactic(tuple(tactic), k, n)
    assert (launch.mma_m, launch.mma_n, launch.split_k) == tuple(tactic[:3])
    x, w = _w10_operands(fl, site, rows, 900 + rows)
    sentinel = 0x7FAA  # a BF16 NaN pattern: no product of finite operands takes it
    buf = torch.full((rows + 8, n), sentinel, dtype=torch.int16, device="cuda")
    sk.run_splitk_dense_tiles(x, w.t(), buf[:rows].view(torch.bfloat16), True, launch)
    out = fl.fixed_low_m_gemm(x, w, tactic, True)
    chunks = _w10_chunks(sk, x, w, served, 64)  # the 64-row tactic's launches
    requests = _w10_chunks(sk, x, w, eight, 8)  # the 8-row cell's calls
    torch.cuda.synchronize()
    assert not bool((buf[:rows] == sentinel).any()), "an element of the output was not stored"
    assert bool((buf[rows:] == sentinel).all()), "the launch wrote past its output"
    got = buf[:rows]
    assert torch.equal(out.view(torch.int16), got)
    assert torch.equal(got, chunks.view(torch.int16)), "!= the 64-row tactic's launches"
    assert torch.equal(got, requests.view(torch.int16)), "!= the 8-row cell's calls"
    # cuBLAS (the path it replaces) is a different summation: close, not bitwise.
    assert torch.allclose(out.float(), F.linear(x, w).float(), rtol=2e-2, atol=2e-2)


def test_w10_drafter_call_sites_route_by_max_rows_and_sites(drafter_wide128, monkeypatch):
    """``spec_worker._gemm`` (the convolution coefficient / context K/V / fc shard GEMMs) and
    ``FixedTacticLinearMethod`` (the TP-sharded projections) take the one launch at 65..MAX_ROWS rows
    for SITES only; other sites, rows above MAX_ROWS, the fc shard and the default MAX_ROWS (64) keep
    ``F.linear`` / the built-in method above 64 rows. 64 rows keep the 64-row tactic."""
    sk, fl, sw = drafter_wide128
    calls = []
    real = sw.fixed_low_m_gemm
    monkeypatch.setattr(
        sw, "fixed_low_m_gemm", lambda x, w, t, pdl: calls.append((x.shape[0], t)) or real(x, w, t, pdl)
    )
    # The built-in method the fixed one falls back to: cuBLAS, as ``_gemm``'s ``F.linear``.
    monkeypatch.setattr(
        sw.UnquantizedLinearMethod, "apply", lambda self, module, x, bias: F.linear(x, module.weight)
    )
    method = sw.FixedTacticLinearMethod(True)
    for max_rows, sites in ((64, ("conv_coef", "qkv", "gate_up")), (96, ("conv_coef", "qkv", "gate_up")),
                            (128, W10_SITES)):
        _w10_configure(fl, max_rows, "64x64", sites)
        method.tactics = fl.splitk_tactics()
        for site in W10_SITES:
            for rows in (64, 72, 96, 104, 128):
                calls.clear()
                x, w = _w10_operands(fl, site, rows, rows)
                out = sw._gemm(x, w, True, fl.splitk_tactics())
                module = types.SimpleNamespace(weight=w, use_custom_cublas_mm=False)
                via_method = method.apply(module, x, None)
                torch.cuda.synchronize()
                want = rows <= 64 or (rows <= max_rows and site in sites)
                assert len(calls) == (2 if want else 0), (max_rows, site, rows, calls)
                assert torch.equal(out.view(torch.int16), via_method.view(torch.int16))
                if want:
                    single = real(x, w, fl.tactic_for(rows, *fl._TP8_DRAFTER_WIDE_SHAPES[site]), True)
                    torch.cuda.synchronize()
                    assert torch.equal(out.view(torch.int16), single.view(torch.int16))
        # The fc shard ([rows, 40960] x [1024, 40960]^T) keeps F.linear above 64 rows.
        g = torch.Generator(device="cuda").manual_seed(5)
        calls.clear()
        sw._gemm(_randn(g, 72, 40960), _randn(g, 1024, 40960, 0.02), True, fl.splitk_tactics())
        assert not calls


def test_w10_drafter_one_launch_graph_replays(drafter_wide128):
    """72-, 96- and 128-row calls of every site through ``fixed_low_m_gemm`` captured in one CUDA graph
    replay the eager bits, on each tile."""
    sk, fl, _ = drafter_wide128
    g = torch.Generator(device="cuda").manual_seed(91)
    for tile in ("64x64", "64x128"):
        _w10_configure(fl, 128, tile)
        calls = []
        for site in W10_SITES:
            for rows in (72, 96, 128):
                x, w = _w10_operands(fl, site, rows, 950 + rows)
                calls.append((x, w, fl.tactic_for(rows, *fl._TP8_DRAFTER_WIDE_SHAPES[site])))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for x, w, t in calls:
                fl.fixed_low_m_gemm(x, w, t, True)  # compiles before the capture
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [fl.fixed_low_m_gemm(x, w, t, True) for x, w, t in calls]
        for rep in range(3):
            for x, _, _ in calls:
                x.copy_(_randn(g, *x.shape, scale=1 + rep))
            graph.replay()
            torch.cuda.synchronize()
            for (x, w, t), out in zip(calls, outs):
                want = fl.fixed_low_m_gemm(x, w, t, True)
                torch.cuda.synchronize()
                assert torch.equal(out.view(torch.int16), want.view(torch.int16)), (tile, x.shape, rep)


def test_w10_drafter_one_launch_timing(drafter_wide128, capsys):
    """A timing print, no assertion: us/call of each drafter GEMM site in a CUDA graph of 5 calls on
    distinct weights (cold in L2, as the drafter's 5 layers), the one launch (PDL, as served) vs cuBLAS
    ``F.linear`` (nvjet, plus its splitKreduce where cuBLAS splits K: the path 65..128 rows ran before),
    at 64 (the served 64-row tactic) / 72 / 96 / 128 rows, and the step delta over the drafter's calls
    (conv_coef 10, every other site 5)."""
    sk, fl, _ = drafter_wide128
    g = torch.Generator(device="cuda").manual_seed(92)
    per_step = {"conv_coef": 10, "qkv": 5, "o_proj": 5, "down": 5, "gate_up": 5}

    def per_call(run, x, weights) -> float:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for w in weights[:2]:
                run(x, w)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for w in weights:
                run(x, w)
        graph.replay()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        reps = 20
        start.record()
        for _ in range(reps):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1e3 / (reps * len(weights))

    lines = []
    for site in W10_SITES:
        k, n = fl._TP8_DRAFTER_WIDE_SHAPES[site]
        weights = [_randn(g, n, k, 0.02) for _ in range(5)]
        for tile in ("64x64", "128x64", "64x128"):
            _w10_configure(fl, 128, tile)
            for rows in (64, 72, 96, 128):
                if rows == 64 and tile != "64x64":
                    continue  # 64 rows: the 64-row tactic whatever WIDE_TILE
                x = _randn(g, rows, k)
                tactic = fl.tactic_for(rows, k, n)
                splitk = per_call(lambda a, w: fl.fixed_low_m_gemm(a, w, tactic, True), x, weights)
                cublas = per_call(lambda a, w: F.linear(a, w), x, weights)
                lines.append(
                    f"[W10] {site:9s} {rows:3d} rows, WIDE_TILE {tile:6s}: split-K {splitk:6.2f} us/call, "
                    f"cuBLAS {cublas:6.2f} us/call, x{per_step[site]}/step: "
                    f"{(splitk - cublas) * per_step[site]:+6.1f} us/step"
                )
    with capsys.disabled():
        print("\n" + "\n".join(lines), flush=True)


# --------------------------------------------------------------------------------------------------
# W10 (window): the drafter window attention's multi-tile splits on 9..16-request TEP8 steps
# (``tep8_drafter_wide_gemms__WINDOW_TILES``)
# --------------------------------------------------------------------------------------------------
W10_WINDOW_REQUESTS = [9, 12, 16]


@pytest.fixture
def drafter_window():
    wa = _module("ops.spec.dflash_window_attention")
    try:
        with torch.no_grad():
            yield wa
    finally:
        wa.configure()


def _w10_window_operands(wa, requests: int, seed: int, m: int = 8, hq: int = 8, d: int = 128):
    """A TEP8 rank's drafter attention call: ``requests`` blocks of ``m`` rows, 8 query / 1 KV head."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    capacity = 8_200
    k_cache = _randn(g, requests * capacity, d).view(requests, capacity, 1, d)
    v_cache = _randn(g, requests * capacity, d).view(requests, capacity, 1, d)
    seqlens = torch.randint(3_000, capacity, (requests,), generator=g, device="cuda").to(torch.int32)
    slots = torch.randperm(requests, generator=g, device="cuda").to(torch.int32)
    q = _randn(g, requests * m, hq * d).view(requests, m, hq, d)
    k_new = _randn(g, requests * m, d).view(requests * m, 1, d)
    v_new = _randn(g, requests * m, d).view(requests * m, 1, d)
    return q, k_cache, v_cache, seqlens, slots, k_new, v_new


@pytest.mark.parametrize("tiles", [2, 3, 4])
@pytest.mark.parametrize("requests", W10_WINDOW_REQUESTS)
def test_w10_window_multi_tile_splits_within_the_one_tile_error(drafter_window, requests, tiles):
    """At 9..16 requests (72..128 rows) the window attention on ``tiles``-tile splits is as close to the
    FP64 attention as the one-tile geometry (max error within 1.5x of it plus one BF16 step of the row
    scale), appends the same K/V bytes, and the geometry applies only from WINDOW_MIN_ROWS rows."""
    wa = drafter_window
    q, kc, vc, seqlens, slots, k_new, v_new = _w10_window_operands(wa, requests, 1_000 + requests)
    window = wa.WINDOW
    ref = wa.attention_reference(q, kc, vc, seqlens, slots, window, None, torch.float64, k_new, v_new)
    outs, caches = {}, {}
    for n in (1, tiles):
        wa.configure(multi_tiles=n, multi_min_rows=65)
        k, v = kc.clone(), vc.clone()
        outs[n] = wa.dflash_window_attention(q, k, v, seqlens, slots, window, None, True, k_new, v_new)
        caches[n] = (k, v)
        span, tile_rows = window[0] + q.shape[1], q.shape[2] * q.shape[1]
        assert wa.launch_config(requests, q.shape[1], span, tile_rows)["tiles"] == n
    torch.cuda.synchronize()
    assert torch.equal(caches[1][0], caches[tiles][0]) and torch.equal(caches[1][1], caches[tiles][1])
    scale = ref.abs().amax().item()
    err1 = (outs[1].double() - ref).abs().amax().item()
    errn = (outs[tiles].double() - ref).abs().amax().item()
    assert errn <= 1.5 * err1 + scale * 2**-8, (requests, tiles, err1, errn)
    # Below WINDOW_MIN_ROWS the one-request geometry.
    wa.configure(multi_tiles=tiles, multi_min_rows=requests * 8 + 1)
    assert wa.launch_config(requests, 8, window[0] + 8, 64)["tiles"] == 1


def test_w10_window_multi_tile_graph_replays(drafter_window):
    """A 16-request call on 3-tile splits captured in a CUDA graph replays the eager output."""
    wa = drafter_window
    wa.configure(multi_tiles=3, multi_min_rows=65)
    q, kc, vc, seqlens, slots, k_new, v_new = _w10_window_operands(wa, 16, 1_100)
    run = lambda: wa.dflash_window_attention(  # noqa: E731
        q, kc, vc, seqlens, slots, wa.WINDOW, None, True, k_new, v_new
    )
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    g = torch.Generator(device="cuda").manual_seed(1_101)
    for rep in range(3):
        q.copy_(_randn(g, q.shape[0] * q.shape[1], q.shape[2] * q.shape[3], 1 + rep).view(q.shape))
        graph.replay()
        torch.cuda.synchronize()
        want = run()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), want.view(torch.int16)), rep


def test_w10_window_multi_tile_timing(drafter_window, capsys):
    """A timing print, no assertion: us/call (split + merge) of the window attention in a CUDA graph of
    5 calls on distinct caches (the drafter's 5 layers), WINDOW_TILES 1..4, at 8 / 9 / 12 / 16 requests,
    and the step delta (5 calls) against the one-tile geometry."""
    wa = drafter_window
    lines = []
    for requests in (2, 4, 8, 9, 12, 16):
        layers = [_w10_window_operands(wa, requests, 1_200 + 10 * requests + i) for i in range(5)]
        base = None
        for tiles in (1, 2, 3, 4):
            wa.configure(multi_tiles=tiles, multi_min_rows=9)

            def step():
                for q, kc, vc, seqlens, slots, k_new, v_new in layers:
                    wa.dflash_window_attention(
                        q, kc, vc, seqlens, slots, wa.WINDOW, None, True, k_new, v_new
                    )

            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                step()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
            graph.replay()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            reps = 20
            start.record()
            for _ in range(reps):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            us = start.elapsed_time(end) * 1e3 / (reps * len(layers))
            base = us if base is None else base
            lines.append(
                f"[W10w] {requests:2d} requests, WINDOW_TILES {tiles}: {us:6.2f} us/call, "
                f"x5/step vs 1 tile: {(us - base) * 5:+6.1f} us/step"
            )
    with capsys.disabled():
        print("\n" + "\n".join(lines), flush=True)


# --------------------------------------------------------------------------------------------------
# W10 (prefetch): the drafter's next GEMM weights asked of the L2 from a side stream
# (``tep8_drafter_wide_gemms__PREFETCH``, kernels/drafter_l2_prefetch)
# --------------------------------------------------------------------------------------------------
#: (K, N) of the TP8 drafter's per-layer weights.
_W10P_SHAPES = {
    "conv_attn": (8192, 2048),
    "qkv": (8192, 1280),
    "o_proj": (1024, 8192),
    "conv_mlp": (8192, 2048),
    "gate_up": (8192, 7168),
    "down": (3584, 8192),
}


def _w10p_drafter(layers: int = 5, seed: int = 1_300):
    """A module tree with the drafter's attribute paths and TP8 weight shapes (what the forks read)."""
    ns = types.SimpleNamespace
    g = torch.Generator(device="cuda").manual_seed(seed)

    def lin(name):
        k, n = _W10P_SHAPES[name]
        return ns(weight=_randn(g, n, k, 0.02))

    blocks = [
        ns(
            self_attn=ns(qkv_proj=lin("qkv"), o_proj=lin("o_proj")),
            mlp=ns(gate_up_proj=lin("gate_up"), down_proj=lin("down")),
        )
        for _ in range(layers)
    ]
    return ns(
        model=ns(layers=blocks),
        attention_convs=[ns(kernel_projection=lin("conv_attn")) for _ in range(layers)],
        mlp_convs=[ns(kernel_projection=lin("conv_mlp")) for _ in range(layers)],
    )


def _w10p_sleep_cycles(us: float) -> int:
    """``torch.cuda._sleep`` cycles for about ``us`` microseconds on this device."""
    probe = 200_000
    torch.cuda._sleep(probe)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(probe)
    end.record()
    torch.cuda.synchronize()
    return max(1, int(probe * us / (start.elapsed_time(end) * 1e3)))


def _w10p_forward(drafter, x_rows, pf, sleeps):
    """One drafter-shaped chain per layer: the five GEMM sites in order (cuBLAS, the 65+-row path) with
    sleeps standing for the non-HBM kernels between them, forking the prefetch where the block decode
    does. Returns the last outputs (their bits must not depend on the prefetch)."""
    outs = []
    rows = x_rows.shape[0]
    for i, layer in enumerate(drafter.model.layers):
        a = F.linear(x_rows, drafter.attention_convs[i].kernel_projection.weight)
        torch.cuda._sleep(sleeps["conv"])
        q = F.linear(x_rows, layer.self_attn.qkv_proj.weight)
        if pf is not None:
            pf.fork("attn", i, rows)
        torch.cuda._sleep(sleeps["window"])  # QK norm + RoPE + window attention
        if pf is not None:
            pf.fork("mlp", i, rows)
        o = F.linear(x_rows[:, :1024], layer.self_attn.o_proj.weight)
        torch.cuda._sleep(sleeps["hop"])  # all-reduce + finish conv + add-RMSNorm
        c = F.linear(x_rows, drafter.mlp_convs[i].kernel_projection.weight)
        torch.cuda._sleep(sleeps["conv"])
        gu = F.linear(x_rows, layer.mlp.gate_up_proj.weight)
        d = F.linear(gu[:, :3584].contiguous(), layer.mlp.down_proj.weight)
        if pf is not None:
            pf.fork("next", i, rows)
        torch.cuda._sleep(sleeps["hop"])
        outs = [a, q, o, c, gu, d]
    if pf is not None:
        pf.join()
    return outs


@pytest.fixture
def drafter_prefetch():
    pre = _module("kernels.drafter_l2_prefetch.prefetch")
    try:
        with torch.no_grad():
            yield pre
    finally:
        pre.configure()


@pytest.mark.parametrize("level", [1, 2, 3])
def test_w10_drafter_prefetch_keeps_the_bits_and_rejoins_in_a_graph(drafter_prefetch, level):
    """At every level the forks run on the side stream and the chain's outputs are bit for bit those
    without a prefetch, eagerly and in a CUDA graph capture (where the forward's single join rejoins
    every fork); MIN_ROWS above the call's rows forks nothing."""
    pre = drafter_prefetch
    drafter = _w10p_drafter()
    g = torch.Generator(device="cuda").manual_seed(1_310)
    x = _randn(g, 128, 8192)
    sleeps = {"conv": 1_000, "window": 1_000, "hop": 1_000}
    want = [t.clone() for t in _w10p_forward(drafter, x, None, sleeps)]
    pf = pre.DrafterPrefetch(drafter, level, 1)
    got = _w10p_forward(drafter, x, pf, sleeps)
    torch.cuda.synchronize()
    assert pf.calls == len(pre.forks_of(level, 5))
    for a, b in zip(want, got):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _w10p_forward(drafter, x, pf, sleeps)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = _w10p_forward(drafter, x, pf, sleeps)
    for rep in range(2):
        x.copy_(_randn(g, 128, 8192, 1 + rep))
        graph.replay()
        torch.cuda.synchronize()
        ref = _w10p_forward(drafter, x, None, sleeps)
        torch.cuda.synchronize()
        for a, b in zip(outs, ref):
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), rep
    quiet = pre.DrafterPrefetch(drafter, level, 129)
    _w10p_forward(drafter, x, quiet, sleeps)
    assert quiet.calls == 0


def test_w10_drafter_prefetch_timing(drafter_prefetch, capsys):
    """A timing print, no assertion: the 5-layer drafter-shaped chain (the GEMM sites on cuBLAS / the
    64-row path, sleeps standing for the QK norm + window attention ~20 us and each hop chain ~14 us)
    per step in a CUDA graph, PREFETCH 0..3, at 64 / 96 / 128 rows, and the device's L2 size."""
    pre = drafter_prefetch
    drafter = _w10p_drafter()
    props = torch.cuda.get_device_properties(0)
    sleeps = {
        "conv": _w10p_sleep_cycles(2.3),
        "window": _w10p_sleep_cycles(20.0),
        "hop": _w10p_sleep_cycles(14.0),
    }
    g = torch.Generator(device="cuda").manual_seed(1_320)
    lines = [f"[W10p] {props.name}: L2 {props.L2_cache_size / 2**20:.0f} MiB, {props.multi_processor_count} SMs"]
    # The rest of a step streams far more than the L2 holds between two drafter forwards: a flush
    # buffer read between replays keeps every replay's weights cold, as in a served step.
    flush = torch.empty(512 * 2**20, dtype=torch.uint8, device="cuda")
    for rows in (64, 96, 128):
        x = _randn(g, rows, 8192)
        base = None
        configs = [
            (0, None), (1, None), (2, None), (3, None),
            (3, ("o_proj", "conv_mlp", "down", "conv_attn", "qkv")),
            (3, ("o_proj", "conv_mlp", "gate_up", "conv_attn", "qkv")),
            (3, ("o_proj", "conv_mlp", "conv_attn", "qkv")),
        ]  # fmt: skip
        for level, names in configs:
            pf = pre.DrafterPrefetch(drafter, level, 1, names) if level else None
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                _w10p_forward(drafter, x, pf, sleeps)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _w10p_forward(drafter, x, pf, sleeps)
            times = []
            for _ in range(10):
                flush.add_(1)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end) * 1e3)
            us = sorted(times)[len(times) // 2]
            base = us if base is None else base
            lines.append(
                f"[W10p] {rows:3d} rows, PREFETCH {level} {','.join(names) if names else 'all'}: {us:7.1f} us/step chain, "
                f"vs 0: {us - base:+6.1f} us"
            )
    with capsys.disabled():
        print("\n" + "\n".join(lines), flush=True)


# --------------------------------------------------------------------------------------------------
# W10 (finish norm): a non-fused hop's tail (finish conv + residual add + RMSNorm) in one launch
# (``tep8_drafter_wide_gemms__FINISH_NORM``, ops/spec/drafter_finish_norm)
# --------------------------------------------------------------------------------------------------
def _w10f_operands(rows: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = _randn(g, rows, 8192, 0.5)
    coefficients = _randn(g, rows * 2 * 2 * 512, 1, 0.1).view(rows, 2, 2, 512)
    delta = coefficients[:, 1]  # the finish side's [rows, taps, groups] view, as the drafter's prepare returns
    base = _randn(g, 2, 8192, 0.3)
    residual = _randn(g, rows, 8192)
    weight = (1.0 + _randn(g, 1, 8192, 0.1).float()).to(torch.bfloat16).view(8192)
    return x, delta, base, residual, weight


def _w10f_builtin(x, delta, base, residual, weight, eps, block):
    """The two launches it replaces: dflash2_block_conv (finish side) + the built-in RMSNorm's fused add."""
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    conv_op = _module("ops.spec.dflash2_block_conv").dflash2_block_conv
    norm = RMSNorm(hidden_size=8192, eps=eps, dtype=torch.bfloat16).cuda()
    norm.weight.data.copy_(weight)
    conv = conv_op(x, delta, base, block, 16)
    return norm(conv, residual)


@pytest.mark.parametrize("rows", [8, 72, 96, 128])
def test_w10_finish_norm_equals_the_two_launches(rows):
    """Output and the in-place residual are bit for bit the finish convolution + flashinfer's fused add +
    RMSNorm (the built-in RMSNorm), and close to the FP64 reference."""
    fn = _module("ops.spec.drafter_finish_norm")
    with torch.no_grad():
        x, delta, base, residual, weight = _w10f_operands(rows, 1_700 + rows)
        res_ref = residual.clone()
        out_ref, res_ref = _w10f_builtin(x, delta, base, res_ref, weight, 1e-6, 8)
        res = residual.clone()
        assert fn.takes(x, delta, base, res, weight)
        out = fn.drafter_finish_norm(x, delta, base, res, weight, 1e-6, 8)
        torch.cuda.synchronize()
        assert torch.equal(res.view(torch.int16), res_ref.view(torch.int16)), "residual != the built-in add"
        assert torch.equal(out.view(torch.int16), out_ref.view(torch.int16)), "output != the built-in norm"
        ref = fn.drafter_finish_norm.reference(x, delta, base, residual, weight, 1e-6, 8)
        assert torch.allclose(out.float(), ref.float(), rtol=3e-2, atol=3e-2)


def test_w10_finish_norm_graph_and_timing(capsys):
    fn = _module("ops.spec.drafter_finish_norm")
    lines = []
    with torch.no_grad():
        for rows in (72, 96, 128):
            x, delta, base, residual, weight = _w10f_operands(rows, 1_800 + rows)
            res = residual.clone()

            def fused():
                fn.drafter_finish_norm(x, delta, base, res, weight, 1e-6, 8)

            from tensorrt_llm._torch.modules.rms_norm import RMSNorm

            conv_op = _module("ops.spec.dflash2_block_conv").dflash2_block_conv
            norm = RMSNorm(hidden_size=8192, eps=1e-6, dtype=torch.bfloat16).cuda()

            def builtin():
                norm(conv_op(x, delta, base, 8, 16), res)

            res_us = {}
            for name, run in (("fused", fused), ("built-in", builtin)):
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    run()
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(10):
                        run()
                times = []
                for _ in range(10):
                    res.copy_(residual)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end) * 1e3 / 10)
                res_us[name] = sorted(times)[5]
            # The graph replays the eager bits.
            res.copy_(residual)
            want_res = residual.clone()
            want = fn.drafter_finish_norm(x, delta, base, want_res, weight, 1e-6, 8)
            res.copy_(residual)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                got = fn.drafter_finish_norm(x, delta, base, res, weight, 1e-6, 8)
            res.copy_(residual)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(got.view(torch.int16), want.view(torch.int16))
            assert torch.equal(res.view(torch.int16), want_res.view(torch.int16))
            lines.append(
                f"[W10f] {rows:3d} rows: one launch {res_us['fused']:6.2f} us, conv + norm {res_us['built-in']:6.2f} us "
                f"per hop, x10 hops: {(res_us['fused'] - res_us['built-in']) * 10:+6.1f} us/step"
            )
    with capsys.disabled():
        print("\n" + "\n".join(lines), flush=True)


# --------------------------------------------------------------------------------------------------
# W10 (head): the draft head's vocab-shard GEMM ([rows, 8192] x [25256, 8192]^T per TP8 rank) on cuBLAS vs split-K tactics
# (a timing print that decides whether a head lever is worth writing; no lever yet)
# --------------------------------------------------------------------------------------------------
def test_w10_draft_head_gemm_timing(capsys):
    sk = _module("ops.gemm.splitk_gemm")
    sk.configure(wide64=True)
    try:
        g = torch.Generator(device="cuda").manual_seed(1_900)
        k, n = 8192, 202048 // 8
        weights = [_randn(g, n, k, 0.02) for _ in range(2)]  # two distinct 414 MB shards: cold every call
        tactics = [(128, 64, 1, 6), (64, 128, 1, 6), (128, 128, 1, 3), (64, 64, 1, 7), (128, 64, 2, 6), (128, 32, 1, 8)]
        lines = []

        def per_call(run, x) -> float:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for w in weights:
                    run(x, w)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for w in weights:
                    run(x, w)
            graph.replay()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            reps = 10
            start.record()
            for _ in range(reps):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) * 1e3 / (reps * len(weights))

        for rows in (7, 56, 84, 112):
            x = _randn(g, rows, k)
            cublas = per_call(lambda a, w: F.linear(a, w), x)
            cells = [f"cuBLAS {cublas:6.2f}"]
            ref = F.linear(x, weights[0])
            for tactic in tactics:
                mma_m, mma_n, split_k, stages = tactic
                if rows > 64 and mma_n < 64:
                    continue
                fit = None
                for s in range(stages, 0, -1):
                    try:
                        sk.validate_tactic(sk.SplitKTactic(mma_m, mma_n, split_k, s), min(rows, 64), n, k)
                        fit = sk.SplitKTactic(mma_m, mma_n, split_k, s)
                        break
                    except ValueError:
                        continue
                if fit is None:
                    cells.append(f"{mma_m}x{mma_n}/{split_k} refused")
                    continue

                def run(a, w, fit=fit):
                    out = torch.empty(a.shape[0], w.shape[0], dtype=a.dtype, device=a.device)
                    if a.shape[0] > 64:
                        return sk.run_splitk_dense_tiles(a, w.t(), out, True, fit)
                    return sk.run_splitk_dense(a, w.t(), None, out, True, fit)

                try:
                    got = run(x, weights[0])
                    torch.cuda.synchronize()
                    ok = torch.allclose(got.float(), ref.float(), rtol=3e-2, atol=3e-2)
                    us = per_call(run, x)
                    cells.append(f"{mma_m}x{mma_n}/{split_k}/{fit.ab_stages} {us:6.2f}{'' if ok else ' WRONG'}")
                except Exception as error:  # a timing print: record and go on
                    cells.append(f"{mma_m}x{mma_n}/{split_k} error {type(error).__name__}")
            lines.append(f"[W10h] head {rows:3d} rows: " + " | ".join(cells) + " us/call")
        with capsys.disabled():
            print("\n" + "\n".join(lines), flush=True)
    finally:
        sk.configure()


# --------------------------------------------------------------------------------------------------
# W10 (qk fold): the QK norm + RoPE folded into the window attention's split programs at 2..16 requests
# (``tep8_drafter_wide_gemms__QK_FOLD``: dr3_folds' QF without drafter_attn_pdl)
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tiles", [1, 3])
@pytest.mark.parametrize("requests", [2, 8, 12, 16])
def test_w10_qk_fold_equals_the_qk_norm_rope_launch(drafter_window, requests, tiles):
    """The window attention on the raw q / k_new with the QK fold returns, and appends, bit for bit what it
    does after the built-in ``fused_qk_norm_rope`` launch on the same qkv rows (TEP8 rank: 8 q / 1 KV head)."""
    wa = drafter_window
    wa.configure(multi_tiles=tiles, multi_min_rows=9)
    m, eps, base = 8, 1e-6, 500000.0
    q0, kc, vc, seqlens, slots, _, v_new = _w10_window_operands(wa, requests, 2_000 + requests)
    g = torch.Generator(device="cuda").manual_seed(2_100 + requests)
    rows = requests * m
    qkv = _randn(g, rows, 1280)
    qw = (1.0 + _randn(g, 1, 128, 0.1).float()).to(torch.bfloat16).view(128)
    kw = (1.0 + _randn(g, 1, 128, 0.1).float()).to(torch.bfloat16).view(128)
    positions = (seqlens.repeat_interleave(m) - m + torch.arange(m, device="cuda").repeat(requests)).to(torch.int32)
    ref_qkv = qkv.clone()
    torch.ops.trtllm.fused_qk_norm_rope(
        ref_qkv, 8, 1, 1, 128, 128, eps, qw, kw, base, True, positions, 1.0, 0.0, 0.0, 1.0, True, False, False, 0, 0
    )
    kc_ref, vc_ref = kc.clone(), vc.clone()
    want = wa.dflash_window_attention(
        ref_qkv[:, :1024].reshape(requests, m, 8, 128), kc_ref, vc_ref, seqlens, slots, wa.WINDOW, None, True,
        ref_qkv[:, 1024:1152].reshape(rows, 1, 128), qkv[:, 1152:].reshape(rows, 1, 128),
    )  # fmt: skip
    fold = wa.QkFold(qw, kw, positions, eps, base, 1.0)
    got = wa.dflash_window_attention(
        qkv[:, :1024].reshape(requests, m, 8, 128), kc, vc, seqlens, slots, wa.WINDOW, None, True,
        qkv[:, 1024:1152].reshape(rows, 1, 128), qkv[:, 1152:].reshape(rows, 1, 128), qk_fold=fold,
    )  # fmt: skip
    torch.cuda.synchronize()
    assert torch.equal(kc.view(torch.int16), kc_ref.view(torch.int16)), "appended keys differ"
    assert torch.equal(vc.view(torch.int16), vc_ref.view(torch.int16)), "appended values differ"
    assert torch.equal(got.view(torch.int16), want.view(torch.int16)), "attention output differs"


# --------------------------------------------------------------------------
# W10i: tep8_drafter_wide_gemms__DEEP_STAGES (the 16..64-row drafter entries at deeper A/B pipelines)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [16, 24, 32, 40, 64])
def test_w10_deep_stages_keep_every_bit(drafter_wide128, rows):
    """Each deepened 16..64-row drafter entry (``_TP8_DRAFTER_DEEP_STAGES``) fits this device (``deep_stage_tactics``)
    and returns, bit for bit, what the plain entry (the same tile and split_k at the opt-in's stages) returns, on
    rows with zeros, in a CUDA graph as eagerly."""
    sk, fl, _ = drafter_wide128
    _w10_configure(fl, 64)
    plain = dict(fl.SM107_SPLITK_TACTICS)
    fl.configure(tep8_wide64=True, tep8_drafter_wide=True, tep8_drafter_deep_stages=True)
    deep = fl.deep_stage_tactics()
    sites = fl._TP8_DRAFTER_DEEP_STAGES[rows]
    assert sites and len(deep) == sum(len(s) for s in fl._TP8_DRAFTER_DEEP_STAGES.values())
    for site, stages in sites.items():
        k, n = fl._TP8_DRAFTER_WIDE_SHAPES[site]
        tactic = fl.tactic_for(rows, k, n)
        assert tuple(tactic) == (*plain[(rows, k, n)][:3], stages), site
        assert deep[(rows, k, n)].ab_stages == stages
        x, w = _w10_operands(fl, site, rows, 1300 + rows)
        want = fl.fixed_low_m_gemm(x, w, plain[(rows, k, n)], True)
        got = fl.fixed_low_m_gemm(x, w, tactic, True)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fl.fixed_low_m_gemm(x, w, tactic, True)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay = fl.fixed_low_m_gemm(x, w, tactic, True)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (site, rows)
        assert torch.equal(replay.view(torch.int16), want.view(torch.int16)), (site, rows)


# --------------------------------------------------------------------------
# W10j: tep8_drafter_wide_gemms__SILU_FOLD (gate_up + silu(gate) * up in one launch on 8..64 rows)
# --------------------------------------------------------------------------
def _w10j_plain(fl, sm, x, w, tactic, pdl):
    return sm.drafter_silu_and_mul(fl.fixed_low_m_gemm(x, w, tactic, pdl), pdl)


@pytest.mark.parametrize("deep", [False, True])
@pytest.mark.parametrize("rows", [8, 16, 24, 32, 40, 56, 64])
def test_w10_silu_fold_equals_gate_up_and_silu(drafter_wide128, rows, deep):
    """``silu_fold_gemm`` on the interleaved gate_up weight at the table's tactic for ``rows`` (``fold_tactic``:
    stages lowered to fit the staging buffer) equals, bit for bit, the table's GEMM on the plain weight followed by
    ``drafter_silu_and_mul``; eager and in a CUDA graph; rows with zeros."""
    sk, fl, _ = drafter_wide128
    sf = _module("ops.gemm.silu_fold_gemm")
    sm = _module("ops.spec.drafter_silu_and_mul")
    fl.configure(tep8_wide64=True, tep8_drafter_wide=True, tep8_drafter_deep_stages=deep)
    x, w = _w10_operands(fl, "gate_up", rows, 1500 + rows)
    k, n = fl._TP8_DRAFTER_WIDE_SHAPES["gate_up"]
    tactic = fl.tactic_for(rows, k, n)
    folded = sf.fold_tactic(tactic)
    assert folded is not None and (folded.mma_m, folded.mma_n, folded.split_k) == tuple(tactic[:3])
    wi = sf.interleave_gate_up(w, tactic[0]).contiguous()
    want = _w10j_plain(fl, sm, x, w, tactic, True)
    got = sf.silu_fold_gemm(x, wi, folded, True)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        sf.silu_fold_gemm(x, wi, folded, True)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = sf.silu_fold_gemm(x, wi, folded, True)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (rows, deep, tuple(folded.__dict__.values()))
    assert torch.equal(replay.view(torch.int16), want.view(torch.int16)), (rows, deep)


@pytest.mark.parametrize("rows", [72, 96, 128])
def test_w10_silu_fold_interleaved_activation_above_64_rows(drafter_wide128, rows):
    """Above the fold (65..128 rows) the folded MLP runs the built-in GEMM on the interleaved weight, then the
    interleaved activation (``half`` 32): bit for bit the built-in GEMM on the plain weight + the plain activation."""
    sk, fl, _ = drafter_wide128
    sf = _module("ops.gemm.silu_fold_gemm")
    sm = _module("ops.spec.drafter_silu_and_mul")
    x, w = _w10_operands(fl, "gate_up", rows, 1600 + rows)
    wi = sf.interleave_gate_up(w, 64).contiguous()
    want = sm.drafter_silu_and_mul(F.linear(x, w), False)
    got = sm.drafter_silu_and_mul(F.linear(x, wi), False, 32)
    torch.cuda.synchronize()
    assert torch.equal(got.view(torch.int16), want.view(torch.int16)), rows


def test_w10_silu_fold_timing(drafter_wide128, capsys):
    """(Info) us per call of chains of 6 cold gate_up weights behind PDL: GEMM + activation vs the fold, deep stages."""
    sk, fl, _ = drafter_wide128
    sf = _module("ops.gemm.silu_fold_gemm")
    sm = _module("ops.spec.drafter_silu_and_mul")
    fl.configure(tep8_wide64=True, tep8_drafter_wide=True, tep8_drafter_deep_stages=True)
    k, n = fl._TP8_DRAFTER_WIDE_SHAPES["gate_up"]
    g = torch.Generator(device="cuda").manual_seed(17)
    ws = [_randn(g, n, k, 0.02) for _ in range(6)]
    wis = [sf.interleave_gate_up(w, 64).contiguous() for w in ws]

    def time_calls(calls):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for f in calls:
                f()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for f in calls:
                f()
        times = []
        for _ in range(11):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1e3 / len(calls))
        return sorted(times)[5]

    lines = []
    for rows in (8, 16, 32, 48, 64):
        x = _randn(g, rows, k)
        tactic = fl.tactic_for(rows, k, n)
        folded = sf.fold_tactic(tactic)
        plain = time_calls([(lambda i=i: _w10j_plain(fl, sm, x, ws[i], tactic, True)) for i in range(6)])
        fold = time_calls([(lambda i=i: sf.silu_fold_gemm(x, wis[i], folded, True)) for i in range(6)])
        lines.append(
            f"[W10j] {rows:2d} rows: gate_up s{tactic[3]} + silu {plain:6.2f} us, fold s{folded.ab_stages} {fold:6.2f} us "
            f"({fold - plain:+.2f}; x5 layers {5 * (fold - plain):+.1f} us/step)"
        )
    with capsys.disabled():
        print("\n" + "\n".join(lines))


# --------------------------------------------------------------------------------------------------
# W10: the split-K publication census at 65..128 rows, the pool on 64x16 and 64x32 (WIDE_POOL_TILE)
# --------------------------------------------------------------------------------------------------
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _census_poll(
    slot_ptr,
    final_ptr,
    bad_ptr,
    early_ptr,
    n,
    SENT: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_ITERS: tl.constexpr,
):
    """Spin on ``slot`` until every element holds its final value (or MAX_ITERS polls): ``bad`` counts
    polls that saw a value other than the sentinel and the final one, ``early`` the polls that saw the
    sentinel (> 0: the element was observed before its store)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    final = tl.load(final_ptr + offs, mask=mask, other=SENT)
    bad = tl.zeros([BLOCK], dtype=tl.int32)
    early = tl.zeros([BLOCK], dtype=tl.int32)
    pending = tl.sum(mask.to(tl.int32))
    it = tl.full([], 0, tl.int32)
    while (pending > 0) & (it < MAX_ITERS):
        v = tl.load(slot_ptr + offs, mask=mask, other=SENT, volatile=True)
        sent = v == SENT
        early += (sent & mask).to(tl.int32)
        bad += ((v != final) & ~sent & mask).to(tl.int32)
        pending = tl.sum(((v != final) & mask).to(tl.int32))
        it += 1
    tl.store(bad_ptr + offs, bad, mask=mask)
    tl.store(early_ptr + offs, early, mask=mask)


def _census_run(pp, launch, slot_i16: torch.Tensor, final_i16: torch.Tensor):
    """Poller on one stream, then (after a ~1 ms device sleep, so the poller polls first) the producer
    ``launch()`` on another: ``(bad, early)`` per element."""
    n = final_i16.numel()
    bad = torch.zeros(n, dtype=torch.int32, device="cuda")
    early = torch.zeros(n, dtype=torch.int32, device="cuda")
    slot_i16.fill_(pp._SENTINEL_I16)
    torch.cuda.synchronize()
    poll_stream, producer_stream = torch.cuda.Stream(), torch.cuda.Stream()
    block = 1024
    with torch.cuda.stream(poll_stream):
        _census_poll[(triton.cdiv(n, block),)](
            slot_i16.view(-1), final_i16.view(-1), bad, early, n,
            SENT=pp._SENTINEL_I16, BLOCK=block, MAX_ITERS=2_000_000,
        )  # fmt: skip
    with torch.cuda.stream(producer_stream):
        torch.cuda._sleep(2_000_000)
        launch()
    torch.cuda.synchronize()
    return bad, early


@pytest.fixture
def census():
    sk = _module("ops.gemm.splitk_gemm")
    pp = _module("ops.comm.projection_publish")
    sk.configure(wide64=True)
    pp.configure(
        tep8_publish_splitk=True,
        tep8_publish_splitk_max_rows=128,
        tep8_publish_splitk_wide_pool_tile="64x32",
        tep8_publish_splitk_wide_unpool_tile="64x64",
    )
    try:
        with torch.no_grad():
            yield sk, pp
    finally:
        pp.configure()
        sk.configure()


def _census_sites(pp):
    """(site, (K, columns), tactic) of every one-launch publication the census drives."""
    unpool = pp.UNPOOL_SLICE
    return (
        ("pool 64x16", pp.POOL_SLICE, pp.TP8_PUBLISH_SPLITK_TACTICS[pp.POOL_SLICE]),
        ("pool 64x32", pp.POOL_SLICE, pp.SPLITK_WIDE_POOL_TILES["64x32"]),
        ("pool 64x64", pp.POOL_SLICE, pp.SPLITK_WIDE_POOL_TILES["64x64"]),
        ("unpool 64x32", unpool, pp.TP8_PUBLISH_SPLITK_TACTICS[unpool]),
        ("unpool 64x64", unpool, pp.SPLITK_WIDE_UNPOOL_TILES["64x64"]),
    )


#: The census sites whose producer keys are certified (W10 job 1027577 added the 64x32 / 64x64 entry).
_CERTIFIED_SITES = ("pool 64x16", "pool 64x32", "pool 64x64", "unpool 64x32", "unpool 64x64")


def test_w10_census_keys_are_certified(census):
    """Every tile the 65..128-row publication launches compiles to a producer key certified at this SM
    count, and ``_splitk_publisher`` takes WIDE_POOL_TILE 64x32 (fails, naming the key, otherwise)."""
    sk, pp = census
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    certified = {
        key for entry in pp.CERTIFIED_SPLITK_PRODUCERS_TP8 if entry[0] == sms for key in entry[1:]
    }
    for site, _, tactic in _census_sites(pp):
        key = pp._splitk_site_key(sk, tactic, (pp.SPLITK_MAX_ROWS,))
        if site in _CERTIFIED_SITES:
            assert key in certified, f"{site}: producer key {key} not certified at {sms} SMs"
        else:
            print(f"W10_CENSUS_KEY {site} at {sms} SMs: {key}", flush=True)
    assert pp._splitk_publisher() is not None
    assert tuple(pp._splitk_state.get("wide_pool") or ()) == (64, 32, 4, 8)


def test_w10_unpool_64x64_is_refused_until_certified(census):
    """WIDE_UNPOOL_TILE 64x64: the gate takes it exactly when an entry names its producer key at this
    SM count; refused (the served unpool tile) otherwise. Certifying it: this file's census passing,
    then an entry naming the printed key."""
    sk, pp = census
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    tactic = pp.SPLITK_WIDE_UNPOOL_TILES["64x64"]
    key = pp._splitk_site_key(sk, tactic, (pp.SPLITK_MAX_ROWS,))
    assert key is not None and key[1:] == (tuple(tactic), "owner", "v4", "opt-in", True), key
    certified = {
        k for entry in pp.CERTIFIED_SPLITK_PRODUCERS_TP8 if entry[0] == sms for k in entry[1:]
    }
    assert pp._splitk_publisher() is not None
    taken = pp._splitk_state.get("wide_unpool")
    assert (taken is not None) == (key in certified), (key, taken)
    if taken is not None:
        assert tuple(taken) == tuple(tactic)


@pytest.mark.parametrize("rows", [65, 72, 96, 104, 120, 127, 128])
def test_w10_census_every_slot_element_once_with_its_final_value(census, rows):
    sk, pp = census
    disp = _module("kernels.dense_gemms.dispatcher")
    for site, (k, n), fields in _census_sites(pp):
        tactic = sk.SplitKTactic(*fields)
        assert tactic.split_k == 4, site
        cell = "pool" if k == pp.POOL_SLICE[0] else "unpool"
        eight = sk.SplitKTactic(*disp.parse_kernel(disp.SM107_KERNELS[cell][-1][1])[1])
        served = sk.SplitKTactic(*pp.TP8_PUBLISH_SPLITK_TACTICS[(k, n)])
        g = torch.Generator(device="cuda").manual_seed(1000 + rows + k)
        for trial in range(2):
            hidden = _randn(g, rows, k, 1.0 + trial)
            if trial:
                hidden[::7] = 0
            weight = _randn(g, n, k, 0.02)
            # The final values: the 33..64-row tactic on 64-row chunks, and the 8-row cells.
            chunks = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
            cells = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
            for row0 in range(0, rows, 64):
                row1 = min(rows, row0 + 64)
                sk.run_splitk_dense(
                    hidden[row0:row1], weight.t(), None, chunks[row0:row1], True, served
                )
            for row0 in range(0, rows, 8):
                row1 = min(rows, row0 + 8)
                sk.run_splitk_dense(
                    hidden[row0:row1], weight.t(), None, cells[row0:row1], True, eight
                )
            torch.cuda.synchronize()
            assert torch.equal(chunks.view(torch.int16), cells.view(torch.int16)), (site, trial)
            buf = torch.full((rows + 8, n), pp._SENTINEL_I16, dtype=torch.int16, device="cuda")
            slot = buf[:rows]
            bad, early = _census_run(
                pp,
                lambda: pp._run_splitk_wide(hidden, weight, slot.view(torch.bfloat16), tactic),
                slot,
                chunks.view(torch.int16),
            )
            where = (site, rows, trial)
            assert int(bad.sum()) == 0, f"{where}: {int((bad > 0).sum())} elements seen non-final"
            assert bool((early > 0).all()), (
                f"{where}: {int((early == 0).sum())} elements unobserved"
            )
            assert torch.equal(slot, chunks.view(torch.int16)), where
            assert bool((buf[rows:] == pp._SENTINEL_I16).all()), f"{where}: wrote past its slot"


# --------------------------------------------------------------------------------------------------
# W11: QKV one-launch candidates at 65..128 rows (WIDE_TILE / WIDE_SPLIT_K) and the unpool tile timing
# --------------------------------------------------------------------------------------------------
#: (name, (mma_m, mma_n), split_k; 0 = the served tactic's).
QKV_CANDIDATES = [
    ("64x64 k4", (64, 64), 0),
    ("64x128 k4", (64, 128), 0),
    ("128x128 k4", (128, 128), 0),
    ("128x128 k8", (128, 128), 8),
    ("128x64 k8", (128, 64), 8),
]


def _w11_kernels(disp, dk, tile, split_k):
    """(one-launch kernel, its split_k's 64-row reference kernel) for a candidate."""
    k, n = dk.QKV_SHAPE
    served = disp.TEP8_WIDE64_KERNELS["qkv"]
    kernel = disp._wide_tiles_kernel(served, tile, k, n, split_k)
    fields = dk.parse_kernel(kernel)[1]
    ref = disp._wide_tiles_kernel(served, (64, 64), k, n, fields[2])
    return kernel, ref


@pytest.mark.parametrize("name, tile, split_k", QKV_CANDIDATES, ids=[c[0] for c in QKV_CANDIDATES])
@pytest.mark.parametrize("rows", [65, 72, 96, 128])
def test_w11_qkv_candidate_bits(qkv_wide128, rows, name, tile, split_k):
    sk, disp, dk = qkv_wide128
    kernel, ref = _w11_kernels(disp, dk, tile, split_k)
    fields = dk.parse_kernel(kernel)[1]
    x, w = _w6_operands(dk, rows, 1100 + rows)
    out = torch.empty(rows, w.shape[0], dtype=torch.bfloat16, device="cuda")
    sk.run_splitk_dense_tiles(x, w.t(), out, True, sk.SplitKTactic(*fields))
    chunks = _w6_reference(sk, dk, x, w, ref, 64)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), chunks.view(torch.int16)), (name, kernel, ref)
    if fields[2] == 4:
        eight = _w6_reference(sk, dk, x, w, disp.SM107_KERNELS["qkv"][-1][1], 8)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), eight.view(torch.int16)), (name, kernel)
    assert torch.allclose(out.float(), F.linear(x, w).float(), rtol=2e-2, atol=2e-2), name


def _w11_per_call(run, x, weights) -> float:
    """us per call of ``run(x, w)`` over a CUDA graph of one call per weight (PDL as served)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for w in weights[:2]:
            run(x, w)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for w in weights:
            run(x, w)
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (20 * len(weights))


@pytest.fixture(scope="module")
def _qkv_weights():
    with torch.no_grad():
        g = torch.Generator(device="cuda").manual_seed(1188)
        k, n = _module("ops.gemm.dense_gemm").QKV_SHAPE
        return [_randn(g, n, k, 0.02) for _ in range(62)]


@pytest.mark.parametrize("name, tile, split_k", QKV_CANDIDATES, ids=[c[0] for c in QKV_CANDIDATES])
def test_w11_qkv_candidate_timing(qkv_wide128, _qkv_weights, capsys, name, tile, split_k):
    """A timing print per candidate (and cuBLAS), no assertion beyond the build."""
    sk, disp, dk = qkv_wide128
    kernel, _ = _w11_kernels(disp, dk, tile, split_k)
    tactic = sk.SplitKTactic(*dk.parse_kernel(kernel)[1])
    g = torch.Generator(device="cuda").manual_seed(1189)
    lines = []
    for rows in (72, 96, 128):
        x = _randn(g, rows, dk.QKV_SHAPE[0])
        out = torch.empty(rows, dk.QKV_SHAPE[1], dtype=torch.bfloat16, device="cuda")
        ours = _w11_per_call(
            lambda a, w: sk.run_splitk_dense_tiles(a, w.t(), out, True, tactic), x, _qkv_weights
        )
        cublas = _w11_per_call(lambda a, w: F.linear(a, w), x, _qkv_weights)
        lines.append(
            f"[W11] QKV {rows:3d} rows {name:11s} {kernel:22s}: {ours:6.2f} us/call, cuBLAS "
            f"{cublas:6.2f}, x62 layers: {(ours - cublas) * 62 / 1e3:+.3f} ms/step"
        )
    with capsys.disabled():
        print("\n" + "\n".join(lines), flush=True)


def test_w11_unpool_tile_timing(pool_tile, capsys):
    """A timing print: the unpool slice's one launch on 64x32 (served) vs 64x64 (WIDE_UNPOOL_TILE),
    62-layer chain of distinct weight slices, at 72 / 96 / 128 rows."""
    sk, pp = pool_tile
    k, n = pp.UNPOOL_SLICE
    g = torch.Generator(device="cuda").manual_seed(1190)
    weights = [_randn(g, n, k, 0.02) for _ in range(62)]
    lines = []
    for rows in (72, 96, 128):
        hidden = _randn(g, rows, k)
        slot = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
        res = {}
        for name, fields in (
            ("64x32", pp.TP8_PUBLISH_SPLITK_TACTICS[pp.UNPOOL_SLICE]),
            ("64x64", pp.SPLITK_WIDE_UNPOOL_TILES["64x64"]),
        ):
            tactic = sk.SplitKTactic(*fields)
            res[name] = _w11_per_call(
                lambda a, w: pp._run_splitk_wide(a, w, slot, tactic), hidden, weights
            )
        lines.append(
            f"[W11] unpool slice {rows:3d} rows: 64x32 {res['64x32']:6.2f} us/call, 64x64 "
            f"{res['64x64']:6.2f} us/call, x62 layers: {(res['64x64'] - res['64x32']) * 62 / 1e3:+.3f}"
            " ms/step"
        )
    with capsys.disabled():
        print("\n" + "\n".join(lines), flush=True)



# --------------------------------------------------------------------------------------------------
# W12: the 65..128-row o_proj scatter producer by mode and stages (PRODUCER / PRODUCER_STAGES)
# --------------------------------------------------------------------------------------------------
#: (mode, stages); stages 0 = the mode's default (DIRECT128_PRODUCER_STAGES).
PRODUCER_CANDIDATES = [
    ("grid", 3),
    ("grid", 4),
    ("grid", 5),
    ("tile128", 3),
    ("tile128", 4),
    ("chunks", 0),
]


def _produce_staged(wide, mode, stages, a, w, window, parity, rank):
    """``_produce`` with an explicit one-launch stage count."""
    if mode == "chunks":
        _produce(wide, mode, a, w, window, parity, rank)
        return
    pub = _Publication(wide, window)
    rows = a.shape[0]
    view = window.view(parity, rank)[:rows]
    wide.run_splitk_scatter_wide(
        a, w.t(), pub.chunk_view(view, 0, rows), True, wide.hop128_tactic(mode, stages),
        window.stride_b, pub.target_slots(),
    )  # fmt: skip


@pytest.mark.parametrize(
    "mode, stages", PRODUCER_CANDIDATES, ids=[f"{m}@{s}" for m, s in PRODUCER_CANDIDATES]
)
def test_w12_producer_stages_bits_and_timing(hop128, capsys, mode, stages):
    wide = hop128
    g = torch.Generator(device="cuda").manual_seed(1200)
    window = _FakeWindow(128)
    weights = [_randn(g, HIDDEN, K_OPROJ, 0.03) for _ in range(62)]
    lines = []
    with torch.no_grad():
        for rows in (72, 96, 128):
            a = _randn(g, rows, K_OPROJ)
            window.reset()
            _produce_staged(wide, mode, stages, a, weights[0], window, 1, 3)
            torch.cuda.synchronize()
            want = F.linear(a, weights[0]).view(torch.int16)
            got = window.polled(1, 3, rows).view(torch.int16)
            assert torch.equal(got, want), (mode, stages, rows)
            per_call = _w11_per_call(
                lambda x, w: _produce_staged(wide, mode, stages, x, w, window, 1, 3), a, weights
            )
            cublas = _w11_per_call(lambda x, w: F.linear(x, w), a, weights)
            lines.append(
                f"[W12] o_proj producer {rows:3d} rows {mode:7s}@{stages}: {per_call:6.2f} us/call "
                f"(cuBLAS F.linear {cublas:6.2f})"
            )
    window.reset()
    with capsys.disabled():
        print("\n" + "\n".join(lines), flush=True)



# --------------------------------------------------------------------------------------------------
# W13: QKV at 65..128 rows through cuBLAS variants (does one skip the splitKreduce launch?)
# --------------------------------------------------------------------------------------------------
_W13_SCRIPT = r"""
import sys, torch, torch.nn.functional as F
rows_list = [int(r) for r in sys.argv[1].split(",")]
mode = sys.argv[2]
torch.manual_seed(0)
k, n = 8192, 2304
ws = [(torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16) for _ in range(62)]
bias = torch.zeros(n, device="cuda", dtype=torch.bfloat16)
if "lt" in mode:
    torch.backends.cuda.preferred_blas_library("cublaslt")
run = (lambda x, w: torch.addmm(bias, x, w.t())) if "addmm" in mode else (lambda x, w: F.linear(x, w))
for rows in rows_list:
    x = (torch.randn(rows, k, device="cuda")).to(torch.bfloat16)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for w in ws[:2]:
            run(x, w)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        run(x, ws[0]); torch.cuda.synchronize()
    names = sorted({e.name[:70] for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for w in ws:
            run(x, w)
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(20):
        g.replay()
    b.record(); torch.cuda.synchronize()
    us = a.elapsed_time(b) * 1e3 / (20 * len(ws))
    print(f"[W13] QKV {rows:3d} rows {mode:18s}: {us:6.2f} us/call kernels {names}", flush=True)
"""


@pytest.mark.parametrize(
    "mode, env",
    [
        ("linear", {}),
        ("addmm", {}),
        ("linear-lt", {}),
        ("addmm-lt", {}),
        ("linear-lt-ws0", {"CUBLASLT_WORKSPACE_SIZE": "0"}),
        ("addmm-lt-ws64m", {"CUBLASLT_WORKSPACE_SIZE": "65536"}),
        ("linear-ws64m", {"CUBLASLT_WORKSPACE_SIZE": "65536", "CUBLAS_WORKSPACE_CONFIG": ":65536:2"}),
    ],
)
def test_w13_qkv_cublas_variants(capsys, mode, env):
    import os
    import subprocess

    full = {**os.environ, **env}
    proc = subprocess.run(
        [sys.executable, "-c", _W13_SCRIPT, "72,96,128", mode],
        env=full,
        capture_output=True,
        text=True,
        timeout=600,
    )
    with capsys.disabled():
        print("\n" + "\n".join(line for line in proc.stdout.splitlines() if "[W13]" in line), flush=True)
    assert proc.returncode == 0, proc.stderr[-2000:]


# --------------------------------------------------------------------------------------------------
# W14: QKV 2-CTA split-K (ops/gemm/splitk_gemm_2cta) at 65..128 rows
# --------------------------------------------------------------------------------------------------
#: (name, split_k, pre-wait L2 prefetch of the weight slice, reduce-scatter epilogue).
W14_ARMS = [
    ("2cta k4", 4, True, False),
    ("2cta k4 nopf", 4, False, False),
    ("2cta k2", 2, True, False),
    ("2cta k4 rs", 4, True, True),
    ("2cta k2 rs", 2, True, True),
]


@pytest.mark.parametrize("name, split_k, prefetch, rs", W14_ARMS, ids=[a[0] for a in W14_ARMS])
@pytest.mark.parametrize("rows", [65, 72, 96, 128])
def test_w14_qkv_2cta_bits(qkv_wide128, rows, name, split_k, prefetch, rs):
    sk, disp, dk = qkv_wide128
    two = _module("ops.gemm.splitk_gemm_2cta")
    k, n = dk.QKV_SHAPE
    x, w = _w6_operands(dk, rows, 1400 + rows)
    padded = torch.full((rows + 8, n), -1, dtype=torch.int16, device="cuda").view(torch.bfloat16)
    out = padded[:rows]
    two.run_splitk_2cta(x, w, out, True, split_k, prefetch_a=prefetch, rs=rs)
    served = disp.TEP8_WIDE64_KERNELS["qkv"]
    ref = disp._wide_tiles_kernel(served, (64, 64), k, n, split_k)
    chunks = _w6_reference(sk, dk, x, w, ref, 64)
    torch.cuda.synchronize()
    assert bool((padded[rows:].view(torch.int16) == -1).all()), f"{name}: wrote past its output"
    differ = int((out.view(torch.int16) != chunks.view(torch.int16)).sum())
    print(f"[W14] {name} {rows} rows: {differ} elements differ from {ref} (64-row launches)", flush=True)
    assert torch.allclose(out.float(), F.linear(x, w).float(), rtol=2e-2, atol=2e-2), name
    assert differ == 0, f"{name} {rows} rows: not the 64-row launches' bits ({differ} elements)"
    if split_k == 4:
        eight = _w6_reference(sk, dk, x, w, disp.SM107_KERNELS["qkv"][-1][1], 8)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), eight.view(torch.int16)), name


@pytest.mark.parametrize("name, split_k, prefetch, rs", W14_ARMS, ids=[a[0] for a in W14_ARMS])
def test_w14_qkv_2cta_timing(qkv_wide128, _qkv_weights, capsys, name, split_k, prefetch, rs):
    """A timing print per arm (and cuBLAS), no assertion beyond the build."""
    sk, disp, dk = qkv_wide128
    two = _module("ops.gemm.splitk_gemm_2cta")
    g = torch.Generator(device="cuda").manual_seed(1401)
    lines = []
    for rows in (72, 96, 128):
        x = _randn(g, rows, dk.QKV_SHAPE[0])
        out = torch.empty(rows, dk.QKV_SHAPE[1], dtype=torch.bfloat16, device="cuda")
        ours = _w11_per_call(
            lambda a, w: two.run_splitk_2cta(a, w, out, True, split_k, prefetch_a=prefetch, rs=rs),
            x,
            _qkv_weights,
        )
        cublas = _w11_per_call(lambda a, w: F.linear(a, w), x, _qkv_weights)
        lines.append(
            f"[W14] QKV {rows:3d} rows {name:13s}: {ours:6.2f} us/call, cuBLAS {cublas:6.2f}, "
            f"x62 layers: {(ours - cublas) * 62 / 1e3:+.3f} ms/step"
        )
    with capsys.disabled():
        print("\n" + "\n".join(lines), flush=True)


def test_w14_qkv_2cta_row_sweep(qkv_wide128, _qkv_weights, capsys):
    """A timing print of the 2-CTA kernel (split_k 4 and 2, pre-wait prefetch) against cuBLAS at every
    8th row count of 65..128 (the band rule of WIDE_TILE 2cta), and the stage depth at 96 / 128 rows."""
    sk, disp, dk = qkv_wide128
    two = _module("ops.gemm.splitk_gemm_2cta")
    g = torch.Generator(device="cuda").manual_seed(1402)
    lines = []
    for rows in (65, 72, 80, 88, 96, 104, 112, 120, 128):
        x = _randn(g, rows, dk.QKV_SHAPE[0])
        out = torch.empty(rows, dk.QKV_SHAPE[1], dtype=torch.bfloat16, device="cuda")
        k4 = _w11_per_call(lambda a, w: two.run_splitk_2cta(a, w, out, True, 4), x, _qkv_weights)
        k2 = _w11_per_call(lambda a, w: two.run_splitk_2cta(a, w, out, True, 2), x, _qkv_weights)
        k4rs = _w11_per_call(
            lambda a, w: two.run_splitk_2cta(a, w, out, True, 4, rs=True), x, _qkv_weights
        )
        cublas = _w11_per_call(lambda a, w: F.linear(a, w), x, _qkv_weights)
        lines.append(
            f"[W14] sweep QKV {rows:3d} rows: 2cta k4 {k4:6.2f} k4 rs {k4rs:6.2f} k2 {k2:6.2f} us/call, "
            f"cuBLAS {cublas:6.2f}, k4 rs - cuBLAS x62: {(k4rs - cublas) * 62 / 1e3:+.3f} ms/step"
        )
    for rows in (96, 128):
        x = _randn(g, rows, dk.QKV_SHAPE[0])
        out = torch.empty(rows, dk.QKV_SHAPE[1], dtype=torch.bfloat16, device="cuda")
        per = {
            stages: _w11_per_call(
                lambda a, w, s=stages: two.run_splitk_2cta(a, w, out, True, 4, ab_stages=s),
                x,
                _qkv_weights,
            )
            for stages in (4, 6, 8)
        }
        lines.append(
            f"[W14] stages QKV {rows:3d} rows k4: "
            + ", ".join(f"{s} stages {t:6.2f}" for s, t in per.items())
            + " us/call"
        )
    with capsys.disabled():
        print("\n" + "\n".join(lines), flush=True)


@pytest.mark.parametrize("rows", [33, 40, 48, 56, 64])
def test_w14_qkv_2cta_low_rows_bits(qkv_wide128, rows):
    """At 33..64 rows (the served wide64 split-K's band) the 2-CTA kernel (reduce-scatter, split_k 4)
    writes exactly the served launch's bits: a candidate replacement with no numeric change."""
    sk, disp, dk = qkv_wide128
    two = _module("ops.gemm.splitk_gemm_2cta")
    x, w = _w6_operands(dk, rows, 1500 + rows)
    out = torch.empty(rows, w.shape[0], dtype=torch.bfloat16, device="cuda")
    two.run_splitk_2cta(x, w, out, True, 4, rs=True)
    served = _w6_reference(sk, dk, x, w, disp.TEP8_WIDE64_KERNELS["qkv"], 64)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), served.view(torch.int16)), rows


def test_w14_qkv_2cta_low_rows_timing(qkv_wide128, _qkv_weights, capsys):
    """A timing print at 33..64 rows: the 2-CTA kernel (reduce-scatter, split_k 4) against the served
    wide64 split-K tactic and cuBLAS (62-layer chain of distinct weights, PDL)."""
    sk, disp, dk = qkv_wide128
    two = _module("ops.gemm.splitk_gemm_2cta")
    served = sk.SplitKTactic(*dk.parse_kernel(disp.TEP8_WIDE64_KERNELS["qkv"])[1])
    g = torch.Generator(device="cuda").manual_seed(1403)
    lines = []
    for rows in (33, 40, 48, 56, 64):
        x = _randn(g, rows, dk.QKV_SHAPE[0])
        out = torch.empty(rows, dk.QKV_SHAPE[1], dtype=torch.bfloat16, device="cuda")
        rs = _w11_per_call(
            lambda a, w: two.run_splitk_2cta(a, w, out, True, 4, rs=True), x, _qkv_weights
        )
        wide64 = _w11_per_call(
            lambda a, w: sk.run_splitk_dense(a, w.t(), None, out, True, served), x, _qkv_weights
        )
        cublas = _w11_per_call(lambda a, w: F.linear(a, w), x, _qkv_weights)
        lines.append(
            f"[W14] low QKV {rows:3d} rows: 2cta k4 rs {rs:6.2f}, served {served} {wide64:6.2f}, "
            f"cuBLAS {cublas:6.2f} us/call; 2cta - served x62: {(rs - wide64) * 62 / 1e3:+.3f} ms/step"
        )
    with capsys.disabled():
        print("\n" + "\n".join(lines), flush=True)


@pytest.mark.parametrize("rows", [65, 96, 128])
def test_w14_qkv_2cta_k128_bits(qkv_wide128, rows):
    """The 2-CTA kernel on 128-wide K stages (reduce-scatter, split_k 4) writes the served 64-row
    launches' bits (the MMAs still run in ascending k)."""
    sk, disp, dk = qkv_wide128
    two = _module("ops.gemm.splitk_gemm_2cta")
    x, w = _w6_operands(dk, rows, 1600 + rows)
    out = torch.empty(rows, w.shape[0], dtype=torch.bfloat16, device="cuda")
    two.run_splitk_2cta(x, w, out, True, 4, rs=True, k_tile=128)
    chunks = _w6_reference(sk, dk, x, w, disp.TEP8_WIDE64_KERNELS["qkv"], 64)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), chunks.view(torch.int16)), rows


def test_w14_qkv_2cta_k128_timing(qkv_wide128, _qkv_weights, capsys):
    """A timing print: 64- vs 128-wide K stages of the 2-CTA kernel (reduce-scatter, split_k 4), the
    128-wide one at 4 / 5 / 6 stages, against cuBLAS at 64 / 96 / 128 rows (64 rows: and the served
    wide64 split-K)."""
    sk, disp, dk = qkv_wide128
    two = _module("ops.gemm.splitk_gemm_2cta")
    served = sk.SplitKTactic(*dk.parse_kernel(disp.TEP8_WIDE64_KERNELS["qkv"])[1])
    g = torch.Generator(device="cuda").manual_seed(1404)
    lines = []
    for rows in (64, 96, 128):
        x = _randn(g, rows, dk.QKV_SHAPE[0])
        out = torch.empty(rows, dk.QKV_SHAPE[1], dtype=torch.bfloat16, device="cuda")
        per = {}
        for name, kw in (
            ("k64", dict(k_tile=64)),
            ("k128", dict(k_tile=128)),
            ("k128 s4", dict(k_tile=128, ab_stages=4)),
            ("k128 s5", dict(k_tile=128, ab_stages=5)),
        ):
            per[name] = _w11_per_call(
                lambda a, w, kw=kw: two.run_splitk_2cta(a, w, out, True, 4, rs=True, **kw),
                x,
                _qkv_weights,
            )
        per["cuBLAS"] = _w11_per_call(lambda a, w: F.linear(a, w), x, _qkv_weights)
        if rows <= 64:
            per["served"] = _w11_per_call(
                lambda a, w: sk.run_splitk_dense(a, w.t(), None, out, True, served), x, _qkv_weights
            )
        lines.append(
            f"[W14] ktile QKV {rows:3d} rows: " + ", ".join(f"{k} {v:6.2f}" for k, v in per.items())
        )
    with capsys.disabled():
        print("\n" + "\n".join(lines), flush=True)


@pytest.mark.parametrize("rows", [64, 65, 96, 128])
def test_w14_qkv_2cta_prestage_bits(qkv_wide128, rows):
    """The 2-CTA kernel with its first stages' weights loaded before the PDL wait (prestage;
    reduce-scatter, split_k 4) writes the served 64-row launches' bits; a CUDA graph of two calls
    (the second launched under PDL behind the first) too."""
    sk, disp, dk = qkv_wide128
    two = _module("ops.gemm.splitk_gemm_2cta")
    x, w = _w6_operands(dk, rows, 1700 + rows)
    out = torch.empty(rows, w.shape[0], dtype=torch.bfloat16, device="cuda")
    two.run_splitk_2cta(x, w, out, True, 4, rs=True, prestage=True)
    chunks = _w6_reference(sk, dk, x, w, disp.TEP8_WIDE64_KERNELS["qkv"], 64)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), chunks.view(torch.int16)), rows
    out2 = torch.empty_like(out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        two.run_splitk_2cta(x, w, out2, True, 4, rs=True, prestage=True)
        two.run_splitk_2cta(x, w, out, True, 4, rs=True, prestage=True)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), chunks.view(torch.int16)), rows
    assert torch.equal(out2.view(torch.int16), chunks.view(torch.int16)), rows


def test_w14_qkv_2cta_prestage_timing(qkv_wide128, _qkv_weights, capsys):
    """A timing print: the 2-CTA kernel (reduce-scatter, split_k 4) with and without prestage (and
    without the L2 prefetch) against cuBLAS and, at 64 rows, the served wide64 split-K."""
    sk, disp, dk = qkv_wide128
    two = _module("ops.gemm.splitk_gemm_2cta")
    served = sk.SplitKTactic(*dk.parse_kernel(disp.TEP8_WIDE64_KERNELS["qkv"])[1])
    g = torch.Generator(device="cuda").manual_seed(1405)
    lines = []
    for rows in (64, 96, 128):
        x = _randn(g, rows, dk.QKV_SHAPE[0])
        out = torch.empty(rows, dk.QKV_SHAPE[1], dtype=torch.bfloat16, device="cuda")
        per = {}
        for name, kw in (
            ("rs", dict()),
            ("rs prestage", dict(prestage=True)),
            ("rs prestage nopf", dict(prestage=True, prefetch_a=False)),
        ):
            per[name] = _w11_per_call(
                lambda a, w, kw=kw: two.run_splitk_2cta(a, w, out, True, 4, rs=True, **kw),
                x,
                _qkv_weights,
            )
        per["cuBLAS"] = _w11_per_call(lambda a, w: F.linear(a, w), x, _qkv_weights)
        if rows <= 64:
            per["served"] = _w11_per_call(
                lambda a, w: sk.run_splitk_dense(a, w.t(), None, out, True, served), x, _qkv_weights
            )
        lines.append(
            f"[W14] prestage QKV {rows:3d} rows: "
            + ", ".join(f"{k} {v:6.2f}" for k, v in per.items())
        )
    with capsys.disabled():
        print("\n" + "\n".join(lines), flush=True)


# --------------------------------------------------------------------------
# W10k: drafter_kv_side_chain__PAGED (the paged side chain's integer prologue and store, one launch each)
# --------------------------------------------------------------------------
def _w10k_setup(gens: int, rows: int, layers: int, windowed: bool, seed: int):
    """A TP8 drafter's paged context as the DFlash worker keeps it: the manager's encoded block offsets (with
    placeholders past each request's extent), the decoded table / extents, lengths, slots and 5 HND layer pools."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    page, width, pools_n, divisor, offset, nkv, hd = 32, 512, 2, 4, 3, 1, 128
    pages = 16 * width + 1
    src = torch.full((pools_n, gens + 3, 2, width), -1, dtype=torch.int32, device="cuda")
    extents = torch.randint(40, width, (gens,), generator=g, device="cuda")
    # Distinct pages per request (page 0 is what a placeholder decodes to; no live column maps there).
    perm = torch.randperm(pages - 1, generator=g, device="cuda") + 1
    for b in range(gens):
        e = int(extents[b])
        src[1, b, :, :e] = (perm[b * width : b * width + e].to(torch.int32) * divisor + offset)
        # a placeholder inside the allocation, below the step's rows (an evicted page of a sliding layer)
        src[1, b, :, 3] = -7
    ctx_len = torch.zeros(16, dtype=torch.int64, device="cuda")
    slots = torch.randperm(16, generator=g, device="cuda")[:gens].to(torch.int64)
    for b in range(gens):
        cap = int(extents[b]) * page
        ctx_len[slots[b]] = int(torch.randint(8 * page, cap - 10, (1,), generator=g, device="cuda"))
    # one request at its allocation's end: its last rows clamp
    ctx_len[slots[0]] = int(extents[0]) * page - 2
    ctx_base = (ctx_len // 3).to(torch.int64) if windowed else None
    pool_list = [
        torch.zeros((pages, 2, nkv, page, hd), dtype=torch.bfloat16, device="cuda") for _ in range(layers)
    ]
    return dict(src=src, slots=slots, ctx_len=ctx_len, ctx_base=ctx_base, pools=pool_list, page=page,
                width=width, divisor=divisor, offset=offset, nkv=nkv, hd=hd, g=g)


def _w10k_v2_views(storage: torch.Tensor, layers: int, pages: int, nkv: int, page: int, hd: int):
    """KVCacheManagerV2-shaped layer pools: each layer's view starts at its own buffer inside one page-interleaved
    allocation ([slot, layer, K/V, Hkv, page, D]) and runs to the allocation's end, so the page counts differ
    (shape[0] = pages + layers - 1 - layer) while the strides are shared -- what vrg-zk1b's refusal met."""
    pair = 2 * nkv * page * hd
    return [
        torch.as_strided(storage, (pages + layers - 1 - layer, 2, nkv, page, hd),
                         (layers * pair, nkv * page * hd, page * hd, hd, 1), layer * pair)
        for layer in range(layers)
    ]


@pytest.mark.parametrize("v2_pools", [False, True])
@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("gens", [1, 3, 8, 16])
def test_w10_paged_side_chain_prologue_and_store_equal_the_built_in(gens, windowed, v2_pools):
    """``context_kv_paged_prep`` writes the block table, the extents, the columns, the positions and the int32 RoPE
    positions the built-in statements write (``_refresh_ctx_block_tables`` + ``_store_step_context``'s index math,
    the op's ``reference``); ``context_kv_paged_scatter`` leaves every layer's pool bit for bit as the built-in's
    per-layer ``flashinfer.append_paged_kv_cache`` calls do (rows clamped onto one slot excepted: the built-in
    leaves their winner unspecified, the scatter keeps the last row; the test skips those slots). ``v2_pools``: the
    layers are KVCacheManagerV2-shaped views of one interleaved allocation whose page counts differ, and the whole
    allocation (every layer, every other page) must come out bit for bit."""
    import flashinfer

    ps = _module("ops.spec.context_kv_store_paged")
    rows, layers = 8, 5
    s = _w10k_setup(gens, rows, layers, windowed, 2000 + gens + 100 * windowed)
    storage = want_storage = None
    if v2_pools:
        pages = s["pools"][0].shape[0]
        pair = 2 * s["nkv"] * s["page"] * s["hd"]
        storage = _randn(s["g"], (pages + layers) * layers * pair, 1).view(-1)
        want_storage = storage.clone()
        s["pools"] = _w10k_v2_views(storage, layers, pages, s["nkv"], s["page"], s["hd"])
        assert len({p.shape[0] for p in s["pools"]}) == layers and len({p.stride() for p in s["pools"]}) == 1
    src = s["src"][1, :, 0]
    n = gens * rows
    outs = []
    for fn in (ps.context_kv_paged_prep, ps.context_kv_paged_prep.reference):
        table = torch.full((16, s["width"]), 77, dtype=torch.int32, device="cuda")
        counts = torch.full((16,), 5, dtype=torch.int64, device="cuda")
        positions = torch.empty(n, dtype=torch.int64, device="cuda")
        col = torch.empty(n, dtype=torch.int64, device="cuda")
        pos_i32 = torch.empty(n * layers, dtype=torch.int32, device="cuda")
        fn(src, table, counts, s["ctx_len"], s["ctx_base"], s["slots"], positions, col, pos_i32,
           rows, layers, s["offset"], s["divisor"], s["page"])
        outs.append((table, counts, positions, col, pos_i32))
    torch.cuda.synchronize()
    for got, want, name in zip(outs[0], outs[1], ("table", "counts", "positions", "col", "pos_i32")):
        assert torch.equal(got, want), name
    table, counts, positions, col, _ = outs[0]
    # The store: K (post norm + RoPE) [n, layers, nkv * hd] and the fused K/V GEMM's output (V read in place).
    g = s["g"]
    nkv, hd = s["nkv"], s["hd"]
    kv_flat = _randn(g, n, layers * 2 * nkv * hd)
    k_new = _randn(g, n * layers, nkv * hd).view(n, layers, nkv * hd)
    if v2_pools:
        want_pools = _w10k_v2_views(want_storage, layers, s["pools"][0].shape[0] - layers + 1, nkv, s["page"], hd)
    else:
        want_pools = [p.clone() for p in s["pools"]]
    rows_i32 = torch.arange(gens, device="cuda").repeat_interleave(rows).to(torch.int32)
    kv_indptr = torch.arange(0, (16 + 1) * s["width"], s["width"], dtype=torch.int32, device="cuda")
    last = torch.full((16,), s["page"], dtype=torch.int32, device="cuda")
    v_all = kv_flat.view(n, layers, 2, nkv, hd)[:, :, 1]
    for layer in range(layers):
        flashinfer.append_paged_kv_cache(
            append_key=k_new.view(n, layers, nkv, hd)[:, layer].contiguous(),
            append_value=v_all[:, layer].contiguous(),
            batch_indices=rows_i32,
            positions=col.to(torch.int32),
            paged_kv_cache=want_pools[layer],
            kv_indices=table.flatten().contiguous(),
            kv_indptr=kv_indptr,
            kv_last_page_len=last,
            kv_layout="HND",
        )
    ptrs = torch.tensor([p.data_ptr() for p in s["pools"]], dtype=torch.int64, device="cuda")
    ps.context_kv_paged_scatter(k_new, kv_flat, col, counts, table, ptrs, s["pools"][0], rows, layers, nkv, hd)
    torch.cuda.synchronize()
    # Slots two rows of a request clamp onto: the built-in's winner is unspecified; compare everything else.
    dup = torch.zeros(n, dtype=torch.bool, device="cuda")
    cv = col.view(gens, rows)
    dup.view(gens, rows)[:, :-1] = cv[:, :-1] == cv[:, 1:]
    dup.view(gens, rows)[:, 1:] |= cv[:, 1:] == cv[:, :-1]
    b_idx = torch.arange(gens, device="cuda").repeat_interleave(rows)
    pages_ = table[b_idx, col // s["page"]].long()
    toks = col % s["page"]
    for layer in range(layers):
        got_p, want_p = s["pools"][layer], want_pools[layer]
        for i in torch.nonzero(dup).flatten().tolist():
            got_p[pages_[i], :, :, toks[i]] = 0
            want_p[pages_[i], :, :, toks[i]] = 0
        assert torch.equal(got_p.view(torch.int16), want_p.view(torch.int16)), (layer, gens, windowed, v2_pools)
    if v2_pools:
        # Nothing outside the layers' (overlapping-free) buffers moved either: the whole interleaved allocation.
        assert torch.equal(storage.view(torch.int16), want_storage.view(torch.int16)), (gens, windowed)
    assert windowed or int(dup.sum()) >= 1  # request 0 sits at its allocation's end: its rows clamp


# --------------------------------------------------------------------------------------------------
# W15: the exact one-cluster SWA block (swa_one_cluster_block__EXACT): the replaced path's arithmetic
# --------------------------------------------------------------------------------------------------
W15_EXACT = dict(CL8_BAND, cl8_exact=True)


def _w15_rows(swa, step, groups, cfg):
    swa.configure(**cfg)
    pool = step.pool.clone()
    rows = {}
    for ids in groups:
        out = _swa_call(swa, step, step.inputs(ids), pool)
        for k, b in enumerate(ids):
            rows[b] = out[k * SWA_ROWS : (k + 1) * SWA_ROWS].float()
    torch.cuda.synchronize()
    return rows, pool


def _w15_reference(step, ids, pool):
    """The fp64 SWA block of requests ``ids`` on the appended cache ``pool``: the same fused prep and FP8 q
    (q * bf16(kv_scale)), the cache bytes dequantized, then fp64 softmax, P.V and norm-gate.
    Returns [rows, SWA_HQ, SWA_D] fp64."""
    prep = _module("ops.attention.preattn_prep").preattn_prep
    x = step.inputs(ids)
    rope = (step.cos_sin, x["positions"])
    fused, _ = prep(x["qkv"], *rope, SWA_HQ, SWA_HKV, 1e-6, 1.0, step.permuted)
    scale = float(step.kv_scale.item())
    deq = 1.0 / scale
    q = fused[:, : SWA_HQ * SWA_D].float().view(-1, SWA_HQ, SWA_D)
    q_scale = float(torch.tensor(scale).to(torch.bfloat16).float())
    q8 = (q * q_scale).to(torch.bfloat16).float().to(torch.float8_e4m3fn).double()
    gate = _swa_gate(x["qkv"], step.permuted).double()
    group = SWA_HQ // SWA_HKV
    log2_scale = math.log2(math.e) / math.sqrt(SWA_D) * deq * deq
    out = []
    for k, b in enumerate(ids):
        length = step.lens[b]
        pos = torch.arange(length, device="cuda")
        pages = step.bt[b][:, pos // SWA_PAGE].long()  # [2, L]: the K and the V page of each key
        keys = pool[pages[0], :, pos % SWA_PAGE, :].double()  # [L, HKV, D]
        vals = pool[pages[1], :, pos % SWA_PAGE, :].double()
        o = torch.empty(SWA_ROWS, SWA_HQ, SWA_D, dtype=torch.float64, device="cuda")
        for t in range(SWA_ROWS):
            last = length - SWA_ROWS + t
            first = max(0, last - SWA_WINDOW + 1)
            for h in range(SWA_HKV):
                heads = slice(h * group, (h + 1) * group)
                s = q8[k * SWA_ROWS + t, heads] @ keys[first : last + 1, h].T * log2_scale
                p = torch.exp2(s - s.max(1, keepdim=True).values)
                o[t, heads] = p @ vals[first : last + 1, h] / p.sum(1, keepdim=True) * deq
        normed = o / torch.sqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6)
        out.append(normed * torch.sigmoid(gate[k * SWA_ROWS : (k + 1) * SWA_ROWS]))
    return torch.cat(out)


@pytest.mark.parametrize("n", [2, 5, 9, 11])
def test_w15_cl8_exact_is_the_replaced_path_up_to_order(cl8_tep8, n):
    """CL8 EXACT (fp32 exp2 of the fp32 scaled scores, fp32 sums, bf16 P / V, fp32 merge) against the path it
    replaces (ONE2 at 5..8 requests, the split + combine at 2..4; above 8, two calls of <= 8, each request's
    rows batch-invariant): the same K/V bytes appended. Against the fp64 SWA block, in bf16 ULP at each head
    row's RMS, the exact build's error is the replaced path's (max <= max(1.25x, +4 row-ULP), mean within 0.02). The two
    differ only in the key partition and the sum order, so they are not bitwise: a few row-ULP pairwise where
    a row's largest elements round the other way. A graph replay gives the eager bits. (8 ranks x 2..11
    requests: lanes/gen/wide128/vr_cl8x_bits.py.)"""
    swa, _ = cl8_tep8
    step = _SwaStep(n, False, torch.int32, seed=1500 + n)
    ids = list(range(n))
    groups = [ids] if n <= 8 else [ids[: (n + 1) // 2], ids[(n + 1) // 2 :]]
    ref, ref_pool = _w15_rows(swa, step, groups, dict(one2=True))
    exact, exact_pool = _w15_rows(swa, step, [ids], W15_EXACT)
    assert torch.equal(ref_pool.view(torch.uint8), exact_pool.view(torch.uint8))
    R = torch.cat([ref[b] for b in ids]).view(-1, SWA_HQ, SWA_D).double()
    E = torch.cat([exact[b] for b in ids]).view(-1, SWA_HQ, SWA_D).double()
    assert bool(torch.isfinite(E).all())
    X = _w15_reference(step, ids, exact_pool)
    rms = X.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(2.0**-126)
    ulp = torch.exp2(torch.floor(torch.log2(rms)) - 7)
    err_r, err_e = (R - X).abs() / ulp, (E - X).abs() / ulp
    r_max, r_mean = float(err_r.max()), float(err_r.mean())
    e_max, e_mean = float(err_e.max()), float(err_e.mean())
    pair_max = float(((E - R).abs() / ulp).max())
    stats = (
        f"[W15] n={n} vs fp64, bf16 row-ULP: replaced max {r_max:.2f} mean {r_mean:.3f}, "
        f"exact max {e_max:.2f} mean {e_mean:.3f}, pairwise max {pair_max:.2f}"
    )
    print(stats, flush=True)
    # The max is one element's rounding: bound it multiplicatively or additively (4 row-ULP), whichever is looser.
    assert e_max <= max(1.25 * r_max, r_max + 4.0), stats
    assert abs(e_mean - r_mean) <= 0.02, stats
    # graph replay of the exact build
    swa.configure(**W15_EXACT)
    x = step.inputs(ids)
    pool = step.pool.clone()
    eager = _swa_call(swa, step, x, pool).clone()
    pool.copy_(step.pool)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = _swa_call(swa, step, x, pool)
    pool.copy_(step.pool)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(got, eager)
