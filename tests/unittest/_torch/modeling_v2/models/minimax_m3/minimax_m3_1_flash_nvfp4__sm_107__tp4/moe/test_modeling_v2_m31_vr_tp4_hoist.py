# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4WC_HOIST on 4 sm_107 GPUs: ``mpirun -n 4 python -m pytest -q <this file>`` (one rank per GPU; every
test is collective, run the whole file on every rank). Skipped unless the MPI world is 4.

- the seams install on the TP MPI communicator and the block builds on MNNVL exchanges, including the post's build-time
  warm call on the MNNVL rx slots (S0b: the reference's MNNVL drop-in failed that call's rx-slot / prx-bucket check);
- the engine module is the default (B2LAG on) HYB_ENGINE, or tp4_hoist_block.HYB_ENGINE_B2LAG_OFF with
  TRTLLM_M31_DISABLE_TP4WC_B2LAG=1;
- one eager call: the post's residual_out against the fp64 all-reduce of the 4 ranks' o_proj partials + residual_mid,
  normed2 against the folded-weight RMSNorm of it, the published routes against an fp64 routing of it (flip rate);
- a CUDA graph of the call replays the eager bits; a 3 s late rank does not trap (hang cap default).
"""

import os
import time
from types import SimpleNamespace

import pytest
import torch

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
except ImportError:  # pragma: no cover
    COMM = None

FAM = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3"
H, KO, E, INTER, T, LAYER = (
    6144,
    2048,
    128,
    768,
    8,
    3,
)  # KO: the TP4 per-rank o_proj K (16 heads x 128)
WORLD = 4
pytestmark = pytest.mark.skipif(
    COMM is None
    or COMM.Get_size() != WORLD
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7),
    reason="4 MPI ranks on sm_107",
)
_S = {}


def _env():
    if "env" in _S:
        return _S["env"]
    import importlib

    from tensorrt_llm.mapping import Mapping

    rank = COMM.Get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    mapping = Mapping(world_size=WORLD, rank=rank, gpus_per_node=WORLD, tp_size=WORLD)
    mc = importlib.import_module(
        f"{FAM}.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.m3_mc_allreduce"
    )
    seams = importlib.import_module(
        f"{FAM}.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.hoist.seams"
    )
    env = seams.install(mapping, mc._tp_comm(mapping))
    _S.update(env=env, mapping=mapping, mc=mc, seams=seams, rank=rank)
    return env


def _hw():
    """The block package's router_hoist_weights (loaded by the seams install)."""
    import importlib

    _env()
    return importlib.import_module(
        f"{FAM}.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.hoist.tp4c1.kernels.moe.engine.router_hoist_weights"
    )


def _weights(seed):
    """Layer LAYER in this target's weight / scalar form, synthetic (e4m3 codes, block scales in range)."""
    hw = _hw()

    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"

    def e4m3(*shape, scale=1.0):
        return (scale * torch.randn(*shape, device=dev, generator=g)).to(torch.float8_e4m3fn)

    def u8(*shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, dtype=torch.uint8, device=dev, generator=g)

    def sw(n, k):
        return hw.swizzle_128x4(
            torch.full((n, k // 32), 120, dtype=torch.uint8, device=dev)
        ).contiguous()

    folded = (1.0 + 0.1 * torch.randn(H, device=dev, generator=g)).to(torch.bfloat16)
    W = lambda t: SimpleNamespace(data=t)  # noqa: E731
    w = {
        f"l{LAYER}_o_w": W(e4m3(H, KO)),
        f"l{LAYER}_o_s": W(sw(H, KO)),
        f"l{LAYER}_post_norm": W(folded),
        f"l{LAYER}_e_w31": W(u8(E, 2 * INTER, H // 2)),
        f"l{LAYER}_e_s31": W(u8(E, 2 * INTER, H // 16, lo=0x30, hi=0x40)),
        f"l{LAYER}_e_w2": W(u8(E, H, INTER // 2)),
        f"l{LAYER}_e_s2": W(u8(E, H, INTER // 16, lo=0x30, hi=0x40)),
        f"l{LAYER}_router": W((torch.randn(E, H, device=dev, generator=g) / H**0.5).float()),
        f"l{LAYER}_router_bias": W(torch.zeros(E, device=dev)),
        f"l{LAYER}_sh_gu_w": W(e4m3(2 * INTER, H).view(torch.uint8)),
        f"l{LAYER}_sh_gu_s": W(sw(2 * INTER, H)),
        f"l{LAYER}_sh_dn_w": W(e4m3(H, INTER).view(torch.uint8)),
        f"l{LAYER}_sh_dn_s": W(sw(H, INTER)),
    }
    ga = (1.0 / (2.93 * 1.3 * H**0.5)) * (0.8 + 0.4 * torch.rand(E, device=dev, generator=g))
    moe = SimpleNamespace(
        fc31_scale_c=ga.clone(),
        fc31_alpha=ga.clone(),
        act_alpha=torch.full((E,), 1.702, device=dev),
        act_beta=torch.ones(E, device=dev),
        act_clamp=torch.full((E,), 7.0, device=dev),
        fc2_alpha=(0.8 + 0.4 * torch.rand(E, device=dev, generator=g)) / (2.93 * 1.3 * INTER**0.5),
        fc31_input_scale=torch.ones(1, device=dev),
    )
    return w, moe


def _block():
    if "block" in _S:
        return _S["block"]
    env = _env()
    import importlib

    blockmod = importlib.import_module(
        f"{FAM}.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.hoist.tp4_hoist_block"
    )
    tri = importlib.import_module(
        f"{FAM}.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.triton_sparse_decode"
    )
    loc = importlib.import_module(f"{FAM}.kernels.m3ll.locality")
    w, moe = _weights(1234)  # same weights on every rank (each rank's TP slice, synthetic)
    mx = tri.MERGE_MXFP8.buffers(KO, torch.device("cuda", torch.cuda.current_device()))
    locmap = loc.LocalityMap.get(torch.cuda.current_device())
    assert locmap is not None  # the served DL = 1 path (a None map hid the broken locality import)
    block = blockmod.build(
        env, w, moe, LAYER, mx, locmap
    )  # make(): includes the post's rows-0 warm on MNNVL rx
    assert block.placed
    ws = _S["mc"].get_workspace(_S["mapping"], H)
    b8 = ws.for_tokens(T)
    f8 = b8.flags_for(8)
    eng = importlib.import_module(
        f"{FAM}.minimax_m3_1_flash_nvfp4__sm_107__tp4.ops.moe.m3_moe_engine"
    )
    block.arm(
        eng.ar_push_desc(
            b8.mc.data_ptr(), f8.data_ptr(), env.rank, env.world, b8.max_tokens, f8.device
        )
    )
    _S.update(block=block, w=w, moe=moe, mx=mx, b8=b8, blockmod=blockmod)
    return block


def _rearm():
    """No consumer lands set A in this test: every rank re-arms it (EMPTY), as make_workspace left it."""
    torch.cuda.synchronize()
    COMM.Barrier()
    _S["b8"].uc.fill_(_S["seams"].EMPTY_WORD)
    torch.cuda.synchronize()
    COMM.Barrier()


def _inputs(seed):
    hw = _hw()  # noqa: F841

    g = torch.Generator(device="cuda").manual_seed(seed * 7 + COMM.Get_rank())
    x = (torch.randn(T, KO, device="cuda", generator=g)).to(torch.bfloat16)
    q, s = torch.ops.trtllm.mxfp8_quantize(x, True)
    xq, xsf = _S["mx"]
    xq[:T].copy_(q.view(xq.dtype).view(T, KO))
    xsf.view(torch.uint8)[: s.numel()].copy_(s.view(torch.uint8).reshape(-1))
    g2 = torch.Generator(device="cuda").manual_seed(seed)  # residual_mid: the same on every rank
    _S["block"].residual_mid.copy_(
        torch.randn(T, H, device="cuda", generator=g2).to(torch.bfloat16)
    )
    return xq, xsf


def test_build_on_mnnvl_and_the_engine_is_hyb():
    block = _block()
    bm = _S["blockmod"]
    # B2LAG is default on: HYB_ENGINE; TRTLLM_M31_DISABLE_TP4WC_B2LAG=1 (seams.B2LAG_OFF) builds the shipped engine.
    assert block.engine_config == (
        bm.HYB_ENGINE_B2LAG_OFF if _S["seams"].B2LAG_OFF else bm.HYB_ENGINE
    )
    assert block.engine_config["DL"] == 1 and block.placed
    assert block.armed
    COMM.Barrier()


def test_engagement_line_says_dl_1_placed_1():
    block = _block()
    bm = _S["blockmod"]
    bm._ENGAGED.clear()
    line = bm.log_engaged(COMM.Get_rank(), block, _S["seams"].HANG_CAP_NS)
    assert line.startswith("[M31_TP4WC_HOIST] engaged:") and " dl 1 placed 1 " in line, line
    COMM.Barrier()


def _to_wire(t):
    """A tensor as (uint8 numpy bytes, shape, dtype name) for an mpi4py object collective. mpi4py pickles its
    payload, and a pickled torch tensor (here MXFP8 float8_e4m3fn) does not unpickle in this torch build
    ("UntypedStorage has no attribute dtype"); numpy arrays pickle portably."""
    t = t.detach().contiguous().cpu()
    return t.view(torch.uint8).numpy(), tuple(t.shape), str(t.dtype).rsplit(".", 1)[-1]


def _from_wire(w):
    """Inverse of _to_wire: the same bytes, viewed back as the tensor's dtype and shape (bit-exact)."""
    a, shape, dtype = w
    return torch.from_numpy(a.copy()).view(getattr(torch, dtype)).reshape(shape)


def _ref(block, xq, xsf):
    """fp64: residual_out = residual_mid + sum over ranks of o_proj_r(x_r); normed with the folded weight; routes."""
    hw = _hw()

    w = _S["w"]
    ow, osf = w[f"l{LAYER}_o_w"].data, w[f"l{LAYER}_o_s"].data
    W = hw.mxfp8_dequant(ow.cpu(), hw.unswizzle_128x4(osf.cpu(), H, KO // 32))
    xs = [
        (_from_wire(q), _from_wire(s)) for q, s in COMM.allgather((_to_wire(xq[:T]), _to_wire(xsf)))
    ]
    parts = [hw.mxfp8_dequant(q, hw.unswizzle_128x4(s, T, KO // 32)) @ W.T for q, s in xs]
    h = block.residual_mid.double().cpu() + sum(parts)
    rstd = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    folded = w[f"l{LAYER}_post_norm"].data.double().cpu()
    normed = h * rstd * folded
    logits = normed @ w[f"l{LAYER}_router"].data.double().cpu().T
    score = torch.sigmoid(logits) + w[f"l{LAYER}_router_bias"].data.double().cpu()
    return h, normed, score.topk(4, dim=-1).indices.sort(-1).values


def test_eager_call_against_fp64():
    block = _block()
    xq, xsf = _inputs(7)
    block(xq, xsf)
    torch.cuda.synchronize()
    h, normed, ids_ref = _ref(block, xq, xsf)
    res = block.residual_out.double().cpu()
    nrm = block.blk.normed2.double().cpu()
    rel_res = float((res - h).norm() / h.norm())
    rel_nrm = float((nrm - normed).norm() / normed.norm())
    rb = block.blk.h9_hoist.rb
    hw = _hw()

    words = rb.route.view(rb.repl, rb.maxt, hw.ROUTE_LINE_U64)[0, :T, : hw.TOP_K].cpu()
    _, ids, _ = hw.unpack_route(words)
    got = ids.sort(-1).values
    flips = int((got != ids_ref).any(-1).sum())
    print(
        f"[tp4 hoist UT] rank {COMM.Get_rank()} rel residual {rel_res:.3e} rel normed {rel_nrm:.3e} "
        f"route rows flipped {flips}/{T}"
    )
    _rearm()
    assert rel_res < 1e-2 and rel_nrm < 2e-2
    assert flips <= 1  # ~0.8 % expected (by design); one row of 8 is the band


def test_graph_replay_is_the_eager_bits():
    block = _block()
    xq, xsf = _inputs(11)
    block(xq, xsf)
    torch.cuda.synchronize()
    want = (block.residual_out.clone(), block.blk.normed2.clone(), block.ffn_out.clone())
    _rearm()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(graph, stream=s):
        block(xq, xsf)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(32):
        _rearm()
        graph.replay()
        torch.cuda.synchronize()
        got = (block.residual_out, block.blk.normed2, block.ffn_out)
        assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(got, want))
    _rearm()


def test_late_rank_does_not_trap():
    """Rank 2 starts the call 3 s late; with the default hang cap (the reference 2 s deadline would trap) every rank
    completes with the on-time bits."""
    block = _block()
    xq, xsf = _inputs(13)
    block(xq, xsf)
    torch.cuda.synchronize()
    want = block.residual_out.clone()
    _rearm()
    if COMM.Get_rank() == 2:
        time.sleep(3.0)
    block(xq, xsf)
    torch.cuda.synchronize()
    assert torch.equal(block.residual_out.view(torch.int16), want.view(torch.int16))
    _rearm()


def test_kill_switch_env_reaches_the_target():
    import subprocess
    import sys

    code = (
        f"import importlib,sys; m = importlib.import_module('{FAM}.minimax_m3_1_flash_nvfp4__sm_107__tp4.modeling'); "
        "sys.exit(0 if m.TP4WC_HOIST is False else 1)"
    )
    for env in ("TRTLLM_M31_DISABLE_TP4WC_HOIST", "TRTLLM_M31_DISABLE_TP4_WHOLECHAIN"):
        if COMM.Get_rank() == 0:
            # a fresh, non-MPI interpreter: a child that inherits this mpirun rank's OMPI / PMIx env blocks in MPI init
            # (build_cut26_v1: TimeoutExpired after 600 s)
            base = {
                k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMIX_", "PMI_"))
            }
            r = subprocess.run(
                [sys.executable, "-c", code],
                env={**base, env: "1", "TLLM_DISABLE_MPI": "1"},
                timeout=600,
            )
            assert r.returncode == 0, env
    COMM.Barrier()
