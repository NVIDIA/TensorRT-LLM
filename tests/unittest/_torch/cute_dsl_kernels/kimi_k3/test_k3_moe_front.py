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
"""trtllm::k3_moe_front and K3MoeLayer.front (the Kimi K3 MoE front: sharded head GEMV, head all-gather, top-16
routing, MXFP8 latent, shared gate_up + SiTU; then k3_moe on its grid), one process per GPU over the TP group of this
run, at every M in 1..8, over one K3MoeHeadWorkspace. The head is sharded over the group (TP W: 3584 / W latent +
896 / W router rows and 2 x 6144 / W shared rows per rank; W = 4 on one GB200 tray, the model's TP16 shapes with 16
processes); the routed experts are one rank of experts TP4 x EP4 (224 local experts, intermediate 768), as in the TP16
deployment.
  front : against the unfused chain (the head GEMV in fp32 torch -> the gather ->
          trtllm::kimi_k3_noaux_tc_mxfp8_quant; shared: cuBLAS gate_up -> trtllm::situ_and_mul): top-16 ids per
          token (a mismatch only at a reference
          16th / 17th key margin below 1e-4: the split-K head sums in another order), routing weights, MXFP8 codes and
          scales (> 99.9 % equal, dequantized within one block-scale unit), the shared activation within 2e-2; the same
          routing and latent bits on every rank; the head buffers empty and the buffer index flipped after each call;
  fused : y against the TRTLLM-Gen W4A8_MXFP4_MXFP8 runner on the front's own routing and MXFP8 latent (op-catalog
          gates), the shared activation the front's bits, k3_moe's scratch re-armed;
  head_flags : fused with the ready-word handoff (k3_moe built with head_flags) across the head epoch's int32 wrap:
          no word a call polls already holds the value it waits for, the plain call's bits, every ready word left
          at the next call's epoch (check_head_flags);
  publish_order : the handoff with k3_moe's epoch advance racing the front's epoch read: rank 1 routed nothing, k3_moe
          on half the SMs, each quantization CTA held before its flags read until the epoch moves; every ready word
          left at the next call's epoch (check_publish_order);
front and fused each with run-to-run identical bits and each M's rows bit-identical to the same rows of the 8-token
call (the fused y within one bf16 ulp: k3_moe's slice FC2 groups a token's expert terms by the step's group count).
The buffer and scratch checks run between barriers (peers write this rank's buffers in their next call); a failing
rank's rows are printed after rank 0's.

Run under pytest (a pool of 4 MPI workers) or directly, one process per GPU:
  srun -N1 -n4 --mpi=pmix python3 test_k3_moe_front.py [front fused head_flags publish_order]
"""

import importlib.util
import math
import os
import pickle
import shutil
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

try:
    import cloudpickle
    from mpi4py import MPI
except ImportError:  # the test is skipped below
    cloudpickle = MPI = None

if cloudpickle is not None:
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
HIDDEN, LATENT, EXPERTS, TOP_K, SV = 7168, 3584, 896, 16, 32
SHARED_INTER = 6144  # two shared experts of 3072
GATE_CAP, LINEAR_CAP = 4.0, 25.0
RSF = 2.827
I_TP, E_LOCAL, MOE_TP = 768, 224, 4  # one rank of the routed experts' TP4 x EP4
EMPTY = -(2**31)
ULP = 2.0**-8
M_ALL = list(range(1, 9))


def _supported() -> bool:
    if MPI is None or not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        return False
    return torch.cuda.get_device_capability() == (10, 0)


pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not _supported(), reason=f"needs {WORLD} sm_100 GPUs with MNNVL and mpi4py"),
]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _rand_mxfp4(rows, k, gen):
    codes = torch.randint(0, 16, (rows, k), dtype=torch.uint8, device="cuda", generator=gen)
    base = 127 + round(0.5 * math.log2(0.01057 / k))
    exps = torch.randint(
        base, base + 6, (rows, k // SV), dtype=torch.uint8, device="cuda", generator=gen
    )
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous(), exps


def _experts(seed):
    """224 random MXFP4 experts through TRT-LLM's TRTLLM-Gen loader (this rank's buffers)."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod

    i_full = I_TP * MOE_TP
    method = W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod()
    module = SimpleNamespace(tp_size=MOE_TP, tp_rank=1, scaling_vector_size=SV, intermediate_size=i_full,
                             intermediate_size_per_partition=I_TP, hidden_size=LATENT)  # fmt: skip
    kw = dict(dtype=torch.uint8, device="cuda")
    proc = dict(
        w31=torch.empty(E_LOCAL, 2 * I_TP, LATENT // 2, **kw),
        w31s=torch.empty(E_LOCAL, 2 * I_TP, LATENT // SV, **kw),
        w2=torch.empty(E_LOCAL, LATENT, I_TP // 2, **kw),
        w2s=torch.empty(E_LOCAL, LATENT, I_TP // SV, **kw),
    )
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for e in range(E_LOCAL):
        w1, w1s = _rand_mxfp4(i_full, LATENT, gen)
        w3, w3s = _rand_mxfp4(i_full, LATENT, gen)
        w2, w2s = _rand_mxfp4(LATENT, i_full, gen)
        method.load_expert_w3_w1_weight(module, w1, w3, proc["w31"][e])
        method.load_expert_w2_weight(module, w2, proc["w2"][e])
        method.load_expert_w3_w1_weight_scale_mxfp4(module, w1s, w3s, proc["w31s"][e])
        method.load_expert_w2_weight_scale_mxfp4(module, w2s, proc["w2s"][e])
    torch.cuda.synchronize()
    return proc


def _context(with_experts):
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpus)
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import front_op
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op as moe_op
    from tensorrt_llm._torch.modules import situ  # noqa: F401  (registers trtllm::situ_and_mul)
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=gpus, tp_size=world)
    wl, we, inter = LATENT // world, EXPERTS // world, SHARED_INTER // world
    gen = torch.Generator(device="cuda").manual_seed(11)  # the same on every rank
    bias = (torch.randn(EXPERTS, device="cuda", generator=gen) * 0.05).float()
    x8 = torch.randn(8, HIDDEN, device="cuda", generator=gen).bfloat16()
    wgen = torch.Generator(device="cuda").manual_seed(111 + rank)
    head = (torch.randn(wl + we, HIDDEN, device="cuda", generator=wgen) * 0.02).bfloat16()
    head[wl:] *= 8.0  # router rows: logits of a few units
    gate_up = (torch.randn(2 * inter, HIDDEN, device="cuda", generator=wgen) * 0.02).bfloat16()
    assert front_op.weight_supported(
        world, inter, HIDDEN, torch.device("cuda", torch.cuda.current_device())
    )
    # Fabric handles within one tray, as across trays.
    ws = moe_op.K3MoeHeadWorkspace.create(mapping, fabric_handle=True)
    ctx = SimpleNamespace(
        comm=comm, rank=rank, world=world, wl=wl, we=we, inter=inter, bias=bias, x8=x8, head=head, gate_up=gate_up,
        front=front_op.front_weight(head, gate_up), ws=ws, ag=(ws.uc, ws.mc, ws.flags, ws.rank),
        offset=(rank % 4) * E_LOCAL, experts=_experts(20260928 + rank) if with_experts else None, layer=None,
    )  # fmt: skip
    if with_experts:
        ctx.layer = _layer(ctx)
    return ctx


def _layer(ctx, head_flags=False, config=None):
    """This rank's experts as a layer of a new K3MoeState: the plain build, or the head_flags build."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import op as moe_op

    p = ctx.experts
    state = moe_op.K3MoeState(torch.device("cuda", torch.cuda.current_device()), I_TP, E_LOCAL, head_flags=head_flags,
                              config=config)  # fmt: skip
    return state.layer(p["w31"], p["w31s"], p["w2"], p["w2s"])


def _all_ranks(ctx, good) -> bool:
    return all(ctx.comm.allgather(bool(good)))


def _quiet_check(ctx, fn):
    """fn() with every rank's kernels done before it and no rank's next collective call started until every rank
    has run it: the checks read this rank's all-gather buffers and scratch, which peers write."""
    ctx.comm.Barrier()
    value = fn()
    ctx.comm.Barrier()
    return value


def _front(ctx, x):
    return torch.ops.trtllm.k3_moe_front(x, ctx.front, ctx.bias, RSF, ctx.inter, GATE_CAP, LINEAR_CAP, *ctx.ag,
                                         ctx.world)  # fmt: skip


def _reference(ctx, x):
    """The unfused chain: this rank's head rows in fp32 (torch), every rank's gathered (latent columns rounded to
    bf16), the fused C++ routing + MXFP8 quantization; the shared expert's gate_up (cuBLAS) and SiTU-and-mul."""
    head = x.float() @ ctx.head.float().t()
    parts = [torch.from_numpy(a).cuda() for a in ctx.comm.allgather(head.cpu().numpy())]
    latent = torch.cat([p[:, : ctx.wl] for p in parts], dim=1).bfloat16().contiguous()
    logits = torch.cat([p[:, ctx.wl :] for p in parts], dim=1).contiguous()
    ids, w, q, s = torch.ops.trtllm.kimi_k3_noaux_tc_mxfp8_quant(logits, ctx.bias, latent, RSF)
    shared = torch.ops.trtllm.situ_and_mul(F.linear(x, ctx.gate_up), GATE_CAP, LINEAR_CAP)
    key = torch.sigmoid(logits) + ctx.bias
    top = key.sort(dim=1, descending=True).values
    return ids, w, q, s, shared, top[:, TOP_K - 1] - top[:, TOP_K]


def _dequant(q, s):
    scale = torch.pow(2.0, s.float() - 127.0).repeat_interleave(SV, dim=1)
    return q.float() * scale, scale


def _buffers_empty(ctx) -> bool:
    return bool((ctx.ws.uc == EMPTY).all()) and int(ctx.ws.flags[1].item()) == 0


def check_front(ctx):
    results = []
    out8 = _front(ctx, ctx.x8)
    for m in M_ALL:
        x = ctx.x8[:m].contiguous()
        flag0 = int(ctx.ws.flags[0].item())
        ids, w, q, s, shared = out = _front(ctx, x)
        torch.cuda.synchronize()
        flag1 = int(ctx.ws.flags[0].item())
        empty = _quiet_check(ctx, lambda: _buffers_empty(ctx))
        r_ids, r_w, r_q, r_s, r_shared, margin = _reference(ctx, x)
        # The selected experts per token (their order inside the top 16 may differ at near-equal keys) and each
        # selected expert's weight.
        sets_equal = (ids.sort(dim=1).values == r_ids.sort(dim=1).values).all(dim=1)
        near_tie = margin < 1e-4
        dense = torch.zeros(m, EXPERTS, device="cuda").scatter_(1, ids.long(), w.float())
        r_dense = torch.zeros(m, EXPERTS, device="cuda").scatter_(1, r_ids.long(), r_w.float())
        same_tok = sets_equal.nonzero().flatten()
        dq, scale = _dequant(q, s)
        rdq, rscale = _dequant(r_q, r_s)
        again = [_front(ctx, x) for _ in range(2)]
        gathered = ctx.comm.allgather([_bits(t).cpu() for t in (ids, w, q, s)])
        row = dict(
            op="k3_moe_front", case=f"tp{ctx.world}", M=m, expert_sets_equal=f"{int(sets_equal.sum())}/{m}",
            order_equal=f"{int((ids == r_ids).all(dim=1).sum())}/{m}",
            mismatch_not_near_tie=int((~sets_equal & ~near_tie).sum()),
            weight_max_err=(dense[same_tok] - r_dense[same_tok]).abs().max().item() if same_tok.numel() else 0.0,
            codes_equal=(_bits(q) == _bits(r_q)).float().mean().item(), scales_equal=(s == r_s).float().mean().item(),
            latent_err_scale_units=((dq - rdq).abs() / torch.maximum(scale, rscale)).max().item(),
            shared_rel=((shared.float() - r_shared.float()).abs().max() / r_shared.float().abs().max()).item(),
            det=all(all(_same(a, b) for a, b in zip(r, out)) for r in again),
            rows_as_m8=all(_same(a, b[:m]) for a, b in zip(out, out8)),
            ranks_agree=all(all(torch.equal(a, b) for a, b in zip(g, gathered[0])) for g in gathered),
            buffers_empty=empty and _quiet_check(ctx, lambda: _buffers_empty(ctx)), flag_flipped=flag1 != flag0,
        )  # fmt: skip
        good = (row["mismatch_not_near_tie"] == 0 and row["weight_max_err"] <= 0.01 * RSF and row["codes_equal"] > 0.999
                and row["scales_equal"] > 0.999 and row["latent_err_scale_units"] <= 1.0 and row["shared_rel"] <= 2e-2
                and row["det"] and row["rows_as_m8"] and row["ranks_agree"] and row["buffers_empty"]
                and row["flag_flipped"])  # fmt: skip
        row["rank"], row["good"] = ctx.rank, bool(good)
        row["ok"] = _all_ranks(ctx, good)
        results.append(row)
    return results


def _fused(ctx, x, layer=None, bias=None):
    """K3MoeLayer.front on ``layer`` (default: the plain build's ``ctx.layer``)."""
    layer = layer or ctx.layer
    return layer.front(x, ctx.front, ctx.bias if bias is None else bias, ctx.offset, RSF, ctx.inter, GATE_CAP,
                       LINEAR_CAP, ctx.ws)  # fmt: skip


def _runner(ctx, ids, w, q, s):
    """The TRTLLM-Gen W4A8_MXFP4_MXFP8 MoE, pre-routed (the model's base path for these experts)."""
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    p = ctx.experts
    alpha = torch.full((E_LOCAL,), GATE_CAP, dtype=torch.float32, device="cuda")
    beta = torch.full((E_LOCAL,), LINEAR_CAP, dtype=torch.float32, device="cuda")
    return torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner(
        None, None, q, s.view(-1), p["w31"], p["w31s"], None, alpha, beta, None, p["w2"], p["w2s"], None, EXPERTS,
        TOP_K, 1, 1, I_TP, LATENT, I_TP, ctx.offset, E_LOCAL, 1.0, int(RoutingMethodType.DeepSeekV3),
        int(ActType_TrtllmGen.SiTu), topk_weights=w, topk_ids=ids)  # fmt: skip


def _compare(y, ref):
    o, r = y.float(), ref.float()
    row = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    elt = ((o - r).abs() / row).max().item() / ULP
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-12)).item() / ULP
    return elt, rms, bool(torch.isfinite(o).all()) and elt <= 8.0 and rms <= 4.0


def _scratch_rearmed(ctx):
    """The plain build's intermediate slab armed again and its layer's counters zero."""
    st = ctx.layer.state
    mod = st.mod
    cs = st.cs.view(mod.G_CAP, 8, mod.K2_TILES, mod.SFB_GROUP_BYTES)
    armed = bool((st.c == -128).all()) and bool((cs[..., :4] == -1).all())
    return armed and bool((ctx.layer.counters == 0).all())


def _max_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    def ordered(x):
        i = x.contiguous().view(torch.int16).int()
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return int((ordered(a) - ordered(b)).abs().max().item()) if a.numel() else 0


def check_fused(ctx):
    """y's rows against the 8-token call within one bf16 ulp: k3_moe's slice FC2 groups a token's expert terms by the
    step's group count (bit-identity reported); the shared activation (per token) bit for bit."""
    results = []
    y8, sh8 = _fused(ctx, ctx.x8)
    for m in M_ALL:
        x = ctx.x8[:m].contiguous()
        y, shared = _fused(ctx, x)
        torch.cuda.synchronize()
        rearmed = _quiet_check(ctx, lambda: _scratch_rearmed(ctx) and _buffers_empty(ctx))
        ids, w, q, s, f_shared = _front(ctx, x)
        y_stock = _runner(ctx, ids, w, q, s)
        local = int(((ids >= ctx.offset) & (ids < ctx.offset + E_LOCAL)).sum())
        again = [_fused(ctx, x) for _ in range(2)]
        if local:
            elt, rms, close = _compare(y, y_stock)
        else:
            elt, rms, close = 0.0, 0.0, bool((y.float() == 0).all())
        row = dict(
            op="k3_fused_moe_front", case=f"tp{ctx.world}", M=m, local_pairs=local, vs_stock_elt_ulp=elt,
            vs_stock_rms_ulp=rms, shared_eq_front=_same(shared, f_shared),
            det=all(_same(a, y) and _same(b, shared) for a, b in again),
            rows_as_m8=_same(y, y8[:m]) and _same(shared, sh8[:m]), max_ulp_vs_m8=_max_ulp(y, y8[:m]),
            shared_rows_as_m8=_same(shared, sh8[:m]), scratch_rearmed=rearmed,
        )  # fmt: skip
        good = (close and row["shared_eq_front"] and row["det"] and row["max_ulp_vs_m8"] <= 1
                and row["shared_rows_as_m8"] and rearmed)  # fmt: skip
        row["rank"], row["good"] = ctx.rank, bool(good)
        row["ok"] = _all_ranks(ctx, good)
        results.append(row)
    return results


def _i32(v: int) -> int:
    return (v + 2**31) % 2**32 - 2**31


def check_head_flags(ctx):
    """K3MoeLayer.front with the ready-word handoff (``ag_ready``: k3_moe built with head_flags acquires the front's
    ready words, ready[t] / ready[8 + t] = the head epoch flags[2] + 1 for token t, instead of waiting for its grid)
    across the epoch's int32 wrap. From a new workspace's state (epoch 0, ready words 0), two calls at M 1, then the
    epoch preset to -2, then calls at M 1, 8, 3, 8: the M 8 call at epoch -1 waits for 0, the value of the words that
    no call has published. Per call: no word the call polls already holds its epoch + 1 (such a word would let k3_moe
    read the routing before the front writes it; the call is then not run); y and the shared activation the bits of
    the plain call; afterwards the epoch and every ready word hold the next call's epoch, the head buffers empty."""
    flags, ready = ctx.ws.flags, ctx.ws.ready
    flag_layer = _layer(ctx, head_flags=True)
    plain = {m: _fused(ctx, ctx.x8[:m].contiguous()) for m in (1, 3, 8)}
    torch.cuda.synchronize()

    def set_epoch(epoch):
        flags[2] = epoch

    _quiet_check(ctx, lambda: (ready.zero_(), set_epoch(0)))
    results = []
    for m in (1, 1, None, 1, 8, 3, 8):
        if m is None:  # two calls before the epoch reaches 0
            _quiet_check(ctx, lambda: set_epoch(-2))
            continue
        x = ctx.x8[:m].contiguous()
        epoch, words = _quiet_check(ctx, lambda: (int(flags[2].item()), ready[:16].tolist()))
        want = _i32(epoch + 1)
        pre_matched = [i for i in [*range(m), *range(8, 8 + m)] if words[i] == want]
        row = dict(op="k3_fused_moe_front_ready", case=f"tp{ctx.world}", M=m, epoch=epoch,
                   pre_matched=pre_matched)  # fmt: skip
        if not _all_ranks(ctx, not pre_matched):
            row["rank"], row["good"], row["ok"] = ctx.rank, False, False
            results.append(row)
            break
        y, shared = _fused(ctx, x, flag_layer)
        torch.cuda.synchronize()
        after, words = _quiet_check(ctx, lambda: (int(flags[2].item()), ready[:16].tolist()))
        row.update(
            epoch_advanced=after == want, ready_rearmed=all(w == want for w in words),
            y_as_plain=_same(y, plain[m][0]), shared_as_plain=_same(shared, plain[m][1]),
            buffers_empty=_quiet_check(ctx, lambda: _buffers_empty(ctx)),
        )  # fmt: skip
        good = all(row[k] for k in ("epoch_advanced", "ready_rearmed", "y_as_plain", "shared_as_plain",
                                    "buffers_empty"))  # fmt: skip
        row["rank"], row["good"] = ctx.rank, bool(good)
        row["ok"] = _all_ranks(ctx, good)
        results.append(row)
    return results


# The role CTAs' read of the head workspace's flags (buffer index, epoch) in k3_moe_front.py, and what
# check_publish_order inserts before it: each quantization CTA's thread 0 waits until the epoch moves, or ~20 ms.
_FLAGS_READ = (
    "            if tx == 0:\n"
    "                s_flags.store(flags.load(idx=0, is_volatile=True), idx=0)\n"
    "                s_flags.store(flags.load(idx=2, is_volatile=True), idx=1)\n"
)
_QUANT_HOLD = (
    "            if role >= num_tokens:\n"
    "                if tx == 0:\n"
    "                    hold_epoch = flags.load(idx=2, is_volatile=True)\n"
    "                    hold_polls = cutlass.Int32(0)\n"
    "                    while (flags.load(idx=2, is_volatile=True) == hold_epoch) & (\n"
    "                        hold_polls < cutlass.Int32(10000)\n"
    "                    ):\n"
    "                        prims.nanosleep(2000)\n"
    "                        hold_polls = hold_polls + cutlass.Int32(1)\n"
)


def _held_front(tmp_dir):
    """k3_moe_front.py with _QUANT_HOLD before the role CTAs' flags read, loaded from a file as a sibling module."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import front_op

    with open(os.path.join(os.path.dirname(front_op.__file__), "k3_moe_front.py")) as f:
        src = f.read()
    assert src.count(_FLAGS_READ) == 1, "k3_moe_front.py's role CTAs read the flags elsewhere now"
    path = os.path.join(tmp_dir, "k3_moe_front_quant_hold.py")
    with open(path, "w") as f:
        f.write(src.replace(_FLAGS_READ, _QUANT_HOLD + _FLAGS_READ))
    name = front_op.__name__.rsplit(".", 1)[0] + ".k3_moe_front_quant_hold"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def check_publish_order(ctx, num_ctas=None):
    """The ready-word handoff when k3_moe's epoch advance races the front's epoch read. k3_moe (head_flags) launches
    once every front CTA has triggered and advances the head epoch at its last claim, which on a rank without a routed
    expert waits for no ready word of the quantization CTAs, only for every k3_moe CTA to start. Here the routing bias
    keeps every token off rank 1's experts, k3_moe runs ``num_ctas`` CTAs (default half the SMs, so that all of them
    start beside the front's role CTAs; 0: one per SM, as the op builds it), and the front is k3_moe_front.py with each
    quantization CTA held before its flags read until the epoch moves (_QUANT_HOLD). A role CTA that triggers before
    reading the epoch then publishes the advanced epoch + 1, which the next call's poll takes for its own. After a
    warm-up call, from epoch 0, calls at M 1, 8, 3; per call: no word the call polls already holds its epoch + 1 (else
    the call is not run), afterwards the epoch and every ready word at the next call's epoch, the head buffers empty,
    rank 1's y all zeros."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import front_op

    flags, ready = ctx.ws.flags, ctx.ws.ready
    idle = 1
    bias = ctx.bias.clone()
    bias[idle * E_LOCAL : (idle + 1) * E_LOCAL] = -8.0  # below every other expert's selection key
    device = torch.device("cuda", torch.cuda.current_device())
    ctas = torch.cuda.get_device_properties(device).multi_processor_count
    if num_ctas != 0:
        ctas = num_ctas or ctas // 2
    saved_kernel, saved_compiled = front_op._kernel, dict(front_op._compiled)
    tmp_dir = tempfile.mkdtemp(prefix="k3_moe_front_")
    held_layer = _layer(ctx, head_flags=True, config={"num_ctas": ctas})

    def fused(x):
        return _fused(ctx, x, held_layer, bias)

    def set_epoch(epoch):
        flags[2] = epoch

    results = []
    try:
        held = _held_front(tmp_dir)
        front_op._kernel = lambda: held
        front_op._compiled.clear()
        # Compiles both kernels; the ranks then start each call within the hold.
        _quiet_check(ctx, lambda: (fused(ctx.x8[:1].contiguous()), torch.cuda.synchronize()))
        _quiet_check(ctx, lambda: (ready.zero_(), set_epoch(0)))
        for m in (1, 8, 3):
            x = ctx.x8[:m].contiguous()
            epoch, words = _quiet_check(ctx, lambda: (int(flags[2].item()), ready[:16].tolist()))
            want = _i32(epoch + 1)
            pre_matched = [i for i in [*range(m), *range(8, 8 + m)] if words[i] == want]
            row = dict(op="k3_fused_moe_front_ready_order", case=f"tp{ctx.world}", M=m, num_ctas=ctas, epoch=epoch,
                       pre_matched=pre_matched)  # fmt: skip
            if not _all_ranks(ctx, not pre_matched):
                row["rank"], row["good"], row["ok"] = ctx.rank, False, False
                results.append(row)
                break
            t0 = time.perf_counter()
            y, _ = fused(x)
            torch.cuda.synchronize()
            ms = (
                time.perf_counter() - t0
            ) * 1e3  # ~20 ms: the hold ran out, no epoch moved during it
            after, words = _quiet_check(ctx, lambda: (int(flags[2].item()), ready[:16].tolist()))
            row.update(
                epoch_advanced=after == want, off_epoch=[(i, w) for i, w in enumerate(words) if w != want],
                buffers_empty=_quiet_check(ctx, lambda: _buffers_empty(ctx)),
                y_zero=bool((y == 0).all()) if ctx.rank == idle else True,
                call_ms=[round(v, 2) for v in ctx.comm.allgather(ms)],
            )  # fmt: skip
            good = (
                row["epoch_advanced"]
                and not row["off_epoch"]
                and row["buffers_empty"]
                and row["y_zero"]
            )
            row["rank"], row["good"] = ctx.rank, bool(good)
            row["ok"] = _all_ranks(ctx, good)
            results.append(row)
    finally:
        front_op._kernel = saved_kernel
        front_op._compiled.clear()
        front_op._compiled.update(saved_compiled)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return results


CHECKS = {
    "front": check_front,
    "fused": check_fused,
    "head_flags": check_head_flags,
    "publish_order": check_publish_order,
}


def _run_checks(names):
    try:
        ctx = _context(with_experts=bool({"fused", "head_flags", "publish_order"} & set(names)))
        with torch.inference_mode():
            return [row for name in names for row in CHECKS[name](ctx)]
    except Exception:
        traceback.print_exc()
        raise


def _report(per_rank):
    """Rank 0's rows, then every other rank's rows that failed there."""
    rows = list(per_rank[0]) + [row for rows in per_rank[1:] for row in rows if not row["good"]]
    for row in rows:
        fields = " ".join(f"{k}={(f'{v:.3e}' if isinstance(v, float) else v)}" for k, v in row.items()
                          if k not in ("op", "case", "M"))  # fmt: skip
        print(f"OPCHECK op={row['op']} case={row['case']} M={row['M']} {fields}", flush=True)


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("check", list(CHECKS))
def test_k3_moe_front(mpi_pool_executor, check):
    per_rank = list(mpi_pool_executor.map(_run_checks, [[check]] * WORLD))
    _report(per_rank)
    assert all(row["ok"] for rows in per_rank for row in rows)


def main() -> int:
    names = sys.argv[1:] or list(CHECKS)
    rows = _run_checks(names)
    per_rank = MPI.COMM_WORLD.gather(rows, root=0)
    if MPI.COMM_WORLD.Get_rank() == 0:
        _report(per_rank)
        print("PASS" if all(r["ok"] for r in rows) else "FAIL", flush=True)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
