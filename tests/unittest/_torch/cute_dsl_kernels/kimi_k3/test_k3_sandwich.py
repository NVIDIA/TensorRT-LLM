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
"""trtllm::k3_sandwich_oproj / k3_sandwich_tail / k3_sandwich_plain (projection + MNNVL all-reduce + residual update in
one kernel, M <= 8) at the Kimi K3 TP16 per-rank shapes, one process per GPU over the TP group of this run (4 on one
GB200 tray; the per-rank shapes do not depend on the group size, the reduction is 4-way instead of 16-way), at every M
in 1..8:
  oproj  against o_proj (cuBLAS, rows as in an 8-row call) -> MNNVLAllReduce.allreduce_attn_res_rmsnorm, bit for bit,
         0 / 1 / 3 / 8 snapshots, with and without the prefix sum;
  tail   against the tail in torch (fp32 accumulators, the latent RMS on the latent one) -> allreduce_attn_res_rmsnorm
         (fp32 tolerance: another summation order), and with the DSpark capture tap (the pre-norm mixture against
         trtllm::attn_res_fwd) and updated_out (a snapshot bank row);
  plain  against k3_ctm_gemv -> the MNNVL one-shot RESIDUAL_RMS_NORM all-reduce, bit for bit (drafter o_proj, K 384),
         and the SwiGLU form against k3_ctm_gemv_swiglu split 2 -> the same all-reduce (drafter down, K 896);
each with run-to-run identical bits, each M's rows bit-identical to the same rows of the 8-row call, every rank's
result changed by one rank's perturbed input; then CUDA graphs captured per M and replayed in mixed order with refilled
inputs, against eager calls; and the call counters (the buffer's, the folded latent all-reduce's) across the int32
wrap.

Run under pytest (a pool of 4 MPI workers) or directly, one process per GPU:
  srun -N1 -n4 --mpi=pmix python3 test_k3_sandwich.py [oproj tail plain swiglu replay wrap fold_wrap]
"""

import os
import pickle
import sys
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


def _trtllm():
    """``torch.ops.trtllm``, named only in a nested function: the pool gets this module's functions by value, and
    cloudpickle cannot pickle a function whose own code names ``torch.ops`` (it adds ``sys.modules["torch.ops"]`` to
    the function's state)."""

    def namespace():
        return torch.ops.trtllm

    return namespace()


WORLD = 4
H, K_O, LATENT, WIDTH, PAD, ACT = 7168, 768, 3584, 224, 256, 384
PLAIN_K, DOWN_K = 384, 896
EPS, LAT_EPS = 1e-5, 1e-6
M_ALL = list(range(1, 9))
SNAPSHOTS = (0, 1, 3, 8)
REPLAY_ORDER = (8, 1, 5, 2, 7, 3, 6, 4, 8, 1, 3, 8)


def _supported() -> bool:
    if MPI is None or not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        return False
    return torch.cuda.get_device_capability()[0] == 10


pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not _supported(), reason=f"needs {WORLD} SM100 GPUs with MNNVL and mpi4py"),
]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def _rand(shape, seed, scale=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, generator=gen, device="cuda") * scale).bfloat16().contiguous()


def _norm_w(seed):
    return (1.0 + _rand((H,), seed, 0.1).float()).bfloat16()


def _nan(*shape):
    return torch.full(shape, float("nan"), dtype=torch.bfloat16, device="cuda")


def _context():
    """This process's rank, the MNNVL all-reduce of the TP group (the unfused path) and the sandwiches' workspace.
    One process per GPU, ranks filling the nodes in order (gpus_per_node = the node's GPU count, so a rank's
    local_rank is its device on every node). Within one tray the multicast buffers use fabric handles as across
    trays."""
    os.environ.setdefault("TRTLLM_FORCE_MNNVL_AR", "1")
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpus)
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv import op as _ctm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_sandwich import op as sw_op
    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=gpus, tp_size=world)
    return SimpleNamespace(comm=comm, rank=rank, world=world, mapping=mapping,
                           mnnvl=MNNVLAllReduce(mapping, torch.bfloat16),
                           ws=sw_op.K3SandwichWorkspace.create(mapping, fabric_handle=True))  # fmt: skip


def _all_ranks(ctx, good) -> bool:
    return all(ctx.comm.allgather(bool(good)))


def _oproj(ctx, core, w, prefix, block, res_w, rms_w, out_w):
    ws = ctx.ws
    return _trtllm().k3_sandwich_oproj(core, w, prefix, block, res_w, rms_w, out_w, EPS, EPS, ws.uc, ws.mc,
                                       ws.flags, ws.rank)  # fmt: skip


def _tail(ctx, latent, act, w, lo, prefix, block, res_w, rms_w, out_w, **extra):
    ws = ctx.ws
    return _trtllm().k3_sandwich_tail(latent, act, w, lo, LAT_EPS, prefix, block, res_w, rms_w, out_w, EPS,
                                      EPS, ws.uc, ws.mc, ws.flags, ws.rank, **extra)  # fmt: skip


def _plain(ctx, x, w, residual, norm_w, swiglu=False):
    ws = ctx.ws
    return _trtllm().k3_sandwich_plain(x, w, residual, norm_w, EPS, ws.uc, ws.mc, ws.flags, ws.rank,
                                       swiglu=swiglu)  # fmt: skip


def _attn_res_ar(ctx, partial, prefix, block, res_w, rms_w, out_w):
    """The unfused post-projection step: the MNNVL one-shot all-reduce with the attention-residual epilogue."""
    return ctx.mnnvl.allreduce_attn_res_rmsnorm(
        partial, prefix, block, res_w, rms_w, out_w, EPS, EPS
    )


def _tail_partial(latent, act, w, lo):
    """This rank's row-parallel tail partial with the kernel's arithmetic: fp32 accumulators of the latent slice and of
    the activation, the latent one scaled by the RMS of the whole latent row, one bf16 rounding."""
    lat = latent.float()
    scale = torch.rsqrt(lat.pow(2).mean(dim=1, keepdim=True) + LAT_EPS)
    acc_lat = lat[:, lo : lo + WIDTH] @ w[:, :WIDTH].float().t()
    return (acc_lat * scale + act.float() @ w[:, PAD:].float().t()).bfloat16()


def _residual_rms_ar(ctx, partial, residual, norm_w):
    """The unfused drafter step: the MNNVL all-reduce with the residual add + RMSNorm fusion, sent one-shot (the
    sandwich reproduces the one-shot kernel's order; above 8 ranks two-shot sums the ranks in another order)."""
    from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams

    params = AllReduceParams(fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM, residual=residual, norm_weight=norm_w,
                             eps=EPS)  # fmt: skip
    out = ctx.mnnvl(
        partial, params, one_shot_max_bytes=partial.numel() * ctx.world * partial.element_size()
    )
    return out[0], out[1]


def _row8(fn, x, m):
    """fn on x padded to 8 rows, first m rows: the rows an 8-row call computes (cuBLAS picks its kernel by M)."""
    pad = torch.zeros(8, x.shape[1], dtype=x.dtype, device=x.device)
    pad[:m] = x
    return fn(pad)[:m].contiguous()


def _oproj_inputs(ctx, snapshots, seed):
    r = ctx.rank
    core8 = _rand((8, K_O), seed + 1000 * r + 1)
    w = _rand((H, K_O), seed + 1000 * r + 2, 0.03)
    prefix8 = _rand((8, H), seed + 3)
    block8 = _rand((snapshots, 8, H), seed + 4)
    return (
        core8,
        w,
        prefix8,
        block8,
        _rand((H,), seed + 5, 0.05),
        _norm_w(seed + 6),
        _norm_w(seed + 7),
    )


def _tail_inputs(ctx, snapshots, seed):
    r = ctx.rank
    latent8 = _rand((8, LATENT), seed + 11, 0.8)  # the reduced latent: the same on every rank
    act8 = _rand((8, ACT), seed + 1000 * r + 12, 0.5)
    w = _rand((H, PAD + ACT), seed + 1000 * r + 13, 0.03)
    w[:, WIDTH:PAD] = 0
    prefix8 = _rand((8, H), seed + 14)
    block8 = _rand((snapshots, 8, H), seed + 15)
    return (
        latent8,
        act8,
        w,
        r * WIDTH,
        prefix8,
        block8,
        _rand((H,), seed + 16, 0.05),
        _norm_w(seed + 17),
        _norm_w(seed + 18),
    )


def _first(m, prefix8, block8, with_prefix):
    return (prefix8[:m].contiguous() if with_prefix else None), block8[:, :m].contiguous()


def _perturbed(ctx, t, col=0):
    """t with one element changed on the last rank only."""
    bad = t.clone()
    if ctx.rank == ctx.world - 1:
        bad[0, col] += 1.0
    return bad


def check_oproj(ctx):
    results = []
    for snapshots in SNAPSHOTS:
        for with_prefix in (True, False):
            core8, w, prefix8, block8, res_w, rms_w, out_w = _oproj_inputs(
                ctx, snapshots, 100 * snapshots
            )
            pre8, _ = _first(8, prefix8, block8, with_prefix)
            n8, u8 = _oproj(ctx, core8, w, pre8, block8, res_w, rms_w, out_w)
            for m in M_ALL:
                core = core8[:m].contiguous()
                pre, block = _first(m, prefix8, block8, with_prefix)
                n, u = _oproj(ctx, core, w, pre, block, res_w, rms_w, out_w)
                want_n, want_u = _attn_res_ar(ctx, _row8(lambda x: F.linear(x, w), core, m), pre, block, res_w, rms_w,
                                              out_w)  # fmt: skip
                again = [_oproj(ctx, core, w, pre, block, res_w, rms_w, out_w) for _ in range(2)]
                bad_n, bad_u = _oproj(
                    ctx, _perturbed(ctx, core), w, pre, block, res_w, rms_w, out_w
                )
                row = dict(
                    op="k3_sandwich_oproj", case=f"S{snapshots}_{'prefix' if with_prefix else 'noprefix'}", M=m,
                    eq_unfused=_same(n, want_n) and _same(u, want_u), rel_updated=_rel(u, want_u),
                    rel_normed=_rel(n, want_n), det=all(_same(a, n) and _same(b, u) for a, b in again),
                    rows_as_m8=_same(n, n8[:m]) and _same(u, u8[:m]), control=not _same(bad_u, u),
                )  # fmt: skip
                row["ok"] = _all_ranks(
                    ctx, row["eq_unfused"] and row["det"] and row["rows_as_m8"] and row["control"]
                )
                results.append(row)
    return results


def _tap_mixture(updated, block, res_w, rms_w):
    """The unfused path's pre-norm attention-residual mixture: trtllm::attn_res_fwd on the updated row and the bank."""
    m, s = updated.shape[0], block.shape[0]
    out, _, _, _ = _trtllm().attn_res_fwd(updated.reshape(m, 1, H).contiguous(),
                                          block.reshape(s, m, 1, H).contiguous(), res_w.reshape(-1).contiguous(),
                                          rms_w.contiguous(), EPS)  # fmt: skip
    return out.reshape(m, H)


def check_tail(ctx):
    results = []
    for snapshots in SNAPSHOTS:
        for with_prefix in (True, False):
            latent8, act8, w, lo, prefix8, block8, res_w, rms_w, out_w = _tail_inputs(
                ctx, snapshots, 50 + snapshots
            )
            pre8, _ = _first(8, prefix8, block8, with_prefix)
            n8, u8 = _tail(ctx, latent8, act8, w, lo, pre8, block8, res_w, rms_w, out_w)
            for m in M_ALL:
                latent, act = latent8[:m].contiguous(), act8[:m].contiguous()
                pre, block = _first(m, prefix8, block8, with_prefix)
                n, u = _tail(ctx, latent, act, w, lo, pre, block, res_w, rms_w, out_w)
                part = _tail_partial(latent, act, w, lo)
                want_n, want_u = _attn_res_ar(ctx, part, pre, block, res_w, rms_w, out_w)
                again = [
                    _tail(ctx, latent, act, w, lo, pre, block, res_w, rms_w, out_w)
                    for _ in range(2)
                ]
                bad_n, bad_u = _tail(
                    ctx, latent, _perturbed(ctx, act), w, lo, pre, block, res_w, rms_w, out_w
                )
                row = dict(
                    op="k3_sandwich_tail", case=f"S{snapshots}_{'prefix' if with_prefix else 'noprefix'}", M=m,
                    rel_updated=_rel(u, want_u), rel_normed=_rel(n, want_n),
                    det=all(_same(a, n) and _same(b, u) for a, b in again), rows_as_m8=_same(n, n8[:m]) and _same(
                        u, u8[:m]), control=not _same(bad_u, u),
                )  # fmt: skip
                row["ok"] = _all_ranks(ctx, row["rel_updated"] <= 8e-3 and row["rel_normed"] <= 2e-2 and row["det"]
                                       and row["rows_as_m8"] and row["control"])  # fmt: skip
                results.append(row)
    # The DSpark capture tap and the snapshot bank row (updated_out), at every M.
    latent8, act8, w, lo, prefix8, block8, res_w, rms_w, out_w = _tail_inputs(ctx, 3, 90)
    layers = 5
    for m in M_ALL:
        latent, act = latent8[:m].contiguous(), act8[:m].contiguous()
        pre, block = _first(m, prefix8, block8, True)
        args = (latent, act, w, lo, pre, block, res_w, rms_w, out_w)
        n0, u0 = _tail(ctx, *args)
        cap = _nan(m, layers * H)
        tap = cap[:, 2 * H : 3 * H]
        n1, u1 = _tail(ctx, *args, tap=tap)
        capu = _nan(m, layers * H)
        n2, u2 = _tail(ctx, *args, tap=capu[:, 4 * H :], tap_updated=True)
        bank = _nan(5, m, H)
        n3, u3 = _tail(ctx, *args, updated_out=bank[2])
        mix = _tap_mixture(u0, block, res_w, rms_w)
        torch.cuda.synchronize()
        rest_cap = torch.cat([cap[:, : 2 * H], cap[:, 3 * H :]], dim=1)
        rest_bank = torch.cat([bank[:2], bank[3:]])
        frac = (_bits(tap) != _bits(mix)).float().mean().item()
        row = dict(
            op="k3_sandwich_tail", case="tap_updated_out", M=m,
            unchanged=_same(n1, n0) and _same(u1, u0) and _same(n2, n0) and _same(u2, u0) and _same(n3, n0)
            and u3.numel() == 0, tap_rel=_rel(tap, mix), tap_frac_diff=frac,
            tap_updated=_same(capu[:, 4 * H :], u0), bank_row=_same(bank[2], u0),
            untouched=bool(torch.isnan(rest_cap.float()).all()) and bool(torch.isnan(capu[:, : 4 * H].float()).all())
            and bool(torch.isnan(rest_bank.float()).all()),
        )  # fmt: skip
        row["ok"] = _all_ranks(ctx, row["unchanged"] and row["tap_rel"] <= 4e-3 and frac <= 1e-3
                               and row["tap_updated"] and row["bank_row"] and row["untouched"])  # fmt: skip
        results.append(row)
    return results


def _plain_inputs(ctx, k, seed):
    r = ctx.rank
    x8 = _rand((8, k), seed + 1000 * r + 1)
    w = _rand((H, k if k != 2 * DOWN_K else DOWN_K), seed + 1000 * r + 2, 0.03)
    return x8, w, _rand((8, H), seed + 3), _norm_w(seed + 4)


def _check_plain(ctx, swiglu):
    results = []
    name = "k3_sandwich_plain_swiglu" if swiglu else "k3_sandwich_plain"
    for seed in (0, 1):
        x8, w, res8, norm_w = _plain_inputs(
            ctx, 2 * DOWN_K if swiglu else PLAIN_K, 300 + 10 * seed + int(swiglu)
        )

        def gemv(x):
            if swiglu:
                return torch.ops.trtllm.k3_ctm_gemv_swiglu(x, w, True, 2, True)
            return torch.ops.trtllm.k3_ctm_gemv(x, w, True, 1)

        n8, u8 = _plain(ctx, x8, w, res8, norm_w, swiglu)
        for m in M_ALL:
            x, res = x8[:m].contiguous(), res8[:m].contiguous()
            n, u = _plain(ctx, x, w, res, norm_w, swiglu)
            want_n, want_u = _residual_rms_ar(ctx, gemv(x), res, norm_w)
            again = [_plain(ctx, x, w, res, norm_w, swiglu) for _ in range(2)]
            bad_n, bad_u = _plain(
                ctx, _perturbed(ctx, x, DOWN_K if swiglu else 0), w, res, norm_w, swiglu
            )
            row = dict(
                op=name, case=f"seed{seed}", M=m, eq_unfused=_same(n, want_n) and _same(u, want_u),
                rel_updated=_rel(u, want_u), rel_normed=_rel(n, want_n),
                det=all(_same(a, n) and _same(b, u) for a, b in again),
                rows_as_m8=_same(n, n8[:m]) and _same(u, u8[:m]), control=not _same(bad_u, u),
            )  # fmt: skip
            row["ok"] = _all_ranks(
                ctx, row["eq_unfused"] and row["det"] and row["rows_as_m8"] and row["control"]
            )
            results.append(row)
    return results


def check_plain(ctx):
    return _check_plain(ctx, swiglu=False)


def check_swiglu(ctx):
    return _check_plain(ctx, swiglu=True)


def check_replay(ctx):
    """One graph per M of [oproj, tail, plain] on static inputs, replayed in mixed M order with refilled inputs,
    against eager calls of the same ops (the engine replays the graph of each step's batch size in any order)."""
    graphs = {}
    stream = torch.cuda.Stream()
    for m in M_ALL:
        core8, w_o, prefix8, block8, res_w, rms_w, out_w = _oproj_inputs(ctx, 3, 700)
        latent8, act8, w_t, lo, t_prefix8, t_block8, t_res, t_rms, t_out = _tail_inputs(ctx, 3, 710)
        x8, w_p, res8, norm_w = _plain_inputs(ctx, PLAIN_K, 720)
        a_in = [
            core8[:m].clone(),
            w_o,
            prefix8[:m].clone(),
            block8[:, :m].clone(),
            res_w,
            rms_w,
            out_w,
        ]
        b_in = [latent8[:m].clone(), act8[:m].clone(), w_t, lo, t_prefix8[:m].clone(), t_block8[:, :m].clone(), t_res,
                t_rms, t_out]  # fmt: skip
        c_in = [x8[:m].clone(), w_p, res8[:m].clone(), norm_w]
        _oproj(ctx, *a_in)
        _tail(ctx, *b_in)
        _plain(ctx, *c_in)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph, stream=stream):
                outs = [_oproj(ctx, *a_in), _tail(ctx, *b_in), _plain(ctx, *c_in)]
        torch.cuda.synchronize()
        graphs[m] = (graph, a_in, b_in, c_in, outs)
    results = []
    for it, m in enumerate(REPLAY_ORDER):
        graph, a_in, b_in, c_in, outs = graphs[m]
        seed = 9000 + 97 * it
        a_in[0].copy_(_rand(tuple(a_in[0].shape), seed + ctx.rank))
        a_in[2].copy_(_rand(tuple(a_in[2].shape), seed + 1000))
        a_in[3].copy_(_rand(tuple(a_in[3].shape), seed + 2000))
        b_in[0].copy_(_rand(tuple(b_in[0].shape), seed + 3000, 0.8))
        b_in[1].copy_(_rand(tuple(b_in[1].shape), seed + 4000 + ctx.rank, 0.5))
        b_in[4].copy_(_rand(tuple(b_in[4].shape), seed + 5000))
        c_in[0].copy_(_rand(tuple(c_in[0].shape), seed + 6000 + ctx.rank))
        c_in[2].copy_(_rand(tuple(c_in[2].shape), seed + 7000))
        torch.cuda.synchronize()
        ctx.comm.Barrier()
        graph.replay()
        torch.cuda.synchronize()
        got = [[t.clone() for t in o] for o in outs]
        want = [_oproj(ctx, *a_in), _tail(ctx, *b_in), _plain(ctx, *c_in)]
        torch.cuda.synchronize()
        same = all(_same(g, x) for go, wo in zip(got, want) for g, x in zip(go, wo))
        results.append(
            dict(
                op="k3_sandwich_replay",
                case=f"replay{it}",
                M=m,
                eq_eager=same,
                ok=_all_ranks(ctx, same),
            )
        )
    del graphs
    return results


def check_wrap(ctx):
    """The all-reduce buffer's per-CTA call counters (``flags``, whose parity picks the buffer half) across the int32
    wrap: a sequence of oproj / tail / plain calls from counters preset just below 2**31 gives the bits of the same
    sequence from the counters as they were. Every CTA counts every call, so the counters stay equal; the preset keeps
    their parity, which decides the half the previous call emptied."""
    core8, w_o, prefix8, block8, res_w, rms_w, out_w = _oproj_inputs(ctx, 3, 800)
    latent8, act8, w_t, lo, t_prefix8, t_block8, t_res, t_rms, t_out = _tail_inputs(ctx, 3, 810)
    x8, w_p, res8, norm_w = _plain_inputs(ctx, PLAIN_K, 820)
    calls = []
    for m in (8, 1, 5):
        calls += [
            lambda m=m: _oproj(ctx, core8[:m].contiguous(), w_o, prefix8[:m].contiguous(), block8[:, :m].contiguous(),
                               res_w, rms_w, out_w),
            lambda m=m: _tail(ctx, latent8[:m].contiguous(), act8[:m].contiguous(), w_t, lo,
                              t_prefix8[:m].contiguous(), t_block8[:, :m].contiguous(), t_res, t_rms, t_out),
            lambda m=m: _plain(ctx, x8[:m].contiguous(), w_p, res8[:m].contiguous(), norm_w),
        ]  # fmt: skip

    def run():
        outs = [[t.clone() for t in call()] for call in calls]
        torch.cuda.synchronize()
        return outs

    flags = ctx.ws.flags
    fresh = run()
    count = int(flags[0].item())
    ctx.comm.Barrier()
    flags.fill_(2**31 - 4 + (count & 1))
    torch.cuda.synchronize()
    ctx.comm.Barrier()
    wrapped = run()
    after = int(flags[0].item())
    same = all(_same(a, b) for f_out, w_out in zip(fresh, wrapped) for a, b in zip(f_out, w_out))
    row = dict(
        op="k3_sandwich_wrap",
        case="int32_wrap",
        M=8,
        eq_fresh=same,
        crossed=after < 0,
        calls=len(calls),
    )
    row["ok"] = _all_ranks(ctx, same and row["crossed"])
    return [row]


def check_fold_wrap(ctx):
    """The tail with the latent all-reduce folded in (``lat_uc`` / ``lat_flags``; every rank pushes its partial rows
    into slot [rank] of half ``n & 1`` of every rank's exchange buffer through the multicast mapping, as k3_moe does)
    across the wrap of its call count n: the same calls from n preset just below 2**31 give the bits of the calls from
    n as it was, and no call writes a ``lat_flags`` word other than the count and the scale slab."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_sandwich import k3_sandwich_kernel as kernel
    from tensorrt_llm._torch.cute_dsl_kernels.k3_sandwich import op as sw_op

    ex = sw_op.K3SandwichLatentExchange.create(ctx.mapping, fabric_handle=True)
    flags = ex.flags
    lanes = ex.mc.view(2, 8, ctx.world, LATENT // 2)
    latent8, act8, w, lo, prefix8, block8, res_w, rms_w, out_w = _tail_inputs(ctx, 3, 830)
    part8 = _rand((8, LATENT), 840 + 1000 * ctx.rank, 0.3)
    part8[part8 == 0] = (
        0.0  # pushes never send -0.0: with +0.0 beside it, that word would read as empty
    )
    slab = slice(kernel.LAT_SCALES, kernel.LAT_SCALES + kernel.LAT_SCALE_BUFS * 8)

    def run():
        outs, others = [], []
        for m in (8, 1, 5, 3, 8, 2):
            torch.cuda.synchronize()
            ctx.comm.Barrier()  # every rank's previous call has emptied the half this call's pushes go to
            lanes[int(flags[0].item()) & 1, :m, ctx.rank].copy_(part8[:m].view(torch.int32))
            torch.cuda.synchronize()
            ctx.comm.Barrier()
            before = torch.cat([flags[1 : slab.start], flags[slab.stop :]]).clone()
            pre, block = _first(m, prefix8, block8, True)
            out = _tail(ctx, latent8[:m].contiguous(), act8[:m].contiguous(), w, lo, pre, block, res_w, rms_w, out_w,
                        lat_uc=ex.uc, lat_flags=flags)  # fmt: skip
            torch.cuda.synchronize()
            outs.append([t.clone() for t in out])
            others.append(
                torch.equal(before, torch.cat([flags[1 : slab.start], flags[slab.stop :]]))
            )
        return outs, others

    fresh, fresh_kept = run()
    count = int(flags[0].item())
    ctx.comm.Barrier()
    flags[0] = 2**31 - 4 + (count & 1)
    torch.cuda.synchronize()
    wrapped, wrapped_kept = run()
    same = all(_same(a, b) for f_out, w_out in zip(fresh, wrapped) for a, b in zip(f_out, w_out))
    kept = all(fresh_kept) and all(wrapped_kept)
    row = dict(op="k3_sandwich_tail_fold", case="count_wrap", M=8, eq_fresh=same, other_words_kept=kept,
               count_after=int(flags[0].item()))  # fmt: skip
    row["ok"] = _all_ranks(ctx, same and kept)
    return [row]


CHECKS = {"oproj": check_oproj, "tail": check_tail, "plain": check_plain, "swiglu": check_swiglu,
          "replay": check_replay, "wrap": check_wrap, "fold_wrap": check_fold_wrap}  # fmt: skip


def _run_checks(names):
    """Every rank runs the same checks in the same order (they are collectives); returns this rank's result rows."""
    try:
        ctx = _context()
        with torch.inference_mode():
            return [row for name in names for row in CHECKS[name](ctx)]
    except Exception:
        traceback.print_exc()
        raise


def _report(rows):
    for row in rows:
        fields = " ".join(f"{k}={(f'{v:.3e}' if isinstance(v, float) else v)}" for k, v in row.items()
                          if k not in ("op", "case", "M"))  # fmt: skip
        print(f"OPCHECK op={row['op']} case={row['case']} M={row['M']} {fields}", flush=True)


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("check", list(CHECKS))
def test_k3_sandwich(mpi_pool_executor, check):
    per_rank = list(mpi_pool_executor.map(_run_checks, [[check]] * WORLD))
    _report(per_rank[0])
    assert all(row["ok"] for rows in per_rank for row in rows)


def test_workspaces_refuse_graph_capture():
    """The all-reduce buffer and the latent exchange are created collectively: creating either under CUDA-graph
    capture raises instead of entering the collective, which could hang the group."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_sandwich import op
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        with pytest.raises(RuntimeError, match="outside CUDA-graph capture"):
            op.K3SandwichWorkspace.create(mapping)
        with pytest.raises(RuntimeError, match="outside CUDA-graph capture"):
            op.K3SandwichLatentExchange.create(mapping)


def main() -> int:
    names = sys.argv[1:] or list(CHECKS)
    rows = _run_checks(names)
    if MPI.COMM_WORLD.Get_rank() == 0:
        _report(rows)
        print("PASS" if all(r["ok"] for r in rows) else "FAIL", flush=True)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
