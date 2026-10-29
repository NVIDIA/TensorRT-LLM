# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 o_proj -> attention residual through a Lamport mailbox (TRTLLM_K3_OPROJ_LAMPORT).

The attn_res slots consumer against attn_res on the MNNVL-order sum (payloads with +-0,
subnormals and cancellations, emulated larger groups), the o_proj GEMM's publish against the
stock GEMM (base tactics through the Lamport epilogue, split-K and mixed-cluster tactics through
the copy kernel), and CUDA-graph replays interleaved with eager uses. Every comparison is
bitwise. The CPU tests are tests/unittest/_torch/modules/test_kimi_k3_oproj_lamport.py."""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN = 7168
SLOT_BYTES = HIDDEN * 2
K_FULL = 12288
EPS = 1e-6
# Polls trap after this many failed passes (tens of seconds), so a hang fails the test.
SPIN_LIMIT = 1 << 26
# The production o_proj tactic at TP8 and the runner's fallback.
TACTICS = (("base", (256, 64, 128), (256, 64, 64), (2, 1), True, False, "static", "m"), None)


def _trtllm_op(name):
    """``torch.ops.trtllm.<name>``, looked up by string: this module is pickled by value for the
    MPI workers, and a function that names ``torch.ops`` would capture its ``_Ops`` object, which
    cloudpickle cannot pickle."""
    return getattr(getattr(getattr(torch, "ops"), "trtllm"), name)


def _device_ok(rubin_only):
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap == (10, 7) if rubin_only else cap in ((10, 0), (10, 3), (10, 7))


# ---------------------------------------------------------------------------------------------
# multi-GPU
# ---------------------------------------------------------------------------------------------
def _payloads(seed, world, m, device):
    """[world, m, H] BF16 with +-0, subnormals, 2^-15..2^16 and pairwise cancellations."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    shape = (world, m, HIDDEN)
    kw = dict(generator=g, device=device, dtype=torch.int32)
    bits = (torch.randn(shape, generator=g, device=device) * 4.0).to(torch.bfloat16).view(
        torch.int16
    ).to(torch.int32) & 0xFFFF
    sel = torch.randint(0, 100, shape, **kw)
    sign = torch.randint(0, 2, shape, **kw) << 15
    mant = torch.randint(0, 128, shape, **kw)
    expo = torch.randint(0x70, 0x8F, shape, **kw) << 7
    bits = torch.where(sel < 4, torch.zeros_like(bits), bits)
    bits = torch.where((sel >= 4) & (sel < 8), torch.full_like(bits, 0x8000), bits)
    bits = torch.where((sel >= 8) & (sel < 16), sign | mant.clamp(min=1), bits)
    bits = torch.where((sel >= 16) & (sel < 20), sign | expo | mant, bits)
    pair = bits.clone()
    pair[1::2] = bits[0::2] ^ 0x8000
    bits = torch.where((sel >= 20) & (sel < 24), pair, bits)
    bits = torch.where(bits >= 0x8000, bits - 0x10000, bits).to(torch.int16)
    return bits.view(torch.bfloat16)


def _pos_zero(x):
    bits = x.view(torch.int16)
    return torch.where(bits == -32768, torch.zeros_like(bits), bits).view(torch.bfloat16)


def _mnnvl_order_sum(partials, chunk):
    acc = torch.zeros(partials.shape[1:], dtype=torch.float32, device=partials.device)
    for first in range(0, partials.shape[0], chunk):
        part = torch.zeros_like(acc)
        for r in range(first, first + chunk):
            part = part + partials[r].float()
        acc = acc + part
    return acc.to(torch.bfloat16)


def _all_zero(comm, bad, what):
    """Assert on every rank at once: a rank failing alone would leave its peers waiting."""
    per_rank = comm.allgather(bad)
    assert sum(per_rank) == 0, f"{what}: mismatches per rank {per_rank}"


class _Ctx:
    def __init__(self, tp, rank, device, comm, arena, weights):
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res
        from tensorrt_llm._torch.cute_dsl_kernels.rubin.k3_lamport import oneshot_chunk

        self.tp, self.rank, self.device, self.comm, self.arena = tp, rank, device, comm, arena
        self.weights, self.ar, self.chunk = weights, k3_attn_res, oneshot_chunk
        self.count = 0
        self.mc_words = arena.handle.get_mc_buffer([arena.total_bytes // 4], torch.int32, 0)

    def parity(self):
        p = self.count % 3
        self.count += 1
        return p

    def bufs(self, m, fp8):
        out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=self.device)
        upd = torch.empty_like(out)
        f8 = sc = None
        if fp8:
            f8 = torch.empty(m, HIDDEN, dtype=torch.float8_e4m3fn, device=self.device)
            sc = torch.empty(self.ar.fp8_scale_numel(m), dtype=torch.uint8, device=self.device)
        return out, upd, f8, sc

    def reference(self, summed, prefix, block, fp8, exact):
        out, upd, f8, sc = bufs = self.bufs(summed.shape[0], fp8)
        if prefix is None:
            self.ar.run_k3_attn_res(
                summed,
                (),
                block,
                *self.weights,
                EPS,
                EPS,
                out,
                None,
                exact=exact,
                fp8_output=f8,
                fp8_scale=sc,
            )
            upd.copy_(summed)
        else:
            self.ar.run_k3_attn_res(
                prefix,
                (summed,),
                block,
                *self.weights,
                EPS,
                EPS,
                out,
                upd,
                exact=exact,
                fp8_output=f8,
                fp8_scale=sc,
            )
        return bufs

    def slots(self, parity, world, prefix, block, fp8, exact, **kw):
        out, upd, f8, sc = bufs = self.bufs(block.shape[1], fp8)
        self.ar.run_k3_attn_res_slots(
            prefix,
            self.arena.local_addr(parity),
            world,
            block,
            *self.weights,
            EPS,
            EPS,
            out,
            upd,
            exact=exact,
            fp8_output=f8,
            fp8_scale=sc,
            **kw,
        )
        return bufs

    def publish_words(self, payloads, parity, world):
        per = self.arena.region_bytes // 4
        m = payloads.shape[1]
        words = self.mc_words[parity * per : parity * per + m * world * HIDDEN // 2].view(
            m, world, HIDDEN // 2
        )
        for vr in range(self.rank, world, self.tp):
            words[:, vr].copy_(_pos_zero(payloads[vr].contiguous()).view(torch.int32))


def _bad(a, b):
    return sum(
        int((x.view(torch.uint8) != y.view(torch.uint8)).sum().item())
        for x, y in zip(a, b)
        if x is not None
    )


def _inputs(seed, m, n, device):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    prefix = (torch.randn(m, HIDDEN, generator=g, device=device) * 2.0).to(torch.bfloat16)
    block = (torch.randn(n - 1, m, HIDDEN, generator=g, device=device) * 2.0).to(torch.bfloat16)
    return prefix, block


def _consumer_phase(ctx):
    ran = 0
    for world in (ctx.tp, 2 * ctx.tp):
        for m in (1, 6, 33):
            for n in (3, 6):
                for add in (False, True):
                    for fp8, exact, early, pc in (
                        (False, True, False, None),
                        (True, True, True, None),
                        (True, False, False, None),
                        (False, True, False, 1),
                    ):
                        seed = (
                            101 * world + 13 * m + 7 * n + int(add) + 2 * int(fp8) + 4 * int(exact)
                        )
                        pay = _payloads(seed, world, m, ctx.device)
                        prefix, block = _inputs(seed, m, n, ctx.device)
                        ref = ctx.reference(
                            _mnnvl_order_sum(pay, ctx.chunk(world)),
                            prefix if add else None,
                            block,
                            fp8,
                            exact,
                        )
                        torch.cuda.synchronize()
                        ctx.comm.Barrier()
                        parity = ctx.parity()
                        ctx.publish_words(pay, parity, world)
                        got = ctx.slots(
                            parity,
                            world,
                            prefix if add else None,
                            block,
                            fp8,
                            exact,
                            poll_early=early,
                            poll_ctas=pc,
                        )
                        torch.cuda.synchronize()
                        _all_zero(ctx.comm, _bad(got, ref), f"consumer w{world} m{m} n{n} add{add}")
                        ran += 1
    torch.cuda.synchronize()
    ctx.comm.Barrier()
    _all_zero(ctx.comm, ctx.arena.unarmed_words(), "unarmed words after the consumer phase")
    assert ran == 2 * 3 * 2 * 2 * 4


def _operands(seed, world, m, device):
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    k = K_FULL // world
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    ops = []
    for _ in range(world):
        x = torch.randn(m, k, generator=g, device=device).to(torch.bfloat16)
        w = (torch.randn(HIDDEN, k, generator=g, device=device) * 0.02).to(torch.bfloat16)
        a_fp8, a_sf = _trtllm_op("fp8_quantize_1x128_cutedsl_ue8m0")(x)
        w_fp8, w_sf = _Fp8BlockScaleWeightReadLinear.quantize_weight(w)
        ops.append((a_fp8, a_sf, w_fp8, w_sf))
    return ops


def _kind_tactic(stock, inputs, kind):
    """The first valid tactic of ``kind`` (split-K needs TRTLLM_K3_MXFP8_SPLIT_K=1), or None."""
    from tensorrt_llm._torch.autotuner import OptimizationProfile, StaticDim

    prof = OptimizationProfile([[StaticDim(x) for x in t.size()] for t in inputs])
    for tactic in stock.get_valid_tactics(inputs, prof):
        if isinstance(tactic, tuple) and tactic and tactic[0] == kind:
            return tactic
    return None


def _producer_phase(ctx):
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as cdo

    stock = cdo.CuteDSLKimiK3MXFP8RubinLinear(output_dtype=torch.bfloat16)
    alpha = cdo._get_mxfp8_gemm_alpha(ctx.device)

    def inputs_of(op):
        return [op[0], op[2], op[1], op[3], alpha]

    ran = set()
    for world in (ctx.tp, 2 * ctx.tp):
        for m in (1, 6, 32):
            ops = _operands(7 * world + m, world, m, ctx.device)
            probe = inputs_of(ops[0])
            routed = [_kind_tactic(stock, probe, k) for k in ("splitk", "mixed_clusters")]
            # Every rank runs the same tactics (rank 0's).
            tactics = ctx.comm.bcast(list(TACTICS) + [t for t in routed if t is not None], root=0)
            for tactic in tactics:
                prefix, block = _inputs(world + m, m, 6, ctx.device)
                partials = torch.stack([stock.forward(inputs_of(op), tactic or -1) for op in ops])
                ref = ctx.reference(
                    _mnnvl_order_sum(partials, ctx.chunk(world)), prefix, block, True, True
                )
                torch.cuda.synchronize()
                ctx.comm.Barrier()
                parity = ctx.parity()
                base = ctx.arena.mc_addr(parity)
                for vr in range(ctx.rank, world, ctx.tp):
                    cdo.mxfp8_gemm_rubin_lamport(
                        stock,
                        inputs_of(ops[vr]),
                        tactic or -1,
                        base + vr * SLOT_BYTES,
                        world * HIDDEN,
                    )
                got = ctx.slots(parity, world, prefix, block, True, True)
                torch.cuda.synchronize()
                _all_zero(ctx.comm, _bad(got, ref), f"producer w{world} m{m} {tactic}")
                ran.add(tactic[0] if isinstance(tactic, tuple) else "default")
    # Split-K (offered at these shapes with the switch) must have gone through the copy kernel;
    # mixed-cluster tactics run where valid.
    assert "splitk" in ran, ran
    # The public entry point (autotuner-chosen tactic) publishes the same partials.
    ops = _operands(99, ctx.tp, 4, ctx.device)
    prefix, block = _inputs(98, 4, 3, ctx.device)
    partials = torch.stack(
        [
            _trtllm_op("cute_dsl_mxfp8_gemm_rubin")(op[0], op[2], op[1], op[3], fine_grained_m=True)
            for op in ops
        ]
    )
    ref = ctx.reference(_mnnvl_order_sum(partials, ctx.chunk(ctx.tp)), None, block, False, True)
    torch.cuda.synchronize()
    ctx.comm.Barrier()
    parity = ctx.parity()
    op = ops[ctx.rank]
    cdo.cute_dsl_mxfp8_gemm_rubin_lamport(
        op[0],
        op[2],
        op[1],
        op[3],
        ctx.arena.mc_addr(parity) + ctx.rank * SLOT_BYTES,
        ctx.tp * HIDDEN,
    )
    got = ctx.slots(parity, ctx.tp, None, block, False, True)
    torch.cuda.synchronize()
    _all_zero(ctx.comm, _bad(got, ref), "cute_dsl_mxfp8_gemm_rubin_lamport")


def _graph_phase(ctx):
    """A captured forward of 3 uses (publish + consumer), replayed between eager uses."""
    from tensorrt_llm._torch.distributed.k3_lamport_mailbox import lamport_parities

    m, n, uses = 6, 3, 3
    parities = lamport_parities(uses)
    payloads = [
        torch.empty(ctx.tp, m, HIDDEN, dtype=torch.bfloat16, device=ctx.device) for _ in range(uses)
    ]
    prefix, block = _inputs(5, m, n, ctx.device)
    block2 = block[:, :2].contiguous()  # the bank of the 2-row eager uses
    outs = []

    def forward():
        outs.clear()
        for u, p in enumerate(parities):
            ctx.publish_words(payloads[u], p, ctx.tp)
            outs.append(ctx.slots(p, ctx.tp, prefix, block, False, True))

    for u in range(uses):
        payloads[u].copy_(_payloads(u, ctx.tp, m, ctx.device))
    forward()
    torch.cuda.synchronize()
    ctx.comm.Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    torch.cuda.synchronize()
    ctx.comm.Barrier()
    graph_outs = list(outs)
    for it in range(20):
        refs = []
        for u in range(uses):
            pay = _payloads(1000 + 7 * it + u, ctx.tp, m, ctx.device)
            payloads[u].copy_(pay)
            refs.append(
                ctx.reference(_mnnvl_order_sum(pay, ctx.chunk(ctx.tp)), prefix, block, False, True)
            )
        graph.replay()
        bad = sum(_bad(o, r) for o, r in zip(graph_outs, refs))
        pay = _payloads(5000 + it, ctx.tp, 2, ctx.device)  # an eager use between replays
        ref = ctx.reference(_mnnvl_order_sum(pay, ctx.chunk(ctx.tp)), None, block2, False, True)
        parity = ctx.parity()
        ctx.publish_words(pay, parity, ctx.tp)
        bad += _bad(ctx.slots(parity, ctx.tp, None, block2, False, True), ref)
        torch.cuda.synchronize()
        _all_zero(ctx.comm, bad, f"graph replay {it}")
    torch.cuda.synchronize()
    ctx.comm.Barrier()
    _all_zero(ctx.comm, ctx.arena.unarmed_words(), "unarmed words after the replays")


def _rank_body(tp, rank, rubin):
    from tensorrt_llm._torch.distributed.k3_lamport_mailbox import K3LamportArena

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    comm = MPI.COMM_WORLD
    mapping = Mapping(world_size=tp, rank=rank, gpus_per_node=tp, tp_size=tp)
    arena = K3LamportArena.get(mapping, region_bytes=64 * 2 * tp * SLOT_BYTES)
    g = torch.Generator(device=device)
    g.manual_seed(3)
    weights = tuple(
        (torch.randn(HIDDEN, generator=g, device=device) * s + b).to(torch.bfloat16)
        for s, b in ((0.05, 0.0), (0.1, 1.0), (0.1, 1.0))
    )
    ctx = _Ctx(tp, rank, device, comm, arena, weights)
    _consumer_phase(ctx)
    if rubin:
        _producer_phase(ctx)
    _graph_phase(ctx)
    return True


def _run_rank(tp, rubin):
    rank = tensorrt_llm.mpi_rank()
    env = {
        "TLLM_TEST_MNNVL": "1",
        "TRTLLM_K3_LAMPORT_SPIN_LIMIT": str(SPIN_LIMIT),
        "TRTLLM_K3_MXFP8_SPLIT_K": "1",  # offer split-K tactics (published by the copy kernel)
    }
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)  # spawned workers may not inherit the parent's environment
    try:
        return _rank_body(tp, rank, rubin)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.mark.skipif(
    torch.cuda.device_count() < 4 or not _device_ok(rubin_only=False),
    reason="needs 4 Blackwell/Rubin GPUs with multicast",
)
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_k3_oproj_lamport_bitwise(mpi_pool_executor):
    tp = mpi_pool_executor.num_workers
    rubin = _device_ok(rubin_only=True)
    results = mpi_pool_executor.map(_run_rank, [tp] * tp, [rubin] * tp)
    for r in results:
        assert r is True
