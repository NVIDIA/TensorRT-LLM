# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 decode MoE engine (one persistent launch: FC1 -> SwiGLU -> NVFP4 -> FC2) against the trtllm-gen FP4 MoE
runner's pre-routed un-finalized run on the same routing: every (token, k) gemm2 row bit for bit.

One TEP8 rank's shapes: 128 local experts of 512, hidden 4096, intermediate 512 per rank, top-10. Random
checkpoint-format NVFP4 experts through the loader's own shuffle; block scales over 2^-6..2^6 so FP32 sums round."""

import functools
import sys
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._utils import is_blackwell_datacenter

pytestmark = pytest.mark.skipif(not is_blackwell_datacenter(), reason="needs SM 10.0 / 10.3")

H, INTER, E, E_LOCAL, TOPK = 4096, 512, 512, 128, 10
LAMPORT_MAX_ROWS = 128  # the model's (the slot all-reduce's rows)


def _experts(seed=20261106, w2_identity=False):
    from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4TRTLLMGenFusedMoEMethod

    gen = torch.Generator(device="cuda").manual_seed(seed)

    def sf(*shape):
        return torch.randint(
            0x08, 0x70, shape, dtype=torch.uint8, device="cuda", generator=gen
        ).view(torch.float8_e4m3fn)

    w31 = torch.randint(
        0, 256, (E_LOCAL, 2 * INTER, H // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    w31s = sf(E_LOCAL, 2 * INTER, H // 16)
    if w2_identity:
        w2 = torch.zeros((E_LOCAL, H, INTER // 2), dtype=torch.uint8, device="cuda")
        n = torch.arange(INTER, device="cuda")
        w2[:, n, n // 2] = torch.where(n % 2 == 0, 0x02, 0x20).to(torch.uint8)
        w2s = torch.full((E_LOCAL, H, INTER // 16), 0x38, dtype=torch.uint8, device="cuda").view(
            torch.float8_e4m3fn
        )
    else:
        w2 = torch.randint(
            0, 256, (E_LOCAL, H, INTER // 2), dtype=torch.uint8, device="cuda", generator=gen
        )
        w2s = sf(E_LOCAL, H, INTER // 16)
    module = SimpleNamespace(
        w3_w1_weight=w31,
        w2_weight=w2,
        w3_w1_weight_scale=w31s,
        w2_weight_scale=w2s,
        is_gated_activation=True,
        bias=False,
    )
    NVFP4TRTLLMGenFusedMoEMethod()._shuffle_all_experts(module)
    x_scale, h_scale = 448.0 * 6.0 / 8.0, 448.0 * 6.0 / 24.0
    a = (
        1.5
        / (x_scale * 2.93 * 1.3 * H**0.5)
        * (0.8 + 0.4 * torch.rand(E_LOCAL, device="cuda", generator=gen))
    )
    a2 = (
        torch.ones(E_LOCAL, device="cuda")
        if w2_identity
        else (
            (0.8 + 0.4 * torch.rand(E_LOCAL, device="cuda", generator=gen))
            / (h_scale * 2.93 * 1.3 * INTER**0.5)
        )
    )
    return SimpleNamespace(
        w31=w31,
        w31s=w31s,
        w2=w2,
        w2s=w2s,
        scale_c=(h_scale * a).float().contiguous(),
        alpha1=a.float().contiguous(),
        alpha2=a2.float().contiguous(),
        x_scale=x_scale,
    )


def _inputs(m, ex, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, H, device="cuda", generator=gen).to(torch.bfloat16)
    xq, xsf = torch.ops.trtllm.fp4_quantize(
        x, torch.tensor([ex.x_scale], device="cuda"), 16, False, False
    )
    w = torch.full((E,), 0.5 / (E - E_LOCAL), device="cuda")
    w[:E_LOCAL] = 0.5 / E_LOCAL
    ids = (
        torch.multinomial(w.unsqueeze(0).repeat(m, 1), TOPK, replacement=False, generator=gen)
        .int()
        .contiguous()
    )
    wts = torch.softmax(torch.randn(m, TOPK, device="cuda", generator=gen), -1).to(torch.bfloat16)
    return xq, xsf.view(torch.uint8).reshape(m, -1), ids, wts


def _tables(ids, tile):
    """trtllm-gen's routing tables (decode_front.prerouted_table_sizes order) built on the host."""
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_front import (
        max_num_tiles,
        prerouted_table_sizes,
    )

    m, k = ids.shape
    lid = ids.flatten().cpu().long()
    local = (lid >= 0) & (lid < E_LOCAL)
    counts = torch.bincount(lid[local], minlength=E_LOCAL)
    ncta = (counts + tile - 1) // tile
    offs = torch.cumsum(ncta, 0) - ncta
    sizes = prerouted_table_sizes(m, k, E_LOCAL, tile)
    flat = torch.zeros(sum(sizes), dtype=torch.int32)
    e2p, p2t, batch, limit, nct, npad = torch.split(flat, sizes)
    for e in range(E_LOCAL):
        for c in range(int(ncta[e])):
            cta = int(offs[e]) + c
            batch[cta] = e
            limit[cta] = min((cta + 1) * tile, int(offs[e]) * tile + int(counts[e]))
    rank = torch.zeros(E_LOCAL, dtype=torch.long)
    e2p.fill_(-1)
    for i in range(m * k):
        if local[i]:
            e = int(lid[i])
            p = int(offs[e]) * tile + int(rank[e])
            rank[e] += 1
            e2p[i] = p
            p2t[p] = i // k
    nct[0] = int(ncta.sum())
    npad[0] = int(ncta.sum()) * tile
    return list(torch.split(flat.cuda(), sizes)), max_num_tiles(m, k, E_LOCAL, tile) * tile


def _stock(ex, xq, xsf, ids, wts):
    from tensorrt_llm._torch.custom_ops import trtllm_gen_custom_ops as G

    m = xq.shape[0]
    r = G.FP4BlockScaleMoERunner(
        E, TOPK, None, None, INTER, 0, E_LOCAL, None, 1, False, G.ActType_TrtllmGen.SwiGlu.value
    )
    tactics = [
        list(map(int, t)) for t in r.get_runner().get_valid_configs(TOPK, H, INTER, E_LOCAL, m)
    ]
    tile = min(t[0] for t in tactics)
    tac = next(t for t in tactics if t[0] == tile)
    views, _ = _tables(ids, tile)
    inputs = [
        None,
        None,
        xq,
        xsf.view(torch.float8_e4m3fn).flatten(),
        ex.w31,
        ex.w31s,
        None,
        None,
        None,
        None,
        ex.w2,
        ex.w2s,
        None,
        ex.scale_c,
        ex.alpha1,
        ex.alpha2,
        wts,
        ids,
    ]
    out = r.forward(inputs, tactic=tac, prerouted=(views, ids, wts))
    e2p = views[0]
    valid = e2p >= 0
    return out[0][e2p[valid].long()], valid


def _engine(
    ex,
    xq,
    xsf,
    ids,
    tile,
    n_max,
    state,
    out=None,
    tile_k=256,
    release_after_wait=False,
    spare_sms=0,
):
    from tensorrt_llm._torch.kernels.blackwell.moe_nvfp4_engine import moe_nvfp4_engine

    views, cap = _tables(ids, tile)
    e2p, p2t, batch, limit, nct, _ = views
    h = torch.zeros((cap, INTER // 2), dtype=torch.uint8, device="cuda")
    hsf = torch.zeros((cap * INTER // 16,), dtype=torch.uint8, device="cuda")
    if out is None:
        out = torch.zeros((cap, H), dtype=torch.bfloat16, device="cuda")
    moe_nvfp4_engine(
        xq,
        xsf,
        p2t,
        batch,
        limit,
        nct,
        ex.w31,
        ex.w31s.view(torch.uint8),
        ex.w2,
        ex.w2s.view(torch.uint8),
        ex.scale_c,
        ex.alpha1,
        ex.alpha2,
        h,
        hsf,
        out,
        state,
        tile=tile,
        n_max=n_max,
        tile_k=tile_k,
        release_after_wait=release_after_wait,
        spare_sms=spare_sms,
    )
    torch.cuda.synchronize()
    return out, h, hsf, e2p


@pytest.fixture(scope="module")
def state():
    from tensorrt_llm._torch.kernels.blackwell.moe_nvfp4_engine import new_state

    return new_state(torch.device("cuda"))


@pytest.mark.parametrize(
    "m,n_max,tile",
    [
        (8, 16, 8),
        (56, 64, 8),
        (112, 64, 8),
        (224, 64, 8),
        (224, 16, 16),
        (896, 64, 8),
        (896, 32, 64),
    ],
)
def test_engine_rows_are_trtllm_gen_bit_for_bit(m, n_max, tile, state):
    import tensorrt_llm  # noqa: F401

    ex = _experts()
    xq, xsf, ids, wts = _inputs(m, ex, seed=11 + m)
    ref, valid = _stock(ex, xq, xsf, ids, wts)
    out, _, _, e2p = _engine(ex, xq, xsf, ids, tile, n_max, state)
    got = out[e2p[valid].long()]
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    again, _, _, _ = _engine(ex, xq, xsf, ids, tile, n_max, state)
    assert torch.equal(again[e2p[valid].long()].view(torch.int16), got.view(torch.int16))


@pytest.mark.parametrize("release_after_wait", [False, True])
@pytest.mark.parametrize("m", [56, 63, 64, 112])
def test_engine_hook_tiles_bit_for_bit(m, release_after_wait, state):
    """The hook's n_max / K tile per row count, with either PDL release point, leave the rows unchanged."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_moe import n_max_for, tile_k_for

    ex = _experts()
    xq, xsf, ids, wts = _inputs(m, ex, seed=13 + m)
    ref, valid = _stock(ex, xq, xsf, ids, wts)
    n_max = n_max_for(m)
    out, _, _, e2p = _engine(
        ex,
        xq,
        xsf,
        ids,
        8,
        n_max,
        state,
        tile_k=tile_k_for(n_max, H, INTER),
        release_after_wait=release_after_wait,
    )
    assert torch.equal(out[e2p[valid].long()].view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize("m", [448, 896])
def test_engine_rows_do_not_depend_on_the_grid(m, state):
    """Beside a shared expert the engine leaves SMs free (decode_moe.ENGINE_SPARE_SMS): its rows are the full grid's,
    bit for bit (the FC1 units balance over the grid, the FC2 units are claimed as tickets)."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_moe import (
        ENGINE_SPARE_SMS,
        n_max_for,
        tile_k_for,
    )

    ex = _experts()
    xq, xsf, ids, _ = _inputs(m, ex, seed=29 + m)
    n_max = n_max_for(m)
    opts = dict(tile_k=tile_k_for(n_max, H, INTER), release_after_wait=True)
    full, _, _, e2p = _engine(ex, xq, xsf, ids, 8, n_max, state, **opts)
    valid = e2p[e2p >= 0].long()
    for spare in (ENGINE_SPARE_SMS, 2 * ENGINE_SPARE_SMS):
        out, _, _, _ = _engine(ex, xq, xsf, ids, 8, n_max, state, spare_sms=spare, **opts)
        assert torch.equal(out[valid].view(torch.int16), full[valid].view(torch.int16)), spare


@pytest.mark.parametrize(
    "m,tile,release_after_wait",
    [
        (7, 8, False),
        (28, 8, False),
        (56, 8, False),
        (63, 16, False),
        (64, 8, True),
        (112, 16, True),
    ],
)
def test_engine_writes_only_real_rows_of_a_sentinel_buffer(m, tile, release_after_wait, state):
    """Lamport FC2 (decode_moe.LamportFc2Buffer) hands the engine a persistent buffer that holds 0xFFFFFFFF
    wherever no row is written; the MoE-output all-reduce's fold polls each word of the e2p-valid rows until it is not
    0xFFFFFFFF, then writes it back. So the engine must write exactly those rows, bit for bit its default output and
    never a 0xFFFFFFFF word, and leave every other word alone (padded rows, rows past the call's). Two calls in a row
    with the fold's reset between them, as consecutive layers share the buffer; the hook's n_max and tile_k."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_front import max_num_tiles
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_moe import n_max_for, tile_k_for

    ex = _experts()
    n_max = n_max_for(m)
    opts = dict(tile_k=tile_k_for(n_max, H, INTER), release_after_wait=release_after_wait)
    cap = max_num_tiles(m, TOPK, E_LOCAL, tile) * tile
    buf = torch.full((cap + 64, H), -1, dtype=torch.int16, device="cuda")
    for seed in (19 + m, 23 + m):
        xq, xsf, ids, _ = _inputs(m, ex, seed=seed)
        ref, _, _, e2p = _engine(ex, xq, xsf, ids, tile, n_max, state, **opts)
        rows = e2p[e2p >= 0].long()
        _engine(ex, xq, xsf, ids, tile, n_max, state, out=buf[:cap].view(torch.bfloat16), **opts)
        assert torch.equal(buf[rows], ref.view(torch.int16)[rows])
        assert not (buf[rows].view(torch.int32) == -1).any()
        rest = torch.ones(buf.shape[0], dtype=torch.bool, device="cuda")
        rest[rows] = False
        assert (buf[rest] == -1).all()
        buf[rows] = -1  # the fold's write-back


def _op_inputs(ex, xq, xsf):
    """The FP4 runner's inputs without router logits / top-k (FP4BlockScaleMoEInputs order)."""
    return [
        None,
        None,
        xq,
        xsf.view(torch.float8_e4m3fn).flatten(),
        ex.w31,
        ex.w31s,
        None,
        None,
        None,
        None,
        ex.w2,
        ex.w2s,
        None,
        ex.scale_c,
        ex.alpha1,
        ex.alpha2,
        None,
        None,
    ]


def _decode_runner(decode_tuning=True, decode_spare_sms=0):
    from tensorrt_llm._torch.custom_ops import trtllm_gen_custom_ops as G

    return G.FP4BlockScaleMoERunner(
        E,
        TOPK,
        None,
        None,
        INTER,
        0,
        E_LOCAL,
        None,
        1,
        False,
        G.ActType_TrtllmGen.SwiGlu.value,
        decode_tuning=decode_tuning,
        decode_spare_sms=decode_spare_sms,
    )


def _tile8_tactics(runner, m):
    """(a tile-8 tactic, a tile-8 tactic whose FC2 cubin the CuTe FC2 reproduces); the runner offers one."""
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_moe import cute_fc2_reproduces

    cr = runner.get_runner()
    tactics = [list(map(int, t)) for t in cr.get_valid_configs(TOPK, H, INTER, E_LOCAL, m)]
    tile8 = [t for t in tactics if t[0] == 8]
    bitwise = [t for t in tile8 if cute_fc2_reproduces(cr.get_gemm2_kernel_options(t[0], t[1]))]
    assert bitwise, [cr.get_gemm2_kernel_name(t[0], t[1]) for t in tile8]
    return tile8[0], bitwise[0]


def _spy_select(monkeypatch):
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import decode_moe

    picked, orig = [], decode_moe.select

    def spy(*args, **kwargs):
        picked.append(orig(*args, **kwargs))
        return picked[-1]

    monkeypatch.setattr(decode_moe, "select", spy)
    return picked


def test_decode_kernels_select_the_engine_with_the_runner_abi(monkeypatch):
    """A pre-routed un-finalized call of a runner that declares the decode kernels takes the engine when its backend
    hands in the engine state, and otherwise the FC1-only runner + CuTe FC2 or the cubins; the rows reached through
    the expanded -> permuted index are the stock runner's bit for bit either way, and the (rows, weights [m, k],
    index [m, k] int32) triple is what the MoE-output all-reduce's in-kernel finalize accepts."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.distributed.mnnvl_fused_allreduce import MnnvlAllReduceNorm
    from tensorrt_llm._torch.kernels.blackwell.moe_persistent_engine import new_state
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_moe import DecodeKernel

    ex = _experts()
    m = 56
    xq, xsf, ids, wts = _inputs(m, ex, seed=17)
    ref, valid = _stock(ex, xq, xsf, ids, wts)
    runner = _decode_runner()
    any8, bitwise8 = _tile8_tactics(runner, m)
    views, _ = _tables(ids, 8)
    e2p = views[0].view(m, TOPK)
    state = new_state(torch.device("cuda"))
    picked = _spy_select(monkeypatch)
    cases = [
        (any8, state, DecodeKernel.ENGINE),
        (any8, None, None),
        (bitwise8, None, DecodeKernel.CUTE_FC2),
    ]
    for tactic, engine_state, expect in cases:
        picked.clear()
        out = runner.forward(
            _op_inputs(ex, xq, xsf),
            tactic=tactic,
            prerouted=(views, ids, wts),
            engine_state=engine_state,
        )[0]
        torch.cuda.synchronize()
        assert len(picked) == 1, picked
        if expect is None:
            assert picked[0] != DecodeKernel.ENGINE, picked
        else:
            assert picked[0] == expect, (picked, expect)
        assert MnnvlAllReduceNorm.moe_fold_supported(out, wts.view(m, TOPK), e2p, H)
        got = out[e2p.flatten()[valid].long()]
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), (tactic, picked)
    picked.clear()
    _decode_runner(decode_tuning=False).forward(
        _op_inputs(ex, xq, xsf), tactic=any8, prerouted=(views, ids, wts), engine_state=state
    )
    assert not picked


@pytest.mark.parametrize(
    "e_local,inter",
    # Qwen3.5-397B-A17B experts (512, intermediate 1024, top-10) at EP4 / EP8, layouts that take the engine in
    # production (TP4/EP4, TEP8); the 56-row comparison needs a tile-8 tactic, which 32 local experts do not offer.
    [(128, 1024), (64, 1024)],
)
def test_engine_rows_match_the_runner_at_the_qwen35_layouts(monkeypatch, e_local, inter):
    """The runner-ABI engine / CuTe FC2 / cubin comparison of
    ``test_decode_kernels_select_the_engine_with_the_runner_abi`` at the production expert layouts, not only this
    file's 128 x 512 one."""
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "E_LOCAL", e_local)
    monkeypatch.setattr(module, "INTER", inter)
    test_decode_kernels_select_the_engine_with_the_runner_abi(monkeypatch)


def test_engine_leaves_the_reserved_sms_free(monkeypatch):
    """The runner hands the engine its ``decode_spare_sms`` (the backend's reserve for this call); the rows stay the
    stock runner's."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.kernels.blackwell.moe_persistent_engine import new_state
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import decode_moe

    ex = _experts()
    m = 56
    xq, xsf, ids, wts = _inputs(m, ex, seed=31)
    ref, valid = _stock(ex, xq, xsf, ids, wts)
    views, _ = _tables(ids, 8)
    e2p = views[0].view(m, TOPK)
    state = new_state(torch.device("cuda"))
    seen, orig = [], decode_moe.run_engine

    def spy(*args, **kwargs):
        seen.append(kwargs["spare_sms"])
        return orig(*args, **kwargs)

    monkeypatch.setattr(decode_moe, "run_engine", spy)
    for spare in (0, decode_moe.ENGINE_SPARE_SMS):
        runner = _decode_runner(decode_spare_sms=spare)
        any8, _ = _tile8_tactics(runner, m)
        seen.clear()
        out = runner.forward(
            _op_inputs(ex, xq, xsf), tactic=any8, prerouted=(views, ids, wts), engine_state=state
        )[0]
        torch.cuda.synchronize()
        assert seen == [spare], (spare, seen)
        got = out[e2p.flatten()[valid].long()]
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), spare


def test_engine_reserve_is_the_moe_sm_reserve_hook():
    """The NVFP4 TRTLLM-Gen leaf takes the shared expert's SMs through MoEImplBase's hooks: a ceiling of
    ENGINE_SPARE_SMS only where the model declares the decode kernels of a pre-routed call, and the per-forward
    reserve bounded by it (the model hands 0 for an in-order step)."""
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_moe import ENGINE_SPARE_SMS
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.fp4_block_scale import (
        TRTLLMGenFp4BlockScaleBase,
    )

    def leaf(decode_tuning, prerouted):
        obj = SimpleNamespace(decode_tuning=decode_tuning, supports_prerouted_routing=prerouted)
        obj.sm_reserve_ceiling = functools.partial(
            TRTLLMGenFp4BlockScaleBase.sm_reserve_ceiling, obj
        )
        obj.reserve = functools.partial(
            TRTLLMGenFp4BlockScaleBase.reserve_sms_for_concurrent_kernels, obj
        )
        return obj

    declared = leaf(True, True)
    assert declared.sm_reserve_ceiling() == ENGINE_SPARE_SMS
    assert declared.reserve(ENGINE_SPARE_SMS) == ENGINE_SPARE_SMS
    assert declared._concurrent_sm_reserve == ENGINE_SPARE_SMS
    assert declared.reserve(4 * ENGINE_SPARE_SMS) == ENGINE_SPARE_SMS
    assert declared.reserve(0) == 0 and declared._concurrent_sm_reserve == 0
    with pytest.raises(ValueError):
        declared.reserve(-1)
    for decode_tuning, prerouted in ((False, True), (True, False)):
        other = leaf(decode_tuning, prerouted)
        assert other.sm_reserve_ceiling() == 0
        assert other.reserve(ENGINE_SPARE_SMS) == 0


def test_engine_grid_leaves_the_global_sm_reserve(monkeypatch):
    """The persistent MoE grids also keep out the SMs ``cute_dsl_utils.sm_reserve()`` reserves (DWDP), on top of
    the per-call spare SMs."""
    import tensorrt_llm._torch.cute_dsl_utils as cdu
    from tensorrt_llm._torch.kernels.blackwell import moe_lowm_fc2

    sms = torch.cuda.get_device_properties(torch.device("cuda")).multi_processor_count
    assert moe_lowm_fc2.num_sms(torch.device("cuda")) == sms
    monkeypatch.setattr(cdu, "_sm_reserve", 4)
    assert moe_lowm_fc2.num_sms(torch.device("cuda")) == sms - 4


def test_lamport_out_takes_the_decode_kernels_rows(monkeypatch):
    """With a Lamport FC2 buffer the engine and the CuTe FC2 write their rows there, bit for bit the plain call's,
    leaving the rest of the buffer sentinel; a call of the cubins returns its own rows and leaves the buffer untouched
    (no copy). Through the custom op the Lamport rows come back as an empty placeholder (an op output may not alias
    an input) that the op backend replaces with the buffer."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.kernels.blackwell.moe_persistent_engine import new_state
    from tensorrt_llm._torch.moe.fused_moe.impl_contract import PreroutedRouting
    from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import TRTLLMOpBackend
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_moe import (
        DecodeKernel,
        LamportFc2Buffer,
    )
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    ex = _experts()
    dev = torch.device("cuda", torch.cuda.current_device())
    lamport = LamportFc2Buffer(LAMPORT_MAX_ROWS).get(H, INTER, TOPK, E_LOCAL, dev)
    words = lamport.view(torch.int16)
    assert (words == -1).all()
    state = new_state(dev)
    picked = _spy_select(monkeypatch)
    m = 56
    xq, xsf, ids, wts = _inputs(m, ex, seed=29)
    runner = _decode_runner()
    any8, bitwise8 = _tile8_tactics(runner, m)
    views, _ = _tables(ids, 8)
    rows = views[0][views[0] >= 0].long()
    rest = torch.ones(words.shape[0], dtype=torch.bool, device="cuda")
    rest[rows] = False
    cases = [
        (runner, any8, state, DecodeKernel.ENGINE),
        (runner, bitwise8, None, DecodeKernel.CUTE_FC2),
        (_decode_runner(decode_tuning=False), any8, state, None),
    ]
    for op_runner, tactic, engine_state, expect in cases:
        kw = dict(tactic=tactic, prerouted=(views, ids, wts), engine_state=engine_state)
        plain = op_runner.forward(_op_inputs(ex, xq, xsf), **kw)[0]
        picked.clear()
        out = op_runner.forward(_op_inputs(ex, xq, xsf), lamport_out=lamport, **kw)[0]
        torch.cuda.synchronize()
        assert picked == ([] if expect is None else [expect])
        assert torch.equal(out[rows].view(torch.int16), plain[rows].view(torch.int16))
        if expect is None:  # the cubins: their own rows, the buffer untouched
            assert out.data_ptr() != lamport.data_ptr() and (words == -1).all()
            continue
        assert out.data_ptr() == lamport.data_ptr()
        assert not (words[rows].view(torch.int32) == -1).any()
        assert (words[rest] == -1).all()
        words[rows] = -1  # the fold's write-back
    backend = TRTLLMOpBackend()

    def run(lamport_fc2_out, decode_tuning=True):
        return backend.run_fp4_block_scale_moe(
            None,
            None,
            xq,
            xsf,
            ex.w31,
            ex.w31s,
            None,
            None,
            None,
            None,
            ex.w2,
            ex.w2s,
            None,
            ex.scale_c,
            ex.alpha1,
            ex.alpha2,
            E,
            TOPK,
            None,
            None,
            INTER,
            0,
            E_LOCAL,
            None,
            1,
            do_finalize=False,
            topk_weights=None,
            topk_ids=None,
            gated_act_type=int(ActType_TrtllmGen.SwiGlu),
            prerouted=PreroutedRouting(torch.cat(views), any8, ids, wts),
            decode_tuning=decode_tuning,
            decode_engine_state=state,
            lamport_fc2_out=lamport_fc2_out,
        )

    plain = run(None)
    out = run(lamport)
    torch.cuda.synchronize()
    assert out[0] is lamport
    assert torch.equal(out[0][rows].view(torch.int16), plain[0][rows].view(torch.int16))
    words[rows] = -1
    out = run(lamport, decode_tuning=False)
    torch.cuda.synchronize()
    assert out[0] is not lamport and (words == -1).all()
    assert torch.equal(out[0][rows].view(torch.int16), plain[0][rows].view(torch.int16))


def test_lamport_buffer_holds_every_tile_the_runner_can_tune():
    """One Lamport buffer serves every Lamport call of the model: every token count up to its max rows at every tile
    the FP4 runner offers for it, which bounds the engine's rows and the FC1-only runner's gemm2 rows (at least
    128 KiB of them) the CuTe FC2 writes."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import decode_moe as dm
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_front import max_num_tiles
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    runner = torch.classes.trtllm.FP4BlockScaleMoERunner(int(ActType_TrtllmGen.SwiGlu))
    calls = []
    for m in range(1, LAMPORT_MAX_ROWS + 1):
        tiles = {int(t[0]) for t in runner.get_valid_configs(TOPK, H, INTER, E_LOCAL, m)}
        assert tiles
        calls += [max(max_num_tiles(m, TOPK, E_LOCAL, t) * t, 128 * 1024 // (H * 2)) for t in tiles]
    rows = dm.lamport_buffer_rows(runner, LAMPORT_MAX_ROWS, H, INTER, TOPK, E_LOCAL)
    assert max(calls) <= rows
    dev = torch.device("cuda", torch.cuda.current_device())
    owner = dm.LamportFc2Buffer(LAMPORT_MAX_ROWS)
    buf = owner.get(H, INTER, TOPK, E_LOCAL, dev)
    assert buf.shape == (rows, H) and buf.dtype == torch.bfloat16
    assert (buf.view(torch.int16) == -1).all()
    assert owner.get(H, INTER, TOPK, E_LOCAL, dev).data_ptr() == buf.data_ptr()


def test_fc1_intermediate_matches_identity_w2(state):
    """With W2[n][n] = 1 (scales 1, alpha2 1) the runner's gemm2 columns < I are the dequantized FC1 intermediate."""
    import tensorrt_llm  # noqa: F401

    ex = _experts(w2_identity=True)
    m = 224
    xq, xsf, ids, wts = _inputs(m, ex, seed=5)
    ref, valid = _stock(ex, xq, xsf, ids, wts)
    _, h, hsf, e2p = _engine(ex, xq, xsf, ids, 8, 64, state)
    rows = e2p[valid].long()
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device="cuda",
    )
    hb = h[rows].long()
    vals = torch.stack((lut[hb & 0xF], lut[hb >> 4]), dim=-1).flatten(-2)
    r = rows.view(-1, 1)
    b = torch.arange(INTER // 16, device="cuda").view(1, -1)
    sf = hsf.view(torch.float8_e4m3fn)[
        ((r // 8) * (INTER // 64) + b // 4) * 32 + (r % 8) * 4 + b % 4
    ].float()
    got = vals * sf.repeat_interleave(16, dim=1)
    assert torch.equal(got, ref[:, :INTER].float())


@pytest.mark.parametrize("inter", [256, 512, 768, 1024])
@pytest.mark.parametrize("m", [7, 16, 17, 224, 225, 896])
def test_engine_gate_matches_the_engine_shapes(m, inter):
    """``select``'s engine gate holds exactly where the engine compiles for the K tile ``run_engine`` passes (e.g.
    MoE TP4's intermediate 256 per rank takes K tiles of 256 at every row count)."""
    from tensorrt_llm._torch.kernels.blackwell.moe_nvfp4_engine import MoeNvfp4Engine
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import decode_moe as dm
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    def t(*shape):
        return torch.empty(shape, dtype=torch.uint8, device="meta")

    args = SimpleNamespace(
        hidden_states=t(m, H // 2),
        hidden_states_scale=t(m, H // 16),
        gemm1_weights=t(E_LOCAL, 2 * inter, H // 2),
        gemm2_weights=t(E_LOCAL, H, inter // 2),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        output1_scale_scalar=t(E_LOCAL),
        output1_scale_gate_scalar=t(E_LOCAL),
        output2_scale_scalar=t(E_LOCAL),
    )
    assert dm._engine_fits(args, ActType_TrtllmGen.SwiGlu, TOPK)
    n_max = dm.n_max_for(m)
    tile_k = dm.tile_k_for(n_max, H, inter)
    MoeNvfp4Engine(H, inter, n_max, 4, 8, tile_k, False)  # asserts the shapes it compiles for
    assert tile_k == (256 if inter % 512 or n_max > 32 else 512)


@pytest.mark.parametrize("m,top_k,fits", [(1024, TOPK, True), (1024, 32, False), (64, 32, True)])
def test_engine_gate_bounds_the_segment_table(m, top_k, fits):
    """The engine gate refuses a call whose worst-case (expert, row chunk) segment count exceeds ``SEG_CAP``."""
    from tensorrt_llm._torch.kernels.blackwell.moe_nvfp4_engine import SEG_CAP
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import decode_moe as dm
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    def t(*shape):
        return torch.empty(shape, dtype=torch.uint8, device="meta")

    args = SimpleNamespace(
        hidden_states=t(m, H // 2),
        hidden_states_scale=t(m, H // 16),
        gemm1_weights=t(E_LOCAL, 2 * INTER, H // 2),
        gemm2_weights=t(E_LOCAL, H, INTER // 2),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        output1_scale_scalar=t(E_LOCAL),
        output1_scale_gate_scalar=t(E_LOCAL),
        output2_scale_scalar=t(E_LOCAL),
    )
    assert (E_LOCAL + -(-m * top_k // dm.n_max_for(m)) <= SEG_CAP) == fits
    assert dm._engine_fits(args, ActType_TrtllmGen.SwiGlu, top_k) == fits


def test_fine_grained_sync_keeps_the_runner_fc2():
    """A call keeps the runner's FC2 only where the runner runs its fine-grained-sync pair (the FC1-only entry points
    refuse it), as the runner reports for the call's tile and shape; the FC2 cubin's options are asked once per
    (act type, tactic)."""
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import decode_moe as dm
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    options = dict(dm._CUTE_FC2_OPTIONS, tileK="256")
    asked, fg_calls = [], []

    def runner(fine_grained):
        def uses_fine_grained_sync(*shape):
            fg_calls.append(shape)
            return fine_grained

        return SimpleNamespace(
            get_gemm2_kernel_options=lambda tile, config: asked.append((tile, config)) or options,
            uses_fine_grained_sync=uses_fine_grained_sync,
        )

    args = SimpleNamespace(
        hidden_states=torch.empty((7, H // 2), dtype=torch.uint8, device="meta"),
        gemm2_weights=torch.empty((E_LOCAL, H, INTER // 2), dtype=torch.uint8, device="meta"),
        gemm2_weights_scale=torch.empty(
            (E_LOCAL, H, INTER // 16), dtype=torch.uint8, device="meta"
        ),
        gemm2_bias=None,
    )
    kw = dict(
        top_k=TOPK,
        do_finalize=False,
        output=None,
        intermediate_size=INTER,
        prerouted=False,
        engine_state=None,
    )
    act = int(ActType_TrtllmGen.SwiGlu)
    dm._FC2_REPRODUCED.clear()
    assert dm.cute_fc2_reproduces(options)
    assert dm.select(runner(False), act, [8, 0], args, **kw) == dm.DecodeKernel.CUTE_FC2
    assert dm.select(runner(True), act, [8, 0], args, **kw) == dm.DecodeKernel.CUBIN
    assert dm.select(runner(False), act, [8, 0], args, **kw) == dm.DecodeKernel.CUTE_FC2
    assert asked == [(8, 0)]
    # (tile, top_k, hidden, intermediate, local experts, tokens), as the runner's own call resolves it.
    assert fg_calls == [(8, TOPK, H, INTER, E_LOCAL, 7)] * 3


def test_fine_grained_sync_without_its_cubins_keeps_the_cute_fc2(monkeypatch):
    """TLLM_USE_FINE_GRAINED_SYNC=1 on SM 10.0 / 10.3, which ship no fine-grained cubins: the runner runs the tuned
    tactic, so the CuTe FC2 still stands in for its FC2 (and the FC1-only entry runs), bit for bit the stock rows."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_moe import DecodeKernel

    monkeypatch.setenv("TLLM_USE_FINE_GRAINED_SYNC", "1")
    ex = _experts()
    m = 56
    xq, xsf, ids, wts = _inputs(m, ex, seed=19)
    ref, valid = _stock(ex, xq, xsf, ids, wts)
    runner = _decode_runner()
    _, bitwise8 = _tile8_tactics(runner, m)
    assert not runner.get_runner().uses_fine_grained_sync(8, TOPK, H, INTER, E_LOCAL, m)
    views, _ = _tables(ids, 8)
    e2p = views[0].view(m, TOPK)
    picked = _spy_select(monkeypatch)
    out = runner.forward(
        _op_inputs(ex, xq, xsf), tactic=bitwise8, prerouted=(views, ids, wts), engine_state=None
    )[0]
    torch.cuda.synchronize()
    assert picked == [DecodeKernel.CUTE_FC2], picked
    got = out[e2p.flatten()[valid].long()]
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


def _router_inputs(m, ex, seed, dup=False):
    """x [m, H] bf16, a router weight [E, H] bf16 (``dup``: identical row pairs, exact logit ties) and the experts'
    NVFP4 global scale."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, H, device="cuda", generator=gen).to(torch.bfloat16)
    w = (torch.randn(E, H, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    if dup:
        w[1::2] = w[0::2]
    return x, w, torch.tensor([ex.x_scale], dtype=torch.float32, device="cuda")


def _poisoned_routing(m, tile, poison):
    """(flat tables, ids, weights) filled with ``poison``, the outputs of a routing engine."""
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_front import prerouted_table_sizes

    size = sum(prerouted_table_sizes(m, TOPK, E_LOCAL, tile))
    flat = torch.full((size,), poison, dtype=torch.int32, device="cuda")
    ids = torch.full((m, TOPK), poison, dtype=torch.int32, device="cuda")
    wts = torch.full((m, TOPK), poison, dtype=torch.int16, device="cuda").view(torch.bfloat16)
    return flat, ids, wts


def _engine_on(
    ex, xq, xsf, views, tile, state, poison, sync=True, release_after_wait=False, **route
):
    """The engine with the decode hook's n_max / K tile on tables ``views`` (``prerouted_table_views``); with
    ``route`` (candidates, ids, weights, local start) it writes them itself."""
    from tensorrt_llm._torch.kernels.blackwell.moe_nvfp4_engine import moe_nvfp4_engine
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_front import max_num_tiles
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_moe import n_max_for, tile_k_for

    m = xq.shape[0]
    e2p, p2t, batch, limit, nct, npad = views
    cap = max_num_tiles(m, TOPK, E_LOCAL, tile) * tile
    h = torch.full((cap, INTER // 2), poison & 0xFF, dtype=torch.uint8, device="cuda")
    hsf = torch.full((cap * INTER // 16,), poison & 0xFF, dtype=torch.uint8, device="cuda")
    out = torch.full((cap, H), poison, dtype=torch.int16, device="cuda").view(torch.bfloat16)
    if route:
        route = dict(route, e2p=e2p, num_padded=npad)
    n_max = n_max_for(m)
    moe_nvfp4_engine(
        xq,
        xsf,
        p2t,
        batch,
        limit,
        nct,
        ex.w31,
        ex.w31s.view(torch.uint8),
        ex.w2,
        ex.w2s.view(torch.uint8),
        ex.scale_c,
        ex.alpha1,
        ex.alpha2,
        h,
        hsf,
        out,
        state,
        tile=tile,
        n_max=n_max,
        tile_k=tile_k_for(n_max, H, INTER),
        release_after_wait=release_after_wait,
        **route,
    )
    if sync:
        torch.cuda.synchronize()
    return out


def _route_front_chain(ex, x, w, gs, ls, tile, state, poison):
    """Today's chain for the same step: the route front steps' CuTe router GEMM, the lean route front (tables in
    ``poison``-filled buffers), the engine on its tables. Returns (xq, xsf, ids, weights, table views, gemm2 rows)."""
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_front import (
        RouteFront,
        RouteFrontRouterGemm,
        prerouted_table_views,
    )

    m = x.shape[0]
    logits = RouteFrontRouterGemm()(x, w)
    front = RouteFront(lean=True)
    fn = front.compiled(x, E, tile, E_LOCAL, TOPK)
    assert logits is not None and fn is not None
    flat, _, _ = _poisoned_routing(m, tile, poison)
    xq, xsf, ids, wts = front(fn, logits, x, gs, flat, tile, ls, E_LOCAL, TOPK)
    views = prerouted_table_views(flat, m, TOPK, E_LOCAL, tile)
    return xq, xsf, ids, wts, views, _engine_on(ex, xq, xsf, views, tile, state, poison)


def _check_routed(got, ref, tag):
    """ids, weights, every table word (both started from the same poison) and the gemm2 rows of the real (token,
    expert) pairs, bitwise."""
    ids, wts, views, out = got
    r_ids, r_wts, r_views, r_out = ref
    assert torch.equal(ids, r_ids), (tag, "ids")
    assert torch.equal(wts.view(torch.int16), r_wts.view(torch.int16)), (tag, "weights")
    names = ("e2p", "p2t", "cta_batch", "cta_limit", "num_ctas", "num_padded")
    for name, a, b in zip(names, views, r_views):
        assert torch.equal(a, b), (tag, name)
    rows = r_views[0][r_views[0] >= 0].long()
    assert torch.equal(out[rows].view(torch.int16), r_out[rows].view(torch.int16)), (tag, "rows")


# The steps the engine routes (33..63 rows); local experts [ls, ls + 128) of 512; dup: exact logit ties.
@pytest.mark.parametrize(
    "m,ls,tile,dup",
    [
        (33, 0, 8, False),
        (35, 0, 8, False),
        (42, 0, 8, True),
        (49, 384, 8, False),
        (56, 0, 16, False),
        (63, 128, 8, False),
    ],
)
def test_engine_routes_bit_for_bit(m, ls, tile, dup, state):
    """``route``: the engine on the epilogue router GEMM's per-tile candidates writes the ids, weights, e2p, p2t,
    tile tables and counts of today's chain (router GEMM, lean route front) bit for bit and the same gemm2 rows;
    every output starts from poison, twice with different poison."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_front import (
        RouteFrontRouterGemm,
        prerouted_table_views,
    )

    ex = _experts()
    x, w, gs = _router_inputs(m, ex, seed=41 + m, dup=dup)
    _, cand, xq, xsf = RouteFrontRouterGemm(engine_route=True)(x, w, gs, TOPK)
    for poison in (-7, 0x2AAA):
        ref = _route_front_chain(ex, x, w, gs, ls, tile, state, poison)
        tag = (m, ls, tile, dup, poison)
        assert torch.equal(xq, ref[0]) and torch.equal(xsf, ref[1]), (tag, "x quant")
        flat, ids, wts = _poisoned_routing(m, tile, poison)
        views = prerouted_table_views(flat, m, TOPK, E_LOCAL, tile)
        out = _engine_on(
            ex,
            xq,
            xsf,
            views,
            tile,
            state,
            poison,
            candidates=cand,
            topk_ids=ids,
            topk_weights=wts,
            local_start=ls,
        )
        _check_routed((ids, wts, views, out), ref[2:], tag)


def _pdl_table_reader():
    """A PDL dependent that copies e2p (int32) and the BF16 weights (as u16) into ``out`` [n, 2] int32 as soon as it
    starts, without its dependency wait, with the loads of the MoE-output all-reduce's Lamport fold."""
    import cutlass.cute as cute
    from cutlass import Int32, Int64

    from tensorrt_llm._torch.kernels.blackwell.mnnvl_slot_allreduce_norm import (
        _ld_relaxed_s32,
        _ld_relaxed_u16,
    )
    from tensorrt_llm._torch.kernels.blackwell.ptx_helpers import _st_global_b32

    class Reader:
        @cute.jit
        def __call__(self, e2p: Int64, wts: Int64, out: Int64, n: Int32, stream) -> None:
            self.kernel(e2p, wts, out, n).launch(
                grid=[1, 1, 1], block=[1024, 1, 1], stream=stream, use_pdl=True
            )

        @cute.kernel
        def kernel(self, e2p: Int64, wts: Int64, out: Int64, n: Int32) -> None:
            tidx, _, _ = cute.arch.thread_idx()
            if tidx < n:
                o = out + Int64(tidx) * Int64(8)
                _st_global_b32(o, _ld_relaxed_s32(e2p + Int64(tidx) * Int64(4)))
                _st_global_b32(o + Int64(4), _ld_relaxed_u16(wts + Int64(tidx) * Int64(2)))

    return Reader()


@pytest.mark.parametrize("release_after_wait", [False, True])
@pytest.mark.parametrize("m", [35, 56, 63])
def test_engine_routes_before_releasing_its_dependents(m, release_after_wait, state):
    """``route``: a dependent launched right behind the engine (PDL, no dependency wait, as the Lamport fold) reads
    the e2p table and the weights the engine wrote, never the poison they started from, with few active experts (most
    CTAs have no FC1 unit and reach the release point first)."""
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass import Int32, Int64

    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_front import (
        RouteFrontRouterGemm,
        prerouted_table_views,
    )

    ex = _experts()
    x, w, gs = _router_inputs(m, ex, seed=71 + m)
    x = x.abs()
    w = torch.full_like(w, -0.02)
    w[: TOPK + 2] = 0.02  # every token routes to 10 of 12 experts
    _, cand, xq, xsf = RouteFrontRouterGemm(engine_route=True)(x, w, gs, TOPK)
    n = m * TOPK
    got = torch.empty((n, 2), dtype=torch.int32, device="cuda")
    reader = _pdl_table_reader()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = cute.compile(reader, Int64(0), Int64(0), Int64(0), Int32(0), stream)
    for it, poison in enumerate((-7, 0x2AAA) * 8):
        flat, ids, wts = _poisoned_routing(m, 8, poison)
        views = prerouted_table_views(flat, m, TOPK, E_LOCAL, 8)
        got.fill_(poison)
        torch.cuda.synchronize()
        _engine_on(
            ex, xq, xsf, views, 8, state, poison, sync=False,
            release_after_wait=release_after_wait,
            candidates=cand, topk_ids=ids, topk_weights=wts, local_start=0,
        )  # fmt: skip
        fn(
            Int64(views[0].data_ptr()),
            Int64(wts.data_ptr()),
            Int64(got.data_ptr()),
            Int32(n),
            stream,
        )
        torch.cuda.synchronize()
        assert torch.equal(got[:, 0], views[0].reshape(-1)[:n]), (m, release_after_wait, it, "e2p")
        assert torch.equal(got[:, 1], wts.view(torch.int16).reshape(-1).to(torch.int32) & 0xFFFF), (
            m,
            release_after_wait,
            it,
            "weights",
        )


def _ref_route(logits, top_k):
    """Host routing of BF16 ``logits`` for ``top_k``: ids by reduceTopK key (equal logits to the smaller expert) and
    the weights with the route front's arithmetic (exp of v - v0, warp-butterfly sum, divide, round to BF16)."""
    from tensorrt_llm._torch.kernels.blackwell.moe_bf16_engine import _butterfly_sum

    m, e = logits.shape
    bits = logits.view(torch.int16).to(torch.int64) & 0xFFFF
    tw = torch.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits | 0x8000)
    keys = (tw << 16) | (65535 - torch.arange(e, device=logits.device))
    top = keys.sort(dim=-1, descending=True).values[:, :top_k]
    ids = (65535 - (top & 0xFFFF)).int()
    vals = torch.gather(logits.float(), 1, ids.long())
    ex = torch.exp(vals - vals[:, :1])

    def tree_sum(t):
        return ex[:, t] if isinstance(t, int) else tree_sum(t[0]) + tree_sum(t[1])

    return ids, (ex / tree_sum(_butterfly_sum(top_k))[:, None]).to(torch.bfloat16)


@pytest.mark.parametrize("m,top_k,tile", [(42, 8, 8), (63, 4, 16)])
def test_engine_routes_another_top_k(m, top_k, tile, state):
    """The routing engine compiled for another top-k (4 lists of 128 experts): ids and tables equal to host
    routing (tables in token order), weights within one BF16 step of it, and the gemm2 rows bitwise those of the
    engine on the host tables."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_front import (
        RouteFrontRouterGemm,
        prerouted_table_sizes,
        prerouted_table_views,
    )
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_moe import n_max_for, tile_k_for

    ex = _experts()
    x, w, gs = _router_inputs(m, ex, seed=61 + m)
    logits, cand, xq, xsf = RouteFrontRouterGemm(engine_route=True)(x, w, gs, top_k)
    assert cand.shape == (m, E // 128, top_k)
    flat = torch.full((sum(prerouted_table_sizes(m, top_k, E_LOCAL, tile)),), -7, dtype=torch.int32)
    flat = flat.cuda()
    ids = torch.full((m, top_k), -7, dtype=torch.int32, device="cuda")
    wts = torch.zeros((m, top_k), dtype=torch.bfloat16, device="cuda")
    views = prerouted_table_views(flat, m, top_k, E_LOCAL, tile)
    out = _engine_on(
        ex,
        xq,
        xsf,
        views,
        tile,
        state,
        -7,
        candidates=cand,
        topk_ids=ids,
        topk_weights=wts,
        local_start=0,
    )
    r_ids, r_wts = _ref_route(logits, top_k)
    assert torch.equal(ids, r_ids)
    step = (wts.view(torch.int16).int() - r_wts.view(torch.int16).int()).abs().max().item()
    assert step <= 1, step
    r_views, _ = _tables(r_ids, tile)
    e2p, p2t, batch, limit, nct, npad = views
    r_e2p, r_p2t, r_batch, r_limit, r_nct, r_npad = r_views
    n = int(r_nct[0])
    assert torch.equal(e2p, r_e2p) and torch.equal(nct, r_nct) and torch.equal(npad, r_npad)
    assert torch.equal(batch[:n], r_batch[:n]) and torch.equal(limit[:n], r_limit[:n])
    rows = r_e2p[r_e2p >= 0].long()
    assert torch.equal(p2t[rows], r_p2t[rows])
    n_max = n_max_for(m)
    ref, _, _, _ = _engine(
        ex, xq, xsf, r_ids, tile, n_max, state, tile_k=tile_k_for(n_max, H, INTER)
    )
    assert torch.equal(out[rows].view(torch.int16), ref[rows].view(torch.int16))


def test_engine_route_bounds():
    """``route_error``: whole 16-byte candidate vectors, at most two per thread for 64 tokens."""
    from tensorrt_llm._torch.kernels.blackwell.moe_nvfp4_engine import route_error

    assert route_error(10, 4) is None and route_error(8, 4) is None and route_error(12, 4) is None
    assert route_error(13, 4) is not None  # 64 x 52 keys
    assert route_error(10, 1) is not None and route_error(0, 4) is not None
    assert route_error(36, 1) is not None  # the weights' warp sum


def test_runner_routes_in_the_engine(monkeypatch):
    """A pre-routed un-finalized call with ``candidates`` takes the engine, which writes the routing today's chain
    would, also through the op backend (``EngineRouting``); a call that would not take the engine
    refuses them."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.kernels.blackwell.moe_persistent_engine import new_state
    from tensorrt_llm._torch.moe.fused_moe.impl_contract import EngineRouting
    from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import TRTLLMOpBackend
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_front import (
        RouteFrontRouterGemm,
        prerouted_table_views,
    )
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.decode_moe import DecodeKernel
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    ex = _experts()
    m, tile = 42, 8
    state = new_state(torch.device("cuda"))
    x, w, gs = _router_inputs(m, ex, seed=43)
    ref = _route_front_chain(ex, x, w, gs, 0, tile, state, -7)[2:]
    _, cand, xq, xsf = RouteFrontRouterGemm(engine_route=True)(x, w, gs, TOPK)
    runner = _decode_runner()
    any8, _ = _tile8_tactics(runner, m)
    picked = _spy_select(monkeypatch)
    flat, ids, wts = _poisoned_routing(m, tile, -7)
    views = prerouted_table_views(flat, m, TOPK, E_LOCAL, tile)
    kw = dict(tactic=any8, prerouted=(views, ids, wts), candidates=cand)
    out = runner.forward(_op_inputs(ex, xq, xsf), engine_state=state, **kw)[0]
    torch.cuda.synchronize()
    assert picked == [DecodeKernel.ENGINE], picked
    _check_routed((ids, wts, views, out), ref, "runner")
    with pytest.raises(ValueError, match="candidates"):
        runner.forward(_op_inputs(ex, xq, xsf), engine_state=None, **kw)
    flat, ids, wts = _poisoned_routing(m, tile, -7)
    rows, weights, e2p = TRTLLMOpBackend().run_fp4_block_scale_moe(
        None,
        None,
        xq,
        xsf,
        ex.w31,
        ex.w31s,
        None,
        None,
        None,
        None,
        ex.w2,
        ex.w2s,
        None,
        ex.scale_c,
        ex.alpha1,
        ex.alpha2,
        E,
        TOPK,
        None,
        None,
        INTER,
        0,
        E_LOCAL,
        None,
        1,
        do_finalize=False,
        topk_weights=None,
        topk_ids=None,
        gated_act_type=int(ActType_TrtllmGen.SwiGlu),
        prerouted=EngineRouting(flat, any8, ids, wts, cand),
        decode_tuning=True,
        decode_engine_state=state,
    )
    torch.cuda.synchronize()
    views = prerouted_table_views(flat, m, TOPK, E_LOCAL, tile)
    _check_routed((ids, wts, views, rows), ref, "op backend")
    assert weights is wts and torch.equal(e2p.flatten(), views[0])


def test_ring_budget_keeps_the_tuned_stage_counts():
    """On the SM 10.x parts the ring takes 200 KiB of the 227 KiB opt-in shared memory, the budget whose stage counts
    (NVFP4 engine: 4 at N 16 / 32 with K tile 512, 7 at N 64 with 256) the engines were measured with."""
    from tensorrt_llm._torch.kernels.blackwell.moe_bf16_engine import stages_for
    from tensorrt_llm._torch.kernels.blackwell.moe_persistent_engine import ring_smem_bytes

    budget = ring_smem_bytes(torch.device("cuda"))
    assert budget == 200 * 1024
    per_stage = {
        (n, k): (128 + n) * k // 2 + 2 * 128 * k // 16 for n in (16, 32, 64) for k in (256, 512)
    }
    assert [min(10, budget // per_stage[nk]) for nk in ((16, 512), (32, 512), (64, 256))] == [
        4,
        4,
        7,
    ]
    assert [stages_for(n, budget) for n in (16, 32)] == [5, 5]
