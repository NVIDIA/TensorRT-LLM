# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QKV_FRONT (the sm_107 tp4 target's ``ops/qkv_front``, native ``minimax_m31_qkv_front``): at split 8 the
one-launch front is bit for bit the served TP4 chain it replaces, on one GPU with a world-4 exchange (uc = mc):

    the MoE boundary m3_mc_allreduce_norm (cluster kernel, oneshot order, prepushed 2 / 3, MXFP8 copy)
    -> m3_mx_gemv (8, 8) -> m3_producer (producerPw4, prewait step inputs)

on the updated residual, the normed rows, q, index-q, every NVFP4 K / V byte and scale and every FP8 index-K byte;
over consecutive calls (both hand-off parities, the exchange re-armed by the kernel), eager and inside a CUDA graph
replayed on fresh inputs; with large / tiny / signed-zero activations, skipped (-1) and out-of-pool slots. Split 4 (the
peer geometry) is exact-class: its outputs are finite and within one BF16 step of the split-8 QKV on most values.
Plus the TP4 wiring: the switch and the deferral's source order."""

import inspect

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (
    native_ops,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.ops import (
    qkv_front,
)

H, HD, NQ, WORLD, MAXT, PAGES, MAX_POS = 6144, 128, 16, 4, 8, 64, 4096
EMPTY = -(2**31)


def _sm107():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


needs_gpu = pytest.mark.skipif(
    not (_sm107() and qkv_front.op_available()),
    reason="needs an SM 10.7 GPU and a build with the tp4 target's minimax_m31_qkv_front",
)


def _served():
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels import (  # noqa: E501
        m3_mc_allreduce as mca,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels import (  # noqa: E501
        m3_prequant as prequant,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.ops.attention.m3_producer import (  # noqa: E501
        m3_producer as prod,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.ops.gemm.m3_mx_gemv import (  # noqa: E501
        m3_mx_gemv as gemv,
    )

    return mca, prequant, gemv, prod


def _exchange(mca):
    words = 4 * MAXT * WORLD * (H // 2)
    uc = torch.full((words,), EMPTY, dtype=torch.int32, device="cuda")
    ws = mca.McAllReduceWorkspace(
        uc=uc,
        mc=uc,
        flags={},
        rank=0,
        world_size=WORLD,
        hidden=H,
        handle=None,
        comm=None,
        max_tokens=MAXT,
    )
    return ws, ws.flags_for(8)


def _canon_words(x):
    """bf16 [rows, H] -> the int32 exchange words (bf16 pairs, -0 halves stored as +0)."""
    h = x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    h = torch.where(h == 0x8000, torch.zeros_like(h), h)
    lo, hi = h[:, 0::2], h[:, 1::2]
    return (lo | (hi << 16)).to(torch.int32)


def _push(uc, flags, parts, set_b=None):
    """Write every rank's partial (parts [WORLD, rows, H]) into set A (and set_b into set B) at each (row, slice)
    parity, as the MoE engine's push does."""
    rows = parts.shape[1]
    half = MAXT * WORLD * (H // 2)
    par = (flags.view(MAXT, 8)[:rows] & 1).cpu()
    for base, src in ((0, parts), (2 * half, set_b)):
        if src is None:
            continue
        for r in range(WORLD):
            words = _canon_words(src[r])
            for t in range(rows):
                for sl in range(8):
                    off = base + int(par[t, sl]) * half + (t * WORLD + r) * (H // 2) + sl * 384
                    uc[off : off + 384] = words[t, sl * 384 : (sl + 1) * 384]


class _Layer:
    def __init__(self, gen, rows):
        x = (torch.randn(NQ * HD + 4 * HD, H, device="cuda", generator=gen) * H**-0.5).to(
            torch.bfloat16
        )
        wq, wsf = torch.ops.trtllm.mxfp8_quantize(x, True)
        self.w = wq.view(torch.float8_e4m3fn).view(NQ * HD + 4 * HD, H)
        self.ws = wsf.view(torch.uint8).reshape(-1)
        self.nw = (1.0 + 0.1 * torch.randn(H, device="cuda", generator=gen)).to(torch.bfloat16)
        self.norms = [
            (torch.randn(HD, device="cuda", generator=gen) * 0.3).bfloat16() for _ in range(4)
        ]
        self.rope = torch.randn(MAX_POS, 2, 32, device="cuda", generator=gen)
        self.kvq = torch.tensor((1.0, 0.37, 0.61), device="cuda")

    def caches(self):
        return (
            torch.zeros(PAGES, 2, 1, 128, 64, dtype=torch.uint8, device="cuda"),
            torch.zeros(PAGES, 2, 1, 128, 8, dtype=torch.uint8, device="cuda"),
            torch.zeros(PAGES, 1, 128, 128, dtype=torch.float8_e4m3fn, device="cuda"),
        )


def _inputs(gen, rows, scale, prepushed):
    parts = (scale * torch.randn(WORLD, rows, H, device="cuda", generator=gen)).to(torch.bfloat16)
    parts[0, 0, :7] = -0.0
    parts[:, 0, 7] = -0.0  # a column whose rank sum is -0
    set_b = None
    if prepushed == 3:
        set_b = (scale * torch.randn(WORLD, rows, H, device="cuda", generator=gen)).to(
            torch.bfloat16
        )
    res = (torch.randn(rows, H, device="cuda", generator=gen) * 2.0).to(torch.bfloat16)
    slots = torch.randperm(PAGES * 128, device="cuda", generator=gen)[:rows].to(torch.int32)
    if rows > 1:
        slots[1] = -1  # skipped
    if rows > 2:
        slots[2] = PAGES * 128 + 5  # past the pool
    pos = torch.randint(0, MAX_POS, (rows,), device="cuda", generator=gen, dtype=torch.int32)
    return parts, set_b, res, slots, pos


def _served_call(S, L, ex, caches, inp, prepushed):
    mca, prequant, gemv, prod = S
    ws, flags = ex
    parts, set_b, res, slots, pos = inp
    rows = res.shape[0]
    dummy = torch.zeros(rows, H, dtype=torch.bfloat16, device="cuda")
    normed, upd = mca.m3_mc_allreduce_norm(
        dummy,
        res,
        L.nw,
        1e-6,
        ws,
        use_pdl=True,
        cluster=8,
        x2=dummy if prepushed == 3 else None,
        quant=True,
        prepushed=prepushed,
        oneshot_cluster=True,
        carveout=True,
    )
    xq, xsf = prequant.SLOT.take(normed)
    y = gemv(xq, xsf, L.w, L.ws, True, (8, 8), True, None)
    kv, ks, ik = caches
    q, iq = prod(y, kv, ks, ik, slots, L.kvq, 1e-6, *L.norms, L.rope, pos, NQ, True, True, True)
    return [upd, normed, q, iq, y]


def _front_call(L, ex, fws, caches, inp, prepushed, split):
    ws, flags = ex
    parts, set_b, res, slots, pos = inp
    out_res, out_normed = torch.empty_like(res), torch.empty_like(res)
    kv, ks, ik = caches
    q, iq = qkv_front.qkv_front(
        res,
        L.nw,
        1e-6,
        out_res,
        out_normed,
        (ws.uc, flags, 0, WORLD, MAXT),
        prepushed,
        L.w,
        L.ws,
        fws,
        kv,
        ks,
        ik,
        slots,
        L.kvq,
        1e-6,
        *L.norms,
        L.rope,
        pos,
        NQ,
        split=split,
    )
    return [out_res, out_normed, q, iq]


def _bits_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@needs_gpu
@pytest.mark.parametrize("rows", [1, 4, 8])
@pytest.mark.parametrize("prepushed", [2, 3])
@pytest.mark.parametrize("scale", [1.0, 40.0, 1e-3])
def test_split8_is_bitwise_the_served_chain(rows, prepushed, scale):
    S = _served()
    gen = torch.Generator(device="cuda").manual_seed(1000 * rows + 10 * prepushed + int(scale > 1))
    L = _Layer(gen, rows)
    ex_s, ex_f = _exchange(S[0]), _exchange(S[0])
    fws = qkv_front.FrontWorkspace(torch.device("cuda"))
    c_s, c_f = L.caches(), L.caches()
    for call in range(3):  # both hand-off parities and the re-armed exchange
        inp = _inputs(gen, rows, scale, prepushed)
        _push(ex_s[0].uc, ex_s[1], inp[0], inp[1])
        _push(ex_f[0].uc, ex_f[1], inp[0], inp[1])
        ref = _served_call(S, L, ex_s, c_s, inp, prepushed)
        got = _front_call(L, ex_f, fws, c_f, inp, prepushed, 8)
        torch.cuda.synchronize()
        for name, a, b in zip(("residual", "normed", "q", "index_q"), got, ref[:4]):
            assert _bits_equal(a, b), f"call {call}: {name} differs"
        for name, a, b in zip(("kv_data", "kv_scale", "index_k"), c_f, c_s):
            assert _bits_equal(a, b), f"call {call}: {name} differs"
        assert torch.equal(ex_s[1], ex_f[1]), "the exchange flags rotate as the served boundary's"
        assert torch.equal(ex_s[0].uc, ex_f[0].uc), (
            "the exchange is re-armed as the served boundary's"
        )


@needs_gpu
def test_split8_in_a_cuda_graph_on_fresh_inputs():
    S = _served()
    rows, prepushed = 4, 2
    gen = torch.Generator(device="cuda").manual_seed(77)
    L = _Layer(gen, rows)
    ex_s, ex_f = _exchange(S[0]), _exchange(S[0])
    fws = qkv_front.FrontWorkspace(torch.device("cuda"))
    c_s, c_f = L.caches(), L.caches()
    inp = list(_inputs(gen, rows, 1.0, prepushed))
    # one eager call each (the kernels' attributes, the GEMV compile), then capture the front alone
    for ex in (ex_s, ex_f):
        _push(ex[0].uc, ex[1], inp[0])
    _served_call(S, L, ex_s, c_s, inp, prepushed)
    _front_call(L, ex_f, fws, c_f, inp, prepushed, 8)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.cuda.graph(g, stream=side):
        got = _front_call(L, ex_f, fws, c_f, inp, prepushed, 8)
    torch.cuda.synchronize()
    for _ in range(3):
        fresh = _inputs(gen, rows, 1.0, prepushed)
        for k in (0, 2, 3, 4):
            inp[k].copy_(fresh[k])
        for ex in (ex_s, ex_f):
            _push(ex[0].uc, ex[1], inp[0])
        ref = _served_call(S, L, ex_s, c_s, inp, prepushed)
        g.replay()
        torch.cuda.synchronize()
        for name, a, b in zip(("residual", "normed", "q", "index_q"), got, ref[:4]):
            assert _bits_equal(a, b), f"graph replay: {name} differs"
        for name, a, b in zip(("kv_data", "kv_scale", "index_k"), c_f, c_s):
            assert _bits_equal(a, b), f"graph replay: {name} differs"


@needs_gpu
def test_split4_is_exact_class():
    S = _served()
    rows, prepushed = 4, 2
    gen = torch.Generator(device="cuda").manual_seed(5)
    L = _Layer(gen, rows)
    ex8, ex4 = _exchange(S[0]), _exchange(S[0])
    f8 = qkv_front.FrontWorkspace(torch.device("cuda"))
    f4 = qkv_front.FrontWorkspace(torch.device("cuda"))
    c8, c4 = L.caches(), L.caches()
    inp = _inputs(gen, rows, 1.0, prepushed)
    _push(ex8[0].uc, ex8[1], inp[0])
    _push(ex4[0].uc, ex4[1], inp[0])
    a = _front_call(L, ex8, f8, c8, inp, prepushed, 8)
    b = _front_call(L, ex4, f4, c4, inp, prepushed, 4)
    torch.cuda.synchronize()
    assert _bits_equal(a[0], b[0]) and _bits_equal(a[1], b[1]), (
        "the boundary does not depend on the split"
    )
    assert torch.isfinite(b[2].float()).all() and torch.isfinite(b[3].float()).all()
    same = (a[2].view(torch.uint8) == b[2].view(torch.uint8)).float().mean()
    assert same > 0.9, f"only {float(same):.3f} of the q bytes agree with split 8"


@needs_gpu
def test_every_cluster_of_the_grid_is_co_resident():
    # the GEMV clusters wait on the boundary clusters' hand-off: the device must hold the whole grid at once (the
    # launcher refuses otherwise, and the TP4 deferral checks fits() first)
    for split in (8, 4):
        n = native_ops.minimax_m31_qkv_front_max_clusters(split)
        assert n >= NQ + 4, f"split {split}: the device holds {n} clusters, the grid has {NQ + 4}"
    assert qkv_front.fits(torch.device("cuda"), NQ, 8)


def test_target_wiring():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (
        m31_switches as switches,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (
        modeling,
    )

    assert modeling.QKV_FRONT == (not switches.m31_disabled("QKV_FRONT"))
    assert modeling.QKV_FRONT_SPLIT == qkv_front.SPLIT_BITWISE == 8
    target = modeling.DecodeTarget
    attention = inspect.getsource(target._attention)
    assert attention.index("self._front_pending is not None") < attention.index(
        "i not in core.sparse_layers"
    )
    layer = inspect.getsource(target._layer)
    for mode in (2, 3):
        assert (
            layer.count(
                f"i, residual, norm_w, quant, {mode}, capture, defer_front, position_ids, step"
            )
            == 1
        )
    forward = inspect.getsource(target.forward)
    assert "defer_front=defer_front" in forward and "never ran" in forward
    defer = inspect.getsource(target._defer_front)
    assert "capture is not None" in defer and "QKV_FRONT_SPLIT, 8" in defer
    assert defer.index("m31_qkv_front.fits(") < defer.index("self._front_pending = dict(")
