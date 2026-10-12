# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the trtllm_sparse_mla_dsv4_fp8 entry.

The entry owns what is certified: `CELLS` (decode and DSpark verify, varlen prefill, the scale pairs and
the per-query compressed buffer) is the list, and `reference` and `compare` are the gate. This file turns
a cell's spec into tensors and keeps the checks a cell cannot carry: the device-tensor `bmm1_scale`,
CUDA-graph capture and replay, and the wrong-reading controls.

The call computes with E4M3 operands, so the reference (`op.reference`) is taken on the SAME E4M3
values: the codes are exact in fp32, gathered per index row, fp32 scores scaled by bmm1_scale (=
softmax_scale * q_scale * kv_scale), fp32 max and sum, fp32 probabilities, the per-head sink added to the
denominator only (natural-log units), and the output scaled by bmm2_scale (= kv_scale). What the kernel
adds on top is its own arithmetic: it quantizes the probabilities to E4M3 (scaled by 448) for the PV MMA
and accumulates in its own order.

Gate (per query row and head): ||o - ref||_2 <= REL * ||ref||_2 + ABS * sqrt(512) * max|V|, with
REL = 2^-4 -- the largest relative rounding of one E4M3 probability, which is the whole error of a row
whose softmax sits on one key -- and ABS = 2^-7 for rows whose output nearly cancels. The discrimination
is measured in test_wrong_variants_fail_the_gate: the correct reference and each deliberately wrong
reading are reported as a multiple of the gate (sm_103: correct <= 0.45, wrong 12-58x). The wrong
readings are the op's own reference on altered arguments (no sink, sink x log2(e), bmm1 without
q_scale, kv_scale twice in bmm2, lengths cut to the window, every window count 128).

The pools are the native-storage target's compact read buffers, now E4M3: the window buffer
[max_num_tokens * 128, 512] viewed [max_num_tokens, 1, 128, 512] (query t's column c is row t*128+c)
and the compressed buffer [max_batch_size * 132 * 128 / r, 512] viewed [max_batch_size * 132, 1,
128 / r, 512] (request b's entry e is row b * L_r + e), at max_num_tokens 8192 and max_batch_size 32.
"""

import math

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.attention.trtllm_sparse_mla_dsv4_fp8 import (  # noqa: E501
    trtllm_sparse_mla_dsv4_fp8 as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "trtllm_sparse_mla_dsv4_fp8 requires a CUDA device"

DEV = torch.device("cuda")
D = 512
WIN = 128
SCALE = D**-0.5
E4 = torch.float8_e4m3fn
MAX_T = 8192
MBS = 32
COLS = 132
REL = 2.0**-4
ABS = 2.0**-7
WS = None


def _ws():
    global WS
    if WS is None:
        WS = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    return WS


def _gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def _sinks(H):
    return torch.linspace(-2.0, 9.0, H, device=DEV, dtype=torch.float32)


class Buffers:
    """Persistent E4M3 compact buffers (allocated once, like the target's)."""

    def __init__(self):
        self.win = torch.zeros(MAX_T * WIN, D, dtype=E4, device=DEV)
        self.comp = torch.zeros(MBS * COLS * WIN, D, dtype=E4, device=DEV)
        self.dummy = torch.zeros(1, 1, 64, D, dtype=E4, device=DEV)

    def win4(self):
        return self.win.view(MAX_T, 1, WIN, D)

    def comp4(self, r):
        return self.comp[: MBS * COLS * WIN // r].view(MBS * COLS, 1, WIN // r, D)


BUF = None


def _buf():
    global BUF
    if BUF is None:
        BUF = Buffers()
    return BUF


def _fill_e4m3(dst, gen, lo=-8, hi=8):
    """Random E4M3 rows with per-row magnitudes over 2^lo..2^hi code units (the transcoded KV spans
    about 2^-9..2^8 at the target's scale)."""
    n = dst.shape[0]
    mag = torch.randint(lo, hi + 1, (n, 1), device=DEV, generator=gen).float()
    v = torch.randn(n, D, device=DEV, generator=gen) * torch.exp2(mag) * 8
    dst.copy_(v.clamp(-448, 448).to(E4))


def _step(reqs, H, ratio, gen, n_comp_valid=None):
    """Build one step: reqs = [(cached, q_len)]; returns inputs for the call and for the reference.
    Window columns of query t name rows t*128 + c; rows past the window count hold finite decoys."""
    b = _buf()
    T = sum(q for _, q in reqs)
    width = WIN + (512 if ratio else 0)
    q = torch.randn(T, H, D, device=DEV, generator=gen) * 4
    q8 = q.clamp(-448, 448).to(E4)
    _fill_e4m3(b.win[: T * WIN], gen)
    idx = torch.full((T, width), -1, dtype=torch.int32, device=DEV)
    idx[:, :WIN] = torch.arange(T * WIN, dtype=torch.int32, device=DEV).view(T, WIN)
    lens = torch.full((T,), WIN, dtype=torch.int32, device=DEV)
    wc = torch.empty(T, dtype=torch.int64, device=DEV)
    L_r = COLS * WIN // max(ratio, 1)
    t = 0
    for bi, (cached, ql) in enumerate(reqs):
        pos = cached + torch.arange(ql, device=DEV)
        wc[t : t + ql] = (pos + 1).clamp_max(WIN)
        if ratio:
            n_c = (pos + 1) // ratio  # compressed entries visible to each query
            cap = 512 if n_comp_valid is None else n_comp_valid
            k = n_c.clamp_max(cap)
            n_max = int(n_c.max())
            for c0 in range(0, ql, 1024):
                c1 = min(ql, c0 + 1024)
                if n_max == 0:
                    break
                # a random subset of size k of 0..n_c-1 per row, sorted, -1 after it
                sc = torch.rand(c1 - c0, n_max, device=DEV, generator=gen)
                sc = torch.where(torch.arange(n_max, device=DEV)[None] < n_c[c0:c1, None], sc, -1.0)
                kk = min(n_max, cap)
                top = sc.topk(kk, dim=1).indices.sort(dim=1).values
                # rows with n_c < kk: keep only their first k valid entries (all of them, in order)
                keep = torch.arange(kk, device=DEV)[None] < k[c0:c1, None]
                top = torch.where(keep, top, torch.full_like(top, 1 << 30)).sort(dim=1).values
                sel = torch.where(top < (1 << 30), bi * L_r + top, -1).int()
                idx[t + c0 : t + c1, WIN : WIN + kk] = sel
            lens[t : t + ql] = (WIN + k).int()
        t += ql
    if ratio:
        rows = b.comp4(ratio).view(-1, D)
        _fill_e4m3(rows[: len(reqs) * L_r], gen)
    seq = torch.tensor([c + q for c, q in reqs], dtype=torch.int32, device=DEV)
    cu = torch.tensor(
        [0] + list(torch.tensor([q for _, q in reqs]).cumsum(0)), dtype=torch.int32, device=DEV
    )
    return dict(
        q8=q8,
        idx=idx,
        lens=lens,
        wc=wc,
        seq=seq,
        cu=cu,
        T=T,
        width=width,
        ratio=ratio,
        max_q=max(q for _, q in reqs),
        reqs=reqs,
    )


def _pools(st):
    b = _buf()
    comp = b.comp4(st["ratio"]) if st["ratio"] else b.dummy
    return b.win4(), comp


def _reference(
    st,
    sq,
    skv,
    sinks,
    win_count=None,
    drop_comp=False,
    sink_mode="ok",
    bmm2=None,
    pools=None,
):
    """`op.reference` on a step's arguments; returns [T, H, D] fp32.

    The keywords are the controls' wrong readings, each expressed through the op's own arguments:
    `win_count` gives every row its own q_len-1 request with that virtual seq_lens, `drop_comp` cuts
    every length to the window, `sink_mode` "none" / "log2e" passes no sink / sinks x log2(e), and
    `bmm2` replaces bmm2_scale. bmm1_scale is softmax_scale * sq * skv.
    """
    swa, comp = _pools(st) if pools is None else pools
    T = st["T"]
    if win_count is None:
        seq, cu = st["seq"], st["cu"]
    else:
        seq = win_count.to(torch.int32)
        cu = torch.arange(T + 1, dtype=torch.int32, device=DEV)
    lens = torch.full_like(st["lens"], WIN) if drop_comp else st["lens"]
    sk = {"ok": sinks, "none": None, "log2e": sinks * 1.4426950408889634}[sink_mode]
    expected, _ = op.reference(
        st["q8"],
        swa,
        comp,
        None,
        st["idx"],
        lens,
        seq,
        sk,
        SCALE * sq * skv,
        skv if bmm2 is None else bmm2,
        cum_seq_lens_q=cu,
        max_q_len=st["max_q"],
    )
    return expected


def _gate_ratio(o, ref, vmax):
    err = (o.float() - ref).norm(dim=-1)
    tol = REL * ref.norm(dim=-1) + ABS * math.sqrt(D) * vmax
    return (err / tol).max().item()


def _vmax(st, skv, pools=None):
    if pools is not None:
        return max(p.float().abs().max().item() for p in pools) * skv
    swa, comp = _pools(st)
    T = st["T"]
    m = swa.reshape(-1, D)[: T * WIN].float().abs().max().item()
    if st["ratio"]:
        m = max(
            m,
            comp.reshape(-1, D)[: len(st["reqs"]) * COLS * WIN // st["ratio"]]
            .float()
            .abs()
            .max()
            .item(),
        )
    return m * skv


def _call_decode(st, H, bmm1, skv, out=None):
    swa, comp = _pools(st)
    G = len(st["reqs"])
    nn = st["reqs"][0][1]
    return op(
        st["q8"].view(G, nn, H, D),
        swa,
        comp,
        _ws(),
        st["idx"],
        st["lens"],
        st["seq"],
        _sinks(H),
        bmm1,
        skv,
        out=out,
    )


def _call_varlen(st, H, bmm1, skv):
    swa, comp = _pools(st)
    return op(
        st["q8"],
        swa,
        comp,
        _ws(),
        st["idx"],
        st["lens"],
        st["seq"],
        _sinks(H),
        bmm1,
        skv,
        cum_seq_lens_q=st["cu"],
        max_q_len=st["max_q"],
    )


SQ, SKV = 2.0**-3, 2.0**-6


def _bmm1(tensor=False):
    v = SCALE * SQ * SKV
    return torch.tensor([v], dtype=torch.float32, device=DEV) if tensor else v


@pytest.mark.parametrize("H", [16, 64])
def test_tensor_bmm1_equals_float_bmm1_bitwise(H):
    gen = _gen(7 + H)
    st = _step([(100, 6), (5000, 6), (9, 6)], H, 2, gen)
    a = _call_decode(st, H, _bmm1(False), SKV)
    b = _call_decode(st, H, _bmm1(True), SKV)
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("next_n", [1, 6])
@pytest.mark.parametrize("H", [16, 64])
def test_cuda_graph_capture_replay(H, next_n):
    """Decode capture at the target's capture batches with a device bmm1 scale; each replay equals a
    fresh eager call bitwise and passes the reference."""
    for G in (1, 2, 4, 8, 16, 24, 32):
        gen = _gen(G + 50 * next_n + H)
        st = _step(
            [(int(c), next_n) for c in torch.randint(0, 16000, (G,), generator=gen, device=DEV)],
            H,
            2,
            gen,
        )
        bmm1 = _bmm1(True)
        swa, comp = _pools(st)
        q = st["q8"].view(G, next_n, H, D).clone()
        idx, lens, seq = st["idx"].clone(), st["lens"].clone(), st["seq"].clone()
        sinks = _sinks(H)

        def call():
            return op(q, swa, comp, _ws(), idx, lens, seq, sinks, bmm1, SKV)

        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = call()
        for rep in range(3):
            gen2 = _gen(1000 + rep + G)
            st2 = _step(
                [
                    (int(c), next_n)
                    for c in torch.randint(0, 16000, (G,), generator=gen2, device=DEV)
                ],
                H,
                2,
                gen2,
            )
            q.copy_(st2["q8"].view(G, next_n, H, D))
            idx.copy_(st2["idx"])
            lens.copy_(st2["lens"])
            seq.copy_(st2["seq"])
            bmm1.fill_(SCALE * SQ * SKV * (1.0 if rep != 1 else 0.5))
            graph.replay()
            eager = call()
            assert torch.equal(out.view(torch.int16), eager.view(torch.int16))
            sq = SQ * (1.0 if rep != 1 else 0.5)
            ref = _reference(st2, sq, SKV, sinks)
            op.compare(out.view(st2["T"], H, D), (ref, _vmax(st2, SKV)))


def test_wrong_variants_fail_the_gate():
    """Discrimination: the correct reference vs deliberately wrong readings, as multiples of the gate."""
    H = 16
    gen = _gen(99)
    st = _step([(3000, 1), (50, 1), (16000, 1), (0, 300), (9000, 64)], H, 2, gen)
    o = _call_varlen(st, H, _bmm1(), SKV)
    vmax = _vmax(st, SKV)
    sinks = _sinks(H)
    correct = _reference(st, SQ, SKV, sinks)
    good = _gate_ratio(o, correct, vmax)
    wrong = {
        "sink ignored": _reference(st, SQ, SKV, sinks, sink_mode="none"),
        "sink x log2e": _reference(st, SQ, SKV, sinks, sink_mode="log2e"),
        "q scale ignored (bmm1 without q_scale)": _reference(st, 1.0, SKV, sinks),
        "kv scale twice in bmm2": _reference(st, SQ, SKV, sinks, bmm2=SKV * SKV),
        "compressed segment dropped": _reference(st, SQ, SKV, sinks, drop_comp=True),
        "window count 128 for every row": _reference(
            st, SQ, SKV, sinks, win_count=torch.full_like(st["wc"], WIN)
        ),
    }
    print(f"correct reference: {good:.3f} of the gate")
    for name, ref in wrong.items():
        g = _gate_ratio(o, ref, vmax)
        print(f"  {name}: {g:.1f} x the gate")
        assert g > 5.0, name
    op.compare(o, (correct, vmax))


PQ = {}


def _per_query_buffer(max_t, ratio):
    """The target's per-query compressed read buffer (used when it is smaller than the per-request
    one): [max_num_tokens * 512, 512] E4M3, viewed as [max_t * 512 / (128 / r), 1, 128 / r, 512]."""
    key = (max_t, ratio)
    if key not in PQ:
        PQ.clear()
        PQ[key] = torch.zeros(max_t * 512, D, dtype=E4, device=DEV)
    return PQ[key], PQ[key].view(max_t * 512 // (WIN // ratio), 1, WIN // ratio, D)


def _per_query_step(max_t, reqs, H, ratio, gen):
    """The per-query form: query t's top-k column k at row t * 512 + k, every compressed column
    renumbered (a row's valid entries first, then rows past sparse_topk_lens holding stale finite
    decoys that would dominate the softmax if attended), long contexts (cached up to 2^20).

    Returns the step (window rows as in the other cells, compressed half built here) and the
    compressed view.
    """
    st = _step(reqs, H, 0, gen)  # window rows as in the other tests; compressed half built here
    T = st["T"]
    buf, comp4 = _per_query_buffer(max_t, ratio)
    buf[: T * 512].copy_(torch.full((T * 512, D), 300.0, device=DEV).to(E4))  # stale decoys
    count = torch.empty(T, dtype=torch.int64, device=DEV)
    t = 0
    for c, ql in reqs:
        pos = c + torch.arange(ql, device=DEV)
        count[t : t + ql] = ((pos + 1) // ratio).clamp_max(512)
        t += ql
    rows = torch.arange(T * 512, device=DEV).view(T, 512)
    valid = torch.arange(512, device=DEV)[None] < count[:, None]
    mag = torch.randint(-8, 9, (T, 512, 1), device=DEV, generator=gen).float()
    vals = (
        (torch.randn(T, 512, D, device=DEV, generator=gen) * torch.exp2(mag) * 8)
        .clamp(-448, 448)
        .to(E4)
    )
    buf[: T * 512].view(T, 512, D)[valid] = vals[valid]
    idx = torch.cat([st["idx"][:, :WIN], rows.to(torch.int32)], 1).contiguous()
    lens = (WIN + count).to(torch.int32)
    st.update(idx=idx, lens=lens, width=WIN + 512, ratio=ratio)
    return st, comp4


def _cell_call(spec):
    """The step, the op's positional arguments and its keyword arguments for one cell.

    Seeds are per kind and per configuration, so a cell's data does not depend on which other cells
    run.
    """
    kind, H, sq, skv = spec["kind"], spec["H"], spec["q_scale"], spec["kv_scale"]
    if kind == "decode":
        G, nn, ratio = spec["G"], spec["next_n"], spec["ratio"]
        gen = _gen(G * 1000 + H * 10 + ratio * 3 + nn)
        cached = torch.randint(0, 16398 - nn, (G,), generator=gen, device=DEV).tolist()
        cached[0] = 16398 - nn  # the DSpark tail of max_seq_len 16398
        if G > 1:
            cached[1] = 3
        st = _step([(c, nn) for c in cached], H, ratio, gen)
        swa, comp = _pools(st)
    elif kind == "prefill":
        reqs = [tuple(r) for r in spec["requests"]]
        st = _step(reqs, H, spec["ratio"], _gen(len(reqs) * 100 + H + spec["ratio"]))
        swa, comp = _pools(st)
    elif kind == "scales":
        reqs = [tuple(r) for r in spec["requests"]]
        st = _step(reqs, H, spec["ratio"], _gen(int(1 / sq) + int(1 / skv)))
        swa, comp = _pools(st)
    else:
        reqs = [tuple(r) for r in spec["requests"]]
        gen = _gen(spec["max_t"] + len(reqs) + H + spec["ratio"])
        st, comp = _per_query_step(spec["max_t"], reqs, H, spec["ratio"], gen)
        swa = _buf().win4()
    dense = kind in ("decode", "scales") or (kind == "per_query" and spec["decode"])
    if dense:
        q = st["q8"].view(len(st["reqs"]), st["reqs"][0][1], H, D)
        kwargs = {}
    else:
        q = st["q8"]
        kwargs = dict(cum_seq_lens_q=st["cu"], max_q_len=st["max_q"])
    args = (q, swa, comp, _ws(), st["idx"], st["lens"], st["seq"], _sinks(H), SCALE * sq * skv, skv)
    return st, args, kwargs


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    st, args, kwargs = _cell_call(cell.spec)
    o = op(*args, **kwargs)
    assert o.shape == args[0].shape, (o.shape, args[0].shape)
    if cell.spec["kind"] == "per_query":
        assert int(st["idx"][:, WIN:].max()) == st["T"] * 512 - 1, "the buffer's last row is named"
    op.compare(o, op.reference(*args, **kwargs))
