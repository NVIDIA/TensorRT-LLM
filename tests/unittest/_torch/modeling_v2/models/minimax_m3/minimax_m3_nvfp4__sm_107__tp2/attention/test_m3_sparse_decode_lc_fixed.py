# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family library's fixed copy of the sm_107 tp2 long-context sparse decode (LC2,
``minimax_m3_sm107_tp2_sparse_decode_lc``), which the MiniMax-M3 tp2 target now calls.

Its fix moves the query reads after ``griddepcontrol.wait``; on inputs that are complete before the launch it writes
what ``torch.ops.trtllm.minimax_m3_sm107_tp2_sparse_decode_lc`` writes, byte for byte, for every variant (linear
softmax, pre-wait block-table staging, MXFP8 hand-off), with ``allow_split=False`` (the MiniMax-M3 callers) and
with ``allow_split=True`` (the split form the MiniMax-M3.1-Flash callers may take).
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

PAGE, DIM, TOPK, GROUP, KV_HEADS = 128, 128, 16, 16, 2
WIDTH = KV_HEADS * GROUP * DIM
SM_SCALE = DIM**-0.5

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 tp2 sparse decode runs on SM107 only",
)


def _e4m3_bytes(shape, gen, lo=0x28, hi=0x48):
    mag = torch.randint(lo, hi, shape, dtype=torch.int32, device="cuda", generator=gen)
    sign = torch.randint(0, 2, shape, dtype=torch.int32, device="cuda", generator=gen) << 7
    return (mag | sign).to(torch.uint8)


def _case(seq_lens, seed, dql=4, overlap=14, q_hi=0x48, tie=False):
    """A packed-NVFP4 paged cache, ascending top-16 tables with a -1 tail, and FP8 queries for TP2 geometry."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    reqs, total_q = len(seq_lens), len(seq_lens) * dql
    nblk = [(s + PAGE - 1) // PAGE for s in seq_lens]
    pages = sum(nblk) + 8
    k = torch.randint(
        0, 256, (pages, KV_HEADS, PAGE, DIM // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    v = torch.randint(
        0, 256, (pages, KV_HEADS, PAGE, DIM // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    if tie:
        k[:] = k[:1, :, :1, :]
    ksc = _e4m3_bytes((pages, KV_HEADS, PAGE, DIM // 16), gen)
    vsc = _e4m3_bytes((pages, KV_HEADS, PAGE, DIM // 16), gen)
    perm = torch.randperm(pages, device="cuda", generator=gen).int()
    bt = torch.full((reqs, max(nblk)), -1, dtype=torch.int32, device="cuda")
    start = 0
    for r, n in enumerate(nblk):
        bt[r, :n] = perm[start : start + n]
        start += n
    topk = torch.full((KV_HEADS, total_q, TOPK), -1, dtype=torch.int32, device="cuda")
    for r, s in enumerate(seq_lens):
        for h in range(KV_HEADS):
            last = (s + PAGE - 1) // PAGE
            base = torch.randperm(max(last - 1, 1), device="cuda", generator=gen)[
                : TOPK - 1
            ].tolist()
            for qo in range(dql):
                nb = (s - dql + qo + 1 + PAGE - 1) // PAGE
                sel = [b for b in base[:overlap] if b < nb - 1]
                for b in torch.randperm(max(nb - 1, 1), device="cuda", generator=gen).tolist():
                    if len(sel) >= min(TOPK, nb) - 1:
                        break
                    if b not in sel and b < nb - 1:
                        sel.append(b)
                sel = sorted(set(sel + [nb - 1]))[: min(TOPK, nb)]
                topk[h, r * dql + qo, : len(sel)] = torch.tensor(
                    sel, dtype=torch.int32, device="cuda"
                )
    q = _e4m3_bytes((total_q, KV_HEADS * GROUP, DIM), gen, lo=0x20, hi=q_hi).view(
        torch.float8_e4m3fn
    )
    return dict(
        q=q,
        k=k,
        v=v,
        ksc=ksc,
        vsc=vsc,
        kgs=torch.tensor([0.03], device="cuda"),
        vgs=torch.tensor([0.05], device="cuda"),
        topk=topk,
        bt=bt,
        seq=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        dql=dql,
    )


def _run(op, c, linear, prewait, mxfp8, use_pdl, **kwargs):
    total_q = c["q"].shape[0]
    out = torch.full((total_q, WIDTH), -7.0, dtype=torch.bfloat16, device="cuda")
    q8 = torch.zeros(total_q, WIDTH, dtype=torch.uint8, device="cuda") if mxfp8 else None
    sf_bytes = 128 * (WIDTH // 32) * ((total_q + 127) // 128)
    sf = torch.zeros(sf_bytes, dtype=torch.uint8, device="cuda") if mxfp8 else None
    op(
        c["q"].view(torch.uint8),
        c["k"],
        c["v"],
        c["ksc"],
        c["vsc"],
        c["kgs"],
        c["vgs"],
        c["topk"],
        c["bt"],
        c["seq"],
        out,
        q8,
        sf,
        WIDTH // 128,
        c["dql"],
        SM_SCALE,
        linear,
        use_pdl,
        prewait,
        None,
        **kwargs,
    )
    return out, q8, sf


_CASES = {
    "b1-165k": dict(seq_lens=[165521]),
    "b2-short": dict(seq_lens=[130, 5000]),
    "b1-tiny": dict(seq_lens=[4]),
    "b4": dict(seq_lens=[165521, 120000, 80000, 30000]),
}


def _same(a, b):
    return (a is None and b is None) or torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("name", list(_CASES))
@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("prewait", [False, True])
@pytest.mark.parametrize("mxfp8", [False, True])
@pytest.mark.parametrize("allow_split", [False, True])
def test_the_fixed_copy_writes_the_stock_bytes(name, linear, prewait, mxfp8, allow_split):
    c = _case(seed=3, **_CASES[name])
    flags = dict(linear=linear, prewait=prewait, mxfp8=mxfp8, use_pdl=False)
    stock = _run(torch.ops.trtllm.minimax_m3_sm107_tp2_sparse_decode_lc, c, **flags)
    fixed = _run(
        family_ops().minimax_m3_sm107_tp2_sparse_decode_lc, c, **flags, allow_split=allow_split
    )
    torch.cuda.synchronize()
    for want, got in zip(stock, fixed):
        assert _same(want, got)
