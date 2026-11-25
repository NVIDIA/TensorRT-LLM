# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The generation-step side streams of the target's attention layers (index weights, step descriptors, late query,
compressor): layers 2 (ratio-2 owner), 20 (candidate source) and 24 (candidate consumer) on steps of 40 x 4, 80 x 4
and 25 x 6 rows give with the side streams on the picks and output they give inline, bit for bit, and an owner's
step descriptors are built off the model stream (the engine's descriptor stream).
"""

import importlib
import itertools
from types import SimpleNamespace as NS

import pytest
import torch

from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

TARGET = "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="sm_103 (GB300)",
)
att = importlib.import_module(f"{TARGET}.attention")
rope_mod = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rope"
)
DEV, MAXSEQ = "cuda", 16384
RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
CFG = NS(
    rms_norm_eps=1e-6,
    hidden_size=5120,
    head_dim=512,
    qk_rope_head_dim=64,
    num_attention_heads=64,
    q_lora_rank=1280,
    o_lora_rank=1024,
    o_groups=8,
    sliding_window=128,
    compress_ratios=RATIOS,
    kv_source_layer_ids=[2, 8, 14, 20],
    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    candidate_source_layer_id=20,
    candidate_block_size=8,
    candidate_topk_blocks=2048,
    index_topk=512,
    index_n_heads=32,
    index_head_dim=128,
)


_COLS = MAXSEQ // 128 + 1  # page-table columns of a request
_MAXREQ = 80  # requests of the widest step below


class _Kvm:
    """Random pools and page tables in which every request owns its pages (a step's row writes never collide); the
    index pages carry sane UE8M0 scales (finite scores without ties), the fp8 rows no NaN bytes."""

    def __init__(self, nreq):
        self.g = torch.Generator(device=DEV).manual_seed(7)
        self.kvm, self.pools = NS(max_seq_len=MAXSEQ), {}
        for layer, role in itertools.product(
            (2, 20), ("SWA", "GLOBAL", "INDEX", "COMPRESSOR_KV", "COMPRESSOR_SCORE")
        ):
            self.pool(
                layer, role
            )  # every pool in a fixed order, whatever order (and stream) the layer asks in
        self.swa_tables = torch.stack(
            [self.step_table(layer, "SWA", nreq) for layer in range(len(RATIOS))]
        ).contiguous()

    def pool(self, layer, role):
        r = max(RATIOS[layer], 1)
        pages = _MAXREQ * _COLS
        shape, dt = {
            "SWA": ((pages, 128, 528), torch.uint8),
            "GLOBAL": ((pages, 128 // r, 288), torch.uint8),
            "INDEX": ((pages, 128 // r, 68), torch.uint8),
            "COMPRESSOR_KV": ((pages, 128, 512), torch.float32),
            "COMPRESSOR_SCORE": ((pages, 128, 512), torch.float32),
        }[role]
        key = (role, shape)
        if key not in self.pools:
            if (
                role == "INDEX"
            ):  # native 64-row footer pages: 64 x 64 data bytes, then 64 x 4 UE8M0 scale bytes
                p = torch.randint(0, 256, shape, dtype=dt, device=DEV, generator=self.g)
                scales = p.view(-1, 64 * 68)[:, 64 * 64 :]
                scales.copy_(
                    torch.randint(124, 130, scales.shape, dtype=dt, device=DEV, generator=self.g)
                )
            elif dt == torch.uint8:
                p = torch.randint(0, 100, shape, dtype=dt, device=DEV, generator=self.g)
                tail = {"GLOBAL": (256, 56), "SWA": (512, 120)}[role]
                p[..., tail[0] :] = tail[1]
            else:
                p = torch.randn(shape, device=DEV, generator=self.g)
            self.pools[key] = p
        return self.pools[key]

    def step_table(self, layer, role, nreq):
        return torch.arange(nreq * _COLS, device=DEV, dtype=torch.int32).view(nreq, _COLS)


def _ctx(nreq, qlen):
    bases = 3000 + (torch.arange(nreq, device=DEV) * 997) % 12000  # a different history per request
    pos = (bases.repeat_interleave(qlen) + torch.arange(qlen, device=DEV).repeat(nreq)).int()
    rid = torch.arange(nreq, device=DEV).repeat_interleave(qlen)
    cs = pos - (torch.arange(nreq * qlen, device=DEV) % qlen).int()
    rope = {
        0: rope_mod.csa2_rope_table(64, MAXSEQ + 8, 0, 10000.0, 1.0, 32, 1, DEV),
        1: rope_mod.csa2_rope_table(64, MAXSEQ + 8, 65536, 160000.0, 16.0, 32, 1, DEV),
    }
    return att.StepCtx(
        pos=pos,
        rid=rid,
        cs=cs,
        ce=cs + qlen,
        segs=[],
        kvm=_Kvm(nreq),
        rope=rope,
        nreq=nreq,
        paged=True,
    )


def _layer(L):
    torch.manual_seed(L)
    m = att._Attention(CFG, L, NS(enable_attention_dp=True, world_size=1, tp_size=1, tp_rank=0)).to(
        DEV
    )
    for _, p in m.named_parameters():
        if p.dtype == torch.float8_e4m3fn:
            p.data.copy_(torch.randn(p.shape, device=p.device).to(p.dtype))
        elif p.dtype == torch.uint8:
            p.data.fill_(121)
        elif p.dtype.is_floating_point:
            p.data.normal_(0, 0.02)
    m.attn_sink.data.normal_(0, 1)
    m.wo_a_scale.data.fill_(2.0**-6)
    for nm in ("q_norm", "kv_norm", "c_norm", "i_knorm"):
        if hasattr(m, nm):
            getattr(m, nm).data.fill_(1.0)
    return m


def _run(chain, nreq, qlen, side=False, seen=None):
    """The chain's layers on one generation step; returns (each layer's output, each layer's picks)."""
    ctx = _ctx(nreq, qlen)
    x = torch.randn(
        nreq * qlen, 5120, device=DEV, generator=torch.Generator(device=DEV).manual_seed(3)
    ).bfloat16()
    streams = (
        (torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream())
        if side
        else (None, None, None)
    )
    outs, picks = [], []
    with with_multi_stream(side):
        for L in chain:
            m = _layer(L)
            m.side_stream, m.weights_stream, m.query_stream = streams
            outs.append(m(x, ctx))
            picks.append(ctx.topk.clone())
    torch.cuda.synchronize()
    return outs, picks


@pytest.mark.parametrize(
    "chain, nreq, qlen", [((2,), 40, 4), ((20, 24), 40, 4), ((2,), 80, 4), ((20, 24), 25, 6)]
)
def test_side_streams_give_the_inline_result(chain, nreq, qlen, monkeypatch):
    built = []
    fn = att.csa2_indexer_descriptors
    monkeypatch.setattr(
        att,
        "csa2_indexer_descriptors",
        lambda *a: (built.append(torch.cuda.current_stream().cuda_stream), fn(*a))[1],
    )
    main = torch.cuda.current_stream().cuda_stream
    outs, picks = _run(chain, nreq, qlen, side=True)
    assert built and all(s != main for s in built), (
        "the step descriptors are built off the model stream"
    )
    ref_outs, ref_picks = _run(chain, nreq, qlen, side=False)
    for got, ref in zip(picks, ref_picks):
        assert torch.equal(got, ref)
    for got, ref in zip(outs, ref_outs):
        assert torch.equal(got, ref)
