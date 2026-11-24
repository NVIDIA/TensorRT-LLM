# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the decode GVR Top-K's EARLY RELEASE (``enable_gvr_early_release``).

The pool-row GVR launches release their dependent grid right after their own grid-dependency wait.
The rows they write must be the default launch's bits, eager and replayed, and a PDL consumer that
reads them after its own wait must see this launch's rows, never the previous launch's.
"""

from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass.cute import runtime as _crt

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention import (
    gvr_topk_decode as _dev,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention import (
    gvr_topk_decode_host as _host,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_decode_gvr import (  # noqa: E501
    enable_gvr_early_release,
    indexer_topk_decode_gvr_paged,
)

__extra_import_path__ = ["../../../..", "."]
import _k31_topk_inputs as inputs  # noqa: E402

pytestmark = inputs.on_sm103

_SPLIT_KV = 256
_BLOCK = 64
_SCALE = 24
_LAYER = 7
_ENVELOPE = 1 << 20


@pytest.fixture(autouse=True)
def _restore_flag():
    saved = _dev._EARLY_RELEASE
    yield
    _dev._EARLY_RELEASE = saved
    _host._VARLEN_CACHE.clear()


def _build(batch, next_n, context_lens, seed, ties=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    lo, hi = context_lens
    lens = torch.randint(lo, hi + 1, (batch,), generator=gen, device="cuda", dtype=torch.int32)
    stride = -(-_ENVELOPE // _SPLIT_KV) * _SPLIT_KV
    rows = batch * next_n
    logits = torch.full((rows, stride), float("nan"), dtype=torch.float32, device="cuda")
    for r in range(rows):
        n = int(lens[r // next_n]) - next_n + r % next_n + 1
        if ties:
            logits[r, :n] = torch.randint(0, 64, (n,), generator=gen, device="cuda").float()
        else:
            bulk = torch.randn(n, generator=gen, device="cuda") * 1.5 - 2.0
            tail = torch.rand(n, generator=gen, device="cuda") < 0.02
            logits[r, :n] = torch.where(
                tail, bulk.abs() + 6.0 * torch.rand(n, generator=gen, device="cuda"), bulk
            )
    width = -(-hi // _BLOCK) + 1
    table = torch.randint(
        0, 50_000, (batch, width), generator=gen, dtype=torch.int32, device="cuda"
    )
    return dict(
        logits=logits[:, :_ENVELOPE],
        kv_lens=lens,
        indices=torch.full((rows, 2048), -5, dtype=torch.int32, device="cuda"),
        next_n=next_n,
        max_seq_len=_ENVELOPE,
        block_table=table,
        block_size=_BLOCK,
        stride_factor=_SCALE * _BLOCK,
        layer_id=_LAYER,
    )


def _run(args, early: bool) -> torch.Tensor:
    _dev._EARLY_RELEASE = early
    _host._VARLEN_CACHE.clear()
    args["indices"].fill_(-5)
    indexer_topk_decode_gvr_paged(**args)
    torch.cuda.synchronize()
    return args["indices"].clone()


_SHAPES = [
    (1, 5, (118_000, 122_000), False),
    (2, 5, (115_000, 125_000), False),
    (3, 5, (110_000, 130_000), False),
    (2, 5, (170_000, 250_000), False),
    (3, 5, (200_000, 250_000), False),
    (7, 5, (100_000, 140_000), False),
    (2, 5, (60_000, 80_000), True),
]


@pytest.mark.parametrize(
    "batch,next_n,lens,ties",
    _SHAPES,
    ids=[f"{s[0]}x{s[1]}-{s[2][1]}-{'ties' if s[3] else 'real'}" for s in _SHAPES],
)
def test_early_release_rows_are_the_default_bits(batch, next_n, lens, ties):
    args = _build(batch, next_n, lens, seed=batch * 31 + lens[1] // 1000, ties=ties)
    plain = _run(args, early=False)
    early = _run(args, early=True)
    assert any("early" in key for key in _dev._COMPILE_CACHE if isinstance(key, tuple))
    assert torch.equal(early, plain)
    args["indices"].fill_(-5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indexer_topk_decode_gvr_paged(**args)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(args["indices"], plain)


class _PdlCopy:
    """A PDL consumer: waits for its primary grid, then copies ``src`` to ``dst``."""

    @cute.kernel
    def kern(self, src: cute.Tensor, dst: cute.Tensor):
        cute.arch.griddepcontrol_wait()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 256 + tidx
        if i < cute.size(src):
            dst[i] = src[i]

    @cute.jit
    def __call__(self, src: cute.Tensor, dst: cute.Tensor, stream):
        n = cute.size(src)
        self.kern(src, dst).launch(
            grid=((n + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream, use_pdl=True
        )


def _pdl_copy():
    a, b = cute.sym_int(), cute.sym_int()
    src = _crt.make_fake_compact_tensor(cutlass.Int32, (a,), stride_order=(0,), assumed_align=16)
    dst = _crt.make_fake_compact_tensor(cutlass.Int32, (b,), stride_order=(0,), assumed_align=16)
    return cute.compile(
        _PdlCopy(),
        src,
        dst,
        _crt.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def test_pdl_consumer_reads_this_launchs_rows():
    """Rotating inputs: each replayed GVR overwrites the previous call's rows; the PDL consumer
    right behind it must copy the rows of its own call."""
    sets = [_build(2, 5, (110_000, 130_000), seed=100 + i) for i in range(4)]
    want = [_run(a, early=False) for a in sets]
    copy = _pdl_copy()
    _dev._EARLY_RELEASE = True
    _host._VARLEN_CACHE.clear()
    rows = torch.full_like(want[0], -5)
    outs = [torch.full_like(want[0], -7) for _ in range(12)]
    for i, a in enumerate(sets):
        indexer_topk_decode_gvr_paged(**dict(a, indices=rows))
        copy(rows.view(-1), outs[i].view(-1))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(len(outs)):
            indexer_topk_decode_gvr_paged(**dict(sets[i % len(sets)], indices=rows))
            copy(rows.view(-1), outs[i].view(-1))
    for _ in range(20):
        for o in outs:
            o.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        for i, o in enumerate(outs):
            assert torch.equal(o, want[i % len(sets)])


@pytest.mark.parametrize(
    "tp_size,attention_dp,armed",
    [(8, False, True), (16, False, False), (32, False, False), (8, True, False), (4, False, False)],
)
def test_armed_on_attention_tp8_only(tp_size, attention_dp, armed):
    _dev._EARLY_RELEASE = False
    enable_gvr_early_release(SimpleNamespace(tp_size=tp_size, enable_attention_dp=attention_dp))
    assert _dev._EARLY_RELEASE == armed
