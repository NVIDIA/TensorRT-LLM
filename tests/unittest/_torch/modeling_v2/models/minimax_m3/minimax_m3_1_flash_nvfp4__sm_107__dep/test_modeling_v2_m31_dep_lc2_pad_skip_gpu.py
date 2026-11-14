# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PAD_MASK's LC2 skip on the DEP LC2 decode: with the CUDA-graph padding requests' KV lengths read as 0, the real rows
are bitwise those of the call with the padding requests' own (one short block) lengths, and the padding rows are finite
zeros. At the 96-row step with 8 padding rows, the 128-row graph bucket (24 padding rows) and a rank whose requests are
all padding; one case also through a captured CUDA graph."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    lc2_decode,
    pad_mask,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 (Rubin) only"
)

KV_HEADS, HEADS, HEAD_DIM, PAGE, TOPK, ROWS = 4, 64, 128, 128, 16, 8
CONTEXTS = (1000, 5000, 20000, 40000, 3000)
#: A padding request's KV length: one short block.
PAD_CONTEXT = 8


def _case(bucket, real, seed=0):
    seqs = [CONTEXTS[i % len(CONTEXTS)] for i in range(real)] + [PAD_CONTEXT] * (bucket - real)
    pages = [(s + PAGE - 1) // PAGE for s in seqs]
    gen = torch.Generator().manual_seed(seed)
    dgen = torch.Generator(device="cuda").manual_seed(seed)
    perm = torch.randperm(sum(pages), generator=gen).to(torch.int32)
    table = torch.full((bucket, max(pages)), -1, dtype=torch.int32)
    for r, start in enumerate(torch.tensor([0, *pages]).cumsum(0)[:-1].tolist()):
        table[r, : pages[r]] = perm[start : start + pages[r]]
    shape = (sum(pages), KV_HEADS, PAGE, HEAD_DIM // 2)
    sshape = (sum(pages), KV_HEADS, PAGE, HEAD_DIM // 16)

    def scales():
        return (
            (torch.rand(sshape, device="cuda", generator=dgen) * 1.75 + 0.25)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )

    rows = bucket * ROWS
    topk = torch.full((KV_HEADS, rows, TOPK), -1, dtype=torch.int32)
    for req, seq in enumerate(seqs):
        for r in range(ROWS):
            blocks = (max(seq - ROWS + r + 1, 0) + PAGE - 1) // PAGE
            for h in range(KV_HEADS):
                topk[h, req * ROWS + r, : min(blocks, TOPK)] = torch.randperm(
                    blocks, generator=gen
                )[:TOPK]
    args = (
        (torch.randn((rows, HEADS, HEAD_DIM), device="cuda", generator=dgen) * 2.0).to(
            torch.float8_e4m3fn
        ),
        torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda", generator=dgen),
        torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda", generator=dgen),
        topk.cuda(),
        table.cuda(),
        torch.tensor(seqs, dtype=torch.int32, device="cuda"),
    )
    kwargs = dict(
        sm_scale=HEAD_DIM**-0.5,
        decode_query_len=ROWS,
        k_block_scale=scales(),
        v_block_scale=scales(),
        k_global_scale=torch.tensor([0.05], device="cuda"),
        v_global_scale=torch.tensor([0.03], device="cuda"),
    )
    requests = torch.arange(bucket, device="cuda") >= real
    return args, kwargs, requests


@pytest.fixture(scope="module", autouse=True)
def _split_workspace():
    lc2_decode.reserve_workspace(torch.device("cuda"))


@pytest.fixture(autouse=True)
def _lc2_only(monkeypatch):
    """These are LC2's contracts: OWN-T's padding rows are test_modeling_v2_m31_dep_own_t_gpu.py's."""
    monkeypatch.setattr(lc2_decode, "DEP_OWNT", False)


def _output(args):
    return torch.empty((args[0].shape[0], HEADS, HEAD_DIM), dtype=torch.bfloat16, device="cuda")


def _decode(args, kwargs, requests=None):
    out = _output(args)
    state = None
    if requests is not None:
        state = pad_mask.PadMask()
        state.lc2_layers, state.requests = 1, requests
    lc2_decode.set_pad_rows(state)
    try:
        lc2_decode.dep_lc2_decode(*args, output=out, **kwargs)
    finally:
        lc2_decode.set_pad_rows(None)
    return out


@pytest.mark.parametrize("bucket, real", [(12, 11), (16, 13), (12, 0)])
def test_the_real_rows_are_bitwise_and_the_padding_rows_zero(bucket, real):
    args, kwargs, requests = _case(bucket, real)
    scales = [
        kwargs[k] for k in ("k_block_scale", "v_block_scale", "k_global_scale", "v_global_scale")
    ]
    assert lc2_decode.decline_reason(*args, _output(args), ROWS, None, *scales, None) is None
    stock = _decode(args, kwargs)
    skip = _decode(args, kwargs, requests)
    torch.cuda.synchronize()
    assert torch.equal(skip[: real * ROWS], stock[: real * ROWS])
    assert torch.isfinite(skip).all()
    assert not skip[real * ROWS :].any()
    assert args[5].tolist()[real:] == [PAD_CONTEXT] * (bucket - real)


def test_the_skip_replays_in_a_cuda_graph():
    args, kwargs, requests = _case(12, 11, seed=1)
    eager = _decode(args, kwargs, requests)
    state = pad_mask.PadMask()
    state.lc2_layers, state.requests = 1, requests
    out = torch.empty_like(eager)
    lc2_decode.set_pad_rows(state)
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            lc2_decode.dep_lc2_decode(*args, output=out, **kwargs)
        torch.cuda.current_stream().wait_stream(side)
        state.end()
        state.lc2_layers, state.requests = 1, requests
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            lc2_decode.dep_lc2_decode(*args, output=out, **kwargs)
    finally:
        lc2_decode.set_pad_rows(None)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
