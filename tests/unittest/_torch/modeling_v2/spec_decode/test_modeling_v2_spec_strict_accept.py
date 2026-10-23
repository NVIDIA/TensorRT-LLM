# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`spec_decode/strict_accept`: both entries bitwise to their in-tree torch path over every cell of their `.CELLS`.

One GPU; each cell is also run on a second random draw and inside a CUDA graph (captured after a warm call, replayed
on new inputs written into the captured buffers).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode.strict_accept import (
    SpecKvRewind,
    SpecNextNewTokens,
    SpecStrictAccept,
    spec_kv_rewind,
    spec_next_new_tokens,
    spec_strict_accept,
)

assert torch.cuda.is_available(), "strict_accept requires a CUDA device"

VOCAB = 200064


def _strict_inputs(spec, gen):
    n_req, draft_len = spec["n_req"], spec["draft_len"]
    t_dtype = spec.get("target_dtype", torch.int32)
    k1 = draft_len + 1
    target = torch.randint(0, VOCAB, (n_req * k1,), generator=gen, device="cuda").to(t_dtype)
    draft = target.view(n_req, k1)[:, :draft_len].clone().to(spec.get("draft_dtype", torch.int32))
    # A random prefix of each request's drafts matches: break the draft at a random position (or never).
    cut = torch.randint(0, draft_len + 1, (n_req,), generator=gen, device="cuda")
    pos = torch.arange(draft_len, device="cuda").unsqueeze(0)
    breaks = pos >= cut.unsqueeze(1)
    draft = torch.where(breaks & (pos == cut.unsqueeze(1)), (draft + 1) % VOCAB, draft)
    # Past the first mismatch the drafts are random again (a later match must not count).
    noise = torch.randint(0, VOCAB, draft.shape, generator=gen, device="cuda").to(draft.dtype)
    draft = torch.where(breaks & (pos > cut.unsqueeze(1)), noise, draft)
    if spec.get("pad_drafts"):
        draft[::2, draft_len // 2 :] = -1
    if spec.get("extremes"):
        draft[: n_req // 2] = target.view(n_req, k1)[: n_req // 2, :draft_len].to(draft.dtype)
        draft[n_req // 2 :] = (
            target.view(n_req, k1)[n_req // 2 :, :draft_len].to(draft.dtype) + 1
        ) % VOCAB
    return target, draft


def _next_new_inputs(spec, gen):
    n_req, draft_len = spec["n_req"], spec["draft_len"]
    width = draft_len + 1 + (4 if spec.get("wide") else 0)
    rows = n_req + 3
    accepted = torch.randint(
        0, VOCAB, (rows, width), generator=gen, device="cuda", dtype=torch.int32
    )
    num = torch.randint(1, draft_len + 2, (n_req,), generator=gen, device="cuda", dtype=torch.int32)
    batch_idx = torch.arange(n_req, device="cuda", dtype=torch.int32)
    if spec.get("shuffle"):
        batch_idx = torch.randperm(rows, generator=gen, device="cuda")[:n_req].to(torch.int32)
    draft_dtype = spec.get("draft_dtype", torch.int32)
    if spec.get("wide"):
        # A strided view: rows of draft_len + 5 elements, the first draft_len used.
        wide_rows = torch.randint(0, VOCAB, (n_req, draft_len + 5), generator=gen, device="cuda")
        next_draft = wide_rows.to(draft_dtype)[:, :draft_len]
    else:
        next_draft = torch.randint(0, VOCAB, (n_req, draft_len), generator=gen, device="cuda").to(
            draft_dtype
        )
    return accepted, num, batch_idx, next_draft


def _assert_same(out, ref):
    for o, r in zip(
        out if isinstance(out, (list, tuple)) else [out],
        ref if isinstance(ref, (list, tuple)) else [ref],
    ):
        assert o.dtype == r.dtype, (o.dtype, r.dtype)
        assert o.shape == r.shape, (o.shape, r.shape)
        assert torch.equal(o, r), (o, r)


@pytest.mark.parametrize("cell", SpecStrictAccept.CELLS, ids=lambda c: c.why[:60])
def test_strict_accept_cells(cell):
    gen = torch.Generator(device="cuda").manual_seed(7)
    for _ in range(2):
        target, draft = _strict_inputs(dict(cell.spec), gen)
        _assert_same(spec_strict_accept(target, draft), spec_strict_accept.reference(target, draft))
    # Graph replay on new values written into the captured inputs.
    target, draft = _strict_inputs(dict(cell.spec), gen)
    spec_strict_accept(target, draft)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = spec_strict_accept(target, draft)
    new_target, new_draft = _strict_inputs(dict(cell.spec), gen)
    target.copy_(new_target)
    draft.copy_(new_draft)
    graph.replay()
    torch.cuda.synchronize()
    _assert_same(out, spec_strict_accept.reference(target, draft))


@pytest.mark.parametrize("cell", SpecNextNewTokens.CELLS, ids=lambda c: c.why[:60])
def test_next_new_tokens_cells(cell):
    gen = torch.Generator(device="cuda").manual_seed(11)
    for _ in range(2):
        args = _next_new_inputs(dict(cell.spec), gen)
        spec_next_new_tokens.is_valid(*args)
        _assert_same(spec_next_new_tokens(*args), spec_next_new_tokens.reference(*args))
    args = _next_new_inputs(dict(cell.spec), gen)
    spec_next_new_tokens(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = spec_next_new_tokens(*args)
    new_args = _next_new_inputs(dict(cell.spec), gen)
    for dst, src in zip(args, new_args):
        dst.copy_(src)
    graph.replay()
    torch.cuda.synchronize()
    _assert_same(out, spec_next_new_tokens.reference(*args))


def _kv_rewind_inputs(spec, gen):
    nc, n = spec["nc"], spec["n"]
    rows = nc + n + 2  # rows past the generation block must stay untouched
    num = torch.randint(1, 5, (rows,), generator=gen, device="cuda", dtype=torch.int32)
    kv = torch.randint(0, 1 << 20, (rows,), generator=gen, device="cuda").to(
        spec.get("kv_dtype", torch.int32)
    )
    rewind = torch.empty(64, dtype=torch.int32, device="cuda")
    return num, kv, rewind


@pytest.mark.parametrize("cell", SpecKvRewind.CELLS, ids=lambda c: c.why[:60])
def test_kv_rewind_cells(cell):
    spec = dict(cell.spec)
    nc, n = spec["nc"], spec["n"]
    gen = torch.Generator(device="cuda").manual_seed(13)
    for _ in range(2):
        num, kv, rewind = _kv_rewind_inputs(spec, gen)
        kv_ref = kv.clone()
        out = spec_kv_rewind(num, kv, rewind, nc, n)
        _assert_same(out, spec_kv_rewind.reference(num, kv_ref, rewind, nc, n))
        _assert_same(kv, kv_ref)
    # Graph replay on new values written into the captured inputs.
    num, kv, rewind = _kv_rewind_inputs(spec, gen)
    spec_kv_rewind(num, kv.clone(), rewind, nc, n)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = spec_kv_rewind(num, kv, rewind, nc, n)
    new_num, new_kv, _ = _kv_rewind_inputs(spec, gen)
    num.copy_(new_num)
    kv.copy_(new_kv)
    kv_ref = kv.clone()
    graph.replay()
    torch.cuda.synchronize()
    _assert_same(out, spec_kv_rewind.reference(num, kv_ref, rewind, nc, n))
    _assert_same(kv, kv_ref)
