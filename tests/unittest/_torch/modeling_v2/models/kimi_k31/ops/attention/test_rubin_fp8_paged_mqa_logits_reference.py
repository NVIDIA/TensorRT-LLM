# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the rubin_fp8_paged_mqa_logits entry: its reference on small shapes, its guard and
its cell table. Nothing here launches the kernel, so no GPU and no CuTe DSL is needed.

The reference is checked against the logits written out from the dense keys, on integer data
where every product and sum is exact, so the two must agree bit for bit.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_fp8_paged_mqa_logits import (  # noqa: E501
    MAX_N_TILE,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_fp8_paged_mqa_logits import (  # noqa: E501
    rubin_fp8_paged_mqa_logits as op,
)

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7),
        reason="CPU checks of a Rubin (sm_107) entry: run without a GPU or on sm_107",
    ),
]

HEADS, DIM = 32, 128
TOKEN_BYTES = DIM + 4


def _case(lens, next_n, max_context_len, page=16, pad=0, seed=0):
    """Integer keys on random pages of a small cache, and the same keys dense, per request."""
    gen = torch.Generator().manual_seed(seed)
    pages = [-(-n // page) for n in lens]
    num_pages = sum(pages) + 3
    base = torch.randint(0, 256, (num_pages, page * TOKEN_BYTES + pad), generator=gen)
    base = base.to(torch.uint8)
    perm = torch.randperm(num_pages, generator=gen)
    table = torch.zeros(len(lens), max(pages), dtype=torch.int32)
    dense, used = [], 0
    for b, (n, count) in enumerate(zip(lens, pages)):
        ids = perm[used : used + count]
        used += count
        table[b, :count] = ids.to(torch.int32)
        k = torch.randint(-3, 4, (count * page, DIM), generator=gen).float()
        s = torch.pow(2.0, torch.randint(-6, 2, (count * page,), generator=gen).float())
        for i, pid in enumerate(ids.tolist()):
            rows = slice(i * page, (i + 1) * page)
            base[pid, : page * DIM] = k[rows].to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1)
            base[pid, page * DIM : page * TOKEN_BYTES] = s[rows].contiguous().view(torch.uint8)
        dense.append(k[:n].double() * s[:n, None].double())
    rows = len(lens) * next_n
    q = torch.randint(-3, 4, (len(lens), next_n, HEADS, DIM), generator=gen).float()
    args = dict(
        q=q.to(torch.float8_e4m3fn),
        kv_cache=base[:, : page * TOKEN_BYTES].view(num_pages, page, 1, TOKEN_BYTES),
        weights=torch.randint(-3, 4, (rows, HEADS), generator=gen).float() / 16,
        context_lens=torch.tensor(lens, dtype=torch.int32),
        block_table=table,
        # A CPU schedule: the guard checks its SM count on the GPU only (the GPU test).
        schedule_meta=torch.zeros((2, 2), dtype=torch.int32),
        max_context_len=max_context_len,
        num_next_n_atoms=1,
        atom=None,
        kv_evict_first=True,
        out=torch.full((rows, max_context_len), 7.0),
    )
    return args, dense


def _expected(args, dense) -> torch.Tensor:
    """The logits written out from the dense keys: a row sees keys up to its causal end."""
    q, w = args["q"].double(), args["weights"].double()
    batch, next_n = q.shape[:2]
    out = torch.full((batch * next_n, args["max_context_len"]), float("-inf"))
    for b in range(batch):
        for t in range(next_n):
            end = dense[b].shape[0] - next_n + t + 1
            if end > 0:
                scores = torch.einsum("hd,jd->hj", q[b, t], dense[b][:end]).relu()
                out[b * next_n + t, :end] = torch.einsum(
                    "h,hj->j", w[b * next_n + t], scores
                ).float()
    return out


@pytest.mark.parametrize(
    "lens,next_n,pad",
    [
        ([70, 130], 2, 0),  # several pages, a two-row verify window
        ([1, 40], 8, 0),  # a context shorter than the window: rows with no key
        ([33], 1, 0),  # plain decode
        ([90, 17, 64], 6, 48),  # a padded page stride (the V2 per-layer view)
    ],
)
def test_reference_is_the_causal_weighted_relu_sum(lens, next_n, pad):
    args, dense = _case(lens, next_n, max_context_len=160, pad=pad, seed=len(lens) * 7 + next_n)
    before = args["out"].clone()
    ref = op.reference(**args)
    assert ref.dtype == torch.float32 and ref.shape == (len(lens) * next_n, 160)
    assert torch.equal(ref, _expected(args, dense))
    assert torch.equal(args["out"], before), "the reference must leave out untouched"


def test_reference_ignores_the_work_partition_and_the_cache_policy():
    args, _ = _case([50, 20], 4, max_context_len=64, seed=3)
    ref = op.reference(**args)
    other = dict(args, num_next_n_atoms=2, atom=2, kv_evict_first=False, schedule_meta=None)
    assert torch.equal(ref, op.reference(**other))


def test_guard_admits_a_well_formed_call():
    args, _ = _case([70, 130], 2, max_context_len=160)
    op.is_valid(**args)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda a: dict(a, q=a["q"].view(torch.uint8)), "E4M3"),
        (lambda a: dict(a, max_context_len=100, out=a["out"][:, :100]), "max_context_len"),
        (lambda a: dict(a, out=a["out"][:-1]), "out must be"),
        (lambda a: dict(a, out=a["out"][:, :-1]), "out must be"),
        (lambda a: dict(a, out=a["out"].double()), "fp32"),
        (lambda a: dict(a, weights=a["weights"][:, :16].contiguous()), "one row per query row"),
        (lambda a: dict(a, q=a["q"].transpose(2, 3).contiguous().transpose(2, 3)), "packed"),
        (lambda a: dict(a, kv_cache=_unpacked(a["kv_cache"])), "packed"),
        (lambda a: dict(a, kv_cache=_token_width(a["kv_cache"], TOKEN_BYTES + 4)), "head_dim"),
        (lambda a: dict(a, block_table=a["block_table"][:, :-1]), "block_table"),
    ],
    ids=[
        "q dtype",
        "context past max",
        "short out",
        "narrow out",
        "out dtype",
        "weights",
        "q",
        "kv",
        "kv token width",
        "short table",
    ],
)
def test_guard_refuses(change, match):
    args, _ = _case([70, 130], 2, max_context_len=160)
    with pytest.raises(AssertionError, match=match):
        op.is_valid(**change(args))


def _unpacked(kv):
    """The same page view with each token's 132 bytes spread over 264."""
    wide = torch.zeros(kv.shape[0], kv.shape[1], 1, 2 * kv.shape[3], dtype=torch.uint8)
    return wide[..., : kv.shape[3]]


def _token_width(kv, width):
    """A packed page view with ``width`` bytes per token instead of 132."""
    return torch.zeros(kv.shape[0], kv.shape[1], 1, width, dtype=torch.uint8)


def test_cells_are_the_rubin_shapes_the_kernel_takes_in_one_atom():
    assert Arch.SM_107 in op.ARCHS
    whys = [c.why for c in op.CELLS]
    assert len(set(whys)) == len(whys)
    specs = [repr(sorted(c.spec.items())) for c in op.CELLS]
    assert len(set(specs)) == len(specs)
    for cell in op.CELLS:
        s = cell.spec
        assert s["num_heads"] == HEADS and s["head_dim"] == DIM and s["tokens_per_block"] == 64
        assert s["num_next_n_atoms"] == 1 and s["next_n"] * s["num_heads"] <= MAX_N_TILE
        assert s["context_lens"][0] <= s["context_lens"][1] <= s["max_context_len"]
    # The specialized targets' GEN shapes, (batch per rank, next_n), with the L2 hint:
    # DEP16 b12 k7, helix8 b16 k5 and TEP8 b1 k7.
    points = {(c.spec["batch"], c.spec["next_n"]) for c in op.CELLS if c.spec["kv_evict_first"]}
    assert {(12, 8), (16, 6), (1, 8)} <= points
    assert any(not c.spec["kv_evict_first"] for c in op.CELLS)
