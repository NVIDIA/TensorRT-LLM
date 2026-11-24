# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``ZeroPaddedReduce``: the drafter's TP16 copies as one all-reduce of a zero-padded buffer.

One process plays the 16 ranks: each rank's call hands its padded buffer to a recording
all-reduce, and the reduction is the FP32 sum of the 16 buffers rounded to the dtype (what the
MNNVL kernels compute, with a -0.0 written as +0.0). The gathers must equal the all-gather's
layout bitwise up to the sign of zero: fc's 448-column BF16 shares and the top-32 FP32
``(global id, value)`` pairs, with -0.0, infinities, NaN, subnormals and the largest ids; the
broadcast must give every rank rank 0's draft ids. ``fits`` is the same on every rank, a second
call writes only the rank's own columns, and ``build`` refuses a non-MNNVL all-reduce, another
dtype, or a workspace that would have to grow. The worker's agreement override returns rank 0's
drafts and keeps the engine's path when unbound.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

_PKG = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31."
    "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
)
WORLD = 16


def _mod():
    return importlib.import_module(_PKG + ".modeling")


class _Recording:
    """An all-reduce that keeps each rank's input; ``reduce`` sums them as the kernel does."""

    def __init__(self):
        self.inputs = []

    def __call__(self, padded):
        self.inputs.append(padded.clone())
        return padded

    def reduce(self):
        total = torch.zeros_like(self.inputs[0], dtype=torch.float32)
        for x in self.inputs:
            total += torch.where(x == 0, torch.zeros_like(x), x).float()
        return total.to(self.inputs[0].dtype)


def _ranks(mod, width, columns, dtype, max_rows, writers):
    recording = _Recording()
    ranks = [
        mod.ZeroPaddedReduce(
            recording,
            writers(r),
            columns,
            torch.zeros(max_rows, width, dtype=dtype),
        )
        for r in range(WORLD)
    ]
    return recording, ranks


def _special(values):
    flat = values.view(-1)
    flat[:6] = torch.tensor([-0.0, float("inf"), float("-inf"), float("nan"), 1e-40, -1e-40])
    return values


def _same_bits_up_to_zero_sign(a, b):
    a = torch.where(a == 0, torch.zeros_like(a), a)
    b = torch.where(b == 0, torch.zeros_like(b), b)
    bits = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[a.dtype]
    return torch.equal(a.view(bits), b.view(bits))


@pytest.mark.parametrize(
    "dtype,columns,rows",
    [
        (torch.bfloat16, 448, 8),  # fc's row shares at one request's block
        (torch.bfloat16, 448, 3),
        (torch.float32, 64, 7),  # top-32 pairs at one request's 7 positions
        (torch.float32, 64, 28),
    ],
)
def test_gather_is_the_all_gather(dtype, columns, rows):
    mod = _mod()
    recording, ranks = _ranks(mod, WORLD * columns, columns, dtype, 32, lambda r: r * columns)
    g = torch.Generator().manual_seed(rows * columns)
    shares = [_special(torch.randn(rows, columns, generator=g).to(dtype)) for _ in range(WORLD)]
    if dtype == torch.float32:  # ids as the pairs carry them, up to the vocabulary's last
        for r, share in enumerate(shares):
            share[:, 0::2] = torch.randint(0, 163840, (rows, columns // 2), generator=g).float()
            share[0, 0] = 163839.0
    for rank, share in zip(ranks, shares):
        out = rank(share, rows)
        assert out.shape == (rows, WORLD * columns)
    reduced = recording.reduce()
    assert _same_bits_up_to_zero_sign(reduced, torch.cat(shares, dim=1))
    # A second, smaller call reads only rows written by it: every other column is still zero.
    recording.inputs.clear()
    for rank, share in zip(ranks, shares):
        rank(share[:1] * 2, 1)
    assert _same_bits_up_to_zero_sign(recording.reduce(), torch.cat(shares, dim=1)[:1] * 2)
    for r, rank in enumerate(ranks):
        others = torch.ones(WORLD * columns, dtype=torch.bool)
        others[r * columns : (r + 1) * columns] = False
        assert not rank.buffer[:, others].any()


def test_fp32_carries_every_vocabulary_id():
    """Every id of the 163,840-entry vocabulary, the largest (163,839) included, survives the FP32
    round trip of the pairs and the agreement, alone and summed with zeros."""
    ids = torch.arange(163840, dtype=torch.int32)
    as_fp32 = ids.to(torch.float32)
    assert torch.equal(as_fp32.to(torch.int32), ids)
    assert torch.equal((as_fp32 + torch.zeros_like(as_fp32)).to(torch.int32), ids)
    assert int(as_fp32[-1].item()) == 163839


def test_broadcast_gives_rank_zero_drafts():
    mod = _mod()
    k = 7
    recording, ranks = _ranks(mod, 8, k, torch.float32, 4, lambda r: 0 if r == 0 else None)
    drafts = [torch.randint(0, 163840, (4, k), dtype=torch.int32) for _ in range(WORLD)]
    drafts[0][0, 0] = 163839
    for rank, mine in zip(ranks, drafts):
        rank(mine, 4)
    agreed = recording.reduce()[:, :k].to(torch.int32)
    assert torch.equal(agreed, drafts[0])
    assert all(not rank.buffer.any() for rank in ranks[1:])


def test_fits_is_rank_independent():
    mod = _mod()
    _, ranks = _ranks(mod, 8, 7, torch.float32, 4, lambda r: 0 if r == 0 else None)
    for rows, cols in ((1, 7), (4, 7), (5, 7), (0, 7), (2, 6), (2, 8)):
        assert len({rank.fits(rows, cols) for rank in ranks}) == 1
    assert ranks[3].fits(4, 7) and not ranks[3].fits(5, 7) and not ranks[3].fits(2, 6)


def test_build_refuses(monkeypatch):
    mod = _mod()
    mapping = SimpleNamespace(tp_size=WORLD)
    mnnvl = SimpleNamespace(
        dtype=torch.bfloat16,
        mapping=mapping,
        get_required_workspace_size=lambda rows, width, world, dtype: (
            rows * width * world * dtype.itemsize
        ),
    )
    workspace = {"buffer_size_bytes": 8 * 7168 * WORLD * 2}
    monkeypatch.setattr(mod, "get_or_scale_allreduce_mnnvl_workspace", lambda m, d: workspace)
    args = (0, 448, 7168)
    cpu = torch.device("cpu")
    assert mod.ZeroPaddedReduce.build(SimpleNamespace(), *args, 8, torch.bfloat16, cpu) is None
    ar = SimpleNamespace(mnnvl_allreduce=mnnvl)
    assert mod.ZeroPaddedReduce.build(ar, *args, 8, torch.float32, cpu) is None
    assert mod.ZeroPaddedReduce.build(ar, *args, 9, torch.bfloat16, cpu) is None
    built = mod.ZeroPaddedReduce.build(ar, *args, 8, torch.bfloat16, cpu)
    assert built is not None and built.buffer.shape == (8, 7168) and not built.buffer.any()
    # Rows round up to 16-byte vectors: 7 FP32 columns are a row of 8.
    mnnvl.dtype = torch.float32
    built = mod.ZeroPaddedReduce.build(ar, None, 7, 7, 4, torch.float32, cpu)
    assert built.buffer.shape == (4, 8)
    with pytest.raises(ValueError):
        mod.ZeroPaddedReduce.build(ar, 4, 7, 7, 4, torch.float32, cpu)


def test_worker_agreement():
    mod = _mod()
    from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

    worker = object.__new__(mod.DFlashWorker)
    recording = _Recording()
    worker._draft_agreement = mod.ZeroPaddedReduce(
        recording, 0, 7, torch.zeros(4, 8, dtype=torch.float32, device="cuda")
    )
    drafts = torch.randint(0, 163840, (2, 7), dtype=torch.int32, device="cuda")
    drafts[0, 0] = 163839
    first = torch.tensor([[5], [6]], dtype=torch.int32, device="cuda")
    new_tokens = torch.cat([first, drafts], dim=1)
    outputs = {"next_draft_tokens": drafts, "next_new_tokens": new_tokens.clone()}
    worker._agree_draft_tokens(outputs)
    assert torch.equal(outputs["next_draft_tokens"], drafts)
    assert outputs["next_draft_tokens"].dtype == torch.int32
    assert torch.equal(outputs["next_new_tokens"], new_tokens)
    # Another rank's call: its own drafts and tokens differ; the reduced row is rank 0's.
    other = torch.randint(0, 163840, (2, 7), dtype=torch.int32, device="cuda")
    rank1 = mod.ZeroPaddedReduce(
        lambda padded: recording.inputs[0].clone(), None, 7, torch.zeros(4, 8, device="cuda")
    )
    worker._draft_agreement = rank1
    out1 = {"next_draft_tokens": other, "next_new_tokens": torch.cat([first, other], dim=1)}
    worker._agree_draft_tokens(out1)
    assert torch.equal(out1["next_draft_tokens"], drafts)
    assert torch.equal(out1["next_new_tokens"], new_tokens)
    assert not rank1.buffer.any()
    # Without next-step tokens only the drafts are agreed.
    out2 = {"next_draft_tokens": other}
    worker._agree_draft_tokens(out2)
    assert torch.equal(out2["next_draft_tokens"], drafts) and "next_new_tokens" not in out2
    worker._draft_agreement = mod.ZeroPaddedReduce(
        recording, 0, 7, torch.zeros(4, 8, dtype=torch.float32, device="cuda")
    )
    # Unbound, or a shape it was not built for: the engine's all-gather path.
    calls = []
    original = SpecWorkerBase._agree_draft_tokens
    try:
        SpecWorkerBase._agree_draft_tokens = lambda self, out: calls.append(out)
        worker._draft_agreement = None
        worker._agree_draft_tokens(outputs)
        worker._draft_agreement = mod.ZeroPaddedReduce(
            recording, 0, 7, torch.zeros(1, 8, dtype=torch.float32)
        )
        worker._agree_draft_tokens(outputs)
    finally:
        SpecWorkerBase._agree_draft_tokens = original
    assert len(calls) == 2


def test_agreement_row_launches_a_wide_block():
    """The agreement's all-reduce clears an earlier call's Lamport buffer with its own threads: its
    row must be wide (FP32, four columns a thread), not the K drafts' two threads."""
    mod = _mod()
    assert mod._DRAFT_AGREEMENT_WIDTH // 4 >= 512


def test_own_columns_is_the_staged_region():
    mod = _mod()
    rank = mod.ZeroPaddedReduce(_Recording(), 128, 64, torch.zeros(8, 1024))
    view = rank.own_columns(3)
    assert view.shape == (3, 64) and view.stride() == (1024, 1)
    view.fill_(2.0)
    assert rank.buffer[:3, 128:192].eq(2.0).all() and rank.buffer.sum() == 2.0 * 3 * 64
    with pytest.raises(ValueError):
        mod.ZeroPaddedReduce(_Recording(), None, 7, torch.zeros(4, 8)).own_columns(1)


@pytest.mark.parametrize("requests", [1, 2])
def test_global_top_k_pairs_into_the_buffer(requests):
    """The worker's vocab-parallel top-32 with the pairs written straight into each rank's columns
    of the exchange buffer: every rank's staged buffer, and its merged candidates, bitwise the
    path that selects into a scratch tensor and stages a copy of it."""
    mod = _mod()
    k, top_k, shard = 7, 32, 163840 // WORLD
    rows = requests * k
    g = torch.Generator(device="cuda").manual_seed(requests)
    shards = [
        (8.0 * torch.randn(requests, k, shard, generator=g, device="cuda")).bfloat16()
        for _ in range(WORLD)
    ]
    shards[3][0, 0, :40] = shards[3][0, 0, :40].abs().max() + 1.0  # ties across the 32nd

    def worker(r, all_reduce):
        w = object.__new__(mod.DFlashWorker)
        w._vocab_parallel = SimpleNamespace(head=SimpleNamespace(col_base=r * shard))
        w._pairs_gather = mod.ZeroPaddedReduce(
            all_reduce, r * 2 * top_k, 2 * top_k, torch.zeros(16 * k, 1024, device="cuda")
        )
        return w

    # Pass 1: each rank's buffer as it reaches the all-reduce.
    recording = _Recording()
    for r in range(WORLD):
        worker(r, recording)._dflash2_global_top_k(shards[r], None, top_k, 163840, block=False)
    staged = [x.clone() for x in recording.inputs]
    total = torch.stack(staged).sum(0)
    # The staging path: scratch pairs, copied into the columns.
    for r in range(WORLD):
        scratch = mod.k31_dflash2_shard_top32(shards[r], top_k, r * shard, True)
        old = mod.ZeroPaddedReduce(
            _Recording(), r * 64, 64, torch.zeros(16 * k, 1024, device="cuda")
        )
        old.stage(scratch.view(rows, 64), rows)
        assert torch.equal(staged[r].view(torch.int32), old.buffer[:rows].view(torch.int32))
    # Pass 2: every rank merges the reduced buffer as the gathered pairs.
    gathered = torch.cat(
        [mod.k31_dflash2_shard_top32(shards[r], top_k, r * shard, True) for r in range(WORLD)], -1
    )
    want_values, want_ids = mod.k31_dflash2_merge_top32(gathered.contiguous(), top_k)
    for r in range(WORLD):
        ids, values, block = worker(r, lambda padded: total.clone())._dflash2_global_top_k(
            shards[r], None, top_k, 163840, block=False
        )
        assert block is None
        assert torch.equal(ids, want_ids) and torch.equal(
            values.view(torch.int32), want_values.view(torch.int32)
        )
