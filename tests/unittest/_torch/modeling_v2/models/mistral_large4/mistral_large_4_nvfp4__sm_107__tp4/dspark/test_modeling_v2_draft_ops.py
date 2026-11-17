# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the sm_107 tp4 target's DSpark draft ops: every entry's cells against its own
reference, CUDA graph replay on changing inputs, and the edges each entry's docstring states."""

from contextlib import contextmanager

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.ops.dspark import (  # noqa: E501
    draft_ops,
)

__extra_import_path__ = ["../../../.."]
from _validating import ULP, assert_within_ulp, certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires VR200/SM107",
)

attention = draft_ops.triton_draft_attention
latent_tail = draft_ops.triton_latent_tail
markov_input = draft_ops.triton_markov_input
markov_gate = draft_ops.triton_markov_gate
markov_logits = draft_ops.triton_markov_logits

LAT, PE = draft_ops.LAT, draft_ops.PE
#: The draft's softmax scale: the 192-wide MLA query/key head.
SCALE = 192**-0.5
VOCAB = 204800
MARKOV_RANK = 512
HIDDEN = 6144
#: Rope table rows and the latent RMSNorm's eps.
ROPE_ROWS = 65536
EPS = 1e-6


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _randn(*shape, gen, dtype=torch.bfloat16, std=1.0):
    return (torch.randn(shape, device="cuda", generator=gen) * std).to(dtype)


def _rope_table() -> torch.Tensor:
    """`[cos | sin]` rows rounded through bf16, as `_rope_tables` holds them."""
    inv = 1.0 / 10000 ** (torch.arange(0, PE, 2, dtype=torch.float32) / PE)
    angle = torch.arange(ROPE_ROWS, dtype=torch.float32)[:, None] * inv[None, :]
    cos = angle.cos().to(torch.bfloat16).float()
    sin = angle.sin().to(torch.bfloat16).float()
    return torch.cat((cos, sin), dim=1).cuda().contiguous()


# --- triton_draft_attention -------------------------------------------------------------------


class _Context:
    """A paged draft context the way the drafter holds one: a `[pages, 1, 1, page, 576]` layer
    of the arena, one page-table row per batch slot read through `table_rows`, and every pool
    row a sequence has not written yet holding NaN. Page 0 belongs to no sequence and is all
    NaN; the table's unused entries point at it, as a zero-initialized block table does."""

    def __init__(self, spec, seed: int):
        gen = _gen(seed)
        self.page = page = spec["page"]
        self.batch = batch = spec["batch"]
        self.lengths = list(spec["lengths"])
        # Room for every sequence to grow by 8 steps of up to 3 tokens during graph replays.
        capacity = [length + 32 for length in self.lengths]
        owned = [-(-c // page) for c in capacity]
        slots = batch + 1
        self.table = torch.zeros(slots, max(owned), dtype=torch.int32, device="cuda")
        pages = torch.randperm(sum(owned), device="cuda", generator=gen).to(torch.int32) + 1
        self.pool5 = torch.full(
            (sum(owned) + 1, 1, 1, page, LAT + PE),
            float("nan"),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.table_rows = torch.randperm(slots, device="cuda", generator=gen)[:batch]
        start = 0
        for b in range(batch):
            row = int(self.table_rows[b])
            self.table[row, : owned[b]] = pages[start : start + owned[b]]
            start += owned[b]
            self._write(b, 0, self.lengths[b], gen)
        self.pool = self.pool5[:, 0, 0]
        self.length_t = torch.tensor(self.lengths, dtype=torch.int64, device="cuda")

    def _write(self, b: int, first: int, last: int, gen) -> None:
        """Store fresh keys at sequence `b`'s positions `[first, last)`."""
        if last <= first:
            return
        row = self.table[int(self.table_rows[b])].long()
        positions = torch.arange(first, last, device="cuda")
        values = _randn(last - first, LAT + PE, gen=gen)
        self.pool5[row[positions // self.page], 0, 0, positions % self.page] = values

    def grow(self, b: int, by: int, gen) -> None:
        self._write(b, self.lengths[b], self.lengths[b] + by, gen)
        self.lengths[b] += by
        self.length_t[b] = self.lengths[b]


def _attention_args(ctx: _Context, spec, gen):
    query = _randn(ctx.batch, spec["block"], spec["heads"], LAT + PE, gen=gen)
    block_kv = _randn(ctx.batch, spec["block"], LAT + PE, gen=gen)
    return (
        query,
        ctx.pool,
        ctx.table,
        ctx.table_rows,
        ctx.length_t,
        block_kv,
        SCALE,
        spec["window"],
    )


def _attention_truth(args) -> torch.Tensor:
    """The attention in float64 over the reference's own bf16 operands."""
    query, pool, table, rows, lengths, block_kv, scale, window = args
    batch, block = query.shape[:2]
    page = pool.shape[1]
    positions = (lengths.long() - window).clamp_min(0)[:, None] + torch.arange(
        window, device=query.device
    )
    valid = positions < lengths.long()[:, None]
    safe = positions.clamp(max=table.shape[1] * page - 1)
    pages = table.index_select(0, rows.long()).long().gather(1, safe // page)
    context = pool[pages, safe % page].masked_fill(~valid.unsqueeze(-1), 0)
    keys = torch.cat((context, block_kv), dim=1).double()
    key_valid = torch.cat(
        (valid, torch.ones(batch, block, dtype=torch.bool, device=valid.device)), 1
    )
    scores = torch.einsum("bshd,btd->bsht", query.double(), keys) * scale
    scores = scores.masked_fill(~key_valid[:, None, None, :], float("-inf"))
    return torch.einsum("bsht,btc->bshc", torch.softmax(scores, dim=-1), keys[..., :LAT])


def _rel_rms(out: torch.Tensor, truth: torch.Tensor) -> float:
    return ((out.double() - truth).pow(2).mean().sqrt() / truth.pow(2).mean().sqrt()).item()


@contextmanager
def _fp32_reduced_bf16_matmul():
    """Keep the reference's bf16 products reducing in fp32.

    torch lets cuBLAS reduce a bf16 product's split-K partials in bf16 by default, and it does
    so for the one-sequence cells' value product: the reference alone then moves 3.4 ulp in
    relative RMS (6.9 element-wise) and becomes the inaccurate side of the comparison."""
    matmul = torch.backends.cuda.matmul
    previous = matmul.allow_bf16_reduced_precision_reduction
    matmul.allow_bf16_reduced_precision_reduction = False
    try:
        yield
    finally:
        matmul.allow_bf16_reduced_precision_reduction = previous


def _check_attention(out: torch.Tensor, args) -> None:
    """Measured on VR200 over the cells, 3 draws each: within 1.98 ulp element-wise of the
    reference and 0.74 ulp of relative RMS, gated at 4 and 1.5. Both round the scores to bf16
    first, which is most of their distance from float64; the op's relative RMS from float64
    was 0.84 to 1.04 times the reference's, gated at 1.1."""
    with _fp32_reduced_bf16_matmul():
        ref = attention.reference(*args)
    assert not out.isnan().any()
    assert_within_ulp(out, ref, element_ulp=4, rms_ulp=1.5)
    truth = _attention_truth(args)
    assert _rel_rms(out, truth) <= 1.1 * _rel_rms(ref, truth)


@pytest.mark.parametrize("cell", attention.CELLS, ids=[cell.why for cell in attention.CELLS])
@pytest.mark.parametrize("draw", range(2))
def test_draft_attention_cells(cell, draw, certifying) -> None:  # noqa: F811
    certifying(attention)
    seed = cell.seed_base() + 2 * draw
    ctx = _Context(cell.spec, seed)
    args = _attention_args(ctx, cell.spec, _gen(seed + 1))
    _check_attention(attention(*args), args)


def test_draft_attention_graph_replay_as_contexts_grow() -> None:
    """One captured call replayed as a decode step replays it: the context grows by each
    step's accepted tokens and the block's query and keys are new. The first sequence starts
    one key short of the window, so its replays cross the point where masking stops."""
    spec = dict(batch=4, heads=16, block=3, window=2048, page=64, lengths=(2047, 5, 4096, 70000))
    gen = _gen(7)
    ctx = _Context(spec, 7)
    args = _attention_args(ctx, spec, gen)
    attention(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = attention(*args)
    query, block_kv = args[0], args[5]
    for step in range(8):
        for b in range(ctx.batch):
            ctx.grow(b, 1 + (step + b) % 3, gen)
        query.copy_(_randn(*query.shape, gen=gen))
        block_kv.copy_(_randn(*block_kv.shape, gen=gen))
        graph.replay()
        assert torch.equal(out, attention(*args))
        _check_attention(out, args)


# --- triton_latent_tail -----------------------------------------------------------------------


class _LatentTail:
    """The target's two call shapes. A draft layer writes `[batch, 3, 576]` and the rope tail of
    a `[batch, 3, heads, 576]` query; a context-features layer writes its slice of the
    `[rows, 3, 1, 576]` buffer, whose other layers must keep what they held."""

    def __init__(self, spec, seed: int, table: torch.Tensor):
        gen = _gen(seed)
        rows = self.rows = spec["rows"]
        self.kv = _randn(rows, LAT + PE, gen=gen, std=3.0)
        self.weight = (1 + 0.1 * torch.randn(1, LAT, device="cuda", generator=gen)).to(
            torch.bfloat16
        )
        self.positions = torch.randint(0, ROPE_ROWS, (rows,), device="cuda", generator=gen)
        self.table = table
        self.with_q = spec["with_q"]
        if self.with_q:
            self.heads, self.nope = spec["heads"], 128
            self.q = _randn(rows, self.heads * (self.nope + PE), gen=gen)
            self.query = _randn(rows // 3, 3, self.heads, LAT + PE, gen=gen)
            self.buffer = torch.full(
                (rows // 3, 3, LAT + PE), float("nan"), dtype=torch.bfloat16, device="cuda"
            )
        else:
            self.buffer = torch.full(
                (rows, 3, 1, LAT + PE), float("nan"), dtype=torch.bfloat16, device="cuda"
            )

    def args(self, buffer: torch.Tensor, query: torch.Tensor | None = None) -> dict:
        args = dict(kv=self.kv, weight=self.weight, eps=EPS, positions=self.positions)
        if not self.with_q:
            return dict(args, table=self.table, out=buffer[:, 1])
        return dict(args, table=self.table, out=buffer, q=self.q, query=query, nope=self.nope)

    def check(self, buffer: torch.Tensor, query: torch.Tensor | None, before) -> None:
        """The rope is bitwise; the RMSNorm within one bf16 ulp of the fp32 reference, which is
        up to 2 of the row-maximum units `assert_within_ulp` counts, and 1/4 ulp in RMS.
        Measured on VR200, 6 draws a cell: no element differed at decode sizes and 15 to 28 of
        8.4M by 1 ulp in the prefill cell (1.42 units, 0.003 in RMS); 3460 of 1.1e9 in the
        served model. Nothing else in the buffer or the query moves."""
        expected_buffer, expected_query = before[0].clone(), None
        if self.with_q:
            expected_query = before[1].clone()
        latent_tail.reference(**self.args(expected_buffer, expected_query))
        out, ref = buffer, expected_buffer
        if not self.with_q:
            out, ref = buffer[:, 1], expected_buffer[:, 1]
            others = torch.cat((buffer[:, :1], buffer[:, 2:]), dim=1)
            assert others.isnan().all()
        assert torch.equal(out[..., LAT:], ref[..., LAT:])
        assert_within_ulp(out[..., :LAT], ref[..., :LAT], element_ulp=2, rms_ulp=0.25)
        if self.with_q:
            assert torch.equal(query[..., LAT:], expected_query[..., LAT:])
            assert torch.equal(query[..., :LAT], before[1][..., :LAT])

    def snapshot(self):
        return self.buffer.clone(), self.query.clone() if self.with_q else None


@pytest.mark.parametrize("cell", latent_tail.CELLS, ids=[cell.why for cell in latent_tail.CELLS])
@pytest.mark.parametrize("draw", range(3))
def test_latent_tail_cells(cell, draw, certifying) -> None:  # noqa: F811
    certifying(latent_tail)
    case = _LatentTail(cell.spec, cell.seed_base() + 2 * draw, _rope_table())
    before = case.snapshot()
    query = case.query if case.with_q else None
    latent_tail(**case.args(case.buffer, query))
    case.check(case.buffer, query, before)


@pytest.mark.parametrize("with_q", [True, False], ids=["draft layer", "context features"])
def test_latent_tail_graph_replay(with_q) -> None:
    spec = dict(rows=12, heads=16, with_q=with_q)
    table = _rope_table()
    case = _LatentTail(spec, 3, table)
    query = case.query if with_q else None
    latent_tail(**case.args(case.buffer, query))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        latent_tail(**case.args(case.buffer, query))
    for seed in range(10):
        fresh = _LatentTail(spec, 100 + seed, table)
        for name in ("kv", "positions", "q", "query", "buffer"):
            if hasattr(case, name):
                getattr(case, name).copy_(getattr(fresh, name))
        before = case.snapshot()
        graph.replay()
        case.check(case.buffer, query, before)


# --- the Markov head --------------------------------------------------------------------------


def _markov_step(batch: int, first: bool, seed: int):
    """One recurrent step's inputs: the previous tokens (one off the vocabulary at either end
    when the batch has room), the embedding, the state, and one block slot of the hidden
    states, a strided `[batch, hidden]` view as `hidden_states[:, i]` is."""
    gen = _gen(seed)
    previous = torch.randint(0, VOCAB, (batch,), device="cuda", generator=gen)
    if batch > 1:
        previous[0] = -1
        previous[-1] = VOCAB
    embedding = _randn(VOCAB, MARKOV_RANK, gen=gen)
    state = None if first else _randn(batch, MARKOV_RANK, gen=gen)
    hidden = _randn(batch, 3, HIDDEN, gen=gen)[:, 1]
    joint = _randn(batch, 3 * MARKOV_RANK, gen=gen, std=4.0)
    return previous, embedding, state, hidden, joint


def _bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("cell", markov_input.CELLS, ids=[cell.why for cell in markov_input.CELLS])
def test_markov_input_cells(cell, certifying) -> None:  # noqa: F811
    """Copies only, so bitwise; a token off the vocabulary embeds as zeros."""
    certifying(markov_input)
    batch, first = cell.spec["batch"], cell.spec["first"]
    previous, embedding, state, hidden, _ = _markov_step(batch, first, cell.seed_base())
    out = markov_input(previous, embedding, state, hidden)
    assert _bitwise(out, markov_input.reference(previous, embedding, state, hidden))
    if batch > 1:
        assert not out[0, MARKOV_RANK : 2 * MARKOV_RANK].any()
        assert not out[-1, MARKOV_RANK : 2 * MARKOV_RANK].any()


@pytest.mark.parametrize("cell", markov_gate.CELLS, ids=[cell.why for cell in markov_gate.CELLS])
def test_markov_gate_cells(cell, certifying) -> None:  # noqa: F811
    """Every op rounds to bf16 where the torch chain does, so bitwise."""
    certifying(markov_gate)
    _, _, state, _, joint = _markov_step(cell.spec["batch"], cell.spec["first"], cell.seed_base())
    new_state, emitted = markov_gate(joint, state)
    ref_state, ref_emitted = markov_gate.reference(joint, state)
    assert _bitwise(new_state, ref_state) and _bitwise(emitted, ref_emitted)


def _logits_inputs(batch: int, seed: int):
    """Three block slots of fp32 base logits and one bf16 bias, written into a strided
    `refined[:, i]` as the chain writes them. Each row's maximum is tied at two indices in
    different 4096-column chunks and once more within a chunk."""
    gen = _gen(seed)
    base = torch.randn(batch, 3, VOCAB, device="cuda", generator=gen) * 5
    bias = _randn(batch, VOCAB, gen=gen)
    for column in (1000, 1001, 150000):
        base[:, :, column] = 100.0
        bias[:, column] = 0
    return base, bias


@pytest.mark.parametrize(
    "cell", markov_logits.CELLS, ids=[cell.why for cell in markov_logits.CELLS]
)
def test_markov_logits_cells(cell, certifying) -> None:  # noqa: F811
    """The sums are bitwise and the argmax takes the first of tied maxima, as torch.argmax."""
    certifying(markov_logits)
    base, bias = _logits_inputs(cell.spec["batch"], cell.seed_base())
    refined = torch.empty_like(base)
    expected = torch.empty_like(base)
    for i in range(3):
        tokens = markov_logits(base[:, i], bias, refined[:, i])
        ref_tokens = markov_logits.reference(base[:, i], bias, expected[:, i])
        assert tokens.dtype == torch.int64 and torch.equal(tokens, ref_tokens)
        assert bool((tokens == 1000).all())
    assert torch.equal(refined, expected)


def test_markov_logits_argmax_untied_and_last_column() -> None:
    """Without ties: each row's maximum in a different chunk, the last column included."""
    batch = 4
    base, bias = _logits_inputs(batch, 11)
    base[:, :, (1000, 1001, 150000)] = 0
    winners = torch.tensor([0, 4095, 4096, VOCAB - 1], device="cuda")
    base[torch.arange(batch), 0, winners] = 1000.0
    refined = torch.empty_like(base)
    tokens = markov_logits(base[:, 0], bias, refined[:, 0])
    assert torch.equal(tokens, winners)
    assert torch.equal(tokens, torch.argmax(base[:, 0] + bias, dim=-1))


def test_markov_chain_graph_replay() -> None:
    """One recurrent step captured whole, replayed on fresh inputs: the input, the gate and the
    logits all match the torch chain bitwise."""
    batch = 4
    previous, embedding, state, hidden, joint = _markov_step(batch, False, 21)
    base, bias = _logits_inputs(batch, 22)
    refined = torch.empty_like(base)

    def step():
        joint_in = markov_input(previous, embedding, state, hidden)
        new_state, emitted = markov_gate(joint, state)
        return joint_in, new_state, emitted, markov_logits(base[:, 0], bias, refined[:, 0])

    step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = step()
    for seed in range(10):
        fresh_previous, _, fresh_state, fresh_hidden, fresh_joint = _markov_step(batch, False, seed)
        fresh_base, fresh_bias = _logits_inputs(batch, 50 + seed)
        for dst, src in (
            (previous, fresh_previous),
            (state, fresh_state),
            (hidden, fresh_hidden),
            (joint, fresh_joint),
            (base, fresh_base),
            (bias, fresh_bias),
        ):
            dst.copy_(src)
        graph.replay()
        joint_in, new_state, emitted, tokens = outputs
        assert _bitwise(joint_in, markov_input.reference(previous, embedding, state, hidden))
        ref_state, ref_emitted = markov_gate.reference(joint, state)
        assert _bitwise(new_state, ref_state) and _bitwise(emitted, ref_emitted)
        expected = torch.empty_like(base)
        assert torch.equal(tokens, markov_logits.reference(base[:, 0], bias, expected[:, 0]))
        assert torch.equal(refined[:, 0], expected[:, 0])
