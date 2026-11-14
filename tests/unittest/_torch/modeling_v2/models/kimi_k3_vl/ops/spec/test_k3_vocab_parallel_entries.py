# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vocab-parallel selection entries (``ops/spec``) over their certified cells: one rank of a
vocab-sharded head on one GPU, the exchange emulated as the sum of every rank's zero-filled pair
buffer (what the MNNVL all-reduce computes)."""

import types
import zlib

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.dspark.worker import (  # noqa: E402, E501
    _PairExchange,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.spec import (  # noqa: E402, E501
    _reference,
    k3_markov_add_local_argmax,
    k3_markov_reduce_embed,
    k3_vocab_local_select,
    k3_vocab_reduce_pairs,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

VOCAB = 163840
MARKOV_RANK = 256
_ARCH = "sm_{}{}".format(*torch.cuda.get_device_capability())


def _require_arch(op) -> None:
    if _ARCH not in {a.value for a in op.ARCHS}:
        pytest.skip(f"{type(op).__name__} is not certified on {_ARCH}")


def _gen(cell) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)


def _layout(tp: int, rank: int, rows: int, shard: int, sentinel: float = 3.0):
    """The MNNVL exchange's layout of ``rank``, its buffer filled with ``sentinel`` so that the
    zero fill shows."""
    exchange = types.SimpleNamespace(backend="mnnvl", tp_size=tp, tp_rank=rank)
    layout = _PairExchange.layout(exchange, rows, shard, torch.device("cuda"))
    layout.buffer.fill_(sentinel)
    return layout


def _logits(rows: int, cols: int, gen, ties: bool) -> torch.Tensor:
    """BF16 logits of the head's scale; with ``ties`` some rows repeat their maximum in other
    columns (other splits and shards) and some hold NaNs."""
    x = (2.0 * torch.randn(rows, cols, device="cuda", generator=gen)).to(torch.bfloat16)
    if ties:
        cols_at = torch.randint(0, cols, (rows, 3), device="cuda", generator=gen)
        for r in range(0, rows, 3):
            top = x[r].float().max() + 1.0
            x[r, cols_at[r]] = top.to(torch.bfloat16)
        for r in range(1, rows, 7):
            x[r, cols_at[r, :2]] = float("nan")
    return x


def _assert_pairs_equal(out: torch.Tensor, ref: torch.Tensor) -> None:
    # Exact, NaN == NaN (the kernel keeps the logit's NaN payload) and -0 == +0.
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0, equal_nan=True)


def _own_slots(buf: torch.Tensor, layout) -> tuple[torch.Tensor, torch.Tensor]:
    cols = buf[:, 2 * layout.slot0 : 2 * (layout.slot0 + layout.splits)]
    return cols[:, 0::2], cols[:, 1::2].to(torch.int64)


# ---------------------------------------------------------------------------------------------
# k3_vocab_local_select
# ---------------------------------------------------------------------------------------------

_SELECT = k3_vocab_local_select.k3_vocab_local_select


@pytest.mark.parametrize("cell", _SELECT.CELLS, ids=[c.why[:48] for c in _SELECT.CELLS])
def test_vocab_local_select_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    _require_arch(_SELECT)
    certifying(_SELECT)
    spec, gen = cell.spec, _gen(cell)
    tp, rank, rows = spec["tp"], spec["rank"], spec["rows"]
    shard = VOCAB // tp
    col_base = rank * shard
    local = _logits(rows, shard, gen, spec["ties"])
    layout = _layout(tp, rank, rows, shard)
    before = layout.buffer.clone()
    local_before = local.clone()
    sampling = None
    if spec["sampling"]:
        from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD

        temperatures = 0.3 + 1.2 * torch.rand(rows, device="cuda", generator=gen)
        temperatures[::5] = 0.0  # greedy rows inside a sampled batch
        seeds = torch.randint(0, 2**62, (rows,), device="cuda", generator=gen)
        offsets = torch.randint(0, 2**40, (rows,), device="cuda", generator=gen)
        sampling = (temperatures, seeds, offsets, float(GREEDY_TEMPERATURE_THRESHOLD))

    _SELECT(local, col_base, layout, sampling)
    # Bitwise: the tied cells hold NaNs.
    assert torch.equal(local.view(torch.int16), local_before.view(torch.int16)), "logits written"
    ref = _SELECT.reference(local, col_base, layout._replace(buffer=before), sampling)
    out = layout.buffer
    # Columns outside this rank's slots: zero (the all-reduce sums them away).
    own = torch.zeros_like(out, dtype=torch.bool)
    own[:, layout.zero_lo : layout.zero_hi] = True
    assert torch.equal(out[~own], torch.zeros_like(out[~own])), "other ranks' columns not zeroed"
    if sampling is None:
        _assert_pairs_equal(out, ref)
        return
    _assert_sampled_pairs(out, ref, layout, local, col_base, sampling)


def _assert_sampled_pairs(out, ref, layout, local, col_base, sampling) -> None:
    """The kernel scores in FP32, the reference in FP64. A score is at most about 32 in
    magnitude (logits / T with T >= 0.3, plus Gumbel noise below 17) and the kernel's carries
    at most four roundings of 1 ulp at that scale (the temperature product, the two logs of the
    noise and the sum): 2**-16. The pairs agree within 2**-14 (4x that), and where the winning
    columns differ the kernel's must score within 2**-14 of the reference's winner in the same
    split (a near-tie the FP32 evaluation resolves the other way)."""
    tol = 2.0**-14
    k_val, k_idx = _own_slots(out, layout)
    r_val, r_idx = _own_slots(ref, layout)
    scores = k3_vocab_local_select.sampled_scores(local, col_base, sampling)
    same = k_idx == r_idx
    torch.testing.assert_close(k_val[same], r_val[same], rtol=0.0, atol=tol)
    if not bool(same.all()):
        rows, splits = (~same).nonzero(as_tuple=True)
        k_cols, r_cols = k_idx[rows, splits] - col_base, r_idx[rows, splits] - col_base
        assert bool((k_cols // layout.chunk == splits).all()), "a winner outside its split"
        gap = scores[rows, r_cols] - scores[rows, k_cols]
        assert float(gap.max()) <= tol, f"winners differ beyond a near-tie: {gap.max():.3e}"
        # A near-tie is rare: a handful of the row x split winners at most.
        assert int((~same).sum()) <= max(2, same.numel() // 100)


# ---------------------------------------------------------------------------------------------
# k3_vocab_reduce_pairs
# ---------------------------------------------------------------------------------------------

_REDUCE = k3_vocab_reduce_pairs.k3_vocab_reduce_pairs


def _exchanged_pairs(values: torch.Tensor, tp: int) -> tuple[torch.Tensor, object]:
    """Every rank's published pairs of ``values`` ``[rows, vocab]`` summed: the exchange's
    result (interleaved slots: row stride ``stride_out``, slot stride 2)."""
    rows, vocab = values.shape
    shard = vocab // tp
    total, layout = None, None
    for rank in range(tp):
        layout = _layout(tp, rank, rows, shard, sentinel=0.0)
        part = _reference.publish_pairs(
            layout, values[:, rank * shard : (rank + 1) * shard].float(), rank * shard
        )
        total = part if total is None else total + part
    return total, layout


@pytest.mark.parametrize("cell", _REDUCE.CELLS, ids=[c.why[:48] for c in _REDUCE.CELLS])
def test_vocab_reduce_pairs_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    _require_arch(_REDUCE)
    certifying(_REDUCE)
    spec, gen = cell.spec, _gen(cell)
    tp, rows = spec["tp"], spec["rows"]
    full = _logits(rows, VOCAB, gen, spec["ties"])
    pairs, layout = _exchanged_pairs(full, tp)
    n_slots = tp * layout.splits
    pairs_before = pairs.clone()
    out_dtype = getattr(torch, spec["out_dtype"])
    out = torch.full((rows,), -5, dtype=out_dtype, device="cuda")
    ret = _REDUCE(pairs, layout.stride_out, 2, n_slots, out)
    assert ret is out
    assert torch.equal(pairs.view(torch.int32), pairs_before.view(torch.int32)), "pairs written"
    ref = _REDUCE.reference(pairs, layout.stride_out, 2, n_slots, out)
    assert ref.dtype == out_dtype and torch.equal(out, ref)
    # Cross-check: the argmax of the gathered logits.
    cols = torch.arange(VOCAB, device="cuda").expand(rows, -1)
    _, gathered = _reference.argmax_pairs(full.float(), cols)
    assert torch.equal(out.to(torch.int64), gathered)
    if not spec["ties"]:
        assert torch.equal(out.to(torch.int64), torch.argmax(full.float(), dim=-1))


# ---------------------------------------------------------------------------------------------
# The multi-launch Markov chain: k3_markov_reduce_embed and k3_markov_add_local_argmax
# ---------------------------------------------------------------------------------------------

_EMBED = k3_markov_reduce_embed.k3_markov_reduce_embed
_ADD = k3_markov_add_local_argmax.k3_markov_add_local_argmax


def _markov_w1(gen) -> torch.Tensor:
    # [vocab, R] with entries of 0.25: a bias GEMV over R = 256 of unit scale.
    return (0.25 * torch.randn(VOCAB, MARKOV_RANK, device="cuda", generator=gen)).to(torch.bfloat16)


@pytest.mark.parametrize("cell", _EMBED.CELLS, ids=[c.why[:48] for c in _EMBED.CELLS])
def test_markov_reduce_embed_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    _require_arch(_EMBED)
    certifying(_EMBED)
    spec, gen = cell.spec, _gen(cell)
    tp, reqs, block, step = spec["tp"], spec["reqs"], spec["block"], spec["step"]
    markov_w1 = _markov_w1(gen)
    anchors = torch.randint(0, VOCAB, (reqs,), device="cuda", generator=gen, dtype=torch.int64)
    anchors[3] = -1
    anchors[7] = VOCAB
    emb = torch.full((reqs, MARKOV_RANK), 9.0, dtype=torch.bfloat16, device="cuda")
    tokens = torch.full((reqs, block), -7, dtype=torch.int32, device="cuda")
    if step == 0:
        pairs, stride_row, stride_slot, n_slots = anchors, 0, 0, 1
        values = None
    else:
        values = _logits(reqs, VOCAB, gen, ties=False)
        pairs, layout = _exchanged_pairs(values, tp)
        stride_row, stride_slot, n_slots = layout.stride_out, 2, tp * layout.splits
    args = (pairs, stride_row, stride_slot, n_slots, anchors, markov_w1)
    ref_emb, ref_tokens = _EMBED.reference(*args, emb, tokens, VOCAB, step)
    _EMBED(*args, emb, tokens, VOCAB, step)
    assert torch.equal(emb, ref_emb) and torch.equal(tokens, ref_tokens)
    if step == 0:
        assert torch.equal(emb[3], torch.zeros_like(emb[3])), "an out-of-vocab anchor's row"
        assert torch.equal(tokens, torch.full_like(tokens, -7)), "step 0 stores no token"
    else:
        prev = torch.argmax(values.float(), dim=-1)
        assert torch.equal(tokens[:, step - 1].to(torch.int64), prev)
        assert torch.equal(emb, markov_w1[prev])
        others = [c for c in range(block) if c != step - 1]
        assert bool((tokens[:, others] == -7).all()), "only token step - 1 is stored"


@pytest.mark.parametrize("cell", _ADD.CELLS, ids=[c.why[:48] for c in _ADD.CELLS])
def test_markov_add_local_argmax_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    _require_arch(_ADD)
    certifying(_ADD)
    spec, gen = cell.spec, _gen(cell)
    tp, rank, reqs, block, step = (
        spec["tp"],
        spec["rank"],
        spec["reqs"],
        spec["block"],
        spec["step"],
    )
    shard = VOCAB // tp
    col_base = rank * shard
    local_block = _logits(reqs * block, shard, gen, ties=False).view(reqs, block, shard)
    w2_shard = (0.25 * torch.randn(shard, MARKOV_RANK, device="cuda", generator=gen)).to(
        torch.bfloat16
    )
    emb = (0.25 * torch.randn(reqs, MARKOV_RANK, device="cuda", generator=gen)).to(torch.bfloat16)
    bias = torch.nn.functional.linear(emb, w2_shard)
    step_logits = local_block[:, step]
    assert not step_logits.is_contiguous()
    layout = _layout(tp, rank, reqs, shard)
    before = layout.buffer.clone()
    block_before = local_block.clone()
    _ADD(step_logits, bias, col_base, layout)
    assert torch.equal(local_block, block_before), "the op must not write the logits"
    ref = _ADD.reference(step_logits, bias, col_base, layout._replace(buffer=before))
    _assert_pairs_equal(layout.buffer, ref)
