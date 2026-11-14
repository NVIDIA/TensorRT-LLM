# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The K3 speculative-tail entries (``ops/spec``) against their references, over their cells.

Every entry here is integer bookkeeping, a copy, or a masked store whose arithmetic is one
multiply by 0 or 1 and an optional FP8 conversion, so each written buffer must match its
reference bit for bit, including the rows and slots the op must leave alone. The selection
entry is also checked against ``KDAReplayState.record_acceptance`` for the replay record.
"""

import importlib
import zlib

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("the speculative-tail kernels need a GPU", allow_module_level=True)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KDAReplayState  # noqa: E402
from tensorrt_llm._torch.speculative.interface import (  # noqa: E402
    _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
    _FORCE_ACCEPT_RNG_POOL_SIZE,
    _FORCE_ACCEPT_RNG_SEED,
    _FORCE_ACCEPT_RNG_SLOT_STRIDE,
)

_OPS = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.spec"


def _entry(name):
    return getattr(importlib.import_module(f"{_OPS}.{name}"), name)


k3_draft_block_table = _entry("k3_draft_block_table")
k3_fused_strict_accept = _entry("k3_fused_strict_accept")
k3_fused_select_accept = _entry("k3_fused_select_accept")
k3_next_new_tokens = _entry("k3_next_new_tokens")
k3_dflash_ctx_positions = _entry("k3_dflash_ctx_positions")
k3_dflash_gen_prepare = _entry("k3_dflash_gen_prepare")
k3_dflash_paged_latent_prepare = _entry("k3_dflash_paged_latent_prepare")
k3_dflash_noise_embedding = _entry("k3_dflash_noise_embedding")
k3_store_context_latents = _entry("k3_store_context_latents")
k3_scatter_rows_to_pool = _entry("k3_scatter_rows_to_pool")
k3_vocab_shard_embedding = _entry("k3_vocab_shard_embedding")

_ARCH = "sm_{}{}".format(*torch.cuda.get_device_capability())
if not any(
    _ARCH in {a.value for a in type(op).ARCHS}
    for op in (k3_draft_block_table, k3_fused_select_accept, k3_dflash_gen_prepare)
):
    pytest.skip(f"no speculative-tail entry is certified on {_ARCH}", allow_module_level=True)

VOCAB = 163840
HIDDEN = 7168
LATENT = 576  # kv_lora 512 + rope 64
LAYERS = 5
PAGE = 128
MAX_CTX = 997696
EMPTY_INDEX = 2**24


def _ids(op):
    return [c.why[:44] for c in op.CELLS]


def _gen(cell) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)


def _require(op) -> None:
    if _ARCH not in {a.value for a in type(op).ARCHS}:
        pytest.skip(f"{type(op).__name__} is not certified on {_ARCH}")


def _randint(low, high, shape, gen, dtype=torch.int64):
    return torch.randint(low, high, shape, generator=gen, device="cuda", dtype=dtype)


def _bits(t: torch.Tensor) -> torch.Tensor:
    """The tensor's raw bits, so that -0.0 and +0.0 differ."""
    width = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return t.contiguous().view(width)


def _same(out: torch.Tensor, ref: torch.Tensor, what: str) -> None:
    assert out.dtype == ref.dtype, (what, out.dtype, ref.dtype)
    assert out.shape == ref.shape, (what, tuple(out.shape), tuple(ref.shape))
    assert torch.equal(_bits(out), _bits(ref)), f"{what} differs from the reference"


def _force(cell, draft_len: int, gen) -> tuple:
    """The ``force`` operand ``K3DFlashWorker._force_accept_kernel_args`` builds, and the counter."""
    value = float(cell.spec["force"])
    if value == 0.0:
        return None, None
    int_part = int(value)
    max_total = draft_len + 1
    base_total = min(int_part + 1, max_total)
    frac = value - int_part
    if frac == 0.0 or base_total >= max_total:
        return (base_total,), None
    cpu = torch.Generator(device="cpu").manual_seed(_FORCE_ACCEPT_RNG_SEED)
    pool = torch.rand(_FORCE_ACCEPT_RNG_POOL_SIZE, dtype=torch.float32, generator=cpu).cuda()
    counter = _randint(0, 1000, (1,), gen)
    force = (
        base_total,
        frac,
        max_total,
        pool,
        counter,
        _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
        _FORCE_ACCEPT_RNG_SLOT_STRIDE,
    )
    return force, counter


def _drafts_agreeing(target: torch.Tensor, draft_len: int, gen) -> torch.Tensor:
    """``[G, K]`` int32 drafts that match ``target[:, :K]`` on a random prefix per row, behind
    a wider row stride."""
    num_gens = target.shape[0]
    agree = _randint(0, draft_len + 1, (num_gens,), gen)
    cols = torch.arange(draft_len, device="cuda")
    wrong = (target[:, :draft_len] + 1) % VOCAB
    picked = torch.where(cols[None, :] < agree[:, None], target[:, :draft_len], wrong)
    buf = torch.zeros((num_gens, draft_len + 3), dtype=torch.int32, device="cuda")
    buf[:, :draft_len] = picked.to(torch.int32)
    return buf[:, :draft_len]


# --------------------------------------------------------------------------------------------
# Page tables, acceptance and next tokens
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cell", k3_draft_block_table.CELLS, ids=_ids(k3_draft_block_table))
def test_k3_draft_block_table_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    _require(k3_draft_block_table)
    certifying(k3_draft_block_table)
    g = _gen(cell)
    s = cell.spec
    rows, blocks, divisor, block_offset = s["rows"], s["blocks"], s["divisor"], s["block_offset"]
    # The metadata's [pools, rows, K/V, blocks] offsets; the op reads plane 0 of pool 1.
    meta = _randint(-5, 2**20, (2, rows + 3, 2, blocks), g, torch.int32)
    held = _randint(0, blocks + 1, (rows,), g)
    held[0] = blocks
    if rows > 1:
        held[1] = 0
    encoded = _randint(0, 2**18, (rows, blocks), g) * divisor + block_offset
    cols = torch.arange(blocks, device="cuda")
    # Unallocated columns hold the -1 placeholder; one allocated column sits below the offset.
    plane = torch.where(cols[None, :] < held[:, None], encoded, -1)
    plane[0, 0] = block_offset - 1
    meta[1, :rows, 0] = plane.to(torch.int32)
    src = meta[1, :rows, 0]
    meta_before = meta.clone()
    num_slots = rows + 1
    table = torch.full((num_slots, blocks), -7, dtype=torch.int32, device="cuda")
    counts = torch.full((num_slots,), -7, dtype=torch.int64, device="cuda")
    partial = None
    if s["partial"]:
        chunks = -(-blocks // 2048)
        partial = torch.full((num_slots * chunks,), -7, dtype=torch.int32, device="cuda")

    ref_table, ref_counts = k3_draft_block_table.reference(
        src, divisor, table, counts, partial, block_offset=block_offset
    )
    k3_draft_block_table(src, divisor, table, counts, partial, block_offset=block_offset)
    _same(table, ref_table, "table")
    _same(counts, ref_counts, "counts")
    assert int(counts[rows]) == -7 and bool((table[rows] == -7).all()), "a row past rows moved"
    assert torch.equal(meta, meta_before), "the op must not write the metadata"


@pytest.mark.parametrize("cell", k3_fused_strict_accept.CELLS, ids=_ids(k3_fused_strict_accept))
def test_k3_fused_strict_accept_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    _require(k3_fused_strict_accept)
    certifying(k3_fused_strict_accept)
    g = _gen(cell)
    s = cell.spec
    num_contexts, num_gens, draft_len = s["num_contexts"], s["num_gens"], s["draft_len"]
    batch = num_contexts + num_gens
    width = draft_len + 1
    target = _randint(0, VOCAB, (num_contexts + num_gens * width,), g)
    draft = _drafts_agreeing(target[num_contexts:].view(num_gens, width), draft_len, g)
    force, counter = _force(cell, draft_len, g)
    before = (target.clone(), draft.clone())
    counter_before = None if counter is None else counter.clone()

    ref_accepted, ref_num = k3_fused_strict_accept.reference(
        target, draft, num_contexts, batch, force
    )
    accepted, num_accepted = k3_fused_strict_accept(target, draft, num_contexts, batch, force)
    _same(num_accepted, ref_num, "num_accepted")
    # Context rows define column 0 only.
    _same(accepted[:num_contexts, 0], ref_accepted[:num_contexts, 0], "context tokens")
    _same(accepted[num_contexts:], ref_accepted[num_contexts:], "generation tokens")
    if counter is not None:
        assert int(counter) == int(counter_before) + 1, "the force counter advances by one"
    assert torch.equal(target, before[0]) and torch.equal(draft, before[1])


def _pairs(cell, rows: int, gen) -> tuple[tuple, torch.Tensor]:
    """Every rank's interleaved (value, global index) pairs as the MNNVL exchange lays them out,
    and the token each row selects (largest value, lowest index among equal values)."""
    s = cell.spec
    tp, splits = s["tp"], s["splits"]
    n_slots = tp * splits
    stride = 1 << (2 * n_slots - 1).bit_length()
    shard = VOCAB // tp
    # Quarter steps make equal maxima across slots common; -0.0 against +0.0 in row 0.
    vals = torch.randn((rows, n_slots), generator=gen, device="cuda").mul(4).round().div(4)
    vals[0] = -1.0
    vals[0, n_slots // 2] = 0.0
    vals[0, n_slots // 3] = -0.0
    rank = torch.arange(n_slots, device="cuda") // splits
    idx = rank[None, :] * shard + _randint(0, shard, (rows, n_slots), gen)
    # A split without valid columns reports (-inf, EMPTY_INDEX).
    empty = torch.rand((rows, n_slots), generator=gen, device="cuda") < 0.05
    vals = torch.where(empty, float("-inf"), vals)
    idx = torch.where(empty, EMPTY_INDEX, idx)
    # Lanes past the slots hold NaN: the op must not read them.
    buf = torch.full((rows, stride), float("nan"), dtype=torch.float32, device="cuda")
    buf[:, 0 : 2 * n_slots : 2] = vals
    buf[:, 1 : 2 * n_slots : 2] = idx.to(torch.float32)
    best = vals.amax(dim=1, keepdim=True)
    selected = torch.where(vals == best, idx, 2**31 - 1).amin(dim=1)
    return (buf, stride, 2, n_slots), selected


@pytest.mark.parametrize("cell", k3_fused_select_accept.CELLS, ids=_ids(k3_fused_select_accept))
def test_k3_fused_select_accept_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    _require(k3_fused_select_accept)
    certifying(k3_fused_select_accept)
    g = _gen(cell)
    s = cell.spec
    num_contexts, num_gens, draft_len = s["num_contexts"], s["num_gens"], s["draft_len"]
    batch = num_contexts + num_gens
    width = draft_len + 1
    pairs, selected = _pairs(cell, num_contexts + num_gens * width, g)
    draft = _drafts_agreeing(selected[num_contexts:].view(num_gens, width), draft_len, g)
    force, counter = _force(cell, draft_len, g)
    counter_before = None if counter is None else counter.clone()

    kda_replay = kv_lens = None
    if s["replay"]:
        # The replay count is the first int32 of each slot's 16-byte record.
        records = _randint(0, 9, (3 * batch + 2, 4), g, torch.int32)
        perm = torch.randperm(records.shape[0], generator=g, device="cuda")
        state_indices = perm[:batch].to(torch.int32).contiguous()
        dummy = torch.zeros(batch, dtype=torch.bool, device="cuda")
        if num_gens > 1:
            dummy[num_contexts + 1] = True
        kda_replay = (records[:, 0], state_indices, dummy)
        kv_lens = _randint(10, 9000, (batch,), g, torch.int32)
        records_before = records.clone()
    kv_before = None if kv_lens is None else kv_lens.clone()
    pair_buf_before, draft_before = pairs[0].clone(), draft.clone()

    ref_accepted, ref_num, ref_rewind, ref_prev, ref_kv = k3_fused_select_accept.reference(
        pairs, draft, num_contexts, batch, force, kda_replay, kv_lens
    )
    accepted, num_accepted, rewind = k3_fused_select_accept(
        pairs, draft, num_contexts, batch, force, kda_replay=kda_replay, kv_lens=kv_lens
    )
    _same(num_accepted, ref_num, "num_accepted")
    _same(accepted[:num_contexts, 0], ref_accepted[:num_contexts, 0], "context tokens")
    _same(accepted[num_contexts:], ref_accepted[num_contexts:], "generation tokens")
    assert torch.equal(
        ref_accepted[num_contexts:, 0], selected[num_contexts::width].to(torch.int32)
    )
    if force is None:
        target = selected[num_contexts:].view(num_gens, width)
        assert torch.equal(ref_accepted[num_contexts:], target.to(torch.int32))
    if counter is not None:
        assert int(counter) == int(counter_before) + 1, "the force counter advances by one"
    if s["replay"]:
        _same(rewind, ref_rewind, "rewind")
        _same(kv_lens, ref_kv, "kv_lens")
        _same(records[:, 0], ref_prev, "replay count")
        assert torch.equal(records[:, 1:], records_before[:, 1:]), "the rest of each record moved"
        # Main's eager record of the same acceptance.
        main = records_before.clone()
        state = KDAReplayState(draft_len)
        state.prev_num_accepted_tokens = main[:, 0]
        state.record_acceptance(
            kda_replay[1][num_contexts:], ref_num[num_contexts:] - 1, kda_replay[2][num_contexts:]
        )
        _same(main[:, 0], ref_prev, "record_acceptance")
        assert torch.equal(kv_before[:num_contexts], kv_lens[:num_contexts])
    else:
        assert rewind is None and ref_rewind is None and ref_prev is None and ref_kv is None
    assert torch.equal(_bits(pairs[0]), _bits(pair_buf_before)) and torch.equal(draft, draft_before)


@pytest.mark.parametrize("cell", k3_next_new_tokens.CELLS, ids=_ids(k3_next_new_tokens))
def test_k3_next_new_tokens_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    _require(k3_next_new_tokens)
    certifying(k3_next_new_tokens)
    g = _gen(cell)
    s = cell.spec
    batch, draft_len = s["batch"], s["draft_len"]
    # The worker's [max_batch, K + 1] accepted buffer, of which the batch rows are passed.
    accepted = _randint(0, VOCAB, (batch + 2, draft_len + 1), g, torch.int32)[:batch]
    num_accepted = _randint(1, draft_len + 2, (batch,), g, torch.int32)
    drafts = _randint(0, VOCAB, (batch, draft_len), g, torch.int32)
    kv_lens = rewind = None
    if s["rewind"]:
        kv_lens = _randint(100, 9000, (batch,), g, torch.int32)
        rewind = (1 - num_accepted).contiguous()
    before = (accepted.clone(), num_accepted.clone(), drafts.clone())

    ref_out, ref_kv = k3_next_new_tokens.reference(accepted, drafts, num_accepted, kv_lens, rewind)
    out = k3_next_new_tokens(accepted, drafts, num_accepted, kv_lens, rewind)
    _same(out, ref_out, "next new tokens")
    if kv_lens is not None:
        _same(kv_lens, ref_kv, "kv_lens")
    assert all(torch.equal(a, b) for a, b in zip((accepted, num_accepted, drafts), before))


# --------------------------------------------------------------------------------------------
# The DFlash generation step's context bookkeeping
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cell", k3_dflash_ctx_positions.CELLS, ids=_ids(k3_dflash_ctx_positions))
def test_k3_dflash_ctx_positions_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    _require(k3_dflash_ctx_positions)
    certifying(k3_dflash_ctx_positions)
    g = _gen(cell)
    s = cell.spec
    num_gens, width, max_batch = s["num_gens"], s["width"], s["max_batch"]
    dummy = max_batch
    ctx_len = _randint(0, MAX_CTX - width, (max_batch + 1,), g)
    ctx_len[dummy] = 0
    slots = torch.randperm(max_batch, generator=g, device="cuda")[:num_gens].contiguous()
    if num_gens > 2:
        slots[-2:] = dummy  # graph padding rows share the dummy slot
    before = ctx_len.clone()
    ref = k3_dflash_ctx_positions.reference(ctx_len, slots, width)
    out = k3_dflash_ctx_positions(ctx_len, slots, width)
    _same(out, ref, "positions")
    assert torch.equal(ctx_len, before), "the op must not write ctx_len"


def _step_state(cell, num_rows: int, first_row: int, cp_size: int, gen) -> dict:
    """A generation step's worker state: context lengths per slot, the decoded page table and
    its counts per batch row, the accepted tokens. Generation request ``g`` is batch row
    ``first_row + g``; the last ``dummies`` requests are padding on the dummy slot. One request
    sits at the context ceiling and one has fewer pages than its block needs."""
    s = cell.spec
    num_gens, width = s["num_gens"], s["draft_len"] + 1
    max_batch = num_gens + 2
    dummy = max_batch
    live = num_gens - s["dummies"]
    slots = torch.randperm(max_batch, generator=gen, device="cuda")[:num_gens].contiguous()
    slots[live:] = dummy
    ctx_len = _randint(0, 60000, (max_batch + 1,), gen)
    ctx_len[dummy] = 0
    ctx_len[slots[0]] = MAX_CTX - 3
    old = ctx_len[slots]
    span = PAGE * cp_size
    counts = torch.zeros(num_rows + 2, dtype=torch.int64, device="cuda")
    rows = first_row + torch.arange(num_gens, device="cuda")
    need = (old + 2 * width + span - 1) // span
    counts[rows] = need + _randint(0, 2, (num_gens,), gen)
    if live > 1:
        counts[first_row + 1] = (old[1] + span - 1) // span  # the block does not fit
    max_blocks = int(counts.max()) + 1
    tables = _randint(0, 2**20, (num_rows + 2, max_blocks + 3), gen, torch.int32)[:, :max_blocks]
    return dict(
        slots=slots,
        ctx_len=ctx_len,
        counts=counts,
        tables=tables,
        dummy=dummy,
        num_accepted=_randint(1, width + 1, (num_rows + 2,), gen, torch.int32),
        accepted=_randint(0, VOCAB, (num_rows + 2, width), gen, torch.int32),
        seq_lens=_randint(1, 9000, (num_rows + 2,), gen, torch.int32),
    )


@pytest.mark.parametrize("cell", k3_dflash_gen_prepare.CELLS, ids=_ids(k3_dflash_gen_prepare))
def test_k3_dflash_gen_prepare_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    _require(k3_dflash_gen_prepare)
    certifying(k3_dflash_gen_prepare)
    g = _gen(cell)
    s = cell.spec
    num_gens, width, cp_size = s["num_gens"], s["draft_len"] + 1, s["cp_size"]
    st = _step_state(cell, num_gens, 0, cp_size, g)
    args = (
        st["num_accepted"],
        st["accepted"],
        st["slots"],
        st["ctx_len"],
        st["counts"],
        st["tables"],
        st["seq_lens"],
        0,
        num_gens,
        st["dummy"],
        MAX_CTX,
        163837,
        PAGE,
        width,
        width,
    )
    before = {k: st[k].clone() for k in ("num_accepted", "accepted", "slots", "counts", "tables")}
    ref, ref_ctx, ref_seq = k3_dflash_gen_prepare.reference(
        *args, cp_size=cp_size, cp_rank=s["cp_rank"]
    )
    out = k3_dflash_gen_prepare(*args, cp_size=cp_size, cp_rank=s["cp_rank"])
    assert set(out) == set(ref), (sorted(out), sorted(ref))
    for name in out:
        _same(out[name], ref[name], name)
    _same(st["ctx_len"], ref_ctx, "ctx_len")
    _same(st["seq_lens"], ref_seq, "seq_lens_cuda")
    assert all(torch.equal(st[k], v) for k, v in before.items()), "an input moved"


@pytest.mark.parametrize(
    "cell", k3_dflash_paged_latent_prepare.CELLS, ids=_ids(k3_dflash_paged_latent_prepare)
)
def test_k3_dflash_paged_latent_prepare_cells(cell, certifying) -> None:  # noqa: F811
    _require(k3_dflash_paged_latent_prepare)
    certifying(k3_dflash_paged_latent_prepare)
    g = _gen(cell)
    s = cell.spec
    num_gens, width, first_row = s["num_gens"], s["draft_len"] + 1, s["first_row"]
    st = _step_state(cell, first_row + num_gens, first_row, 1, g)
    args = (
        st["num_accepted"],
        st["accepted"],
        st["slots"],
        st["ctx_len"],
        st["counts"],
        st["tables"],
        st["seq_lens"],
        first_row,
        st["dummy"],
        MAX_CTX,
        PAGE,
        width,
        width,
    )
    ref, ref_ctx, ref_seq = k3_dflash_paged_latent_prepare.reference(*args)
    out = k3_dflash_paged_latent_prepare(*args)
    assert type(out) is type(ref)
    for name in out._fields:
        _same(getattr(out, name), getattr(ref, name), name)
    _same(st["ctx_len"], ref_ctx, "ctx_len")
    _same(st["seq_lens"], ref_seq, "seq_lens_cuda")


@pytest.mark.parametrize(
    "cell", k3_dflash_noise_embedding.CELLS, ids=_ids(k3_dflash_noise_embedding)
)
def test_k3_dflash_noise_embedding_cells(cell, certifying) -> None:  # noqa: F811
    _require(k3_dflash_noise_embedding)
    certifying(k3_dflash_noise_embedding)
    g = _gen(cell)
    s = cell.spec
    combined = torch.randn((s["num_gens"] + 1, HIDDEN), generator=g, device="cuda")
    combined = combined.to(torch.bfloat16)
    combined[0, :3] = -0.0
    before = combined.clone()
    ref = k3_dflash_noise_embedding.reference(combined, s["block_size"])
    out = k3_dflash_noise_embedding(combined, s["block_size"])
    _same(out, ref, "noise embedding")
    assert torch.equal(_bits(combined), _bits(before))


# --------------------------------------------------------------------------------------------
# Paged-pool stores and the vocab-shard lookup
# --------------------------------------------------------------------------------------------


def _latent_rows(shape, gen) -> torch.Tensor:
    """BF16 latents with a spread of magnitudes, a few past the FP8 E4M3 range (448)."""
    x = torch.randn(shape, generator=gen, device="cuda")
    x = x * torch.logspace(-3, 1, shape[-1], device="cuda")
    x[..., :4] *= 300.0
    return x.to(torch.bfloat16)


def _arena(pages: int, dtype, gen) -> torch.Tensor:
    """Every layer's pages interleaved, ``[pages, layers, 1, page, D]``."""
    x = torch.randn((pages, LAYERS, 1, PAGE, LATENT), generator=gen, device="cuda")
    return x.to(dtype)


def _distinct_slots(n: int, pages: int, gen) -> tuple[torch.Tensor, torch.Tensor]:
    flat = torch.randperm(pages * PAGE, generator=gen, device="cuda")[:n]
    return flat // PAGE, flat % PAGE


@pytest.mark.parametrize("cell", k3_store_context_latents.CELLS, ids=_ids(k3_store_context_latents))
def test_k3_store_context_latents_cells(cell, certifying) -> None:  # noqa: F811
    _require(k3_store_context_latents)
    certifying(k3_store_context_latents)
    g = _gen(cell)
    s = cell.spec
    num_tokens = s["num_gens"] * s["width"]
    # The drafter's layer-major latents, read through the [N, L, D] transpose.
    latent = _latent_rows((LAYERS, num_tokens, LATENT), g).transpose(0, 1)
    dtype = torch.float8_e4m3fn if s["pool"] == "fp8" else torch.bfloat16
    arena = _arena(24, dtype, g)
    pools = [arena[:, layer : layer + 1] for layer in range(LAYERS)]
    pool_ptrs = torch.tensor([p.data_ptr() for p in pools], dtype=torch.int64, device="cuda")
    pages, offsets = _distinct_slots(num_tokens, arena.shape[0], g)
    # 1 for an accepted token, 0 past the accepted prefix, -1 on another rank's page.
    mask = _randint(0, 2, (num_tokens,), g, torch.int8)
    if s["cp_size"] > 1:
        other = _randint(0, s["cp_size"], (num_tokens,), g) != 0
        mask = torch.where(other, torch.full_like(mask, -1), mask)
    latent_before = latent.clone()

    ref = k3_store_context_latents.reference(
        latent, pools, pool_ptrs, pages, offsets, mask, mla=s["mla"]
    )
    k3_store_context_latents(latent, pools, pool_ptrs, pages, offsets, mask, mla=s["mla"])
    for layer, (pool, expected) in enumerate(zip(pools, ref)):
        _same(pool, expected, f"pool of layer {layer}")
    assert torch.equal(_bits(latent), _bits(latent_before))


@pytest.mark.parametrize("cell", k3_scatter_rows_to_pool.CELLS, ids=_ids(k3_scatter_rows_to_pool))
def test_k3_scatter_rows_to_pool_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    _require(k3_scatter_rows_to_pool)
    certifying(k3_scatter_rows_to_pool)
    g = _gen(cell)
    s = cell.spec
    rows = s["rows"]
    src = _latent_rows((rows, LATENT), g)
    arena = _arena(16, torch.float8_e4m3fn, g)
    pool = arena[:, 2, 0]  # one layer's [pages, page, D] view of the interleaved pool
    pages, offsets = _distinct_slots(rows, arena.shape[0], g)
    rows_mask = None
    if s["rows_mask"]:
        # Block positions on another rank's pages map onto slots this rank's rows also write:
        # only the masked-in rows may land.
        rows_mask = (_randint(0, 3, (rows,), g) == 0).to(torch.int8)
        rows_mask[0] = 1
        pages = torch.where(rows_mask.bool(), pages, pages[0])
        offsets = torch.where(rows_mask.bool(), offsets, offsets[0])
    src_before = src.clone()
    ref = k3_scatter_rows_to_pool.reference(src, pool, pages, offsets, rows_mask)
    k3_scatter_rows_to_pool(src, pool, pages, offsets, rows_mask=rows_mask)
    _same(pool, ref, "pool")
    assert torch.equal(_bits(src), _bits(src_before))


@pytest.mark.parametrize("cell", k3_vocab_shard_embedding.CELLS, ids=_ids(k3_vocab_shard_embedding))
def test_k3_vocab_shard_embedding_cells(cell, certifying) -> None:  # noqa: F811
    _require(k3_vocab_shard_embedding)
    certifying(k3_vocab_shard_embedding)
    g = _gen(cell)
    s = cell.spec
    shard = VOCAB // s["tp"]
    start = s["rank"] * shard
    weight = torch.randn((shard, HIDDEN), generator=g, device="cuda").to(torch.bfloat16)
    weight[0, :5] = -0.0
    # Bonus ids from anywhere in the vocab, this shard's first and last rows, then the mask id.
    ids = _randint(0, VOCAB, (s["num_ids"],), g)
    ids[0] = start
    if s["num_ids"] > 2:
        ids[1] = start + shard - 1
    ids[-1] = 163837
    ref = k3_vocab_shard_embedding.reference(ids, weight, start, start + shard)
    out = k3_vocab_shard_embedding(ids, weight, start, start + shard)
    _same(out, ref, "shard lookup")
