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
"""``trtllm::k3_spec_accept`` (one decode step's speculative acceptance and the block drafter's inputs) against the
torch op sequence it replaces, bit for bit, on one GPU at the in-model shapes (V = 163840 target logits, hidden 7168).

The reference, as the model runs it for a decode step of B generation requests with K drafts each (no context requests,
greedy strict acceptance): DFlashWorker._refresh_ctx_block_tables, SpecWorkerBase._sample_and_accept_draft_tokens_base
with _apply_force_accepted_tokens (the model's RNG pool and strides), the KDA replay record of
MambaHybridCacheManagerV2.update_mamba_states, the kv_lens update of DFlashWorker._prepare_kv_for_draft_forward and
DFlashWorker.prepare_1st_drafter_inputs up to the fc (bonus tokens, positions, noise embedding).

Every split B = 1 .. 8 requests x K + 1 in {2, 4, 8} tokens at the drafter block widths the model passes (K + 1: DFlash;
K: DSpark's shift_label; 8: a draft-length schedule running K < max_draft_len = 7), 20 consecutive steps per case with
the state carried: natural acceptance with every accepted-prefix length 0 .. K (mixed over the requests), ties across
CTAs (the lowest index wins), NaN rows (the first NaN wins, torch's rule, also over +inf), +-inf and +-0 maxima, forced
acceptance (an integer, a fractional and an above-K value), dummy requests (and CUDA-graph padding sharing one state and
one context slot), placeholder (-1) and zero block offsets, block tables of 72 / 300 / 4100 columns, context lengths at
the max_ctx clamp. At every step every output and every in-place state tensor is bit-identical to the reference, the
inputs are untouched and a second run on a clone of the state gives the same bits (determinism). CUDA graphs: one
captured call per split family, replayed 20 times with the logits, drafts, dummy mask, block offsets and context lengths
rewritten in place and the state carried.

Table: ``python3 test_k3_spec_accept.py report``. Timing: ``python3 test_k3_spec_accept.py time`` (batch 1, then every
split: CUDA graphs of 20 back-to-back calls of the reference ops and of the kernel, median over 12 replays in
alternating order). The vocabulary-sharded variant: test_k3_spec_accept_sharded.py (MPI).
"""

import itertools
import statistics
import sys

import pytest
import torch
import torch.nn.functional as F

V = 163840  # target vocabulary
HIDDEN = 7168  # drafter hidden size (the noise embedding rows)
CTA_COLS = 1024  # logits columns one CTA of the kernel reduces (ties straddle these boundaries)
TP16_COLUMNS = 10240  # a TP16 rank's vocabulary shard
SEQS = 16  # rows of the per-request state; the rows past the batch must stay untouched
SLOTS = 20  # KDA replay / context slots
POOLS, POOL_IDX = 2, 1  # KV pools in the block offsets; the draft pool's index
DIVISOR = 10  # encoded block offset -> pool block (5 drafter layers x K / V)
MAX_CTX = 4100
# Block-table width by batch % 3: one, two and 17 passes of the kernel's 256 threads.
MAX_BLOCKS = (72, 300, 4100)
PADDING_FROM = 5  # batches of 5 or more end with two CUDA-graph padding requests
STEPS = 20
# Forced acceptance per K: (an integer value, a fractional value) of TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS.
FORCES = {1: (1.0, 0.4), 3: (2.0, 1.6), 7: (5.0, 5.4)}

# (B requests, K + 1 tokens each), batch 1 first.
SPLITS = [(b, t) for b in range(1, 9) for t in (2, 4, 8)]
GRAPH_SPLITS = [(b, t) for b in (1, 4, 8) for t in (2, 4, 8)]
WIDE_SPLITS = [(b, t) for b in (1, 4, 8) for t in (2, 4)]
# (name, logits kind, forced acceptance)
CASES = (
    ("natural", "plain", None),
    ("ties", "ties", None),
    ("nan", "nan", None),
    ("inf / +-0", "signed", None),
    ("forced int", "plain", "int"),
    ("forced frac", "plain", "frac"),
    ("forced > K", "plain", "over"),
)
GRAPH_KINDS = ("plain", "ties", "nan", "signed")

OUTPUTS = ("accepted", "num_acc", "rewind", "bonus", "qpos", "cpos", "noise")
STATE_KEYS = ("block_counts", "block_tables", "prev_acc", "kv_lens", "rng_counter")
INPUT_KEYS = ("block_off", "state_idx", "dummy", "batch_to_slot", "ctx_len", "mask_row")
SHARED_KEYS = ("embed", "rng_pool")  # read-only and large: shared by the clones of a state


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _sm100(), reason="needs SM100 (CTM kernel)")


def _op():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    from tensorrt_llm._torch.cute_dsl_kernels.k3_spec_accept import op

    return op


def _spec():
    """The model's forced-acceptance constants live here."""
    from tensorrt_llm._torch.speculative import interface

    return interface


_cache = {}


def embedding() -> torch.Tensor:
    """The drafter embedding, bf16 [V, HIDDEN] (2.3 GB), built once: the kernel only reads it."""
    embed = _cache.get("embed")
    if embed is None:
        gen = torch.Generator(device="cuda").manual_seed(7)
        embed = torch.randn(V, HIDDEN, generator=gen, device="cuda", dtype=torch.bfloat16)
        _cache["embed"] = embed
    return embed


def rng_pool() -> torch.Tensor:
    """The model's fixed forced-acceptance pool (SpecWorkerBase._ensure_force_accept_rng_state), built once."""
    pool = _cache.get("rng_pool")
    if pool is None:
        spec = _spec()
        gen = torch.Generator(device="cpu").manual_seed(spec._FORCE_ACCEPT_RNG_SEED)
        pool = torch.rand(spec._FORCE_ACCEPT_RNG_POOL_SIZE, generator=gen).cuda()
        _cache["rng_pool"] = pool
    return pool


def force_value(drafts: int, forced) -> float:
    if forced is None:
        return 0.0
    if forced == "over":  # min(int(f) + 1, K + 1) = K + 1: the fraction is ignored
        return drafts + 2.5
    return FORCES[drafts][0 if forced == "int" else 1]


# ----------------------------------------------------------------------------------------------------------------
# The decode step's state and inputs
# ----------------------------------------------------------------------------------------------------------------


def block_offsets(gen, max_blocks: int, salt: int = 0) -> torch.Tensor:
    """The draft KV manager's encoded block offsets [pools, seqs, K / V, max_blocks]: request b holds 5 + 9 b (+ 3 salt)
    blocks for even b, max_blocks + 1 - b for odd b (request 1 the whole row), placeholders (-1) after them, a
    placeholder hole and a zero offset in request 0; the other pool and the V plane hold other values."""
    off = torch.randint(
        0, 1 << 22, (POOLS, SEQS, 2, max_blocks), generator=gen, device="cuda", dtype=torch.int32
    )
    seq = torch.arange(SEQS, device="cuda")
    used = torch.where(seq % 2 == 0, 5 + 9 * seq + 3 * salt, max_blocks + 1 - seq)
    tail = torch.arange(max_blocks, device="cuda")[None, :] >= used[:, None]
    off.masked_fill_(tail[None, :, None, :], -1)
    off[POOL_IDX, 0, 0, 1] = -1
    off[POOL_IDX, 0, 0, 3] = 0
    return off


def context_lengths(gen, batch_to_slot: torch.Tensor, batch: int) -> torch.Tensor:
    """Context lengths [SLOTS] (int64): request 0's one below max_ctx (its query positions clamp once it accepts 2
    tokens), request 1's at max_ctx, the others random."""
    ctx = torch.randint(100, MAX_CTX - 16, (SLOTS,), generator=gen, device="cuda")
    ctx[batch_to_slot[0]] = MAX_CTX - 1
    if batch > 1:
        ctx[batch_to_slot[1]] = MAX_CTX
    return ctx


def make_state(gen, batch: int) -> dict:
    """One decode step's state with the model's dtypes: distinct KDA slots and context slots per request, except the
    two CUDA-graph padding requests of a batch of 5 or more (one state slot, one context slot); stale block counts and
    tables the kernel overwrites for the batch's rows only."""
    max_blocks = MAX_BLOCKS[batch % len(MAX_BLOCKS)]
    state_idx = torch.randperm(SLOTS, generator=gen, device="cuda")[:SEQS].to(torch.int32)
    batch_to_slot = torch.randperm(SLOTS, generator=gen, device="cuda")[:SEQS]
    if batch >= PADDING_FROM:
        state_idx[batch - 1] = state_idx[batch - 2]
        batch_to_slot[batch - 1] = batch_to_slot[batch - 2]
    return dict(
        pool_idx=POOL_IDX,
        divisor=DIVISOR,
        block_off=block_offsets(gen, max_blocks),
        block_counts=torch.randint(0, 99, (SEQS,), generator=gen, device="cuda"),
        block_tables=torch.randint(0, 99, (SEQS, max_blocks), generator=gen, device="cuda", dtype=torch.int32),
        prev_acc=torch.randint(0, 8, (SLOTS,), generator=gen, device="cuda", dtype=torch.int32),
        state_idx=state_idx,
        dummy=torch.zeros(SEQS, dtype=torch.bool, device="cuda"),
        kv_lens=torch.randint(100, 2000, (SEQS,), generator=gen, device="cuda", dtype=torch.int32),
        batch_to_slot=batch_to_slot,
        ctx_len=context_lengths(gen, batch_to_slot, batch),
        embed=embedding(),
        mask_row=torch.randn(HIDDEN, generator=gen, device="cuda").bfloat16(),
        rng_pool=rng_pool(),
        rng_counter=torch.full((1,), 41, dtype=torch.int64, device="cuda"),
    )  # fmt: skip


def clone_state(st: dict) -> dict:
    return {
        k: v.clone() if torch.is_tensor(v) and k not in SHARED_KEYS else v for k, v in st.items()
    }


def set_dummies(states, batch: int, step: int) -> None:
    """A step's dummy-request mask: request b when (b + step) % 4 == 3, and the CUDA-graph padding (the last two
    requests of a batch of 5 or more) at every step."""
    seq = torch.arange(SEQS, device="cuda")
    mask = (seq + step) % 4 == 3
    if batch >= PADDING_FROM:
        mask |= (seq == batch - 1) | (seq == batch - 2)
    for st in states:
        st["dummy"].copy_(mask)


def shape_logits(logits: torch.Tensor, gen, kind: str, step: int, shards: int = 1) -> None:
    """Shapes a step's logits [rows, vocab] (fp32 or bf16: the values are exact in both) in place for ``kind``.

    ties: 9.0 at 3 random columns, the column after the first, both sides of a CTA boundary (and of a shard boundary).
    rank_ties: 7.0 at one column of every shard (rank 0's wins). nan: a whole NaN row; rows with +inf at column 2 before
    NaNs in both halves, rows with +inf before two random NaNs, rows with one NaN in the last CTA (shard). signed: two
    +inf, a -inf row, -0.0 / +0.0 maxima in either order. negzero: -1.0 rows with -0.0 / +0.0 maxima across shards, a
    -0.0 in the last shard only, and none (every column ties)."""
    rows, vocab = logits.shape
    dev = logits.device
    r = torch.arange(rows, device=dev)
    half = vocab // 2
    shard = vocab // shards
    nan, inf = float("nan"), float("inf")
    if kind == "ties":
        cols = torch.randint(0, vocab, (rows, 3), generator=gen, device=dev)
        edge = torch.randint(1, vocab // CTA_COLS, (rows, 1), generator=gen, device=dev) * CTA_COLS
        extra = [(cols[:, :1] + 1) % vocab, edge - 1, edge]
        if shards > 1:
            k = 1 + (r[:, None] + step) % (shards - 1)
            extra += [k * shard - 1, k * shard]
        logits.scatter_(1, torch.cat([cols] + extra, dim=1), 9.0)
    elif kind == "rank_ties":
        q = torch.arange(shards, device=dev)[None, :]
        logits.scatter_(1, q * shard + (r[:, None] * 37 + q * 11 + step) % shard, 7.0)
    elif kind == "nan":
        first = torch.randint(1, half, (rows,), generator=gen, device=dev)
        later = first + torch.randint(1, half, (rows,), generator=gen, device=dev)
        pattern = (r + step) % 3
        a, b, c = r[pattern == 0], r[pattern == 1], r[pattern == 2]
        logits[a, 2] = inf
        logits[a, half + 3] = nan
        logits[a, vocab - 1] = nan
        logits[b, first[b] // 2] = inf
        logits[b, first[b]] = nan
        logits[b, later[b]] = nan
        logits[c, vocab - 5] = nan
        logits[step % rows] = nan
    elif kind == "signed":
        lo = torch.randint(0, half, (rows,), generator=gen, device=dev)
        hi = lo + torch.randint(1, half, (rows,), generator=gen, device=dev)
        pattern = (r + step) % 4
        p0, p1, p2, p3 = (r[pattern == p] for p in range(4))
        logits[p0, lo[p0]] = inf
        logits[p0, hi[p0]] = inf
        logits[p1] = -inf
        logits[p2] = -1.0
        logits[p2, lo[p2]] = -0.0
        logits[p2, hi[p2]] = 0.0
        logits[p3] = -1.0
        logits[p3, lo[p3]] = 0.0
        logits[p3, hi[p3]] = -0.0
    elif kind == "negzero":
        pattern = (r + step) % 4
        p0, p1, p2 = (r[pattern == p] for p in range(3))
        logits.fill_(-1.0)
        logits[p0, 12] = -0.0
        logits[p0, half + 5] = 0.0
        logits[p1, vocab // max(shards, 2) + 7] = 0.0
        logits[p1, vocab - 1] = -0.0
        logits[p2, vocab - 1] = -0.0
    elif kind != "plain":
        raise ValueError(kind)


def drafts_for(gen, logits: torch.Tensor, batch: int, drafts: int, step: int) -> torch.Tensor:
    """Drafts [B, K] (int32) whose accepted prefix under the target tokens of ``logits`` is (step + b) % (K + 1) for
    request b: the drafts after the prefix are random, the first of them a mismatch."""
    vocab = logits.shape[1]
    dev = logits.device
    target = torch.argmax(logits.float(), dim=-1).view(batch, drafts + 1)[:, :drafts]
    target = target.to(torch.int32)
    draft = torch.randint(0, vocab, (batch, drafts), generator=gen, device=dev, dtype=torch.int32)
    prefix = ((torch.arange(batch, device=dev) + step) % (drafts + 1))[:, None]
    j = torch.arange(drafts, device=dev)[None, :]
    draft = torch.where(j < prefix, target, draft)
    return torch.where(j == prefix, (target + 1) % vocab, draft).contiguous()


def step_inputs(gen, batch: int, drafts: int, kind: str, step: int):
    """A step's fp32 target logits [B (K + 1), V] of ``kind`` and its drafts."""
    logits = torch.randn(batch * (drafts + 1), V, generator=gen, device="cuda")
    shape_logits(logits, gen, kind, step)
    return logits, drafts_for(gen, logits, batch, drafts, step)


# ----------------------------------------------------------------------------------------------------------------
# The reference and the kernel
# ----------------------------------------------------------------------------------------------------------------


def reference(st: dict, logits, draft, force: float, block: int) -> list:
    """The model's torch op sequence for the step, K = draft.shape[1]; updates the state tensors of ``st`` in place."""
    spec = _spec()
    batch, drafts = draft.shape
    dev = logits.device
    # DFlashWorker._refresh_ctx_block_tables
    encoded = st["block_off"][st["pool_idx"], :batch, 0].to(torch.int64).clone()
    st["block_counts"][:batch].copy_((encoded >= 0).sum(dim=1))
    decoded = encoded.clamp_(min=0).div_(st["divisor"], rounding_mode="floor")
    st["block_tables"][:batch].copy_(decoded.to(torch.int32))
    # SpecWorkerBase._sample_and_accept_draft_tokens_base (greedy: argmax)
    accepted = torch.empty((batch, drafts + 1), dtype=torch.int, device=dev)
    num_acc = torch.ones(batch, dtype=torch.int, device=dev)
    gen_target = torch.argmax(logits, dim=-1).reshape(batch, drafts + 1)
    accepted[:, : drafts + 1] = gen_target
    num_acc[0:] += torch.cumprod((draft == gen_target[:, :drafts]).int(), dim=-1).sum(1)
    # SpecWorkerBase._apply_force_accepted_tokens (no context requests)
    if force != 0.0:
        int_part = int(force)
        frac = force - int_part
        max_total = drafts + 1
        base_total = min(int_part + 1, max_total)
        if frac > 0.0 and base_total < max_total:
            st["rng_counter"] += 1
            slot_ids = torch.arange(batch, device=dev, dtype=torch.int64)
            hashed = st["rng_counter"] * spec._FORCE_ACCEPT_RNG_COUNTER_STRIDE
            hashed = hashed + slot_ids * spec._FORCE_ACCEPT_RNG_SLOT_STRIDE
            indices = hashed & (spec._FORCE_ACCEPT_RNG_POOL_SIZE - 1)
            extra = (st["rng_pool"][indices] < frac).to(num_acc.dtype)
            num_acc[0:] = (base_total + extra).clamp_(max=max_total)
        else:
            num_acc[0:] = base_total
    # MambaHybridCacheManagerV2.update_mamba_states -> _record_kda_replay_acceptance
    nad = (num_acc[0:batch] - 1).to(torch.int32)
    slots = st["state_idx"][0:batch].to(torch.int32).to(torch.long)
    acc = nad.clamp(min=0)
    current = st["prev_acc"][slots]
    acc = torch.where(st["dummy"][0:batch], current, acc)
    st["prev_acc"][slots] = acc
    # DFlashWorker._prepare_kv_for_draft_forward
    rewind = 1 - num_acc[0:batch]
    st["kv_lens"][0:batch] += 1
    # DFlashWorker.prepare_1st_drafter_inputs up to the fc
    bonus_idx = (num_acc - 1).clamp_min(0).long().unsqueeze(1)
    bonus = accepted.gather(1, bonus_idx).squeeze(1).long()
    ctx_len_gen = st["ctx_len"][st["batch_to_slot"][0:batch]]
    j_block = torch.arange(block, dtype=torch.long, device=dev)
    offsets = torch.arange(drafts + 1, dtype=torch.long, device=dev)
    now = (ctx_len_gen + num_acc.long()).clamp_(max=MAX_CTX)
    qpos = now.unsqueeze(1) + j_block.unsqueeze(0)
    cpos = ctx_len_gen.unsqueeze(1) + offsets.unsqueeze(0)
    noise = st["mask_row"].expand(batch, block, -1).clone()
    noise[:, 0, :] = F.embedding(bonus, st["embed"])
    return [accepted, num_acc, rewind, bonus, qpos, cpos, noise]


def fused(st: dict, logits, draft, force: float, block: int, shard=None) -> list:
    """``trtllm::k3_spec_accept`` on the state ``st``; ``shard``: (workspace, first column) when ``logits`` are this
    rank's bf16 vocabulary shard."""
    ws_args = ()
    if shard is not None:
        ws, first = shard
        ws_args = (ws["uc"], ws["mc"], ws["flags"])
        ws_args += (ws["rank"], ws["slots"], ws["push_copies"], first)
    return torch.ops.trtllm.k3_spec_accept(
        logits, draft, st["block_off"], st["pool_idx"], st["divisor"], st["block_counts"], st["block_tables"],
        st["prev_acc"], st["state_idx"], st["dummy"], st["kv_lens"], st["batch_to_slot"], st["ctx_len"], MAX_CTX,
        st["embed"], st["mask_row"], st["rng_pool"], st["rng_counter"], force, block, *ws_args,
    )  # fmt: skip


def state_of(st: dict) -> list:
    return [st[k] for k in STATE_KEYS]


def same(xs, ys) -> list:
    """Per pair: the same dtype, shape and bits."""
    return [
        x.dtype == y.dtype
        and x.shape == y.shape
        and torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
        for x, y in zip(xs, ys)
    ]


def new_result(**fields) -> dict:
    return dict(fields, checks={}, bad=[])


def tally(res: dict, check: str, ok: bool, step: int, names=()) -> None:
    """Counts one step of a check; the first failures are kept with the names of what differed."""
    passed, total = res["checks"].get(check, (0, 0))
    res["checks"][check] = (passed + bool(ok), total + 1)
    if not ok and len(res["bad"]) < 4:
        res["bad"].append((step, check, list(names)))


def record(res: dict, step: int, pairs: dict) -> None:
    """Tallies one step's comparisons; ``pairs``: check -> (names, got, want)."""
    for check, (names, got, want) in pairs.items():
        flags = same(got, want)
        tally(res, check, all(flags), step, [n for n, f in zip(names, flags) if not f])


def finish(res: dict) -> dict:
    res["ok"] = all(passed == total for passed, total in res["checks"].values())
    return res


def cell(res: dict, check: str) -> str:
    passed, total = res["checks"].get(check, (None, None))
    return "-" if total is None else f"{passed}/{total}"


# ----------------------------------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------------------------------


def measure(batch: int, drafts: int, block: int, case: int, steps: int = STEPS) -> dict:
    """One case at one split: ``steps`` consecutive steps with the state carried, the kernel against the reference
    (every output and state tensor), its inputs untouched and a rerun on a clone of its state."""
    _op()
    name, kind, forced = CASES[case]
    force = force_value(drafts, forced)
    seed = 20260930 + 1000 * batch + 100 * drafts + 10 * block + case
    gen = torch.Generator(device="cuda").manual_seed(seed)
    st_ref = make_state(gen, batch)
    st_new = clone_state(st_ref)
    res = new_result(split=f"{batch}x{drafts + 1}", block=block, case=name, force=force)
    for step in range(steps):
        set_dummies((st_ref, st_new), batch, step)
        logits, draft = step_inputs(gen, batch, drafts, kind, step)
        logits_in, draft_in = logits.clone(), draft.clone()
        st_again = clone_state(st_new)
        want = reference(st_ref, logits, draft, force, block)
        got = fused(st_new, logits, draft, force, block)
        again = fused(st_again, logits, draft, force, block)
        record(res, step, {
            "outputs": (OUTPUTS, got, want),
            "state": (STATE_KEYS, state_of(st_new), state_of(st_ref)),
            "inputs": (("logits", "draft") + INPUT_KEYS, [logits, draft] + [st_new[k] for k in INPUT_KEYS],
                       [logits_in, draft_in] + [st_ref[k] for k in INPUT_KEYS]),
            "rerun": (OUTPUTS + STATE_KEYS, again + state_of(st_again), got + state_of(st_new)),
        })  # fmt: skip
    return finish(res)


def measure_graph(batch: int, drafts: int, block: int, forced, replays: int = STEPS) -> dict:
    """One call captured in a CUDA graph and replayed ``replays`` times with the logits (every kind), drafts, dummy
    mask, block offsets and context lengths rewritten in place and the state carried: every replay against the
    reference."""
    _op()
    force = force_value(drafts, forced)
    gen = torch.Generator(device="cuda").manual_seed(777 + 100 * batch + 10 * drafts + block)
    st_ref = make_state(gen, batch)
    st_new = clone_state(st_ref)
    logits, draft = step_inputs(gen, batch, drafts, "plain", 0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused(clone_state(st_new), logits, draft, force, block)  # compiled outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = fused(st_new, logits, draft, force, block)
    torch.cuda.synchronize()
    name = f"graph x{replays} ({'natural' if forced is None else 'forced ' + forced})"
    res = new_result(split=f"{batch}x{drafts + 1}", block=block, case=name, force=force)
    max_blocks = st_ref["block_off"].shape[3]
    for rep in range(replays):
        kind = GRAPH_KINDS[rep % len(GRAPH_KINDS)]
        new_logits, new_draft = step_inputs(gen, batch, drafts, kind, rep)
        logits.copy_(new_logits)
        draft.copy_(new_draft)
        set_dummies((st_ref, st_new), batch, rep)
        offsets = block_offsets(gen, max_blocks, salt=rep)
        ctx = context_lengths(gen, st_ref["batch_to_slot"], batch)
        for st in (st_ref, st_new):
            st["block_off"].copy_(offsets)
            st["ctx_len"].copy_(ctx)
        graph.replay()
        want = reference(st_ref, logits, draft, force, block)
        record(res, rep, {
            "outputs": (OUTPUTS, outs, want),
            "state": (STATE_KEYS, state_of(st_new), state_of(st_ref)),
            "inputs": (INPUT_KEYS, [st_new[k] for k in INPUT_KEYS], [st_ref[k] for k in INPUT_KEYS]),
        })  # fmt: skip
    del graph
    return finish(res)


def failures(results: list) -> list:
    return [
        {k: r[k] for k in ("split", "block", "case", "checks", "bad")}
        for r in results
        if not r["ok"]
    ]


@pytest.mark.parametrize("block_delta", [1, 0], ids=["block=K+1", "block=K"])
@pytest.mark.parametrize("batch,tokens", SPLITS, ids=[f"{b}x{t}" for b, t in SPLITS])
def test_split(batch, tokens, block_delta):
    """Every case at one split and block width (DFlash's K + 1, DSpark's K), 20 steps each."""
    with torch.inference_mode():
        results = [
            measure(batch, tokens - 1, tokens - 1 + block_delta, c) for c in range(len(CASES))
        ]
    assert not failures(results), failures(results)


@pytest.mark.parametrize("batch,tokens", WIDE_SPLITS, ids=[f"{b}x{t}" for b, t in WIDE_SPLITS])
def test_wide_block(batch, tokens):
    """Block 8 at K = 1 and 3: DFlash's block under max_draft_len 7 when a draft-length schedule runs fewer drafts."""
    with torch.inference_mode():
        results = [measure(batch, tokens - 1, 8, c) for c in range(len(CASES))]
    assert not failures(results), failures(results)


@pytest.mark.parametrize("forced", [None, "frac"], ids=["natural", "forced-frac"])
@pytest.mark.parametrize("block_delta", [1, 0], ids=["block=K+1", "block=K"])
@pytest.mark.parametrize("batch,tokens", GRAPH_SPLITS, ids=[f"{b}x{t}" for b, t in GRAPH_SPLITS])
def test_graph_replay(batch, tokens, block_delta, forced):
    """A captured call replayed 20 times with its inputs rewritten in place, the state carried."""
    with torch.inference_mode():
        res = measure_graph(batch, tokens - 1, tokens - 1 + block_delta, forced)
    assert res["ok"], failures([res])


def test_force_mode():
    """force_mode follows _apply_force_accepted_tokens: (mode, base total, fraction) of a forced value at K drafts."""
    op = _op()
    kern = op._kernel_module()
    for drafts, (integer, fractional) in FORCES.items():
        assert op.force_mode(0.0, drafts) == (kern.FORCE_OFF, 0, 0.0)
        want = (kern.FORCE_INT, min(int(integer) + 1, drafts + 1), 0.0)
        assert op.force_mode(integer, drafts) == want, drafts
        mode, total, frac = op.force_mode(fractional, drafts)
        assert (mode, total) == (kern.FORCE_FRAC, int(fractional) + 1) and 0.0 < frac < 1.0, drafts
        # Above K the count clamps to K + 1 and the fraction is ignored.
        want = (kern.FORCE_INT, drafts + 1, 0.0)
        assert op.force_mode(force_value(drafts, "over"), drafts) == want, drafts


def test_supports():
    """Every split the engine runs is supported (block K .. 8, the whole vocabulary or a TP shard); the limits are
    rejected, and unsupported calls raise before launching."""
    op = _op()
    for batch, tokens in SPLITS:
        drafts = tokens - 1
        for block in (drafts, tokens, 8):
            assert op.supports(V, batch, block, drafts, HIDDEN), (batch, tokens, block)
        for ranks in (2, 4, 8, 16):
            assert op.supports(V // ranks, batch, tokens, drafts, HIDDEN, ranks), (batch, ranks)
        assert op.supports(TP16_COLUMNS, batch, tokens, drafts, HIDDEN, 16), (batch, tokens)
    assert not op.supports(V, 9, 8, 7, HIDDEN)  # 9 requests
    assert not op.supports(V, 5, 16, 15, HIDDEN)  # 80 rows
    assert not op.supports(V, 1, 17, 7, HIDDEN)  # block 17
    assert not op.supports(V, 1, 0, 7, HIDDEN)  # no block
    assert not op.supports(V, 1, 8, 16, HIDDEN)  # K + 1 = 17
    assert not op.supports(V + CTA_COLS // 2, 1, 8, 7, HIDDEN)  # a partial CTA of columns
    assert not op.supports(V, 1, 8, 7, HIDDEN + 4)  # hidden % 8
    assert not op.supports(TP16_COLUMNS, 1, 8, 7, HIDDEN, 3)  # an odd slot count
    assert not op.supports(TP16_COLUMNS, 1, 8, 7, HIDDEN, 32)  # more than 16 slots
    with torch.inference_mode():
        gen = torch.Generator(device="cuda").manual_seed(3)
        st = make_state(gen, 1)
        logits, draft = step_inputs(gen, 1, 7, "plain", 0)
        # 7 rows for 1 x 8, bf16 logits without a workspace, int64 drafts.
        bad_calls = ((logits[:7], draft), (logits.bfloat16(), draft), (logits, draft.long()))
        for bad_logits, bad_draft in bad_calls:
            with pytest.raises(ValueError):
                fused(st, bad_logits, bad_draft, 0.0, 8)
        with pytest.raises(ValueError):  # a state tensor of another dtype
            fused(dict(st, kv_lens=st["kv_lens"].long()), logits, draft, 0.0, 8)
        with pytest.raises(ValueError):  # bf16 shards without a workspace
            torch.ops.trtllm.k3_spec_accept(
                logits.bfloat16(), draft, st["block_off"], POOL_IDX, DIVISOR, st["block_counts"], st["block_tables"],
                st["prev_acc"], st["state_idx"], st["dummy"], st["kv_lens"], st["batch_to_slot"], st["ctx_len"],
                MAX_CTX, st["embed"], st["mask_row"], st["rng_pool"], st["rng_counter"], 0.0, 8, None, None, None, 0,
                2, 1, 0,
            )  # fmt: skip


# ----------------------------------------------------------------------------------------------------------------
# Report (python3 test_k3_spec_accept.py report) and timing (python3 test_k3_spec_accept.py time)
# ----------------------------------------------------------------------------------------------------------------


def report() -> int:
    """Every check as a markdown table."""
    _op()
    print(f"{torch.cuda.get_device_name()}; V {V}, hidden {HIDDEN}; {STEPS} consecutive steps per case, "
          "state carried")  # fmt: skip
    print("| split | block | case | force | outputs identical | state identical | inputs untouched | rerun identical "
          "| result |")  # fmt: skip
    print("| :-- | --: | :-- | --: | --: | --: | --: | --: | :-- |")
    ok_all = True
    runs = [(b, t, t - 1 + d, c) for b, t in SPLITS for d in (1, 0) for c in range(len(CASES))]
    runs += [(b, t, 8, c) for b, t in WIDE_SPLITS for c in range(len(CASES))]
    with torch.inference_mode():
        results = (measure(b, t - 1, block, c) for b, t, block, c in runs)
        graphs = (measure_graph(b, t - 1, t - 1 + d, forced) for b, t in GRAPH_SPLITS for d in (1, 0)
                  for forced in (None, "frac"))  # fmt: skip
        for res in itertools.chain(results, graphs):
            ok_all &= res["ok"]
            print(f"| {res['split']} | {res['block']} | {res['case']} | {res['force']:g} | {cell(res, 'outputs')} | "
                  f"{cell(res, 'state')} | {cell(res, 'inputs')} | {cell(res, 'rerun')} | "
                  f"{'PASS' if res['ok'] else 'FAIL ' + str(res['bad'])} |", flush=True)  # fmt: skip
    print("ALL PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


def capture(body, calls: int) -> torch.cuda.CUDAGraph:
    """A CUDA graph of ``calls`` back-to-back bodies (the first body, which compiles, runs outside capture)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        body()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(calls):
                body()
    torch.cuda.synchronize()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def replay_us(graph: torch.cuda.CUDAGraph, calls: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / calls


def timing(calls: int = 20, rounds: int = 12) -> None:
    """us per call of the reference ops and of the kernel (natural acceptance, block K + 1), batch 1 first."""
    _op()
    print(f"{torch.cuda.get_device_name()}; V {V}, hidden {HIDDEN}; CUDA graphs of {calls} back-to-back calls, "
          f"median over {rounds} replays in alternating order: us per call")  # fmt: skip
    print("| split | block | reference (torch ops) | k3_spec_accept | speedup |")
    print("| :-- | --: | --: | --: | --: |")
    with torch.inference_mode():
        for batch, tokens in SPLITS:
            drafts = tokens - 1
            gen = torch.Generator(device="cuda").manual_seed(31 + 10 * batch + tokens)
            st_ref = make_state(gen, batch)
            st_new = clone_state(st_ref)
            logits, draft = step_inputs(gen, batch, drafts, "plain", 3)
            graphs = [
                capture(lambda: reference(st_ref, logits, draft, 0.0, tokens), calls),
                capture(lambda: fused(st_new, logits, draft, 0.0, tokens), calls),
            ]
            per_call = [[], []]
            for rd in range(rounds):
                for arm in (0, 1) if rd % 2 == 0 else (1, 0):
                    per_call[arm].append(replay_us(graphs[arm], calls))
            ref_us, kernel_us = (statistics.median(t) for t in per_call)
            print(f"| {batch}x{tokens} | {tokens} | {ref_us:.2f} | {kernel_us:.2f} | {ref_us / kernel_us:.1f}x |",
                  flush=True)  # fmt: skip
            del graphs


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "time":
        timing()
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        sys.exit(report())
    else:
        sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider", *sys.argv[1:]]))
