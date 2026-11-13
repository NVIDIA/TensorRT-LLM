# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the target's spec/k3_spec_accept op on one GPU: the cells without ``columns`` (fp32 logits
of the whole vocabulary, V = 163840, drafter hidden 7168).

Every cell runs its steps with the state carried, through the op under ``validating`` and through the entry's
``reference`` on a clone of the state: every output and every in-place state tensor bit for bit (the kernel
reproduces the torch op sequence exactly). The sharded cells run on four ranks in
``_k3_spec_accept_op_matrix.py``, which builds its steps from the helpers here.
"""

import importlib
import zlib

import pytest
import torch

__extra_import_path__ = ["../../.."]
from _validating import validating  # noqa: E402

k3_spec_accept = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
    "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.ops.spec.k3_spec_accept"
).k3_spec_accept

V = 163840  # target vocabulary
HIDDEN = 7168  # drafter hidden size
CTA_COLS = 1024  # logits columns one CTA reduces (ties straddle these boundaries)
SEQS = 16  # rows of the per-request state; the rows past the batch stay untouched
SLOTS = 20  # KDA replay and context slots
POOLS, POOL_IDX = 2, 1  # KV pools in the block offsets; the draft pool's index
DIVISOR = 10  # encoded block offset -> pool block
MAX_CTX = 4100
MAX_BLOCKS = 300
PADDING_FROM = 5  # batches of 5 or more end with two CUDA-graph padding requests

OUTPUTS = ("accepted", "num_acc", "rewind", "bonus", "qpos", "cpos", "noise")
STATE_KEYS = ("block_counts", "block_tables", "prev_acc", "kv_lens", "rng_counter")
SHARED_KEYS = ("embed", "rng_pool")  # read-only: shared by the clones of a state

UNSHARDED = [c for c in k3_spec_accept.CELLS if "columns" not in c.spec]


def sm10x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not sm10x(), reason="needs sm_100 or sm_103 (CuTe DSL CTM kernel)")

_cache = {}


def embedding(rows: int) -> torch.Tensor:
    """The drafter embedding's first ``rows`` rows, bf16 [rows, HIDDEN] (the whole table is 2.3 GB, built once)."""
    embed = _cache.get("embed")
    if embed is None:
        gen = torch.Generator(device="cuda").manual_seed(7)
        embed = _cache["embed"] = torch.randn(
            V, HIDDEN, generator=gen, device="cuda", dtype=torch.bfloat16
        )
    return embed[:rows]


def rng_pool() -> torch.Tensor:
    """A forced-acceptance RNG pool of the model's size (65536 uniform fp32 draws), built once."""
    pool = _cache.get("rng_pool")
    if pool is None:
        gen = torch.Generator(device="cpu").manual_seed(0xACCE9D)
        pool = _cache["rng_pool"] = torch.rand(1 << 16, generator=gen).cuda()
    return pool


def make_state(gen: torch.Generator, batch: int, vocab: int) -> dict:
    """One decode step's state, keyed by the op's parameter names: distinct KDA and context slots per request except
    the two CUDA-graph padding requests of a batch of 5 or more (one KDA slot, one context slot); encoded block offsets
    with placeholders (-1) after each request's blocks, a placeholder hole and a zero offset; stale block counts and
    tables; request 0's context one below ``max_ctx`` and request 1's at it (the query positions clamp)."""
    state_idx = torch.randperm(SLOTS, generator=gen, device="cuda")[:SEQS].to(torch.int32)
    batch_to_slot = torch.randperm(SLOTS, generator=gen, device="cuda")[:SEQS]
    if batch >= PADDING_FROM:
        state_idx[batch - 1] = state_idx[batch - 2]
        batch_to_slot[batch - 1] = batch_to_slot[batch - 2]
    off = torch.randint(
        0, 1 << 22, (POOLS, SEQS, 2, MAX_BLOCKS), generator=gen, device="cuda", dtype=torch.int32
    )
    seq = torch.arange(SEQS, device="cuda")
    used = torch.where(seq % 2 == 0, 5 + 9 * seq, MAX_BLOCKS + 1 - seq)
    off.masked_fill_(
        (torch.arange(MAX_BLOCKS, device="cuda")[None, :] >= used[:, None])[None, :, None, :], -1
    )
    off[POOL_IDX, 0, 0, 1] = -1
    off[POOL_IDX, 0, 0, 3] = 0
    ctx = torch.randint(100, MAX_CTX - 16, (SLOTS,), generator=gen, device="cuda")
    ctx[batch_to_slot[0]] = MAX_CTX - 1
    if batch > 1:
        ctx[batch_to_slot[1]] = MAX_CTX
    return dict(
        block_off=off,
        pool_idx=POOL_IDX,
        divisor=DIVISOR,
        block_counts=torch.randint(0, 99, (SEQS,), generator=gen, device="cuda"),
        block_tables=torch.randint(0, 99, (SEQS, MAX_BLOCKS), generator=gen, device="cuda", dtype=torch.int32),
        prev_acc=torch.randint(0, 8, (SLOTS,), generator=gen, device="cuda", dtype=torch.int32),
        state_idx=state_idx,
        dummy=torch.zeros(SEQS, dtype=torch.bool, device="cuda"),
        kv_lens=torch.randint(100, 2000, (SEQS,), generator=gen, device="cuda", dtype=torch.int32),
        batch_to_slot=batch_to_slot,
        ctx_len=ctx,
        max_ctx=MAX_CTX,
        embed=embedding(vocab),
        mask_row=torch.randn(HIDDEN, generator=gen, device="cuda").bfloat16(),
        rng_pool=rng_pool(),
        rng_counter=torch.full((1,), 41, dtype=torch.int64, device="cuda"),
    )  # fmt: skip


def clone_state(st: dict) -> dict:
    return {
        k: v.clone() if torch.is_tensor(v) and k not in SHARED_KEYS else v for k, v in st.items()
    }


def state_of(st: dict) -> list:
    return [st[k] for k in STATE_KEYS]


def set_dummies(states, batch: int, step: int) -> None:
    """A step's dummy-request mask: request b when (b + step) % 4 == 3, and the padding requests at every step."""
    seq = torch.arange(SEQS, device="cuda")
    mask = (seq + step) % 4 == 3
    if batch >= PADDING_FROM:
        mask |= (seq == batch - 1) | (seq == batch - 2)
    for st in states:
        st["dummy"].copy_(mask)


def step_logits(
    gen: torch.Generator, rows: int, vocab: int, kind: str, step: int, shards: int = 1
) -> torch.Tensor:
    """A step's fp32 logits [rows, vocab]. ``ties``: 9.0 at three random columns, the column after the first, both
    sides of a CTA boundary and, with ``shards`` > 1, both sides of a shard boundary (the values are exact in bf16)."""
    logits = torch.randn(rows, vocab, generator=gen, device="cuda")
    if kind == "ties":
        r = torch.arange(rows, device="cuda")
        cols = torch.randint(0, vocab, (rows, 3), generator=gen, device="cuda")
        edge = (
            torch.randint(1, vocab // CTA_COLS, (rows, 1), generator=gen, device="cuda") * CTA_COLS
        )
        extra = [(cols[:, :1] + 1) % vocab, edge - 1, edge]
        if shards > 1:
            shard = vocab // shards
            k = 1 + (r[:, None] + step) % (shards - 1)
            extra += [k * shard - 1, k * shard]
        logits.scatter_(1, torch.cat([cols] + extra, dim=1), 9.0)
    elif kind != "plain":
        raise ValueError(kind)
    return logits


def drafts_for(
    gen: torch.Generator, logits: torch.Tensor, batch: int, drafts: int, step: int
) -> torch.Tensor:
    """Drafts [B, K] (int32) whose accepted prefix under the target tokens of ``logits`` is (step + b) % (K + 1) for
    request b; the draft after the prefix is a mismatch."""
    vocab = logits.shape[1]
    target = (
        torch.argmax(logits.float(), dim=-1).view(batch, drafts + 1)[:, :drafts].to(torch.int32)
    )
    draft = torch.randint(
        0, vocab, (batch, drafts), generator=gen, device="cuda", dtype=torch.int32
    )
    prefix = ((torch.arange(batch, device="cuda") + step) % (drafts + 1))[:, None]
    j = torch.arange(drafts, device="cuda")[None, :]
    draft = torch.where(j < prefix, target, draft)
    return torch.where(j == prefix, (target + 1) % vocab, draft).contiguous()


def differing(got, want, names) -> list:
    """The names whose tensors differ in dtype, shape or bits."""
    return [
        n
        for n, x, y in zip(names, got, want)
        if x.dtype != y.dtype
        or x.shape != y.shape
        or not torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
    ]


def cell_seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


@pytest.mark.parametrize("cell", UNSHARDED, ids=[c.why[:40] for c in UNSHARDED])
def test_k3_spec_accept_cells(cell) -> None:
    """The cell's steps with the state carried: the op against its reference, every output and state tensor bit for
    bit."""
    spec = cell.spec
    batch, drafts = spec["batch"], spec["drafts"]
    gen = torch.Generator(device="cuda").manual_seed(cell_seed(cell))
    with torch.inference_mode(), validating(k3_spec_accept):
        st_op = make_state(gen, batch, V)
        st_ref = clone_state(st_op)
        for step in range(spec["steps"]):
            set_dummies((st_op, st_ref), batch, step)
            logits = step_logits(gen, batch * (drafts + 1), V, spec["logits"], step)
            draft = drafts_for(gen, logits, batch, drafts, step)
            got = k3_spec_accept(
                logits=logits, draft=draft, force=spec["force"], block=spec["block"], **st_op
            )
            want = k3_spec_accept.reference(
                logits=logits, draft=draft, force=spec["force"], block=spec["block"], **st_ref
            )
            torch.cuda.synchronize()
            assert not differing(got, want, OUTPUTS), (step, differing(got, want, OUTPUTS))
            bad = differing(state_of(st_op), state_of(st_ref), STATE_KEYS)
            assert not bad, (step, bad)


def test_is_valid_rejects_wrong_inputs() -> None:
    """``is_valid`` admits a valid step and refuses each input the kernel would take and answer wrongly."""
    vocab, batch, drafts = 4 * CTA_COLS, 6, 7
    gen = torch.Generator(device="cuda").manual_seed(11)
    st = make_state(gen, batch, vocab)
    set_dummies((st,), batch, 0)
    logits = step_logits(gen, batch * (drafts + 1), vocab, "plain", 0)
    args = dict(
        st, logits=logits, draft=drafts_for(gen, logits, batch, drafts, 0), force=0.0, block=drafts
    )
    k3_spec_accept.is_valid(**args)
    shared = st["state_idx"].clone()
    shared[1] = shared[0]  # two live requests write one KDA slot
    shard = logits[:, : vocab // 4].bfloat16()
    bad = {
        "pool_idx": dict(pool_idx=POOLS),
        "divisor": dict(divisor=0),
        "kv_lens": dict(kv_lens=st["kv_lens"][: batch - 1]),
        "state_idx range": dict(state_idx=st["state_idx"].clone().fill_(SLOTS)),
        "shared KDA slot": dict(state_idx=shared),
        "batch_to_slot range": dict(batch_to_slot=st["batch_to_slot"].clone().fill_(SLOTS)),
        "short embed": dict(embed=st["embed"][: vocab - 1]),
        "shard_offset": dict(logits=shard, slots=4, push_copies=1, rank=1, shard_offset=0),
        "short rng pool": dict(force=5.4, rng_pool=st["rng_pool"][:1024]),
    }
    for name, change in bad.items():
        with pytest.raises(AssertionError):
            k3_spec_accept.is_valid(**dict(args, **change))
            pytest.fail(f"is_valid admitted {name}")
