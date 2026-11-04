# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep4 target's context indexer under the engine's query split, at tep4 (four ranks) and at
tep8 (eight ranks over two nodes of four).

Under chunked prefill the engine splits a step's context queries evenly across the TP ranks
once they reach `q_split_threshold` (8192): `Indexer.prepare_for_chunked_prefill` sets
`indexer_prefill_q_split = (q_begin, q_end, q_sizes)` and plans `indexer_prefill_chunks` over
this rank's `[q_begin, q_end)` only. The target selects those rows and all-gathers every
rank's share into rows `[0, tc)` of its selection buffer before a layer reads them. The engine
also packs consecutive whole requests into one chunk while `sum(q) * sum(kv)` stays within its
logits budget, so short tails over long prefixes make a chunk of more keys than `max_seq_len`;
the target scores such a chunk, and only such a chunk, per request, each over its own keys
(`_indexer_blocks`).

The ranks, on the MPI session's communicator a tep4 or tep8 engine under mpirun reaches:

* tep4: four ranks this pytest starts on its node with mpirun (the launching allocation's
  PMI/PMIx/Open MPI/Slurm variables removed, as the launch prefix does);
* tep8: eight ranks over two nodes of four, each a pytest process an MPI launcher started --
  `bash <infra>/scripts/mn/mn_mpirun.sh 4 python -m pytest -k tep8 <this file>` -- not
  trtllm-llmapi-launch, under which only rank 0 runs pytest.

Each rank builds the target's core on the meta device with layers 1 (a RoPE indexer), 3 (a NoPE
indexer) and 17 (a layer that reuses the last selection) materialized from
Mistral-Large-4-1T-A52B-NVFP4 through the production weight table at its own mapping, at the
engine's max_seq_len, and drives `PrefillTarget._attention` -- the code the forward runs -- over
the engine's own DSA metadata (`MistralLarge4CacheManager`'s fp8_ds_mla pool on the KV owners at
max_seq_len 65536 at tep4, 32768 at tep8, the most keys a scoring block holds; chunked prefill
on, the world's mapping at that rank) and the context workspace the forward builds per step
(`PrefillTarget._ctx_workspace`), one step after another, so every key and latent a step reads
was appended at its position by an earlier one. The core takes its sizes from the first step's
metadata (`_derive_first_step`): at max_num_tokens 32768 the indexer's logits budget is 2**26 at
tep4's max_seq_len and 2**24 at tep8's (`_prefill_logits_budget`), and with chunked prefill on
the held context workspace is 2**19 rows (`_ctx_workspace_rows`). The steps:

* 3500 tokens (under the threshold: no split, the chunks cover every row);
* 20480 tokens, then 12288 more over those 20480 cached (keys to 32768);
* 8193 tokens of one request, every share boundary inside it (shares 2048, 2048, 2048, 2049 at
  tep4; seven shares of 1024 and one of 1025 at tep8);
* four requests summing 8200, one over a cached prefix, with a decode row behind them;
* five requests within `index_topk` summing 8500: the split is planned, the context indexer
  skipped;

and at tep8, whose packed chunks pass its pool's max_seq_len of 32768:

* two requests of 16384, one 32768-token chunk (4096 rows per rank);
* two 5-token tails over those 16384-token prefixes and an 8200-token prompt: the split, and on
  rank 0 a chunk the engine packs from the two tails over 32778 keys;
* two 5-token tails over 16389-token prefixes: no split, one packed chunk over 32788 keys.

Every step runs as the engine planned it (split) and as the same step with no split (the
chunks `plan_mla_chunked_prefill_chunks` gives over every context row, every row on every
rank); a step that packs requests also runs with each request in a chunk of its own (the
planner at a zero budget). Checked on each rank, per layer:

* every row the indexer selects for starts as a sentinel and none is left after the split;
* each row's scores (its whole window of fp8_mqa_logits output, by two positional checksums of
  the fp32 bits) are bitwise the unsplit ones on this rank's share, and the per-request ones;
* the split's selections are bitwise the unsplit ones and the per-request ones, and the same on
  every rank;
* the layer's sparse-MLA output is bitwise the unsplit one and the per-request one;
* one gather per selecting layer, of this rank's share, over every rank, none on the reusing
  layer, none without a split and none when the context indexer is skipped;
* no key gather, scorer or top-k call is over more than `max_seq_len` keys, though the engine's
  packed chunks may be; each packed chunk's parts are the per-request chunks exactly.

Both top-k entries promise neither an order nor which of the keys tied at the k-th score they
keep (their catalog notes); measured here, three calls of the prefill top-k on one input kept
different tied keys on 2 of 5120 rows. So a wrapper around each top-k call first checks that the
kernel's selection is an exact top-k of the scores it was given (every key above the k-th score
kept, the rest at it, no duplicate, short windows whole), then rewrites it canonically: each
row ascending, `-1` last, a tie at the k-th score given to the lowest-index tied keys. The
rewrite happens before the gather, so the gather moves it. How many rows tied, and on how many
of those the kernel kept other tied keys, is reported. Also: the split without the gather leaves
the other ranks' rows as the sentinel and the coverage check catches them (negative control);
the selections at the budget the core took from this engine are bitwise those at the 1M
config's `1 << 28` and at the 32768 config's `1 << 24` (`_prefill_logits_budget`), over up to
32768 keys; the gather of a full 32768-token chunk's shares is timed (over the two nodes at
tep8); and the (query rows, keys) tiles the target cuts at the 1M config (max_seq_len 1048576,
its budget 2**28) for one 32768-token chunk of a 1048576-token prompt, planned by the engine's
own planner over cached prefixes up to 1015808, for an unsplit step of 8191 tokens over a
near-1M prefix, and for two 5-token tails over 983040-token prefixes, are reported and checked
against the budget and max_seq_len.

Skipped where the checkpoint is absent (looked up as `mistral-large-4-nvfp4` or
`Mistral-Large-4-1T-A52B-NVFP4` under `LLM_MODELS_ROOT`); tep4 where fewer than four GPUs are
visible, inside a launch of another world size or in a pytest that is itself a launched rank;
tep8 unless this pytest is one of eight launched ranks. A tep8 rank that fails prints its
traceback and leaves at once, which ends the launcher's job; after a pass every tep8 rank
finalizes MPI once pytest has reported.
"""

from __future__ import annotations

import faulthandler
import hashlib
import itertools
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from tensorrt_llm.executor.utils import get_spawn_proxy_process_env
from tensorrt_llm.llmapi.mpi_session import get_mpi_world_size

#: The ranks of a node: tep4 is one node of four, tep8 two.
RANKS_PER_NODE = 4
CHECKPOINTS = ("mistral-large-4-nvfp4", "Mistral-Large-4-1T-A52B-NVFP4")
SHARD_INDEX = "consolidated.safetensors.index.json"
#: A world's whole run, compiles and the checkpoint reads included: past it tep4's mpirun and its
#: ranks are ended, and each tep8 rank dumps its stacks and exits, which ends the launcher's job.
DEADLINE_S = 2400
_RANK_FLAG = "--q-split-rank"

LAYERS = 72
TOKENS_PER_BLOCK = 32
LATENT = 576
INDEX_TOPK = 2048
#: The run's chunk (max_num_tokens), batch and the engine's default split threshold.
MAX_NUM_TOKENS = 32768
MAX_BATCH = 16
Q_SPLIT_THRESHOLD = 8192
#: The pool's and the attention metadata's max_seq_len, and the pool's blocks, by world; the
#: core is built at the same max_seq_len (`_build`). At tep8 every top-k call stays within the
#: 32768 columns the prefill top-k entry's cells hold, so max_seq_len is 32768 there, and its
#: packed steps (`PACKED`) pass it.
MAX_SEQ_LEN = {4: 65536, 8: 32768}
NUM_BLOCKS = {4: 2560, 8: 4096}
#: Layer 1 selects with RoPE, layer 3 without; layer 17 reuses the selection in the buffer.
DRIVEN = (1, 3, 17)
SENTINEL = -(2**31)
_INT32_MAX = 2**31 - 1
#: Rows of the checkpoint's embedding the tokens are drawn from: one contiguous read, every
#: token distinct, so a NoPE key never ties another of its window by repeating a token.
FIRST_TOKEN = 1000
#: Rows the top-k check and rewrite handle at once (bounds its temporaries at 32768 columns).
_CHECK_ROWS = 1024

#: The 1M config's max_seq_len, its longest prompt, at which the tiles the target cuts are
#: reported (`_envelope_tiles`), and the cached prefixes before one 32768-token chunk whose last
#: key is 32768, 65536, ..., 1048576.
ENVELOPE_MAX_SEQ_LEN = 1 << 20
ENVELOPE_CACHED = (0, 32768, 98304, 229376, 491520, 1015808)
#: An unsplit step just under the threshold whose last key is max_seq_len.
ENVELOPE_UNSPLIT = Q_SPLIT_THRESHOLD - 1
#: Two 5-token tails over 983040-token prefixes: one packed chunk of 1966090 keys.
ENVELOPE_TAILS = ((983040, 5), (983040, 5))

#: Each request's prompt length.
PROMPTS = {0: 32768, 1: 8193, 2: 6000, 3: 3000, 4: 1800, 5: 900, 6: 1700}
PROMPTS.update({r: 1700 for r in range(7, 11)})
#: The requests of tep8's packed steps.
PACKED_PROMPTS = {11: 16394, 12: 16394, 13: 8200}


@dataclass(frozen=True)
class Step:
    name: str
    #: (request, cached tokens, new tokens) per context request, in batch order.
    ctx: tuple
    #: Requests decoding one token behind the context rows.
    gen: tuple = ()


STEPS = (
    Step("3500 tokens, under the threshold", ((2, 0, 3500),)),
    Step("20480 tokens", ((0, 0, 20480),)),
    Step("12288 tokens over 20480 cached", ((0, 20480, 12288),)),
    Step("8193 tokens of one request", ((1, 0, 8193),)),
    Step(
        "four requests summing 8200 and a decode row",
        ((2, 3500, 2500), (3, 0, 3000), (4, 0, 1800), (5, 0, 900)),
        gen=(1,),
    ),
    Step(
        "five requests within index_topk summing 8500",
        tuple((r, 0, 1700) for r in range(6, 11)),
    ),
)
#: tep8's further steps, over its pool's max_seq_len of 32768.
PACKED = (
    Step("two requests of 16384, one 32768-token chunk", ((11, 0, 16384), (12, 0, 16384))),
    Step(
        "two 5-token tails over 16384 cached and 8200 new tokens",
        ((11, 16384, 5), (12, 16384, 5), (13, 0, 8200)),
    ),
    Step("two 5-token tails over 16389 cached", ((11, 16389, 5), (12, 16389, 5))),
)
#: The steps each world runs, and the requests they draw their tokens for.
WORLD_STEPS = {4: STEPS, 8: STEPS + PACKED}
WORLD_PROMPTS = {4: PROMPTS, 8: {**PROMPTS, **PACKED_PROMPTS}}
#: The step the negative control and the budget comparison run on.
NEGATIVE_STEP = 3
BUDGET_STEP = 2
#: The steps whose engine-packed chunk passes max_seq_len keys (on rank 0 only under the split):
#: tep8's, at its 32768.
PACKED_STEPS = (7, 8)


def _launched_elsewhere(world: int) -> bool:
    """A launcher started an MPI world of another size: the ranks of `world` do not start
    inside it."""
    return get_mpi_world_size() not in (1, world)


def _a_rank_of_a_launch() -> bool:
    """This pytest is itself one rank of an MPI launch, not a process on its own (unlaunched, or
    trtllm-llmapi-launch's proxy, which runs with the launch's MPI variables removed)."""
    return get_mpi_world_size() > 1 and not get_spawn_proxy_process_env()


#: The world sizes, each started its own way: at tep4 this pytest, a process on its own, starts
#: the four ranks on its node under mpirun; at tep8 each of the eight ranks over two nodes of
#: four is a pytest process an MPI launcher started.
WORLDS = [
    pytest.param(
        4,
        marks=[
            pytest.mark.skipif(
                _launched_elsewhere(4), reason="tep4 needs a launch of four ranks or none"
            ),
            pytest.mark.skipif(
                _a_rank_of_a_launch(),
                reason="tep4's ranks are started by one pytest under mpirun, not by a launch",
            ),
        ],
        id="tep4",
    ),
    pytest.param(
        8,
        marks=[
            pytest.mark.skipif(
                get_mpi_world_size() != 8 or not _a_rank_of_a_launch(),
                reason="tep8 needs eight ranks over two nodes of four, each a pytest process an "
                "MPI launcher started (mn_mpirun.sh 4); under trtllm-llmapi-launch only rank 0 "
                "runs pytest",
            ),
        ],
        id="tep8",
    ),
]

pytestmark = [
    # The ranks' communicators and the CuTe DSL compiles keep threads past the test.
    pytest.mark.threadleak(enabled=False),
]


# ---------------------------------------------------------------- the rank body


def _canonicalize(values, starts, ends, out, top_k: int):
    """Check that `out` is an exact top-k of each row's window `[starts, ends)` of `values`
    (window-local indices), then rewrite it in canonical form: ascending, `-1` last, a tie at
    the k-th score given to the lowest-index tied keys. Returns per row: whether its window has
    a tie at the k-th score, whether the kernel kept other tied keys than the canonical ones,
    and two positional checksums of the window's fp32 score bits."""
    import torch

    n, width = values.shape
    dev = values.device
    cols = torch.arange(width, device=dev, dtype=torch.int64)
    ident = torch.arange(top_k, device=dev, dtype=torch.int64)
    big = torch.iinfo(torch.int64).max
    tie_all, moved_all, sums_all = [], [], []
    for r0 in range(0, n, _CHECK_ROWS):
        r1 = min(n, r0 + _CHECK_ROWS)
        s = starts[r0:r1].long()[:, None]
        e = ends[r0:r1].long()[:, None]
        length = (e - s)[:, 0]
        inside = (cols >= s) & (cols < e)
        v = values[r0:r1]
        bits = torch.where(inside, v.view(torch.int32).long(), 0)
        sums = torch.stack([bits.sum(1), (bits * (cols - s + 1)).sum(1)], dim=1)
        sc = torch.where(inside, v.float(), float("-inf"))
        raw = out[r0:r1].long()
        raw_sorted = torch.where(raw < 0, big, raw).sort(dim=1).values
        raw_sorted = torch.where(raw_sorted == big, -1, raw_sorted)
        long_row = length > top_k
        # A window within top_k is kept whole: every position once, then -1.
        short = torch.where(ident < length[:, None], ident, -1)
        bad_short = (~long_row) & (raw_sorted != short).any(1)
        assert not bad_short.any(), f"{int(bad_short.sum())} short windows not kept whole"
        canon = short
        tie = torch.zeros_like(long_row)
        moved = torch.zeros_like(long_row)
        if long_row.any():
            kth = sc.topk(top_k, dim=1).values[:, -1:]
            above = (sc > kth).sum(1)
            at = sc == kth
            inside_idx = (raw_sorted >= 0) & (raw_sorted < length[:, None])
            distinct = (raw_sorted[:, 1:] != raw_sorted[:, :-1]).all(1)
            picked = sc.gather(1, (s + raw_sorted.clamp(min=0)).clamp(max=width - 1))
            exact = (
                inside_idx.all(1)
                & distinct
                & (picked.min(1).values == kth[:, 0])
                & ((picked > kth).sum(1) == above)
            )
            bad = long_row & ~exact
            assert not bad.any(), f"{int(bad.sum())} rows are not an exact top-{top_k}"
            take = (sc > kth) | (at & (at.int().cumsum(1) <= (top_k - above)[:, None]))
            best = torch.where(take, cols - s, big).sort(dim=1).values[:, :top_k]
            canon = torch.where(long_row[:, None], best, short)
            tie = long_row & (above + at.sum(1) > top_k)
            moved = long_row & (raw_sorted != canon).any(1)
        out[r0:r1].copy_(canon.to(out.dtype))
        tie_all.append(tie)
        moved_all.append(moved)
        sums_all.append(sums)
    return torch.cat(tie_all), torch.cat(moved_all), torch.cat(sums_all)


def _unwritten_rows(raw, lo: int, hi: int) -> int:
    """Rows of `raw[lo:hi]` still holding the sentinel anywhere."""
    return int((raw[lo:hi] == SENTINEL).any(dim=1).sum())


def _assert_overwritten(raw, lo: int, hi: int, where: str) -> None:
    left = _unwritten_rows(raw, lo, hi)
    assert left == 0, f"{where}: {left} of rows [{lo}, {hi}) still hold the sentinel"


class _Checkpoint:
    """Raw tensors by Mistral name, read lazily from the shard files, and the configuration the
    engine adapts from its params.json."""

    def __init__(self, path: Path) -> None:
        from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader

        self.path = path
        self.index = json.loads((path / SHARD_INDEX).read_text())["weight_map"]
        self.config = MistralConfigLoader().load_pretrained_config(str(path))

    def entry(self, key: str):
        from safetensors import safe_open

        return safe_open(self.path / self.index[key], framework="pt", device="cpu").get_slice(key)


def _materialize(core, ckpt: _Checkpoint, keys, W) -> None:
    """Fill the named parameters of a meta-declared core through the production manifest:
    its slices (this rank's heads), multi-source rows and load-time transforms."""
    import torch

    manifest = W.MODEL_WEIGHTS.manifest(core)
    for key in keys:
        meta = core.w[key]
        dst = torch.empty(meta.shape, dtype=meta.dtype, device="cuda")
        for ckpt_key, index, transform, shard in manifest[key]:
            entry = ckpt.entry(ckpt_key)
            src = (entry[...] if shard is None else shard(entry)).to("cuda").contiguous()
            if transform is not None:
                src = transform(src, core)
            (dst if index is None else dst[index]).copy_(src)
        core.w[key] = torch.nn.Parameter(dst, requires_grad=False)


def _sparse_config():
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

    return DeepSeekSparseAttentionConfig(
        index_n_heads=32, index_head_dim=128, index_topk=INDEX_TOPK, indexer_rope_interleave=True
    )


def _manager(pretrained_config, world: int):
    """The engine's KV cache manager for this checkpoint (the sparse registry's pick for a
    MistralLarge4Config): fp8_ds_mla latent pages on the 63 KV owners, the indexer keys on the
    44 full-indexer layers; this rank's own pool (the latent and the indexer keys are
    replicated over the ranks), at the world's max_seq_len."""
    from tensorrt_llm._torch.attention.backends.sparse.mistral_large4 import (
        MistralLarge4CacheManager,
    )
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    return MistralLarge4CacheManager(
        KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=NUM_BLOCKS[world] * TOKENS_PER_BLOCK,
            host_cache_size=0,
        ),
        CacheType.SELFKONLY,
        num_layers=LAYERS,
        num_kv_heads=1,
        head_dim=LATENT,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN[world],
        max_batch_size=MAX_BATCH,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.FP8,
        sparse_attention_config=_sparse_config(),
        pretrained_config=pretrained_config,
        max_num_tokens=MAX_NUM_TOKENS,
    )


def _allocate(env, step: Step) -> None:
    """Grow the pool for one step as the scheduler does: a first chunk adds the request, a
    later chunk or a decode token extends it."""
    manager = env.manager
    for rid, cached, new in step.ctx:
        if cached == 0:
            manager.add_dummy_requests([rid], token_nums=[new])
        else:
            cache = manager.kv_cache_map[rid]
            assert cache.resize(cache.capacity + new)
    for rid in step.gen:
        cache = manager.kv_cache_map[rid]
        assert cache.resize(cache.capacity + 1)


def _metadata(env, step: Step):
    """`DSAtrtllmAttentionMetadata` for one step, prepared as a tep4 or tep8 engine with
    chunked prefill prepares it: the cached-KV MLA path on, this rank's mapping."""
    import torch

    from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
    from tensorrt_llm._torch.attention.backends.sparse.dsa import DSAtrtllmAttentionMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    rids = [r for r, _, _ in step.ctx] + list(step.gen)
    md = DSAtrtllmAttentionMetadata(
        seq_lens=torch.tensor([n for _, _, n in step.ctx] + [1] * len(step.gen), dtype=torch.int),
        request_ids=rids,
        max_num_requests=MAX_BATCH,
        num_contexts=len(step.ctx),
        prompt_lens=[env.prompts[r] for r in rids],
        max_num_tokens=MAX_NUM_TOKENS,
        kv_cache_manager=env.manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[c for _, c, _ in step.ctx] + [env.kv[r] for r in step.gen],
        ),
        mapping=env.mapping,
        sparse_attention_config=_sparse_config(),
        runtime_features=AttentionRuntimeFeatures(
            chunked_prefill=True, cache_reuse=False, chunk_size=MAX_NUM_TOKENS
        ),
        enable_context_mla_with_cached_kv=True,
    )
    md.prepare()
    return md


def _planned_chunks(md, step: Step, budget: int):
    """The same step's chunks with no split, by the engine's planner over every context row at
    the given packed-logits budget: the engine's own at `_packed_logits_elem_budget`, one request
    per chunk at zero."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa import indexer as engine

    q_lens = [n for _, _, n in step.ctx]
    kv_lens = [c + n for _, c, n in step.ctx]
    specs = engine.plan_mla_chunked_prefill_chunks(q_lens, kv_lens, budget)
    return [engine.Indexer.prepare_one_prefill_chunk(md, s) for s in specs], specs


def _step_inputs(env, step: Step):
    """The step's token rows (context rows in batch order, then the decode rows) and their
    positions."""
    import torch

    ids, pos = [], []
    for rid, cached, new in step.ctx:
        ids.append(env.ids[rid][cached : cached + new])
        pos.append(torch.arange(cached, cached + new, dtype=torch.int32))
    for rid in step.gen:
        ids.append(env.ids[rid][env.kv[rid] : env.kv[rid] + 1])
        pos.append(torch.tensor([env.kv[rid]], dtype=torch.int32))
    return torch.cat(ids), torch.cat(pos).cuda()


def _layer_input(env, rows, i: int):
    """Layer `i`'s input: the tokens' embeddings through its own attention norm."""
    import torch

    xf = env.embed[rows].float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + env.eps)
    return xf.to(torch.bfloat16) * env.core.w[f"l{i}_attn_norm"]


def _run(env, md, step: Step, chunks, q_split, *, indexer_only: bool = False):
    """One pass of the step over the driven layers through the target's own code, with the
    given chunk plan. Returns per layer: the selection buffer as the indexer left it (None on
    a layer that selects nothing), the selection attention read, the layer output, and per
    row the scores' checksums and whether a top-k call wrote them (and tied, and moved)."""
    import torch

    prefill = env.prefill
    md.indexer_prefill_chunks, md.indexer_prefill_q_split = chunks, q_split
    s = prefill.step_args(md)
    rows, tc = s["num_tokens"], s["num_ctx_tokens"]
    gen = rows - tc
    skip_ctx = s["skip_ctx"]
    skip_gen = s["skip_gen"] if gen else True
    token_rows, pos = env.step_rows, env.step_pos
    req_idx = s["req_idx_per_token"][:rows]
    topk_buf = prefill._topk_buf(s, rows, tc, skip_ctx, skip_gen)
    scratch = prefill._scratch(rows, torch.bfloat16, "cuda")
    ctx = prefill._ctx_workspace(s, tc, "cuda") if tc else None
    env.topk_base = topk_buf
    out = {}
    for i in DRIVEN:
        L = env.layers[i]
        x = _layer_input(env, token_rows, i)
        selects = L.indexer is not None and not (skip_ctx and skip_gen)
        if selects:
            if not skip_ctx:
                topk_buf[:tc].fill_(SENTINEL)
            if not skip_gen:
                topk_buf[tc:].fill_(SENTINEL)
        env.raw = None
        env.rows_seen = SimpleNamespace(
            scored=torch.zeros(rows, dtype=torch.bool, device="cuda"),
            tie=torch.zeros(rows, dtype=torch.bool, device="cuda"),
            moved=torch.zeros(rows, dtype=torch.bool, device="cuda"),
            sums=torch.zeros(rows, 2, dtype=torch.int64, device="cuda"),
        )
        if indexer_only:
            if L.indexer is not None:
                prefill._indexer(s, x, pos, L, topk_buf, tc, skip_ctx, skip_gen)
            o = None
        else:
            o = prefill._attention(
                s, L, x, pos, req_idx, topk_buf, scratch, ctx, tc, skip_ctx, skip_gen
            ).clone()
        out[i] = SimpleNamespace(
            raw=env.raw if selects else None,
            sel=topk_buf.clone(),
            out=o,
            rows=rows,
            tc=tc,
            seen=env.rows_seen,
        )
    torch.cuda.synchronize()
    return out


def _digest(t) -> str:
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()


def _build(ckpt_path: Path, rank: int, world: int):
    """The rank's core at the `world` mapping and the world's max_seq_len, its prefill target
    with the driven layers materialized, the pool at the same max_seq_len and the tokens."""
    import torch

    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tep4 import (  # noqa: E501
        modeling as M,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tep4 import (  # noqa: E501
        weights as W,
    )
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.mapping import Mapping

    ck = _Checkpoint(ckpt_path)
    mapping = Mapping(
        world_size=world,
        rank=rank,
        gpus_per_node=RANKS_PER_NODE,
        tp_size=world,
        pp_size=1,
        moe_ep_size=world,
        moe_tp_size=1,
    )
    # The core at the world's max_seq_len (`MAX_SEQ_LEN`), the pool's and the metadata's: the
    # step's `max_seq_len`, the most keys `_indexer_blocks` leaves in one block, the decode
    # scorer's columns and the rope tables' rows, which cover every position the pool holds. Its
    # logits budget and held context workspace are sized on the first step, from the metadata.
    model_config = ModelConfig(
        pretrained_config=ck.config,
        mapping=mapping,
        max_num_tokens=MAX_NUM_TOKENS,
        max_seq_len=MAX_SEQ_LEN[world],
    )
    with torch.device("meta"):
        core = M.MistralLarge4Tep4ModelingV2(model_config)
    keys = ["lerope_alpha", "final_norm"]
    skip = ("fc", "router", "mlp", "wo", "ffn", "isc", "ws2")
    for i in DRIVEN:
        keys += [
            k
            for k in core.w.keys()
            if k.startswith(f"l{i}_") and not k.split("_", 1)[1].startswith(skip)
        ]
    _materialize(core, ck, keys, W)
    core.build_kernels(torch.cuda.current_device())
    prefill = M.PrefillTarget(core)

    # Every request's tokens, distinct, drawn in one seeded order shared by all ranks.
    prompts = WORLD_PROMPTS[world]
    total = sum(prompts.values()) + 1
    assert FIRST_TOKEN + total <= ck.config.vocab_size, total
    table = ck.entry("tok_embeddings.weight")[FIRST_TOKEN : FIRST_TOKEN + total]
    order = torch.randperm(total, generator=torch.Generator().manual_seed(1051))
    embed = table[order].cuda()
    ids, at = {}, 0
    for rid, length in prompts.items():
        n = length + (1 if rid == 1 else 0)
        ids[rid] = torch.arange(at, at + n)
        at += n

    env = SimpleNamespace(
        M=M,
        core=core,
        prefill=prefill,
        layers={L.index: L for L in prefill._layers},
        mapping=mapping,
        world=world,
        max_seq_len=MAX_SEQ_LEN[world],
        prompts=prompts,
        steps=WORLD_STEPS[world],
        manager=_manager(model_config.pretrained_config, world),
        embed=embed,
        ids=ids,
        kv={},
        eps=ck.config.rms_norm_eps,
        raw=None,
    )
    # Record the selection buffer as the indexer leaves it, after its gather.
    indexer = M.PrefillTarget._indexer

    def recorded(self, step, x, pos, L, topk_buf, tc, skip_ctx, skip_gen):
        indexer(self, step, x, pos, L, topk_buf, tc, skip_ctx, skip_gen)
        env.raw = topk_buf.clone()

    prefill._indexer = MethodType(recorded, prefill)
    return env


def _instrument(env):
    """Count the target's gathers, record the keys its key gather, prefill scorer and top-k get,
    and check and canonicalize every top-k call's selection where it is made, through the module
    names the target calls them by: the decode rows' selector is the trtllm op, or, at a
    max_seq_len cute_dsl_radix_topk_decode serves (both worlds' here), that entry."""
    import torch

    M = env.M
    env.gathers, env.logits_calls, env.key_calls = [], [], []
    gather, logits, topk = M.allgather, M.fp8_mqa_logits, M.cute_dsl_gvr_topk_prefill
    key_gather, topk_decode = M.indexer_k_cache_gather_op, M.indexer_topk_decode
    radix_decode = M.cute_dsl_radix_topk_decode

    def counted_gather(input, sizes, group, process_group=None):
        env.gathers.append(
            (tuple(input.shape), str(input.dtype), tuple(sizes), tuple(group), process_group)
        )
        return gather(input, sizes, group, process_group)

    def recorded_key_gather(k_pool, slots_fp8, slots_scale, k_token_start, n_k, *args, **kwargs):
        env.key_calls.append(("gather", int(n_k)))
        return key_gather(k_pool, slots_fp8, slots_scale, k_token_start, n_k, *args, **kwargs)

    def recorded_logits(q, kv, weights, ks, ke, *args, **kwargs):
        result = logits(q, kv, weights, ks, ke, *args, **kwargs)
        env.logits_calls.append((q.shape[0], kv[0].shape[0], result.stride(0)))
        env.key_calls.append(("logits", int(kv[0].shape[0])))
        return result

    def first_row(out) -> int:
        base = env.topk_base
        offset = out.data_ptr() - base.data_ptr()
        row_bytes = base.stride(0) * base.element_size()
        assert offset >= 0 and offset % row_bytes == 0, (offset, row_bytes)
        return offset // row_bytes

    def note(out, tie, moved, sums) -> None:
        r0 = first_row(out)
        r1 = r0 + out.shape[0]
        seen = env.rows_seen
        assert not seen.scored[r0:r1].any(), f"rows [{r0}, {r1}) selected twice"
        seen.scored[r0:r1] = True
        seen.tie[r0:r1] = tie
        seen.moved[r0:r1] = moved
        seen.sums[r0:r1] = sums

    def checked_topk(values, ks, ke, out, top_k, *args, **kwargs):
        env.logits_calls[-1] += ("topk", values.shape[0], values.shape[1])
        env.key_calls.append(("topk", int(values.shape[1])))
        result = topk(values, ks, ke, out, top_k, *args, **kwargs)
        note(out, *_canonicalize(values, ks, ke, out, top_k))
        return result

    def checked_topk_decode(logits_, seq_lens, indices, next_n, index_topk, *args, **kwargs):
        assert next_n == 1, next_n
        result = topk_decode(logits_, seq_lens, indices, next_n, index_topk, *args, **kwargs)
        starts = torch.zeros_like(seq_lens)
        note(indices, *_canonicalize(logits_, starts, seq_lens, indices, index_topk))
        return result

    def checked_radix_decode(logits_, seq_lens, indices, *, next_n, compiled):
        assert next_n == 1, next_n
        result = radix_decode(logits_, seq_lens, indices, next_n=next_n, compiled=compiled)
        starts = torch.zeros_like(seq_lens)
        note(indices, *_canonicalize(logits_, starts, seq_lens, indices, indices.shape[1]))
        return result

    M.allgather = counted_gather
    M.indexer_k_cache_gather_op = recorded_key_gather
    M.fp8_mqa_logits = recorded_logits
    M.cute_dsl_gvr_topk_prefill = checked_topk
    M.indexer_topk_decode = checked_topk_decode
    M.cute_dsl_radix_topk_decode = checked_radix_decode


def _chunks_summary(chunks):
    return [(c.token_start, c.token_end, c.k_token_end - c.k_token_start) for c in chunks or ()]


def _max_keys(chunks) -> int:
    return max((c.k_token_end - c.k_token_start for c in chunks or ()), default=0)


def _same_chunks(a, b) -> bool:
    """Two chunk plans with the same rows, keys and windows."""
    import torch

    return len(a) == len(b) and all(
        (x.token_start, x.token_end, x.k_token_start, x.k_token_end)
        == (y.token_start, y.token_end, y.k_token_start, y.k_token_end)
        and torch.equal(x.cu_seqlen_ks, y.cu_seqlen_ks)
        and torch.equal(x.cu_seqlen_ke, y.cu_seqlen_ke)
        for x, y in zip(a, b)
    )


def _compare(k: int, i: int, what: str, got, ref, rows_mask) -> dict:
    """Selections, layer output and (on `rows_mask`) scores of one layer of two runs."""
    import torch

    bad_rows = int((got.sel != ref.sel).any(dim=1).sum())
    result = dict(
        same_selection=torch.equal(got.sel, ref.sel),
        rows_differing=bad_rows,
        same_output=torch.equal(got.out, ref.out),
    )
    if got.raw is not None:
        result["same_scores"] = torch.equal(got.seen.sums[rows_mask], ref.seen.sums[rows_mask])
        assert result["same_scores"], f"step {k} layer {i}: scores differ from {what}"
    assert result["same_selection"], f"step {k} layer {i}: {bad_rows} rows differ from {what}"
    assert result["same_output"], f"step {k} layer {i}: the attention output differs from {what}"
    return result


def _check_step(env, k: int, step: Step, log) -> dict:
    """Run step `k` split, unsplit and (when the engine packs requests) one request per chunk;
    every assertion of the step, on this rank."""
    import torch

    from tensorrt_llm._torch.attention.backends.sparse.dsa import indexer as engine

    M = env.M
    rank = env.mapping.tp_rank
    _allocate(env, step)
    md = _metadata(env, step)
    env.core._derive_first_step(md)
    env.step_rows, env.step_pos = _step_inputs(env, step)
    tc = md.num_ctx_tokens
    ctx = md.num_contexts
    assert tc == sum(n for _, _, n in step.ctx)
    assert md.max_seq_len == env.max_seq_len, md.max_seq_len

    # The engine's split, re-derived from its own helper: rank r's share of the context rows.
    split = md.indexer_prefill_q_split
    expect = engine._split_prefill_queries(tc, env.mapping, Q_SPLIT_THRESHOLD)
    if expect[2] is None:
        assert split is None, split
    else:
        assert split == expect, (split, expect)
        q_begin, q_end, q_sizes = split
        assert sum(q_sizes) == tc and q_end - q_begin == q_sizes[rank] and len(q_sizes) == env.world
    split_chunks = md.indexer_prefill_chunks
    full_chunks, full_specs = _planned_chunks(md, step, engine._packed_logits_elem_budget(md))
    lo, hi = (0, tc) if split is None else split[:2]
    covered = sorted((c.token_start, c.token_end) for c in split_chunks)
    assert covered[0][0] == lo and covered[-1][1] == hi, (covered, lo, hi)
    assert all(a[1] == b[0] for a, b in zip(covered, covered[1:])), covered
    full = sorted((c.token_start, c.token_end) for c in full_chunks)
    assert (
        full[0][0] == 0 and full[-1][1] == tc and all(a[1] == b[0] for a, b in zip(full, full[1:]))
    ), full

    # What `_indexer_blocks` reads of the step: the target's scoring blocks for a chunk plan,
    # a chunk past max_seq_len keys cut at its requests.
    plan = dict(
        num_ctx_tokens=tc,
        num_ctx_kv_tokens=md.num_ctx_kv_tokens,
        cu_seqlen_ks=md.cu_seqlen_ks,
        cu_seqlen_ke=md.cu_seqlen_ke,
        max_seq_len=md.max_seq_len,
        **M._ctx_host_offsets(md, ctx),
    )

    def bounded(chunks):
        return M._indexer_blocks(chunks, plan)

    packs = any(len(specs) > 1 for specs in full_specs)
    alone_chunks = _planned_chunks(md, step, 0)[0] if packs else None
    if k in PACKED_STEPS:
        # The engine's packed chunk passes max_seq_len keys; the target's parts are the chunks
        # the engine plans for each request alone, rows, keys and windows.
        assert _max_keys(full_chunks) > env.max_seq_len, _chunks_summary(full_chunks)
        assert _same_chunks(bounded(full_chunks), alone_chunks), (
            _chunks_summary(bounded(full_chunks)),
            _chunks_summary(alone_chunks),
        )
    if rank == 0 and k in PACKED_STEPS:
        assert _max_keys(split_chunks) > env.max_seq_len, _chunks_summary(split_chunks)

    env.key_calls.clear()
    env.gathers.clear()
    env.logits_calls.clear()
    got = _run(env, md, step, split_chunks, split)
    split_gathers, split_logits = list(env.gathers), list(env.logits_calls)
    env.gathers.clear()
    env.logits_calls.clear()
    ref = _run(env, md, step, full_chunks, None)
    full_gathers, full_logits = list(env.gathers), list(env.logits_calls)
    alone = None
    if alone_chunks is not None:
        env.gathers.clear()
        alone = _run(env, md, step, alone_chunks, None)
        assert not env.gathers, env.gathers

    skip_ctx = bool(md.skip_indexer_for_ctx_reqs)
    selecting = [i for i in DRIVEN if env.layers[i].indexer is not None]
    expected_gathers = 0 if split is None or skip_ctx else len(selecting)
    assert len(split_gathers) == expected_gathers, split_gathers
    assert not full_gathers, full_gathers
    for shape, dtype, sizes, group, pg in split_gathers:
        assert shape == (split[1] - split[0], INDEX_TOPK) and dtype == "torch.int32", shape
        assert sizes == tuple(split[2]) and group == tuple(range(env.world)) and pg is None
    # No call reads more keys than a request can hold, though the engine's plans may pack more.
    widest = max((n for _, n in env.key_calls), default=0)
    assert widest <= env.max_seq_len, (widest, sorted(set(env.key_calls)))

    report = dict(
        step=k,
        name=step.name,
        tc=tc,
        rows=got[DRIVEN[0]].rows,
        split=None if split is None else (split[0], split[1], list(split[2])),
        skip_ctx=skip_ctx,
        split_chunks=_chunks_summary(split_chunks),
        split_chunks_as_run=_chunks_summary(bounded(split_chunks)),
        full_chunks=_chunks_summary(full_chunks),
        full_chunks_as_run=_chunks_summary(bounded(full_chunks)),
        engine_max_keys=max(_max_keys(split_chunks), _max_keys(full_chunks)),
        widest_call_keys=widest,
        gathers=len(split_gathers),
        split_logits=sorted(set(split_logits)),
        full_logits=sorted(set(full_logits)),
        layers={},
    )
    for i in DRIVEN:
        g, r = got[i], ref[i]
        layer = dict(selects=g.raw is not None)
        mine = torch.zeros(g.rows, dtype=torch.bool, device="cuda")
        every = torch.zeros(g.rows, dtype=torch.bool, device="cuda")
        if g.raw is not None:
            # Every row the indexer selects for was written: the context rows through the
            # gather, the decode rows by the decode scorer.
            _assert_overwritten(g.raw, 0, g.rows, f"step {k} layer {i} split")
            _assert_overwritten(r.raw, 0, r.rows, f"step {k} layer {i} unsplit")
            # This rank scored exactly its share (and the decode rows); the unsplit run every
            # row. On the rows this rank scored, the scores are the unsplit ones bit for bit.
            if not skip_ctx:
                mine[lo:hi] = True
                every[:tc] = True
            if g.rows > tc and not bool(md.skip_indexer_for_gen_reqs):
                mine[tc:] = True
                every[tc:] = True
            assert torch.equal(g.seen.scored, mine), f"step {k} layer {i}: rows scored, split"
            assert torch.equal(r.seen.scored, every), f"step {k} layer {i}: rows scored, unsplit"
            layer.update(
                scored_rows=int(mine.sum()),
                tie_rows_split=int(g.seen.tie.sum()),
                tie_rows_unsplit=int(r.seen.tie.sum()),
                kernel_kept_other_tied_split=int(g.seen.moved.sum()),
                kernel_kept_other_tied_unsplit=int(r.seen.moved.sum()),
            )
        layer.update(_compare(k, i, "unsplit", g, r, mine))
        if alone is not None:
            a = alone[i]
            if a.raw is not None:
                _assert_overwritten(a.raw, 0, a.rows, f"step {k} layer {i} per request")
                assert torch.equal(a.seen.scored, every), f"step {k} layer {i}: rows, per request"
            layer["per_request"] = _compare(k, i, "per request", r, a, every)
        layer["digest"] = _digest(g.sel)
        report["layers"][i] = layer
        log(f"step {k} layer {i}: {layer}")

    if k == NEGATIVE_STEP:
        # The split's chunks without the gather: the rows of the other ranks' shares are left
        # as the sentinel, and the coverage check sees them.
        neg = _run(env, md, step, split_chunks, None, indexer_only=True)
        for i in selecting:
            left = _unwritten_rows(neg[i].raw, 0, tc)
            assert left == tc - (split[1] - split[0]), (left, tc, split)
            with pytest.raises(AssertionError, match="still hold the sentinel"):
                _assert_overwritten(neg[i].raw, 0, tc, "negative control")
            report[f"negative_layer{i}_sentinel_rows"] = left
        assert not env.gathers
    if k == BUDGET_STEP:
        # The selections do not move with the logits budget: the 1M config's (fewer, longer
        # tiles) and the 32768 config's (more, shorter ones) against the core's own.
        budget = env.core.prefill_logits_budget
        others = {
            M._prefill_logits_budget(MAX_NUM_TOKENS, ENVELOPE_MAX_SEQ_LEN),
            M._prefill_logits_budget(MAX_NUM_TOKENS, 32768),
        }
        assert others == {1 << 28, 1 << 24}, others
        for other in sorted(others - {budget}):
            at = f"2^{other.bit_length() - 1}"
            env.logits_calls.clear()
            env.core.prefill_logits_budget = other
            try:
                moved = _run(env, md, step, split_chunks, split, indexer_only=True)
            finally:
                env.core.prefill_logits_budget = budget
            report[f"budget_{at}_logits"] = sorted(set(env.logits_calls))
            for i in selecting:
                _assert_overwritten(moved[i].raw, 0, tc, f"step {k} layer {i} at {at}")
                same = torch.equal(moved[i].sel, got[i].sel)
                same_scores = torch.equal(moved[i].seen.sums, got[i].seen.sums)
                report[f"budget_{at}_same_layer{i}"] = (same, same_scores)
                assert same, f"layer {i}: selections at {at} differ from the core's {budget}"
                assert same_scores, f"layer {i}: scores at {at} differ from the core's {budget}"
    for rid, cached, new in step.ctx:
        env.kv[rid] = cached + new
    for rid in step.gen:
        env.kv[rid] += 1
    return report


def _time_gather(env, comm) -> dict:
    """What one gather of a full chunk's shares costs: 32768 context rows split over the ranks,
    8192 x 2048 int32 (64 MiB) per rank at tep4 and 4096 x 2048 (32 MiB) at tep8, over the two
    nodes, into rows [0, 32768) of a selection buffer."""
    import torch

    from tensorrt_llm._torch.attention.backends.sparse.dsa import indexer as engine

    core = env.core
    tc = MAX_NUM_TOKENS
    q_begin, q_end, q_sizes = engine._split_prefill_queries(tc, env.mapping, Q_SPLIT_THRESHOLD)
    buf = torch.randint(0, tc, (tc, INDEX_TOPK), dtype=torch.int32, device="cuda")
    times, gather_only = [], []
    for it in range(25):
        comm.Barrier()
        a, b, c = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        a.record()
        gathered = env.M.allgather(buf[q_begin:q_end], q_sizes, core.tp_group, core.tp_pg)
        b.record()
        buf[:tc].copy_(gathered)
        c.record()
        torch.cuda.synchronize()
        if it >= 5:
            gather_only.append(a.elapsed_time(b))
            times.append(a.elapsed_time(c))
    times.sort()
    gather_only.sort()
    return dict(
        share_rows=q_end - q_begin,
        share_mib=(q_end - q_begin) * INDEX_TOPK * 4 / 2**20,
        median_ms=times[len(times) // 2],
        min_ms=times[0],
        gather_median_ms=gather_only[len(gather_only) // 2],
        gather_min_ms=gather_only[0],
    )


def _envelope_md(cached: list[int], new: list[int]):
    """The planner's view of a step at the 1M config (max_seq_len 1048576, max_num_tokens
    32768): context requests with these cached and new lengths."""
    import torch

    cu_cached = [0]
    cu_kv = [0]
    for c, n in zip(cached, new):
        cu_cached.append(cu_cached[-1] + c)
        cu_kv.append(cu_kv[-1] + c + n)
    return SimpleNamespace(
        cu_seqlen_ks=torch.empty(0, dtype=torch.int32, device="cuda"),
        compress_ratios=[1],
        host_ctx_cached_token_indptr=torch.tensor(cu_cached),
        host_ctx_kv_indptr=torch.tensor(cu_kv),
        max_num_tokens=MAX_NUM_TOKENS,
        kv_cache_manager=SimpleNamespace(max_seq_len=ENVELOPE_MAX_SEQ_LEN),
    )


def _tiles_row(M, chunk, budget: int, **extra) -> dict:
    """The target's tiles of one chunk as run: the (query rows, keys) of each scorer and top-k
    call, checked against the budget."""
    n_k = chunk.k_token_end - chunk.k_token_start
    assert n_k <= ENVELOPE_MAX_SEQ_LEN, n_k
    assert int(chunk.cu_seqlen_ke[-1]) <= n_k and int(chunk.cu_seqlen_ks[0]) >= 0
    tiles = M._prefill_tiles(chunk.token_start, chunk.token_end, n_k, budget)
    assert tiles[0][0] == chunk.token_start and tiles[-1][1] == chunk.token_end
    assert all(a[1] == b[0] for a, b in zip(tiles, tiles[1:])), tiles
    full_rows = min(chunk.token_end - chunk.token_start, budget // n_k)
    assert all(b - a == full_rows for a, b in tiles[:-1]), tiles
    assert all((b - a) * n_k <= budget for a, b in tiles), tiles
    return dict(
        extra,
        rows=(chunk.token_start, chunk.token_end),
        n_k=n_k,
        tiles=len(tiles),
        tile_rows=sorted({b - a for a, b in tiles}, reverse=True),
        topk_bucket=1 << (n_k - 1).bit_length(),
        tile_mib=full_rows * n_k * 4 / 2**20,
    )


def _envelope_tiles(env) -> list:
    """The (query rows, keys) tiles this rank's fp8_mqa_logits and prefill top-k get at the 1M
    config, from the chunks the engine's own planner gives at max_seq_len 1048576 and
    max_num_tokens 32768, run through the target's own cut and tiling under the budget it takes
    from that engine (`_prefill_logits_budget`: 2**28): one 32768-token chunk of a
    1048576-token prompt split over the ranks, over each cached prefix; an unsplit step of 8191
    tokens whose last key is the 1048576th; two 5-token tails over 983040-token prefixes. The
    top-k compiles one kernel per next power of two of the keys."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa import indexer as engine

    M = env.M
    budget = M._prefill_logits_budget(MAX_NUM_TOKENS, ENVELOPE_MAX_SEQ_LEN)
    assert budget == M._PREFILL_LOGITS_CEILING == 1 << 28, budget
    rows = []

    def plan(cached, new, q_begin=0, q_end=None):
        md = _envelope_md(cached, new)
        specs = engine.plan_mla_chunked_prefill_chunks(
            new,
            [c + n for c, n in zip(cached, new)],
            engine._packed_logits_elem_budget(md),
            q_begin,
            q_end,
        )
        chunks = [engine.Indexer.prepare_one_prefill_chunk(md, s) for s in specs]
        kv = [c + n for c, n in zip(cached, new)]
        step = dict(
            num_ctx_tokens=sum(new),
            max_seq_len=ENVELOPE_MAX_SEQ_LEN,
            host_cu_ctx_new=[0, *itertools.accumulate(new)],
            host_cu_ctx_kv=[0, *itertools.accumulate(kv)],
        )
        return chunks, M._indexer_blocks(chunks, step)

    new = MAX_NUM_TOKENS
    q_begin, q_end, q_sizes = engine._split_prefill_queries(new, env.mapping, Q_SPLIT_THRESHOLD)
    assert q_sizes == [new // env.world] * env.world, q_sizes
    for cached in ENVELOPE_CACHED:
        chunks, as_run = plan([cached], [new], q_begin, q_end)
        assert len(chunks) == 1 and len(as_run) == 1, _chunks_summary(chunks)
        chunk = as_run[0]
        assert (chunk.token_start, chunk.token_end) == (q_begin, q_end)
        n_k = chunk.k_token_end - chunk.k_token_start
        assert n_k == cached + q_end, (n_k, cached, q_end)
        assert int(chunk.cu_seqlen_ke[-1]) == n_k
        rows.append(_tiles_row(M, chunk, budget, case="split 32768", cached=cached))

    cached = ENVELOPE_MAX_SEQ_LEN - ENVELOPE_UNSPLIT
    unsplit = engine._split_prefill_queries(ENVELOPE_UNSPLIT, env.mapping, Q_SPLIT_THRESHOLD)
    assert unsplit[2] is None, unsplit
    chunks, as_run = plan([cached], [ENVELOPE_UNSPLIT])
    assert len(as_run) == 1 and as_run[0].k_token_end - as_run[0].k_token_start == 1 << 20
    rows.append(_tiles_row(M, as_run[0], budget, case=f"unsplit {ENVELOPE_UNSPLIT}", cached=cached))

    tails_cached = [c for c, _ in ENVELOPE_TAILS]
    tails_new = [n for _, n in ENVELOPE_TAILS]
    chunks, as_run = plan(tails_cached, tails_new)
    packed = _max_keys(chunks)
    assert len(chunks) == 1 and packed == sum(c + n for c, n in ENVELOPE_TAILS), packed
    assert len(as_run) == len(ENVELOPE_TAILS), _chunks_summary(as_run)
    for part in as_run:
        rows.append(_tiles_row(M, part, budget, case="tails", cached=tails_cached, packed=packed))
    return rows


def _spawned_rank(world: int):
    """This process as a rank of the `world` this pytest started under mpirun (`_rank_entry`),
    on its own device: its rank and the MPI session's communicator."""
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    import torch

    torch.cuda.set_device(int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]))
    from tensorrt_llm._utils import mpi_comm, mpi_rank, mpi_world_size

    comm = mpi_comm()
    assert os.environ.get("TLLM_DISABLE_MPI") != "1", "the gather runs on the MPI session"
    assert size == world == mpi_world_size() == comm.Get_size(), (size, mpi_world_size())
    assert rank == mpi_rank() == comm.Get_rank(), (rank, mpi_rank())
    return rank, comm


def _launched_rank(world: int):
    """This pytest process as a rank of the `world` an MPI launcher started, on its own device:
    its rank and the MPI session's communicator."""
    import torch

    from tensorrt_llm._utils import mpi_comm, mpi_disabled, mpi_rank

    comm = mpi_comm()
    rank, size = mpi_rank(), comm.Get_size()
    torch.cuda.set_device(int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]))
    assert not mpi_disabled(), "the gather runs on the MPI session"
    assert size == world == get_mpi_world_size(), (size, get_mpi_world_size())
    return rank, comm


def _rank_main(ckpt_path: Path, world: int, rank: int, comm) -> None:
    import torch

    device = torch.cuda.current_device()
    devices = comm.allgather(str(torch.cuda.get_device_properties(device).uuid))
    assert len(set(devices)) == world, "every rank must own its own device"
    hosts = comm.allgather(socket.gethostname())

    def log(msg: str) -> None:
        print(f"[rank {rank}] {msg}", flush=True)

    log(f"on {hosts[rank]} device {device}; ranks over {len(set(hosts))} node(s): {hosts}")
    started = time.monotonic()
    env = _build(ckpt_path, rank, world)
    _instrument(env)
    assert env.core.tp_group == list(range(world)) and env.core.tp_pg is None
    # Nothing is sized before the first step (`_check_step` hands the core its metadata).
    assert (env.core.prefill_logits_budget, env.core.ctx_workspace_rows) == (0, 0)
    assert _sparse_config().q_split_threshold == Q_SPLIT_THRESHOLD
    L = env.layers
    assert L[1].indexer.rope and not L[3].indexer.rope and L[17].indexer is None
    log(f"built in {time.monotonic() - started:.1f}s")
    reports = []
    for k, step in enumerate(env.steps):
        t0 = time.monotonic()
        reports.append(_check_step(env, k, step, log))
        log(f"step {k} ({step.name}) passed in {time.monotonic() - t0:.1f}s")
    # The sizes the first step's metadata gave: the logits budget at max_num_tokens 32768 and
    # the world's max_seq_len, and, chunked prefill on, the held workspace's 2**19 rows.
    budget = env.M._prefill_logits_budget(MAX_NUM_TOKENS, env.max_seq_len)
    assert env.core.prefill_logits_budget == budget == {4: 1 << 26, 8: 1 << 24}[world], budget
    assert env.core.ctx_workspace_rows == env.M._CTX_WORKSPACE_ROWS == 1 << 19
    assert env.prefill._ws_buf.shape == (1 << 19, LATENT)
    log(f"sized on the first step: logits budget {budget}, held context workspace 2**19 rows")
    timing = _time_gather(env, comm)
    tiles = _envelope_tiles(env)

    everyone = comm.allgather(
        dict(host=hosts[rank], reports=reports, timing=timing, envelope_tiles=tiles)
    )
    for k in range(len(env.steps)):
        for i in DRIVEN:
            digests = {r["reports"][k]["layers"][i]["digest"] for r in everyone}
            assert len(digests) == 1, f"step {k} layer {i}: the ranks hold different selections"
    if rank == 0:
        print(json.dumps(dict(ranks=everyone), indent=1, default=str), flush=True)
        for r, member in enumerate(everyone):
            for t in member["envelope_tiles"]:
                print(f"[envelope tiles] rank {r} {t}", flush=True)
            print(f"[gather] rank {r} {member['host']} {member['timing']}", flush=True)
        print(f"[q-split] every check passed on every rank, all {world}", flush=True)
    comm.Barrier()


def _rank_entry() -> None:
    """A rank `_start_ranks` started: `<this file> --q-split-rank <checkpoint> <world>`."""
    try:
        flag = sys.argv.index(_RANK_FLAG)
        world = int(sys.argv[flag + 2])
        rank, comm = _spawned_rank(world)
        _rank_main(Path(sys.argv[flag + 1]), world, rank, comm)
        sys.stdout.flush()
        # mpirun fails a rank that exits without finalizing.
        from mpi4py import MPI

        MPI.Finalize()
    except BaseException:
        print(f"[rank {os.environ.get('OMPI_COMM_WORLD_RANK')}] FAILED", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(13)
    os._exit(0)


def _finalize(config, session) -> None:
    """After pytest's report: MPI finalized (mpirun fails a rank that leaves without it), then
    out at once with pytest's own status, ahead of the interpreter's teardown of the ranks'
    communicators."""
    from mpi4py import MPI

    config.get_terminal_writer().flush()
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        stream.flush()
    MPI.Finalize()
    os._exit(int(session.exitstatus))


# ---------------------------------------------------------------- the launcher (pytest)


def _checkpoint_dir() -> Path | None:
    from test_common.llm_data import llm_models_root

    root = llm_models_root()
    if root is None:
        return None
    for name in CHECKPOINTS:
        path = Path(root) / name
        if (path / "params.json").is_file() and (path / SHARD_INDEX).is_file():
            return path
    return None


def _session_members(sid: int) -> list[int]:
    """Every process in session `sid`, read from /proc."""
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        fields = stat[stat.rindex(")") + 2 :].split()
        if int(fields[3]) == sid:
            members.append(int(entry.name))
    return members


def _end_job(job) -> None:
    """End mpirun and every rank it started (each rank sits in a process group of its own,
    inside mpirun's session)."""
    if job.poll() is None:
        job.terminate()
        try:
            job.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    for pid in _session_members(job.pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if job.poll() is None:
        job.kill()
    job.wait()


def _start_ranks(path: Path, world: int, tmp_path: Path) -> None:
    """tep4: the `world` ranks on this node, started under mpirun as `_rank_entry`, their log
    read back."""
    import torch

    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs, {torch.cuda.device_count()} visible")
    mpirun = shutil.which("mpirun")
    assert mpirun is not None, "the ranks start under mpirun; none on PATH"
    repo_root = Path(__file__).resolve().parents[7]
    assert (repo_root / "tensorrt_llm").is_dir(), repo_root
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_")) and k != "TLLM_DISABLE_MPI"
    }
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)
    command = [mpirun, "-n", str(world), "--oversubscribe", "--allow-run-as-root"]
    command += [sys.executable, str(Path(__file__).resolve()), _RANK_FLAG, str(path), str(world)]
    log_path = tmp_path / "ranks.log"
    with open(log_path, "w") as log:
        job = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = job.wait(timeout=DEADLINE_S)
        except subprocess.TimeoutExpired:
            _end_job(job)
            code = None
    output = log_path.read_text()
    print(output)
    assert code is not None, (
        f"the {world} ranks did not finish in {DEADLINE_S}s:\n{output[-20000:]}"
    )
    assert code == 0, f"mpirun exited {code}:\n{output[-20000:]}"
    assert "[q-split] every check passed on every rank" in output


def _run_as_rank(path: Path, world: int, request) -> None:
    """tep8: this pytest process as one of the `world` ranks."""
    request.config.add_cleanup(lambda: _finalize(request.config, request.session))
    faulthandler.dump_traceback_later(DEADLINE_S, exit=True)
    try:
        rank, comm = _launched_rank(world)
        _rank_main(path, world, rank, comm)
    except BaseException:
        # The other ranks wait in a collective for this one: leave now, so the launcher ends
        # the job instead of the deadline.
        print(f"[rank {os.environ.get('OMPI_COMM_WORLD_RANK')}] FAILED", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(13)
    finally:
        faulthandler.cancel_dump_traceback_later()


@pytest.mark.no_xdist
@pytest.mark.parametrize("world", WORLDS)
def test_indexer_q_split_on_every_rank(world, request):
    """Every check above on every rank of the world: at tep4 the four ranks this pytest starts,
    at tep8 the eight launched pytest processes, this one among them."""
    path = _checkpoint_dir()
    if path is None:
        pytest.skip("checkpoint Mistral-Large-4-1T-A52B-NVFP4 is not under LLM_MODELS_ROOT")
    if world > RANKS_PER_NODE:
        _run_as_rank(path, world, request)
    else:
        _start_ranks(path, world, request.getfixturevalue("tmp_path"))


if __name__ == "__main__" and _RANK_FLAG in sys.argv:
    _rank_entry()
