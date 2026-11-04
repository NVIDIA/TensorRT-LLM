# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp4 target's indexer query split across its TP ranks: at tp4 (four ranks, on one node of
four GPUs) and at tp8 (eight ranks over two nodes of four).

Under chunked prefill, once a step's context rows reach `q_split_threshold` (8192), the engine
splits them over the TP ranks: `Indexer.prepare_for_chunked_prefill` (`_split_prefill_queries`,
then `plan_mla_chunked_prefill_chunks` over this rank's share) sets
`indexer_prefill_q_split = (q_begin, q_end, q_sizes)` and plans `indexer_prefill_chunks` over
rows `[q_begin, q_end)` only. `PrefillTarget._indexer` selects for those rows, then every full
indexer layer all-gathers every rank's share -- four at tp4, eight at tp8 -- (int32
`[rows, 2048]`, uneven sizes in rank order) into rows `[0, tc)` of the step's selection buffer,
through the catalog's `comm/allgather` on the MPI session, before any layer reads it.

Each rank builds the tp4 core on the meta device at its world's mapping and fills, through the
weight table's own manifest (its tensor-parallel slices -- 16 heads a rank at tp4, 8 at tp8 --
and load-time transforms), the attention and indexer weights of three layers of the checkpoint:
1 (a full-indexer RoPE layer), 3 (a full-indexer NoPE layer: its indexer reads the identity
table) and 17 (a 'shared' layer, which attends with layer 3's selection). The inputs are seeded:
real token embeddings of seeded random ids from an 8192-token block of the vocabulary, through
each layer's own attention norm. The KV pools are the engine's `MistralLarge4CacheManager` built
from the checkpoint's own config (the fp8_ds_mla latent pool on the 63 KV owners, the indexer key
pool on the 44 full-indexer layers), and every step's metadata is `DSAtrtllmAttentionMetadata`
prepared by the engine itself (`prepare()` runs the indexer's planning) with chunked prefill on,
as the engine builds it (`enable_context_mla_with_cached_kv`, `chunk_size` 32768). Every key and
latent in the pools was written by the target's own appends (`ds_mla_rope_append` for the
latent), over the steps before. The engine is at max_seq_len 131072 with chunked prefill, and
the target sizes itself from that on the first step (`_derive_first_step`): its indexer logits
budget `_prefill_logits_budget` is the ceiling, 1 << 28 (2048-row tiles at 131072 keys), and its
held context workspace `_ctx_workspace_rows` 2**19 rows, since a step can attend cached prefixes.
Every step's context KV fits it, so the context rows attend on the workspace path: each
request's cached and new latent rows dequantized into the held workspace
(`PrefillTarget._ctx_workspace`, `ds_mla_dequant_gather`), then FlashMLA's sparse prefill over
it; every step after R's first chunk reads a cached prefix there.

What "the same selection" can mean here is set by the top-k entries, not by the split. The
decode top-k returns a long row's 2048 winners in unspecified order and may break a tie at the
2048th score either way, and the prefill top-k (`cute_dsl_gvr_topk_prefill`) keeps the lowest
columns among such a tie, which another top-k need not; `fp8_mqa_logits` is bitwise per query
tile. So every step runs three times from identical pools, each through the target's
`_attention`: on the engine's split plan at this rank's TP mapping, and twice on the unsplit
plan (the same metadata with `q_split_threshold` past any step, so the chunks cover every
context row); the second unsplit run is the kernel's own run-to-run variation. An observe-only
wrapper around the target module's two top-k entries keeps, for every row it selects, what it
wrote and the scores of the keys it chose, sorted. Asserted for every step, layer and rank:

* the engine's split is the expected one and its chunks tile exactly this rank's rows;
* the context workspace holds every context request's cached and new rows (the engine's
  `num_ctx_kv_tokens` counts the new ones only);
* with the selection buffer filled with a sentinel before each full layer, no sentinel survives
  anywhere in the step's rows;
* after the gather, this rank's rows are bitwise what its own top-k calls wrote, and the
  context rows of the gathered buffer are bitwise the same on every rank;
* per row, the selected keys' scores are bitwise the unsplit computation's, as a multiset: the
  two are the same top-k answer, up to the kernel's order and its choice among keys tied at the
  2048th score. Decode rows behind the context rows, which every rank scores itself, likewise;
* each layer's attention output (this rank's heads), recomputed through `_attention` with the
  layer's indexer detached over each selection put in ascending order, is bitwise the unsplit
  one on every row whose selection is the same set; the rows that differ (exact ties) are
  counted and their output difference printed beside the unsplit-vs-unsplit control's.

The steps, in order, on one pool, with the expected q_sizes at tp4; at tp8 they are the same
split over eight ranks (rank r takes `[tc * r // 8, tc * (r + 1) // 8)`):

* request R, 98304 tokens in three 32768-row chunks (even split, 8192 rows per rank at tp4, 4096
  at tp8): over no prefix, a 32768-token prefix, then a 65536-token prefix, where each rank's
  keys reach past 65536 and the 1 << 28 logits budget cuts its share into 3 or 4 tiles at tp4, 2
  at tp8;
* three first chunks in one step, 4096 + 20000 + 3000 = 27096 rows (6774 per rank at tp4, 3387
  at tp8): rank boundaries cut a request;
* 32766 context rows over three requests with cached prefixes of 4096, 0 and 20000 tokens
  (q_sizes 8191, 8192, 8191, 8192; at tp8 4095, then 4096 three times, twice over), with two
  decode rows behind them (R at 98304 and a 3000-token request): 32768 rows, a full step;
* 32769 context rows (q_sizes 8192, 8192, 8192, 8193; at tp8 seven of 4096 and one of 4097), one
  request over a 30766-token prefix and a fresh one: one row past max_num_tokens, the split's
  uneven-by-one case;
* 8191 context rows, just below the threshold, with R's next decode row: no split;
* 8192 context rows, at the threshold: split 2048 per rank at tp4, 1024 at tp8.

A negative control on R's third chunk runs layer 1's indexer with the split's chunks and the
gather withheld: the other ranks' rows keep the sentinel, which is what the target without the
gather attended over.

At tp4 the test starts the ranks itself: one `mpirun -n 4 --oversubscribe --allow-run-as-root`
job on this node (the launch line's MPI world, without `trtllm-llmapi-launch`, which runs a
command on rank 0 only), its environment scrubbed of the allocation's SLURM/PMI/Open MPI
variables, and reads the four ranks' results back; it runs unlaunched or inside a launch of four
ranks, on four GPUs. At tp8 each of the eight ranks is a pytest process an MPI launcher started
over two nodes of four, not under `trtllm-llmapi-launch` (e.g. `srun --mpi=pmix --nodes=2
--ntasks-per-node=4 python -m pytest <this file> -k tp8`): each runs its rank in place on a device
of its node and every rank gathers every rank's results. A tp8 rank that fails prints its
traceback and leaves at once, which ends the launcher's job; after a pass every rank finalizes
MPI once pytest has reported. Each world is skipped inside a launch of another. Needs sm_103 and
the checkpoint under `LLM_MODELS_ROOT`; the per-step numbers are printed (run with -rA).
"""

from __future__ import annotations

import faulthandler
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

#: A node's GPUs: tp4 runs its four ranks on one node, tp8 its eight over two.
RANKS_PER_NODE = 4
#: The worlds: tp4, four ranks one mpirun job this test starts runs; tp8, eight ranks over two
#: nodes, each a pytest process an MPI launcher started.
WORLDS = [pytest.param(4, id="tp4"), pytest.param(8, id="tp8")]
CHECKPOINTS = ("Mistral-Large-4-1T-A52B-NVFP4", "mistral-large-4-nvfp4")
SHARD_INDEX = "consolidated.safetensors.index.json"

LAYERS = 72
TOKENS_PER_BLOCK = 32
LATENT = 576
INDEX_TOPK = 2048
#: The checkpoint's index_sharing as the engine's DSA pattern: 17 indexer layers, then each
#: indexer pair followed by two layers that reuse its selection.
PATTERN = "F" * 17 + "SSFF" * 13 + "SSF"
#: The engine's token budget, which is also its chunked-prefill chunk.
MAX_NUM_TOKENS = 32768
#: The engine's default `q_split_threshold`: a step of this many context rows or more has its
#: indexer queries split over the TP ranks.
Q_SPLIT_THRESHOLD = 8192
#: The metadata's token capacity: room for the 32769-row step, one row past the engine's
#: max_num_tokens, which the split's arithmetic asks for.
STEP_TOKENS = 32776
MAX_SEQ_LEN = 131072
NUM_BLOCKS = 8192
#: The layers driven, in order: full-indexer RoPE, full-indexer NoPE, and a 'shared' layer.
DRIVEN = (1, 3, 17)
#: The attention and indexer weight roles of a driven layer.
ATTN_ROLES = (
    "attn_norm", "wq", "wq_s", "ssmax_s", "ssmax_b", "wkv_a", "kv_a_norm", "wkv_b",
    "ix_q_norm_w", "ix_q_norm_b", "ix_wq_b", "ix_wq_b_s", "ix_wk_wp", "ix_k_norm_w",
    "ix_k_norm_b", "ix_ssmax_s", "ix_ssmax_b",
)  # fmt: skip
#: int32, neither -1 (an empty slot) nor a position.
SENTINEL = -(2**31) + 7
#: A threshold no step reaches: the engine then plans every context row on every rank.
UNSPLIT_THRESHOLD = 1 << 30
SEED = 1051
#: The token embeddings the inputs are drawn from: a contiguous block of the vocabulary.
EMB_LO, EMB_ROWS = 1000, 8192
#: The tp4 job, and each tp8 rank's whole run (compiles and checkpoint reads included); past it a
#: tp8 rank dumps its stacks and exits, which ends the launcher's job.
DEADLINE_S = 3600
_RANK_FLAG = "--q-split-rank"

#: Request id -> prompt length, allocated when the request is first scheduled.
PROMPTS = {1: 98304, 2: 14096, 3: 47151, 4: 3000, 5: 12000, 6: 16384, 7: 8191, 8: 8192}
#: (name, context rows [(request, cached, new)], decode rows [(request, cached)], expected
#: q_sizes at tp4 or None; `_q_sizes` gives them at any world).
STEPS = (
    ("R chunk 1, no prefix", [(1, 0, 32768)], [], [8192] * 4),
    ("R chunk 2, 32768 prefix", [(1, 32768, 32768)], [], [8192] * 4),
    ("R chunk 3, 65536 prefix", [(1, 65536, 32768)], [], [8192] * 4),
    ("three first chunks", [(2, 0, 4096), (3, 0, 20000), (4, 0, 3000)], [], [6774] * 4),
    (
        "three prefixes + 2 decode rows",
        [(2, 4096, 10000), (5, 0, 12000), (3, 20000, 10766)],
        [(1, 98304), (4, 3000)],
        [8191, 8192, 8191, 8192],
    ),
    ("32769 rows", [(3, 30766, 16385), (6, 0, 16384)], [], [8192, 8192, 8192, 8193]),
    ("below the threshold + 1 decode row", [(7, 0, 8191)], [(1, 98305)], None),
    ("at the threshold", [(8, 0, 8192)], [], [2048] * 4),
)
#: The step whose layer-1 indexer also runs with the gather withheld.
CONTROL_STEP = 2


# ===================================================================== the ranks


def _mapping(rank: int, world: int):
    """The engine's mapping at tp4 or tp8: tensor parallel throughout over `world` ranks on nodes
    of four, the routed experts tensor parallel, no attention DP."""
    from tensorrt_llm.mapping import Mapping

    return Mapping(
        world_size=world,
        rank=rank,
        gpus_per_node=RANKS_PER_NODE,
        tp_size=world,
        moe_tp_size=world,
        moe_ep_size=1,
    )


def _sparse_config(**kwargs):
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

    return DeepSeekSparseAttentionConfig(
        index_n_heads=32,
        index_head_dim=128,
        index_topk=INDEX_TOPK,
        indexer_rope_interleave=True,
        **kwargs,
    )


def _manager(pretrained_config):
    """The engine's KV cache manager for this checkpoint, from its own config: the fp8_ds_mla
    latent pool (656 B per token on each of the 63 KV owners), page 32, and the 44-layer indexer
    key pool."""
    from tensorrt_llm._torch.attention.backends.sparse.mistral_large4 import (
        MistralLarge4CacheManager,
    )
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    assert pretrained_config.index_topk_pattern == PATTERN
    return MistralLarge4CacheManager(
        KvCacheConfig(
            enable_block_reuse=False, max_tokens=NUM_BLOCKS * TOKENS_PER_BLOCK, host_cache_size=0
        ),
        CacheType.SELFKONLY,
        num_layers=LAYERS,
        num_kv_heads=1,
        head_dim=LATENT,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=16,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.FP8,
        sparse_attention_config=_sparse_config(),
        pretrained_config=pretrained_config,
        max_num_tokens=STEP_TOKENS,
    )


def _prepare(manager, ctx, gen, rank: int, world: int, **sparse):
    """`DSAtrtllmAttentionMetadata` for one step, prepared as the engine prepares it under
    chunked prefill: contexts first, each over its cached prefix, then the decode rows."""
    from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
    from tensorrt_llm._torch.attention.backends.sparse.dsa import DSAtrtllmAttentionMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    rids = [r for r, _, _ in ctx] + [r for r, _ in gen]
    md = DSAtrtllmAttentionMetadata(
        seq_lens=torch.tensor([n for _, _, n in ctx] + [1] * len(gen), dtype=torch.int),
        request_ids=rids,
        max_num_requests=16,
        max_num_sequences=16,
        num_contexts=len(ctx),
        prompt_lens=[PROMPTS[r] for r in rids],
        max_num_tokens=STEP_TOKENS,
        kv_cache_manager=manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[c for _, c, _ in ctx] + [c for _, c in gen],
        ),
        mapping=_mapping(rank, world),
        runtime_features=AttentionRuntimeFeatures(
            chunked_prefill=True, cache_reuse=False, chunk_size=MAX_NUM_TOKENS
        ),
        enable_context_mla_with_cached_kv=True,
        sparse_attention_config=_sparse_config(**sparse),
    )
    md.prepare()
    return md


class _Checkpoint:
    """Raw tensors by Mistral name, read lazily from the safetensors files, and the
    configuration the engine adapts from its params.json."""

    def __init__(self, path: Path) -> None:
        from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader

        self.path = path
        self.index = json.loads((path / SHARD_INDEX).read_text())["weight_map"]
        self.config = MistralConfigLoader().load_pretrained_config(str(path))

    def entry(self, key: str):
        from safetensors import safe_open

        f = safe_open(self.path / self.index[key], framework="pt", device="cpu")
        return f.get_slice(key)


def _materialize(W, core, ck: _Checkpoint, keys) -> None:
    """Fill the named parameters of a meta-declared core through the weight table's manifest:
    this rank's slices, multi-source rows and load-time transforms."""
    manifest = W.MODEL_WEIGHTS.manifest(core)
    for key in keys:
        meta = core.w[key]
        dst = torch.empty(meta.shape, dtype=meta.dtype, device="cuda")
        for ckpt_key, index, transform, shard in manifest[key]:
            entry = ck.entry(ckpt_key)
            src = entry[...] if shard is None else shard(entry)
            src = src.to("cuda").contiguous()
            if transform is not None:
                src = transform(src, core)
            (dst if index is None else dst[index]).copy_(src)
        core.w[key] = torch.nn.Parameter(dst, requires_grad=False)


class _Recorder:
    """Observe-only wrapper of the target module's two top-k entries.

    While armed with a step's selection buffer, it keeps per absolute row of that buffer what
    each top-k call wrote there (`local`) and the scores of the keys it chose, sorted
    descending (`scores`, -inf for an empty slot): the multiset that makes two top-k answers
    the same up to the kernel's order and its choice among keys tied at the 2048th score.
    Rows no call wrote stay SENTINEL / NaN."""

    def __init__(self, M) -> None:
        self.prefill_topk = M.cute_dsl_gvr_topk_prefill
        self.decode_topk = M.indexer_topk_decode
        M.cute_dsl_gvr_topk_prefill = self._prefill
        M.indexer_topk_decode = self._decode
        self.buf = None

    def arm(self, buf: torch.Tensor) -> None:
        self.buf = buf
        self.local = torch.full_like(buf, SENTINEL)
        self.scores = torch.full(buf.shape, float("nan"), dtype=torch.float32, device=buf.device)

    def take(self) -> SimpleNamespace:
        self.buf = None
        return SimpleNamespace(local=self.local, scores=self.scores)

    def _record(self, out, logits, cols, valid) -> None:
        if self.buf is None:
            return
        row_bytes = self.buf.stride(0) * self.buf.element_size()
        r0 = (out.data_ptr() - self.buf.data_ptr()) // row_bytes
        n = out.shape[0]
        self.local[r0 : r0 + n] = out
        chosen = torch.where(valid, torch.gather(logits, 1, cols), float("-inf"))
        self.scores[r0 : r0 + n] = torch.sort(chosen, dim=1, descending=True).values

    def _prefill(self, logits, ks, ke, out, top_k, *args, **kwargs):
        res = self.prefill_topk(logits, ks, ke, out, top_k, *args, **kwargs)
        idx = out.long()
        # Window-local indices: the score's column is the index plus the row's window start.
        self._record(out, logits, (idx + ks.long()[:, None]).clamp(min=0), idx >= 0)
        return res

    def _decode(self, logits, kv_lens, out, *args, **kwargs):
        res = self.decode_topk(logits, kv_lens, out, *args, **kwargs)
        idx = out.long()
        self._record(out, logits, idx.clamp(min=0), idx >= 0)
        return res


def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf.to(torch.bfloat16) * w


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


def _sentinel_rows(sel: torch.Tensor) -> int:
    return int((sel == SENTINEL).any(dim=1).sum())


def _canonical(sel: torch.Tensor) -> torch.Tensor:
    """Each row's selection in ascending order, empty slots (-1) last."""
    big = torch.iinfo(torch.int32).max
    ordered = sel.masked_fill(sel < 0, big).sort(dim=1).values
    return ordered.masked_fill(ordered == big, -1)


def _chunks(step, budget: int) -> list[list[int]]:
    """`[token_start, token_end, n_k, tiles, largest logits tile in bytes]` per engine chunk."""
    out = []
    for c in step["indexer_prefill_chunks"] or []:
        rows, n_k = c.token_end - c.token_start, c.k_token_end - c.k_token_start
        tile = max(1, budget // max(1, n_k))
        out.append([c.token_start, c.token_end, n_k, -(-rows // tile), min(tile, rows) * n_k * 4])
    return out


def _setup(prefill, md):
    """The step's set-up, as the target's forward does it."""
    step = prefill.step_args(md)
    rows, tc = int(step["num_tokens"]), int(step["num_ctx_tokens"])
    gen = rows - tc
    skip_ctx = step["skip_ctx"]
    skip_gen = step["skip_gen"] if gen else True
    req_idx = step["req_idx_per_token"][:rows]
    return SimpleNamespace(
        step=step,
        rows=rows,
        tc=tc,
        skip_ctx=skip_ctx,
        skip_gen=skip_gen,
        req_idx=req_idx,
        scratch=prefill._scratch(rows, torch.bfloat16, "cuda"),
        ctx=prefill._ctx_workspace(step, tc, "cuda") if tc else None,
    )


def _drive(env, md, xs, pos) -> SimpleNamespace:
    """One step through the driven layers with the target's own `_attention`; the selection
    buffer is refilled with the sentinel, and the recorder armed, before each full layer."""
    prefill = env.prefill
    s = _setup(prefill, md)
    topk_buf = prefill._topk_buf(s.step, s.rows, s.tc, s.skip_ctx, s.skip_gen)
    outs, sels, recs, peak, ms = {}, {}, {}, {}, {}
    for i in DRIVEN:
        L = env.layers[i]
        if L.indexer is not None:
            topk_buf.fill_(SENTINEL)
            env.recorder.arm(topk_buf)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        out = prefill._attention(
            s.step, L, xs[i], pos, s.req_idx, topk_buf, s.scratch, s.ctx, s.tc, s.skip_ctx,
            s.skip_gen,
        )  # fmt: skip
        torch.cuda.synchronize()
        ms[i] = (time.monotonic() - started) * 1e3
        peak[i] = torch.cuda.max_memory_allocated() - base
        if L.indexer is not None:
            recs[i] = env.recorder.take()
        outs[i] = out.clone()
        sels[i] = topk_buf.clone()
    s.outs, s.sels, s.recs, s.peak, s.ms = outs, sels, recs, peak, ms
    return s


def _attend(env, md, xs, pos, sels) -> dict:
    """Each driven layer's attention through `_attention` with its indexer detached, over the
    given selection put in ascending order."""
    prefill = env.prefill
    s = _setup(prefill, md)
    outs = {}
    for i in DRIVEN:
        detached = SimpleNamespace(**{**vars(env.layers[i]), "indexer": None})
        buf = _canonical(sels[i])
        out = prefill._attention(
            s.step, detached, xs[i], pos, s.req_idx, buf, s.scratch, s.ctx, s.tc, s.skip_ctx,
            s.skip_gen,
        )  # fmt: skip
        outs[i] = out.clone()
    return outs


def _sets_differ(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (_canonical(a) != _canonical(b)).any(dim=1)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().max())


def _rank_run(rank: int, world: int, ckpt_path: Path) -> dict:
    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tp4 import (  # noqa: E501
        modeling as M,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tp4 import (  # noqa: E501
        weights as W,
    )
    from tensorrt_llm._torch.model_config import ModelConfig

    ck = _Checkpoint(ckpt_path)
    model_config = ModelConfig(
        pretrained_config=ck.config,
        mapping=_mapping(rank, world),
        max_num_tokens=MAX_NUM_TOKENS,
        max_seq_len=MAX_SEQ_LEN,
    )
    with torch.device("meta"):
        core = M.MistralLarge4Tp4ModelingV2(model_config)
    assert core.tp_group == list(range(world)) and core.tp_pg is None
    # The engine's sizes are fixed on the first step (`_derive_first_step`), not at construction.
    assert (core.prefill_logits_budget, core.ctx_workspace_rows) == (0, 0)
    assert _sparse_config().q_split_threshold == Q_SPLIT_THRESHOLD
    keys = ["lerope_alpha", "final_norm"]
    keys += [f"l{i}_{r}" for i in DRIVEN for r in ATTN_ROLES if f"l{i}_{r}" in core.w]
    _materialize(W, core, ck, keys)
    core.build_kernels(torch.cuda.current_device())
    prefill = M.PrefillTarget(core)
    env = SimpleNamespace(
        prefill=prefill,
        layers={L.index: L for L in prefill._layers},
        recorder=_Recorder(M),
    )
    assert [env.layers[i].indexer is not None for i in DRIVEN] == [True, True, False]
    assert env.layers[3].indexer.rope is False and env.layers[1].indexer.rope is True

    emb = ck.entry("tok_embeddings.weight")[EMB_LO : EMB_LO + EMB_ROWS].to("cuda")
    eps = ck.config.rms_norm_eps
    norms = {i: core.w[f"l{i}_attn_norm"] for i in DRIVEN}
    gen_ids = torch.Generator().manual_seed(SEED)
    ids = {r: torch.randint(0, EMB_ROWS, (n + 8,), generator=gen_ids) for r, n in PROMPTS.items()}

    manager = _manager(model_config.pretrained_config)
    allocated: set[int] = set()
    result = dict(rank=rank, budget=0, held_rows=0, steps=[])
    try:
        for n_step, (name, ctx, gen, _) in enumerate(STEPS):
            for r, _, _ in ctx:
                if r not in allocated:
                    manager.add_dummy_requests([r], token_nums=[PROMPTS[r]])
                    allocated.add(r)
            for r, cached in gen:
                # One more token, as a decode step allocates it.
                cache = manager.kv_cache_map[r]
                if cache.capacity < cached + 1:
                    assert cache.resize(cached + 1)
            md_split = _prepare(manager, ctx, gen, rank, world)
            md_ref = _prepare(manager, ctx, gen, rank, world, q_split_threshold=UNSPLIT_THRESHOLD)
            assert md_split.enable_context_mla_with_cached_kv
            assert md_split.runtime_features.chunked_prefill
            core._derive_first_step(md_split)
            # At max_seq_len 131072 with chunked prefill: the logits budget's ceiling, and the
            # held workspace's 2**19 rows.
            assert md_split.max_seq_len == MAX_SEQ_LEN
            assert core.prefill_logits_budget == M._prefill_logits_budget(
                md_split.max_num_tokens, MAX_SEQ_LEN
            )
            assert core.ctx_workspace_rows == M._ctx_workspace_rows(md_split) == 1 << 19
            result["budget"] = core.prefill_logits_budget
            result["held_rows"] = core.ctx_workspace_rows
            tok = torch.cat(
                [ids[r][c : c + n] for r, c, n in ctx] + [ids[r][c : c + 1] for r, c in gen]
            ).to("cuda")
            pos = torch.cat(
                [torch.arange(c, c + n) for _, c, n in ctx] + [torch.tensor([c]) for _, c in gen]
            ).to(device="cuda", dtype=torch.int32)
            h = emb[tok]
            xs = {i: _rmsnorm(h, norms[i], eps) for i in DRIVEN}

            split = _drive(env, md_split, xs, pos)
            ref = _drive(env, md_ref, xs, pos)
            ref2 = _drive(env, md_ref, xs, pos)
            q_split = split.step["indexer_prefill_q_split"]
            assert ref.step["indexer_prefill_q_split"] is None
            rows, tc = split.rows, split.tc
            begin, end = (0, tc) if q_split is None else (q_split[0], q_split[1])
            canon = dict(
                split=_attend(env, md_split, xs, pos, split.sels),
                ref=_attend(env, md_ref, xs, pos, ref.sels),
            )
            rec = dict(
                name=name,
                rows=rows,
                tc=tc,
                skip_ctx=bool(split.skip_ctx),
                q_split=None if q_split is None else [q_split[0], q_split[1], list(q_split[2])],
                chunks_split=_chunks(split.step, core.prefill_logits_budget),
                chunks_ref=_chunks(ref.step, core.prefill_logits_budget),
                gather_bytes=0 if q_split is None else tc * INDEX_TOPK * 4,
                # The context workspace's rows (every context request's cached + new latent
                # rows) beside the engine's count of the step's new context keys.
                workspace_rows=0 if split.ctx is None else int(split.ctx.ws_buf.shape[0]),
                num_ctx_kv_tokens=int(split.step["num_ctx_kv_tokens"]),
                layers={},
            )
            for i in DRIVEN:
                full = env.layers[i].indexer is not None
                a, b, b2 = split.sels[i], ref.sels[i], ref2.sels[i]
                differ = _sets_differ(a, b)
                same = ~differ
                lay = dict(
                    full=full,
                    sentinel_rows=[_sentinel_rows(a), _sentinel_rows(b), _sentinel_rows(b2)],
                    rows_order_differ=int((a != b).any(dim=1).sum()),
                    rows_sets_differ=int(differ.sum()),
                    control_rows_order_differ=int((b2 != b).any(dim=1).sum()),
                    control_rows_sets_differ=int(_sets_differ(b2, b).sum()),
                    digest_split_ctx=_digest(a[:tc]),
                    out_native_max_abs=_max_abs(split.outs[i], ref.outs[i]),
                    control_out_native_max_abs=_max_abs(ref2.outs[i], ref.outs[i]),
                    out_canonical_same_rows_bitwise=bool(
                        torch.equal(canon["split"][i][same], canon["ref"][i][same])
                    ),
                    out_canonical_same_rows=int(same.sum()),
                    out_canonical_tie_rows_max_abs=_max_abs(
                        canon["split"][i][differ], canon["ref"][i][differ]
                    ),
                    peak=[int(split.peak[i]), int(ref.peak[i])],
                    ms=[round(split.ms[i], 1), round(ref.ms[i], 1)],
                )
                if full:
                    rs, rr, rr2 = split.recs[i], ref.recs[i], ref2.recs[i]
                    lay.update(
                        # This rank's rows of the gathered buffer are what its own calls wrote.
                        share_is_local=bool(torch.equal(a[begin:end], rs.local[begin:end])),
                        gen_is_local=bool(torch.equal(a[tc:], rs.local[tc:])),
                        scores_share_equal=bool(
                            torch.equal(rs.scores[begin:end], rr.scores[begin:end])
                        ),
                        scores_gen_equal=bool(torch.equal(rs.scores[tc:], rr.scores[tc:])),
                        scores_ref_complete=bool(not torch.isnan(rr.scores[:rows]).any()),
                        control_scores_equal=bool(torch.equal(rr2.scores, rr.scores)),
                    )
                rec["layers"][str(i)] = lay
            if n_step == CONTROL_STEP:
                # The split's chunks with the gather withheld: the target without the gather.
                withheld = dict(split.step, indexer_prefill_q_split=None)
                buf = torch.full([rows, INDEX_TOPK], SENTINEL, dtype=torch.int32, device="cuda")
                env.recorder.arm(buf)
                prefill._indexer(withheld, xs[1], pos, env.layers[1], buf, tc, False, True)
                kept = env.recorder.take()
                rec["control"] = dict(
                    unwritten_rows=int((buf[:tc] == SENTINEL).all(dim=1).sum()),
                    expected=tc - (end - begin),
                    share_scores_equal=bool(
                        torch.equal(kept.scores[begin:end], split.recs[1].scores[begin:end])
                    ),
                )
            result["steps"].append(rec)
            print(f"[rank {rank}] step {n_step} ({name}) done", flush=True)
            del split, ref, ref2, canon, xs, h
    finally:
        manager.shutdown()
    return result


def _rank_main(ckpt_dir: str, out_dir: str) -> None:
    """Body of one rank of tp4's mpirun job. A failing rank exits nonzero without finalizing,
    and mpirun then ends its siblings and fails."""
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    world = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]))
    from mpi4py import MPI

    from tensorrt_llm._utils import mpi_comm

    comm = mpi_comm()
    try:
        assert world == RANKS_PER_NODE == comm.Get_size() and comm.Get_rank() == rank, (
            world,
            comm.Get_size(),
            comm.Get_rank(),
            rank,
        )
        result = _rank_run(rank, world, Path(ckpt_dir))
        (Path(out_dir) / f"rank{rank}.json").write_text(json.dumps(result, indent=1))
    except BaseException:
        print(f"[rank {rank}] FAILED", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(13)
    comm.Barrier()
    print(f"[rank {rank}] finished", flush=True)
    sys.stdout.flush()
    MPI.Finalize()
    os._exit(0)


def _rank_in_place(ckpt_path: Path, world: int) -> tuple[int, list[dict]]:
    """tp8: this pytest process is one rank of the launcher's world, two nodes of four. It runs
    its rank on a device of its node, and every rank's results are gathered on every rank."""
    from mpi4py import MPI

    from tensorrt_llm._utils import mpi_comm, mpi_disabled

    comm = mpi_comm()
    rank = comm.Get_rank()
    # The node's ranks, one device each.
    local = comm.Split_type(MPI.COMM_TYPE_SHARED)
    torch.cuda.set_device(local.Get_rank() % torch.cuda.device_count())
    assert not mpi_disabled(), "the gather runs on the MPI session"
    assert comm.Get_size() == world, (comm.Get_size(), world)
    device = torch.cuda.current_device()
    devices = comm.allgather(str(torch.cuda.get_device_properties(device).uuid))
    assert len(set(devices)) == world, f"every rank must own its own device: {devices}"
    hosts = comm.allgather(socket.gethostname())
    print(
        f"[rank {rank}] on {hosts[rank]} device {device}; ranks over {len(set(hosts))} node(s): "
        f"{hosts}",
        flush=True,
    )
    result = _rank_run(rank, world, ckpt_path)
    # As tp4's ranks hand theirs over: through JSON.
    results = [json.loads(json.dumps(r)) for r in comm.allgather(result)]
    print(f"[rank {rank}] finished", flush=True)
    return rank, results


def _finalize(config, session) -> None:
    """tp8, after pytest's report: MPI finalized (the launcher fails a rank that leaves without
    it), then out at once with pytest's own status, ahead of the interpreter's teardown of the
    ranks' communicators."""
    from mpi4py import MPI

    config.get_terminal_writer().flush()
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        stream.flush()
    MPI.Finalize()
    os._exit(int(session.exitstatus))


def _join_ranks(request, ckpt_path: Path, world: int) -> tuple[int, list[dict]]:
    """tp8: this rank's run among the launcher's ranks. A rank that fails prints its traceback
    and leaves at once (the others wait in a collective for it), so the launcher ends the job
    instead of the deadline."""
    from tensorrt_llm._utils import mpi_rank

    request.config.add_cleanup(lambda: _finalize(request.config, request.session))
    faulthandler.dump_traceback_later(DEADLINE_S, exit=True)
    try:
        return _rank_in_place(ckpt_path, world)
    except BaseException:
        print(f"[rank {mpi_rank()}] FAILED", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(13)
    finally:
        faulthandler.cancel_dump_traceback_later()


# ===================================================================== the launcher


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


def _end_job(job: subprocess.Popen) -> None:
    """End mpirun and every rank it started: mpirun is asked first, then its session's members
    are killed (each rank sits in a process group of its own inside mpirun's session)."""
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


def _devices(world: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [d.strip() for d in visible.split(",") if d.strip()]
    else:
        devices = [str(i) for i in range(torch.cuda.device_count())]
    return ",".join(devices[:world])


def _launch(ckpt: Path, out_dir: Path, world: int) -> str:
    """tp4: the four ranks as one mpirun job on this node; its log."""
    mpirun = shutil.which("mpirun")
    assert mpirun is not None, "the ranks are started by mpirun; none on PATH"
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_")) and k != "TLLM_DISABLE_MPI"
    }
    env["CUDA_VISIBLE_DEVICES"] = _devices(world)
    # tests/unittest/_torch/modeling_v2/models/mistral_large4/<target>/ -> the repo root.
    repo_root = Path(__file__).resolve().parents[7]
    assert (repo_root / "tensorrt_llm").is_dir(), repo_root
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH")) if p)
    command = [mpirun, "-n", str(world), "--oversubscribe", "--allow-run-as-root"]
    command += [sys.executable, str(Path(__file__).resolve()), _RANK_FLAG, str(ckpt), str(out_dir)]
    job = subprocess.Popen(
        command,
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        log, _ = job.communicate(timeout=DEADLINE_S)
    except subprocess.TimeoutExpired:
        _end_job(job)
        raise AssertionError(f"the {world}-rank job did not finish in {DEADLINE_S} s") from None
    if job.returncode != 0:
        _end_job(job)
        raise AssertionError(f"the {world}-rank job failed: mpirun exited {job.returncode}\n{log}")
    return log


def _q_sizes(ctx, tp4_sizes, world: int) -> list[int] | None:
    """A step's expected q_sizes at `world` ranks: `_split_prefill_queries` over its `tc` context
    rows, rank r taking `[tc * r // world, tc * (r + 1) // world)` from Q_SPLIT_THRESHOLD rows on
    and no split below. At four ranks it is the step's listed tp4 expectation."""
    tc = sum(n for _, _, n in ctx)
    if tc < Q_SPLIT_THRESHOLD:
        sizes = None
    else:
        sizes = [tc * (r + 1) // world - tc * r // world for r in range(world)]
    if world == RANKS_PER_NODE:
        assert sizes == tp4_sizes, (ctx, sizes, tp4_sizes)
    return sizes


def _expected_split(sizes, rank: int):
    if sizes is None:
        return None
    begin = sum(sizes[:rank])
    return [begin, begin + sizes[rank], list(sizes)]


def _report(results: list[dict]) -> str:
    lines = [
        f"prefill_logits_budget {results[0]['budget']}, held context workspace rows "
        f"{results[0]['held_rows']}"
    ]
    for n_step, (name, *_rest) in enumerate(STEPS):
        r0 = results[0]["steps"][n_step]
        lines.append(
            f"step {n_step}: {name}: rows {r0['rows']}, context rows {r0['tc']}, context "
            f"workspace rows {r0['workspace_rows']} (num_ctx_kv_tokens {r0['num_ctx_kv_tokens']})"
        )
        for res in results:
            st = res["steps"][n_step]
            tiles = sum(c[3] for c in st["chunks_split"])
            n_k = max((c[2] for c in st["chunks_split"]), default=0)
            tile_mib = max((c[4] for c in st["chunks_split"]), default=0) / 2**20
            lines.append(
                f"  rank {res['rank']}: q_split {st['q_split']}; {len(st['chunks_split'])} "
                f"chunk(s) {[c[:2] for c in st['chunks_split']]}, largest n_k {n_k}, {tiles} "
                f"tile(s), largest tile {tile_mib:.0f} MiB, gathered block "
                f"{st['gather_bytes'] / 2**20:.0f} MiB; unsplit {len(st['chunks_ref'])} "
                f"chunk(s), {sum(c[3] for c in st['chunks_ref'])} tile(s)"
            )
            for i, lay in st["layers"].items():
                head = f"    layer {i}{'' if lay['full'] else ' (shared)'}"
                lines.append(
                    f"{head}: sentinel rows {lay['sentinel_rows']}; rows differing in order "
                    f"{lay['rows_order_differ']} (control {lay['control_rows_order_differ']}), as "
                    f"sets {lay['rows_sets_differ']} (control {lay['control_rows_sets_differ']}); "
                    f"native output max |diff| {lay['out_native_max_abs']} (control "
                    f"{lay['control_out_native_max_abs']}); canonical output bitwise on "
                    f"{lay['out_canonical_same_rows']} same-set rows: "
                    f"{lay['out_canonical_same_rows_bitwise']}, tie rows max |diff| "
                    f"{lay['out_canonical_tie_rows_max_abs']}; peak {lay['peak'][0] / 2**30:.2f} "
                    f"/ {lay['peak'][1] / 2**30:.2f} GiB, {lay['ms'][0]} / {lay['ms'][1]} ms"
                )
                if lay["full"]:
                    lines.append(
                        f"      share is local {lay['share_is_local']}, decode rows local "
                        f"{lay['gen_is_local']}, scores equal: share {lay['scores_share_equal']}, "
                        f"decode rows {lay['scores_gen_equal']}, control "
                        f"{lay['control_scores_equal']}"
                    )
            if "control" in st:
                lines.append(f"    gather withheld (layer 1): {st['control']}")
    return "\n".join(lines)


def _problems(results: list[dict], world: int) -> list[str]:
    problems = []
    assert sorted(res["rank"] for res in results) == list(range(world))
    for n_step, (name, ctx, _gen, tp4_sizes) in enumerate(STEPS):
        sizes = _q_sizes(ctx, tp4_sizes, world)
        tc = sum(n for _, _, n in ctx)
        for res in results:
            rank, st = res["rank"], res["steps"][n_step]
            where = f"step {n_step} ({name}) rank {rank}"
            want = _expected_split(sizes, rank)
            if st["q_split"] != want:
                problems.append(f"{where}: q_split {st['q_split']}, expected {want}")
            begin, end = (0, tc) if want is None else want[:2]
            kv_rows = sum(c + n for _, c, n in ctx)
            if st["workspace_rows"] != kv_rows:
                problems.append(
                    f"{where}: the context workspace holds {st['workspace_rows']} rows, the "
                    f"context requests' cached + new KV is {kv_rows}"
                )
            covered = sorted(c[:2] for c in st["chunks_split"])
            spans_ok = bool(covered) and covered[0][0] == begin and covered[-1][1] == end
            spans_ok = spans_ok and all(a[1] == b[0] for a, b in zip(covered, covered[1:]))
            if not spans_ok:
                problems.append(f"{where}: chunks {covered} do not tile [{begin}, {end})")
            for i, lay in st["layers"].items():
                checks = [any(lay["sentinel_rows"]), not lay["out_canonical_same_rows_bitwise"]]
                if lay["full"]:
                    checks += [
                        not lay["share_is_local"],
                        not lay["gen_is_local"],
                        not lay["scores_share_equal"],
                        not lay["scores_gen_equal"],
                        not lay["scores_ref_complete"],
                        not lay["control_scores_equal"],
                    ]
                if any(checks):
                    problems.append(f"{where} layer {i}: {lay}")
        # Gathered rows are one buffer on every rank; without the split each rank selects every
        # row itself, in the kernel's own order.
        for i in map(str, DRIVEN if sizes is not None else ()):
            digests = {res["steps"][n_step]["layers"][i]["digest_split_ctx"] for res in results}
            if len(digests) != 1:
                problems.append(f"step {n_step} layer {i}: gathered rows differ by rank {digests}")
    # Past 65536 keys the 1 << 28 budget cuts each rank's share of R's third chunk into tiles.
    for res in results:
        chunks = res["steps"][CONTROL_STEP]["chunks_split"]
        if not (len(chunks) == 1 and chunks[0][2] > 65536 and chunks[0][3] > 1):
            problems.append(f"rank {res['rank']}: R chunk 3 is not tiled past 65536: {chunks}")
        control = res["steps"][CONTROL_STEP]["control"]
        if not (control["unwritten_rows"] == control["expected"] > 0):
            problems.append(f"rank {res['rank']}: the withheld gather left {control}")
        if not control["share_scores_equal"]:
            problems.append(f"rank {res['rank']}: the withheld gather's share {control}")
    return problems


def _skip_unless_launched_for(world: int) -> None:
    """tp4 starts its four ranks itself, so it runs unlaunched or inside a launch of four (pytest
    on rank 0), on four GPUs. tp8 runs inside its ranks: this process must be one of eight MPI
    ranks a launcher started, each a pytest process."""
    from tensorrt_llm._utils import mpi_world_size
    from tensorrt_llm.executor.utils import get_spawn_proxy_process_env
    from tensorrt_llm.llmapi.mpi_session import get_mpi_world_size

    launched = get_mpi_world_size()
    if world == RANKS_PER_NODE:
        if launched not in (1, world):
            pytest.skip(f"tp4 starts its {world} ranks itself; this is a launch of {launched}")
        if torch.cuda.device_count() < world:
            pytest.skip(f"needs {world} GPUs, {torch.cuda.device_count()} visible")
        return
    in_place = mpi_world_size()
    if get_spawn_proxy_process_env() or in_place != world:
        pytest.skip(
            f"tp8 runs in its {world} ranks, each a pytest process an MPI launcher started over "
            f"two nodes of {RANKS_PER_NODE} (not under trtllm-llmapi-launch, which runs pytest on "
            f"rank 0 only); this process is one of {in_place} (launched world {launched})"
        )


@pytest.mark.parametrize("world", WORLDS)
def test_q_split_matches_the_unsplit_computation(world, request, tmp_path):
    _skip_unless_launched_for(world)
    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() != 103:
        pytest.skip("the mistral_large4 modeling_v2 targets are certified on sm_103 only")
    ckpt = _checkpoint_dir()
    if ckpt is None:
        pytest.skip(f"checkpoint {CHECKPOINTS} is not under LLM_MODELS_ROOT")

    if world == RANKS_PER_NODE:
        log = _launch(ckpt, tmp_path, world)
        results = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(world)]
        print("\n" + _report(results))
        print("\n--- rank log (tail) ---\n" + "\n".join(log.splitlines()[-40:]))
    else:
        rank, results = _join_ranks(request, ckpt, world)
        if rank == 0:
            print("\n" + _report(results), flush=True)

    assert all(res["budget"] == 1 << 28 and res["held_rows"] == 1 << 19 for res in results)
    problems = _problems(results, world)
    assert not problems, "\n".join(problems)


if __name__ == "__main__" and _RANK_FLAG in sys.argv:
    _at = sys.argv.index(_RANK_FLAG)
    _rank_main(sys.argv[_at + 1], sys.argv[_at + 2])
