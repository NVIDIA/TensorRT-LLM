# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Key-sharded DSA decode Top-K of the attention-TP perf target: every TP rank scores 1/TP of the index keys.

Without attention DP every TP rank holds the whole index-K cache, and the replicated decode indexer has every
rank score every key of every sequence. On a key-sharded step rank ``r`` of ``W`` scores only the 64-token cache
blocks ``b % W == r`` (a compacted block table), takes each row's exact local Top-``L``, and the ranks exchange
and merge them into the global Top-K over NVLink: exact, with a certificate and a re-selection from every rank's
local scores for the rare row a local Top-``L`` cannot certify. The selected set is the global Top-K of the full
row -- ties at the K-th score go to the smaller token index, the deterministic GVR's rule -- and the per-key
logits are the same kernel's, so after the order + pool remap the rows are bit for bit the replicated path's.

Two exchanges, by layout (``KeyShardLayout.pruned``):

* TP 16 / 32: the pruned exchange (``kernels/attention/dsa_key_shard_prune``, three CuTe DSL launches): the ranks
  share ladders of their sorted local keys, derive a common lower bound of the K-th key and send only their keys
  above it (about 2.2-2.3k keys per row instead of ``W * L``), then each rank selects the K largest received keys.
* TP 8: every rank sends all ``L`` local keys and each rank merges ``W * L`` (``kernels/attention/dsa_key_shard``,
  Triton); at 8 ranks the pruned exchange's extra round trip costs what its smaller transfer saves.

``KeyShardDecode`` holds a metadata's per-step views (compacted block table, local context and row lengths, the
logits schedule over the local lengths) and the TP group's merge workspace; ``prepare`` builds the views for a
generation step and ``topk`` runs a layer's decode Top-K.

The split is on. Its limits are per TP size (``LAYOUTS``). A
step takes the split on uniform verify rows, at least ``min_seqs`` sequences and
at most ``max_rows`` rows; smaller batches keep the replicated path, which is faster there in the model (the
exchange and merge add a near-constant ~10-20 us per layer, and some rows need the merge's exact re-selection).
The decision is a host function of the batch shape, the same on every TP rank, so each CUDA graph captures one
path.

Kernels: the local Top-``L`` is the CuTe DSL radix decode Top-K. Exchange + merge, GB300, 5 verify rows, 120k-token
contexts, us per layer (pruned vs all-keys): TP16 55 / 80 / 100 rows 17.4 / 19.8 / 20.8 vs 25.5 / 28.7 / 30.9; TP32
40 / 55 / 80 rows 19.7 / 20.4 / 21.3 vs 23.9 / 26.2 / 34.0; TP8 35 / 80 rows 15.1 / 17.5 vs 14.9 / 18.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.attention.dsa_key_shard import (
    KEY_SHARD_TOKENS_PER_BLOCK,
    KeyShardTopKMerge,
    get_key_shard_merge,
    key_shard_local_count,
    key_shard_prepare_rows,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.attention.dsa_key_shard_prune import (  # noqa: E501
    KeyShardPrunedExchange,
    get_key_shard_pruned_exchange,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.attention.dsa_topk_order_remap import (  # noqa: E501
    dsa_topk_order_remap_into,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.cute_dsl_fp8_paged_mqa_logits import (  # noqa: E501
    cute_dsl_fp8_paged_mqa_logits_pdl,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.paged_mqa_schedule import (
    paged_mqa_schedule,
)
from tensorrt_llm.logger import logger


@dataclass(frozen=True)
class KeyShardLayout:
    """Per TP size: ``local_k`` keys of each rank's local Top-L (bounds the exchange; any value selects the same
    rows), the fewest decode sequences of a split step, its most decode rows, and whether it takes the pruned
    exchange (else every rank sends all ``local_k`` keys)."""

    local_k: int
    min_seqs: int
    max_rows: int
    pruned: bool = False


#: Most decode rows a split step can serve (the workspace keeps every row's local scores).
MAX_ROWS = 288
#: The layouts the split serves (attention TP 8 / 16 / 32, no attention DP), with the limits the deployed TEP
#: configurations run. ``local_k`` bounds the exchange (any value selects the same rows; a larger one needs the exact
#: re-selection on fewer rows). ``min_seqs`` is 11 on every layout. In the model the split already shortens decode
#: steps from fewer sequences (GB300, 5 verify rows, ms per step at 120k / 200k-token contexts: TEP8 from 7, 7 / 8 /
#: 10 sequences ~0 / -0.45, ~0 / -0.54, ~0 / -0.69; TEP16 from 8, 7 sequences +0.12 / -0.12, 8 sequences -0.01 /
#: -0.24, 10 sequences -0.12 / -0.59; TEP32 from 11). ``max_rows`` is ``MAX_ROWS`` on every layout, which covers
#: DFlash2 w6 / w8 at large batches; every measured deployment's decode row count, max batch x (draft + 1) = 32 x 5 at
#: TP8, 48 x 5 at TP16, 32 x 8 at TP32 (160 / 240 / 256), is below it, so it splits the same steps.
LAYOUTS = {
    8: KeyShardLayout(local_k=512, min_seqs=11, max_rows=MAX_ROWS),
    16: KeyShardLayout(local_k=512, min_seqs=11, max_rows=MAX_ROWS, pruned=True),
    32: KeyShardLayout(local_k=256, min_seqs=11, max_rows=MAX_ROWS, pruned=True),
}


def key_shard_layout(tp_size: int) -> Optional[KeyShardLayout]:
    """The split's layout at ``tp_size`` (``LAYOUTS``), or None for a TP size it does not serve."""
    return LAYOUTS.get(tp_size)


class KeyShardDecode:
    """One metadata's key-shard state: per-step views and the TP group's merge workspace.

    ``next_n`` is the step's verify width when the step is split, else 0 (``prepare``).
    """

    def __init__(self, metadata, mapping, layout: KeyShardLayout):
        world = mapping.tp_size
        self.world = world
        self.rank = mapping.tp_rank
        self.layout = layout
        self.top_k = metadata.num_sparse_topk
        max_seqs = metadata.max_num_sequences
        local_blocks = -(-metadata.kv_cache_manager.max_blocks_per_seq // world)
        local_blocks = -(-local_blocks // 4) * 4  # whole 256-key splits of the logits kernel
        self.max_local_len = local_blocks * KEY_SHARD_TOKENS_PER_BLOCK
        capture = metadata.is_cuda_graph
        buffers = metadata.cuda_graph_buffers

        def empty(shape, name):
            return metadata.get_empty(
                buffers, shape, cache_name=name, dtype=torch.int32, capture_graph=capture
            )

        self.block_table = empty([max_seqs, local_blocks], "k31_key_shard_block_table")
        self.ctx_lens = empty((max_seqs,), "k31_key_shard_ctx_lens")
        self.row_lens = empty((MAX_ROWS,), "k31_key_shard_row_lens")
        self.sched = empty((metadata.num_sms + 1, 2), "k31_key_shard_sched")
        self.local_topk = empty((MAX_ROWS, layout.local_k), "k31_key_shard_local_topk")
        # The pruned exchange reads each sequence's key count on the device.
        self.kv_lens = empty((max_seqs,), "k31_key_shard_kv_lens") if layout.pruned else None
        if layout.pruned:
            self.merge = get_key_shard_pruned_exchange(
                mapping, MAX_ROWS, self.top_k, self.max_local_len, layout.local_k
            )
        else:
            self.merge = get_key_shard_merge(
                mapping, MAX_ROWS, self.top_k, self.max_local_len, layout.local_k
            )
        self.num_sms = metadata.num_sms
        self.next_n = 0
        logger.info_once(
            f"Kimi K3.1 DSA indexer: key-sharded decode Top-K on (tp {world}, local top-{layout.local_k}, "
            f">= {layout.min_seqs} sequences, <= {layout.max_rows} rows, "
            f"{'pruned' if layout.pruned else 'all-keys'} exchange)",
            key="k31_dsa_key_shard_on",
        )

    @staticmethod
    def create(metadata) -> Optional["KeyShardDecode"]:
        """The state for a perf-target metadata whose layout the split serves when it is on, else None."""
        mapping = getattr(metadata, "mapping", None)
        layout = key_shard_layout(mapping.tp_size) if mapping is not None else None
        if (
            mapping is None
            or layout is None
            or getattr(metadata, "is_cross", False)
            or mapping.enable_attention_dp
            or mapping.cp_size != 1
            or mapping.tp_size not in LAYOUTS
            or metadata.kv_cache_manager is None
            or getattr(metadata, "_indexer_compress_ratio", 1) != 1
            or getattr(metadata, "_tokens_per_block", None) != KEY_SHARD_TOKENS_PER_BLOCK
        ):
            return None
        return KeyShardDecode(metadata, mapping, layout)

    def prepare(self, metadata, gen_indexer_kv_lens: Optional[torch.Tensor]) -> None:
        """The per-step local views for the generation rows (``gen_indexer_kv_lens``, None without any;
        graph-capturable); ``next_n`` = 0 when the step keeps the replicated path. Mixed context + generation steps
        keep it too."""
        self.next_n = 0
        num_gen = metadata.num_generations
        num_gen_tokens = metadata.num_tokens - metadata.num_ctx_tokens
        if (
            gen_indexer_kv_lens is None
            or num_gen == 0
            or metadata.num_contexts != 0
            or getattr(metadata, "is_ragged_verify", False)
            or num_gen_tokens % num_gen
            or num_gen < self.layout.min_seqs
            or num_gen_tokens > self.layout.max_rows
        ):
            return
        next_n = num_gen_tokens // num_gen
        key_shard_prepare_rows(
            gen_indexer_kv_lens,
            metadata.indexer_k_cache_block_offsets[metadata.num_contexts : metadata.num_seqs],
            self.block_table,
            self.ctx_lens,
            self.row_lens,
            self.rank,
            self.world,
            next_n,
            self.kv_lens,
        )
        paged_mqa_schedule(self.ctx_lens[:num_gen].view(-1, 1), 64, self.num_sms, self.sched)
        self.next_n = next_n

    def topk(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        weights: torch.Tensor,
        rows_out: torch.Tensor,
        num_generations: int,
        next_n: int,
        pool_remap: Optional[tuple],
    ) -> bool:
        """A layer's decode Top-K on a split step into ``rows_out`` [rows, K]: KV-pool rows (ordered) with
        ``pool_remap`` ``(block_table, block_size, stride_factor, layer_id)``, else the token indices (unordered).
        Returns False (nothing written) when this step keeps the replicated path."""
        if (
            self.next_n == 0
            or next_n != self.next_n
            or rows_out.shape[1] != self.top_k
            # FP8 index keys: 128 bytes + an fp32 scale.
            or k_cache.shape[-1] != q.shape[-1] + 4
        ):
            return False
        k31_key_shard_topk(
            q.view(num_generations, next_n, *q.shape[-2:]),
            k_cache,
            weights,
            self.ctx_lens[:num_generations],
            self.block_table[:num_generations],
            self.sched,
            self.row_lens[: num_generations * next_n],
            self.local_topk[: num_generations * next_n],
            self.max_local_len,
            self.merge,
            rows_out,
            None if self.kv_lens is None else self.kv_lens[:num_generations],
        )
        if pool_remap is not None:
            dsa_topk_order_remap_into(rows_out, rows_out, None, *pool_remap, rows_per_req=next_n)
        return True


def _spec(batch, next_n, context_lens, world, **overrides) -> dict:
    spec = dict(
        batch=batch,
        next_n=next_n,
        context_lens=context_lens,
        world=world,
        local_k=LAYOUTS.get(world, LAYOUTS[16]).local_k,
        pruned=LAYOUTS.get(world, LAYOUTS[16]).pruned,
        topk=2048,
        data="real",
    )
    spec.update(overrides)
    return spec


class K31KeyShardTopk(OpWrapper):
    """One TP rank's key-sharded decode Top-K of the ``B * n`` verify rows (``n`` = ``q.shape[1]``).

    Scores the rank's index keys (``block_table`` = its compacted blocks, ``context_lens`` its local key counts,
    ``schedule_meta`` the logits schedule over them) into ``merge.local_logits``, takes each row's exact local
    Top-``merge.local_k`` over its ``row_lens`` local keys into ``local_topk`` and exchanges them with every rank:
    ``out`` [rows, K] gets row ``r``'s global Top-K token indices (largest score; ties at the K-th score to the
    smaller token index; all ``n_r`` tokens and then ``-1`` when ``n_r <= K``), in rank-major order from a
    ``KeyShardTopKMerge`` (every rank sends all its keys) and in no particular order from a
    ``KeyShardPrunedExchange``, which also needs ``kv_lens`` [B] (each sequence's key count). A collective: every
    rank of ``merge``'s group makes the same call.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="M2 TEP16: 12 sequences x 5 verify rows at 130-170k",
            spec=_spec(12, 5, (130_000, 170_000), 16),
        ),
        Cell(why="M2 TEP16: 24 sequences x 5 rows", spec=_spec(24, 5, (130_000, 170_000), 16)),
        Cell(
            why="TEP32: 6 sequences x 8 rows at 150-190k", spec=_spec(6, 8, (150_000, 190_000), 32)
        ),
        Cell(why="TEP8: 8 sequences x 5 rows", spec=_spec(8, 5, (100_000, 140_000), 8)),
        Cell(
            why="short sequences mixed in: rows with n_r <= K take every token",
            spec=_spec(12, 5, (500, 40_000), 16),
        ),
        Cell(
            why="threshold ties: integer-valued scores, the smaller token indices win the ties",
            spec=_spec(12, 5, (60_000, 80_000), 16, data="ties"),
        ),
        Cell(
            why="one rank holds most of the Top-K: the certificate fails and the row is re-selected exactly",
            spec=_spec(12, 5, (100_000, 120_000), 16, data="skewed"),
        ),
        Cell(
            why="c48 TEP16: 20 sequences x 5 rows at 110-130k (pruned exchange)",
            spec=_spec(20, 5, (110_000, 130_000), 16),
        ),
        Cell(
            why="TEP32: 16 sequences x 5 rows at 110-130k (pruned exchange)",
            spec=_spec(16, 5, (110_000, 130_000), 32),
        ),
        Cell(
            why="every score +-0 (zero queries): all keys tie, the smallest token indices win",
            spec=_spec(12, 5, (60_000, 80_000), 16, data="zeros"),
        ),
        Cell(
            why="every row at most K keys: [0, n) and -1 without an exchange",
            spec=_spec(12, 5, (500, 2_000), 16),
        ),
        Cell(
            why="TEP16 with the all-keys exchange (the TP8 path)",
            spec=_spec(12, 5, (100_000, 140_000), 16, pruned=False),
        ),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300; the cells emulate the ranks on one GPU with plain buffers,
    ``pruned`` picks the exchange; the pruned exchange also ran on 4 to 32 real ranks over MNNVL) and
    sm_107 (VR200, the emulated cells).
    The attention-TP perf target calls it (``KeyShardDecode.topk``) in the DSA indexer's decode branch in
    place of the replicated logits + GVR Top-K on split steps, then orders and remaps the rows
    (``dsa_topk_order_remap``): bitwise the replicated path's rows, since the per-key logits come from the
    same kernel and the exchange selects the same set (exact). Every rank must make the same sequence of calls
    (call counters on the device pick the receive slots); the local scores live in the symmetric workspace and
    are overwritten by the next call, which the TP all-reduces between DSA layers order after every peer's merge.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        row_lens: torch.Tensor,
        local_topk: torch.Tensor,
        max_local_len: int,
        merge: KeyShardTopKMerge | KeyShardPrunedExchange,
        out: torch.Tensor,
        kv_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        rows = row_lens.shape[0]
        logits = merge.local_logits[:rows, :max_local_len]
        cute_dsl_fp8_paged_mqa_logits_pdl(
            q,
            k_cache,
            weights,
            context_lens,
            block_table,
            schedule_meta,
            max_local_len,
            1,
            None,
            logits,
        )
        torch.ops.trtllm.cute_dsl_indexer_topk_decode(
            logits, row_lens, local_topk, merge.local_k, 1
        )
        if isinstance(merge, KeyShardPrunedExchange):
            merge.exchange(local_topk, kv_lens, q.shape[1], out)
        else:
            merge.pack(local_topk, row_lens)
            merge.merge(out)
        return out

    def reference(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        row_lens: torch.Tensor,
        local_topk: torch.Tensor,
        max_local_len: int,
        merge: KeyShardTopKMerge | KeyShardPrunedExchange,
        out: torch.Tensor,
        kv_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The collective exception: ``q`` .. ``schedule_meta`` / ``block_table`` are the GROUP's full-row inputs
        (``block_table`` the full table, ``context_lens`` the full key counts); ``row_lens``, ``local_topk``,
        ``max_local_len``, ``merge`` and ``kv_lens`` are ignored. Each row's stable descending sort of its causal
        prefix, first K token indices, ``-1`` padded (sorted ascending, as the order + remap step would see the set)."""
        batch, next_n, heads, dim = q.shape
        k = out.shape[1]
        page = k_cache.shape[1]
        rows = batch * next_n
        out.fill_(-1)
        kv = k_cache.reshape(k_cache.shape[0], -1)
        for b in range(batch):
            n = int(context_lens[b])
            pages = block_table[b, : -(-n // page)].long()
            blob = kv[pages]
            keys = blob[:, : page * dim].reshape(-1, dim)[:n].view(torch.float8_e4m3fn).float()
            scales = blob[:, page * dim :].contiguous().view(torch.float32).reshape(-1)[:n]
            for t in range(next_n):
                r = b * next_n + t
                qt = q[b, t].float()
                s = (torch.relu(qt @ keys.T) * weights[r].float()[:, None]).sum(0) * scales
                valid = n - next_n + t + 1
                if valid <= k:
                    out[r, :valid] = torch.arange(valid, dtype=torch.int32, device=out.device)
                    continue
                order = torch.sort(s[:valid], descending=True, stable=True).indices[:k]
                out[r] = order.sort().values.int()
        return out[:rows]

    def is_valid(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        row_lens: torch.Tensor,
        local_topk: torch.Tensor,
        max_local_len: int,
        merge: KeyShardTopKMerge | KeyShardPrunedExchange,
        out: torch.Tensor,
        kv_lens: Optional[torch.Tensor] = None,
    ) -> None:
        rows = q.shape[0] * q.shape[1]
        assert row_lens.shape[0] == rows and out.shape[0] == rows and local_topk.shape[0] == rows
        assert local_topk.shape[1] == merge.local_k and local_topk.is_contiguous()
        assert out.shape[1] == merge.top_k and out.stride(1) == 1
        assert rows <= merge.rows_max and max_local_len <= merge.local_width
        assert k_cache.shape[1] == KEY_SHARD_TOKENS_PER_BLOCK, "64-token index-key pages"
        if isinstance(merge, KeyShardPrunedExchange):
            assert kv_lens is not None and kv_lens.dtype == torch.int32
            assert kv_lens.shape[0] == q.shape[0], "one key count per sequence"


k31_key_shard_topk = K31KeyShardTopk()

__all__ = [
    "KeyShardDecode",
    "KeyShardLayout",
    "K31KeyShardTopk",
    "KeyShardPrunedExchange",
    "LAYOUTS",
    "MAX_ROWS",
    "k31_key_shard_topk",
    "key_shard_layout",
    "key_shard_local_count",
]
