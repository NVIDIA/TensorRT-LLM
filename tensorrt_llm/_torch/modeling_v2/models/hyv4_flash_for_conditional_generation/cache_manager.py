# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HY4-Flash's KV cache manager: the engine's DSA manager with the checkpoint's grouped index pool.

The engine resolves ``DSACacheManagerV2`` for HY4-Flash: its ``sparse_attention_config`` is the DSA
config (``algorithm == "dsa"``) and KV cache manager V2 is the default. This subclass keeps every
pool that class derives and changes the one geometry the checkpoint adds to it:

* the MLA latent is ``kv_lora_rank + qk_rope_head_dim`` = 576 wide with one head on all 42 layers,
  replicated across the attention ranks, so every rank holds one 576-wide row per token per layer;
* the 30 windowed layers keep ``sliding_window`` tokens of history and the 12 sparse layers the
  full context. The engine derives both from the checkpoint's ``layer_types``
  (``_derive_v2_layer_type_attention_windows``) and hands them over in
  ``kv_cache_config.max_attention_window``, so this class only has to keep them;
* the indexer K cache holds the checkpoint's *group* keys, not token keys: the compressed indexer
  pools ``index_kpool`` consecutive tokens into one group key and stores that key as
  ``index_head_dim`` quantized bytes followed by its fp32 scale
  (``_get_indexer_k_cache_bytes_per_token``). One page of ``tokens_per_block`` tokens therefore
  holds ``tokens_per_block // index_kpool`` records of ``index_head_dim + index_head_dim // 128 *
  4`` bytes -- 8 records of 132 bytes for this checkpoint's 128-wide fp8 key -- and only the 12
  sparse layers own a pool (``index_topk_pattern`` is 'F' there and 'S' on the windowed layers).

The base sizes ``Role.INDEX_KEY`` as one record per token, which for a grouped pool would allocate
``index_kpool`` times the bytes and address records at the wrong stride. This class sizes the role
per group instead, so the base's allocation, its flat page bookkeeping and its ``Role.ALL``
accounting all follow from :meth:`get_layer_bytes_per_token`.

The pool views the target's attention reads are :meth:`get_indexer_k_cache_buffers` (a page as
``[pages, groups, 1, record]``), :meth:`get_indexer_k_cache_index_plane` (every group key's
quantized data) and :meth:`get_indexer_k_cache_scale_plane` (their fp32 scales); the latent is the
base's ``get_buffers`` / ``get_unique_primary_pool``. The record layout is the checkpoint's, so the
stock DSA indexer, which reads one record per token, must not be pointed at this pool.

This is a V2 manager. The target declares it through ``RuntimeHooks.kv_cache_manager_cls`` and
``get_preferred_kv_cache_manager_version`` -> ``"V2"`` (with ``requires_kv_cache_manager_v2``
True), and the engine then builds it in place of the class it resolved.

The manager also owns the DFly2 draft layers' picks state (:class:`DraftPicks`, through
:meth:`Hyv4FlashCacheManager.draft_picks`): the five target sparse layers' top-``index_topk``
selections as pool rows, which the core writes from the selection it already computes and the
drafter reads as ``DflyBlockIndices``' ``history``. It is not pool memory and no region: it
belongs to no layer, one object serves the whole model, and it is allocated on the first forward
that needs it.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import torch

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import (
    DSACacheManagerV2,
    _get_indexer_k_cache_bytes_per_token,
)
from tensorrt_llm._torch.pyexecutor.config_utils import unwrap_hyv4_flash_text_config
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm.runtime.kv_cache_manager_v2 import DataRole

__all__ = ["DraftPicks", "Hyv4FlashCacheManager"]

#: A picks-state entry that names no pick. `DflyBlockIndices` compacts the
#: valid picks of a row to its front and reads the rest as absent, so the
#: writer leaves every entry it did not fill at this value.
NO_PICK = -1


class DraftPicks:
    """The five draft-source layers' picks: one row per row of a target step.

    DFly2's draft layers attend over their target sparse layer's top-`topk`
    picks of the *anchor's source token* -- the last token the target accepted
    for the request -- and this is where those picks are published. The core
    computes each sparse layer's selection for every row of every step anyway
    (`Hyv4FlashCore._layer_forward` -> `_decode_selection` -> `_pool_rows`);
    the write here is a side effect of that selection and changes none of it.

    What a state row is: a row of a target step, i.e. one token the target
    processed, named by its position among the step's rows -- the packed-row
    axis of the step (`Hyv4FlashCore._row_batch`), *not* the batch row of the
    request that owns the token. A context row and a generation row of the same
    step are different rows; the row numbering is the step's, and the state is
    reused (overwritten row by row) by the next step. The writer fills:

    * every generation row of the step, because the drafter's anchor may be any
      of the rows a verify step accepted, and which one is known only after the
      step's logits are sampled;
    * each context request's last row, the prompt tail a request's first draft
      block is built from.

    The reader (`Hyv4FlashDraft`, target layers 25/29/33/37/41 at checkpoint
    ids) calls:

        picks = md.kv_cache_manager.draft_picks(
            layers, topk, rows=<the engine's per-step token budget>, device=...)
        DflyBlockIndices(history=picks.history,
                         state_rows=<row of the anchor's source token>,
                         block_slots=..., active=..., packed=..., lengths=...)

    The writer allocates the state on the first step that needs it, so a reader
    that arrives later reaches the same buffer with `rows` and `device` left
    out: `draft_picks(layers, topk)` returns what was allocated, never a second
    state of its own.

    `history` is int32 `[len(layers), rows, topk]`, dense, `-1` (`NO_PICK`)
    where a row holds no pick; `history[k]` is `layers[k]`'s picks as rows of
    the primary latent pool, the form `SparseMlaDecode` and
    `DflyBlockIndices.block_slots` both index. `state_rows` names one row of
    the *step the target last processed*: for a generation request whose verify
    step carried a block, that is the row its last accepted token occupied
    (`DflyBlockPrepare`'s `source_rows`, the target's packed row, not the batch
    row); for a request whose last target step was its prefill, it is the row
    of the prompt's last token, which is the row the writer filled for it. A
    reader must pass `state_rows` of that step, never a row of another step:
    rows the writer did not fill still hold the previous step's picks.
    """

    def __init__(self, layers: Sequence[int], rows: int, topk: int, device) -> None:
        self.layers = tuple(int(layer) for layer in layers)
        self.topk = int(topk)
        if not self.layers:
            raise ValueError("a picks state needs at least one draft-source layer")
        if int(rows) <= 0 or self.topk <= 0:
            raise ValueError(f"a picks state is 2D at least; got rows={rows}, topk={topk}")
        #: `[layers, rows, topk]` int32: each draft layer's picks, `NO_PICK`
        #: where the row holds none. Dense, and the last dimension at stride 1:
        #: `DflyBlockIndices` reads each row's `topk` columns at stride 1.
        self.history = torch.full(
            (len(self.layers), int(rows), self.topk), NO_PICK, dtype=torch.int32, device=device
        )
        self._slots = {layer: k for k, layer in enumerate(self.layers)}

    def slot(self, layer: int) -> int | None:
        """`history`'s first-axis slot of decoder layer `layer`, or None if it is no draft layer."""
        return self._slots.get(int(layer))

    def write(self, layer: int, packed: torch.Tensor, rows: torch.Tensor) -> None:
        """Copy `packed`'s `rows` into `layer`'s slot; a layer outside `layers` writes nothing.

        `packed` is `[T, topk]` (the step's rows, `NO_PICK` padded) and `rows`
        is `[N]` int64 with the rows to keep; both are the writer's, so the
        state never re-reads a selection the layer did not make.
        """
        slot = self._slots.get(int(layer))
        if slot is None:
            return
        self.history[slot].index_copy_(0, rows, packed.index_select(0, rows))


class Hyv4FlashCacheManager(DSACacheManagerV2):
    """The engine's DSA cache manager, sizing the indexer pool in the checkpoint's group keys."""

    @override
    def __init__(
        self,
        *args,
        pretrained_config=None,
        model_config=None,
        **kwargs,
    ) -> None:
        """Build the manager with the checkpoint's indexer group width.

        ``index_kpool`` is read from the checkpoint (``index_kpool``, published from
        ``compressed_indexer_compress_ratio``) before the base resolves the sparse parameters,
        because the base sizes and builds the indexer pools from it.
        """
        config = pretrained_config
        if config is None:
            config = getattr(model_config, "pretrained_config", None)
        index_kpool = getattr(unwrap_hyv4_flash_text_config(config), "index_kpool", None)
        if index_kpool is None:
            raise ValueError(
                "Hyv4FlashCacheManager needs the checkpoint's index_kpool, the compressed "
                "indexer's tokens per group key; the pretrained config publishes none."
            )
        self.index_kpool = int(index_kpool)
        if self.index_kpool <= 0:
            raise ValueError(f"HY4-Flash's index_kpool must be positive, got {self.index_kpool}.")
        tokens_per_block = kwargs.get("tokens_per_block")
        if tokens_per_block is not None and int(tokens_per_block) % self.index_kpool:
            raise ValueError(
                f"HY4-Flash's compressed indexer pools {self.index_kpool} tokens per group key, "
                f"so a page must hold whole groups; got tokens_per_block={tokens_per_block}."
            )
        super().__init__(
            *args,
            pretrained_config=pretrained_config,
            model_config=model_config,
            **kwargs,
        )
        self._draft_picks: DraftPicks | None = None

    def draft_picks(
        self,
        layers: Sequence[int],
        topk: int,
        rows: int | None = None,
        device=None,
    ) -> DraftPicks:
        """The draft-source layers' picks state, allocated on the first call and kept.

        The manager owns it because both ends reach the manager and neither owns
        the other: the target's core writes it from its forward
        (`md.kv_cache_manager`, the same object a draft model that shares this
        cache -- `should_use_separate_draft_kv_cache() -> False` -- reads it
        from). `DraftPicks` documents what a row is and which rows a step
        fills.

        `layers` is the checkpoint's `dfly_target_kv_layers`, `topk` its
        `dfly_target_index_topk` (`index_topk`: the width of the selection the
        indexer computes) and `rows` the engine's per-step token budget, the
        most rows a step can carry. Later calls with the same geometry return
        the state that was built; a different geometry is refused rather than
        silently served by a buffer of the wrong shape.
        """
        wanted = tuple(int(layer) for layer in layers)
        state = self._draft_picks
        if state is None:
            if rows is None:
                rows = self.max_num_tokens
            if rows is None:
                raise ValueError(
                    "this manager knows no per-step token budget, so the picks state's row "
                    "count must be passed: draft_picks(layers, topk, rows=...)"
                )
            if device is None:
                raise ValueError(
                    "the picks state is device memory; pass the device of the step's pools "
                    "(device=...)"
                )
            state = DraftPicks(wanted, int(rows), int(topk), device)
            self._draft_picks = state
        elif wanted != state.layers or int(topk) != state.topk:
            raise ValueError(
                f"this manager's picks state already holds layers {list(state.layers)} at "
                f"topk={state.topk}; asked for {list(wanted)} at topk={topk}"
            )
        return state

    @property
    def indexer_record_bytes(self) -> int:
        """Bytes of one group key: its quantized data followed by its fp32 scale."""
        return _get_indexer_k_cache_bytes_per_token(
            self.index_head_dim, self.quant_block_size, self.use_fp4
        )

    @property
    def indexer_data_bytes(self) -> int:
        """Bytes of one group key's quantized data, at the front of its record."""
        return self.index_head_dim // 2 if self.use_fp4 else self.index_head_dim

    @property
    def indexer_scale_bytes(self) -> int:
        """Bytes of one group key's scales, after its data in the record."""
        return self.indexer_record_bytes - self.indexer_data_bytes

    @property
    def indexer_bytes_per_token(self) -> int:
        """One token's share of the grouped pool, for the base's per-token ledger."""
        record = self.indexer_record_bytes
        if record % self.index_kpool:
            raise ValueError(
                f"HY4-Flash's indexer record ({record} bytes) does not divide into "
                f"index_kpool={self.index_kpool} whole per-token bytes."
            )
        return record // self.index_kpool

    def _indexer_groups_per_block(self, tokens_per_block: int) -> int:
        """Group-key records one page of ``tokens_per_block`` tokens holds."""
        if tokens_per_block % self.index_kpool:
            raise ValueError(
                f"HY4-Flash's compressed indexer pools {self.index_kpool} tokens per group key, "
                f"so a page must hold whole groups; got tokens_per_block={tokens_per_block}."
            )
        return tokens_per_block // self.index_kpool

    @override
    def get_layer_bytes_per_token(self, local_layer_idx: int, data_role: DataRole) -> int:
        """Size the indexer role per group key; every other role is the base's."""
        if data_role == Role.INDEX_KEY:
            return self.indexer_bytes_per_token
        return super().get_layer_bytes_per_token(local_layer_idx, data_role)

    @override
    def get_indexer_k_cache_buffers(self, layer_idx: int) -> torch.Tensor:
        """Return one indexer K-cache page as ``[pages, groups, 1, record]`` uint8.

        ``groups`` is ``tokens_per_block // index_kpool`` and ``record`` is
        :attr:`indexer_record_bytes`; each record is one group key's quantized data followed by
        its fp32 scale. The base's view of the same pool is one record per token, which this
        checkpoint's grouped pool does not have.
        """
        layer_offset = self.layer_offsets[layer_idx]
        pool = self.indexer_k_cache_pool_per_layer[layer_offset]
        assert pool is not None, (
            f"Layer {layer_idx} runs windowed MLA and owns no indexer k-cache; only the "
            f"checkpoint's sparse layers may access it."
        )
        return pool.view(
            pool.shape[0],
            self._indexer_groups_per_block(self.tokens_per_block),
            1,
            self.indexer_record_bytes,
        )

    def get_indexer_k_cache_index_plane(self, layer_idx: int) -> torch.Tensor:
        """Return every group key's quantized data as ``[pages, groups, 1, index_head_dim]``.

        A strided view of :meth:`get_indexer_k_cache_buffers`: group ``g`` of a page starts at
        byte ``g * indexer_record_bytes`` and its data runs for :attr:`indexer_data_bytes`, so the
        plane is not contiguous across groups and must be indexed by group, not by token.
        """
        records = self.get_indexer_k_cache_buffers(layer_idx)
        pages, groups = records.shape[0], records.shape[1]
        return torch.as_strided(
            records,
            size=(pages, groups, 1, self.indexer_data_bytes),
            stride=(
                groups * self.indexer_record_bytes,
                self.indexer_record_bytes,
                self.indexer_data_bytes,
                1,
            ),
        )

    def get_indexer_k_cache_scale_plane(self, layer_idx: int) -> torch.Tensor:
        """Return every group key's fp32 scales as ``[pages, groups, 1, scales per record]``.

        A strided view of the tail of :meth:`get_indexer_k_cache_buffers`, at the byte offset
        :attr:`indexer_data_bytes` inside each record.
        """
        records = self.get_indexer_k_cache_buffers(layer_idx)
        return records[..., self.indexer_data_bytes :].view(torch.float32)
