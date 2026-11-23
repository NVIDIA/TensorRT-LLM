# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HY4-Flash's KV cache manager: the engine's DSA V2 manager with the checkpoint's grouped index pool.

The compressed indexer pools ``index_kpool`` tokens into one group key, stored as ``index_head_dim``
quantized bytes followed by its fp32 scale, so a page holds ``tokens_per_block // index_kpool``
records. The base sizes ``Role.INDEX_KEY`` one record per token; this class sizes it per group.

The manager also owns the DFly2 draft layers' picks state (:class:`DraftPicks`).
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

#: A picks entry that names no pick; `DflyBlockIndices` reads it as absent.
NO_PICK = -1


class DraftPicks:
    """The draft-source layers' top-`topk` picks, one row per packed row of a target step.

    The target core writes every generation row and each context request's last row of a step;
    the drafter reads `history` through `DflyBlockIndices` with `state_rows` of that same step
    (rows the writer did not fill still hold an earlier step's picks). `history` is int32
    `[len(layers), rows, topk]`, dense, `NO_PICK` where a row holds no pick, the picks as rows of
    the primary latent pool.
    """

    def __init__(self, layers: Sequence[int], rows: int, topk: int, device) -> None:
        self.layers = tuple(int(layer) for layer in layers)
        self.topk = int(topk)
        if not self.layers:
            raise ValueError("a picks state needs at least one draft-source layer")
        if int(rows) <= 0 or self.topk <= 0:
            raise ValueError(f"a picks state is 2D at least; got rows={rows}, topk={topk}")
        self.history = torch.full(
            (len(self.layers), int(rows), self.topk), NO_PICK, dtype=torch.int32, device=device
        )
        self._slots = {layer: k for k, layer in enumerate(self.layers)}

    def slot(self, layer: int) -> int | None:
        """`history`'s first-axis slot of decoder layer `layer`, or None if it is no draft layer."""
        return self._slots.get(int(layer))

    def write(self, layer: int, packed: torch.Tensor, rows: torch.Tensor) -> None:
        """Copy `packed[rows]` (`[T, topk]`, `[N]` int64) into `layer`'s slot; no-op off `layers`."""
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
        config = pretrained_config
        if config is None:
            config = getattr(model_config, "pretrained_config", None)
        index_kpool = getattr(unwrap_hyv4_flash_text_config(config), "index_kpool", None)
        if index_kpool is None:
            raise ValueError("Hyv4FlashCacheManager needs the checkpoint's index_kpool.")
        # Set before the base builds the indexer pools from it.
        self.index_kpool = int(index_kpool)
        if self.index_kpool <= 0:
            raise ValueError(f"HY4-Flash's index_kpool must be positive, got {self.index_kpool}.")
        tokens_per_block = kwargs.get("tokens_per_block")
        if tokens_per_block is not None:
            self._indexer_groups_per_block(int(tokens_per_block))
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
        """The picks state, allocated on the first call; later calls must ask the same geometry."""
        wanted = tuple(int(layer) for layer in layers)
        state = self._draft_picks
        if state is None:
            if rows is None:
                rows = self.max_num_tokens
            if rows is None:
                raise ValueError(
                    "this manager knows no per-step token budget; pass draft_picks(..., rows=...)"
                )
            if device is None:
                raise ValueError(
                    "the picks state is device memory; pass draft_picks(..., device=...)"
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
        if tokens_per_block % self.index_kpool:
            raise ValueError(
                f"HY4-Flash's compressed indexer pools {self.index_kpool} tokens per group key, "
                f"so a page must hold whole groups; got tokens_per_block={tokens_per_block}."
            )
        return tokens_per_block // self.index_kpool

    @override
    def get_layer_bytes_per_token(self, local_layer_idx: int, data_role: DataRole) -> int:
        if data_role == Role.INDEX_KEY:
            return self.indexer_bytes_per_token
        return super().get_layer_bytes_per_token(local_layer_idx, data_role)

    @override
    def get_indexer_k_cache_buffers(self, layer_idx: int) -> torch.Tensor:
        """Return one indexer K-cache page as ``[pages, groups, 1, record]`` uint8."""
        layer_offset = self.layer_offsets[layer_idx]
        pool = self.indexer_k_cache_pool_per_layer[layer_offset]
        assert pool is not None, f"Layer {layer_idx} runs windowed MLA and owns no indexer k-cache"
        return pool.view(
            pool.shape[0],
            self._indexer_groups_per_block(self.tokens_per_block),
            1,
            self.indexer_record_bytes,
        )
