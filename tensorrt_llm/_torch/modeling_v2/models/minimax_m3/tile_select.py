# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The prefill top-16 block select as the tile select (#20629 TILE_SELECT) on the sm_107 context targets.

``minimax_m31_select_prefill_tile`` reads each 32-byte sector of the head-major proxy scores once for 8 queries of a KV
head; the built-in ``minimax_m3_select_blocks`` reads one score per sector per pass. Its table equals the built-in one
bit for bit on sm_107. ``TileSelect.select`` is registered on every layer's MSA indexer ``selector_hook``; steps
outside its rows / blocks range, graph captures and builds without the op keep the built-in select.

Row-major select (exact, ``rows_enabled``): the exact Q8KV8 prefill scorer's scores (its [T * H, pitch] buffer as a
zero-copy [H, pages, T] view with unit block stride, the target's idx_scorer) take
``minimax_m31_select_prefill_tile_rows``, the same kernel with a row-major staging: the shared memory its unchanged
select phase reads is the same, so the table is the tile op's on the same values bit for bit. ``select_rows`` is the
indexer's ``prefill_scorer_selector`` (the target's idx_scorer registers it), which only the scorer's steps reach:
``select`` (every other select, the NVFP4 decode reader's row-major scores on mixed / eager decode rows included) still
takes head-major scores only, so those rows keep their other select.

Each target owns one ``TileSelect`` built from its own kill switches (``TileSelectSettings``): ``<switch>`` turns the
tile select off, ``<switch>_ROWS`` the row-major select, ``TSEL_NARROW`` the op's narrow-row kernel.
"""

from dataclasses import dataclass

import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    MSA_REQUIRED_TOPK,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm.logger import logger

TSEL_MIN_ROWS = 256  # #20629's bound: below it the built-in one-CTA-per-row grid serves the select
TSEL_MAX_BLOCKS = 5856  # the op's shared-memory bound (kMinimaxM3SelectPrefillTileMaxBlocks)


@dataclass(frozen=True)
class TileSelectSettings:
    """One target's tile-select settings, read from its own kill switches."""

    target: str  # the target's log tag, e.g. "tp2ctx"
    switch: str  # the tile select's switch id, e.g. "TP2CTX_TSEL" (log text)
    enabled: bool  # the tile select is on (not <switch>)
    rows_enabled: bool  # the row-major select is on (False for a target without it)
    narrow: bool  # the op runs tiles of few blocks on its narrow-row kernel (not TSEL_NARROW)


class TileSelect:
    """A target's tile select: its settings and the ops it resolved (once per instance)."""

    def __init__(self, settings: TileSelectSettings):
        self.settings = settings
        self._state: dict = {}

    def op(self):
        """``family_ops().minimax_m31_select_prefill_tile`` on sm_107, or None. Decided once."""
        if "op" not in self._state:
            s = self.settings
            op = None
            if torch.cuda.get_device_capability() == (10, 7):
                op = getattr(family_ops(), "minimax_m31_select_prefill_tile", None)
            self._state["op"] = op
            logger.info(
                f"[M3.1 {s.target}] {s.switch} prefill tile select: "
                f"{'engaged' if op is not None else 'not taken: not sm_107 or the op is not built'} "
                f"(rows >= {TSEL_MIN_ROWS}, blocks <= {TSEL_MAX_BLOCKS})"
            )
        return self._state["op"]

    def rows_op(self):
        """``family_ops().minimax_m31_select_prefill_tile_rows`` on sm_107 (the row-major select), or None. Decided
        once."""
        if "rows_op" not in self._state:
            s = self.settings
            op = None
            if s.rows_enabled and torch.cuda.get_device_capability() == (10, 7):
                op = getattr(family_ops(), "minimax_m31_select_prefill_tile_rows", None)
            self._state["rows_op"] = op
            logger.info(
                f"[M3.1 {s.target}] {s.switch}_ROWS row-major prefill tile select: "
                + (
                    "engaged"
                    if op is not None
                    else f"not taken: TRTLLM_M31_DISABLE_{s.switch}(_ROWS)=1"
                    if not s.rows_enabled
                    else "not taken: not sm_107 or the op is not built"
                )
            )
        return self._state["rows_op"]

    @staticmethod
    def _in_range(max_score_kv) -> bool:
        """Whether the tile select's host-side gate admits these scores (any layout)."""
        return not (
            torch.cuda.is_current_stream_capturing()  # the host-side rows / blocks gate is not graph-stable
            or max_score_kv.dtype != torch.float32
            or max_score_kv.dim() != 3
            or max_score_kv.shape[2] < TSEL_MIN_ROWS
            or max_score_kv.shape[1] > TSEL_MAX_BLOCKS
        )

    def select(self, max_score_kv, n_valid_blocks, init_blocks, local_blocks):
        """The MSA indexer's selector hook: the head-major [total_q, num_kv_heads, 16] table of
        ``select_blocks_from_maxscore``, or None (outside the tile select's rows / blocks range, not head-major, or no
        op) to keep it."""
        if not self._in_range(max_score_kv) or max_score_kv.stride(2) != 1:
            return None
        return self._run(self.op(), max_score_kv, n_valid_blocks, init_blocks, local_blocks)

    def select_rows(self, max_score_kv, n_valid_blocks, init_blocks, local_blocks):
        """The MSA indexer's prefill_scorer_selector (the row-major select): ``select``'s table for the prefill
        scorer's row-major scores (unit block stride), or None (outside the range, another layout, or no rows op) to
        leave them to the indexer's other selects."""
        if not self._in_range(max_score_kv) or max_score_kv.stride(1) != 1:
            return None
        return self._run(self.rows_op(), max_score_kv, n_valid_blocks, init_blocks, local_blocks)

    def _run(self, op, max_score_kv, n_valid_blocks, init_blocks, local_blocks):
        """The table ``op`` (the tile op or its row-major twin) selects, or None without the op."""
        if op is None:
            return None
        num_kv_heads, num_blocks, total_q = max_score_kv.shape
        out = torch.empty(
            (num_kv_heads, total_q, MSA_REQUIRED_TOPK),
            dtype=torch.int32,
            device=max_score_kv.device,
        ).permute(1, 0, 2)
        nvb = n_valid_blocks.to(device=max_score_kv.device, dtype=torch.int32).contiguous()
        op(max_score_kv, nvb, out, init_blocks, local_blocks, narrow=self.settings.narrow)
        return out

    def register(self, model) -> int:
        """Register ``select`` on every layer's MSA indexer that has an unset hook; the number registered (0 when
        off)."""
        s = self.settings
        if not s.enabled:
            logger.info(
                f"[M3.1 {s.target}] {s.switch} off (TRTLLM_M31_DISABLE_{s.switch}=1): built-in select"
            )
            return 0
        registered = 0
        for layer in model.model.layers:
            indexer = getattr(layer.self_attn.attn, "indexer", None)  # only the MSA backend has one
            if indexer is not None and getattr(indexer, "selector_hook", False) is None:
                indexer.selector_hook = self.select
                registered += 1
        return registered
