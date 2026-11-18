# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row layout of a QSA index-key page.

Plain layout: a page holds `tokens_per_block` rows, row `pos % tokens_per_block`: the token's raw
index key, overwritten at the last token of a `compress_ratio` group (the anchor) by the group's
compressed key. Index scoring reads only anchors; the raw rows are read once more, to pool their
group, and are dead after it.

Compact layout (`TRTLLM_QSA_INDEX_PLANE_COMPACT=1`): a page holds the `tokens_per_block //
compress_ratio` compressed keys in its first rows (group `g` of the page at row `g`), then two
sets of `compress_ratio - 1` raw rows, one per group parity: raw key `pos` at row
`groups + ((pos // cr) & 1) * (cr - 1) + pos % cr`. A group's raw keys live until the group two
groups later starts, so any window of up to `cr + 1` consecutive positions maps to distinct rows
and a group's members survive the next group's writes. A raw key is only stored where a later
step may pool it (a request's last positions of a step); every reuse point must be a group edge
(the KV cache manager moves recurrent-state snapshot points onto group edges under this layout).
At 32 tokens a page and groups of 4: 14 rows instead of 32.
"""

from __future__ import annotations

import os

COMPACT_ENV = "TRTLLM_QSA_INDEX_PLANE_COMPACT"


def compact_enabled() -> bool:
    """Whether QSA index pages use the compact layout (process-wide; both sides of a
    disaggregated deployment must agree, since the page size changes)."""
    return os.environ.get(COMPACT_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def plane_rows(tokens_per_block: int, compress_ratio: int, compact: bool | None = None) -> int:
    """Rows of one index page."""
    if compact is None:
        compact = compact_enabled()
    if not compact:
        return tokens_per_block
    assert tokens_per_block % compress_ratio == 0, (tokens_per_block, compress_ratio)
    return tokens_per_block // compress_ratio + 2 * (compress_ratio - 1)


def row_of(pos: int, tokens_per_block: int, compress_ratio: int, compact: bool) -> int:
    """The page row of position `pos` (its raw key, or the compressed key at a group's last
    token). Reference for the kernels, which inline the same arithmetic."""
    if not compact:
        return pos % tokens_per_block
    within = pos % tokens_per_block
    if pos % compress_ratio == compress_ratio - 1:
        return within // compress_ratio
    groups = tokens_per_block // compress_ratio
    return groups + ((pos // compress_ratio) & 1) * (compress_ratio - 1) + pos % compress_ratio


def anchor_row(block: int, tokens_per_block: int, compress_ratio: int, compact: bool) -> int:
    """The page row of compressed key `block` (the group of positions `block * cr ..`)."""
    if not compact:
        return (block * compress_ratio + compress_ratio - 1) % tokens_per_block
    return block % (tokens_per_block // compress_ratio)
