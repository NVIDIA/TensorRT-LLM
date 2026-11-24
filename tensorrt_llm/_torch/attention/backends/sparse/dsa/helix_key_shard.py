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
"""Key-sharded decode Top-K merge for the DSA indexer under Helix CP.

Under Helix every CP rank already holds only its own KV pages: global page ``g`` (64 tokens) of a
sequence lives on CP rank ``g % cp`` as that rank's local page ``g // cp``. That is the ownership
the key-sharded TP indexer (``key_shard.py``) builds by compaction, so the Helix decode indexer
reuses its path: logits over the local pages into the merge workspace, an exact local Top-L, and
the pack / merge over the CP group instead of ``merge_helix_topk`` (all-gather of every rank's full
local Top-K plus a Top-K over ``cp * K`` candidates per row and layer). ``merge_owned`` writes this
rank's winners as local token indices (the rows ``merge`` followed by :func:`keep_owned_local_rows`
would give), which is what the Helix sparse MLA reads.

A Kimi K3.1 generation server takes this path by default (``TRTLLM_DSA_HELIX_KEY_SHARD=0``
restores ``merge_helix_topk``); every other server keeps ``merge_helix_topk`` unless
``TRTLLM_DSA_HELIX_KEY_SHARD=1``. See :func:`helix_key_shard_enabled`.
"""

import os

import torch

from .params import disagg_role, is_kimi_k31_dsa

# 1 / 0 force the Helix key-sharded merge on / off; unset: see helix_key_shard_enabled.
HELIX_KEY_SHARD_ENV = "TRTLLM_DSA_HELIX_KEY_SHARD"
# Debug: in eager (non-captured) decode steps, recompute every Helix key-sharded merge with
# merge_helix_topk over the same local scores and log the rows whose kept sets differ.
HELIX_KEY_SHARD_VERIFY_ENV = "TRTLLM_DSA_HELIX_KEY_SHARD_VERIFY"


def helix_key_shard_enabled() -> bool:
    """Whether the Helix decode indexer merges its Top-K with the key-sharded kernels.

    ``TRTLLM_DSA_HELIX_KEY_SHARD=1`` / ``0`` forces it on / off. Unset, it follows the gate of the
    key-sharded TP indexer (``key_shard.key_shard_enabled``): on for Kimi K3.1 on a disaggregated
    generation server, off elsewhere. The static layout gates are in
    ``DSAtrtllmAttentionMetadata._key_shard_eligible``. Every rank of the CP group must see the same
    value (set it through ``env_overrides``): the merge is a collective.
    """
    value = os.environ.get(HELIX_KEY_SHARD_ENV)
    if value is None:
        return is_kimi_k31_dsa() and disagg_role() == "generation"
    return value == "1"


def helix_key_shard_verify_enabled() -> bool:
    """Debug switch: check the key-sharded merge against ``merge_helix_topk`` in eager steps (1 or 2)."""
    return os.environ.get(HELIX_KEY_SHARD_VERIFY_ENV, "0") in ("1", "2")


def helix_key_shard_verify_full() -> bool:
    """``TRTLLM_DSA_HELIX_KEY_SHARD_VERIFY=2``: the reference also recomputes the local scores the way the default
    Helix path does (DeepGEMM paged MQA logits over the per-token expanded rows and Helix bounds)."""
    return os.environ.get(HELIX_KEY_SHARD_VERIFY_ENV, "0") == "2"


def helix_local_topk(local_logits: torch.Tensor, row_lens: torch.Tensor, out: torch.Tensor) -> None:
    """Write this rank's exact local Top-L rows by score (``-1`` padded) into ``out``.

    Always the single-CTA radix-filter kernel, called without the dispatch of ``cute_dsl_indexer_topk_decode``:
    the op sizes a CTA cluster per row from the padded row width and the row count (``auto_cluster_size``), while
    the valid part of a Helix row is only this rank's share of the context, where one CTA is at least as fast.

    Args:
        local_logits: ``[num_rows, local_len]`` fp32 local indexer scores.
        row_lens: ``[num_rows]`` int32 number of valid local scores per row.
        out: ``[num_rows, L]`` int32 local token indices, ``-1`` past the valid entries.
    """
    # Defined only when the CuTe DSL is available.
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLTopKDecodeSingleCTARunner

    CuteDSLTopKDecodeSingleCTARunner.forward(
        local_logits, row_lens, top_k=out.shape[1], next_n=1, output_indices=out
    )


def reference_local_topk(
    local_logits: torch.Tensor, row_lens: torch.Tensor, top_k: int
) -> torch.Tensor:
    """This rank's exact local Top-K rows (``-1`` padded) over each row's first ``row_lens`` scores.

    Args:
        local_logits: ``[num_rows, local_len]`` local indexer scores.
        row_lens: ``[num_rows]`` int32 number of valid local scores per row.
        top_k: Selection size.

    Returns:
        ``[num_rows, top_k]`` int32 local token indices, ``-1`` past the valid entries.
    """
    rows, cols = local_logits.shape
    pos = torch.arange(cols, device=local_logits.device)
    masked = local_logits.float().masked_fill(
        pos[None, :] >= row_lens[:, None].long(), float("-inf")
    )
    k = min(top_k, cols)
    values, indices = torch.topk(masked, k, dim=1)
    out = torch.full((rows, top_k), -1, dtype=torch.int32, device=local_logits.device)
    out[:, :k] = torch.where(torch.isfinite(values), indices.to(torch.int32), -1)
    return out


def count_set_mismatches(got: torch.Tensor, want: torch.Tensor) -> tuple:
    """``(rows whose kept sets differ, entries in the symmetric difference, entries kept in want)``."""
    a = torch.sort(got, dim=1).values
    b = torch.sort(want, dim=1).values
    differs = (a != b).any(dim=1)
    kept_want = int((want >= 0).sum())
    sym_diff = 0
    for i in torch.nonzero(differs).flatten().tolist():
        sa = set(a[i][a[i] >= 0].tolist())
        sb = set(b[i][b[i] >= 0].tolist())
        sym_diff += len(sa ^ sb)
    return int(differs.sum()), sym_diff, kept_want


def kept_scores(logits: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``[num_rows, top_k]`` fp32 scores of the entries a rank keeps, ``-inf`` where a row keeps none."""
    scores = torch.gather(logits.float(), 1, rows.clamp(min=0).long())
    return scores.masked_fill(rows < 0, float("-inf"))


def count_score_mismatches(got: torch.Tensor, want: torch.Tensor, tolerance: float = 0.0) -> int:
    """Rows whose kept scores differ, compared over all ranks and in descending order.

    Two valid Top-K selections of a row keep the same scores even where they break a tie at the cut-off
    differently (and so differ as index sets, which ``count_set_mismatches`` reports). ``got`` / ``want`` are
    the ``[cp, num_rows, top_k]`` kept scores of every rank, ``-inf`` padded; ``tolerance`` allows for scores
    that come from two kernels.
    """
    cp, rows, k = got.shape

    def top(x: torch.Tensor) -> torch.Tensor:
        return torch.sort(x.transpose(0, 1).reshape(rows, cp * k), dim=1, descending=True).values[
            :, :k
        ]

    a, b = top(got), top(want)
    # Padding compares equal to padding; a kept score against padding is a mismatch.
    differs = (a != b) & ~((a - b).abs() <= tolerance)
    return int(differs.any(dim=1).sum())


def keep_owned_local_rows(
    rows: torch.Tensor, cp_rank: int, cp_size: int, tokens_per_block: int = 64
) -> None:
    """Rewrite merged global Top-K rows into this rank's local Top-K rows, in place.

    ``rows`` holds key-shard global token indices (``-1`` padded): index ``p`` belongs to rank
    ``(p // tokens_per_block) % cp_size`` at local position
    ``(p // tokens_per_block) // cp_size * tokens_per_block + p % tokens_per_block``. Entries of
    other ranks become ``-1`` and this rank's entries move to the front of each row in their
    original order, as ``merge_helix_topk`` leaves them.

    Args:
        rows: ``[num_rows, top_k]`` int32 merged Top-K.
        cp_rank: This rank's position in the CP group.
        cp_size: CP group size.
        tokens_per_block: Tokens per KV page.
    """
    if rows.numel() == 0:
        return
    block = torch.div(rows, tokens_per_block, rounding_mode="floor")
    mine = (rows >= 0) & (torch.remainder(block, cp_size) == cp_rank)
    local = torch.div(block, cp_size, rounding_mode="floor") * tokens_per_block + torch.remainder(
        rows, tokens_per_block
    )
    values = torch.where(mine, local, torch.full_like(rows, -1))
    order = torch.sort((~mine).to(torch.int8), dim=1, stable=True).indices
    rows.copy_(torch.gather(values, 1, order))
