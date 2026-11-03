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
"""Key-sharded decode Top-K merge for the DSA indexer under Helix CP (opt-in).

Under Helix every CP rank already holds only its own KV pages: global page ``g`` (64 tokens) of a
sequence lives on CP rank ``g % cp`` as that rank's local page ``g // cp``. That is the ownership
the key-sharded TP indexer (``key_shard.py``) builds by compaction, so with
``TRTLLM_DSA_HELIX_KEY_SHARD=1`` the Helix decode indexer reuses its path: logits over the local
pages into the merge workspace, an exact local Top-L, and the pack / merge over the CP group
instead of ``merge_helix_topk`` (all-gather of every rank's full local Top-K plus a Top-K over
``cp * K`` candidates per row and layer). The merge returns the global Top-K in the key-shard
index space; :func:`keep_owned_local_rows` keeps this rank's winners as local token indices, which
is what the Helix sparse MLA reads.
"""

import os

import torch

HELIX_KEY_SHARD_ENV = "TRTLLM_DSA_HELIX_KEY_SHARD"
# Debug: in eager (non-captured) decode steps, recompute every Helix key-sharded merge with
# merge_helix_topk over the same local scores and log the rows whose kept sets differ.
HELIX_KEY_SHARD_VERIFY_ENV = "TRTLLM_DSA_HELIX_KEY_SHARD_VERIFY"


def helix_key_shard_enabled() -> bool:
    """Opt-in switch of the key-sharded merge under Helix CP."""
    return os.environ.get(HELIX_KEY_SHARD_ENV, "0") == "1"


def helix_key_shard_verify_enabled() -> bool:
    """Debug switch: check the key-sharded merge against ``merge_helix_topk`` in eager steps (1 or 2)."""
    return os.environ.get(HELIX_KEY_SHARD_VERIFY_ENV, "0") in ("1", "2")


def helix_key_shard_verify_full() -> bool:
    """``TRTLLM_DSA_HELIX_KEY_SHARD_VERIFY=2``: the reference also recomputes the local scores the way the default
    Helix path does (DeepGEMM paged MQA logits over the per-token expanded rows and Helix bounds)."""
    return os.environ.get(HELIX_KEY_SHARD_VERIFY_ENV, "0") == "2"


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
