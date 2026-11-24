# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row capacity of the key-sharded decode Top-K merge. No torch import, so it is testable anywhere."""


def key_shard_max_rows(max_batch_size: int, max_draft_len: int) -> int:
    """Rows of the largest decode step the merge workspace serves: every sequence verifies its draft tokens and
    its current token, so ``max_batch_size * (max_draft_len + 1)``. Sized from the server's limits at start, the
    workspace holds every step the server can run; a step with more rows is an error, not a silent fallback to
    the unsharded merge."""
    if max_batch_size < 1 or max_draft_len < 0:
        raise ValueError(
            "key-sharded decode rows need max_batch_size >= 1 and max_draft_len >= 0, "
            f"got {max_batch_size} and {max_draft_len}"
        )
    return max_batch_size * (max_draft_len + 1)
