# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded host dispatch to the existing shared FSEL kernel for newly ported decode paths."""

from .ops import m3_fsel_topk as _fsel


def prepare(
    device, *, enabled: bool, target: str, rank: int, max_rows: int, trigger_phase: int
) -> bool:
    """Prepare the identity table before capture, or retain the original selector."""
    if not enabled or _fsel.fsel_op() is None:
        return False
    _fsel.prepare(device)
    _fsel.log_engaged(target, rank, device, max_rows, trigger_phase)
    return True


def select(
    scores,
    *,
    n_valid_blocks,
    num_tokens: int,
    init_blocks: int,
    local_blocks: int,
    enabled: bool,
    max_rows: int,
    use_pdl: bool,
    trigger_phase: int,
    num_kv_heads: int = 1,
):
    """Use only the existing certified forcing/head domain; unsupported calls keep stock.

    The index scorer supplies finite valid-block maxima; this adapter does not add a device sync to inspect them.
    """
    if (
        not enabled
        or not 1 <= num_tokens <= max_rows
        or scores.dim() != 3
        or num_kv_heads not in (1, 2)
        or scores.shape[0] != num_kv_heads
        or (init_blocks, local_blocks) != (0, 1)
        or not 128 < scores.shape[1] <= _fsel.MAX_BLOCKS
    ):
        return None
    return _fsel.fsel_select_blocks(
        scores,
        n_valid_blocks=n_valid_blocks,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        use_pdl=use_pdl,
        trigger_phase=trigger_phase,
    )
