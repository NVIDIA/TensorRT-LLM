# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam component shapes for thop_attention, using the shared operator comparison."""

import pytest

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch.modeling_v2.catalog.attention.thop_attention import STANDARD
from tensorrt_llm._torch.modeling_v2.catalog.attention.thop_attention import thop_attention as op

__extra_import_path__ = ["../../../../attention", "../../../.."]
import test_modeling_v2_thop_attention as shared  # noqa: E402
from _validating import certifying  # noqa: E402,F401

_BEAM_DENSE = dict(
    input_type=STANDARD,
    num_heads=48,
    num_kv_heads=8,
    head_size=128,
    v_head_dim=128,
    sinks=False,
    window=512,
    quant_mode=0,
    mask_type=1,
    tokens_per_block=32,
    num_blocks=16,
    max_num_requests=1,
    use_paged_context_fmha=False,
    num_contexts=1,
    predicted_tokens_per_seq=1,
)

_BEAM_SLIDING = dict(
    _BEAM_DENSE,
    num_blocks=64,
    max_num_requests=3,
    max_blocks_per_seq=20,
    use_paged_context_fmha=True,
)

_BEAM_FULL = dict(_BEAM_SLIDING, window=None)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "beam/full_model_nvfp4__sm_103__tp4": tuple(
        Cell(
            why=f"beam TP4 {name}, {'full' if window is None else 'sliding'} attention",
            spec=dict(
                _BEAM_SLIDING,
                num_heads=12,
                num_kv_heads=2,
                window=window,
                cached=cached,
                new=new,
                num_contexts=contexts,
                predicted_tokens_per_seq=1,
            ),
        )
        for window in (512, None)
        for name, cached, new, contexts in (
            ("fresh decode-sized prefill", (0,), (1,), 1),
            ("fresh prefill", (0,), (129,), 1),
            ("prefix reuse", (64,), (17,), 1),
            ("prefill beyond window", (0,), (513,), 1),
            ("prefix at window", (512,), (1,), 1),
            ("decode across window", (511, 512, 513), (1, 1, 1), 0),
        )
    ),
    "beam/six_layer_model": (
        *(
            Cell(
                why=f"beam dense block: GQA 48q/8kv d128, fresh {tokens}-row context",
                spec=dict(_BEAM_DENSE, cached=(0,), new=(tokens,)),
            )
            for tokens in (1, 17, 129)
        ),
        Cell(
            why="beam sliding, paged context: 17 rows over a 64-token reused prefix",
            spec=dict(
                _BEAM_SLIDING, cached=(64,), new=(17,), num_contexts=1, predicted_tokens_per_seq=1
            ),
        ),
        Cell(
            why="beam sliding, paged context: a fresh 513-row context, key 0 leaves the "
            "window at row 512",
            spec=dict(
                _BEAM_SLIDING, cached=(0,), new=(513,), num_contexts=1, predicted_tokens_per_seq=1
            ),
        ),
        Cell(
            why="beam sliding, paged context: one row at position 512 after a 512-token "
            "reused prefix",
            spec=dict(
                _BEAM_SLIDING, cached=(512,), new=(1,), num_contexts=1, predicted_tokens_per_seq=1
            ),
        ),
        Cell(
            why="beam sliding decode across the window: histories 511, 512 and 513",
            spec=dict(
                _BEAM_SLIDING,
                cached=(511, 512, 513),
                new=(1, 1, 1),
                num_contexts=0,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="beam full layer decode past the sliding window: histories 511, 512 and 513",
            spec=dict(
                _BEAM_FULL,
                cached=(511, 512, 513),
                new=(1, 1, 1),
                num_contexts=0,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="beam full layer, paged context: 17 rows over a 64-token reused prefix",
            spec=dict(
                _BEAM_FULL, cached=(64,), new=(17,), num_contexts=1, predicted_tokens_per_seq=1
            ),
        ),
    ),
}

# A shared shape used by multiple components needs only one kernel comparison.
CELLS = tuple(
    {
        repr(sorted(cell.spec.items())): cell for cells in TARGET_CELLS.values() for cell in cells
    }.values()
)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why for c in CELLS])
def test_certified_cells(cell: Cell, certifying) -> None:  # noqa: F811
    assert shared.op is op
    shared.test_certified_cells(cell, certifying)
