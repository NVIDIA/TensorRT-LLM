# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The (token, slot) -> row contract of ``expert_rows.py``, host-only.

* ``slot_row_map`` is ``t * 8 + k`` for an id inside the window and ``-1``
  otherwise, for every row count and both windows; a token that repeats an
  in-window id keeps only the first such slot.
* ``routed_row_valid`` is ``0 <= row < rows`` for ints and tensors alike.
* ``gather_routed_rows`` selects the same rows the built-in
  ``post_expert_norm_rows`` selects (and zeroes invalid slots without reading
  them), for a permuted map as TRTLLM-gen writes it and for the slot map.
* The built-in consumers still follow the contract: the fused finalize
  all-reduce kernel and the unfused combine both bound the map on both sides,
  so a finalize reading the persistent experts' map needs no change. The ``finalize_transport``
  kernel selects its rows through ``routed_row_valid`` / ``routed_row_offset``
  (its ``select_row`` and its per-slot scale) and has no selection rule of its own.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.persistent_experts import (  # noqa: E501
    rows as expert_rows,
)

pytestmark = pytest.mark.cpu_only

TOP_K = expert_rows.TOP_K


@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("window", [32, 64])
@pytest.mark.parametrize("id_range", [256, 12])
def test_slot_row_map(rows, window, id_range):
    gen = torch.Generator().manual_seed(rows * 100 + window + id_range)
    # id_range 12: every id local and most rows repeat one
    ids = torch.randint(-1, id_range, (rows, TOP_K), generator=gen, dtype=torch.int32)
    got = expert_rows.slot_row_map(ids, window)
    assert got.dtype == torch.int32 and tuple(got.shape) == (rows, TOP_K)
    for t in range(rows):
        row = ids[t].tolist()
        for k in range(TOP_K):
            first = 0 <= row[k] < window and row[k] not in row[:k]
            assert int(got[t, k]) == (t * TOP_K + k if first else -1)


def test_repeated_expert_keeps_the_first_slot():
    """A token that names one in-window expert in several slots is routed to it once."""
    ids = torch.tensor(
        [
            [5, 5, 100, 101, 102, 103, 104, 105],
            [100, 7, 101, 102, 103, 104, 105, 7],
            [3, 3, 3, 3, 3, 3, 3, 3],
            [1, 2, 2, 1, -1, -1, 40, 40],
        ],
        dtype=torch.int32,
    )
    got = expert_rows.slot_row_map(ids, 32)
    assert got.tolist() == [
        [0, -1, -1, -1, -1, -1, -1, -1],
        [-1, 9, -1, -1, -1, -1, -1, -1],
        [16, -1, -1, -1, -1, -1, -1, -1],
        [24, 25, -1, -1, -1, -1, -1, -1],
    ]
    assert expert_rows.first_local_slots(ids, 64)[3].tolist() == [1, 1, 0, 0, 0, 0, 1, 0]


def test_routed_row_valid_on_ints_and_tensors():
    assert expert_rows.routed_row_valid(0, 8) and expert_rows.routed_row_valid(7, 8)
    assert not expert_rows.routed_row_valid(-1, 8) and not expert_rows.routed_row_valid(8, 8)
    rows = torch.tensor([-5, -1, 0, 3, 7, 8, 100], dtype=torch.int32)
    assert expert_rows.routed_row_valid(rows, 8).tolist() == [0, 0, 1, 1, 1, 0, 0]
    assert expert_rows.routed_row_offset(3, 4096, 16) == 3 * 4096 + 16


def _builtin_rows(gemm2, index, num_tokens, hidden):
    from tensorrt_llm._torch.models.modeling_musespark_open import post_expert_norm_rows

    # post_expert_norm_rows applies the norm after selecting; with a unit gain
    # and a huge eps the norm is a fixed scale, so compare the selection only
    # through which rows are non-zero and through a norm-free re-selection.
    _, valid = post_expert_norm_rows(
        gemm2,
        index,
        torch.zeros(hidden, dtype=torch.float32),
        1.0,
        num_tokens=num_tokens,
        top_k=TOP_K,
        hidden_size=hidden,
    )
    return valid


@pytest.mark.parametrize("producer", ["permuted", "slot"])
def test_gather_selects_the_builtin_rows(producer):
    gen = torch.Generator().manual_seed(7)
    tokens, hidden = 5, 64
    ids = torch.randint(0, 96, (tokens, TOP_K), generator=gen, dtype=torch.int32)
    local = expert_rows.first_local_slots(ids, 32)  # a repeated expert counts once
    if producer == "slot":
        index = expert_rows.slot_row_map(ids, 32)
        gemm2 = torch.randn(tokens * TOP_K, hidden + 8, generator=gen).to(torch.bfloat16)
    else:
        # Any injective placement of the local slots into a larger padded buffer.
        n = int(local.sum())
        perm = torch.randperm(4 * tokens * TOP_K, generator=gen)[:n].to(torch.int32)
        index = torch.full((tokens, TOP_K), -1, dtype=torch.int32)
        index[local] = perm
        gemm2 = torch.randn(4 * tokens * TOP_K, hidden + 8, generator=gen).to(torch.bfloat16)
    # Rows no local slot addresses hold NaN: the selection must never read them.
    addressed = torch.zeros(gemm2.shape[0], dtype=torch.bool)
    addressed[index[index >= 0].long()] = True
    gemm2[~addressed] = float("nan")

    got, valid = expert_rows.gather_routed_rows(gemm2, index, tokens, hidden)
    assert torch.equal(valid, local)
    assert torch.equal(valid, _builtin_rows(gemm2, index, tokens, hidden))
    assert not torch.isnan(got).any()
    for t, k in local.nonzero().tolist():
        assert torch.equal(got[t, k], gemm2[int(index[t, k]), :hidden].float())
    assert not got[~local].any()
