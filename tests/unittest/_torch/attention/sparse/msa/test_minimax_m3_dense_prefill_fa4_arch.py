# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 FP8 dense prefill: FA4 engages on SM107 only (CPU-level; FA4 itself is stubbed)."""

import pytest

from tensorrt_llm._torch.attention.backends.fmha import msa_prefill

HEAD_DIM, PAGE = 128, 128


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "sm, expected", [(107, True), (100, False), (103, False), (110, False), (90, False)]
)
def test_dense_prefill_fa4_engages_on_sm107_only(
    monkeypatch: pytest.MonkeyPatch, sm: int, expected: bool
) -> None:
    # FA4 is deterministic but not bitwise to fmha_sm100, so every other architecture keeps the stock dense prefill
    # even with a usable FA4 installed. SM107 is the positive control.
    monkeypatch.setattr(msa_prefill, "_dense_fa4_forward", lambda: object())
    monkeypatch.setattr(msa_prefill, "get_sm_version", lambda: sm)
    assert msa_prefill.dense_prefill_fa4_enabled(True, HEAD_DIM, PAGE) is expected


@pytest.mark.cpu_only
def test_dense_prefill_fa4_needs_a_usable_fa4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(msa_prefill, "_dense_fa4_forward", lambda: None)
    monkeypatch.setattr(msa_prefill, "get_sm_version", lambda: 107)
    assert not msa_prefill.dense_prefill_fa4_enabled(True, HEAD_DIM, PAGE)
