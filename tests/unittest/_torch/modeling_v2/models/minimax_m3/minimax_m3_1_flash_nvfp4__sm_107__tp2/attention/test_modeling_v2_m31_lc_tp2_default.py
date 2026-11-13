# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LC_TP2 and TP2_LC2_SMALL_DRAFT_LEN_3 (opt-ins of the tp2 sparse decode): both off by default.

LC_TP2 only silences the Triton route's engagement line for the long-context rows the long-context decode does not
take; TP2_LC2_SMALL_DRAFT_LEN_3 extends the long-context decode's 9-row lower bound to draft length 3 (4 query rows per
request). The opt-in set is patched and the kernel module re-imported (both are read at import); no GPU needed."""

import importlib

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2 import (  # noqa: E501
        m31_switches,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
        triton_sparse_decode,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


def test_off_by_default() -> None:
    if m31_switches._ENABLED:
        pytest.skip("an opt-in is set in this environment")
    assert triton_sparse_decode.LC_TP2 is False
    assert triton_sparse_decode.LC_TRITON_PATH == "m3_sparse_decode_lc_tp2_triton"
    assert triton_sparse_decode.LC2_SMALL_QLENS == frozenset({8})


@pytest.mark.parametrize(
    "enabled, lc_tp2, qlens",
    [
        (frozenset(), False, frozenset({8})),
        (frozenset({"LC_TP2"}), True, frozenset({8})),
        (frozenset({"TP2_LC2_SMALL_DRAFT_LEN_3"}), False, frozenset({4, 8})),
    ],
    ids=["default", "lc_tp2", "draft_len_3"],
)
def test_the_opt_ins(monkeypatch, enabled, lc_tp2, qlens) -> None:
    monkeypatch.setattr(m31_switches, "_ENABLED", enabled)
    try:
        tsd = importlib.reload(triton_sparse_decode)
        assert tsd.LC_TP2 is lc_tp2
        assert tsd.LC2_SMALL_QLENS == qlens
    finally:
        monkeypatch.undo()
        importlib.reload(triton_sparse_decode)
