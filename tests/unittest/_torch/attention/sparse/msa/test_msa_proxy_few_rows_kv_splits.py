# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate of the SM107 few-row KV split count of the MiniMax-M3 indexer proxy plan."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_backend


@pytest.fixture(autouse=True)
def _sm_count(monkeypatch):
    monkeypatch.setattr(msa_backend, "_sm_count", lambda device_index: 212)


@pytest.mark.parametrize(
    "sm_version,enabled,num_rows,expected",
    [
        (107, True, 1, 212),
        (107, True, 2, 106),
        (107, True, 4, 53),
        (107, True, 5, -1),
        (107, True, 128, -1),
        (103, True, 1, -1),
        (107, False, 1, -1),
    ],
)
def test_proxy_kv_splits(monkeypatch, sm_version, enabled, num_rows, expected):
    monkeypatch.setattr(msa_backend, "get_sm_version", lambda: sm_version)
    monkeypatch.setattr(msa_backend, "_SM107_PROXY_FEW_ROWS_KV_SPLITS", enabled)
    assert msa_backend.proxy_kv_splits(num_rows, torch.device("cuda", 0)) == expected
