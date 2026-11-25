# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The verify rows the shell gathers for its heads (`_head_rows`): under attention DP up to
`_VERIFY_ROWS_GATHER_MAX` rows in one `embed_rows` launch (a copy, bitwise the embedding), else
`nn.functional.embedding`. The op is a recorder; no kernel runs."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only


class _Cuda(torch.Tensor):
    """A host tensor that answers `is_cuda` (the dispatch reads only host properties)."""

    @property
    def is_cuda(self):
        return True


@pytest.mark.parametrize(
    "dp_size, rows, dtype, strided, expected",
    [
        (8, 12, torch.int32, False, "embed_rows"),
        (4, 31, torch.int64, False, "embed_rows"),
        (8, 32, torch.int32, False, "embedding"),  # PyTorch's vectorized gather from 32 rows
        (1, 12, torch.int32, False, "embedding"),  # attention TP keeps its path
        (8, 12, torch.int16, False, "embedding"),
        (8, 12, torch.int32, True, "embedding"),  # a column-strided table
    ],
)
def test_dispatch(monkeypatch, dp_size, rows, dtype, strided, expected):
    calls = []
    monkeypatch.setattr(
        M,
        "embed_rows",
        lambda ids, table, copies, clamp: calls.append(("embed_rows", copies, clamp)),
    )
    monkeypatch.setattr(
        M.nn.functional, "embedding", lambda ids, table: calls.append(("embedding",))
    )
    final = torch.zeros(64, 16)
    if strided:
        final = torch.zeros(16, 64).t()
    final = final.as_subclass(_Cuda)
    ids = torch.arange(rows, dtype=dtype)
    M._head_rows(SimpleNamespace(dp_size=dp_size), final, ids)
    assert calls == ([("embed_rows", 1, False)] if expected == "embed_rows" else [("embedding",)])
