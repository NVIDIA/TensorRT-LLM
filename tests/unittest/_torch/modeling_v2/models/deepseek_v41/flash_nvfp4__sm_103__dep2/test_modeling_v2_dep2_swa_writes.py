# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A dep2 context chunk writes every live row to its SWA slot (`_chunk_consts`' `keep`), as stock writes each row's
slot: prefix reuse publishes whole 128-row pages, so the chunk's rows before its last window must be in the pool too.
A graph bucket's padding rows (they copy the last row's position, not its values) write nothing. Before, only the last
`window` rows of each request were written."""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="imports the target (CUDA build)"
)
attn = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep2.attention"
)


def _ctx(n, live=None):
    pos = torch.arange(1000, 1000 + n, dtype=torch.int32, device="cuda")
    return NS(pos=pos, ce=pos.new_full((n,), 1000 + n), live=live, cache={})


def test_every_row_of_an_eager_chunk_is_written():
    keep, arange_n = attn._chunk_consts(_ctx(700), 128, 700)
    assert (
        bool(keep.all()) and keep.shape == (700,) and torch.equal(arange_n.cpu(), torch.arange(700))
    )


def test_padding_rows_of_a_graph_bucket_are_not_written():
    live = torch.arange(1024, device="cuda") < 700
    keep, _ = attn._chunk_consts(_ctx(1024, live), 128, 1024)
    assert torch.equal(keep, live)
