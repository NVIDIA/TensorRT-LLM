# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep target takes the shared-bank context path only within the engine's bound (`_shared_context_plan`): a bank of
at most `rows x (window + the selection's 128-key tiles)` rows. A short chunk over a very long history (whose bank would
hold every compressed row of the history: about 640 MB of grow-only workspace after one ~1M-position step) takes the
selected-row path instead."""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="imports the target (CUDA build)"
)
attn = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep.attention"
)


def _ctx(history: int, rows: int):
    pos = torch.arange(history, history + rows, dtype=torch.int32, device="cuda")
    rid = torch.zeros(rows, dtype=torch.int64, device="cuda")
    return NS(pos=pos, rid=rid, segs=[(0, rows, 0, history + rows)], nreq=1, floor=None, cache={})


@pytest.mark.parametrize("ratio", [1, 4])
def test_short_history_takes_the_shared_bank(ratio):
    assert attn._shared_bank_fits(_ctx(2000, 512), 128, ratio, 512, 512)


def test_long_history_falls_back_to_the_selected_rows():
    assert not attn._shared_bank_fits(_ctx(1 << 20, 128), 128, 4, 512, 128)
