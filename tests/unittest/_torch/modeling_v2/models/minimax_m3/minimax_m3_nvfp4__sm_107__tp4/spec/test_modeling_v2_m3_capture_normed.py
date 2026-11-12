# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU test: a first-draft-inputs reader that skips the drafter's fc_norm (the capture norm is armed) gets the captured
rows unchanged when every captured layer was written with the step's norm, and otherwise a copy with only the raw
layers normalized by their own fc_norm (M3Eagle3Worker._m3_renorm_raw_captures); the per-forward reset clears the
record and the unconsumed fc fork result."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4 import (  # noqa: E501
    modeling,
)

M3Eagle3Worker = modeling.M3Eagle3Worker

H = 4
LAYERS = (1, 29, 56)


def _worker(normed):
    w = object.__new__(M3Eagle3Worker)
    w._m3_capture_normed = set(normed)
    return w


@pytest.fixture(autouse=True)
def _scaling_norm(monkeypatch):
    """The drafter's RMSNorm as a stand-in that multiplies by its weight (CPU, no FlashInfer)."""
    monkeypatch.setattr(modeling, "flashinfer_rmsnorm", lambda x, w, eps: x * w)


def _drafter():
    # fc_norm i scales by (i + 2): enough to tell which blocks were normalized here.
    return SimpleNamespace(w={f"fc_norm{i}": i + 2 for i in range(len(LAYERS))}, rms_eps=1e-6)


def test_all_captured_layers_normed_returns_the_rows_as_they_lie() -> None:
    hs = torch.randn(4, H * len(LAYERS))
    out = _worker(set(LAYERS))._m3_renorm_raw_captures(
        hs, SimpleNamespace(layers_to_capture=LAYERS, hidden_size=H), _drafter()
    )
    assert out is hs


@pytest.mark.parametrize("normed", [set(), {1, 29}, {29, 56}, {1, 56}])
def test_raw_layers_get_their_own_fc_norm(normed) -> None:
    hs = torch.randn(4, H * len(LAYERS))
    ref = hs.clone()
    out = _worker(normed)._m3_renorm_raw_captures(
        hs, SimpleNamespace(layers_to_capture=LAYERS, hidden_size=H), _drafter()
    )
    assert torch.equal(hs, ref)  # the capture buffer itself is not modified
    for i, layer_id in enumerate(LAYERS):
        block, src = out[:, i * H : (i + 1) * H], ref[:, i * H : (i + 1) * H]
        assert torch.equal(block, src if layer_id in normed else src * (i + 2)), layer_id


def test_arm_resets_the_record_and_the_fork_result() -> None:
    w = _worker(set(LAYERS))
    w._m3_fc_hidden = object()
    w._m3_fc_early_stream = None  # no fork pending
    w._m3_join_pending = lambda: None  # (tp8) nothing else pending
    w.arm_capture_norm(
        None, None, None, None
    )  # no drafter: resets the per-forward state and returns
    assert w._m3_capture_normed == set()
    assert w._m3_fc_hidden is None
