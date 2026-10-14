# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decoder layer's feed-forward exit on Eagle3 capture layers (``M3DecoderLayer._feed_forward_boundary``).

A capture layer with a next norm keeps the fused boundary (all-reduce + residual + next norm in one kernel) and
captures the reduced residual sum it writes; the last layer captures its raw output with the residual, as the
in-tree capture does. CPU only: the boundary and the capture are stand-ins.
"""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8 import (  # noqa: E501
    layers,
)


class _Layer:
    """The attributes the boundary reads, with a recording capture and next-norm stand-in."""

    _fused_feed_forward_boundary = layers.M3DecoderLayer._fused_feed_forward_boundary
    _feed_forward_boundary = layers.M3DecoderLayer._feed_forward_boundary

    def __init__(self, has_next_norm: bool):
        self.layer_idx = 56
        self.next_layer_layernorm = object() if has_next_norm else None
        self.captured = []
        self.boundaries = []

    def capture(self, spec_metadata, layer_id, hidden_states, residual):
        self.captured.append((layer_id, hidden_states, residual))

    def _next_layer_norm(self, hidden_states, residual, fused_boundary):
        self.boundaries.append(fused_boundary)
        if self.next_layer_layernorm is None:
            return hidden_states, residual
        return ("normed", hidden_states), ("updated", hidden_states, residual)


def test_a_capture_layer_keeps_the_fused_boundary_and_captures_its_residual_sum() -> None:
    layer = _Layer(has_next_norm=True)
    fused = layer._fused_feed_forward_boundary(capture=True)
    assert fused
    h, r = torch.ones(2), torch.zeros(2)
    out = layer._feed_forward_boundary(h, r, fused, True, None)
    assert layer.boundaries == [True]
    assert layer.captured == [(56, ("updated", h, r), None)]
    assert out == (("normed", h), ("updated", h, r))


def test_the_last_layer_captures_its_raw_output_with_the_residual() -> None:
    layer = _Layer(has_next_norm=False)
    fused = layer._fused_feed_forward_boundary(capture=True)
    assert not fused
    h, r = torch.ones(2), torch.zeros(2)
    out = layer._feed_forward_boundary(h, r, fused, True, None)
    assert layer.captured == [(56, h, r)] and layer.boundaries == [False]
    assert out == (h, r)


def test_a_layer_without_capture_is_unchanged() -> None:
    layer = _Layer(has_next_norm=True)
    assert layer._fused_feed_forward_boundary(capture=False)
    h, r = torch.ones(2), torch.zeros(2)
    layer._feed_forward_boundary(h, r, True, False, None)
    assert not layer.captured and layer.boundaries == [True]
