# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""When may the residual add be folded into the projection's GEMM epilogue?

The fold writes the projection's result *into* the accumulator. That is only sound when the
accumulator is ours to write and nobody reads it afterwards expecting the pre-add value, so
each condition below is a case where folding would be silently wrong — the fusion must
decline, not produce a plausible answer.
"""

import pytest
import torch
from torch.fx import Graph

from tensorrt_llm._torch.auto_deploy.transform.library.fuse_nanojet_gemm_fp8_add import (
    _is_fp8,
    _sole_residual_add,
)


def _fake(shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="meta")


def _build(
    *,
    accumulator_is_input=False,
    extra_reader_after=False,
    extra_reader_before=False,
    accumulator_dtype=torch.bfloat16,
    accumulator_shape=(4, 8),
    two_users=False,
):
    """A projection feeding an add, with one condition varied at a time."""
    graph = Graph()
    activation = graph.placeholder("activation")
    activation.meta["val"] = _fake((4, 8))

    if accumulator_is_input:
        accumulator = graph.placeholder("residual")
    else:
        accumulator = graph.call_function(torch.ops.aten.mul.Tensor, args=(activation, 1.0))
    accumulator.meta["val"] = _fake(accumulator_shape, accumulator_dtype)

    if extra_reader_before:
        before = graph.call_function(torch.ops.aten.relu.default, args=(accumulator,))
        before.meta["val"] = _fake(accumulator_shape, accumulator_dtype)

    projection = graph.call_function(torch.ops.aten.mm.default, args=(activation, activation))
    projection.meta["val"] = _fake((4, 8))

    add = graph.call_function(torch.ops.aten.add.Tensor, args=(accumulator, projection))
    add.meta["val"] = _fake((4, 8))

    if two_users:
        graph.call_function(torch.ops.aten.relu.default, args=(projection,))
    if extra_reader_after:
        graph.call_function(torch.ops.aten.neg.default, args=(accumulator,))

    graph.output(add)
    order = {n: i for i, n in enumerate(graph.nodes)}
    return projection, order


def test_folds_the_ordinary_residual():
    """The supported shape must fold, or every rejection below proves nothing."""
    projection, order = _build()
    assert _sole_residual_add(projection, order) is not None


def test_declines_when_projection_has_another_reader():
    """Folding erases the add; a second consumer of the projection would lose its producer."""
    projection, order = _build(two_users=True)
    assert _sole_residual_add(projection, order) is None


def test_declines_when_accumulator_is_a_graph_input():
    """A placeholder belongs to the caller — writing through it corrupts their tensor."""
    projection, order = _build(accumulator_is_input=True)
    assert _sole_residual_add(projection, order) is None


def test_declines_when_accumulator_read_after_the_add():
    """That reader wants the pre-add value; the in-place write would hand it the sum."""
    projection, order = _build(extra_reader_after=True)
    assert _sole_residual_add(projection, order) is None


def test_allows_accumulator_read_before_the_add():
    """A reader that already ran is fine — this is the norm in a transformer block."""
    projection, order = _build(extra_reader_before=True)
    assert _sole_residual_add(projection, order) is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float8_e4m3fn])
def test_declines_on_non_bf16_accumulator(dtype):
    projection, order = _build(accumulator_dtype=dtype)
    assert _sole_residual_add(projection, order) is None


def test_declines_on_shape_mismatch():
    """The accumulator is written into, not broadcast against."""
    projection, order = _build(accumulator_shape=(1, 8))
    assert _sole_residual_add(projection, order) is None


# --------------------------------------------------------------------------------------
# The fusion decides by the dtype of the activation it reads, nothing else.
# --------------------------------------------------------------------------------------


def test_declines_when_attention_is_another_backend():
    """Flashinfer's attention emits BF16; treating it as e4m3 would feed the wrong dtype.

    This is the mixed-backend case: nanojet fusions on, attention deliberately left to
    another backend. o_proj must simply not fold.
    """
    graph = Graph()
    hidden = graph.placeholder("hidden")
    hidden.meta["val"] = _fake((4, 8))
    other_attention = graph.call_function(torch.ops.aten.mm.default, args=(hidden, hidden))
    other_attention.meta["val"] = _fake((4, 8))  # BF16, as any non-nanojet attention emits
    assert not _is_fp8(other_attention)


def test_declines_on_bf16_even_behind_a_view():
    """Views are transparent; the dtype behind them still decides."""
    graph = Graph()
    hidden = graph.placeholder("hidden")
    hidden.meta["val"] = _fake((4, 8))
    view = graph.call_function(torch.ops.aten.reshape.default, args=(hidden, [4, 8]))
    view.meta["val"] = _fake((4, 8))
    assert not _is_fp8(view)


def test_accepts_fp8_producer():
    graph = Graph()
    hidden = graph.placeholder("hidden")
    hidden.meta["val"] = _fake((4, 8), torch.float8_e4m3fn)
    assert _is_fp8(hidden)
