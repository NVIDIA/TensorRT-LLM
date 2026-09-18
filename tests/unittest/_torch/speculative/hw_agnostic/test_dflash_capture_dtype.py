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
"""DFlash hidden-state capture when the tap dtype is wider than the buffer.

The capture buffer is allocated in the target's ``torch_dtype``, but a model
may tap a wider stream (an FP32 residual, say), and folding ``residual`` inside
the capture call can promote the result as well. The capture converts the tap
to the buffer dtype so the write is well-defined and the drafter keeps seeing
buffer-dtype-rounded values.

These tests require CUDA: ``DFlashSpecMetadata`` allocates the capture buffer
on ``cuda`` and ``inplace_slice_copy`` is a CUDA-only custom op.
"""

import pytest
import torch

from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

requires_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="DFlashSpecMetadata allocates its bfloat16 capture buffer on cuda "
    "and inplace_slice_copy is a CUDA-only custom op",
)

HIDDEN_SIZE = 16
MAX_TOKENS = 8


def _make_metadata(buffer_dtype: torch.dtype):
    from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata

    return DFlashSpecMetadata(
        max_draft_len=4,
        max_total_draft_tokens=4,
        spec_dec_mode=SpeculativeDecodingMode.DFLASH,
        max_num_requests=2,
        max_num_tokens=MAX_TOKENS,
        hidden_size=HIDDEN_SIZE,
        layers_to_capture=[1, 3],
        dtype=buffer_dtype,
    )


@requires_cuda
def test_capture_converts_wider_tap_to_buffer_dtype():
    """An FP32 tap into a bf16 buffer must store the bf16-rounded values."""
    md = _make_metadata(torch.bfloat16)

    h1 = torch.randn(MAX_TOKENS, HIDDEN_SIZE, dtype=torch.float32, device="cuda")
    h3 = torch.randn_like(h1)
    md.maybe_capture_hidden_states(1, h1, None)
    md.maybe_capture_hidden_states(3, h3, None)

    captured = md.get_hidden_states(MAX_TOKENS)
    assert captured.dtype == torch.bfloat16
    assert captured.shape == (MAX_TOKENS, 2 * HIDDEN_SIZE)
    # Bitwise, not approximate: the drafter consumes buffer-dtype-rounded
    # values, so the capture must store exactly the bf16-rounded tap.
    assert torch.equal(captured[:, :HIDDEN_SIZE], h1.to(torch.bfloat16))
    assert torch.equal(captured[:, HIDDEN_SIZE:], h3.to(torch.bfloat16))


@requires_cuda
def test_capture_converts_when_residual_fold_promotes():
    """``hidden_states + residual`` can promote past the buffer dtype."""
    md = _make_metadata(torch.bfloat16)

    hidden = torch.randn(MAX_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(MAX_TOKENS, HIDDEN_SIZE, dtype=torch.float32, device="cuda")
    assert (hidden + residual).dtype == torch.float32

    md.maybe_capture_hidden_states(1, hidden, residual)

    captured = md.get_hidden_states(MAX_TOKENS)
    assert captured.dtype == torch.bfloat16
    assert torch.equal(captured[:, :HIDDEN_SIZE], (hidden + residual).to(torch.bfloat16))


@requires_cuda
def test_capture_leaves_matching_dtype_untouched():
    """Control: a tap already in the buffer dtype is stored verbatim."""
    md = _make_metadata(torch.bfloat16)

    h1 = torch.randn(MAX_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    md.maybe_capture_hidden_states(1, h1, None)

    captured = md.get_hidden_states(MAX_TOKENS)
    assert torch.equal(captured[:, :HIDDEN_SIZE], h1)
