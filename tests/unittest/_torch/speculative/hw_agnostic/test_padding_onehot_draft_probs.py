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
"""CUDA-graph padding slots' one-hot draft-prob rows for rejection sampling.

Each padding slot's row becomes a one-hot distribution at draft token 0 over
the draft length, every other row and column is left alone. With
TRTLLM_DFLASH_DEVICE_ONEHOT_SCALARS set the rows are written with index_fill_
and a pinned, non-blocking slot copy; both paths write the same bytes, and the
gated one never synchronizes the host.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative import interface
from tensorrt_llm._torch.speculative.interface import SpecMetadata

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DRAFT_LEN = 3
VOCAB = 6


def _metadata(device, last_dim=0, use_rejection_sampling=True):
    # One more step and two more vocab columns than the write covers, so the
    # test sees anything it touches outside [:DRAFT_LEN, :width].
    return SimpleNamespace(
        use_rejection_sampling=use_rejection_sampling,
        draft_probs=torch.full((5, DRAFT_LEN + 1, VOCAB + 2), 0.5, device=device),
        draft_probs_vocab_size=VOCAB,
        draft_probs_last_dim=last_dim,
    )


def _write(metadata, slots):
    SpecMetadata.write_padding_onehot_draft_probs(metadata, slots, DRAFT_LEN)


def _expected(before, slots, width):
    expected = before.clone()
    expected[slots, :DRAFT_LEN, :width] = 0.0
    expected[slots, :DRAFT_LEN, 0] = 1.0
    return expected


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize(
    "last_dim,width", [(0, VOCAB), (4, 4)], ids=["vocab_size_width", "published_width"]
)
def test_padding_rows_are_onehot(monkeypatch, device, gate, last_dim, width):
    monkeypatch.setattr(interface, "_DEVICE_ONEHOT_SCALARS", gate)
    metadata = _metadata(device, last_dim)
    before = metadata.draft_probs.clone()

    _write(metadata, [3, 1])

    assert torch.equal(metadata.draft_probs, _expected(before, [3, 1], width))
    assert metadata.draft_probs_last_dim == last_dim


@pytest.mark.parametrize("gate", [False, True])
def test_padding_write_is_a_noop_without_slots_or_rejection_sampling(monkeypatch, gate):
    monkeypatch.setattr(interface, "_DEVICE_ONEHOT_SCALARS", gate)
    metadata = _metadata("cpu")
    before = metadata.draft_probs.clone()

    _write(metadata, [])
    assert torch.equal(metadata.draft_probs, before)

    metadata.use_rejection_sampling = False
    _write(metadata, [3, 1])
    assert torch.equal(metadata.draft_probs, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_padding_onehot_write_does_not_synchronize_with_the_gate(monkeypatch):
    monkeypatch.setattr(interface, "_DEVICE_ONEHOT_SCALARS", True)
    metadata = _metadata("cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _write(metadata, [3, 1])
    finally:
        torch.cuda.set_sync_debug_mode("default")
