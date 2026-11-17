# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PAD_MASK on the MiniMax-M3.1 DEP target: a generation step's padding-request rows reach the MoE backend's
``set_pad_mask`` as a uint8 row mask, only for a launch of exactly those rows and only on layers whose backend takes
the mask; with the MoE mask installed, the LC2 decode reads the padding requests' KV lengths as 0. CPU only."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._router_index import NoTarget

try:
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        lc2_decode,
        pad_mask,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

ROWS = 8
DUMMY = 9


class _Backend:
    def __init__(self, decline=None):
        self.masks = []
        self._decline = decline

    def set_pad_mask(self, mask):
        self.masks.append(mask)

    def pad_mask_decline_reason(self):
        return self._decline


class _Experts(torch.nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def forward(self, hidden_states, router_logits, **kwargs):
        return hidden_states


def _layer(backend):
    return SimpleNamespace(block_sparse_moe=SimpleNamespace(experts=_Experts(backend)))


def _metadata(slots, dummy=DUMMY, num_contexts=0):
    spec = SimpleNamespace(
        batch_slot_ids=torch.tensor(slots), dummy_slot_row=dummy, is_ragged_verify=False
    )
    return SimpleNamespace(num_contexts=num_contexts, num_seqs=len(slots)), spec


def test_the_mask_marks_the_padding_requests_rows():
    state = pad_mask.PadMask()
    attn, spec = _metadata([3, DUMMY, DUMMY, 1])
    state.begin(attn, spec, 4 * ROWS)
    expected = torch.tensor([0, 1, 1, 0], dtype=torch.uint8).repeat_interleave(ROWS)
    assert state.mask.dtype == torch.uint8
    assert torch.equal(state.mask, expected)
    state.end()
    assert state.mask is None


@pytest.mark.parametrize("case", ["context", "uneven", "unpublished", "no_spec", "no_attn"])
def test_no_mask_outside_a_uniform_generation_step(case):
    attn, spec = _metadata(
        [3, DUMMY, DUMMY, 1],
        dummy=0 if case == "unpublished" else DUMMY,
        num_contexts=1 if case == "context" else 0,
    )
    state = pad_mask.PadMask()
    state.begin(
        None if case == "no_attn" else attn,
        None if case == "no_spec" else spec,
        4 * ROWS + (1 if case == "uneven" else 0),
    )
    assert state.mask is None


def test_a_launch_of_the_mask_rows_takes_it_and_others_take_none():
    backend = _Backend()
    layer = _layer(backend)
    state = pad_mask.PadMask()
    assert pad_mask.install([layer], state) == (1, None)
    attn, spec = _metadata([DUMMY, 2])
    state.begin(attn, spec, 2 * ROWS)
    mask = state.mask
    experts = layer.block_sparse_moe.experts
    experts(torch.zeros(2 * ROWS, 4), None)
    experts(torch.zeros(2 * ROWS + 1, 4), None)
    state.end()
    experts(torch.zeros(2 * ROWS, 4), None)
    assert backend.masks[0] is mask
    assert backend.masks[1:] == [None, None]


def _armed(slots, lc2_layers=2):
    state = pad_mask.PadMask()
    state.lc2_layers = lc2_layers
    attn, spec = _metadata(slots)
    state.begin(attn, spec, len(slots) * ROWS)
    return state


def test_lc2_reads_zero_kv_for_the_padding_requests_once_per_forward():
    state = _armed([3, DUMMY, DUMMY, 1])
    seq_lens = torch.tensor([100, 8, 8, 50], dtype=torch.int32)
    lens = state.lc2_seq_lens(seq_lens)
    assert lens.dtype == torch.int32 and lens.is_contiguous()
    assert lens.tolist() == [100, 0, 0, 50]
    assert seq_lens.tolist() == [100, 8, 8, 50]
    assert state.lc2_seq_lens(seq_lens) is lens
    state.end()
    assert state.lc2_seq_lens(seq_lens) is seq_lens


def test_a_taken_lc2_call_reads_the_masked_lengths(monkeypatch):
    calls = []
    monkeypatch.setattr(lc2_decode, "lc2_op", lambda: lambda *args: calls.append(args))
    monkeypatch.setattr(lc2_decode, "decline_reason", lambda *args: None)
    monkeypatch.setattr(pad_mask, "LC2_PAD_SKIP", True)
    monkeypatch.setattr(lc2_decode, "_PAD_ROWS", None)
    state = pad_mask.PadMask()
    pad_mask.arm_lc2(state, 1, 1)
    attn, spec = _metadata([DUMMY, 2])
    state.begin(attn, spec, 2 * ROWS)
    q = torch.zeros((2 * ROWS, 64, 128), dtype=torch.float8_e4m3fn)
    out = torch.zeros((2 * ROWS, 64, 128), dtype=torch.bfloat16)
    seq_lens = torch.tensor([8, 300], dtype=torch.int32)
    lc2_decode.dep_lc2_decode(
        q, None, None, None, None, seq_lens, sm_scale=1.0, output=out, decode_query_len=8
    )
    assert calls[0][9].tolist() == [0, 300]
