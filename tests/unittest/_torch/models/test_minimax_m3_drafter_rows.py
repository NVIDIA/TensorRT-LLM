# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of the drafter's live-row selection and unused-row output contract."""

from unittest.mock import Mock, create_autospec

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.models.modeling_minimaxm3 import (
    Eagle3DecoderLayer,
    _drafter_live_rows,
    _rows_only,
    mark_drafter_dead_rows,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "lengths,contexts,rows,expected",
    [
        ([], 0, 0, None),
        ([1, 1], 0, 2, None),
        ([1, 1], 2, 2, None),
        ([5, 3, 1, 1], 2, 10, [4, 7, 8, 9]),
        ([3, 2], 2, 5, [2, 4]),
        ([4], 1, 4, [3]),
    ],
)
def test_drafter_live_rows_selects_context_tails_and_generation(lengths, contexts, rows, expected):
    metadata = AttentionMetadata(
        max_num_requests=max(1, len(lengths)), max_num_tokens=rows, seq_lens=None, num_contexts=0
    )
    # The selection is device independent; real metadata tensor fields are supplied on CPU here.
    metadata._num_contexts = contexts
    metadata._num_ctx_tokens = sum(lengths[:contexts])
    metadata._seq_lens_cuda = torch.tensor(lengths, dtype=torch.int32)
    selected = _drafter_live_rows(metadata, rows)
    if expected is None:
        assert selected is None
    else:
        assert selected.dtype == torch.long
        assert selected.tolist() == expected


def test_rows_only_computes_live_rows_and_restores_output_positions(monkeypatch):
    module = torch.nn.Linear(4, 3)
    with torch.no_grad():
        module.weight.copy_(torch.arange(12).reshape(3, 4) / 16)
        module.bias.copy_(torch.tensor([0.25, -0.5, 1.0]))
    x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    reference = module(x)
    call = Mock(spec_set=module.forward, wraps=module.forward)
    monkeypatch.setattr(module, "forward", call)
    live = [torch.tensor([4, 1], dtype=torch.long)]
    _rows_only(module, live)
    actual = module(x)
    torch.testing.assert_close(call.call_args.args[0], x[[4, 1]], rtol=0, atol=0)
    torch.testing.assert_close(actual[[4, 1]], reference[[4, 1]], rtol=0, atol=0)
    assert torch.count_nonzero(actual[[0, 2, 3, 5]]) == 0
    live[0] = None
    torch.testing.assert_close(module(x), reference, rtol=0, atol=0)
    assert call.call_args.args[0] is x
    live[0] = torch.empty(0, dtype=torch.long)
    assert torch.count_nonzero(module(x)) == 0
    assert call.call_args.args[0].shape == (0, 4)


@pytest.mark.parametrize("layers", [0, 1, 2])
def test_mark_drafter_dead_rows_restores_full_rows_after_exception(layers: int) -> None:
    draft = torch.nn.Module()
    decoders = torch.nn.ModuleList()
    for _ in range(layers):
        layer = Eagle3DecoderLayer.__new__(Eagle3DecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.self_attn = torch.nn.Module()
        layer.self_attn.o_proj = torch.nn.Identity()
        layer.mlp = torch.nn.Identity()
        decoders.append(layer)
    draft.layers = decoders
    hidden = torch.arange(20, dtype=torch.float32).view(5, 4)
    metadata = AttentionMetadata(
        max_num_requests=2, max_num_tokens=5, seq_lens=None, num_contexts=0
    )
    metadata._num_contexts = 1
    metadata._num_ctx_tokens = 4
    metadata._seq_lens_cuda = torch.tensor([4, 1], dtype=torch.int32)
    if layers != 1:
        originals = [layer.forward for layer in decoders]
        assert mark_drafter_dead_rows(draft) == 0
        assert [layer.forward for layer in decoders] == originals
        return
    layer = decoders[0]

    def run(
        position_ids: None,
        embeds: None,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return layer.mlp(layer.self_attn.o_proj(hidden_states)), hidden_states

    call = create_autospec(layer.forward, side_effect=run, spec_set=True)
    layer.forward = call
    assert mark_drafter_dead_rows(draft) == 1
    result, residual = layer(None, None, hidden, metadata, None)
    assert residual is hidden
    torch.testing.assert_close(result[3:], hidden[3:], rtol=0, atol=0)
    assert torch.count_nonzero(result[:3]) == 0
    call.side_effect = RuntimeError("controlled drafter failure")
    with pytest.raises(RuntimeError, match="controlled drafter failure"):
        layer(None, None, hidden, metadata, None)
    # The wrapper must clear shared row state even when the layer raises.
    torch.testing.assert_close(layer.mlp(hidden), hidden, rtol=0, atol=0)
    torch.testing.assert_close(layer.self_attn.o_proj(hidden), hidden, rtol=0, atol=0)
    call.side_effect = run
    metadata._num_contexts = 0
    metadata._num_ctx_tokens = 0
    result, residual = layer(None, None, hidden, metadata, None)
    assert residual is hidden
    torch.testing.assert_close(result, hidden, rtol=0, atol=0)
