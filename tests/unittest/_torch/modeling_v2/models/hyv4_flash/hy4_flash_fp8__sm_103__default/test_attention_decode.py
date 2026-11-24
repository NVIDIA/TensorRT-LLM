# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a layer's decode hands the MLA generation entry (CPU only)."""

import torch
from _target import Step, core, mla_layer, requires_checkpoint

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.dims import (  # noqa: E501
    dims,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.modeling import (  # noqa: E501
    _recent_rows,
)


def test_recent_rows_are_a_valid_prefix_followed_by_padding():
    rows = _recent_rows(torch.tensor([0, 1, 2, 7], dtype=torch.int64), 4)
    assert rows.dtype == torch.int32
    assert rows.tolist() == [
        [0, -1, -1, -1],
        [1, 0, -1, -1],
        [2, 1, 0, -1],
        [7, 6, 5, 4],
    ]


@requires_checkpoint
def test_a_generation_step_hands_the_entry_a_bound_that_does_not_move():
    """A captured graph freezes the host KV length, so it must not depend on the step."""
    d = dims(core())
    max_seq_len = 5120
    rope = {"rotary_cos_sin": None}
    sparse = mla_layer(d.sparse_layers[2])
    windowed = mla_layer(next(i for i in range(len(d.layers)) if i not in d.sparse_layers))
    selection = torch.zeros((1, d.index_topk), dtype=torch.int32)

    for kv in (6, max_seq_len - 1):
        md = Step(kv, max_seq_len)
        sparse_bound = sparse._generation_args(md, rope, selection)["host_past_key_value_lengths"]
        window_bound = windowed._generation_args(md, rope, None)["host_past_key_value_lengths"]
        assert sparse_bound.dtype == md.kv_lens_runtime.dtype
        assert sparse_bound.tolist() == [d.index_topk]
        assert window_bound.tolist() == [max_seq_len]
        assert window_bound.item() > windowed.geom.window
