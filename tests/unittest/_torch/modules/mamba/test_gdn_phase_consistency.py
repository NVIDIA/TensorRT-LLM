# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare whole prefill with prefill followed by real one-token updates."""

import pytest
import torch
from test_gdn_mixed_decode import _make_layer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("split", [0, 1, 63, 64, 127, 128])
@pytest.mark.parametrize("initial_history", [False, True])
def test_gdn_prefill_decode_state_continuity(monkeypatch, split, initial_history):
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    torch.manual_seed(413)
    layer = _make_layer()
    length, slots = 129, 5
    projected = torch.randn(length, 3072, device="cuda", dtype=torch.bfloat16) * 0.1
    gates = torch.randn(length, 16, device="cuda", dtype=torch.bfloat16) * 0.1
    conv = torch.randn(slots, 2048, 3, device="cuda", dtype=torch.bfloat16) * 0.1
    state = torch.randn(slots, 8, 128, 128, device="cuda", dtype=torch.float32) * 0.1
    if not initial_history:
        conv[3].zero_()
        state[3].zero_()
    index = torch.tensor([3], device="cuda", dtype=torch.int32)

    def prefill(n, conv_pool, state_pool):
        cu = torch.tensor([0, n], device="cuda", dtype=torch.int32)
        return layer.forward_extend(
            conv_pool,
            state_pool,
            mixed_qkv=projected[:n, :2048].clone(),
            a=gates[:n, 8:],
            b=gates[:n, :8],
            batch_size=1,
            has_initial_states=torch.tensor([initial_history], device="cuda"),
            cache_indices=index,
            query_start_loc=cu,
            query_start_loc_long=cu.long(),
            num_prefill_tokens=n,
            num_decode_tokens=0,
            state_indices_p=index,
            state_indices_d=index[:0],
            num_prefill=1,
            num_decodes=0,
        )

    full_conv, full_state = conv.clone(), state.clone()
    full = prefill(length, full_conv, full_state)
    split_conv, split_state = conv.clone(), state.clone()
    pieces = [prefill(split, split_conv, split_state)] if split else []
    for token in range(split, length):
        pieces.append(
            layer.forward_decode(
                split_conv,
                split_state,
                mixed_qkv=projected[token : token + 1, :2048].clone(),
                a=gates[token : token + 1, 8:],
                b=gates[token : token + 1, :8],
                cache_indices=index,
                num_decodes=1,
                query_start_loc_long=torch.tensor([0, 1], device="cuda", dtype=torch.int64),
            )
        )
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=0, atol=0)
    torch.testing.assert_close(split_conv, full_conv, rtol=0, atol=0)
    torch.testing.assert_close(split_state, full_state, rtol=0, atol=0)
    torch.testing.assert_close(split_state[[0, 1, 2, 4]], state[[0, 1, 2, 4]], rtol=0, atol=0)
