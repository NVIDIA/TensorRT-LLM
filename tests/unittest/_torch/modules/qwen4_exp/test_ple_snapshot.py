# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.modules.qwen4_exp.ple_snapshot import store_ple_snapshots

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_stores_ple_windows_in_reserved_slots():
    torch.manual_seed(817)
    slots_cpu, points = [4, 1, 7], [64, 128, 192]
    conv_width, history_width, channels = 6, 4, 12
    conv_input = torch.randn(256, channels, device="cuda", dtype=torch.bfloat16)
    input_ids = torch.arange(256, device="cuda", dtype=torch.int64)
    # A strided leading dimension models layer views within coalesced cache slots.
    conv_states = torch.randn(20, channels, conv_width, device="cuda", dtype=torch.bfloat16)[::2]
    ngram = torch.full((20, history_width), -1, device="cuda", dtype=torch.int64)[::2]
    original_conv, original_ngram = conv_states.clone(), ngram.clone()
    slots = torch.tensor(slots_cpu, device="cuda", dtype=torch.int32)
    conv_indices = torch.tensor(
        [i for point in points for i in range(point - conv_width, point)],
        device="cuda",
        dtype=torch.int64,
    )
    token_indices = torch.tensor(
        [i for point in points for i in range(point - history_width, point)],
        device="cuda",
        dtype=torch.int64,
    )
    store_ple_snapshots(
        conv_input, input_ids, conv_states, ngram, conv_indices, token_indices, slots
    )

    for slot, point in zip(slots_cpu, points):
        torch.testing.assert_close(
            conv_states[slot], conv_input[point - conv_width : point].T, rtol=0, atol=0
        )
        torch.testing.assert_close(
            ngram[slot], input_ids[point - history_width : point], rtol=0, atol=0
        )
    untouched = [slot for slot in range(10) if slot not in slots_cpu]
    assert torch.equal(conv_states[untouched], original_conv[untouched])
    assert torch.equal(ngram[untouched], original_ngram[untouched])
