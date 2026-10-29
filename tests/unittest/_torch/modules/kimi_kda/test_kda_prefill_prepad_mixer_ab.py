# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixer-level A/B: forward_prefill with 3(b) pre-padding must be bit-identical to baseline."""

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.modules.kimi_kda import kimi_kda_mixer  # noqa: E402
from tests.unittest._torch.modules.kimi_kda.test_kimi_kda_bf16_state_pool import (  # noqa: E402
    CONV_WIDTH,
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_HEADS,
    _make_runtime,
    _prefill_metadata,
)


@pytest.mark.parametrize(
    "sequence_lengths,expect_prepad",
    [
        ([1], True),
        ([17], True),
        ([63], True),
        ([64], False),
        ([65], True),
        ([1000], True),
        ([4097], True),
        ([17, 31], False),
    ],
)
@pytest.mark.parametrize("has_initial", [False, True])
@pytest.mark.parametrize("pool_dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_prepad_matches_baseline(
    sequence_lengths, expect_prepad, has_initial, pool_dtype, monkeypatch
):
    torch.manual_seed(0)
    runtime = _make_runtime()
    assert runtime._dispatch.prefill_kernel_path == "optimized"
    n = len(sequence_lengths)
    slots = n + 3
    dim = NUM_HEADS * HEAD_DIM
    slot_indices = torch.arange(2, 2 + n, device="cuda", dtype=torch.int32)
    cu = torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()], device="cuda", dtype=torch.long
    )
    meta = _prefill_metadata(sequence_lengths, has_initial, [has_initial] * n, cu, slot_indices)
    hidden = torch.randn(sum(sequence_lengths), HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    hidden *= 0.05
    conv_seed = torch.randn(slots, 3 * dim, CONV_WIDTH - 1, dtype=torch.bfloat16, device="cuda")
    conv_seed *= 0.02
    ssm_seed = (torch.randn(slots, NUM_HEADS, HEAD_DIM, HEAD_DIM, device="cuda") * 0.01).to(
        pool_dtype
    )

    padded_calls = []
    post_conv = kimi_kda_mixer.fused_kda_post_conv

    def _record(*args, **kwargs):
        padded_calls.append(kwargs.get("padded_tokens"))
        return post_conv(*args, **kwargs)

    def _strip_prepad(*args, **kwargs):
        # Baseline shim: ignore the mixer's pre-padding request so the
        # dispatch falls back to its original F.pad path, exactly as before
        # the pre-padding fusion existed.
        padded_calls.append(kwargs.get("padded_tokens"))
        beta = kwargs.pop("beta", None)
        kwargs.pop("padded_tokens", None)
        out = post_conv(*args, **kwargs)
        return out if beta is None else (*out, beta.float())

    results = {}
    for prepad in (False, True):
        monkeypatch.setattr(
            kimi_kda_mixer, "fused_kda_post_conv", _record if prepad else _strip_prepad
        )
        conv, ssm = conv_seed.clone(), ssm_seed.clone()
        out = runtime.forward_prefill(hidden, cu, meta, n, conv, ssm, slot_indices)
        results[prepad] = (out.clone(), conv, ssm)

    assert (padded_calls[0] is not None) == expect_prepad, padded_calls
    assert padded_calls[0] == padded_calls[1]
    for a, b in zip(results[False], results[True]):
        assert a.shape == b.shape
        assert torch.equal(
            a.view(torch.int16) if a.element_size() == 2 else a.view(torch.int32),
            b.view(torch.int16) if b.element_size() == 2 else b.view(torch.int32),
        )
