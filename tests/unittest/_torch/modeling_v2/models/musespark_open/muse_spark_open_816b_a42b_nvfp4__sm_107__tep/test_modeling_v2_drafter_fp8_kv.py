# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sm107 TEP drafter block decode over an FP8 draft KV pool (``TRTLLM_DFLASH_FP8_DRAFT_KV``).

The tests pin, without a GPU:

* ``fast_path_refusal``: BF16 activations over a BF16 or FP8 E4M3 context take the written-out
  block decode; FP16 activations, an FP16 context, K and V of different dtypes, a strided head
  dimension, a block wider than the drafter's and a drafter without the fused QK norm + RoPE
  run the built-in forward;
* the built-in fallback's log line: once per (reason, block rows), so a reason first met while a
  graph is captured is still reported for every captured width;
* the install's geometry check takes both context dtypes for the served drafter's geometry.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.drafter_block_decode import (  # noqa: E501
    block_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.spec.dflash_window_attention import (  # noqa: E501
    geometry_unsupported_reason,
)

FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
BLOCK = 8


def _drafter(fused=True):
    return SimpleNamespace(_use_fused_qk_norm_rope=fused, block_size=BLOCK)


def _context(dtype, strided=False):
    pool = torch.zeros(6, 2, 2, 32, 128).to(dtype)
    k, v = pool[:, 0].transpose(1, 2), pool[:, 1].transpose(1, 2)
    if strided:
        k = torch.zeros(6, 32, 2, 256).to(dtype)[..., ::2]
    return k, v


@pytest.mark.parametrize("dtype", [BF16, FP8])
def test_bf16_activations_over_bf16_or_fp8_context_take_the_fast_path(dtype):
    k, v = _context(dtype)
    noise = torch.zeros(3, BLOCK, 64, dtype=BF16)
    assert block_decode.fast_path_refusal(_drafter(), noise, k, v) is None


@pytest.mark.parametrize(
    "case,match",
    [
        ("fp16_activations", "activations are torch.float16"),
        ("fp16_context", "context K/V are torch.float16"),
        ("mixed_context", "context K/V are torch.float8_e4m3fn/torch.bfloat16"),
        ("strided", "strided"),
        ("wide_block", "wider than"),
        ("unfused", "fused QK norm"),
    ],
)
def test_other_steps_run_the_built_in_forward(case, match):
    noise = torch.zeros(3, BLOCK, 64, dtype=torch.float16 if case == "fp16_activations" else BF16)
    k, v = _context(torch.float16 if case == "fp16_context" else FP8, strided=case == "strided")
    if case == "mixed_context":
        v = _context(BF16)[1]
    if case == "wide_block":
        noise = torch.zeros(3, BLOCK + 1, 64, dtype=BF16)
    drafter = _drafter(fused=case != "unfused")
    reason = block_decode.fast_path_refusal(drafter, noise, k, v)
    assert reason is not None and match in reason, reason


def test_fallback_is_logged_once_per_reason_and_width(monkeypatch):
    messages = []
    monkeypatch.setattr(
        block_decode, "logger", SimpleNamespace(info=lambda message: messages.append(message))
    )
    monkeypatch.setattr(block_decode, "_LOGGED_FALLBACKS", set())
    for reason, rows in [("a", 8), ("a", 8), ("a", 16), ("b", 8), ("a", 16)]:
        block_decode._log_fallback(reason, rows)
    assert messages == [
        "[musespark_open] drafter block decode ran the built-in forward (a) for a 8-row block decode",
        "[musespark_open] drafter block decode ran the built-in forward (a) for a 16-row block decode",
        "[musespark_open] drafter block decode ran the built-in forward (b) for a 8-row block decode",
    ]


@pytest.mark.parametrize("kv_dtype", block_decode.BLOCK_DECODE_KV_DTYPES)
def test_served_geometry_takes_both_context_dtypes(kv_dtype):
    # A TP4 rank of the served drafter: 8 query / 2 KV heads of 128, block 8, window 2047.
    assert geometry_unsupported_reason(BLOCK, 8, 2, 128, BF16, (2047, 2047), kv_dtype) is None
