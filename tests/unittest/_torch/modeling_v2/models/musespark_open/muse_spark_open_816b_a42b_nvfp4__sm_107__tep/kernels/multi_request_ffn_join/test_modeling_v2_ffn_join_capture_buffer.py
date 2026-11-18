# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The capture the ``multi_request_ffn_join`` kernel stores, on one SM107 GPU.

A wide generation server runs this lever over either drafter context arena (``levers.
_wide_generation_refusal``): the kernel stores the DFlash capture into the capture view
``capture_fold`` hands it, a column block of the speculative metadata's capture buffer
(``[max_num_tokens, capture layers * 8192]`` BF16), and the drafter's context is written from that
buffer afterwards, into whichever arena the worker keeps. The tests pin, at the decode rows above
the 512-row cap up to 768 (96 requests at seven draft tokens):

* the capture view holds, bit for bit, the BF16 round-to-nearest-even of the FP32 residual the
  join returns (the bytes the built-in capture's cast writes);
* nothing else of the buffer is written: the other capture layers' column blocks and the rows past
  the call keep their poison.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.multi_request_ffn_join import (  # noqa: E501
    join as ffn_join,
)

HIDDEN = ffn_join.HIDDEN
#: The decode rows of the generation shape and the capture layers of the served drafter.
MAX_NUM_TOKENS = 768
CAPTURE_LAYERS = 5
#: A BF16 NaN bit pattern no kernel output takes here.
POISON = 0x7FC1


def _supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(
    not _supported(), reason="the multi-request post-FFN join kernel runs on SM107"
)


class _NextNorm:
    """The next layer's Gemma RMSNorm, as the join reads it."""

    use_gemma = True

    def __init__(self, weight: torch.Tensor, eps: float) -> None:
        self.weight = weight
        self.variance_epsilon = eps


class _Join:
    """The post-FFN join the kernel serves, as the kernel reads it."""

    _fused_enabled = True
    _preqkv_norm_fp32 = False

    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor, capture: torch.Tensor) -> None:
        self._alpha = alpha
        self._beta = beta
        self._capture_fold = capture
        self._capture_stored = False

    def coefficients(self):
        return self._alpha, self._beta


@pytest.mark.parametrize("rows", [520, 744, 768])
@pytest.mark.parametrize("index", [0, CAPTURE_LAYERS - 1])
def test_the_capture_lands_in_its_column_block_only(rows, index):
    torch.manual_seed(rows * 8 + index)
    device = torch.device("cuda")
    residual = torch.randn(rows, HIDDEN, device=device, dtype=torch.float32)
    branch = torch.randn(rows, HIDDEN, device=device).to(torch.bfloat16)
    gain = (torch.rand(HIDDEN, device=device) * 0.5).to(torch.bfloat16)
    alpha = torch.rand(HIDDEN, device=device, dtype=torch.float32)
    beta = torch.rand(HIDDEN, device=device, dtype=torch.float32)
    next_norm = _NextNorm((torch.randn(HIDDEN, device=device) * 0.1).to(torch.bfloat16), 1e-6)
    buffer = torch.full(
        (MAX_NUM_TOKENS, CAPTURE_LAYERS * HIDDEN), POISON, dtype=torch.int16, device=device
    ).view(torch.bfloat16)
    columns = slice(index * HIDDEN, (index + 1) * HIDDEN)
    capture = buffer[:rows, columns]
    owner = _Join(alpha, beta, capture)
    op = ffn_join.MultiRequestFfnJoin(capture_only=True, pdl=False)

    joined = op.join(owner, residual, branch, 1e-6, next_norm, gain)
    torch.cuda.synchronize()

    assert joined is not None, "the kernel declined the call"
    updated, _ = joined
    assert owner._capture_stored
    expected = updated.to(torch.bfloat16)
    assert torch.equal(capture.view(torch.int16), expected.view(torch.int16))
    written = torch.zeros_like(buffer, dtype=torch.bool)
    written[:rows, columns] = True
    untouched = buffer.view(torch.int16)[~written]
    assert bool((untouched == POISON).all())
