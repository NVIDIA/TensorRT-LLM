# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's logits processor returns the shell's logits bit for bit: the same row gather,
the same fp32 values, its head's bf16 logits widened by one vectorized kernel."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as T,
)
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor

assert torch.cuda.is_available(), "the logits processor test requires a CUDA device"

VOCAB, HIDDEN = 248320, 2560


@pytest.fixture(scope="module")
def head():
    g = torch.Generator(device="cuda").manual_seed(0)
    lin = torch.nn.Linear(HIDDEN, VOCAB, bias=False, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(VOCAB, HIDDEN, generator=g, device="cuda") * 0.02)
    return lin


@pytest.mark.parametrize("seq_lens", [[1] * 128, [5000, 3000, 1] + [1] * 121])
def test_matches_the_shell_processor_bitwise(head, seq_lens) -> None:
    g = torch.Generator(device="cuda").manual_seed(len(seq_lens))
    lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    hidden = torch.randn(int(lens.sum()), HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    md = SimpleNamespace(seq_lens_cuda=lens)
    with torch.no_grad():
        want = LogitsProcessor().forward(hidden, head, md)
        got = T._WideningLogitsProcessor().forward(hidden, head, md)
    assert got.dtype == torch.float32 and got.shape == (len(seq_lens), VOCAB)
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))


def test_context_logits_and_no_metadata(head) -> None:
    g = torch.Generator(device="cuda").manual_seed(9)
    hidden = torch.randn(33, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    with torch.no_grad():
        for md, ctx in ((None, False), (SimpleNamespace(seq_lens_cuda=None), True)):
            want = LogitsProcessor().forward(hidden, head, md, ctx)
            got = T._WideningLogitsProcessor().forward(hidden, head, md, ctx)
            assert torch.equal(got.view(torch.int32), want.view(torch.int32))
