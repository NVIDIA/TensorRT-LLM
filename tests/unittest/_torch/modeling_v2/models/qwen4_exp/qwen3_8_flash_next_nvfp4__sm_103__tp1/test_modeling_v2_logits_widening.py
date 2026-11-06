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


@pytest.mark.parametrize("rows", [1, 4, 256])
def test_unwidened_logits_give_the_same_greedy_tokens(head, rows) -> None:
    """`widen=False` returns the head's own bf16 logits, whose greedy samplers (the verify
    sampler's `greedy_argmax`, the draft sampler's int32 one) pick the widened logits' tokens."""
    from tensorrt_llm._torch.speculative.greedy_argmax import greedy_argmax

    g = torch.Generator(device="cuda").manual_seed(rows)
    hidden = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    proc = T._WideningLogitsProcessor()
    with torch.no_grad():
        wide = proc.forward(hidden, head, None, True)
        narrow = proc.forward(hidden, head, None, True, widen=False)
        # Without return_context_logits the shell's gather and fp32 contract stay.
        gathered = proc.forward(hidden, head, None, False, widen=False)
    assert narrow.dtype == torch.bfloat16 and wide.dtype == torch.float32
    assert gathered.dtype == torch.float32
    assert torch.equal(narrow.float().view(torch.int32), wide.view(torch.int32))
    narrow[0, 17] = narrow[0].max()  # an exact tie, broken to the lower index either way
    wide = narrow.float()
    assert torch.equal(greedy_argmax(narrow), greedy_argmax(wide))
    assert torch.equal(greedy_argmax(narrow, torch.int32), greedy_argmax(wide, torch.int32))
    assert torch.equal(greedy_argmax(narrow), torch.argmax(wide, dim=-1))


def _gate(guided=None, greedy=True, rejection=False, penalty=False, relaxed=False):
    worker = SimpleNamespace(
        guided_decoder=guided,
        spec_config=SimpleNamespace(use_relaxed_acceptance_for_thinking=relaxed),
    )
    md = SimpleNamespace(
        is_all_greedy_sample=greedy,
        wants_advanced_draft_sampling=rejection and not greedy,
        enable_penalty=penalty,
    )
    return T._spec_argmax_only(worker, md)


def test_argmax_only_gate() -> None:
    assert _gate() == (True, True)
    # A non-greedy batch: the verify sampler samples; the draft stays argmax unless
    # rejection sampling needs its distribution.
    assert _gate(greedy=False) == (False, True)
    assert _gate(greedy=False, rejection=True) == (False, False)
    # The occurrence penalties and relaxed acceptance read the verify logits only.
    assert _gate(penalty=True) == (False, True)
    assert _gate(relaxed=True) == (False, True)
    # A guided decoder edits both.
    assert _gate(guided=object()) == (False, False)
