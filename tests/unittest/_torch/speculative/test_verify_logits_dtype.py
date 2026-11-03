# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Verify logits kept in the LM head's dtype sample as their fp32 copy does."""

import types

import pytest
import torch

from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

VOCAB = 129280
DISABLE_TOPK = torch.iinfo(torch.int32).max

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class _StubSpecWorker(SpecWorkerBase):
    @property
    def max_draft_len(self) -> int:
        return 3

    def _forward_impl(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError


def _metadata(rows: int, *, greedy: bool, offset_base: int = 0):
    return types.SimpleNamespace(
        is_all_greedy_sample=greedy,
        advanced_sampling_mode=AdvancedSamplingMode.FULL,
        temperatures=torch.ones(rows, dtype=torch.float32, device="cuda"),
        top_ks=torch.full((rows,), DISABLE_TOPK, dtype=torch.int32, device="cuda"),
        top_ps=torch.ones(rows, dtype=torch.float32, device="cuda"),
        min_ps=torch.zeros(rows, dtype=torch.float32, device="cuda"),
        seeds=torch.full((rows,), 1234, dtype=torch.int64, device="cuda"),
        offsets=torch.arange(rows, dtype=torch.int64, device="cuda") + offset_base,
    )


def _peaked_logits(rows: int, dtype: torch.dtype, *, ties: bool = False) -> torch.Tensor:
    """bf16-representable rows whose mass sits on a few dozen tokens, as decode logits do."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn(rows, VOCAB, device="cuda", generator=gen) * 2.0
    hot = torch.randint(0, VOCAB, (32,), device="cuda", generator=gen)
    logits[:, hot] += torch.linspace(8.0, 12.0, 32, device="cuda")
    if ties:
        # Coarse values make equal maxima common; argmax must keep the lowest index.
        logits = (logits * 2).round() / 2
    return logits.to(dtype)


@pytest.mark.parametrize("rows", [6, 1024])
def test_greedy_tokens_match_the_fp32_copy(rows: int) -> None:
    worker = _StubSpecWorker()
    logits = _peaked_logits(rows, torch.bfloat16, ties=True)
    meta = _metadata(rows, greedy=True)
    kept = worker._sample_tokens_for_batch(logits, meta, 0, rows)
    widened = worker._sample_tokens_for_batch(logits.float(), meta, 0, rows)
    assert torch.equal(kept, widened)


def test_sampled_tokens_follow_the_same_distribution() -> None:
    """Draws from bf16 logits and from their fp32 copy both match the exact softmax.

    The fused op's reduction order depends on the dtype, so tokens may differ per draw.
    """
    rows, calls = 4096, 16
    worker = _StubSpecWorker()
    row = _peaked_logits(1, torch.bfloat16)
    logits = row.expand(rows, VOCAB).contiguous()
    exact = torch.softmax(row.double(), dim=-1)[0]
    support = torch.topk(exact, 32).indices
    expected = torch.cat([exact[support], (1 - exact[support].sum()).unsqueeze(0)])

    def histogram(sample_logits: torch.Tensor) -> torch.Tensor:
        counts = torch.zeros(VOCAB, dtype=torch.float64, device="cuda")
        for call in range(calls):
            meta = _metadata(rows, greedy=False, offset_base=call * rows)
            tokens = worker._sample_tokens_for_batch(sample_logits, meta, 0, rows)
            counts += torch.bincount(tokens.long(), minlength=VOCAB).double()
        counts /= rows * calls
        return torch.cat([counts[support], (1 - counts[support].sum()).unsqueeze(0)])

    for sample_logits in (logits, logits.float()):
        tv = 0.5 * (histogram(sample_logits) - expected).abs().sum().item()
        assert tv < 0.03, f"{sample_logits.dtype}: total-variation distance {tv:.4f}"
