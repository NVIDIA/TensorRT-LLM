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
"""TRTLLM_GREEDY_BF16_LOGITS: the BF16 argmax picks torch.argmax's index over the
FP32 copy of the logits (first maximum, NaN counts as the maximum)."""

import pytest
import torch

from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.pyexecutor.sampler import greedy_bf16

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or greedy_bf16.triton is None, reason="needs CUDA and Triton"
)


@pytest.mark.parametrize("rows", [1, 7, 256, 1024])
@pytest.mark.parametrize("vocab", [1000, 4096, 129280])
def test_bf16_argmax_matches_fp32_argmax(rows, vocab):
    torch.manual_seed(rows + vocab)
    logits = (torch.randn(rows, vocab, device="cuda") * 3).to(torch.bfloat16)
    if rows > 3:
        row_max = logits[3].max()
        logits[3, vocab // 3] = row_max  # tie: the first maximum wins
        logits[3, vocab - 1] = row_max
    if rows > 5:
        logits[5, 7] = float("nan")  # NaN counts as the maximum
        logits[5, 9] = float("nan")
    if rows > 6:
        logits[6] = float("-inf")
    assert greedy_bf16.bf16_argmax_supported(logits)
    expected = torch.argmax(logits.float(), dim=-1)
    assert torch.equal(greedy_bf16.bf16_argmax_last_dim(logits), expected)


def test_bf16_argmax_row_stride():
    logits = torch.randn(64, 2 * 5000, device="cuda").to(torch.bfloat16)
    view = logits[:, :5000]
    assert greedy_bf16.bf16_argmax_supported(view)
    assert torch.equal(greedy_bf16.bf16_argmax_last_dim(view), torch.argmax(view.float(), dim=-1))
    assert not greedy_bf16.bf16_argmax_supported(view.float())
    assert not greedy_bf16.bf16_argmax_supported(logits.t())


def test_logits_processor_keeps_bf16_only_when_asked():
    lm_head = torch.nn.Linear(32, 100, bias=False).cuda().to(torch.bfloat16)
    hidden = torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)
    processor = LogitsProcessor()
    assert processor(hidden, lm_head, None, True).dtype == torch.float32
    processor.keep_bf16_logits = True
    logits = processor(hidden, lm_head, None, True)
    assert logits.dtype == torch.bfloat16
    assert torch.equal(logits.float(), LogitsProcessor()(hidden, lm_head, None, True))
