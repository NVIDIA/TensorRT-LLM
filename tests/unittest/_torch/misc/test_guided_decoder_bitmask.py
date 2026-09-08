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
import math

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.guided_decoder import row_has_valid_token


def _empty_row(vocab_size_padded: int) -> torch.Tensor:
    return torch.zeros(math.ceil(vocab_size_padded / 32), dtype=torch.int32)


# 128000 is word-aligned; 128001 and 128031 leave a partial last word.
@pytest.mark.parametrize("vocab_size_padded", [128000, 128001, 128031])
def test_dead_end_row_has_no_valid_token(vocab_size_padded: int):
    """A grammar state with no valid continuation must be reported as dead."""
    assert not row_has_valid_token(_empty_row(vocab_size_padded), vocab_size_padded)


@pytest.mark.parametrize("vocab_size_padded", [128000, 128001, 128031])
@pytest.mark.parametrize("token_id_from_end", [0, 1, 32, 12345])
def test_single_valid_token_is_detected(vocab_size_padded: int, token_id_from_end: int):
    """A single set bit anywhere below vocab_size_padded keeps the row alive."""
    token_id = vocab_size_padded - 1 - token_id_from_end
    row = _empty_row(vocab_size_padded)
    row[token_id // 32] |= 1 << (token_id % 32)
    assert row_has_valid_token(row, vocab_size_padded)


@pytest.mark.parametrize("vocab_size_padded", [128001, 128031])
def test_trailing_padding_bits_do_not_count(vocab_size_padded: int):
    """Padding bits above vocab_size_padded must not mark a dead row valid.

    The apply kernel never reads them and the backends do not guarantee they
    are cleared, so counting them would let a fully masked row reach the
    sampler and produce a NaN logits row.
    """
    row = _empty_row(vocab_size_padded)
    num_words, num_tail_bits = divmod(vocab_size_padded, 32)
    # Set every bit of the last word that lies at or above vocab_size_padded.
    row[num_words] = torch.tensor(-1 << num_tail_bits, dtype=torch.int32)
    assert not row_has_valid_token(row, vocab_size_padded)

    # The highest in-range bit of that same partial word must still count.
    row[num_words] |= 1 << (num_tail_bits - 1)
    assert row_has_valid_token(row, vocab_size_padded)


def test_sign_bit_counts_as_valid_token():
    """Bit 31 of a word makes the int32 negative; it is still a valid token."""
    vocab_size_padded = 128000
    row = _empty_row(vocab_size_padded)
    row[0] = torch.tensor(-2147483648, dtype=torch.int32)  # only bit 31 set
    assert row_has_valid_token(row, vocab_size_padded)
