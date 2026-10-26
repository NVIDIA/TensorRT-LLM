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
import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.embedding import pre_comm_embedding_ops
from tensorrt_llm._torch.modules.linear import TensorParallelMode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("tp_size", [1, 8])
@pytest.mark.parametrize("tokens", [6, 12, 33])
@pytest.mark.parametrize("hidden", [512, 1288, 5120])
@torch.inference_mode()
def test_vocab_slice_lookup_sums_to_full_embedding(tp_size, tokens, hidden):
    """Every rank's slice lookup is exact and the slices sum to the full-table lookup."""
    torch.manual_seed(0)
    vocab = 1000
    table = torch.randn(vocab, hidden, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, vocab, (tokens,), device="cuda")
    ids[0] = -1  # A padded rejected draft token.
    slice_width = -(-vocab // tp_size)
    total = torch.zeros(tokens, hidden, dtype=torch.float32, device="cuda")
    for rank in range(tp_size):
        start, end = rank * slice_width, min((rank + 1) * slice_width, vocab)
        rows = table[start:end].contiguous()
        out = pre_comm_embedding_ops(
            ids, rows, tp_size, rank, TensorParallelMode.COLUMN, start, end, False, 0
        )
        local = ids - start
        inside = (local >= 0) & (local < end - start)
        expected = F.embedding(torch.where(inside, local, 0), rows)
        expected[~inside] = 0
        assert torch.equal(out, expected)
        total += out.float()
    expected_total = F.embedding(ids.clamp(min=0), table).float()
    expected_total[0] = 0
    assert torch.equal(total, expected_total)
