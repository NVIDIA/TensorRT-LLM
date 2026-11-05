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

from itertools import accumulate

import pytest
import torch

import tensorrt_llm  # noqa: F401 -- Registers the native merge operator.

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _reference_merge(
    previous: torch.Tensor,
    previous_stats: torch.Tensor,
    current: torch.Tensor,
    current_stats: torch.Tensor,
    offsets: list[int],
    operations: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge normalized partial outputs using float64 log-partition weights."""
    expected = previous.clone()
    expected_stats = previous_stats.clone()
    for request, operation in enumerate(operations):
        rows = slice(offsets[request], offsets[request + 1])
        if operation == 0:
            continue
        if operation == 2:
            expected[rows] = current[rows]
            expected_stats[rows] = current_stats[rows]
            continue

        stats = torch.stack((previous_stats[rows], current_stats[rows]), dim=-1).double()
        maxima, sums = stats.unbind(dim=-2)
        log_partitions = maxima + sums.log()
        weights = log_partitions.softmax(dim=-1)
        partial_outputs = torch.stack((previous[rows], current[rows]), dim=-1).double()
        expected[rows] = (partial_outputs * weights.unsqueeze(-2)).sum(dim=-1).to(previous.dtype)
        merged_max = maxima.amax(dim=-1)
        merged_sum = (log_partitions.logsumexp(dim=-1) - merged_max).exp()
        expected_stats[rows] = torch.stack((merged_max, merged_sum), dim=-1).float()
    return expected, expected_stats


@pytest.mark.parametrize(
    "dtype,head_size",
    [
        (torch.float16, 128),
        (torch.bfloat16, 128),
        (torch.float32, 128),
        (torch.float16, 256),
        (torch.bfloat16, 256),
    ],
)
def test_mla_context_merge_ragged_in_place(dtype: torch.dtype, head_size: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(42)
    lengths = [5, 0, 17, 9, 1]
    offsets = list(accumulate(lengths, initial=0))
    num_heads = 3
    shape = (offsets[-1], num_heads, head_size)
    stats_shape = shape[:2] + (2,)
    cu_q_seq_lens = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    operations_by_partition = ([0, 1, 2, 1, 2], [2, 0, 1, 0, 1], [1, 2, 0, 1, 2])

    # Guard rows detect writes outside the packed query range.
    output_storage = torch.full(
        (shape[0] + 2, num_heads, head_size), -7.0, dtype=dtype, device="cuda"
    )
    stats_storage = torch.full((shape[0] + 2, num_heads, 2), -11.0, device="cuda")
    merged = output_storage[1:-1]
    merged_stats = stats_storage[1:-1]
    merged.copy_(torch.randn(shape, generator=generator, device="cuda"))
    merged_stats[..., 0] = torch.linspace(-80, 80, shape[0], device="cuda").unsqueeze(-1)
    merged_stats[..., 1] = torch.rand(shape[:2], generator=generator, device="cuda") + 1
    for request, operation in enumerate(operations_by_partition[0]):
        if operation == 2:
            rows = slice(offsets[request], offsets[request + 1])
            merged[rows] = torch.nan
            merged_stats[rows] = torch.nan

    tolerances = {torch.float16: 1e-3, torch.bfloat16: 8e-3, torch.float32: 2e-6}
    for partition, operations in enumerate(operations_by_partition):
        current = torch.randn(shape, generator=generator, device="cuda").to(dtype)
        current_stats = torch.empty(stats_shape, device="cuda")
        # Alternating dominant partitions exercise rescaling in both directions.
        current_stats[..., 0] = torch.linspace(80, -80, shape[0], device="cuda").unsqueeze(-1)
        current_stats[..., 0] *= (-1) ** partition
        current_stats[..., 1] = torch.rand(shape[:2], generator=generator, device="cuda") * 7 + 1
        previous, previous_stats = merged.clone(), merged_stats.clone()
        current_before, current_stats_before = current.clone(), current_stats.clone()
        expected, expected_stats = _reference_merge(
            previous, previous_stats, current, current_stats, offsets, operations
        )

        # The native binding aliases both destination tensors with its previous-partition inputs.
        torch.ops.trtllm.merge_chunked_attention_for_mla(
            merged,
            current,
            merged_stats,
            current_stats,
            len(lengths),
            cu_q_seq_lens,
            max(lengths),
            torch.tensor(operations, dtype=torch.int64, device="cuda"),
            num_heads,
            head_size,
        )

        torch.testing.assert_close(merged, expected, atol=tolerances[dtype], rtol=tolerances[dtype])
        torch.testing.assert_close(merged_stats, expected_stats, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(current, current_before, atol=0, rtol=0)
        torch.testing.assert_close(current_stats, current_stats_before, atol=0, rtol=0)
        for request, operation in enumerate(operations):
            rows = slice(offsets[request], offsets[request + 1])
            if operation in (0, 2):
                torch.testing.assert_close(merged[rows], expected[rows], atol=0, rtol=0)
                torch.testing.assert_close(merged_stats[rows], expected_stats[rows], atol=0, rtol=0)
        assert torch.all(output_storage[[0, -1]] == -7)
        assert torch.all(stats_storage[[0, -1]] == -11)


@pytest.mark.parametrize(
    "dtype,head_size", [(torch.float32, 256), (torch.float16, 64), (torch.bfloat16, 192)]
)
def test_mla_context_merge_rejects_unsupported_width(dtype: torch.dtype, head_size: int) -> None:
    output = torch.zeros((1, 1, head_size), dtype=dtype, device="cuda")
    stats = torch.zeros((1, 1, 2), device="cuda")
    with pytest.raises(RuntimeError, match="head dim"):
        torch.ops.trtllm.merge_chunked_attention_for_mla(
            output,
            output.clone(),
            stats,
            stats.clone(),
            1,
            torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
            1,
            torch.tensor([2], dtype=torch.int64, device="cuda"),
            1,
            head_size,
        )
