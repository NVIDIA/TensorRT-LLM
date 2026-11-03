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
"""The column-limited KV-V2 page-table copy matches the full-width CUDA copy on the copied columns."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("num_pools", [1, 2])
@pytest.mark.parametrize("num_cols", [1, 37, 1024, 2048, 3000])
def test_copy_batch_block_offsets_columns(num_pools, num_cols):
    from tensorrt_llm._torch.pyexecutor.kv_cache.block_offsets_copy import (
        copy_batch_block_offsets_columns,
    )
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
        copy_batch_block_offsets_to_device,
    )

    slots, rows, width = 24, 16, 4096
    gen = torch.Generator().manual_seed(num_pools * 7 + num_cols)
    host = torch.randint(
        -1, 1 << 18, (num_pools, slots, 2, width), generator=gen, dtype=torch.int32
    )
    host = host.pin_memory()
    copy_idx = torch.randperm(slots, generator=gen)[:13].to(torch.int32).pin_memory()
    scales = torch.tensor([3, 5][:num_pools], dtype=torch.int32).pin_memory()
    kv_offset = torch.tensor([0, 11][:num_pools], dtype=torch.int32).pin_memory()
    stream = torch.cuda.current_stream()
    ref = torch.full((num_pools, rows, 2, width), -5, dtype=torch.int32, device="cuda")
    copy_batch_block_offsets_to_device(host, ref, copy_idx, scales, kv_offset, stream.cuda_stream)
    got = torch.full_like(ref, -5)
    assert copy_batch_block_offsets_columns(
        host, got, copy_idx, scales, kv_offset, num_cols, stream
    )
    torch.cuda.synchronize()
    n = copy_idx.shape[0]
    assert torch.equal(got[:, :n, :, :num_cols], ref[:, :n, :, :num_cols])
    assert bool((got[:, :, :, num_cols:] == -5).all()) and bool((got[:, n:] == -5).all())


def test_copy_batch_block_offsets_columns_rejects_layouts():
    """Layouts the column copy does not serve launch nothing, so the caller runs the full copy."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.block_offsets_copy import (
        copy_batch_block_offsets_columns,
    )

    host = torch.zeros((1, 8, 2, 64), dtype=torch.int32).pin_memory()
    copy_idx = torch.arange(4, dtype=torch.int32).pin_memory()
    scales = torch.ones(1, dtype=torch.int32).pin_memory()
    kv_offset = torch.zeros(1, dtype=torch.int32).pin_memory()
    stream = torch.cuda.current_stream()
    dst = torch.full((1, 4, 2, 64), -5, dtype=torch.int32, device="cuda")
    unpinned = torch.zeros((1, 8, 2, 64), dtype=torch.int32)
    assert not copy_batch_block_offsets_columns(
        unpinned, dst, copy_idx, scales, kv_offset, 8, stream
    )
    assert not copy_batch_block_offsets_columns(
        host, dst[..., :32], copy_idx, scales, kv_offset, 8, stream
    )
    assert not copy_batch_block_offsets_columns(host, dst, copy_idx, scales, kv_offset, 65, stream)
    torch.cuda.synchronize()
    assert bool((dst == -5).all())
