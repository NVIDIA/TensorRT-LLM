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
"""The direct FlashInfer GDN launch casts the metadata's int64 cu_seqlens to
int32 once per iteration: the cast is cached per buffer until
``Mamba2Metadata.prepare`` invalidates it, and inference-mode tensors (no
version counter) are handled."""
import pytest
import torch

from tensorrt_llm._torch.modules.fla import flashinfer_chunk as fc

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _clean_cache():
    fc.invalidate_int32_cu_seqlens_cache()
    yield
    fc.invalidate_int32_cu_seqlens_cache()


def test_cast_is_shared_within_an_iteration_and_dropped_on_invalidate():
    buf = torch.zeros(9, dtype=torch.int64)
    buf[:4].copy_(torch.tensor([0, 5, 9, 12]))
    view = buf[:4]
    first = fc._int32_cu_seqlens(view)
    assert first.dtype == torch.int32 and first.tolist() == [0, 5, 9, 12]
    # Same buffer, same length -> the same cast object (no new copy kernel).
    assert fc._int32_cu_seqlens(buf[:4]) is first
    # A different length of the same buffer is a different key.
    assert fc._int32_cu_seqlens(buf[:3]) is not first
    # The metadata rewrites the buffer and invalidates: the next call recasts.
    buf[:4].copy_(torch.tensor([0, 7, 8, 20]))
    fc.invalidate_int32_cu_seqlens_cache()
    second = fc._int32_cu_seqlens(buf[:4])
    assert second is not first and second.tolist() == [0, 7, 8, 20]


def test_int32_input_passes_through_and_inference_mode_tensors_work():
    already = torch.tensor([0, 3], dtype=torch.int32)
    assert fc._int32_cu_seqlens(already) is already
    with torch.inference_mode():
        buf = torch.tensor([0, 2, 6], dtype=torch.int64)
        out = fc._int32_cu_seqlens(buf)
        assert out.dtype == torch.int32 and out.tolist() == [0, 2, 6]
        assert fc._int32_cu_seqlens(buf) is out


def test_cache_is_bounded():
    for i in range(fc._INT32_CU_SEQLENS_CACHE_MAX + 3):
        fc._int32_cu_seqlens(torch.arange(i + 2, dtype=torch.int64))
    assert len(fc._int32_cu_seqlens_cache) <= fc._INT32_CU_SEQLENS_CACHE_MAX


def test_recycled_address_of_a_freed_buffer_does_not_alias():
    """A cast for an ad-hoc buffer must not be handed to a later buffer that
    lands on the same address with the same length (the entry keeps its source
    alive, so the allocator cannot recycle the address while it is cached)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    first = torch.tensor([0, 23], dtype=torch.int64, device=dev)
    assert fc._int32_cu_seqlens(first).tolist() == [0, 23]
    del first
    second = torch.tensor([0, 17], dtype=torch.int64, device=dev)
    assert fc._int32_cu_seqlens(second).tolist() == [0, 17]
