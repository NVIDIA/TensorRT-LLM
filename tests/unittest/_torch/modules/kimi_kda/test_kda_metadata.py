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

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_kda.kda_metadata import KDAMetadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prepare_materializes_aligned_generation_state_indices():
    class KdaCacheManager:
        def __init__(self) -> None:
            self.state_indices = torch.tensor([9, 4, 7], dtype=torch.int32, device="cuda")

        def get_state_indices(self, request_ids, is_padding):
            return self.state_indices[: len(request_ids)]

    manager = KdaCacheManager()
    metadata = KDAMetadata(max_batch_size=3, chunk_size=8)
    seq_lens = torch.tensor([2, 1, 1], dtype=torch.int)
    attn_metadata = SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens.cuda(),
        num_contexts=1,
        num_ctx_tokens=2,
        kv_cache_manager=manager,
        request_ids=[10, 11, 12],
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=torch.tensor([0], dtype=torch.int),
        ),
    )

    metadata.prepare(attn_metadata)

    assert metadata.state_indices[1:].data_ptr() % 16 != 0
    assert metadata.generation_state_indices.data_ptr() % 16 == 0
    torch.testing.assert_close(
        metadata.generation_state_indices[:2],
        torch.tensor([4, 7], dtype=torch.int32, device="cuda"),
    )

    aligned_ptr = metadata.generation_state_indices.data_ptr()
    manager.state_indices.copy_(torch.tensor([8, 3, 6], dtype=torch.int32, device="cuda"))
    metadata.prepare(attn_metadata)

    assert metadata.generation_state_indices.data_ptr() == aligned_ptr
    torch.testing.assert_close(
        metadata.generation_state_indices[:2],
        torch.tensor([3, 6], dtype=torch.int32, device="cuda"),
    )
