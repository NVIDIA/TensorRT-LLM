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

from __future__ import annotations

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata


class KDAMetadata(Mamba2Metadata):
    """Mamba metadata extended with KDA replay generation indices."""

    def __init__(self, max_batch_size: int, chunk_size: int):
        super().__init__(max_batch_size, chunk_size)
        # KDA's CuTe verify kernel requires a 16-byte-aligned DLPack pointer.
        # The generation slice may follow context rows at an unaligned offset,
        # so keep one stable aligned buffer for CUDA graph capture and replay.
        self.generation_state_indices = torch.zeros(
            max_batch_size, dtype=torch.int32, device="cuda"
        )

    def prepare(self, attn_metadata: AttentionMetadata):
        super().prepare(attn_metadata)

        batch_size = attn_metadata.seq_lens.shape[0]
        num_contexts = attn_metadata.num_contexts
        num_generations = batch_size - num_contexts
        if num_generations > 0:
            self.generation_state_indices[:num_generations].copy_(
                self.state_indices[num_contexts:batch_size]
            )
