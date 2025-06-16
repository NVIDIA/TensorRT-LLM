# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata


class Mamba2Metadata:

    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size

        # cumulative sequence lengths for prefill requests [batch_size+1]
        self.cu_seqlens = torch.zeros(max_batch_size + 1,
                                      dtype=torch.int,
                                      device="cuda")

        # sequence index for prefill requests [num_prefill_tokens] - specifies which request each token belongs to
        self.seq_idx: torch.Tensor = None

    def prepare(self, attn_metadata: AttentionMetadata):
        num_contexts = attn_metadata.num_contexts
        context_lens = attn_metadata.seq_lens_cuda[:num_contexts]
        if num_contexts > 0:
            torch.cumsum(context_lens,
                         dim=0,
                         dtype=torch.int,
                         out=self.cu_seqlens[1:num_contexts + 1])
            self.seq_idx = torch.repeat_interleave(
                torch.arange(num_contexts,
                             dtype=torch.int,
                             device=self.cu_seqlens.device),
                repeats=context_lens,
                output_size=self.cu_seqlens[num_contexts]).unsqueeze(0)
