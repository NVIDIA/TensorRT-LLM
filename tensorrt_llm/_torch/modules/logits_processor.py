# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch.nn as nn

from tensorrt_llm._utils import nvtx_range

from ..attention.backends import AttentionMetadata
from .linear import Linear


class LogitsProcessor(nn.Module):

    def __init__(self):
        super().__init__()

    @nvtx_range("LogitsProcessor")
    def forward(self,
                hidden_states: torch.Tensor,
                lm_head: Linear,
                attn_metadata: AttentionMetadata,
                return_context_logits: bool = False,
                keep_dtype: bool = False) -> torch.Tensor:
        # keep_dtype: return the lm_head's own output dtype and skip
        # the vocab-wide fp32 copy. Only for a caller whose consumer reads the
        # logits exactly in that dtype (an argmax); every other caller keeps
        # the default and gets fp32 logits as before.

        if not return_context_logits:
            if attn_metadata is not None:
                last_tokens = torch.cumsum(
                    attn_metadata.seq_lens_cuda,
                    dim=0,
                    dtype=torch.long,
                ) - 1
                hidden_states = hidden_states[last_tokens]
            else:
                hidden_states = hidden_states[-1]

        logits = lm_head(hidden_states)
        if keep_dtype:
            return logits
        logits = logits.float()
        return logits
