# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""SkipSoftmax sparse attention backend.

Inherits TrtllmAttention and only overrides sparse_params() to provide
skip-softmax threshold scale factors. No index prediction is needed.
"""

from __future__ import annotations

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention, TrtllmAttentionMetadata

from ..params import SparseParams


class SkipSoftmaxTrtllmAttention(TrtllmAttention):
    """TrtllmAttention subclass for SkipSoftmax sparse attention.

    Only overrides sparse_params() to populate threshold scale factors.
    sparse_kv_predict and sparse_attn_predict remain as base (return None).
    """

    def sparse_params(self, metadata: TrtllmAttentionMetadata) -> SparseParams:
        return SparseParams(
            skip_softmax_threshold_scale_factor_prefill=(
                self.sparse_attention_config.threshold_scale_factor_prefill
            ),
            skip_softmax_threshold_scale_factor_decode=(
                self.sparse_attention_config.threshold_scale_factor_decode
            ),
        )
