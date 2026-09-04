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

"""VisualGen TRTLLM backend for shared SkipSoftmax attention."""

from typing import Optional

import torch

from .....attention_backend.interface import AttentionForwardArgs, AttentionMetadata
from .....attention_backend.sparse.params import SparseRuntimeParams
from .....attention_backend.sparse.skip_softmax import SkipSoftmaxParams
from ...trtllm import TrtllmAttention


class SkipSoftmaxTrtllmAttention(TrtllmAttention):
    """Bind VisualGen metadata handling to the core TRTLLM SkipSoftmax path."""

    def __init__(self, *, sparse_params: SkipSoftmaxParams | None = None, **kwargs) -> None:
        if not isinstance(sparse_params, SkipSoftmaxParams):
            raise TypeError("SkipSoftmaxTrtllmAttention requires SkipSoftmaxParams")
        super().__init__(sparse_params=sparse_params, **kwargs)

    def predict_sparse_attention(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: AttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> SparseRuntimeParams:
        """Apply SkipSoftmax scheduling to the aggregate prediction."""
        runtime_params = super().predict_sparse_attention(q, k, v, metadata, forward_args)
        return self.sparse_params.scheduler.get_runtime_params(
            runtime_params=runtime_params,
            timestep=forward_args.timestep,
        )


__all__ = ["SkipSoftmaxTrtllmAttention"]
