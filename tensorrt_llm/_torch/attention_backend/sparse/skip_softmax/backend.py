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

"""TRTLLM sparse prediction for SkipSoftmax attention."""

from typing import Optional

import torch

from ...interface import AttentionForwardArgs, AttentionMetadata
from ...trtllm import TrtllmAttention
from ..params import SparseRuntimeParams
from .params import SkipSoftmaxParams


class SkipSoftmaxTrtllmAttention(TrtllmAttention):
    """TRTLLM backend with SkipSoftmax runtime prediction."""

    def predict_sparse_attention(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: AttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> SparseRuntimeParams:
        prediction = super().predict_sparse_attention(q, k, v, metadata, forward_args)
        sparse_params = self.sparse_params
        if not isinstance(sparse_params, SkipSoftmaxParams):
            raise TypeError("SkipSoftmax prediction requires SkipSoftmaxParams")
        runtime_params = sparse_params.scheduler.get_runtime_params(
            runtime_params=prediction,
            timestep=forward_args.timestep,
        )
        return runtime_params


__all__ = ["SkipSoftmaxTrtllmAttention"]
