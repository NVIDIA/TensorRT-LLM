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
"""SkipSoftmax sparse attention parameter preparation."""

from __future__ import annotations

from ..params import SparseParams


def prepare_skip_softmax_params(
    sparse_attention_config,
    metadata,
) -> SparseParams:
    """Prepare SparseParams for the SkipSoftmax attention method."""
    params = SparseParams(
        sparse_mla_topk=(metadata.sparse_mla_topk if hasattr(metadata, "sparse_mla_topk") else 0),
        skip_softmax_threshold_scale_factor_prefill=(
            sparse_attention_config.threshold_scale_factor_prefill
        ),
        skip_softmax_threshold_scale_factor_decode=(
            sparse_attention_config.threshold_scale_factor_decode
        ),
    )
    return params
