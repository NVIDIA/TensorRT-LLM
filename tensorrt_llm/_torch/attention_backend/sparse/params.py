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
"""Base sparse parameter interfaces for attention backend boundaries.

User-facing SparseAttentionConfig live in LLM / VisualGen args and
ModelConfig. They lower through ``to_sparse_params()`` for per-backend runtime
state and ``to_sparse_metadata_params()`` for metadata allocation/update state.

Concrete sparse algorithm params live with their backend implementations;
shared kernel-facing carriers live here when they are part of the generic
attention-forward contract.
"""

from dataclasses import dataclass


class SparseParams:
    """Base for per AttentionBackend instance sparse runtime parameters.

    User-facing SparseAttentionConfig objects lower into this type through
    ``to_sparse_params()`` before an AttentionBackend is constructed. That
    lowering can resolve per-model or per-layer settings without passing
    configs into backend instances.
    """

    algorithm: str


class SparseMetadataParams:
    """Base for sparse settings needed by AttentionMetadata.

    Derived from the same user-facing SparseAttentionConfig through
    ``to_sparse_metadata_params()``, but kept separate from SparseParams because
    metadata owns batch/runtime buffers rather than per-layer
    ``AttentionBackend`` behavior.
    """


@dataclass
class SkipSoftmaxKernelParams:
    """Skip-softmax thresholds passed to attention backend kernels."""

    # The kernel divides this by the context length to get the skip threshold;
    # zero turns skip-softmax off.
    threshold_scale_factor_prefill: float = 0.0
    # Only autoregressive (LLM) decoding has a decode phase; diffusion and
    # visual generation leave this at zero.
    threshold_scale_factor_decode: float = 0.0
