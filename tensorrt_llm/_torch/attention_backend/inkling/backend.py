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
"""Attention backend that carries :class:`InklingAttentionMetadata`."""

from ..trtllm import TrtllmAttention
from .metadata import InklingAttentionMetadata


class InklingTritonAttention(TrtllmAttention):
    """Carries :class:`InklingAttentionMetadata`.

    Inkling never routes attention through a backend ``forward``:
    ``InklingAttention.forward`` overrides the base module entirely and calls
    the Triton kernels above. The backend object exists so the model engine
    picks the right ``Metadata`` class (``metadata_cls = attn_backend.Metadata``)
    and so the base module can assign ``local_layer_idx``. Subclassing
    ``TrtllmAttention`` rather than ``AttentionBackend`` keeps construction and
    every non-Inkling code path byte-identical to the TRTLLM backend Inkling
    used before.
    """

    Metadata = InklingAttentionMetadata
