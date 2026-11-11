# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for the shared multimodal runtime."""

from ..multimodal.model import (
    EncoderCachePartition,
    EncoderGroup,
    MultimodalEncoderContractError,
    MultimodalModelContractError,
    MultimodalModelMixin,
    MultimodalRuntimeMixin,
    PreparedLlmInputs,
    PreparedMultimodalEmbeddings,
    _assemble_multimodal_encoder_embeddings,
    _build_request_multimodal_input,
    _collect_cross_iter_prefetch_candidates,
    _dispatch_cross_iter_prefetch,
    _get_mm_aux_stream,
    _lengths_by_modality,
    _reorder_embeds_by_manifest,
    _run_on_aux_stream,
    encode_multimodal_by_groups,
    maybe_prefetch_mm_encoder_for_next_iter,
)

__all__ = [
    "MultimodalEncoderContractError",
    "_assemble_multimodal_encoder_embeddings",
    "EncoderGroup",
    "_lengths_by_modality",
    "_reorder_embeds_by_manifest",
    "encode_multimodal_by_groups",
    "_build_request_multimodal_input",
    "_get_mm_aux_stream",
    "_run_on_aux_stream",
    "PreparedLlmInputs",
    "PreparedMultimodalEmbeddings",
    "EncoderCachePartition",
    "MultimodalModelContractError",
    "MultimodalModelMixin",
    "MultimodalRuntimeMixin",
    "_collect_cross_iter_prefetch_candidates",
    "_dispatch_cross_iter_prefetch",
    "maybe_prefetch_mm_encoder_for_next_iter",
]
