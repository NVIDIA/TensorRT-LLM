# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import TYPE_CHECKING

from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig


def get_sparse_attn_kv_cache_manager(sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        from .rocket import RocketKVCacheManager

        return RocketKVCacheManager
    elif sparse_attn_config.algorithm == "dsa":
        from .dsa import DSACacheManager

        return DSACacheManager
    elif sparse_attn_config.algorithm == "skip_softmax":
        return KVCacheManager
    else:
        raise ValueError(f"Unsupported sparse attention algorithm: {sparse_attn_config.algorithm}")


def get_vanilla_sparse_attn_attention_backend(sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        from .rocket import RocketVanillaAttention

        return RocketVanillaAttention
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in vanilla attention backend: {sparse_attn_config.algorithm}"
        )


def get_trtllm_sparse_attn_attention_backend(sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        from .rocket import RocketTrtllmAttention

        return RocketTrtllmAttention
    elif sparse_attn_config.algorithm == "dsa":
        from .dsa import DSATrtllmAttention

        return DSATrtllmAttention
    elif sparse_attn_config.algorithm == "skip_softmax":
        from .skip_softmax.backend import SkipSoftmaxTrtllmAttention

        return SkipSoftmaxTrtllmAttention
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in trtllm attention backend: {sparse_attn_config.algorithm}"
        )


def get_flashinfer_sparse_attn_attention_backend(sparse_attn_config: "SparseAttentionConfig"):
    raise ValueError(
        f"Unsupported sparse attention algorithm in flashinfer attention backend: {sparse_attn_config.algorithm}"
    )
