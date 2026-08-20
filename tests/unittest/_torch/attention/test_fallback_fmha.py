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

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.attention_backend.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    CustomAttentionMask,
)


class _TestAttention:
    pass


def test_fallback_rejects_q_only_self_attention() -> None:
    fmha = FallbackFmha(_TestAttention())
    q = torch.empty((1, 64))
    metadata = SimpleNamespace(is_cross=False)
    forward_args = AttentionForwardArgs(is_fused_qkv=False, update_kv_cache=False)

    assert not fmha.is_supported(q, None, None, metadata, forward_args)


def test_fallback_accepts_other_qkv_forms() -> None:
    fmha = FallbackFmha(_TestAttention())
    q = torch.empty((1, 64))

    fused_qkv_args = AttentionForwardArgs(is_fused_qkv=True, update_kv_cache=True)
    assert fmha.is_supported(
        q,
        None,
        None,
        SimpleNamespace(is_cross=False),
        fused_qkv_args,
    )

    unfused_qkv_args = AttentionForwardArgs(is_fused_qkv=False, update_kv_cache=True)
    kv = torch.empty((1, 32))
    assert fmha.is_supported(
        q,
        kv,
        kv,
        SimpleNamespace(is_cross=False),
        unfused_qkv_args,
    )

    cached_cross_args = AttentionForwardArgs(is_fused_qkv=False, update_kv_cache=False)
    assert fmha.is_supported(
        q,
        None,
        None,
        SimpleNamespace(is_cross=True),
        cached_cross_args,
    )


def test_fallback_still_rejects_custom_mask() -> None:
    fmha = FallbackFmha(_TestAttention())
    qkv = torch.empty((1, 64))
    metadata = SimpleNamespace(is_cross=False)
    forward_args = AttentionForwardArgs(
        attention_mask=CustomAttentionMask.CUSTOM,
        is_fused_qkv=True,
        update_kv_cache=True,
    )

    assert not fmha.is_supported(qkv, None, None, metadata, forward_args)
