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

"""Tests for algorithm-independent sparse attention framework plumbing.

Kernel-specific regression coverage lives in dedicated modules such as
``test_sparse_mqa_gqa.py``. This file verifies how sparse algorithms register
hooks and pass predictions through ``SparseRuntimeParams``.
"""

from types import ModuleType
from unittest.mock import Mock

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
from tensorrt_llm._torch.attention_backend.sparse.hooks import (
    AttentionSparseHooks,
    MLASparseHooks,
    get_sparse_attention_hooks,
    get_sparse_mla_hooks,
    prepare_sparse_runtime_params,
    register_attention_sparse_hooks,
    register_mla_sparse_hooks,
)
from tensorrt_llm._torch.attention_backend.sparse.params import SparseParams, SparseRuntimeParams
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.modules.mla import MLA


class _StubSparseParams(SparseParams):
    """Minimal sparse parameters for framework-level tests."""

    algorithm: str = "test_sparse"

    @property
    def indices_block_size(self) -> int:
        return 1


class _StaticPredictionAttention(TrtllmAttention):
    """Backend stub that returns predetermined sparse predictions."""

    def sparse_kv_predict(self, q, k, metadata, forward_args: AttentionForwardArgs):
        return self._sparse_kv_indices, self._sparse_kv_offsets

    def sparse_attn_predict(self, q, k, metadata, forward_args: AttentionForwardArgs):
        return self._sparse_attn_indices, self._sparse_attn_offsets


def test_prepare_sparse_runtime_params_from_predictions() -> None:
    attention = _StaticPredictionAttention.__new__(_StaticPredictionAttention)
    attention.sparse_params = _StubSparseParams()
    attention._sparse_kv_indices = torch.tensor([1], dtype=torch.int32)
    attention._sparse_kv_offsets = torch.tensor([0, 1], dtype=torch.int32)
    attention._sparse_attn_indices = torch.tensor([2], dtype=torch.int32)
    attention._sparse_attn_offsets = None
    forward_args = AttentionForwardArgs(
        sparse_runtime_params=SparseRuntimeParams(sparse_attn_kv_lens=torch.tensor([3]))
    )

    runtime_params = prepare_sparse_runtime_params(
        attention, torch.empty(0), None, None, forward_args
    )

    assert runtime_params.sparse_kv_indices is attention._sparse_kv_indices
    assert runtime_params.sparse_kv_offsets is attention._sparse_kv_offsets
    assert runtime_params.sparse_attn_indices is attention._sparse_attn_indices
    assert runtime_params.sparse_attn_offsets is None
    assert runtime_params.sparse_attn_indices_block_size == 1
    assert (
        runtime_params.sparse_attn_kv_lens is forward_args.sparse_runtime_params.sparse_attn_kv_lens
    )


def test_sparse_attn_hook_registration() -> None:
    hook_module = ModuleType("sparse_attn_hook_registration")
    hook_module.sparse_params = _StubSparseParams()

    hook_module.sparse_params.algorithm = "dsa"
    dsa_hooks = get_sparse_mla_hooks(hook_module)
    assert isinstance(dsa_hooks, MLASparseHooks)
    assert dsa_hooks.mqa_rope_append
    assert dsa_hooks.need_absorption
    assert get_sparse_attention_hooks(hook_module) is None

    hook_module.sparse_params.algorithm = "deepseek_v4"
    dsv4_hooks = get_sparse_mla_hooks(hook_module)
    assert isinstance(dsv4_hooks, MLASparseHooks)
    assert not dsv4_hooks.mqa_rope_append
    assert not dsv4_hooks.need_absorption

    hook_module.sparse_params.algorithm = "rocket"
    rocket_hooks = get_sparse_attention_hooks(hook_module)
    assert isinstance(rocket_hooks, AttentionSparseHooks)
    assert get_sparse_mla_hooks(hook_module) is None

    register_mla_sparse_hooks("test_mla_hooks", type(dsa_hooks))
    hook_module.sparse_params.algorithm = "test_mla_hooks"
    assert isinstance(get_sparse_mla_hooks(hook_module), type(dsa_hooks))
    assert get_sparse_mla_hooks(hook_module) is not get_sparse_mla_hooks(hook_module)

    register_attention_sparse_hooks("test_attention_hooks", type(rocket_hooks))
    hook_module.sparse_params.algorithm = "test_attention_hooks"
    assert isinstance(get_sparse_attention_hooks(hook_module), type(rocket_hooks))
    assert get_sparse_attention_hooks(hook_module) is not get_sparse_attention_hooks(hook_module)


def test_mla_backend_only_forward_uses_default_path() -> None:
    backend_only_module = ModuleType("backend_only_sparse_attention")
    backend_only_module.sparse_params = _StubSparseParams()
    backend_only_module.sparse_params.algorithm = "skip_softmax"
    hooks = get_sparse_mla_hooks(backend_only_module)
    assert hooks is None

    mla = MLA.__new__(MLA)
    torch.nn.Module.__init__(mla)
    mla.sparse_attn_hooks = hooks
    default_forward = Mock()
    mla._forward_impl = default_forward
    hidden_states = torch.empty(0)
    attn_output = [torch.empty(0)]

    MLA.forward_impl(mla, None, hidden_states, None, attn_output)

    default_forward.assert_called_once_with(
        None,
        hidden_states,
        None,
        attn_output[0],
        latent_cache_gen=None,
    )


def test_prepare_sparse_runtime_params_without_predictions() -> None:
    attention = TrtllmAttention.__new__(TrtllmAttention)
    attention.sparse_params = _StubSparseParams()

    runtime_params = prepare_sparse_runtime_params(
        attention, torch.empty(0), None, None, AttentionForwardArgs()
    )

    assert runtime_params == SparseRuntimeParams()
