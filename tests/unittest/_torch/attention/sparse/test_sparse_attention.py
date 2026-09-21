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
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends import trtllm as trtllm_backend
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.sparse.hooks import (
    AttentionSparseHooks,
    MLASparseHooks,
    get_sparse_attention_hooks,
    get_sparse_mla_hooks,
    prepare_sparse_runtime_params,
    register_attention_sparse_hooks,
    register_mla_sparse_hooks,
)
from tensorrt_llm._torch.attention.backends.sparse.params import (
    BlockSparseForwardInputs,
    SparseBackendForwardArgs,
    SparseParams,
    SparseRuntimeParams,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.mla import MLA


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
        attention, torch.empty(0), None, None, None, forward_args
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


@pytest.mark.parametrize(
    "sparse_params", [None, _StubSparseParams()], ids=["dense_backend", "sparse_backend"]
)
def test_prepare_sparse_runtime_params_without_predictions(sparse_params) -> None:
    attention = TrtllmAttention.__new__(TrtllmAttention)
    attention.sparse_params = sparse_params

    runtime_params = prepare_sparse_runtime_params(
        attention, torch.empty(0), None, None, None, AttentionForwardArgs()
    )

    assert runtime_params == SparseRuntimeParams()


def test_prepare_sparse_runtime_params_runs_index_hooks_once() -> None:
    attention = _StaticPredictionAttention.__new__(_StaticPredictionAttention)
    attention.sparse_params = _StubSparseParams()
    sparse_kv_indices = torch.tensor([1], dtype=torch.int32)
    sparse_kv_offsets = torch.tensor([0, 1], dtype=torch.int32)
    sparse_attn_indices = torch.tensor([2], dtype=torch.int32)
    sparse_attn_offsets = torch.tensor([0, 1], dtype=torch.int32)
    attention.sparse_kv_predict = Mock(return_value=(sparse_kv_indices, sparse_kv_offsets))
    attention.sparse_attn_predict = Mock(return_value=(sparse_attn_indices, sparse_attn_offsets))
    q = torch.empty((1, 4))
    k = torch.empty((1, 4))
    v = torch.empty((1, 4))
    metadata = Mock()
    caller_kv_lens = torch.tensor([3])
    forward_args = AttentionForwardArgs(
        sparse_runtime_params=SparseRuntimeParams(sparse_attn_kv_lens=caller_kv_lens)
    )

    runtime_params = prepare_sparse_runtime_params(attention, q, k, v, metadata, forward_args)

    assert isinstance(runtime_params, SparseRuntimeParams)
    assert runtime_params.block_sparse_inputs is None
    assert runtime_params.sparse_kv_indices is sparse_kv_indices
    assert runtime_params.sparse_kv_offsets is sparse_kv_offsets
    assert runtime_params.sparse_attn_indices is sparse_attn_indices
    assert runtime_params.sparse_attn_offsets is sparse_attn_offsets
    assert runtime_params.sparse_attn_indices_block_size == 1
    assert runtime_params.sparse_attn_kv_lens is caller_kv_lens
    attention.sparse_kv_predict.assert_called_once_with(q, k, metadata, forward_args)
    attention.sparse_attn_predict.assert_called_once_with(q, k, metadata, forward_args)


def test_prepare_sparse_runtime_params_schedules_skip_softmax_thresholds() -> None:
    from tensorrt_llm._torch.attention.backends.sparse.skip_softmax import SkipSoftmaxParams

    attention = TrtllmAttention.__new__(TrtllmAttention)
    attention.sparse_params = SkipSoftmaxParams()
    scheduler = attention.sparse_params.scheduler
    timestep = torch.tensor(0.5)
    forward_args = AttentionForwardArgs(timestep=timestep)

    with patch.object(
        scheduler, "get_runtime_params", wraps=scheduler.get_runtime_params
    ) as schedule:
        runtime_params = prepare_sparse_runtime_params(
            attention, torch.empty(0), None, None, None, forward_args
        )

    schedule.assert_called_once_with(runtime_params=SparseRuntimeParams(), timestep=timestep)
    assert runtime_params == scheduler.get_runtime_params(timestep=timestep)


def _make_block_sparse_inputs() -> BlockSparseForwardInputs:
    return BlockSparseForwardInputs(
        q_block_size=64,
        kv_block_size=64,
        exact_block_bits=torch.zeros((1, 1), dtype=torch.int32),
    )


def test_block_sparse_attn_predict_hands_through_backend_args() -> None:
    attention = TrtllmAttention.__new__(TrtllmAttention)
    attention.sparse_params = None
    block_sparse_inputs = _make_block_sparse_inputs()
    forward_args = AttentionForwardArgs(
        sparse_backend_args=SparseBackendForwardArgs(block_sparse_inputs=block_sparse_inputs)
    )

    runtime_params = prepare_sparse_runtime_params(
        attention, torch.empty(0), None, None, None, forward_args
    )

    assert runtime_params.block_sparse_inputs is block_sparse_inputs
    assert runtime_params == SparseRuntimeParams(block_sparse_inputs=block_sparse_inputs)


def test_block_sparse_attn_predict_override_composes_with_index_predictors() -> None:
    attention = _StaticPredictionAttention.__new__(_StaticPredictionAttention)
    attention.sparse_params = _StubSparseParams()
    sparse_attn_indices = torch.tensor([2], dtype=torch.int32)
    sparse_attn_offsets = torch.tensor([0, 1], dtype=torch.int32)
    attention.sparse_kv_predict = Mock(return_value=(None, None))
    attention.sparse_attn_predict = Mock(return_value=(sparse_attn_indices, sparse_attn_offsets))
    block_sparse_inputs = _make_block_sparse_inputs()
    attention.block_sparse_attn_predict = Mock(return_value=block_sparse_inputs)
    q = torch.empty((1, 4))
    k = torch.empty((1, 4))
    v = torch.empty((1, 4))
    metadata = Mock()
    forward_args = AttentionForwardArgs()

    runtime_params = prepare_sparse_runtime_params(attention, q, k, v, metadata, forward_args)

    assert runtime_params.block_sparse_inputs is block_sparse_inputs
    assert runtime_params.sparse_attn_indices is sparse_attn_indices
    assert runtime_params.sparse_attn_indices_block_size == 1
    attention.block_sparse_attn_predict.assert_called_once_with(q, k, v, metadata, forward_args)


def test_attention_forward_args_default_to_empty_sparse_runtime_params() -> None:
    assert AttentionForwardArgs().sparse_runtime_params == SparseRuntimeParams()


class _StopAfterShapeValidation(Exception):
    pass


def _make_sparse_prediction_forward_backend() -> TrtllmAttention:
    attention = _StaticPredictionAttention.__new__(_StaticPredictionAttention)
    attention.sparse_params = None
    attention.is_mla_enable = False
    attention.num_heads = 1
    attention.num_kv_heads = 1
    attention.head_dim = 4
    attention.get_local_layer_idx = Mock(return_value=1)
    attention._ensure_rope_table_size = Mock(side_effect=_StopAfterShapeValidation)
    return attention


def _make_sparse_prediction_forward_metadata() -> TrtllmAttentionMetadata:
    metadata = object.__new__(TrtllmAttentionMetadata)
    seq_lens = torch.tensor([2], dtype=torch.int32)
    metadata._seq_lens = seq_lens
    metadata._seq_lens_kv = seq_lens
    metadata._seq_lens_cuda = None
    metadata.kv_cache_manager = None
    metadata._max_seq_len_storage = 4
    metadata.use_paged_context_fmha = False
    metadata.cu_q_seqlens = None
    metadata.cu_kv_seqlens = None
    metadata.enable_flash_mla = False
    metadata.spec_bl_tree_first_sparse_mask_offset_kv = None
    metadata.kv_lens_cuda_runtime = torch.tensor([2], dtype=torch.int32)
    metadata.kv_lens_runtime = torch.tensor([2], dtype=torch.int32)
    metadata.prompt_lens_cuda_runtime = torch.tensor([2], dtype=torch.int32)
    metadata.prompt_lens_cpu_runtime = torch.tensor([2], dtype=torch.int32)
    metadata.host_request_types_runtime = torch.tensor([0], dtype=torch.int32)
    return metadata


def test_forward_materializes_dynamic_block_sparse_prediction_before_shape_validation() -> None:
    attention = _make_sparse_prediction_forward_backend()
    block_sparse_inputs = _make_block_sparse_inputs()
    prediction = SparseRuntimeParams(
        sparse_attn_kv_lens=torch.tensor([4]),
        block_sparse_inputs=block_sparse_inputs,
    )
    q = torch.empty((2, 4))
    k = torch.empty((2, 4))
    v = torch.empty((2, 4))
    metadata = _make_sparse_prediction_forward_metadata()
    forward_args = AttentionForwardArgs(
        output=torch.empty_like(q),
        attention_input_type=AttentionInputType.context_only,
    )

    with patch.object(
        trtllm_backend, "prepare_sparse_runtime_params", return_value=prediction
    ) as prepare:
        for _ in range(2):
            with pytest.raises(_StopAfterShapeValidation):
                attention.forward(q, k, v, metadata, forward_args)

    assert prepare.call_count == 2
    prepare.assert_called_with(attention, q, k, v, metadata, forward_args)
    assert attention._ensure_rope_table_size.call_count == 2
    assert forward_args.sparse_runtime_params is prediction


@pytest.mark.parametrize("has_block_sparse_inputs", [False, True])
def test_forward_assigns_prepared_sparse_runtime_params(
    has_block_sparse_inputs: bool,
) -> None:
    attention = _make_sparse_prediction_forward_backend()
    block_sparse_inputs = _make_block_sparse_inputs() if has_block_sparse_inputs else None
    prediction = SparseRuntimeParams(
        sparse_attn_kv_lens=torch.tensor([2]),
        block_sparse_inputs=block_sparse_inputs,
    )
    caller_params = SparseRuntimeParams(sparse_attn_kv_lens=torch.tensor([7]))
    q = torch.empty((2, 4))
    k = torch.empty((2, 4))
    v = torch.empty((2, 4))
    metadata = _make_sparse_prediction_forward_metadata()
    forward_args = AttentionForwardArgs(
        output=torch.empty_like(q),
        attention_input_type=AttentionInputType.context_only,
        sparse_runtime_params=caller_params,
    )

    with patch.object(
        trtllm_backend, "prepare_sparse_runtime_params", return_value=prediction
    ) as prepare:
        for _ in range(2):
            with pytest.raises(_StopAfterShapeValidation):
                attention.forward(q, k, v, metadata, forward_args)

    assert prepare.call_count == 2
    assert attention._ensure_rope_table_size.call_count == 2
    assert forward_args.sparse_runtime_params is prediction


@pytest.mark.parametrize("backend_name", ["FLASHINFER", "unknown"])
def test_sparse_attention_backend_fallback_does_not_redispatch(
    backend_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensorrt_llm._torch.attention.backends import utils as attention_backend_utils
    from tensorrt_llm._torch.attention.backends.sparse.skip_softmax import SkipSoftmaxParams

    monkeypatch.setattr(attention_backend_utils, "IS_FLASHINFER_AVAILABLE", False)

    with patch.object(
        attention_backend_utils,
        "get_trtllm_sparse_attn_attention_backend",
    ) as trtllm_sparse_resolver:
        backend = attention_backend_utils.get_attention_backend(
            backend_name, sparse_params=SkipSoftmaxParams()
        )

    assert backend is TrtllmAttention
    trtllm_sparse_resolver.assert_not_called()


@pytest.mark.parametrize("backend_name", ["FLASHINFER", "unknown"])
def test_trtllm_fallback_without_sparse_params_remains_dense(
    backend_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensorrt_llm._torch.attention.backends import utils as attention_backend_utils

    monkeypatch.setattr(attention_backend_utils, "IS_FLASHINFER_AVAILABLE", False)

    assert attention_backend_utils.get_attention_backend(backend_name) is TrtllmAttention
