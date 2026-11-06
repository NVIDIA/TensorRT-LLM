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

import pytest
import torch
from fmha_test_utils import FakeAttention, FakeFmha, FakePhasedFmha

from tensorrt_llm._torch.attention.backends.fmha.combined import (
    CombinedFmha,
    generation_rows_uniform,
)
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role


def test_combined_fmha_delegates_phases_and_prepares_max_workspace() -> None:
    events: list[tuple] = []
    attn = FakeAttention()
    context_fmha = FakePhasedFmha(
        attn,
        {FmhaPhase.CONTEXT},
        "context",
        events,
        workspace_size=8,
    )
    generation_fmha = FakePhasedFmha(
        attn,
        {FmhaPhase.GENERATION},
        "generation",
        events,
        workspace_size=4,
    )
    combined_fmha = CombinedFmha(attn)
    assert not combined_fmha.supports_workspace_reclamation
    for context_support, generation_support, expected in (
        (False, False, False),
        (True, False, False),
        (True, True, True),
        (False, True, False),
    ):
        context_fmha.supports_workspace_reclamation = context_support
        generation_fmha.supports_workspace_reclamation = generation_support
        combined_fmha.set_fmha_impls(context_fmha, generation_fmha)
        assert combined_fmha.supports_workspace_reclamation is expected
    metadata = SimpleNamespace(
        kv_cache_block_offsets=object(),
        effective_workspace=torch.empty(0, dtype=torch.uint8),
        num_contexts=2,
        num_ctx_tokens=3,
        num_generations=4,
        kv_lens_cuda_runtime=torch.tensor([2, 1, 5, 5, 5, 5], dtype=torch.int32),
        kv_lens_runtime=torch.tensor([2, 1, 5, 5, 5, 5], dtype=torch.int32),
        prompt_lens_cuda_runtime=torch.tensor([2, 1, 1, 1, 1, 1], dtype=torch.int32),
        prompt_lens_cpu_runtime=torch.tensor([2, 1, 1, 1, 1, 1], dtype=torch.int32),
        beam_width=2,
        cache_indirection=None,
        tokens_per_block=32,
        kv_cache_manager=None,
        is_cross=False,
        is_spec_decoding_enabled=False,
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((7, 4)),
        attention_input_type=AttentionInputType.mixed,
        attention_window_size=8,
    )

    combined_fmha.forward(torch.empty((7, 4)), None, None, metadata, forward_args)

    assert events == [
        ("prepare", "context"),
        ("prepare", "generation"),
        ("run", "context", FmhaPhase.CONTEXT, 3, 2, 2),
        ("run", "generation", FmhaPhase.GENERATION, 4, 4, 2),
    ]
    assert metadata.effective_workspace.numel() == 8


def test_combined_fmha_uses_flattened_v2_page_bound() -> None:
    attn = FakeAttention(local_layer_idx=3)
    combined_fmha = CombinedFmha(attn)
    calls: list[tuple[int, object]] = []

    def get_page_index_upper_bound(local_layer_idx: int, role: object) -> int:
        calls.append((local_layer_idx, role))
        return 23

    kv_cache_manager = object.__new__(KVCacheManagerV2)
    kv_cache_manager.impl = SimpleNamespace(get_page_index_upper_bound=get_page_index_upper_bound)

    assert (
        combined_fmha._get_total_num_blocks(SimpleNamespace(kv_cache_manager=kv_cache_manager))
        == 23
    )
    assert calls == [(3, Role.KEY)]


class _ContextOnlyFakeFmha(FakeFmha):
    serves_context_only_input = True

    def forward(self, q, k, v, metadata, forward_args) -> None:
        self._events.append(("forward", self._name, forward_args.attention_input_type, q.shape[0]))


def _mixed_metadata(gen_lens: list[int], *, max_total_draft_tokens=None) -> SimpleNamespace:
    ctx_lens = [3, 2]
    lens = ctx_lens + gen_lens
    return SimpleNamespace(
        kv_cache_block_offsets=object(),
        effective_workspace=torch.empty(0, dtype=torch.uint8),
        num_contexts=len(ctx_lens),
        num_ctx_tokens=sum(ctx_lens),
        num_generations=len(gen_lens),
        seq_lens=torch.tensor(lens, dtype=torch.int32),
        max_total_draft_tokens=max_total_draft_tokens,
        kv_lens_cuda_runtime=torch.tensor([9] * len(lens), dtype=torch.int32),
        kv_lens_runtime=torch.tensor([9] * len(lens), dtype=torch.int32),
        prompt_lens_cuda_runtime=torch.tensor(lens, dtype=torch.int32),
        prompt_lens_cpu_runtime=torch.tensor(lens, dtype=torch.int32),
        beam_width=1,
        cache_indirection=None,
        tokens_per_block=32,
        kv_cache_manager=None,
        is_cross=False,
        is_spec_decoding_enabled=False,
    )


_ATTN = FakeAttention()  # the libraries keep a weak reference to their attention layer


def _whole_batch_combined(events):
    attn = _ATTN
    context_fmha = _ContextOnlyFakeFmha(attn, "fallback", events)
    generation_fmha = FakePhasedFmha(attn, {FmhaPhase.GENERATION}, "generation", events)
    combined_fmha = CombinedFmha(attn)
    combined_fmha.set_fmha_impls(context_fmha, generation_fmha)
    return combined_fmha


@pytest.mark.parametrize("max_total_draft_tokens", [None, 3])
def test_combined_fmha_splits_a_mixed_batch_with_a_whole_batch_context_library(
    max_total_draft_tokens,
) -> None:
    """The whole-batch library gets the full tensors with a context-only call, then the
    generation library serves the generation rows; the input type is restored."""
    events: list[tuple] = []
    combined_fmha = _whole_batch_combined(events)
    assert not combined_fmha.supports_workspace_reclamation
    metadata = _mixed_metadata([4, 4, 4], max_total_draft_tokens=max_total_draft_tokens)
    forward_args = AttentionForwardArgs(
        output=torch.empty((17, 4)),
        attention_input_type=AttentionInputType.mixed,
        attention_window_size=8,
    )

    combined_fmha.forward(torch.empty((17, 4)), None, None, metadata, forward_args)

    assert events == [
        ("forward", "fallback", AttentionInputType.context_only, 17),
        ("prepare", "generation"),
        ("run", "generation", FmhaPhase.GENERATION, 12, 3, 3),
    ]
    assert forward_args.attention_input_type == AttentionInputType.mixed


def test_combined_fmha_restores_input_type_when_the_context_call_fails() -> None:
    events: list[tuple] = []
    combined_fmha = _whole_batch_combined(events)

    def fail(*args):
        raise RuntimeError("boom")

    combined_fmha._get_context_impl().forward = fail
    forward_args = AttentionForwardArgs(
        output=torch.empty((17, 4)), attention_input_type=AttentionInputType.mixed
    )
    with pytest.raises(RuntimeError, match="boom"):
        combined_fmha.forward(
            torch.empty((17, 4)), None, None, _mixed_metadata([4, 4, 4]), forward_args
        )
    assert forward_args.attention_input_type == AttentionInputType.mixed


@pytest.mark.parametrize("gen_lens", [[4, 2, 3], [1, 4, 4]])
def test_combined_fmha_sends_ragged_generation_rows_to_the_whole_batch_library(gen_lens) -> None:
    """Rows of different lengths, even when they divide evenly, stay on the whole-batch op."""
    events: list[tuple] = []
    combined_fmha = _whole_batch_combined(events)
    tokens = 5 + sum(gen_lens)
    forward_args = AttentionForwardArgs(
        output=torch.empty((tokens, 4)), attention_input_type=AttentionInputType.mixed
    )

    combined_fmha.forward(
        torch.empty((tokens, 4)),
        None,
        None,
        _mixed_metadata(gen_lens, max_total_draft_tokens=3),
        forward_args,
    )

    assert events == [("forward", "fallback", AttentionInputType.mixed, tokens)]


def test_generation_rows_uniform() -> None:
    def check(gen_lens, max_total_draft_tokens=None, tokens=None):
        metadata = _mixed_metadata(gen_lens, max_total_draft_tokens=max_total_draft_tokens)
        q = torch.empty((tokens if tokens is not None else 5 + sum(gen_lens), 4))
        return generation_rows_uniform(q, metadata)

    assert check([1, 1, 1])
    assert check([4, 4], max_total_draft_tokens=3)
    assert check([3, 3, 3])  # from the host lengths
    assert check([4, 4], max_total_draft_tokens=8)
    assert not check([4, 2, 3])
    assert not check([4, 2, 3], max_total_draft_tokens=8)
    assert not check([4, 4], tokens=12)  # not divisible
    metadata = _mixed_metadata([3, 3, 3])
    metadata.seq_lens = None
    assert not generation_rows_uniform(torch.empty((14, 4)), metadata)


def test_combined_fmha_refuses_a_whole_batch_context_library_without_context_only_calls() -> None:
    attn = FakeAttention()
    combined_fmha = CombinedFmha(attn)
    with pytest.raises(TypeError):
        combined_fmha.set_fmha_impls(
            FakeFmha(attn, "plain", []),
            FakePhasedFmha(attn, {FmhaPhase.GENERATION}, "generation", []),
        )
