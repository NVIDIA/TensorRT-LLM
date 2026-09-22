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
from fmha_test_utils import FakeAttention, FakePhasedFmha

from tensorrt_llm._torch.attention.backends.fmha.combined import CombinedFmha
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role


@pytest.mark.parametrize(
    "input_type", [AttentionInputType.mixed, AttentionInputType.generation_only]
)
def test_combined_fmha_delegates_phases_and_prepares_max_workspace(
    monkeypatch: pytest.MonkeyPatch, input_type: AttentionInputType
) -> None:
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
        prompt_lens_cuda_runtime=torch.tensor([2, 1, 3, 3, 4, 4], dtype=torch.int32),
        prompt_lens_cpu_runtime=torch.tensor([2, 1, 3, 3, 4, 4], dtype=torch.int32),
        beam_width=2,
        cache_indirection=None,
        tokens_per_block=32,
        kv_cache_manager=None,
        is_cross=False,
        is_spec_decoding_enabled=False,
    )
    run_generation = generation_fmha.run_generation

    def check_generation_lengths(params) -> None:
        assert params.context_lengths.tolist() == [3, 3, 4, 4]
        assert params.context_lengths.data_ptr() == metadata.prompt_lens_cuda_runtime[2:].data_ptr()
        run_generation(params)

    monkeypatch.setattr(generation_fmha, "run_generation", check_generation_lengths)
    num_tokens = 4 if input_type == AttentionInputType.generation_only else 7
    forward_args = AttentionForwardArgs(
        output=torch.empty((num_tokens, 4)),
        attention_input_type=input_type,
        attention_window_size=8,
    )

    combined_fmha.forward(torch.empty((num_tokens, 4)), None, None, metadata, forward_args)

    expected_events = [
        ("prepare", "context"),
        ("prepare", "generation"),
    ]
    if input_type == AttentionInputType.mixed:
        expected_events.append(("run", "context", FmhaPhase.CONTEXT, 3, 2, 2))
    expected_events.append(("run", "generation", FmhaPhase.GENERATION, 4, 4, 2))
    assert events == expected_events
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
