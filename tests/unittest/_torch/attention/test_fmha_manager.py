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

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from fmha_test_utils import FakeAttention, FakeFmha, FakePhasedFmha

from tensorrt_llm._torch.attention.backends.fmha import manager as fmha_manager
from tensorrt_llm._torch.attention.backends.fmha.combined import CombinedFmha
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.fmha.manager import FmhaManager
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def _make_manager(*, sanity_check_enabled: bool = False) -> tuple[FakeAttention, FmhaManager]:
    attn = FakeAttention()
    with patch.object(fmha_manager, "get_enabled_fmha_lib_classes", return_value=[]):
        manager = FmhaManager(attn)
    manager._cache_sanity_check_enabled = sanity_check_enabled
    return attn, manager


def _make_metadata(
    *,
    num_contexts: int,
    num_generations: int,
    num_ctx_tokens: int = 0,
    use_spec_decoding: bool = False,
    is_cuda_graph: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_contexts=num_contexts,
        num_generations=num_generations,
        num_ctx_tokens=num_ctx_tokens,
        use_spec_decoding=use_spec_decoding,
        is_cuda_graph=is_cuda_graph,
    )


def test_manager_does_not_retain_attention() -> None:
    attn, manager = _make_manager()
    attn_ref = weakref.ref(attn)

    del attn
    gc.collect()

    assert attn_ref() is None
    assert manager.fmha_libs == []


def test_select_non_mla_fmha_combines_supported_phases() -> None:
    events: list[tuple] = []
    attn, manager = _make_manager()
    context_fmha = FakePhasedFmha(attn, {FmhaPhase.CONTEXT}, "context", events)
    generation_fmha = FakePhasedFmha(attn, {FmhaPhase.GENERATION}, "generation", events)
    manager.fmha_libs = [context_fmha, generation_fmha]

    selected = manager.select(
        attn,
        torch.empty((2, 4)),
        None,
        None,
        _make_metadata(num_contexts=1, num_generations=1, num_ctx_tokens=1),
        AttentionForwardArgs(attention_input_type=AttentionInputType.mixed),
    )

    assert isinstance(selected, CombinedFmha)
    assert selected.attn is attn
    assert selected._get_context_impl() is context_fmha
    assert selected._get_generation_impl() is generation_fmha
    assert events == [
        ("support", "context", None),
        ("support", "context", FmhaPhase.CONTEXT),
        ("support", "context", FmhaPhase.GENERATION),
        ("support", "generation", None),
        ("support", "generation", FmhaPhase.GENERATION),
    ]


def test_select_non_mla_fmha_checks_followup_support() -> None:
    events: list[tuple] = []
    attn, manager = _make_manager()
    context_fmha = FakePhasedFmha(attn, {FmhaPhase.CONTEXT}, "context", events)
    unsupported_followup = FakePhasedFmha(attn, set(), "followup", events)
    manager.fmha_libs = [context_fmha, unsupported_followup]

    selected = manager.select(
        attn,
        torch.empty((2, 4)),
        None,
        None,
        _make_metadata(num_contexts=1, num_generations=1, num_ctx_tokens=1),
        AttentionForwardArgs(attention_input_type=AttentionInputType.mixed),
    )

    assert selected is None
    assert events == [
        ("support", "context", None),
        ("support", "context", FmhaPhase.CONTEXT),
        ("support", "context", FmhaPhase.GENERATION),
        ("support", "followup", None),
        ("support", "followup", FmhaPhase.GENERATION),
    ]


def test_select_non_mla_fmha_reuses_one_implementation() -> None:
    events: list[tuple] = []
    attn, manager = _make_manager()
    fmha = FakePhasedFmha(
        attn,
        {None, FmhaPhase.CONTEXT, FmhaPhase.GENERATION},
        "both",
        events,
    )
    manager.fmha_libs = [fmha]

    selected = manager.select(
        attn,
        torch.empty((2, 4)),
        None,
        None,
        _make_metadata(num_contexts=1, num_generations=1, num_ctx_tokens=1),
        AttentionForwardArgs(attention_input_type=AttentionInputType.mixed),
    )

    assert selected is fmha
    assert events == [("support", "both", None)]


def test_select_non_mla_fmha_skips_same_phased_implementation_after_full_rejection() -> None:
    events: list[tuple] = []
    attn, manager = _make_manager()
    phased_fmha = FakePhasedFmha(
        attn,
        {FmhaPhase.CONTEXT, FmhaPhase.GENERATION},
        "phased",
        events,
    )
    fallback_fmha = FakeFmha(attn, "fallback", events)
    manager.fmha_libs = [phased_fmha, fallback_fmha]

    selected = manager.select(
        attn,
        torch.empty((2, 4)),
        None,
        None,
        _make_metadata(num_contexts=1, num_generations=1),
        AttentionForwardArgs(attention_input_type=AttentionInputType.mixed),
    )

    assert selected is fallback_fmha
    assert events == [
        ("support", "phased", None),
        ("support", "phased", FmhaPhase.CONTEXT),
        ("support", "phased", FmhaPhase.GENERATION),
        ("support", "fallback", None),
    ]


def test_select_non_mla_fmha_uses_context_only_phased_implementation() -> None:
    events: list[tuple] = []
    attn, manager = _make_manager()
    context_fmha = FakePhasedFmha(attn, {FmhaPhase.CONTEXT}, "context", events)
    manager.fmha_libs = [context_fmha]

    selected = manager.select(
        attn,
        torch.empty((2, 4)),
        None,
        None,
        _make_metadata(num_contexts=1, num_generations=0),
        AttentionForwardArgs(attention_input_type=AttentionInputType.mixed),
    )

    assert selected is context_fmha
    assert events == [
        ("support", "context", None),
        ("support", "context", FmhaPhase.CONTEXT),
    ]


def test_select_non_mla_fmha_preserves_registry_order() -> None:
    events: list[tuple] = []
    attn, manager = _make_manager()
    non_phased_fmha = FakeFmha(attn, "non-phased", events)
    phased_fmha = FakePhasedFmha(
        attn,
        {FmhaPhase.CONTEXT, FmhaPhase.GENERATION},
        "phased",
        events,
    )
    manager.fmha_libs = [non_phased_fmha, phased_fmha]

    selected = manager.select(
        attn,
        torch.empty((2, 4)),
        None,
        None,
        _make_metadata(num_contexts=1, num_generations=1, num_ctx_tokens=1),
        AttentionForwardArgs(attention_input_type=AttentionInputType.mixed),
    )

    assert selected is non_phased_fmha
    assert events == [("support", "non-phased", None)]


@pytest.mark.parametrize(("lower_q_length", "upper_q_length"), [(3, 4), (7, 8)])
@pytest.mark.parametrize("upper_first", [False, True])
def test_fmha_cache_separates_generation_q_boundaries(
    lower_q_length: int,
    upper_q_length: int,
    upper_first: bool,
) -> None:
    events: list[tuple] = []
    attn, manager = _make_manager()
    attn.is_mla_enable = True
    upper_boundary_fmha = FakeFmha(
        attn,
        "upper-boundary",
        events,
        request_support_predicate=lambda q, metadata: q.shape[0] // metadata.num_generations
        == upper_q_length,
    )
    fallback = FakeFmha(attn, "fallback", events)
    manager.fmha_libs = [upper_boundary_fmha, fallback]
    batch_size = 128
    metadata = _make_metadata(num_contexts=0, num_generations=batch_size)
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)
    request_order = (
        (upper_q_length, lower_q_length) if upper_first else (lower_q_length, upper_q_length)
    )

    with patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=True):
        selected_by_q_length = {
            q_length: manager.select(
                attn, torch.empty((batch_size * q_length, 4)), None, None, metadata, forward_args
            )
            for q_length in request_order
        }
        for q_length in request_order:
            assert (
                manager.select(
                    attn,
                    torch.empty((batch_size * q_length, 4)),
                    None,
                    None,
                    metadata,
                    forward_args,
                )
                is selected_by_q_length[q_length]
            )

    assert selected_by_q_length == {
        lower_q_length: fallback,
        upper_q_length: upper_boundary_fmha,
    }
    assert len(manager._cache) == 2
    assert events.count(("support", "upper-boundary", None)) == 2
    assert events.count(("support", "fallback", None)) == 1


@pytest.mark.parametrize(
    ("lower_batch_size", "upper_batch_size", "q_length"),
    [(31, 32, 4), (63, 64, 1)],
)
@pytest.mark.parametrize("upper_first", [False, True])
def test_fmha_cache_separates_generation_batch_boundaries(
    lower_batch_size: int,
    upper_batch_size: int,
    q_length: int,
    upper_first: bool,
) -> None:
    events: list[tuple] = []
    attn, manager = _make_manager()
    attn.is_mla_enable = True
    upper_boundary_fmha = FakeFmha(
        attn,
        "upper-boundary",
        events,
        request_support_predicate=lambda _q, metadata: metadata.num_generations == upper_batch_size,
    )
    fallback = FakeFmha(attn, "fallback", events)
    manager.fmha_libs = [upper_boundary_fmha, fallback]
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)
    request_order = (
        (upper_batch_size, lower_batch_size)
        if upper_first
        else (lower_batch_size, upper_batch_size)
    )

    with patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=True):
        selected_by_batch_size = {
            batch_size: manager.select(
                attn,
                torch.empty((batch_size * q_length, 4)),
                None,
                None,
                _make_metadata(num_contexts=0, num_generations=batch_size),
                forward_args,
            )
            for batch_size in request_order
        }
        for batch_size in request_order:
            assert (
                manager.select(
                    attn,
                    torch.empty((batch_size * q_length, 4)),
                    None,
                    None,
                    _make_metadata(num_contexts=0, num_generations=batch_size),
                    forward_args,
                )
                is selected_by_batch_size[batch_size]
            )

    assert selected_by_batch_size == {
        lower_batch_size: fallback,
        upper_batch_size: upper_boundary_fmha,
    }
    assert len(manager._cache) == 2
    assert events.count(("support", "upper-boundary", None)) == 2
    assert events.count(("support", "fallback", None)) == 1


def test_fmha_cache_reuses_grid_cell() -> None:
    events: list[tuple] = []
    attn, manager = _make_manager()
    boundary_fmha = FakeFmha(
        attn,
        "boundary",
        events,
        request_support_predicate=lambda q, metadata: (
            metadata.num_generations == 64 and q.shape[0] // metadata.num_generations == 8
        ),
    )
    fallback = FakeFmha(attn, "fallback", events)
    manager.fmha_libs = [boundary_fmha, fallback]
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)

    with patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=True):
        for batch_size, q_length in ((57, 5), (63, 7)):
            assert (
                manager.select(
                    attn,
                    torch.empty((batch_size * q_length, 4)),
                    None,
                    None,
                    _make_metadata(num_contexts=0, num_generations=batch_size),
                    forward_args,
                )
                is fallback
            )
        assert (
            manager.select(
                attn,
                torch.empty((64 * 8, 4)),
                None,
                None,
                _make_metadata(num_contexts=0, num_generations=64),
                forward_args,
            )
            is boundary_fmha
        )

    assert len(manager._cache) == 2
    assert events == [
        ("support", "boundary", None),
        ("support", "fallback", None),
        ("support", "boundary", None),
    ]


def test_context_fmha_cache_uses_batch_grid_only() -> None:
    events: list[tuple] = []
    attn, manager = _make_manager()
    fmha = FakeFmha(attn, "fmha", events)
    manager.fmha_libs = [fmha]
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.context_only)

    with patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=True):
        first = manager.select(
            attn,
            torch.empty((57 * 3, 4)),
            None,
            None,
            _make_metadata(num_contexts=57, num_generations=0, num_ctx_tokens=57 * 3),
            forward_args,
        )
        second = manager.select(
            attn,
            torch.empty((63 * 17, 4)),
            None,
            None,
            _make_metadata(num_contexts=63, num_generations=0, num_ctx_tokens=63 * 17),
            forward_args,
        )
        same_batch_different_q_length = manager.select(
            attn,
            torch.empty((63 * 3, 4)),
            None,
            None,
            _make_metadata(num_contexts=63, num_generations=0, num_ctx_tokens=63 * 3),
            forward_args,
        )

    assert first is fmha
    assert second is fmha
    assert same_batch_different_q_length is fmha
    assert len(manager._cache) == 1
    assert events == [("support", "fmha", None)]


@pytest.mark.parametrize("graph_first", [False, True])
def test_fmha_cache_separates_cuda_graph_mode(graph_first: bool) -> None:
    events: list[tuple] = []
    attn, manager = _make_manager()
    eager_fmha = FakeFmha(
        attn,
        "eager",
        events,
        request_support_predicate=lambda _q, metadata: not metadata.is_cuda_graph,
    )
    fallback = FakeFmha(attn, "fallback", events)
    manager.fmha_libs = [eager_fmha, fallback]
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.context_only)
    request_order = (True, False) if graph_first else (False, True)

    with patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=True):
        selected_by_graph_mode = {
            is_cuda_graph: manager.select(
                attn,
                torch.empty((1, 4)),
                None,
                None,
                _make_metadata(
                    num_contexts=1,
                    num_generations=0,
                    num_ctx_tokens=1,
                    is_cuda_graph=is_cuda_graph,
                ),
                forward_args,
            )
            for is_cuda_graph in request_order
        }
        for is_cuda_graph in request_order:
            assert (
                manager.select(
                    attn,
                    torch.empty((1, 4)),
                    None,
                    None,
                    _make_metadata(
                        num_contexts=1,
                        num_generations=0,
                        num_ctx_tokens=1,
                        is_cuda_graph=is_cuda_graph,
                    ),
                    forward_args,
                )
                is selected_by_graph_mode[is_cuda_graph]
            )

    assert selected_by_graph_mode == {False: eager_fmha, True: fallback}
    assert len(manager._cache) == 2
    assert events.count(("support", "eager", None)) == 2
    assert events.count(("support", "fallback", None)) == 1


@pytest.mark.parametrize("reverse_order", [False, True])
def test_fmha_cache_separates_qkv_input_modes(reverse_order: bool) -> None:
    events: list[tuple] = []
    attn, manager = _make_manager()
    fused_fmha = FakeFmha(
        attn,
        "fused",
        events,
        support_predicate=lambda forward_args: forward_args.is_fused_qkv,
    )
    unfused_fmha = FakeFmha(
        attn,
        "unfused",
        events,
        support_predicate=lambda forward_args: (
            not forward_args.is_fused_qkv and forward_args.update_kv_cache
        ),
    )
    q_only_fmha = FakeFmha(
        attn,
        "q-only",
        events,
        support_predicate=lambda forward_args: (
            not forward_args.is_fused_qkv and not forward_args.update_kv_cache
        ),
    )
    manager.fmha_libs = [fused_fmha, unfused_fmha, q_only_fmha]
    metadata = _make_metadata(num_contexts=1, num_generations=0, num_ctx_tokens=1)
    request_order = [(True, True), (False, True), (False, False)]
    if reverse_order:
        request_order.reverse()

    def select(is_fused_qkv: bool, update_kv_cache: bool) -> FakeFmha | None:
        q = torch.empty((1, 8 if is_fused_qkv else 4))
        k = torch.empty((1, 2)) if update_kv_cache and not is_fused_qkv else None
        v = torch.empty((1, 2)) if update_kv_cache and not is_fused_qkv else None
        return manager.select(
            attn,
            q,
            k,
            v,
            metadata,
            AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                is_fused_qkv=is_fused_qkv,
                update_kv_cache=update_kv_cache,
            ),
        )

    with patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=True):
        selected_by_input_mode = {input_mode: select(*input_mode) for input_mode in request_order}
        for input_mode in request_order:
            assert select(*input_mode) is selected_by_input_mode[input_mode]

    assert selected_by_input_mode == {
        (True, True): fused_fmha,
        (False, True): unfused_fmha,
        (False, False): q_only_fmha,
    }
    assert len(manager._cache) == 3
    assert events.count(("support", "fused", None)) == 3
    assert events.count(("support", "unfused", None)) == 2
    assert events.count(("support", "q-only", None)) == 1


@pytest.mark.parametrize("generation_first", [False, True])
def test_fmha_cache_separates_compacted_mla_phases(generation_first: bool) -> None:
    events: list[tuple] = []
    attn, manager = _make_manager()
    attn.is_mla_enable = True
    context_fmha = FakePhasedFmha(attn, {FmhaPhase.CONTEXT}, "context", events)
    generation_fmha = FakePhasedFmha(attn, {FmhaPhase.GENERATION}, "generation", events)
    manager.fmha_libs = [context_fmha, generation_fmha]
    metadata = _make_metadata(num_contexts=3, num_generations=2, num_ctx_tokens=6)
    requests = {
        FmhaPhase.CONTEXT: (
            torch.empty((6, 4)),
            AttentionForwardArgs(attention_input_type=AttentionInputType.context_only),
        ),
        FmhaPhase.GENERATION: (
            torch.empty((2, 4)),
            AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only),
        ),
    }
    request_order = (
        (FmhaPhase.GENERATION, FmhaPhase.CONTEXT)
        if generation_first
        else (FmhaPhase.CONTEXT, FmhaPhase.GENERATION)
    )

    with patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=True):
        selected_by_phase = {
            phase: manager.select(
                attn,
                requests[phase][0],
                None,
                None,
                metadata,
                requests[phase][1],
            )
            for phase in request_order
        }
        for phase in request_order:
            assert (
                manager.select(
                    attn,
                    requests[phase][0],
                    None,
                    None,
                    metadata,
                    requests[phase][1],
                )
                is selected_by_phase[phase]
            )

    assert selected_by_phase == {
        FmhaPhase.CONTEXT: context_fmha,
        FmhaPhase.GENERATION: generation_fmha,
    }
    assert len(manager._cache) == 2
    assert events.count(("support", "context", FmhaPhase.CONTEXT)) == 1
    assert events.count(("support", "context", FmhaPhase.GENERATION)) == 1
    assert events.count(("support", "generation", FmhaPhase.GENERATION)) == 1


def test_fmha_cache_tracks_attention_mask_data() -> None:
    for mask_data_first in (True, False):
        events: list[tuple] = []
        attn, manager = _make_manager()
        implicit_mask_only = FakeFmha(
            attn,
            "implicit-mask-only",
            events,
            support_predicate=lambda forward_args: forward_args.attention_mask_data is None,
        )
        fallback = FakeFmha(attn, "fallback", events)
        manager.fmha_libs = [implicit_mask_only, fallback]
        metadata = _make_metadata(num_contexts=0, num_generations=1)
        q = torch.empty((1, 4))
        implicit_mask_args = AttentionForwardArgs(
            attention_input_type=AttentionInputType.generation_only,
            attention_mask=PredefinedAttentionMask.CAUSAL,
        )
        mask_data_args = AttentionForwardArgs(
            attention_input_type=AttentionInputType.generation_only,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            attention_mask_data=torch.empty((1, 1)),
        )
        request_order = (
            (mask_data_args, implicit_mask_args)
            if mask_data_first
            else (implicit_mask_args, mask_data_args)
        )

        with patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=True):
            selected = [
                manager.select(attn, q, None, None, metadata, forward_args)
                for forward_args in request_order
            ]

        expected = (
            [fallback, implicit_mask_only] if mask_data_first else [implicit_mask_only, fallback]
        )
        assert selected == expected
        assert (
            manager.select(attn, q, None, None, metadata, implicit_mask_args) is implicit_mask_only
        )
        assert manager.select(attn, q, None, None, metadata, mask_data_args) is fallback
        assert len(manager._cache) == 2


@pytest.mark.parametrize("speculative_first", [False, True])
def test_fmha_cache_separates_speculative_decoding(speculative_first: bool) -> None:
    events: list[tuple] = []
    attn, manager = _make_manager()
    regular_only = FakeFmha(
        attn,
        "regular-only",
        events,
        request_support_predicate=lambda _q, metadata: not metadata.use_spec_decoding,
    )
    fallback = FakeFmha(attn, "fallback", events)
    manager.fmha_libs = [regular_only, fallback]
    q = torch.empty((4, 4))
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)
    metadata_by_mode = {
        False: _make_metadata(
            num_contexts=0,
            num_generations=1,
            use_spec_decoding=False,
        ),
        True: _make_metadata(
            num_contexts=0,
            num_generations=1,
            use_spec_decoding=True,
        ),
    }
    request_order = (True, False) if speculative_first else (False, True)

    with patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=True):
        selected_by_mode = {
            use_spec_decoding: manager.select(
                attn,
                q,
                None,
                None,
                metadata_by_mode[use_spec_decoding],
                forward_args,
            )
            for use_spec_decoding in request_order
        }
        for use_spec_decoding in request_order:
            assert (
                manager.select(
                    attn,
                    q,
                    None,
                    None,
                    metadata_by_mode[use_spec_decoding],
                    forward_args,
                )
                is selected_by_mode[use_spec_decoding]
            )

    assert selected_by_mode == {False: regular_only, True: fallback}
    assert len(manager._cache) == 2
    assert events.count(("support", "regular-only", None)) == 2
    assert events.count(("support", "fallback", None)) == 1


def test_fmha_cache_tracks_lora_output_representation() -> None:
    for lora_first in (True, False):
        events: list[tuple] = []
        attn, manager = _make_manager()
        unpacked_only = FakeFmha(
            attn,
            "unpacked-only",
            events,
            support_predicate=lambda forward_args: forward_args.output is not None
            and forward_args.output.dtype == torch.bfloat16,
        )
        fallback = FakeFmha(attn, "fallback", events)
        manager.fmha_libs = [unpacked_only, fallback]
        metadata = _make_metadata(num_contexts=0, num_generations=4)
        q = torch.empty((4, 4), dtype=torch.bfloat16)
        lora_args = AttentionForwardArgs(
            output=torch.empty((4, 4), dtype=torch.bfloat16),
            attention_input_type=AttentionInputType.generation_only,
        )
        base_args = AttentionForwardArgs(
            output=torch.empty((4, 2), dtype=torch.uint8),
            output_sf=torch.empty(4, dtype=torch.uint8),
            attention_input_type=AttentionInputType.generation_only,
        )
        request_order = (lora_args, base_args) if lora_first else (base_args, lora_args)

        with patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=True):
            selected = [
                manager.select(attn, q, None, None, metadata, forward_args)
                for forward_args in request_order
            ]
            assert manager.select(attn, q, None, None, metadata, lora_args) is unpacked_only
            assert manager.select(attn, q, None, None, metadata, base_args) is fallback

        expected = [unpacked_only, fallback] if lora_first else [fallback, unpacked_only]
        assert selected == expected
        assert len(manager._cache) == 2


def test_fmha_cache_sanity_check_logs_mismatched_inputs() -> None:
    events: list[tuple] = []
    attn, manager = _make_manager(sanity_check_enabled=True)
    preferred = FakeFmha(
        attn,
        "preferred",
        events,
        request_support_predicate=lambda q, metadata: q.shape[1] == 4,
    )
    fallback = FakeFmha(attn, "fallback", events)
    manager.fmha_libs = [preferred, fallback]
    cached_metadata = _make_metadata(num_contexts=0, num_generations=1)
    cached_metadata.beam_width = 1
    uncached_metadata = _make_metadata(num_contexts=0, num_generations=1)
    uncached_metadata.beam_width = 2
    cached_args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        relative_attention_max_distance=0,
    )
    uncached_args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        relative_attention_max_distance=7,
    )

    with (
        patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=True),
        patch.object(fmha_manager.logger, "error") as log_error,
    ):
        assert (
            manager.select(
                attn,
                torch.empty((1, 4)),
                torch.empty((1, 2)),
                None,
                cached_metadata,
                cached_args,
            )
            is preferred
        )
        with pytest.raises(RuntimeError, match="FMHA cache sanity check failed"):
            manager.select(
                attn,
                torch.empty((1, 8)),
                torch.empty((1, 3)),
                torch.empty((1, 3)),
                uncached_metadata,
                uncached_args,
            )

    log_error.assert_called_once()
    message = log_error.call_args.args[0]
    assert "cached=FakeFmha, uncached=FakeFmha" in message
    assert "q: cached=shape=(1, 4), uncached=shape=(1, 8)" in message
    assert "k: cached=shape=(1, 2), uncached=shape=(1, 3)" in message
    assert "v: cached=None, uncached=shape=(1, 3)" in message
    assert "metadata.beam_width: cached=1, uncached=2" in message
    assert "forward_args.relative_attention_max_distance: cached=0, uncached=7" in message


def test_fmha_cache_sanity_check_accepts_equivalent_combined_fmha() -> None:
    events: list[tuple] = []
    attn, manager = _make_manager(sanity_check_enabled=True)
    context_fmha = FakePhasedFmha(attn, {FmhaPhase.CONTEXT}, "context", events)
    generation_fmha = FakePhasedFmha(attn, {FmhaPhase.GENERATION}, "generation", events)
    manager.fmha_libs = [context_fmha, generation_fmha]
    metadata = _make_metadata(num_contexts=1, num_generations=1, num_ctx_tokens=1)
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.mixed)
    q = torch.empty((2, 4))

    with patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=True):
        first = manager.select(attn, q, None, None, metadata, forward_args)
        second = manager.select(attn, q, None, None, metadata, forward_args)

    assert isinstance(first, CombinedFmha)
    assert second is first
    assert events == [
        ("support", "context", None),
        ("support", "context", FmhaPhase.CONTEXT),
        ("support", "context", FmhaPhase.GENERATION),
        ("support", "generation", None),
        ("support", "generation", FmhaPhase.GENERATION),
        ("support", "context", None),
        ("support", "context", FmhaPhase.CONTEXT),
        ("support", "context", FmhaPhase.GENERATION),
        ("support", "generation", None),
        ("support", "generation", FmhaPhase.GENERATION),
    ]


def test_fmha_cache_keeps_combined_selections_immutable() -> None:
    events: list[tuple] = []
    attn, manager = _make_manager()
    context_small = FakePhasedFmha(
        attn,
        {FmhaPhase.CONTEXT},
        "context-small",
        events,
        support_predicate=lambda metadata, phase: metadata.num_contexts == 1,
    )
    context_large = FakePhasedFmha(
        attn,
        {FmhaPhase.CONTEXT},
        "context-large",
        events,
        support_predicate=lambda metadata, phase: metadata.num_contexts > 1,
    )
    generation_small = FakePhasedFmha(
        attn,
        {FmhaPhase.GENERATION},
        "generation-small",
        events,
        support_predicate=lambda metadata, phase: metadata.num_generations == 1,
    )
    generation_large = FakePhasedFmha(
        attn,
        {FmhaPhase.GENERATION},
        "generation-large",
        events,
        support_predicate=lambda metadata, phase: metadata.num_generations > 1,
    )
    manager.fmha_libs = [
        context_small,
        context_large,
        generation_small,
        generation_large,
    ]
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.mixed)
    small_metadata = _make_metadata(num_contexts=1, num_generations=1, num_ctx_tokens=2)
    large_metadata = _make_metadata(num_contexts=26, num_generations=26, num_ctx_tokens=52)

    with patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=True):
        small = manager.select(attn, torch.empty((3, 4)), None, None, small_metadata, forward_args)
        large = manager.select(attn, torch.empty((78, 4)), None, None, large_metadata, forward_args)
        num_events_after_misses = len(events)
        small_again = manager.select(
            attn, torch.empty((3, 4)), None, None, small_metadata, forward_args
        )

    assert isinstance(small, CombinedFmha)
    assert isinstance(large, CombinedFmha)
    assert small is small_again
    assert small is not large
    assert small._get_context_impl() is context_small
    assert small._get_generation_impl() is generation_small
    assert large._get_context_impl() is context_large
    assert large._get_generation_impl() is generation_large
    assert len(events) == num_events_after_misses


def test_fmha_cache_is_bypassed_while_autotuning() -> None:
    cases = (
        (False, AttentionInputType.mixed, 1, 1, 1, 2),
        (True, AttentionInputType.context_only, 1, 0, 1, 1),
        (True, AttentionInputType.generation_only, 0, 1, 0, 1),
    )
    for (
        is_mla_enable,
        attention_input_type,
        num_contexts,
        num_generations,
        num_ctx_tokens,
        num_q_tokens,
    ) in cases:
        events: list[tuple] = []
        attn, manager = _make_manager()
        attn.is_mla_enable = is_mla_enable
        fmha = FakeFmha(attn, "fmha", events)
        manager.fmha_libs = [fmha]
        metadata = _make_metadata(
            num_contexts=num_contexts,
            num_generations=num_generations,
            num_ctx_tokens=num_ctx_tokens,
        )
        forward_args = AttentionForwardArgs(attention_input_type=attention_input_type)
        q = torch.empty((num_q_tokens, 4))

        with patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=False):
            assert manager.select(attn, q, None, None, metadata, forward_args) is fmha
            assert manager.select(attn, q, None, None, metadata, forward_args) is fmha
        assert not manager._cache
        with patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=True):
            assert manager.select(attn, q, None, None, metadata, forward_args) is fmha
            assert manager.select(attn, q, None, None, metadata, forward_args) is fmha
        with patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=False):
            assert manager.select(attn, q, None, None, metadata, forward_args) is fmha
        with patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=True):
            assert manager.select(attn, q, None, None, metadata, forward_args) is fmha

        assert events == [
            ("support", "fmha", None),
            ("support", "fmha", None),
            ("support", "fmha", None),
            ("support", "fmha", None),
        ]


def test_fmha_cache_does_not_cache_failed_selection() -> None:
    events: list[tuple] = []
    attn, manager = _make_manager()
    unsupported = FakePhasedFmha(attn, set(), "unsupported", events)
    manager.fmha_libs = [unsupported]
    metadata = _make_metadata(num_contexts=1, num_generations=0, num_ctx_tokens=1)
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.context_only)
    q = torch.empty((1, 4))

    with patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=True):
        assert manager.select(attn, q, None, None, metadata, forward_args) is None
        assert manager.select(attn, q, None, None, metadata, forward_args) is None

    assert events == [
        ("support", "unsupported", None),
        ("support", "unsupported", FmhaPhase.CONTEXT),
        ("support", "unsupported", None),
        ("support", "unsupported", FmhaPhase.CONTEXT),
    ]


def test_update_quant_config_replaces_manager_with_fresh_cache() -> None:
    events: list[tuple] = []
    quant_states_during_construction: list[bool] = []

    class _OldFmha(FakeFmha):
        def __init__(self, attn: TrtllmAttention) -> None:
            super().__init__(attn, "old", events)

    class _NewFmha(FakeFmha):
        def __init__(self, attn: TrtllmAttention) -> None:
            quant_states_during_construction.append(attn.has_fp8_kv_cache)
            super().__init__(attn, "new", events)

    attn = TrtllmAttention.__new__(TrtllmAttention)
    attn.is_mla_enable = False
    metadata = _make_metadata(num_contexts=0, num_generations=1)
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)
    q = torch.empty((1, 4))

    with (
        patch.object(fmha_manager, "get_enabled_fmha_lib_classes", return_value=[_OldFmha]),
        patch.object(fmha_manager, "_is_fmha_cache_enabled", return_value=True),
    ):
        attn.update_quant_config(None)
        old_manager = attn._fmha_manager
        old = old_manager.select(attn, q, None, None, metadata, forward_args)

    with patch.object(
        fmha_manager,
        "get_enabled_fmha_lib_classes",
        return_value=[_NewFmha],
    ):
        attn.update_quant_config(QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8))

    assert isinstance(old, _OldFmha)
    assert len(old_manager._cache) == 1
    assert attn._fmha_manager is not old_manager
    assert attn._fmha_manager._cache == {}
    assert isinstance(attn._fmha_manager.fmha_libs[0], _NewFmha)
    assert quant_states_during_construction == [True]
    assert events == [("support", "old", None)]
