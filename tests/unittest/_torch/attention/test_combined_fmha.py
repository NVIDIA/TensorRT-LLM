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
from typing import Callable
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.attention_backend import trtllm as trtllm_backend
from tensorrt_llm._torch.attention_backend.fmha.combined import CombinedFmha
from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import FlashInferTrtllmGenFmha
from tensorrt_llm._torch.attention_backend.fmha.interface import Fmha, FmhaPhase
from tensorrt_llm._torch.attention_backend.fmha.phased import FmhaParams, PhasedFmha
from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs, AttentionInputType
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm.quantization.mode import QuantMode


class _TestAttention:
    def __init__(self) -> None:
        self.is_mla_enable = False
        self.kv_lora_rank = None
        self.v_head_dim = None
        self.head_dim = 4
        self.num_heads = 1
        self.num_kv_heads = 1
        self.predicted_tokens_per_seq = 1
        self.flashinfer_mla_backend = None
        self.has_fp8_kv_cache = False


class _TestPhasedFmha(PhasedFmha):
    def __init__(
        self,
        attn: _TestAttention,
        supported_phases: set[FmhaPhase],
        name: str,
        events: list[tuple],
        workspace_size: int = 0,
        support_predicate: Callable[[object, FmhaPhase | None], bool] | None = None,
    ) -> None:
        super().__init__(attn)
        self._supported_phases = supported_phases
        self._name = name
        self._events = events
        self._workspace_size = workspace_size
        self._support_predicate = support_predicate

    def is_supported(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: object,
        forward_args: AttentionForwardArgs,
        *,
        phase: FmhaPhase | None = None,
    ) -> bool:
        self._events.append(("support", self._name, phase))
        return phase in self._supported_phases and (
            self._support_predicate is None or self._support_predicate(metadata, phase)
        )

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: object,
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        self._events.append(("prepare", self._name))
        if workspace.numel() < self._workspace_size:
            workspace.resize_(self._workspace_size)

    def run_context(self, params: FmhaParams) -> None:
        self._events.append(("run", self._name, FmhaPhase.CONTEXT, params.num_tokens))

    def run_generation(self, params: FmhaParams) -> None:
        self._events.append(("run", self._name, FmhaPhase.GENERATION, params.num_tokens))


class _TestFmha(Fmha):
    def __init__(
        self,
        attn: _TestAttention,
        name: str,
        events: list[tuple],
        support_predicate: Callable[[AttentionForwardArgs], bool] | None = None,
    ) -> None:
        super().__init__(attn)
        self._name = name
        self._events = events
        self._support_predicate = support_predicate

    def is_supported(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: object,
        forward_args: AttentionForwardArgs,
        *,
        phase: FmhaPhase | None = None,
    ) -> bool:
        self._events.append(("support", self._name, phase))
        return self._support_predicate is None or self._support_predicate(forward_args)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: object,
        forward_args: AttentionForwardArgs,
    ) -> None:
        self._events.append(("forward", self._name))


def test_select_non_mla_fmha_combines_supported_phases() -> None:
    events: list[tuple] = []
    attn = _TestAttention()
    context_fmha = _TestPhasedFmha(attn, {FmhaPhase.CONTEXT}, "context", events)
    generation_fmha = _TestPhasedFmha(attn, {FmhaPhase.GENERATION}, "generation", events)
    backend = SimpleNamespace(
        fmha_libs=[context_fmha, generation_fmha],
        phased_fmha_libs=[context_fmha, generation_fmha],
    )

    selected = TrtllmAttention._select_non_mla_fmha(
        backend,
        torch.empty((2, 4)),
        None,
        None,
        SimpleNamespace(num_contexts=1, num_generations=1),
        AttentionForwardArgs(attention_input_type=AttentionInputType.mixed),
    )

    assert isinstance(selected, CombinedFmha)
    assert selected._get_context_impl() is context_fmha
    assert selected._get_generation_impl() is generation_fmha
    assert events == [
        ("support", "context", FmhaPhase.CONTEXT),
        ("support", "context", FmhaPhase.GENERATION),
        ("support", "generation", FmhaPhase.GENERATION),
    ]


def test_select_non_mla_fmha_checks_followup_support() -> None:
    events: list[tuple] = []
    attn = _TestAttention()
    context_fmha = _TestPhasedFmha(attn, {FmhaPhase.CONTEXT}, "context", events)
    unsupported_followup = _TestPhasedFmha(attn, set(), "followup", events)
    backend = SimpleNamespace(
        fmha_libs=[context_fmha, unsupported_followup],
        phased_fmha_libs=[context_fmha, unsupported_followup],
    )

    selected = TrtllmAttention._select_non_mla_fmha(
        backend,
        torch.empty((2, 4)),
        None,
        None,
        SimpleNamespace(num_contexts=1, num_generations=1),
        AttentionForwardArgs(attention_input_type=AttentionInputType.mixed),
    )

    assert selected is None
    assert events[-1] == ("support", "followup", FmhaPhase.GENERATION)


def test_select_non_mla_fmha_reuses_one_implementation() -> None:
    events: list[tuple] = []
    attn = _TestAttention()
    fmha = _TestPhasedFmha(
        attn,
        {FmhaPhase.CONTEXT, FmhaPhase.GENERATION},
        "both",
        events,
    )
    backend = SimpleNamespace(
        fmha_libs=[fmha],
        phased_fmha_libs=[fmha],
    )

    selected = TrtllmAttention._select_non_mla_fmha(
        backend,
        torch.empty((2, 4)),
        None,
        None,
        SimpleNamespace(num_contexts=1, num_generations=1),
        AttentionForwardArgs(attention_input_type=AttentionInputType.mixed),
    )

    assert selected is fmha


def test_select_non_mla_fmha_preserves_registry_order() -> None:
    events: list[tuple] = []
    attn = _TestAttention()
    non_phased_fmha = _TestFmha(attn, "non-phased", events)
    phased_fmha = _TestPhasedFmha(
        attn,
        {FmhaPhase.CONTEXT, FmhaPhase.GENERATION},
        "phased",
        events,
    )
    backend = SimpleNamespace(
        fmha_libs=[non_phased_fmha, phased_fmha],
        phased_fmha_libs=[phased_fmha],
    )

    selected = TrtllmAttention._select_non_mla_fmha(
        backend,
        torch.empty((2, 4)),
        None,
        None,
        SimpleNamespace(num_contexts=1, num_generations=1),
        AttentionForwardArgs(attention_input_type=AttentionInputType.mixed),
    )

    assert selected is non_phased_fmha
    assert events == [("support", "non-phased", None)]


def test_combined_fmha_delegates_phases_and_prepares_max_workspace() -> None:
    events: list[tuple] = []
    attn = _TestAttention()
    context_fmha = _TestPhasedFmha(
        attn,
        {FmhaPhase.CONTEXT},
        "context",
        events,
        workspace_size=8,
    )
    generation_fmha = _TestPhasedFmha(
        attn,
        {FmhaPhase.GENERATION},
        "generation",
        events,
        workspace_size=4,
    )
    combined_fmha = CombinedFmha(attn)
    combined_fmha.set_fmha_impls(context_fmha, generation_fmha)
    metadata = SimpleNamespace(
        kv_cache_block_offsets=object(),
        effective_workspace=torch.empty(0, dtype=torch.uint8),
        num_contexts=1,
        num_ctx_tokens=2,
        num_generations=1,
        kv_lens_cuda_runtime=torch.tensor([2, 5], dtype=torch.int32),
        kv_lens_runtime=torch.tensor([2, 5], dtype=torch.int32),
        prompt_lens_cuda_runtime=torch.tensor([2, 1], dtype=torch.int32),
        prompt_lens_cpu_runtime=torch.tensor([2, 1], dtype=torch.int32),
        beam_width=1,
        cache_indirection=None,
        tokens_per_block=32,
        kv_cache_manager=None,
        is_cross=False,
        is_spec_decoding_enabled=False,
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((3, 4)),
        attention_input_type=AttentionInputType.mixed,
        attention_window_size=8,
    )

    combined_fmha.forward(torch.empty((3, 4)), None, None, metadata, forward_args)

    assert events == [
        ("prepare", "context"),
        ("prepare", "generation"),
        ("run", "context", FmhaPhase.CONTEXT, 2),
        ("run", "generation", FmhaPhase.GENERATION, 1),
    ]
    assert metadata.effective_workspace.numel() == 8


def test_combined_fmha_uses_flattened_v2_page_bound() -> None:
    attn = _TestAttention()
    combined_fmha = CombinedFmha(attn)
    kv_cache_manager = SimpleNamespace(
        impl=SimpleNamespace(get_page_index_upper_bound=lambda: 23),
        blocks_in_primary_pool=23,
        num_local_layers=4,
    )

    assert (
        combined_fmha._get_total_num_blocks(SimpleNamespace(kv_cache_manager=kv_cache_manager))
        == 23
    )


def test_flashinfer_fp8_mode_remains_implementation_local() -> None:
    attn = _TestAttention()
    attn.quant_mode = int(QuantMode.from_description(use_fp8_kv_cache=True))
    fmha = FlashInferTrtllmGenFmha(attn)
    output = torch.empty(1, dtype=torch.bfloat16)

    assert fmha._use_fp8_context_fmha(output, AttentionInputType.context_only)
    assert fmha._use_fp8_context_fmha(output, AttentionInputType.mixed)
    assert not fmha._use_fp8_context_fmha(output, AttentionInputType.generation_only)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sm_version", [100, 103])
@pytest.mark.parametrize("tokens_per_block", [32, 64])
@pytest.mark.parametrize("num_contexts", [1, 4, 5])
def test_flashinfer_context_fallback_scope(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    sm_version: int,
    tokens_per_block: int,
    num_contexts: int,
) -> None:
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen.get_sm_version",
        lambda: sm_version,
    )
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha.kv_factor = 2
    q = torch.empty((num_contexts, 64), dtype=dtype)
    attn = SimpleNamespace(
        is_mla_enable=False,
        sparse_params=None,
        position_embedding_type=0,
        head_dim=64,
        num_heads=1,
        num_kv_heads=1,
    )
    metadata = SimpleNamespace(
        num_contexts=num_contexts,
        helix_position_offsets=None,
        num_sparse_topk=0,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        kv_cache_block_offsets=object(),
        kv_cache_manager=None,
        is_cross=False,
        is_spec_decoding_enabled=False,
        tokens_per_block=tokens_per_block,
        beam_width=1,
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty_like(q),
        attention_input_type=AttentionInputType.mixed,
    )

    supported, reason = fmha._is_supported_with_reason(
        q,
        None,
        None,
        attn,
        metadata,
        forward_args,
        phase=FmhaPhase.CONTEXT,
    )

    expected_fallback = (dtype == torch.bfloat16 and num_contexts <= 4) or sm_version == 103
    if expected_fallback:
        assert not supported
        assert "fallback FMHA" in reason
    else:
        assert supported, reason
        assert reason == ""

    generation_supported, generation_reason = fmha._is_supported_with_reason(
        q,
        None,
        None,
        attn,
        metadata,
        forward_args,
        phase=FmhaPhase.GENERATION,
    )
    assert generation_supported, generation_reason
    assert generation_reason == ""


def _make_selection_backend() -> TrtllmAttention:
    backend = object.__new__(TrtllmAttention)
    backend.is_mla_enable = False
    backend.fmha_libs = []
    backend.phased_fmha_libs = []
    backend.non_phased_fmha_libs = []
    backend._fmha_selection_cache = {}
    return backend


def _make_selection_metadata(
    *,
    num_contexts: int,
    num_generations: int,
    num_ctx_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_contexts=num_contexts,
        num_generations=num_generations,
        num_ctx_tokens=num_ctx_tokens,
    )


def test_normalize_fmha_selection_grid_value() -> None:
    grid = (1, 2, 4)

    assert trtllm_backend._normalize_fmha_selection_grid_value(0, grid) == 0
    assert trtllm_backend._normalize_fmha_selection_grid_value(1, grid) == 1
    assert trtllm_backend._normalize_fmha_selection_grid_value(3, grid) == 4
    assert trtllm_backend._normalize_fmha_selection_grid_value(4, grid) == 4
    assert trtllm_backend._normalize_fmha_selection_grid_value(6, grid) == 4
    assert (
        trtllm_backend._normalize_fmha_selection_grid_value(
            25, trtllm_backend._FMHA_SELECTION_BATCH_SIZE_GRID
        )
        == 26
    )
    assert (
        trtllm_backend._normalize_fmha_selection_grid_value(
            3, trtllm_backend._FMHA_SELECTION_SEQ_LEN_Q_GRID
        )
        == 4
    )


def test_fmha_selection_cache_uses_normalized_batch_and_q_grids() -> None:
    events: list[tuple] = []
    backend = _make_selection_backend()
    fmha = _TestFmha(backend, "fmha", events)
    backend.fmha_libs = [fmha]
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)

    with patch.object(trtllm_backend, "_is_fmha_selection_cache_enabled", return_value=True):
        first = backend._select_fmha(
            torch.empty((25 * 3, 4)),
            None,
            None,
            _make_selection_metadata(num_contexts=0, num_generations=25),
            forward_args,
        )
        second = backend._select_fmha(
            torch.empty((26 * 4, 4)),
            None,
            None,
            _make_selection_metadata(num_contexts=0, num_generations=26),
            forward_args,
        )
        third = backend._select_fmha(
            torch.empty((27 * 4, 4)),
            None,
            None,
            _make_selection_metadata(num_contexts=0, num_generations=27),
            forward_args,
        )

    assert first is fmha
    assert second is fmha
    assert third is fmha
    assert events == [
        ("support", "fmha", None),
        ("support", "fmha", None),
    ]


def test_context_fmha_selection_cache_uses_only_batch_grid() -> None:
    events: list[tuple] = []
    backend = _make_selection_backend()
    fmha = _TestFmha(backend, "fmha", events)
    backend.fmha_libs = [fmha]
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.context_only)

    with patch.object(trtllm_backend, "_is_fmha_selection_cache_enabled", return_value=True):
        first = backend._select_fmha(
            torch.empty((25 * 3, 4)),
            None,
            None,
            _make_selection_metadata(num_contexts=25, num_generations=0, num_ctx_tokens=25 * 3),
            forward_args,
        )
        second = backend._select_fmha(
            torch.empty((26 * 17, 4)),
            None,
            None,
            _make_selection_metadata(num_contexts=26, num_generations=0, num_ctx_tokens=26 * 17),
            forward_args,
        )

    assert first is fmha
    assert second is fmha
    assert events == [("support", "fmha", None)]


def test_fmha_selection_cache_key_tracks_phase_composition() -> None:
    backend = _make_selection_backend()
    removed_fields = {"attention_input_type", "has_context", "has_generation"}
    assert removed_fields.isdisjoint(trtllm_backend._FmhaSelectionCacheKey._fields)

    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)
    pure_generation = backend._make_fmha_selection_cache_key(
        torch.empty((2, 4)),
        _make_selection_metadata(num_contexts=0, num_generations=2),
        forward_args,
    )
    mixed_generation = backend._make_fmha_selection_cache_key(
        torch.empty((2, 4)),
        _make_selection_metadata(num_contexts=3, num_generations=2),
        forward_args,
    )

    assert pure_generation != mixed_generation
    assert pure_generation.context_batch_size == 0
    assert mixed_generation.context_batch_size == 3
    assert pure_generation.generation_batch_size == mixed_generation.generation_batch_size
    assert pure_generation.generation_seq_len_q == mixed_generation.generation_seq_len_q


def test_fmha_selection_cache_key_distinguishes_compacted_phases() -> None:
    backend = _make_selection_backend()
    backend.is_mla_enable = True
    metadata = _make_selection_metadata(num_contexts=3, num_generations=2, num_ctx_tokens=6)
    context = backend._make_fmha_selection_cache_key(
        torch.empty((6, 4)),
        metadata,
        AttentionForwardArgs(attention_input_type=AttentionInputType.context_only),
    )
    generation = backend._make_fmha_selection_cache_key(
        torch.empty((2, 4)),
        metadata,
        AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only),
    )

    assert context != generation
    assert context.context_batch_size == generation.context_batch_size
    assert context.generation_batch_size == generation.generation_batch_size
    assert context.generation_seq_len_q == 0
    assert generation.generation_seq_len_q == 1


def test_fmha_selection_cache_key_tracks_mixed_generation_q_length() -> None:
    backend = _make_selection_backend()
    metadata = _make_selection_metadata(
        num_contexts=1,
        num_generations=1,
        num_ctx_tokens=32,
    )
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.mixed)

    single_token = backend._make_fmha_selection_cache_key(
        torch.empty((33, 4)), metadata, forward_args
    )
    four_tokens = backend._make_fmha_selection_cache_key(
        torch.empty((36, 4)), metadata, forward_args
    )

    assert single_token.context_batch_size == four_tokens.context_batch_size
    assert single_token.generation_batch_size == four_tokens.generation_batch_size
    assert single_token.generation_seq_len_q == 1
    assert four_tokens.generation_seq_len_q == 4
    assert single_token != four_tokens


def test_fmha_selection_cache_tracks_lora_output_representation() -> None:
    for lora_first in (True, False):
        events: list[tuple] = []
        backend = _make_selection_backend()
        unpacked_only = _TestFmha(
            backend,
            "unpacked-only",
            events,
            support_predicate=lambda forward_args: forward_args.output is not None
            and forward_args.output.dtype == torch.bfloat16,
        )
        fallback = _TestFmha(backend, "fallback", events)
        backend.fmha_libs = [unpacked_only, fallback]
        metadata = _make_selection_metadata(num_contexts=0, num_generations=4)
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

        with patch.object(trtllm_backend, "_is_fmha_selection_cache_enabled", return_value=True):
            for forward_args in request_order:
                fresh = backend._select_fmha_uncached(q, None, None, metadata, forward_args)
                cached = backend._select_fmha(q, None, None, metadata, forward_args)

                assert cached is fresh

        assert backend._select_fmha(q, None, None, metadata, lora_args) is unpacked_only
        assert backend._select_fmha(q, None, None, metadata, base_args) is fallback
        assert len(backend._fmha_selection_cache) == 2


def test_fmha_selection_cache_keeps_combined_selections_immutable() -> None:
    events: list[tuple] = []
    backend = _make_selection_backend()
    context_small = _TestPhasedFmha(
        backend,
        {FmhaPhase.CONTEXT},
        "context-small",
        events,
        support_predicate=lambda metadata, phase: metadata.num_contexts == 1,
    )
    context_large = _TestPhasedFmha(
        backend,
        {FmhaPhase.CONTEXT},
        "context-large",
        events,
        support_predicate=lambda metadata, phase: metadata.num_contexts > 1,
    )
    generation_small = _TestPhasedFmha(
        backend,
        {FmhaPhase.GENERATION},
        "generation-small",
        events,
        support_predicate=lambda metadata, phase: metadata.num_generations == 1,
    )
    generation_large = _TestPhasedFmha(
        backend,
        {FmhaPhase.GENERATION},
        "generation-large",
        events,
        support_predicate=lambda metadata, phase: metadata.num_generations > 1,
    )
    backend.fmha_libs = [
        context_small,
        context_large,
        generation_small,
        generation_large,
    ]
    backend.phased_fmha_libs = list(backend.fmha_libs)
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.mixed)
    small_metadata = _make_selection_metadata(num_contexts=1, num_generations=1, num_ctx_tokens=2)
    large_metadata = _make_selection_metadata(
        num_contexts=26, num_generations=26, num_ctx_tokens=52
    )

    with patch.object(trtllm_backend, "_is_fmha_selection_cache_enabled", return_value=True):
        small = backend._select_fmha(torch.empty((3, 4)), None, None, small_metadata, forward_args)
        large = backend._select_fmha(torch.empty((78, 4)), None, None, large_metadata, forward_args)
        num_events_after_misses = len(events)
        small_again = backend._select_fmha(
            torch.empty((3, 4)), None, None, small_metadata, forward_args
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


def test_fmha_selection_cache_is_bypassed_while_autotuning() -> None:
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
        backend = _make_selection_backend()
        backend.is_mla_enable = is_mla_enable
        fmha = _TestFmha(backend, "fmha", events)
        backend.fmha_libs = [fmha]
        metadata = _make_selection_metadata(
            num_contexts=num_contexts,
            num_generations=num_generations,
            num_ctx_tokens=num_ctx_tokens,
        )
        forward_args = AttentionForwardArgs(attention_input_type=attention_input_type)
        q = torch.empty((num_q_tokens, 4))

        with patch.object(trtllm_backend, "_is_fmha_selection_cache_enabled", return_value=False):
            assert backend._select_fmha(q, None, None, metadata, forward_args) is fmha
            assert backend._select_fmha(q, None, None, metadata, forward_args) is fmha
        assert not backend._fmha_selection_cache
        with patch.object(trtllm_backend, "_is_fmha_selection_cache_enabled", return_value=True):
            assert backend._select_fmha(q, None, None, metadata, forward_args) is fmha
            assert backend._select_fmha(q, None, None, metadata, forward_args) is fmha
        with patch.object(trtllm_backend, "_is_fmha_selection_cache_enabled", return_value=False):
            assert backend._select_fmha(q, None, None, metadata, forward_args) is fmha
        with patch.object(trtllm_backend, "_is_fmha_selection_cache_enabled", return_value=True):
            assert backend._select_fmha(q, None, None, metadata, forward_args) is fmha

        assert events == [
            ("support", "fmha", None),
            ("support", "fmha", None),
            ("support", "fmha", None),
            ("support", "fmha", None),
        ]


def test_fmha_selection_cache_does_not_cache_failed_selection() -> None:
    events: list[tuple] = []
    backend = _make_selection_backend()
    unsupported = _TestPhasedFmha(backend, set(), "unsupported", events)
    backend.fmha_libs = [unsupported]
    metadata = _make_selection_metadata(num_contexts=1, num_generations=0, num_ctx_tokens=1)
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.context_only)
    q = torch.empty((1, 4))

    with patch.object(trtllm_backend, "_is_fmha_selection_cache_enabled", return_value=True):
        assert backend._select_fmha(q, None, None, metadata, forward_args) is None
        assert backend._select_fmha(q, None, None, metadata, forward_args) is None

    assert events == [
        ("support", "unsupported", FmhaPhase.CONTEXT),
        ("support", "unsupported", FmhaPhase.CONTEXT),
    ]


def test_create_fmha_libs_invalidates_selection_cache() -> None:
    events: list[tuple] = []

    class _OldFmha(_TestFmha):
        def __init__(self, attn: _TestAttention) -> None:
            super().__init__(attn, "old", events)

    class _NewFmha(_TestFmha):
        def __init__(self, attn: _TestAttention) -> None:
            super().__init__(attn, "new", events)

    backend = _make_selection_backend()
    backend.sparse_params = None
    backend.kv_cache_dtype = "auto"
    metadata = _make_selection_metadata(num_contexts=0, num_generations=1)
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)
    q = torch.empty((1, 4))

    with (
        patch.object(trtllm_backend, "get_enabled_fmha_lib_classes", return_value=[_OldFmha]),
        patch.object(trtllm_backend, "_is_fmha_selection_cache_enabled", return_value=True),
    ):
        backend.create_fmha_libs()
        old = backend._select_fmha(q, None, None, metadata, forward_args)

    with (
        patch.object(trtllm_backend, "get_enabled_fmha_lib_classes", return_value=[_NewFmha]),
        patch.object(trtllm_backend, "_is_fmha_selection_cache_enabled", return_value=True),
    ):
        backend.create_fmha_libs()
        new = backend._select_fmha(q, None, None, metadata, forward_args)

    assert isinstance(old, _OldFmha)
    assert isinstance(new, _NewFmha)
    assert events == [
        ("support", "old", None),
        ("support", "new", None),
    ]
