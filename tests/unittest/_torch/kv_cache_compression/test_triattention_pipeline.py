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

"""Unit tests for the TriAttention compression-manager pipeline.

TriAttention is a pure KV-cache compression method on the PR-15106 framework: it
has NO sparse-attention config and NO attention backend of its own. Decode runs
the model's standard attention over the compacted cache; the manager publishes
the cumulative evicted count on ``LlmRequest.py_num_compressed_tokens`` and the
model engine subtracts it where it builds ``num_cached_tokens_per_seq``. These
tests cover the config, construction, eviction lifecycle, page-table staging,
and the fixed score buffers. Draft co-compression contracts live in
``test_triattention_draft_cocompaction.py``; model-level correctness is covered
by separate end-to-end tests.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from conftest import encode_block_offsets as _encode_block_offsets
from conftest import make_bare_staging as _make_bare_staging
from conftest import make_fake_v2 as _make_fake_v2
from conftest import make_request as _make_request
from conftest import make_staging_manager as _make_staging_manager
from conftest import make_triattention as _make_triattention
from conftest import make_workspace_stubs as _make_workspace_stubs
from conftest import mocked_eviction_internals as _mocked_eviction_internals
from conftest import torch_tri_score_oracle as _torch_tri_score_oracle
from pydantic import ValidationError

# TriAttention lives in the kv_cache_compression package. It exposes only the
# compression manager -- no attention classes or KV-cache-manager subclass.
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import TriAttention

# Framework base class lives in pyexecutor.resource_manager; the factory lives
# in pyexecutor._util (next to _create_kv_cache_manager), matching #15106.
from tensorrt_llm._torch.pyexecutor._util import create_kv_cache_compression_manager
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import Role
from tensorrt_llm.llmapi.llm_args import TriAttentionKvCacheCompressionConfig

# The SM100 CuTe kernel is the only score path, so every test that actually
# launches scores (or builds the real staging buffers, whose constructor
# compiles the kernel) is SM100-only, like the production feature itself.
requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="TriAttention score requires SM100",
)


def _set_request_state(
    manager,
    request_id,
    *,
    generation_steps=0,
    evicted_tokens=0,
    confirmed_kv_length=None,
):
    state = {
        "generation_steps": generation_steps,
        "evicted_tokens": evicted_tokens,
        "confirmed_kv_length": confirmed_kv_length,
    }
    manager._request_states[request_id] = state
    return state


def _prepared_eviction(
    request,
    *,
    seq_len,
    expected_keep_count,
    protected_tail=0,
    request_id=0,
    round_start=None,
    prompt_len=0,
):
    return {
        "request": request,
        "request_id": request_id,
        "seq_len": seq_len,
        "round_start": int(seq_len if round_start is None else round_start),
        "prompt_len": prompt_len,
        "expected_keep_count": expected_keep_count,
        "protected_tail": protected_tail,
    }


@pytest.fixture
def flat_calibration_pt(tmp_path):
    """Build a minimal valid calibration ``.pt`` in our flat runtime schema."""
    path = tmp_path / "tri_calib.pt"
    calibration = {
        "E_q": torch.zeros(2, 2, 4, dtype=torch.complex64),
        "E_q_norm": torch.ones(2, 2, 4, dtype=torch.float32),
        "omega": torch.arange(4, dtype=torch.float32),
        "freq_scale_sq": torch.ones(4, dtype=torch.float32),
    }
    torch.save(calibration, path)
    return str(path)


def _make_hf_config(**values):
    """Expose the normalized Hugging Face text-config contract."""
    text_config = SimpleNamespace(to_dict=lambda: dict(values))
    return SimpleNamespace(get_text_config=lambda: text_config)


class TestConfigAndFactory:
    def test_llm_args_dispatch_and_validation(self):
        from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

        tri_args = TorchLlmArgs(
            model="dummy",
            kv_cache_compression_config={
                "algorithm": "triattention",
                "model_path": "/models/test",
                "calibration_path": "/calib/test.pt",
            },
        )
        assert isinstance(
            tri_args.kv_cache_compression_config,
            TriAttentionKvCacheCompressionConfig,
        )
        assert tri_args.kv_cache_compression_config.top_B == 2048
        assert tri_args.kv_cache_compression_config.beta == 128

        # The union dispatches on the algorithm tag, so an unknown algorithm
        # fails config validation instead of falling back to a base config.
        with pytest.raises(ValidationError):
            TorchLlmArgs(
                model="dummy",
                kv_cache_compression_config={"algorithm": "future_method"},
            )
        with pytest.raises(ValidationError):
            TriAttentionKvCacheCompressionConfig(eviction_mode="made_up_mode")

    def test_factory_returns_triattention_and_propagates_config_fields(self):
        # A plain V2 manager (block reuse off) yields a TriAttention instance.
        # Calibration is deferred to the first request, so construction needs
        # no calibration file or CUDA.
        fake_v2 = _make_fake_v2(enable_block_reuse=False)
        cfg = TriAttentionKvCacheCompressionConfig(
            top_B=32,
            beta=16,
            eviction_mode="per_head",
            model_path="/models/test",
            calibration_path="/calib/test.pt",
        )
        mgr = create_kv_cache_compression_manager(cfg, kv_cache_manager=fake_v2)
        assert isinstance(mgr, TriAttention)
        assert mgr.top_B == 32
        assert mgr.beta == 16
        assert mgr.eviction_mode == "per_head"
        assert mgr.kv_cache_manager is fake_v2


class TestTriAttentionClass:
    def test_cached_layout_checks_page_counts_without_rebuilding_pool_views(self):
        page_count_query = mock.Mock(side_effect=[8, 16, 8, 18])
        manager = SimpleNamespace(
            get_buffers=mock.Mock(side_effect=AssertionError("pool view was rebuilt")),
            impl=SimpleNamespace(get_page_index_upper_bound=page_count_query),
            kv_factor=2,
            layer_offsets={10: 100, 11: 101, 12: 102},
        )
        triattention = _make_triattention()
        triattention.kv_cache_manager = manager
        cached = dict(
            manager=manager,
            num_layers=3,
            global_layers=[10, 11, 12],
            layer_pools=[torch.empty(4), torch.empty(8), torch.empty(4)],
            dense_layers=[0, 1, 2],
            swa_layers=[],
            swa_window=None,
            storage_groups={0: [0, 2], 1: [1]},
            layer_group_representative={0: 0, 1: 1, 2: 0},
            layer_pool_keys=(0, 1, 0),
            # These are local layer slots. Layer 2 shares layer 0's pool.
            pool_representatives=(0, 1),
            pool_page_counts=(4, 8),
            pool_view_fingerprint=(),
        )
        triattention._runtime_kv_layout_cache = cached

        assert triattention._runtime_kv_layout(3) is cached
        manager.get_buffers.assert_not_called()
        assert page_count_query.call_args_list == [
            mock.call(100, Role.KEY),
            mock.call(101, Role.KEY),
        ]

        with pytest.raises(RuntimeError, match="pool layout changed"):
            triattention._runtime_kv_layout(3)
        manager.get_buffers.assert_not_called()
        assert page_count_query.call_args_list == [
            mock.call(100, Role.KEY),
            mock.call(101, Role.KEY),
            mock.call(100, Role.KEY),
            mock.call(101, Role.KEY),
        ]

    @pytest.mark.parametrize("num_extra_kv_tokens,reserved_draft", [(0, 0), (4, 4)])
    def test_request_init_marks_capacity_only_and_tracks_state(
        self, num_extra_kv_tokens, reserved_draft
    ):
        # Speculative capacity (extra KV tokens / reserved draft width) is
        # accepted at request init; the target manager is marked so V2 sizing
        # keeps logical max_seq_len while capacity is reclaimed and reused.
        manager = _make_fake_v2()
        manager.num_extra_kv_tokens = num_extra_kv_tokens
        manager._kv_reserve_draft_tokens = reserved_draft
        triattention = TriAttention(manager, top_B=8, model_path="/models/test")
        triattention._attention_layer_partition_cache = ([], [], None)
        triattention._calibrated = True

        triattention.on_request_init(_make_request(11))
        triattention.on_request_init(_make_request(12))

        assert triattention.adjusts_generation_kv_length is True
        assert manager.kv_compression_manages_history
        assert set(triattention._request_states) == {11, 12}

    def test_resolve_accepts_flat_pt(self, flat_calibration_pt):
        mgr = _make_triattention()
        mgr.calibration_path = flat_calibration_pt
        mgr.model_path = None
        loaded = mgr._resolve_calibration()
        for key in ("E_q", "E_q_norm", "omega", "freq_scale_sq"):
            assert key in loaded


class TestCompressedTokenPublication:
    # The cumulative/monotone publication contract itself (including the
    # uncompressed round_start restoration) is covered end to end by
    # test_triattention_draft_cocompaction.py::
    # test_compressed_count_is_monotone_and_tracks_confirmed_length.

    def test_identity_compaction_is_rejected_instead_of_published(self):
        manager = _make_triattention(top_B=4)
        manager.kv_cache_manager._stream = mock.Mock()
        request = _make_request(7, py_prompt_len=2)
        # Selection keeps every token: seq_len == prompt + budget + 1 evicts
        # one token; seq_len == prompt + budget must never publish.
        _set_request_state(manager, 7, confirmed_kv_length=6)

        with _mocked_eviction_internals(manager):
            assert manager._evict_requests([(request, 7)], 2) == []
        assert request.py_num_compressed_tokens == 0


class TestEvictionLifecycle:
    def test_prepare_does_not_evict_and_update_runs_final_hook_once(self):
        # Structural: TriAttention implements hooks only, never the base
        # template methods; the growth snapshot rides the step-begin hook and
        # the eviction runs from the final on_generation_step_end hook.
        assert "prepare_resources" not in TriAttention.__dict__
        assert "update_resources" not in TriAttention.__dict__
        assert "on_generation_step_begin" in TriAttention.__dict__
        assert "on_generation_step_end" in TriAttention.__dict__

        manager = _make_triattention()
        batch = SimpleNamespace(
            context_requests=[],
            context_requests_last_chunk=[],
            generation_requests=[_make_request(7)],
        )
        with mock.patch.object(manager, "_periodic_evict") as periodic_evict:
            manager.prepare_resources(batch)
            periodic_evict.assert_not_called()

            manager.update_resources(batch)

        periodic_evict.assert_called_once_with(batch)

    @staticmethod
    def _make_due_decode_request(seq_len):
        request = _make_request(
            7,
            py_prompt_len=1024,
            max_beam_num_tokens=seq_len + 1,
        )
        batch = SimpleNamespace(generation_requests=[request])
        mgr = _make_triattention()
        mgr._calibrated = True
        cache = SimpleNamespace(
            capacity=seq_len,
            history_length=1024,
            is_active=True,
            resize=mock.Mock(return_value=True),
        )
        mgr.kv_cache_manager = SimpleNamespace(
            get_buffers=lambda *args, **kwargs: None,
            kv_cache_map={7: cache},
            pp_layers=[0, 1],
            _stream=mock.Mock(),
            num_extra_kv_tokens=0,
        )
        mgr._L = 2
        mgr._request_states = {}
        _set_request_state(mgr, 7, generation_steps=127)
        mgr.beta = 128
        mgr.top_B = 4096
        return mgr, request, batch

    def test_identity_gate_preserves_real_eviction_round(self):
        mgr, request, batch = self._make_due_decode_request(seq_len=1024 + 4096 + 1)
        cache = mgr.kv_cache_manager.kv_cache_map[7]

        def compact(*args, protected_tail_lengths, **_kwargs):
            assert protected_tail_lengths == {7: 0}
            return [(7, 1024 + 4096)]

        with mock.patch.object(mgr, "_evict_requests", side_effect=compact) as evict:
            mgr._periodic_evict(batch)

        evict.assert_called_once_with(
            [(request, 7)],
            2,
            protected_tail_lengths={7: 0},
        )
        mgr.kv_cache_manager._stream.wait_event.assert_not_called()
        cache.resize.assert_called_once_with(1024 + 4096, None)

    def test_suspended_cache_rejects_batch_before_cadence_mutation(self):
        manager, first_request, _ = self._make_due_decode_request(seq_len=1024 + 4096 + 1)
        second_request = _make_request(8, py_prompt_len=1024)
        manager.kv_cache_manager.kv_cache_map[8] = SimpleNamespace(is_active=False)
        first_state = manager._request_states[7]
        second_state = _set_request_state(manager, 8, generation_steps=127)
        batch = SimpleNamespace(generation_requests=[first_request, second_request])

        with pytest.raises(RuntimeError, match="request 8 must be resumed"):
            manager._periodic_evict(batch)

        assert first_state["generation_steps"] == 127
        assert first_state["confirmed_kv_length"] is None
        assert second_state["generation_steps"] == 127
        assert second_state["confirmed_kv_length"] is None

    def test_non_boundary_step_skips_eviction_geometry(self):
        manager, _, batch = self._make_due_decode_request(seq_len=1024 + 4096 + 1)
        state = manager._request_states[7]
        state["generation_steps"] = 126

        with mock.patch.object(manager, "_minimum_evictable_length") as keep_count:
            manager._periodic_evict(batch)

        keep_count.assert_not_called()
        assert state["generation_steps"] == 127
        assert state["confirmed_kv_length"] == 1024 + 4096 + 1

    def test_eager_eviction_runs_large_due_cohort_in_one_round(self):
        manager, _, _ = self._make_due_decode_request(seq_len=1024 + 4096 + 1)
        requests = []
        caches = {}
        for request_id in range(65):
            request = _make_request(request_id, py_prompt_len=1024)
            requests.append(request)
            caches[request_id] = SimpleNamespace(
                capacity=1024 + 4096 + 1,
                history_length=1024,
                is_active=True,
            )
            _set_request_state(manager, request_id, generation_steps=127)
        manager.kv_cache_manager.kv_cache_map = caches
        batch = SimpleNamespace(generation_requests=requests)

        with (
            mock.patch.object(manager, "_evict_requests", return_value=[]) as evict,
            mock.patch.object(manager, "_resize_compacted_requests") as resize,
        ):
            manager._periodic_evict(batch)

        assert [len(call.args[0]) for call in evict.call_args_list] == [65]
        assert resize.call_count == 1

    def test_request_finish_clears_state_but_keeps_buffers_resident(self):
        manager = _make_triattention()
        _set_request_state(
            manager,
            7,
            generation_steps=1,
            evicted_tokens=127,
            confirmed_kv_length=128,
        )
        workspace = object()
        manager._workspace = workspace
        manager._prepared_generation_batch = (SimpleNamespace(), {7: 1})

        manager.on_request_finish(_make_request(7))

        assert manager._request_states == {}
        assert manager._prepared_generation_batch[1] == {}
        # The workspace is sized for the executor limits, not one cohort, so
        # it stays resident for the next generation batch.
        assert manager._workspace is workspace

    @pytest.mark.parametrize("accepted", [0, 1, 2, 3])
    def test_overlap_tail_is_excluded_from_selection_and_compacted(self, accepted):
        confirmed = 1024 + 4096 + 1 + accepted
        reserve = 2
        current_growth = 4
        tail = reserve + current_growth
        retained = 1024 + 4096
        mgr, request, batch = self._make_due_decode_request(seq_len=confirmed)
        request.py_num_accepted_draft_tokens = accepted
        cache = mgr.kv_cache_manager.kv_cache_map[7]
        cache.capacity = confirmed + tail
        mgr.kv_cache_manager.num_extra_kv_tokens = reserve
        mgr._prepared_generation_batch = (
            SimpleNamespace(generation_requests=[request]),
            {7: current_growth},
        )
        draft_manager = _make_fake_v2(is_draft=True)
        draft_cache = SimpleNamespace(is_active=True, resize=mock.Mock(return_value=True))
        draft_manager.kv_cache_map = {7: draft_cache}
        draft_manager._stream = mock.Mock()
        mgr.draft_kv_cache_manager = draft_manager

        def compact(*_args, **_kwargs):
            mgr._request_states[7]["confirmed_kv_length"] = retained
            return [(7, retained)]

        with mock.patch.object(mgr, "_evict_requests", side_effect=compact) as evict:
            mgr._periodic_evict(batch)

        evict.assert_called_once_with(
            [(request, 7)],
            2,
            protected_tail_lengths={7: tail},
        )
        assert mgr._request_states[7]["confirmed_kv_length"] == retained
        cache.resize.assert_called_once_with(retained + tail, None)
        # The draft cache shrinks in the same round, to the same retained
        # length plus the draft's own protected tail.
        draft_cache.resize.assert_called_once_with(retained + 1, None)

    def test_missing_draft_cache_fails_the_due_eviction_round(self):
        mgr, request, batch = self._make_due_decode_request(seq_len=1024 + 4096 + 1)
        mgr.draft_kv_cache_manager = _make_fake_v2(is_draft=True)

        with mock.patch.object(mgr, "_evict_requests") as evict:
            with pytest.raises(RuntimeError, match="missing or.*suspended draft KV cache"):
                mgr._periodic_evict(batch)
        evict.assert_not_called()

    def test_confirmed_length_comes_from_capacity_ledger_not_logical_length(self):
        physical_confirmed = 6100
        manager = _make_triattention(beta=128)
        manager._calibrated = True
        _set_request_state(manager, 7, evicted_tokens=100)
        cache = SimpleNamespace(
            capacity=physical_confirmed,
            history_length=1024,
            is_active=True,
            resize=mock.Mock(return_value=True),
        )
        manager.kv_cache_manager.kv_cache_map = {7: cache}
        manager.kv_cache_manager.pp_layers = [0, 1]
        manager.kv_cache_manager.num_extra_kv_tokens = 0
        request = _make_request(
            7,
            py_prompt_len=1024,
            max_beam_num_tokens=999999,
            py_draft_tokens=[1, 2, 3, 4],
        )

        manager._periodic_evict(SimpleNamespace(generation_requests=[request]))

        assert manager._request_states[7]["confirmed_kv_length"] == physical_confirmed
        cache.resize.assert_not_called()

    def test_mla_selfkonly_cache_is_rejected(self):
        manager = _make_triattention()
        manager.kv_cache_manager.kv_factor = 1

        with pytest.raises(ValueError, match="standard key/value KV cache"):
            manager._validate_v2_compatibility()

    def test_one_model_draft_co_compression_contract_is_accepted(self):
        # Co-compression keeps the draft's physical length equal to the
        # target's, so a draft with a smaller max_seq_len than the target's
        # logical maximum is accepted, and both managers are marked as
        # diverging from the logical length.
        draft_manager = _make_fake_v2(is_draft=True)
        draft_manager.max_seq_len = 8192
        manager = TriAttention(
            _make_fake_v2(),
            top_B=8,
            model_path="/models/test",
            draft_kv_cache_manager=draft_manager,
        )

        manager._validate_v2_compatibility()
        assert manager.kv_cache_manager.kv_compression_manages_history is True
        assert draft_manager.kv_compression_manages_history is True

    def test_resize_shrinks_draft_cache_with_its_own_protected_tail(self):
        retained = 1024 + 4096
        manager = _make_triattention()
        target_cache = SimpleNamespace(
            capacity=retained + 10,
            is_active=True,
            resize=mock.Mock(return_value=True),
        )
        manager.kv_cache_manager = SimpleNamespace(
            kv_cache_map={7: target_cache},
        )
        draft_manager = _make_fake_v2(is_draft=True)
        draft_manager.num_extra_kv_tokens = 2
        draft_manager._kv_reserve_draft_tokens = 3
        draft_cache = SimpleNamespace(is_active=True, resize=mock.Mock(return_value=True))
        draft_manager.kv_cache_map = {7: draft_cache}
        manager.draft_kv_cache_manager = draft_manager

        manager._resize_compacted_requests([(7, retained)], {7: 4})

        target_cache.resize.assert_called_once_with(retained + 4, None)
        # Draft protected tail = num_extra_kv_tokens + reserved draft width + 1.
        draft_cache.resize.assert_called_once_with(retained + 6, None)

    def test_mtp_eagle_paged_draft_length_contract_is_accepted(self):
        # A one-model MTP contract passes the call-site speculative gate, and
        # the factory then builds a manager that validates cleanly.
        from tensorrt_llm._torch.pyexecutor._util import (
            create_kv_cache_compression_manager,
            validate_kv_cache_compression_with_spec,
        )
        from tensorrt_llm.llmapi.llm_args import (
            MTPDecodingConfig,
            TriAttentionKvCacheCompressionConfig,
        )

        config = TriAttentionKvCacheCompressionConfig(
            model_path="/models/test", calibration_path="/calib/test.pt", top_B=8
        )
        assert config.kv_cache_compression_mode.is_eviction_method() is True
        draft_manager = _make_fake_v2(is_draft=True)
        validate_kv_cache_compression_with_spec(
            config, MTPDecodingConfig(max_draft_len=1), draft_manager
        )
        manager = create_kv_cache_compression_manager(
            config,
            _make_fake_v2(),
            draft_kv_cache_manager=draft_manager,
        )

        manager._validate_v2_compatibility()

    @pytest.mark.parametrize(
        "num_extra_kv_tokens,reserved_draft,draft_tokens,expected_growth",
        [
            (2, 0, [1, 2, 3], 4),
            # The reserved draft width protects capacity even when this step's
            # actual draft is shorter.
            (0, 6, [1, 2], 7),
        ],
    )
    def test_prepare_snapshots_fixed_linear_generation_growth(
        self, num_extra_kv_tokens, reserved_draft, draft_tokens, expected_growth
    ):
        manager = _make_fake_v2()
        manager.num_extra_kv_tokens = num_extra_kv_tokens
        manager._kv_reserve_draft_tokens = reserved_draft
        manager.kv_cache_map = {
            7: SimpleNamespace(capacity=106, is_active=True),
        }
        triattention = TriAttention(manager, top_B=8, model_path="/models/test")
        batch = SimpleNamespace(
            context_requests=[],
            generation_requests=[_make_request(7, py_draft_tokens=draft_tokens)],
        )

        triattention.prepare_resources(batch)

        assert triattention._prepared_generation_batch[0] is batch
        assert triattention._prepared_generation_batch[1] == {7: expected_growth}


class TestFixedScoreMetadata:
    @pytest.mark.parametrize("normalize_scores", [False, True])
    @pytest.mark.parametrize("eviction_mode", ["union", "per_head", "per_layer_perhead"])
    def test_workspace_build_receives_mode_and_capacity_kwargs(
        self, eviction_mode, normalize_scores
    ):
        from tensorrt_llm._torch.kv_cache_compression.triattention import triattention as module

        if eviction_mode == "union" and not normalize_scores:
            # The fused pipeline (THE union path) always z-normalizes, so
            # this combination is rejected loudly at construction.
            with pytest.raises(ValueError, match="normalize_scores=True"):
                _make_triattention(
                    top_B=4,
                    eviction_mode=eviction_mode,
                    normalize_scores=normalize_scores,
                )
            return
        manager = _make_triattention(
            top_B=4,
            eviction_mode=eviction_mode,
            normalize_scores=normalize_scores,
        )
        # The workspace follows the executor limits: eight requests (max batch
        # size) by 260 decode tokens (top_B plus two eviction periods).
        layout, workspace = _make_workspace_stubs(manager)
        prepared = [
            _prepared_eviction(
                _make_request(7),
                request_id=7,
                seq_len=8,
                expected_keep_count=4,
            )
        ]

        with mock.patch.object(
            module,
            "prepare_eviction_workspace",
            return_value=workspace,
        ) as build_workspace:
            resources = manager._workspace_for(layout, prepared)

        assert resources is workspace
        kwargs = build_workspace.call_args.kwargs
        assert kwargs["eviction_mode"] == eviction_mode
        assert kwargs["keep_count"] == 4
        assert kwargs["max_requests"] == 8
        assert kwargs["decode_width"] == 260
        assert kwargs["phase"] is manager._phase
        assert kwargs["layer_pool_keys"] == list(layout["layer_pool_keys"])
        # The cached workspace serves later rounds without rebuilding.
        with mock.patch.object(
            module,
            "prepare_eviction_workspace",
            side_effect=AssertionError("workspace was rebuilt"),
        ):
            assert manager._workspace_for(layout, prepared) is workspace

    def test_bulk_page_table_copy_snapshots_and_orders_consumers(self):
        """The bulk copy stages immutable host snapshots, and the next copy
        waits for the previous cohort's consumers.

        Both persistent V2 host inputs (the block-offset table and the
        index-mapper slot assignment) may mutate as soon as ``stage`` returns;
        the staged device tables must reflect the values at staging time. A
        subsequent bulk copy must also wait until the previous round's
        consumers (recorded by ``mark_page_tables_consumed``) are done.
        """
        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            _stage_block_offsets,
            mark_page_tables_consumed,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        current_stream = torch.cuda.current_stream(device)
        manager_stream = torch.cuda.Stream(device=device)
        host_table = torch.zeros(
            1,
            2,
            2,
            12,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        host_table[0, 0, 0, :5] = torch.tensor([3, 4, 5, 6, 7], dtype=torch.int32)
        host_table[0, 1, 0, :5] = torch.tensor([8, 9, 10, 11, 12], dtype=torch.int32)
        selected_slot = [0]

        def gather_k_block_offsets(source, destination, request_ids, num_blocks):
            assert request_ids == [7]
            destination[:, :1, 0, :num_blocks].copy_(
                source[:, selected_slot[0], 0, :num_blocks].unsqueeze(1)
            )

        gather = mock.Mock(side_effect=gather_k_block_offsets)
        staging = _make_bare_staging(device, max_requests=1, copy_block_count=8, page_count=5)
        staging.copy_done.record(current_stream)
        manager = _make_staging_manager(host_table, gather, manager_stream)

        def stage_once():
            # Raises on any staging failure; success returns None.
            _stage_block_offsets(
                staging,
                manager,
                [7],
                current_stream,
                staging._bulk_offsets_src,
                staging.block_offsets_device,
                staging.copy_block_count,
            )

        # Round 1: mutate the host table and the slot assignment right after
        # staging, with the manager stream artificially delayed. The staged
        # result must still reflect the values at staging time.
        with torch.cuda.stream(manager_stream):
            torch.cuda._sleep(50_000_000)
        with mock.patch.object(
            torch,
            "index_select",
            side_effect=AssertionError("page-table staging used torch.index_select"),
        ):
            stage_once()
        assert staging._bulk_offsets_src.shape == (1, 1, 2, 8)
        host_table[0, 0, 0, :5] = torch.tensor([13, 14, 15, 16, 17], dtype=torch.int32)
        selected_slot[0] = 1
        current_stream.synchronize()

        assert staging.block_offsets_device[0, 0, 0, :5].tolist() == [6, 8, 10, 12, 14]
        assert staging.block_offsets_device[0, 0, 1, :5].tolist() == [7, 9, 11, 13, 15]

        # Round 2: same contract on a re-staged cohort.
        host_table[0, 0, 0, :5] = torch.tensor([18, 19, 20, 21, 22], dtype=torch.int32)
        selected_slot[0] = 0
        with torch.cuda.stream(manager_stream):
            torch.cuda._sleep(50_000_000)
        stage_once()
        host_table[0, 0, 0, :5] = torch.tensor([23, 24, 25, 26, 27], dtype=torch.int32)
        selected_slot[0] = 1
        current_stream.synchronize()

        assert staging.block_offsets_device[0, 0, 0, :5].tolist() == [36, 38, 40, 42, 44]
        assert staging.block_offsets_device[0, 0, 1, :5].tolist() == [37, 39, 41, 43, 45]

        # Round 3: a delayed consumer read (snapshot) queued before
        # ``mark_page_tables_consumed`` must complete before the next bulk
        # copy overwrites the device tables.
        selected_slot[0] = 0
        manager_stream.synchronize()
        snapshot = torch.empty_like(staging.block_offsets_device)
        torch.cuda._sleep(20_000_000)
        snapshot.copy_(staging.block_offsets_device)
        staging.page_tables_active = True
        mark_page_tables_consumed(staging, manager_stream)

        stage_once()
        current_stream.synchronize()

        assert snapshot[0, 0, 0, :5].tolist() == [36, 38, 40, 42, 44]
        assert staging.block_offsets_device[0, 0, 0, :5].tolist() == [46, 48, 50, 52, 54]

    def test_staged_page_tables_bypass_per_request_cuda_materialization(self):
        manager = _make_triattention(top_B=4)
        manager.kv_cache_manager.num_extra_kv_tokens = 3
        manager.kv_cache_manager._stream = mock.Mock()
        manager.kv_cache_manager.get_batch_cache_indices = mock.Mock(
            side_effect=AssertionError("eviction staged page tables per request")
        )
        first = _make_request(7, py_prompt_len=3)
        second = _make_request(8, py_prompt_len=5)
        _set_request_state(manager, 7, confirmed_kv_length=8)
        _set_request_state(manager, 8, confirmed_kv_length=10)

        with _mocked_eviction_internals(manager) as internals:
            manager._evict_requests(
                [(first, 7), (second, 8)],
                2,
                protected_tail_lengths={7: 2, 8: 3},
            )

        # One batched staging call carries the whole cohort: request ids,
        # round starts, pinned prompt lengths, and valid lengths. top_B=4:
        # per-request moves are keep + tail = [6, 7]; padded rows repeat the
        # final offset out to the request capacity.
        args = internals.stage.call_args
        assert args.args[0] is internals.workspace
        assert args.args[1] is manager.kv_cache_manager
        assert args.args[2:6] == ([7, 8], [8, 10], [3, 5], [8, 10])
        assert args.kwargs["draft_manager"] is None
        assert args.kwargs["dense_move_offsets"] == [0, 6, 13, 13, 13, 13, 13, 13, 13]
        assert args.kwargs["swa_move_offsets"] is None
        assert args.kwargs["draft_move_offsets"] is None
        internals.consumed.assert_called_once_with(
            internals.workspace, manager.kv_cache_manager._stream
        )

    @requires_sm100
    @pytest.mark.parametrize("request_count", [1, 7, 8])
    def test_staging_stages_dense_and_swa_tables(self, request_count):
        pytest.importorskip("cutlass")
        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            prepare_eviction_workspace,
            stage_eviction_cohort,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        max_requests = 8
        page_count = 3
        # The bucket capacity must be aligned to the score kernel's 64-token
        # compute tile (the staging constructor compiles the kernel).
        seq_len = 64
        page_table_token_capacity = 90
        tokens_per_block = 32
        head_dim = 64
        num_freqs = head_dim // 2
        num_q_heads = 8
        layer_elements = max_requests * page_count * 2 * 1 * tokens_per_block * head_dim
        shared = torch.randn(2 * layer_elements, device=device).to(torch.bfloat16)
        pool_shape = (max_requests * page_count, 2, 1, tokens_per_block, head_dim)
        pools = [
            shared[:layer_elements].view(pool_shape),
            shared[layer_elements:].view(pool_shape),
            torch.randn(pool_shape, device=device).to(torch.bfloat16),
            torch.randn(pool_shape, device=device).to(torch.bfloat16),
        ]
        dense_groups = [[0, 1], [2]]
        representatives = [0, 2, 3]
        q_real = torch.randn(4, num_q_heads, 2 * num_freqs, dtype=torch.float64, device=device)[
            ..., ::2
        ]
        q_imag = torch.randn(4, num_q_heads, 2 * num_freqs, dtype=torch.float64, device=device)[
            ..., ::2
        ]
        mlr = torch.randn(4, num_q_heads, 2 * num_freqs, dtype=torch.float64, device=device)[
            ..., ::2
        ]
        freq = (torch.rand(2 * num_freqs, dtype=torch.float64, device=device) + 0.5)[::2]
        omega = (torch.rand(2 * num_freqs, dtype=torch.float64, device=device) * 0.05)[::2]
        offsets = torch.tensor([1.0, 0.0, 2.0, 0.0], dtype=torch.float64, device=device)[::2]
        assert not q_real.is_contiguous()
        assert not freq.is_contiguous()
        assert not omega.is_contiguous()
        assert not offsets.is_contiguous()
        # The workspace constructor converts every non-contiguous fp64
        # calibration input to contiguous fp32 (the runner's flat views would
        # otherwise fail) and compiles the score kernel here.
        staging = prepare_eviction_workspace(
            eviction_mode="per_head",
            layer_pools=pools,
            dense_groups=dense_groups,
            dense_layers=[0, 1, 2],
            page_representatives=representatives,
            max_requests=max_requests,
            seq_len=seq_len,
            num_q_heads=num_q_heads,
            num_freqs=num_freqs,
            keep_count=4,
            q_real=q_real,
            q_imag=q_imag,
            mlr_coef=mlr,
            freq_scale_sq=freq,
            offsets=offsets,
            omega=omega,
            page_table_token_capacity=page_table_token_capacity,
            layer_group_representative={0: 0, 1: 0, 2: 2},
            layer_pool_keys=[("pool", 0), ("pool", 0), ("pool", 2), ("pool", 3)],
        )
        assert staging.bucket_seq_len == seq_len
        assert staging.page_table_token_capacity == page_table_token_capacity
        # page_count 3 rounds up to the 4-block copy granule.
        assert staging.copy_block_count == (page_count + 3) // 4 * 4
        assert staging.phase["offsets"].dtype == torch.float32
        assert staging.phase["offsets"].is_contiguous()
        assert staging.phase["omega"].dtype == torch.float32
        assert staging.phase["omega"].is_contiguous()
        tables = {
            10: [
                [3 * request, 3 * request + 1, 3 * request + 2] for request in range(request_count)
            ],
            12: [
                [3 * request + 2, 3 * request + 1, 3 * request] for request in range(request_count)
            ],
            13: [
                [23 - 3 * request, 22 - 3 * request, 21 - 3 * request]
                for request in range(request_count)
            ],
        }

        request_ids = list(range(request_count))
        round_starts = [131_071 + request for request in request_ids]
        # Per-request pinned prompt lengths: one cohort may mix them.
        token_starts = list(request_ids)
        host_table = torch.zeros(
            3,
            max_requests,
            2,
            staging.copy_block_count,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        for slot, global_layer in enumerate((10, 12, 13)):
            host_table[slot, :request_count, 0, :page_count].copy_(
                torch.tensor(tables[global_layer], dtype=torch.int32)
            )

        def gather_k_block_offsets(source, destination, requested_ids, num_blocks):
            for destination_row, request_id in enumerate(requested_ids):
                destination[:, destination_row, 0, :num_blocks].copy_(
                    source[:, request_id, 0, :num_blocks]
                )

        gather = mock.Mock(side_effect=gather_k_block_offsets)
        manager = _make_staging_manager(
            host_table, gather, torch.cuda.Stream(device=device), num_slots=3
        )

        # Round starts past the int32 metadata range fail loudly before any
        # GPU work is enqueued.
        with pytest.raises((RuntimeError, OverflowError, ValueError)):
            stage_eviction_cohort(
                staging,
                manager,
                request_ids,
                [2**31] * request_count,
                token_starts,
                [seq_len] * request_count,
            )
        assert gather.call_count == 0
        with mock.patch.object(
            torch,
            "index_select",
            side_effect=AssertionError("page-table staging used torch.index_select"),
        ):
            stage_eviction_cohort(
                staging,
                manager,
                request_ids,
                round_starts,
                token_starts,
                [seq_len] * request_count,
            )
        torch.cuda.current_stream(device).synchronize()
        assert staging.round_starts_device.untyped_storage().data_ptr() == (
            staging.valid_seq_lens_device.untyped_storage().data_ptr()
        )
        assert torch.equal(
            staging.round_starts_device[:request_count],
            torch.tensor(round_starts, dtype=torch.int32, device=device),
        )
        assert torch.equal(
            staging.valid_seq_lens_device[:request_count],
            torch.full((request_count,), seq_len, dtype=torch.int32, device=device),
        )
        assert torch.equal(
            staging.token_starts_device[:request_count],
            torch.tensor(token_starts, dtype=torch.int32, device=device),
        )
        for slot, global_layer in enumerate((10, 12, 13)):
            expected = torch.tensor(tables[global_layer], dtype=torch.int32, device=device) * 2
            assert torch.equal(
                staging.block_offsets_device[slot, :request_count, 0, :page_count],
                expected,
            )

    @requires_sm100
    @pytest.mark.parametrize("request_count", [1, 7, 8])
    def test_fused_score_spans_distinct_storages_and_block_tables(self, request_count):
        """ONE launch over layers in DISTINCT storages with DISTINCT block tables.

        This is the production V2 shape: get_buffers wraps every layer as its
        own TensorWrapper storage and every layer allocates its own pages, so
        the fused path must not assume a shared storage anchor or a shared
        per-request block table. The launch is checked against the independent
        Torch oracle, then relaunched after a round-start advance and a block
        table rebind.
        """
        pytest.importorskip("cutlass")
        from tensorrt_llm._torch.kv_cache_compression.triattention import triattention as module

        device = torch.device("cuda", torch.cuda.current_device())
        torch.manual_seed(20260707 + request_count)
        max_requests = request_count
        page_count = 2
        tokens_per_block = 32
        head_dim = 64
        num_freqs = head_dim // 2
        num_q_heads = 8
        seq_len = page_count * tokens_per_block
        prompt_len = 1
        num_layers = 3
        # Three SEPARATE allocations (distinct storages, like V2 TensorWrapper).
        pools = [
            (
                0.125
                * torch.randn(
                    max_requests * page_count, 2, 1, tokens_per_block, head_dim, device=device
                )
            ).to(torch.bfloat16)
            for _ in range(num_layers)
        ]
        assert len({pool.untyped_storage().data_ptr() for pool in pools}) == num_layers
        # A DIFFERENT block table per layer (per-layer page allocation).
        generator = torch.Generator(device="cpu").manual_seed(7 + request_count)
        page_ids_3d = torch.stack(
            [
                torch.randperm(max_requests * page_count, generator=generator)[
                    : max_requests * page_count
                ]
                .view(max_requests, page_count)
                .to(device=device, dtype=torch.int64)
                for _ in range(num_layers)
            ]
        ).contiguous()
        q_real = 0.125 * torch.randn(num_layers, num_q_heads, num_freqs, device=device)
        q_imag = 0.125 * torch.randn(num_layers, num_q_heads, num_freqs, device=device)
        mlr = 0.125 * torch.randn(num_layers, num_q_heads, num_freqs, device=device)
        freq = torch.rand(num_freqs, device=device) + 0.5
        omega = torch.rand(num_freqs, device=device) * 0.05
        offsets = torch.tensor([1.0, 2.0, 4.0], device=device)
        round_device = torch.arange(max_requests, dtype=torch.int32, device=device) + 9
        round_starts = round_device[:request_count].tolist()
        seq_lens = [seq_len - request % 2 for request in range(request_count)]
        layer_order = list(range(num_layers))
        # One group per layer: slot i holds layer i's own block table.
        ws = module.prepare_eviction_workspace(
            eviction_mode="per_head",
            layer_pools=pools,
            dense_groups=[[layer] for layer in layer_order],
            dense_layers=layer_order,
            page_representatives=layer_order,
            max_requests=max_requests,
            seq_len=seq_len,
            num_q_heads=num_q_heads,
            num_freqs=num_freqs,
            keep_count=4,
            q_real=q_real,
            q_imag=q_imag,
            mlr_coef=mlr,
            freq_scale_sq=freq,
            offsets=offsets,
            omega=omega,
            decode_width=seq_len - prompt_len,
            layer_group_representative={layer: layer for layer in layer_order},
            layer_pool_keys=[("pool", layer) for layer in layer_order],
        )
        valid_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)

        def stage_round():
            # Stage the round metadata straight into the fixed device rows
            # (the cohort staging path is covered by the staging test above).
            ws.round_starts_device.copy_(round_device)
            ws.valid_seq_lens_device[:request_count].copy_(valid_seq_lens)
            ws.token_starts_device.fill_(prompt_len)
            ws.block_offsets_device[:, :, :, :page_count].copy_(_encode_block_offsets(page_ids_3d))

        score_sentinel = -12345.0
        stage_round()
        ws.score_output.fill_(score_sentinel)
        with mock.patch.object(module, "run_cache_compactions"):
            module.run_eviction_round(ws, normalize_scores=False)
        fixed = ws.score_output.clone()
        assert ws.valid_widths.tolist() == [seq_len - prompt_len for seq_len in seq_lens]

        # The deployed fused score must agree with the independent Torch oracle
        # when every layer owns a distinct V2 block table.
        oracle = _torch_tri_score_oracle(
            pools,
            {layer: page_ids_3d[layer, :request_count] for layer in layer_order},
            seq_lens,
            round_starts,
            q_real,
            q_imag,
            mlr,
            freq,
            omega,
            offsets,
            layer_order,
        )
        for request in range(request_count):
            for layer_slot, layer in enumerate(layer_order):
                valid_width = seq_lens[request] - prompt_len
                segment = fixed[request, layer_slot, :, :valid_width]
                expected = oracle[request * num_layers + layer][
                    :, prompt_len : prompt_len + valid_width
                ]
                torch.testing.assert_close(segment, expected, rtol=5e-3, atol=5e-3)

        round_device.add_(17)
        page_ids_3d = page_ids_3d.roll(1, dims=2)
        valid_seq_lens.copy_(
            torch.tensor(
                [seq_len - (request + 1) % 2 for request in range(request_count)],
                dtype=torch.int32,
                device=device,
            )
        )
        expected_second_widths = valid_seq_lens - prompt_len
        stage_round()
        ws.score_output.fill_(score_sentinel)
        ws.valid_widths.fill_(-1)
        with mock.patch.object(module, "run_cache_compactions"):
            module.run_eviction_round(ws, normalize_scores=False)
        second_launch = ws.score_output.clone()
        assert torch.equal(ws.valid_widths, expected_second_widths)
        assert not torch.equal(second_launch, fixed)


class TestKernelMaskedSwa:
    @pytest.mark.parametrize("top_B,fits_window", [(128, True), (127, False)])
    def test_layer_partition_uses_local_config_and_validates_window(self, top_B, fits_window):
        mgr = _make_triattention()
        mgr.model_path = "/models/gpt-oss"
        mgr.top_B = top_B
        mgr.kv_cache_manager = SimpleNamespace(pp_layers=[0, 1, 2, 3])
        config = _make_hf_config(
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            sliding_window=128,
        )

        with mock.patch("transformers.AutoConfig.from_pretrained", return_value=config) as load:
            if not fits_window:
                # The decode budget must cover the kernel-masked SWA window.
                with pytest.raises(ValueError, match="decode budget top_B=127"):
                    mgr._attention_layer_partition(4)
                return
            dense, sliding, window = mgr._attention_layer_partition(4)

        load.assert_called_once_with(
            "/models/gpt-oss", trust_remote_code=True, local_files_only=True
        )
        assert dense == [1, 3]
        assert sliding == [0, 2]
        assert window == 128
