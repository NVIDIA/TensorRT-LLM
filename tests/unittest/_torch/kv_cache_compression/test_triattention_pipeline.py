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
tests cover the config, construction, compressed-count publication, eager
selection, page-table staging, bounded request chunks, and request lifecycle.
Model-level correctness is covered by separate end-to-end tests.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from pydantic import ValidationError

# TriAttention lives in the kv_cache_compression package. It exposes only the
# compression manager -- no attention classes or KV-cache-manager subclass.
from tensorrt_llm._torch.kv_cache_compression.triattention import TriAttention
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
    _BatchedUnionKeepSetSelector,
    _PreparedEviction,
    _PreparedGenerationBatch,
    _RequestCompressionState,
    _RuntimeKVLayout,
)

# Framework base class lives in pyexecutor.resource_manager; the factory lives
# in pyexecutor._util (next to _create_kv_cache_manager), matching #15106.
from tensorrt_llm._torch.pyexecutor._util import create_kv_cache_compression_manager
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm.llmapi.llm_args import TriAttentionKvCacheCompressionConfig

_TORCH_TOPK_ORACLE = torch.topk


def _encode_block_offsets(page_ids: torch.Tensor) -> torch.Tensor:
    """Build the native V2 [pool, request, K/V, block] layout."""
    if page_ids.ndim == 2:
        page_ids = page_ids.unsqueeze(0)
    encoded = torch.empty(
        page_ids.shape[0],
        page_ids.shape[1],
        2,
        page_ids.shape[2],
        dtype=torch.int32,
        device=page_ids.device,
    )
    encoded[:, :, 0] = page_ids.to(torch.int32) * 2
    encoded[:, :, 1] = encoded[:, :, 0] + 1
    return encoded


def _set_request_state(
    manager,
    request_id,
    *,
    generation_steps=0,
    evicted_tokens=0,
    confirmed_kv_length=None,
):
    state = _RequestCompressionState(
        generation_steps=generation_steps,
        evicted_tokens=evicted_tokens,
        confirmed_kv_length=confirmed_kv_length,
    )
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
    return _PreparedEviction(
        request=request,
        request_id=request_id,
        seq_len=seq_len,
        round_start=int(seq_len if round_start is None else round_start),
        prompt_len=prompt_len,
        expected_keep_count=expected_keep_count,
        protected_tail=protected_tail,
    )


def _fake_cute_dsl_topk(
    values: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    top_k: int,
    next_n: int,
) -> None:
    """CPU oracle for the CUDA-only CuTE-DSL selector custom op."""
    assert next_n == 1
    for row in range(int(values.shape[0])):
        width = int(seq_lens[row])
        selected = _TORCH_TOPK_ORACLE(
            values[row, :width],
            top_k,
            sorted=False,
        ).indices
        output[row].copy_(selected.to(torch.int32))


@contextmanager
def _mock_cute_topk_without_fallbacks():
    """Provide the CuTE op while making both retired fallbacks fatal."""
    with (
        mock.patch.object(
            torch.ops.trtllm,
            "cute_dsl_indexer_topk_decode",
            side_effect=_fake_cute_dsl_topk,
            create=True,
        ) as cute_topk,
        mock.patch.object(
            torch.ops.trtllm,
            "indexer_topk_decode",
            side_effect=AssertionError("native IndexerTopK fallback is forbidden"),
            create=True,
        ),
        mock.patch.object(
            torch,
            "topk",
            side_effect=AssertionError("torch.topk production fallback is forbidden"),
        ),
    ):
        yield cute_topk


def _union_oracle(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    """Independent expected-result implementation of union selection."""
    combined = scores.max(dim=0).values
    row_top = _TORCH_TOPK_ORACLE(
        scores,
        keep_count,
        dim=1,
        sorted=False,
    ).indices
    union_mask = torch.zeros(scores.shape[1], dtype=torch.bool, device=scores.device)
    union_mask.scatter_(0, row_top.reshape(-1), True)
    union_indices = torch.nonzero(union_mask, as_tuple=False).flatten()
    if union_indices.numel() >= keep_count:
        candidates = combined.index_select(0, union_indices)
        relative = _TORCH_TOPK_ORACLE(
            candidates,
            keep_count,
            sorted=False,
        ).indices
        return torch.sort(union_indices.index_select(0, relative)).values

    remaining = keep_count - int(union_indices.numel())
    residual = combined.clone()
    residual[union_mask] = float("-inf")
    extra = _TORCH_TOPK_ORACLE(
        residual,
        remaining,
        sorted=False,
    ).indices
    return torch.sort(torch.cat((union_indices, extra))).values


def _distinct_topk_scores(width: int, rows: int = 2) -> torch.Tensor:
    """Create deterministic finite rows without top-k boundary ties."""
    token = torch.arange(width, dtype=torch.float32)
    return torch.stack(
        [
            torch.sin(token * (0.0017 + row * 0.0003))
            + token * (0.00011 + row * 0.000013)
            + row * 0.000001
            for row in range(rows)
        ]
    )


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


def _make_fake_v2(enable_block_reuse=False, *, is_draft=False):
    """Build an unallocated V2 double with TriAttention's production contract."""
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

    fake_v2 = KVCacheManagerV2.__new__(KVCacheManagerV2)
    fake_v2.enable_block_reuse = enable_block_reuse
    fake_v2.is_draft = is_draft
    fake_v2.kv_compression_manages_history = False
    fake_v2.kv_factor = 2
    fake_v2.mapping = SimpleNamespace(enable_attention_dp=False)
    fake_v2.is_disagg = False
    fake_v2.max_beam_width = 1
    fake_v2.max_batch_size = 8
    fake_v2.num_extra_kv_tokens = 0
    fake_v2.max_draft_len = 0
    fake_v2.max_total_draft_tokens = 0
    fake_v2._kv_reserve_draft_tokens = 0
    fake_v2.max_seq_len = 65536
    fake_v2.tokens_per_block = 64
    fake_v2.max_blocks_per_seq = 1028
    fake_v2.get_num_available_tokens = lambda *, token_num_upper_bound, **_: token_num_upper_bound
    fake_v2.max_attention_window_vec = []
    fake_v2.kv_cache_manager_py_config = SimpleNamespace(layers=[])
    fake_v2.impl = object()
    fake_v2.kv_cache_map = {}
    fake_v2.host_kv_cache_block_offsets = torch.empty(1, dtype=torch.int64)
    fake_v2.pp_layers = []
    fake_v2.layer_offsets = {}
    fake_v2.layer_to_pool_mapping_dict = {}
    return fake_v2


def _make_triattention(**overrides):
    """Construct a fully initialized manager for method-level unit tests."""
    options = {"top_B": 8, "model_path": "/models/test"}
    options.update(overrides)
    return TriAttention(_make_fake_v2(), **options)


def _make_request(request_id, **overrides):
    """Build the explicit request fields consumed by TriAttention."""
    fields = {
        "py_request_id": request_id,
        "py_prompt_len": 0,
        "py_max_new_tokens": 65536,
        "py_draft_tokens": [],
        "py_num_accepted_draft_tokens": 0,
        "py_num_compressed_tokens": 0,
        "is_dummy": False,
        "state": LlmRequestState.GENERATION_IN_PROGRESS,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _make_hf_config(**values):
    """Expose the normalized Hugging Face text-config contract."""
    text_config = SimpleNamespace(to_dict=lambda: dict(values))
    return SimpleNamespace(get_text_config=lambda: text_config)


def _torch_tri_score_oracle(
    layer_pools,
    page_ids,
    seq_lens,
    round_starts,
    q_real,
    q_imag,
    mlr_coef,
    freq_scale_sq,
    omega,
    offsets,
    layer_indices,
    aggregation,
):
    """Independent Torch implementation of the paged TriAttention score."""
    scores = []
    num_q_heads = int(q_real.shape[1])
    for request, seq_len in enumerate(seq_lens):
        phase = (round_starts[request] + offsets[:, None]) * omega[None, :]
        mean_cos = torch.cos(phase).mean(dim=0)
        mean_sin = torch.sin(phase).mean(dim=0)
        for layer in layer_indices:
            pool = layer_pools[layer]
            request_page_ids = (
                page_ids[layer][request] if isinstance(page_ids, dict) else page_ids[request]
            )
            keys = (
                pool.index_select(0, request_page_ids)[:, 0]
                .permute(1, 0, 2, 3)
                .reshape(pool.shape[2], -1, pool.shape[4])[:, :seq_len]
                .float()
            )
            num_kv_heads = int(keys.shape[0])
            group_size = num_q_heads // num_kv_heads
            head_scores = []
            for head in range(num_q_heads):
                key = keys[head // group_size]
                num_freqs = int(key.shape[-1]) // 2
                key_real = key[:, :num_freqs]
                key_imag = key[:, num_freqs:]
                product_real = q_real[layer, head] * key_real + q_imag[layer, head] * key_imag
                product_imag = q_imag[layer, head] * key_real - q_real[layer, head] * key_imag
                if aggregation == "mean":
                    position = (
                        freq_scale_sq * (product_real * mean_cos - product_imag * mean_sin)
                    ).sum(dim=-1)
                else:
                    position = (
                        (
                            freq_scale_sq[None, None, :]
                            * (
                                product_real[None] * torch.cos(phase)[:, None, :]
                                - product_imag[None] * torch.sin(phase)[:, None, :]
                            )
                        )
                        .sum(dim=-1)
                        .max(dim=0)
                        .values
                    )
                mlr = (
                    torch.sqrt(key_real.square() + key_imag.square())
                    * mlr_coef[layer, head]
                    * freq_scale_sq
                ).sum(dim=-1)
                head_scores.append(position + mlr)
            scores.append(torch.stack(head_scores))
    return scores


class TestKvCacheCompressionConfig:
    def test_llm_args_dispatches_concrete_and_unknown_algorithms(self):
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

    def test_eviction_mode_validated(self):
        with pytest.raises(ValidationError):
            TriAttentionKvCacheCompressionConfig(eviction_mode="made_up_mode")


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
        cached = _RuntimeKVLayout(
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

    def test_triattention_enables_capacity_only_on_target_manager(self):
        manager = _make_fake_v2()
        triattention = TriAttention(manager, top_B=8, model_path="/models/test")
        triattention._attention_layer_partition_cache = ([], [], None)
        triattention._calibrated = True
        first = _make_request(11)
        second = _make_request(12)

        triattention.on_request_init(first)
        triattention.on_request_init(second)

        assert triattention.adjusts_generation_kv_length is True
        assert manager.kv_compression_manages_history
        assert set(triattention._request_states) == {11, 12}

    def test_request_init_accepts_speculative_capacity(self):
        manager = _make_fake_v2()
        manager.num_extra_kv_tokens = 4
        manager._kv_reserve_draft_tokens = 4
        triattention = TriAttention(manager, top_B=8, model_path="/models/test")
        triattention._attention_layer_partition_cache = ([], [], None)
        triattention._calibrated = True
        request = _make_request(11)

        triattention.on_request_init(request)

        assert manager.kv_compression_manages_history
        assert set(triattention._request_states) == {11}

    def test_resolve_accepts_flat_pt(self, flat_calibration_pt):
        mgr = _make_triattention()
        mgr.calibration_path = flat_calibration_pt
        mgr.model_path = None
        loaded = mgr._resolve_calibration()
        for key in ("E_q", "E_q_norm", "omega", "freq_scale_sq"):
            assert key in loaded


# ---------------------------------------------------------------------------
# Eviction publishes the cumulative evicted count on the request; the model
# engine reads it back where it builds num_cached_tokens_per_seq.
# ---------------------------------------------------------------------------


class TestCompressedTokenPublication:
    def test_manager_is_marked_capacity_only_and_requests_default_to_zero(self):
        mgr = _make_triattention()
        manager = mgr.kv_cache_manager
        # The compression-manager base marks the target manager so V2 sizing
        # keeps logical max_seq_len while capacity is reclaimed and reused.
        assert manager.kv_compression_manages_history
        request = _make_request(7)
        # Default 0 keeps the engine's num_cached subtraction a no-op until
        # the first eviction publishes a count.
        assert request.py_num_compressed_tokens == 0

    @contextmanager
    def _mocked_eviction_internals(self, manager):
        """Run the real ``_evict_requests`` body around mocked GPU launches."""
        score_staging = SimpleNamespace(
            launch_prepared_score=mock.Mock(return_value=torch.zeros(1)),
            mark_page_tables_consumed=mock.Mock(),
        )
        keep_set_selector = SimpleNamespace(
            select_requests=mock.Mock(),
            refresh_row_prompt_offsets=mock.Mock(),
        )
        resources = SimpleNamespace(
            score_staging=score_staging,
            keep_set_selector=keep_set_selector,
        )
        batched_compaction = SimpleNamespace(launch=mock.Mock())
        with (
            mock.patch.object(manager, "_runtime_kv_layout", return_value=SimpleNamespace()),
            mock.patch.object(manager, "_eager_resources_for", return_value=resources),
            mock.patch.object(
                manager,
                "_batched_compaction_for",
                return_value=batched_compaction,
            ),
            mock.patch.object(manager, "_attach_page_ids") as attach,
        ):
            yield SimpleNamespace(
                score_staging=score_staging,
                keep_set_selector=keep_set_selector,
                batched_compaction=batched_compaction,
                attach=attach,
            )

    def test_eviction_bookkeeping_publishes_cumulative_count(self):
        # The eviction bookkeeping writes the cumulative evicted count on the
        # request in the same step that compacts the cache; this is the
        # channel's only producer.
        manager = _make_triattention(top_B=4)
        manager.kv_cache_manager._stream = mock.Mock()
        request = _make_request(7, py_prompt_len=2)
        _set_request_state(manager, 7, confirmed_kv_length=10)

        with self._mocked_eviction_internals(manager) as internals:
            first = manager._evict_requests([(request, 7)], 2)

        assert first == [(7, 6)]
        # 10 confirmed - (2 pinned prompt + 4 decode budget) = 4 evicted.
        assert request.py_num_compressed_tokens == 4
        assert manager._request_states[7].confirmed_kv_length == 6
        internals.batched_compaction.launch.assert_called_once_with()
        internals.score_staging.mark_page_tables_consumed.assert_called_once_with(
            manager.kv_cache_manager._stream
        )

        # Round two: 6 retained + 8 newly confirmed decode tokens.
        manager._request_states[7].confirmed_kv_length = 14
        with self._mocked_eviction_internals(manager) as internals:
            second = manager._evict_requests([(request, 7)], 2)

        assert second == [(7, 6)]
        # The count is cumulative and never decreases: 4 + (14 - 6) = 12.
        assert request.py_num_compressed_tokens == 12
        # The staged logical position restores the uncompressed length.
        prepared = internals.attach.call_args.args[0]
        assert prepared[0].round_start == 14 + 4

    def test_identity_compaction_is_rejected_instead_of_published(self):
        manager = _make_triattention(top_B=4)
        manager.kv_cache_manager._stream = mock.Mock()
        request = _make_request(7, py_prompt_len=2)
        # Selection keeps every token: seq_len == prompt + budget + 1 evicts
        # one token; seq_len == prompt + budget must never publish.
        _set_request_state(manager, 7, confirmed_kv_length=6)

        with self._mocked_eviction_internals(manager):
            assert manager._evict_requests([(request, 7)], 2) == []
        assert request.py_num_compressed_tokens == 0


class TestStepEndHookRefactor:
    def test_triattention_prepare_only_snapshots_and_update_uses_final_hook(self):
        assert "prepare_resources" in TriAttention.__dict__
        assert "update_resources" not in TriAttention.__dict__
        assert "on_generation_step_end" in TriAttention.__dict__

    def test_prepare_does_not_evict_and_update_runs_final_hook_once(self):
        manager = _make_triattention()
        request = _make_request(7)
        batch = SimpleNamespace(
            context_requests=[],
            context_requests_last_chunk=[],
            generation_requests=[request],
        )

        with mock.patch.object(manager, "_periodic_evict") as periodic_evict:
            manager.prepare_resources(batch)
            periodic_evict.assert_not_called()

            manager.update_resources(batch)

        periodic_evict.assert_called_once_with(batch)

    @pytest.mark.parametrize("top_B", [511, 512])
    def test_non_v2_manager_is_always_rejected(self, top_B):
        with pytest.raises(TypeError, match="requires KVCacheManagerV2"):
            TriAttention(SimpleNamespace(), top_B=top_B)

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
        mgr.pin_prefill = True
        mgr.count_prompt_tokens = False
        return mgr, request, batch

    def test_identity_gate_preserves_real_eviction_round(self):
        import contextlib

        import tensorrt_llm._torch.kv_cache_compression.triattention.triattention as tri_module

        mgr, request, batch = self._make_due_decode_request(seq_len=1024 + 4096 + 1)
        timeline = []
        cache = mgr.kv_cache_manager.kv_cache_map[7]

        def compact(*args, protected_tail_lengths, **_kwargs):
            assert protected_tail_lengths == {7: 0}
            timeline.append("compact_dispatch")
            return [(7, 1024 + 4096)]

        @contextlib.contextmanager
        def track_range(name, **kwargs):
            timeline.append(f"enter:{name}")
            yield
            timeline.append(f"exit:{name}")

        with (
            mock.patch.object(mgr, "_evict_requests", side_effect=compact) as evict,
            mock.patch.object(tri_module, "nvtx_range", side_effect=track_range),
        ):
            mgr._periodic_evict(batch)

        evict.assert_called_once_with(
            [(request, 7)],
            2,
            protected_tail_lengths={7: 0},
        )
        mgr.kv_cache_manager._stream.wait_event.assert_not_called()
        cache.resize.assert_called_once_with(1024 + 4096, None)
        assert timeline == [
            "compact_dispatch",
            "enter:triattention.resize",
            "exit:triattention.resize",
        ]

    def test_suspended_cache_rejects_batch_before_cadence_mutation(self):
        manager, first_request, _ = self._make_due_decode_request(seq_len=1024 + 4096 + 1)
        second_request = _make_request(8, py_prompt_len=1024)
        manager.kv_cache_manager.kv_cache_map[8] = SimpleNamespace(is_active=False)
        first_state = manager._request_states[7]
        second_state = _set_request_state(manager, 8, generation_steps=127)
        batch = SimpleNamespace(generation_requests=[first_request, second_request])

        with pytest.raises(RuntimeError, match="request 8 must be resumed"):
            manager._periodic_evict(batch)

        assert first_state.generation_steps == 127
        assert first_state.confirmed_kv_length is None
        assert second_state.generation_steps == 127
        assert second_state.confirmed_kv_length is None

    def test_non_boundary_step_skips_eviction_geometry(self):
        manager, _, batch = self._make_due_decode_request(seq_len=1024 + 4096 + 1)
        state = manager._request_states[7]
        state.generation_steps = 126

        with mock.patch.object(manager, "_minimum_evictable_length") as keep_count:
            manager._periodic_evict(batch)

        keep_count.assert_not_called()
        assert state.generation_steps == 127
        assert state.confirmed_kv_length == 1024 + 4096 + 1

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

    def test_request_finish_keeps_eviction_buffers_resident(self):
        manager = _make_triattention()
        request = _make_request(7)
        _set_request_state(manager, 7)
        buffers = object()
        compaction = object()
        manager._eviction_resources = buffers
        manager._batched_compaction = compaction

        manager.on_request_finish(request)

        # The buffers are sized for the executor limits, not one cohort, so
        # they stay resident for the next generation batch.
        assert manager._eviction_resources is buffers
        assert manager._batched_compaction is compaction

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
        mgr._prepared_generation_batch = _PreparedGenerationBatch(
            batch=SimpleNamespace(generation_requests=[request]),
            growth_by_request={7: current_growth},
        )
        draft_manager = _make_fake_v2(is_draft=True)
        draft_cache = SimpleNamespace(is_active=True, resize=mock.Mock(return_value=True))
        draft_manager.kv_cache_map = {7: draft_cache}
        draft_manager._stream = mock.Mock()
        mgr.draft_kv_cache_manager = draft_manager

        def compact(*_args, **_kwargs):
            mgr._request_states[7].confirmed_kv_length = retained
            return [(7, retained)]

        with mock.patch.object(mgr, "_evict_requests", side_effect=compact) as evict:
            mgr._periodic_evict(batch)

        evict.assert_called_once_with(
            [(request, 7)],
            2,
            protected_tail_lengths={7: tail},
        )
        assert mgr._request_states[7].confirmed_kv_length == retained
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

        assert manager._request_states[7].confirmed_kv_length == physical_confirmed
        cache.resize.assert_not_called()

    def test_mla_selfkonly_cache_is_rejected(self):
        manager = _make_triattention()
        manager.kv_cache_manager.kv_factor = 1

        with pytest.raises(ValueError, match="standard key/value KV cache"):
            manager._validate_v2_compatibility()

    def test_one_model_mtp_co_compression_contract_is_accepted(self):
        draft_manager = _make_fake_v2(is_draft=True)
        manager = TriAttention(
            _make_fake_v2(),
            top_B=8,
            model_path="/models/test",
            draft_kv_cache_manager=draft_manager,
        )

        manager._validate_v2_compatibility()
        assert manager.kv_cache_manager.kv_compression_manages_history is True
        # The draft cache is compacted together with the target, so its
        # physical length diverges from the logical length the same way.
        assert draft_manager.kv_compression_manages_history is True

    def test_draft_co_compression_accepts_smaller_draft_max_seq_len(self):
        # Co-compression keeps the draft's physical length equal to the
        # target's, so the draft does not have to cover the target's logical
        # maximum sequence length.
        draft_manager = _make_fake_v2(is_draft=True)
        draft_manager.max_seq_len = 8192
        manager = TriAttention(
            _make_fake_v2(),
            top_B=8,
            model_path="/models/test",
            draft_kv_cache_manager=draft_manager,
        )

        manager._validate_v2_compatibility()

    @pytest.mark.parametrize("eviction_mode", ["per_head", "per_layer_perhead"])
    def test_draft_co_compression_requires_union_mode(self, eviction_mode):
        manager = TriAttention(
            _make_fake_v2(),
            top_B=8,
            model_path="/models/test",
            eviction_mode=eviction_mode,
            draft_kv_cache_manager=_make_fake_v2(is_draft=True),
        )

        with pytest.raises(ValueError, match="union"):
            manager._validate_v2_compatibility()

    def test_draft_co_compression_rejects_mla_draft_cache(self):
        draft_manager = _make_fake_v2(is_draft=True)
        manager = TriAttention(
            _make_fake_v2(),
            top_B=8,
            model_path="/models/test",
            draft_kv_cache_manager=draft_manager,
        )
        # The base class already rejects a mismatched draft kv_factor at
        # construction; this guards against later runtime divergence too.
        draft_manager.kv_factor = 1

        with pytest.raises(ValueError, match="standard key/value cache"):
            manager._validate_v2_compatibility()

    def test_draft_co_compression_requires_full_attention_draft(self):
        draft_manager = _make_fake_v2(is_draft=True)
        draft_manager.max_attention_window_vec = [128]
        manager = TriAttention(
            _make_fake_v2(),
            top_B=8,
            model_path="/models/test",
            draft_kv_cache_manager=draft_manager,
        )

        with pytest.raises(ValueError, match="full-attention draft"):
            manager._validate_v2_compatibility()

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

    @pytest.mark.parametrize("mode", ["draft_target", "pard"])
    def test_unvalidated_paged_draft_tail_contracts_remain_fail_closed(self, mode):
        from tensorrt_llm.llmapi.llm_args import DraftTargetDecodingConfig, PARDDecodingConfig

        if mode == "draft_target":
            spec_config = DraftTargetDecodingConfig(
                max_draft_len=3,
                speculative_model="/tmp/draft-target-model",
            )
        else:
            spec_config = PARDDecodingConfig(max_draft_len=3)
        # The call-site speculative gate rejects before any manager is created.
        from tensorrt_llm._torch.pyexecutor._util import validate_kv_cache_compression_with_spec
        from tensorrt_llm.llmapi.llm_args import TriAttentionKvCacheCompressionConfig

        with pytest.raises(ValueError, match="standard paged cache compacted together"):
            validate_kv_cache_compression_with_spec(
                TriAttentionKvCacheCompressionConfig(
                    model_path="/models/test", calibration_path="/calib/test.pt", top_B=8
                ),
                spec_config,
                _make_fake_v2(is_draft=True),
            )

    def test_dflash_spec_mode_is_rejected(self):
        # Policy: the DFlash draft reads cross-attention context buffers, not
        # a paged KV cache, so compression cannot cover it. The call-site
        # speculative gate rejects before any manager is created.
        from tensorrt_llm._torch.pyexecutor._util import validate_kv_cache_compression_with_spec
        from tensorrt_llm.llmapi.llm_args import (
            DFlashDecodingConfig,
            TriAttentionKvCacheCompressionConfig,
        )

        with pytest.raises(ValueError, match="standard paged cache compacted together"):
            validate_kv_cache_compression_with_spec(
                TriAttentionKvCacheCompressionConfig(
                    model_path="/models/test", calibration_path="/calib/test.pt", top_B=8
                ),
                DFlashDecodingConfig(max_draft_len=3),
                _make_fake_v2(is_draft=True),
            )

    def test_prepare_snapshots_fixed_linear_generation_growth(self):
        manager = _make_fake_v2()
        manager.num_extra_kv_tokens = 2
        manager.kv_cache_map = {
            7: SimpleNamespace(capacity=106, is_active=True),
        }
        triattention = TriAttention(manager, top_B=8, model_path="/models/test")
        batch = SimpleNamespace(
            context_requests=[],
            generation_requests=[_make_request(7, py_draft_tokens=[1, 2, 3])],
        )

        triattention.prepare_resources(batch)

        assert triattention._prepared_generation_batch.batch is batch
        assert triattention._prepared_generation_batch.growth_by_request == {7: 4}

    def test_prepare_protects_reserved_draft_width(self):
        manager = _make_fake_v2()
        manager._kv_reserve_draft_tokens = 6
        manager.kv_cache_map = {
            7: SimpleNamespace(capacity=106, is_active=True),
        }
        triattention = TriAttention(manager, top_B=8, model_path="/models/test")
        batch = SimpleNamespace(
            context_requests=[],
            generation_requests=[_make_request(7, py_draft_tokens=[1, 2])],
        )

        triattention.prepare_resources(batch)

        assert triattention._prepared_generation_batch.growth_by_request == {7: 7}

    def test_request_finish_clears_compression_state(self):
        request = SimpleNamespace(py_request_id=7)
        mgr = _make_triattention()
        _set_request_state(
            mgr,
            7,
            generation_steps=1,
            evicted_tokens=127,
            confirmed_kv_length=128,
        )
        mgr._prepared_generation_batch = _PreparedGenerationBatch(
            batch=SimpleNamespace(),
            growth_by_request={7: 1},
        )

        mgr.on_request_finish(request)

        assert mgr._request_states == {}
        assert mgr._prepared_generation_batch.growth_by_request == {}


class TestTopKRouting:
    @pytest.mark.parametrize("keep_count", [4096, 8192])
    def test_cross_request_union_uses_cute_without_fallback(self, keep_count):
        width = keep_count + 64
        request_scores = [
            _distinct_topk_scores(width),
            _distinct_topk_scores(width).roll(17, dims=1) + 0.000007,
        ]
        expected = [_union_oracle(scores, keep_count) for scores in request_scores]
        selector = _BatchedUnionKeepSetSelector(
            request_scores[0].shape[0],
            width,
            keep_count,
            dtype=request_scores[0].dtype,
            device=request_scores[0].device,
            max_requests=len(request_scores),
        )

        with _mock_cute_topk_without_fallbacks() as cute_topk:
            selector.select_requests(
                torch.stack(request_scores),
                normalize_scores=False,
            )
            selected = selector.keep[: len(request_scores)].clone()

        assert cute_topk.call_count == 1
        for actual, expected_keep in zip(selected, expected):
            assert torch.equal(actual, expected_keep)


class TestFixedScoreMetadata:
    @pytest.mark.parametrize("normalize_scores", [False, True])
    @pytest.mark.parametrize("eviction_mode", ["union", "per_head", "per_layer_perhead"])
    def test_eager_buffers_bind_score_after_selection(self, eviction_mode, normalize_scores):
        from tensorrt_llm._torch.kv_cache_compression.triattention import triattention as module

        manager = _make_triattention(
            top_B=4,
            eviction_mode=eviction_mode,
            normalize_scores=normalize_scores,
        )
        manager._H = 2
        manager._F = 2
        manager._freq_scale_sq = torch.ones(2)
        manager._offsets = torch.ones(2)
        manager.calibration = {"omega": torch.ones(2)}
        manager._local_score_calibration = mock.Mock(return_value=(torch.ones(2, 2, 2),) * 3)
        manager._page_table_pool_keys = mock.Mock(return_value=[("pool", 0)])
        pool = torch.empty(8, 2, 1, 4, 4)
        layout = SimpleNamespace(
            manager=SimpleNamespace(num_pools=1),
            num_layers=2,
            global_layers=[0, 1],
            layer_pools=[pool, pool],
            dense_layers=[0, 1],
            swa_layers=[],
            storage_groups={0: [0, 1]},
            pool_view_fingerprint=(("fixed",),),
        )
        # The buffers follow the executor limits: eight requests (max batch
        # size) by 260 decode tokens (top_B plus two eviction periods).
        score_staging = SimpleNamespace(
            fused_group=SimpleNamespace(output=torch.empty(8, 4, 260)),
            bind_score_launcher=mock.Mock(),
            token_starts_device=torch.zeros(8, dtype=torch.int32),
        )
        keep_set_selector = SimpleNamespace(
            valid_widths=torch.empty(8, dtype=torch.int32),
            top_indices_i32=torch.zeros(8, 4, dtype=torch.int32),
        )
        prepared = [
            _prepared_eviction(
                _make_request(7),
                request_id=7,
                seq_len=8,
                expected_keep_count=4,
            )
        ]

        with (
            mock.patch.object(
                module,
                "_FixedScoreStagingBuffers",
                return_value=score_staging,
            ),
            mock.patch.object(
                manager,
                "_build_cross_request_keep_set_selector",
                return_value=keep_set_selector,
            ) as build_selection,
        ):
            resources = manager._eager_resources_for(layout, prepared)

        score_staging.bind_score_launcher.assert_called_once_with(
            keep_set_selector.valid_widths,
            manager.score_aggregation,
        )
        plan = build_selection.call_args.args[0]
        assert plan.eviction_mode == eviction_mode
        assert build_selection.call_args.kwargs["normalize_scores"] is normalize_scores
        assert resources.score_staging is score_staging
        assert resources.keep_set_selector is keep_set_selector

    def test_bulk_page_table_copy_uses_immutable_host_snapshots(self):
        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            _FixedScoreStagingBuffers,
        )

        device = torch.device("cuda")
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

        staging = _FixedScoreStagingBuffers.__new__(_FixedScoreStagingBuffers)
        staging.device = device
        staging.max_requests = 1
        staging.page_count = 5
        staging.copy_block_count = 8
        staging.bulk_copy_done = torch.cuda.Event()
        staging.bulk_consume_done = torch.cuda.Event()
        staging.page_tables_active = False
        staging.copy_done = torch.cuda.Event()
        staging.copy_pending = False
        staging._bulk_offsets_src = torch.empty(
            1, 1, 2, 8, dtype=torch.int32, device="cpu", pin_memory=True
        )
        staging._bulk_copy_idx_src = torch.arange(
            1, dtype=torch.int32, device="cpu", pin_memory=True
        )
        staging.block_offsets_device = torch.empty(1, 1, 2, 8, dtype=torch.int32, device=device)
        staging.copy_done.record(current_stream)

        manager = SimpleNamespace(
            host_kv_cache_block_offsets=host_table,
            kv_factor=2,
            layer_offsets={10: 0},
            layer_to_pool_mapping_dict={0: 0},
            index_mapper=SimpleNamespace(gather_k_block_offsets=gather),
            index_scales=torch.tensor([2], dtype=torch.int32, device="cpu", pin_memory=True),
            kv_offset=torch.tensor([1], dtype=torch.int32, device="cpu", pin_memory=True),
            _stream=manager_stream,
        )

        with torch.cuda.stream(manager_stream):
            torch.cuda._sleep(50_000_000)
        with mock.patch.object(
            torch,
            "index_select",
            side_effect=AssertionError("page-table staging used torch.index_select"),
        ):
            assert staging._stage_page_tables_bulk(
                manager,
                [7],
                current_stream,
                staging._bulk_offsets_src,
                staging.block_offsets_device,
                staging.copy_block_count,
            )
        assert staging._bulk_offsets_src.shape[-1] == 8
        assert staging._bulk_offsets_src.shape[1] == 1

        # Mutate both persistent V2 host inputs before the delayed kernel reads.
        # The staged result must still reflect row 0 values [3, 4, 5, 6, 7].
        host_table[0, 0, 0, :5] = torch.tensor([13, 14, 15, 16, 17], dtype=torch.int32)
        selected_slot[0] = 1
        current_stream.synchronize()

        assert staging.block_offsets_device[0, 0, 0, :5].tolist() == [6, 8, 10, 12, 14]
        assert staging.block_offsets_device[0, 0, 1, :5].tolist() == [7, 9, 11, 13, 15]

        host_table[0, 0, 0, :5] = torch.tensor([18, 19, 20, 21, 22], dtype=torch.int32)
        selected_slot[0] = 0
        with torch.cuda.stream(manager_stream):
            torch.cuda._sleep(50_000_000)
        assert staging._stage_page_tables_bulk(
            manager,
            [7],
            current_stream,
            staging._bulk_offsets_src,
            staging.block_offsets_device,
            staging.copy_block_count,
        )
        host_table[0, 0, 0, :5] = torch.tensor([23, 24, 25, 26, 27], dtype=torch.int32)
        selected_slot[0] = 1
        current_stream.synchronize()

        assert staging.block_offsets_device[0, 0, 0, :5].tolist() == [36, 38, 40, 42, 44]
        assert staging.block_offsets_device[0, 0, 1, :5].tolist() == [37, 39, 41, 43, 45]

    def test_next_bulk_copy_waits_for_page_table_consumers(self):
        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            _FixedScoreStagingBuffers,
        )

        device = torch.device("cuda")
        current_stream = torch.cuda.current_stream(device)
        manager_stream = torch.cuda.Stream(device=device)
        host_table = torch.zeros(1, 1, 2, 4, dtype=torch.int32, device="cpu", pin_memory=True)
        host_table[0, 0, 0] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)

        def gather_k_block_offsets(source, destination, request_ids, num_blocks):
            assert request_ids == [7]
            destination[:, :1, 0, :num_blocks].copy_(source[:, :1, 0, :num_blocks])

        staging = _FixedScoreStagingBuffers.__new__(_FixedScoreStagingBuffers)
        staging.device = device
        staging.max_requests = 1
        staging.copy_block_count = 4
        staging.bulk_copy_done = torch.cuda.Event()
        staging.bulk_consume_done = torch.cuda.Event()
        staging.page_tables_active = False
        staging.copy_done = torch.cuda.Event()
        staging.copy_pending = False
        staging._bulk_offsets_src = torch.empty(
            1, 1, 2, 4, dtype=torch.int32, device="cpu", pin_memory=True
        )
        staging._bulk_copy_idx_src = torch.arange(
            1, dtype=torch.int32, device="cpu", pin_memory=True
        )
        staging.block_offsets_device = torch.empty(1, 1, 2, 4, dtype=torch.int32, device=device)
        staging.copy_done.record(current_stream)
        manager = SimpleNamespace(
            host_kv_cache_block_offsets=host_table,
            kv_factor=2,
            index_mapper=SimpleNamespace(
                gather_k_block_offsets=mock.Mock(side_effect=gather_k_block_offsets)
            ),
            index_scales=torch.tensor([2], dtype=torch.int32, device="cpu", pin_memory=True),
            kv_offset=torch.tensor([1], dtype=torch.int32, device="cpu", pin_memory=True),
            _stream=manager_stream,
        )

        assert staging._stage_page_tables_bulk(
            manager,
            [7],
            current_stream,
            staging._bulk_offsets_src,
            staging.block_offsets_device,
            staging.copy_block_count,
        )
        manager_stream.synchronize()
        snapshot = torch.empty_like(staging.block_offsets_device)
        torch.cuda._sleep(20_000_000)
        snapshot.copy_(staging.block_offsets_device)
        staging.page_tables_active = True
        staging.mark_page_tables_consumed(manager_stream)

        host_table[0, 0, 0] = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
        assert staging._stage_page_tables_bulk(
            manager,
            [7],
            current_stream,
            staging._bulk_offsets_src,
            staging.block_offsets_device,
            staging.copy_block_count,
        )
        current_stream.synchronize()

        assert snapshot[0, 0, 0].tolist() == [2, 4, 6, 8]
        assert staging.block_offsets_device[0, 0, 0].tolist() == [10, 12, 14, 16]

    def test_cross_stream_staging_is_rejected_before_page_table_query(self):
        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            _FixedScoreStagingBuffers,
            _FixedScoreStreamMismatch,
        )

        staging = _FixedScoreStagingBuffers.__new__(_FixedScoreStagingBuffers)
        staging.device = torch.device("cuda")
        staging.max_requests = 8
        staging.stream = SimpleNamespace(device=torch.device("cuda:0"), cuda_stream=4)
        staging.page_tables_active = False
        staging.copy_pending = False
        staging.copy_done = SimpleNamespace(query=mock.Mock(), synchronize=mock.Mock())
        staging.draft_block_offsets_device = None
        manager = mock.Mock()

        other_stream = SimpleNamespace(device=torch.device("cuda:0"), cuda_stream=5)
        with mock.patch.object(torch.cuda, "current_stream", return_value=other_stream):
            with pytest.raises(_FixedScoreStreamMismatch, match="first CUDA stream"):
                staging.stage(manager, [1], [8.0], [0])
        staging.copy_done.query.assert_not_called()
        staging.copy_done.synchronize.assert_not_called()

    def test_staged_page_tables_bypass_per_request_cuda_materialization(self):
        manager = _make_triattention()
        get_batch = mock.Mock()
        manager.kv_cache_manager = SimpleNamespace(get_batch_cache_indices=get_batch)
        staging = SimpleNamespace(
            stage=mock.Mock(return_value=True),
        )
        prepared = [
            _prepared_eviction(
                SimpleNamespace(),
                request_id=7,
                round_start=8,
                seq_len=8,
                prompt_len=3,
                expected_keep_count=6,
                protected_tail=2,
            ),
            _prepared_eviction(
                SimpleNamespace(),
                request_id=8,
                round_start=9,
                seq_len=9,
                prompt_len=5,
                expected_keep_count=6,
                protected_tail=3,
            ),
        ]

        manager._attach_page_ids(prepared, staging)

        staging.stage.assert_called_once_with(
            manager.kv_cache_manager,
            [7, 8],
            [8, 9],
            [3, 5],
            [8, 9],
            [10, 12],
            draft_manager=None,
        )
        assert all(not hasattr(item, "page_ids") for item in prepared)

    @pytest.mark.parametrize("request_count", [1, 7, 8])
    def test_staging_stages_dense_and_swa_tables_and_rejects_stream_changes(self, request_count):
        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            _FixedScoreStagingBuffers,
            _FixedScoreStreamMismatch,
        )

        device = torch.device("cuda")
        max_requests = 8
        page_count = 3
        seq_len = 7
        page_table_token_capacity = 11
        layer_elements = max_requests * page_count * 2 * 1 * 4 * 4
        shared = torch.randn(2 * layer_elements, device=device)
        pools = [
            shared[:layer_elements].view(max_requests * page_count, 2, 1, 4, 4),
            shared[layer_elements:].view(max_requests * page_count, 2, 1, 4, 4),
            torch.randn(max_requests * page_count, 2, 1, 4, 4, device=device),
            torch.randn(max_requests * page_count, 2, 1, 4, 4, device=device),
        ]
        dense_groups = [[0, 1], [2]]
        representatives = [0, 2, 3]
        q_real = torch.randn(4, 2, 4, dtype=torch.float64, device=device)[..., ::2]
        q_imag = torch.randn(4, 2, 4, dtype=torch.float64, device=device)[..., ::2]
        mlr = torch.randn(4, 2, 4, dtype=torch.float64, device=device)[..., ::2]
        freq = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64, device=device)[::2]
        omega = torch.tensor([0.01, 0.0, 0.03, 0.0], dtype=torch.float64, device=device)[::2]
        offsets = torch.tensor([1.0, 0.0, 2.0, 0.0], dtype=torch.float64, device=device)[::2]
        assert not q_real.is_contiguous()
        assert not freq.is_contiguous()
        assert not omega.is_contiguous()
        assert not offsets.is_contiguous()
        staging = _FixedScoreStagingBuffers(
            pools,
            dense_groups,
            [0, 1, 2],
            representatives,
            max_requests,
            seq_len,
            2,
            2,
            q_real,
            q_imag,
            mlr,
            freq,
            offsets,
            omega,
            page_table_token_capacity=page_table_token_capacity,
        )
        assert staging.bucket_seq_len == seq_len
        assert staging.page_table_token_capacity == page_table_token_capacity
        assert staging.page_count == page_count
        assert staging.offsets.dtype == torch.float32
        assert staging.offsets.is_contiguous()
        assert staging.omega.dtype == torch.float32
        assert staging.omega.is_contiguous()
        fused = staging.fused_group
        for calibration in (*fused.pointer_middle[2:], *fused.pointer_tail):
            assert calibration.dtype == torch.float32
            assert calibration.is_contiguous()
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
        manager = SimpleNamespace(
            enable_swa_scratch_reuse=False,
            host_kv_cache_block_offsets=host_table,
            kv_factor=2,
            index_mapper=SimpleNamespace(gather_k_block_offsets=gather),
            index_scales=torch.full((3,), 2, dtype=torch.int32, pin_memory=True),
            kv_offset=torch.ones(3, dtype=torch.int32, pin_memory=True),
            _stream=torch.cuda.Stream(device=device),
        )

        assert not staging.stage(
            manager,
            request_ids,
            [2**31] * request_count,
            token_starts,
            [seq_len] * request_count,
            [10] * request_count,
        )
        assert gather.call_count == 0
        with mock.patch.object(
            torch,
            "index_select",
            side_effect=AssertionError("page-table staging used torch.index_select"),
        ):
            assert staging.stage(
                manager,
                request_ids,
                round_starts,
                token_starts,
                [seq_len] * request_count,
                [10] * request_count,
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
        calls = gather.call_count
        other_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(other_stream):
            with pytest.raises(_FixedScoreStreamMismatch, match="first CUDA stream"):
                staging.stage(manager, request_ids, round_starts, token_starts)
        assert gather.call_count == calls

    @pytest.mark.parametrize("request_count", [1, 7, 8])
    @pytest.mark.parametrize("aggregation", ["mean", "max"])
    def test_fixed_score_matches_torch_oracle_across_two_groups(self, request_count, aggregation):
        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
            _FixedScoreGroup,
        )

        device = torch.device("cuda")
        torch.manual_seed(20260703 + request_count)
        max_requests = 8
        page_count = 2
        seq_len = 7
        prompt_len = 2
        page_ids = torch.arange(max_requests * page_count, dtype=torch.int64, device=device).view(
            max_requests, page_count
        )
        layer_elements = max_requests * page_count * 2 * 1 * 4 * 4
        shared = torch.randn(2 * layer_elements, device=device)
        pools = [
            shared[:layer_elements].view(max_requests * page_count, 2, 1, 4, 4),
            shared[layer_elements:].view(max_requests * page_count, 2, 1, 4, 4),
            torch.randn(max_requests * page_count, 2, 1, 4, 4, device=device),
        ]
        storage_groups = [[0, 1], [2]]
        q_real = torch.randn(3, 2, 4, device=device)[..., ::2]
        q_imag = torch.randn(3, 2, 4, device=device)[..., ::2]
        mlr = torch.randn(3, 2, 4, device=device)[..., ::2]
        freq = torch.tensor([0.7, 0.0, 1.3, 0.0], device=device)[::2]
        omega = torch.tensor([0.013, 0.0, 0.071, 0.0], device=device)[::2]
        offsets = torch.tensor([1.0, 0.0, 2.0, 0.0, 4.0, 0.0], device=device)[::2]
        assert not q_real.is_contiguous()
        assert not q_imag.is_contiguous()
        assert not mlr.is_contiguous()
        assert not freq.is_contiguous()
        assert not omega.is_contiguous()
        assert not offsets.is_contiguous()
        round_device = torch.arange(max_requests, dtype=torch.int32, device=device) + 9
        round_starts = round_device[:request_count].tolist()
        token_starts_device = torch.full(
            (max_requests,), prompt_len, dtype=torch.int32, device=device
        )
        seq_lens = [seq_len - request % 2 for request in range(request_count)]
        phase = (round_device[:, None, None] + offsets[None, :, None]) * omega[None, None]
        oracle = _torch_tri_score_oracle(
            pools,
            page_ids[:request_count],
            seq_lens,
            round_starts,
            q_real,
            q_imag,
            mlr,
            freq,
            omega,
            offsets,
            [0, 1, 2],
            aggregation,
        )
        for layers in storage_groups:
            group = _FixedScoreGroup(
                pools,
                layers,
                max_requests,
                page_count,
                seq_len,
                2,
                _encode_block_offsets(page_ids),
                [0] * len(layers),
                q_real,
                q_imag,
                mlr,
                freq,
                omega,
                offsets,
                min_prompt_len=prompt_len,
            )
            valid_widths = torch.empty(request_count, dtype=torch.int32, device=device)
            fixed = group.launch(
                request_count,
                torch.tensor(seq_lens, dtype=torch.int32, device=device),
                valid_widths,
                round_device,
                token_starts_device,
                torch.cos(phase).mean(dim=1),
                torch.sin(phase).mean(dim=1),
                aggregation,
            )
            assert valid_widths.tolist() == [seq_len - prompt_len for seq_len in seq_lens]
            assert fixed.shape == (
                request_count,
                len(layers),
                2,
                seq_len - prompt_len,
            )
            for request in range(request_count):
                for layer_slot, layer in enumerate(layers):
                    valid_width = seq_lens[request] - prompt_len
                    segment = fixed[request, layer_slot, :, :valid_width]
                    expected = oracle[request * len(pools) + layer][:, prompt_len:]
                    torch.testing.assert_close(segment, expected, rtol=5e-3, atol=5e-3)
                    selected = torch.topk(segment.max(dim=0).values, 3).indices.sort().values
                    expected_selected = (
                        torch.topk(expected.max(dim=0).values, 3).indices.sort().values
                    )
                    assert torch.equal(selected, expected_selected)

    @pytest.mark.parametrize("request_count", [1, 7, 8])
    @pytest.mark.parametrize("aggregation", ["mean", "max"])
    def test_fused_score_spans_distinct_storages_and_block_tables(self, request_count, aggregation):
        """ONE launch over layers in DISTINCT storages with DISTINCT block tables.

        This is the production V2 shape: get_buffers wraps every layer as its
        own TensorWrapper storage and every layer allocates its own pages, so
        the fused path must not assume a shared storage anchor or a shared
        per-request block table.
        """
        from tensorrt_llm._torch.kv_cache_compression.triattention import triattention_kernels
        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
            _FixedScoreStagingBuffers,
            _FixedScoreStreamMismatch,
        )
        from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
            _FixedScoreGroup,
        )

        device = torch.device("cuda")
        torch.manual_seed(20260707 + request_count)
        max_requests = request_count
        page_count = 2
        seq_len = 7
        prompt_len = 1
        num_layers = 3
        # Three SEPARATE allocations (distinct storages, like V2 TensorWrapper).
        pools = [
            torch.randn(max_requests * page_count, 2, 1, 4, 4, device=device)
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
        q_real = torch.randn(num_layers, 2, 2, device=device)
        q_imag = torch.randn(num_layers, 2, 2, device=device)
        mlr = torch.randn(num_layers, 2, 2, device=device)
        freq = torch.tensor([0.7, 1.3], device=device)
        omega = torch.tensor([0.013, 0.071], device=device)
        offsets = torch.tensor([1.0, 2.0, 4.0], device=device)
        round_device = torch.arange(max_requests, dtype=torch.int32, device=device) + 9
        round_starts = round_device[:request_count].tolist()
        token_starts = torch.full((max_requests,), prompt_len, dtype=torch.int32, device=device)
        seq_lens = [seq_len - request % 2 for request in range(request_count)]
        layer_order = list(range(num_layers))
        block_offsets = _encode_block_offsets(page_ids_3d)
        group = _FixedScoreGroup(
            pools,
            layer_order,
            max_requests,
            page_count,
            seq_len,
            2,
            block_offsets,
            layer_order,  # slot i holds layer i's tables
            q_real,
            q_imag,
            mlr,
            freq,
            omega,
            offsets,
            min_prompt_len=prompt_len,
        )
        valid_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        valid_widths = torch.empty(request_count, dtype=torch.int32, device=device)
        mean_cos = torch.empty(request_count, 2, dtype=torch.float32, device=device)
        mean_sin = torch.empty_like(mean_cos)
        if aggregation == "mean":
            triattention_kernels.prepare_mean_phase(
                round_device,
                offsets,
                omega,
                mean_cos,
                mean_sin,
                request_count,
            )
        score_sentinel = -12345.0
        group.output.fill_(score_sentinel)
        checked = group.launch(
            request_count,
            valid_seq_lens,
            valid_widths,
            round_device,
            token_starts,
            mean_cos,
            mean_sin,
            aggregation,
        ).clone()
        staging = _FixedScoreStagingBuffers.__new__(_FixedScoreStagingBuffers)
        staging.device = group.output.device
        staging.max_requests = request_count
        staging.fused_group = group
        staging.round_starts_device = round_device
        staging.valid_seq_lens_device = valid_seq_lens
        staging.token_starts_device = token_starts
        staging.mean_cos = mean_cos
        staging.mean_sin = mean_sin
        staging.offsets = offsets
        staging.omega = omega
        staging.stream = None
        staging._phase_runner = None
        staging._phase_args = ()
        staging._score_runner = None
        staging._score_args = ()
        staging.bind_score_launcher(valid_widths, aggregation)
        group.output.fill_(score_sentinel)
        with (
            mock.patch.object(
                triattention_kernels,
                "prepare_mean_phase",
                side_effect=AssertionError("checked phase wrapper was called"),
            ),
            mock.patch.object(
                group,
                "launch",
                side_effect=AssertionError("checked score wrapper was called"),
            ),
        ):
            fixed = staging.launch_prepared_score().clone()
        torch.testing.assert_close(fixed, checked, rtol=0, atol=0)
        assert valid_widths.tolist() == [seq_len - prompt_len for seq_len in seq_lens]

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
            aggregation,
        )
        for request in range(request_count):
            for layer_slot, layer in enumerate(layer_order):
                valid_width = seq_lens[request] - prompt_len
                segment = fixed[request, layer_slot, :, :valid_width]
                expected = oracle[request * num_layers + layer][:, prompt_len:]
                torch.testing.assert_close(segment, expected, rtol=5e-3, atol=5e-3)

        round_device.add_(17)
        block_offsets.copy_(_encode_block_offsets(page_ids_3d.roll(1, dims=2)))
        valid_seq_lens.copy_(
            torch.tensor(
                [seq_len - (request + 1) % 2 for request in range(request_count)],
                dtype=torch.int32,
                device=device,
            )
        )
        if aggregation == "mean":
            triattention_kernels.prepare_mean_phase(
                round_device,
                offsets,
                omega,
                mean_cos,
                mean_sin,
                request_count,
            )
        expected_second_widths = valid_seq_lens - prompt_len
        group.output.fill_(score_sentinel)
        valid_widths.fill_(-1)
        checked_second = group.launch(
            request_count,
            valid_seq_lens,
            valid_widths,
            round_device,
            token_starts,
            mean_cos,
            mean_sin,
            aggregation,
        ).clone()
        group.output.fill_(score_sentinel)
        valid_widths.fill_(-1)
        with (
            mock.patch.object(
                triattention_kernels,
                "prepare_mean_phase",
                side_effect=AssertionError("checked phase wrapper was called"),
            ),
            mock.patch.object(
                group,
                "launch",
                side_effect=AssertionError("checked score wrapper was called"),
            ),
        ):
            second_launch = staging.launch_prepared_score().clone()
        torch.testing.assert_close(second_launch, checked_second, rtol=0, atol=0)
        assert torch.equal(valid_widths, expected_second_widths)
        assert not torch.equal(second_launch, fixed)

        other_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(other_stream):
            with pytest.raises(_FixedScoreStreamMismatch, match="staging CUDA stream"):
                staging.launch_prepared_score()


class TestKernelMaskedSwa:
    def test_layer_partition_uses_local_model_config(self):
        mgr = _make_triattention()
        mgr.model_path = "/models/gpt-oss"
        mgr.top_B = 128
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
            dense, sliding, window = mgr._attention_layer_partition(4)

        load.assert_called_once_with(
            "/models/gpt-oss", trust_remote_code=True, local_files_only=True
        )
        assert dense == [1, 3]
        assert sliding == [0, 2]
        assert window == 128

    def test_layer_partition_rejects_decode_budget_smaller_than_window(self):
        mgr = _make_triattention()
        mgr.model_path = "/models/gpt-oss"
        mgr.top_B = 127
        mgr.kv_cache_manager = SimpleNamespace(pp_layers=[0, 1])
        config = _make_hf_config(
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=128,
        )

        with (
            mock.patch("transformers.AutoConfig.from_pretrained", return_value=config),
            pytest.raises(ValueError, match="decode budget top_B=127"),
        ):
            mgr._attention_layer_partition(2)


class TestFactory:
    def test_returns_triattention_instance_with_v2(self):
        # A plain V2 manager (block reuse off) yields a TriAttention instance.
        # Calibration is deferred to the first request, so construction needs
        # no calibration file or CUDA.
        fake_v2 = _make_fake_v2(enable_block_reuse=False)
        cfg = TriAttentionKvCacheCompressionConfig(
            top_B=32, beta=16, model_path="/models/test", calibration_path="/calib/test.pt"
        )
        mgr = create_kv_cache_compression_manager(cfg, kv_cache_manager=fake_v2)
        assert isinstance(mgr, TriAttention)
        assert mgr.top_B == 32
        assert mgr.beta == 16
        assert mgr.kv_cache_manager is fake_v2

    def test_factory_propagates_eviction_mode(self):
        cfg = TriAttentionKvCacheCompressionConfig(
            top_B=64,
            beta=8,
            eviction_mode="per_head",
            model_path="/models/test",
            calibration_path="/calib/test.pt",
        )
        mgr = create_kv_cache_compression_manager(
            cfg, kv_cache_manager=_make_fake_v2(enable_block_reuse=False)
        )
        assert isinstance(mgr, TriAttention)
        assert mgr.eviction_mode == "per_head"
