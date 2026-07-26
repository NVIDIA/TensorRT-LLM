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

Config, construction, eviction lifecycle, page-table staging, and the fixed
score buffers; the manager publishes evicted counts via
``LlmRequest.py_num_compressed_tokens``. Draft contracts live in
``test_triattention_draft_cocompaction.py``."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from conftest import make_bare_staging as _make_bare_staging
from conftest import make_cute_buffers as _make_cute_buffers
from conftest import make_eviction_input as _make_eviction_input
from conftest import make_fake_v2 as _make_fake_v2
from conftest import make_request as _make_request
from conftest import make_staging_manager as _make_staging_manager
from conftest import make_tri_config as _make_tri_config
from conftest import make_triattention as _make_triattention
from conftest import mocked_eviction_internals as _mocked_eviction_internals
from conftest import torch_tri_score_oracle as _torch_tri_score_oracle

# TriAttention lives in the kv_cache_compression package. It exposes only the
# compression manager -- no attention classes or KV-cache-manager subclass.
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import TriAttention

# Framework base class lives in pyexecutor.resource_manager; the factory lives
# in pyexecutor._util (next to _create_kv_cache_manager), matching #15106.
from tensorrt_llm._torch.pyexecutor._util import create_kv_cache_compression_manager

# The SM100 CuTe kernel is the only score path, so every test that actually
# launches scores (or builds the real staging buffers, whose constructor
# compiles the kernel) is SM100-only, like the production feature itself.
requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="TriAttention score requires SM100",
)


def _set_request_state(manager, request_id, *, generation_steps=0, evicted_tokens=0):
    state = {
        "generation_steps": generation_steps,
        "evicted_tokens": evicted_tokens,
    }
    manager._request_states[request_id] = state
    return state


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
    def test_factory_returns_triattention_and_propagates_config_fields(self):
        # Calibration is deferred to the first request, so construction needs
        # no calibration file or CUDA.
        fake_v2 = _make_fake_v2(enable_block_reuse=False)
        cfg = _make_tri_config(budget=32, beta=16, eviction_mode="per_head")
        mgr = create_kv_cache_compression_manager(cfg, kv_cache_manager=fake_v2)
        assert isinstance(mgr, TriAttention)
        assert mgr.budget == 32
        assert mgr.beta == 16
        assert mgr.eviction_mode == "per_head"
        assert mgr.kv_cache_manager is fake_v2


class TestTriAttentionClass:
    def test_request_init_and_finish_lifecycle(self):
        # Init: speculative capacity accepted, manager marked, state tracked.
        # Finish: state cleared; buffers and the step's batch stay resident.
        manager = _make_fake_v2()
        manager.num_extra_kv_tokens = 4
        manager._kv_reserve_draft_tokens = 4
        triattention = TriAttention(_make_tri_config(budget=8), manager)

        triattention.on_request_init(_make_request(11))
        triattention.on_request_init(_make_request(12))

        assert triattention.adjusts_generation_kv_length is True
        assert manager.kv_compression_manages_history
        assert set(triattention._request_states) == {11, 12}

        buffers = object()
        triattention._buffers_built = True
        triattention._score_scratch = buffers
        batch = SimpleNamespace()
        triattention._inflight_scheduled_batch = batch
        triattention.on_request_finish(_make_request(11))
        triattention.on_request_finish(_make_request(12))
        assert triattention._request_states == {}
        assert triattention._inflight_scheduled_batch is batch
        assert triattention._buffers_built and triattention._score_scratch is buffers

    def test_resolve_accepts_flat_pt(self, flat_calibration_pt):
        mgr = _make_triattention()
        mgr.calibration_path = flat_calibration_pt
        mgr.model_path = None
        loaded = mgr._resolve_calibration()
        for key in ("E_q", "E_q_norm", "omega", "freq_scale_sq"):
            assert key in loaded

    def test_resolve_converts_official_layout(self, tmp_path):
        # PRODUCT CONTRACT: the official R-KV {metadata, stats} layout is
        # converted to the flat runtime schema at load; rope tables derive
        # from the model config.
        pytest.importorskip("transformers")
        num_layers, num_heads, freq_count = 2, 2, 4
        stats, sampled = {}, []
        for layer in range(num_layers):
            for head in range(num_heads):
                stats[f"layer{layer:02d}_head{head:02d}"] = {
                    "q_mean_real": torch.full((freq_count,), float(10 * layer + head)),
                    "q_mean_imag": torch.full((freq_count,), float(layer - head)),
                    "q_abs_mean": torch.full((freq_count,), float(1 + layer + head)),
                }
                sampled.append((layer, head))
        path = tmp_path / "official.pt"
        torch.save({"metadata": {"sampled_heads": sampled}, "stats": stats}, path)
        mgr = _make_triattention()
        mgr.calibration_path = str(path)
        config = _make_hf_config(rope_parameters={"rope_type": "default", "rope_theta": 10000.0})

        with mock.patch("transformers.AutoConfig.from_pretrained", return_value=config):
            converted = mgr._resolve_calibration()

        assert set(converted) == {"E_q", "E_q_norm", "omega", "freq_scale_sq"}
        assert converted["E_q"].shape == (num_layers, num_heads, freq_count)
        torch.testing.assert_close(
            converted["E_q"][1, 0].cpu(),
            torch.complex(torch.full((freq_count,), 10.0), torch.full((freq_count,), 1.0)),
        )
        torch.testing.assert_close(
            converted["E_q_norm"][1, 1].cpu(), torch.full((freq_count,), 3.0)
        )
        assert converted["omega"].numel() == freq_count
        idx = torch.arange(0, 2 * freq_count, 2, dtype=torch.float32)
        torch.testing.assert_close(
            converted["omega"].cpu(), 1.0 / (10000.0 ** (idx / (2 * freq_count)))
        )
        assert torch.equal(converted["freq_scale_sq"].cpu(), torch.ones(freq_count))

    def test_rope_tables_resolve_theta_and_attention_factor(self, tmp_path):
        # transformers>=5.5 folds rope_theta into ``rope_parameters`` and drops
        # "default" from ROPE_INIT_FUNCTIONS; resolution must find the true
        # theta and scaled-rope attention factor on both config generations
        # (the silent base-10000 analytic fallback was the B1 bug).
        pytest.importorskip("transformers")
        import json

        def config_dir(name, body):
            d = tmp_path / name
            d.mkdir()
            (d / "config.json").write_text(json.dumps(body))
            return str(d)

        common = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_hidden_layers": 2,
            "head_dim": 64,
            "max_position_embeddings": 8192,
        }
        plain = config_dir("plain", {**common, "rope_theta": 1000000.0})
        yarn = config_dir(
            "yarn",
            {
                **common,
                "rope_theta": 150000.0,
                "rope_scaling": {
                    "rope_type": "yarn",
                    "factor": 4.0,
                    "original_max_position_embeddings": 2048,
                    "attention_factor": 1.25,
                },
            },
        )
        mgr = _make_triattention()
        freq_count = 32

        mgr.model_path = plain
        omega, freq_scale_sq = mgr._rope_tables(freq_count)
        idx = torch.arange(0, 64, 2, dtype=torch.float32)
        torch.testing.assert_close(omega, (1.0 / (1000000.0 ** (idx / 64)))[:freq_count])
        assert torch.equal(freq_scale_sq, torch.ones(freq_count))

        mgr.model_path = yarn
        omega_yarn, freq_scale_sq_yarn = mgr._rope_tables(freq_count)
        # Routed through transformers' yarn init: the explicit attention
        # factor lands squared, and the ladder leaves the plain-theta curve.
        torch.testing.assert_close(freq_scale_sq_yarn, torch.full((freq_count,), 1.25**2))
        assert not torch.allclose(omega_yarn, omega)


class TestCompressedTokenPublication:
    # The monotone publication contract is covered end to end in
    # test_triattention_draft_cocompaction.py.

    def test_identity_selection_is_filtered_before_launch(self):
        # Identity cohorts (seq_len == prompt + budget) are the pre-launch
        # owner no-op: nothing launches and nothing is published.
        manager = _make_triattention(budget=4)
        manager.kv_cache_manager._stream = mock.Mock()
        request = _make_request(7, py_prompt_len=2)
        # seq_len == prompt + budget: the due filter must drop the request.
        cache = SimpleNamespace(
            capacity=6, history_length=0, is_active=True, resize=mock.Mock(return_value=True)
        )
        manager.kv_cache_manager.kv_cache_map = {7: cache}
        state = _set_request_state(manager, 7, generation_steps=127)

        with _mocked_eviction_internals(manager) as internals:
            manager._evict_due_requests(SimpleNamespace(generation_requests=[request]))

        internals.execute.assert_not_called()
        assert request.py_num_compressed_tokens == 0
        assert state["evicted_tokens"] == 0
        cache.resize.assert_not_called()


class TestEvictionLifecycle:
    def test_prepare_does_not_evict_and_update_runs_final_hook_once(self):
        manager = _make_triattention()
        batch = SimpleNamespace(
            context_requests=[],
            context_requests_last_chunk=[],
            generation_requests=[_make_request(7)],
        )
        with mock.patch.object(manager, "_evict_due_requests") as evict_due:
            manager.prepare_resources(batch)
            evict_due.assert_not_called()

            manager.update_resources(batch)

        evict_due.assert_called_once_with(batch)

    @staticmethod
    def _make_due_decode_request(seq_len, *, num_extra_kv_tokens=0, kv_reserve_draft_tokens=0):
        # The growth and protected-tail capacity constants snapshot the
        # manager at construction, so the reserve widths are set up front.
        request = _make_request(
            7,
            py_prompt_len=1024,
            max_beam_num_tokens=seq_len + 1,
        )
        batch = SimpleNamespace(generation_requests=[request])
        fake_v2 = _make_fake_v2()
        fake_v2.num_extra_kv_tokens = num_extra_kv_tokens
        fake_v2._kv_reserve_draft_tokens = kv_reserve_draft_tokens
        mgr = TriAttention(_make_tri_config(budget=8), fake_v2)
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
            num_extra_kv_tokens=num_extra_kv_tokens,
            _kv_reserve_draft_tokens=kv_reserve_draft_tokens,
        )
        mgr._request_states = {}
        _set_request_state(mgr, 7, generation_steps=127)
        mgr.beta = 128
        mgr.budget = 4096
        return mgr, request, batch

    def test_suspended_cache_defers_that_request_pre_launch(self):
        # A suspended cache is a legal overlap-scheduler transient: that
        # request defers (pre-launch, no cadence mutation) while the rest of
        # the cohort proceeds.
        manager, first_request, _ = self._make_due_decode_request(seq_len=1024 + 4096 + 1)
        second_request = _make_request(8, py_prompt_len=1024)
        manager.kv_cache_manager.kv_cache_map[8] = SimpleNamespace(is_active=False)
        first_state = manager._request_states[7]
        second_state = _set_request_state(manager, 8, generation_steps=127)
        batch = SimpleNamespace(generation_requests=[first_request, second_request])

        with _mocked_eviction_internals(manager) as internals:
            manager._evict_due_requests(batch)

        # Only the active request launched; the suspended one deferred whole.
        eviction_inputs = internals.execute.call_args.args[0]
        assert [item.request.py_request_id for item in eviction_inputs] == [7]
        assert first_state["generation_steps"] == 128
        assert second_state["generation_steps"] == 127

    # ``accepted`` enters the prepared item linearly; the zero and maximal
    # boundary rows pin the whole family.
    @pytest.mark.parametrize("accepted", [0, 3])
    def test_overlap_tail_is_excluded_from_selection_and_compacted(self, accepted):
        confirmed = 1024 + 4096 + 1 + accepted
        reserve = 2
        current_growth = 4
        tail = reserve + current_growth
        retained = 1024 + 4096
        # Growth constant = 1 + _kv_reserve_draft_tokens for batch members.
        mgr, request, batch = self._make_due_decode_request(
            seq_len=confirmed,
            num_extra_kv_tokens=reserve,
            kv_reserve_draft_tokens=current_growth - 1,
        )
        request.py_num_accepted_draft_tokens = accepted
        cache = mgr.kv_cache_manager.kv_cache_map[7]
        cache.capacity = confirmed + tail
        mgr._inflight_scheduled_batch = SimpleNamespace(generation_requests=[request])
        draft_manager = _make_fake_v2(is_draft=True)
        draft_cache = SimpleNamespace(is_active=True, resize=mock.Mock(return_value=True))
        draft_manager.kv_cache_map = {7: draft_cache}
        draft_manager._stream = mock.Mock()
        mgr.draft_kv_cache_manager = draft_manager
        # Injected post-construction: mirror the ctor-cached manager-lifetime tail.
        mgr._draft_protected_tail_capacity = 1

        with _mocked_eviction_internals(mgr) as internals:
            mgr._evict_due_requests(batch)

        # Tail excluded from the source length; keep target = prompt + budget.
        internals.execute.assert_called_once()
        (launched,) = internals.execute.call_args.args[0]
        assert launched.request is request
        assert launched.target_cache is cache
        assert launched.draft_cache is draft_cache
        assert launched.source_length == confirmed
        assert launched.logical_source_length == confirmed
        assert launched.prompt_length == 1024
        assert launched.target_tail_length == tail
        assert request.py_num_compressed_tokens == confirmed - retained
        cache.resize.assert_called_once_with(retained + tail, None)
        draft_cache.resize.assert_called_once_with(retained + 1, None)

    def test_confirmed_length_comes_from_capacity_ledger_not_logical_length(self):
        # The due-branch source length must come from the physical capacity
        # ledger (capacity minus the protected tail), never the logical length.
        physical_confirmed = 6100
        manager = _make_triattention(beta=128)
        _set_request_state(manager, 7, generation_steps=127, evicted_tokens=100)
        cache = SimpleNamespace(
            capacity=physical_confirmed,
            history_length=1024,
            is_active=True,
            resize=mock.Mock(return_value=True),
        )
        manager.kv_cache_manager.kv_cache_map = {7: cache}
        manager.kv_cache_manager.pp_layers = [0, 1]
        request = _make_request(
            7,
            py_prompt_len=1024,
            max_beam_num_tokens=999999,
            py_draft_tokens=[1, 2, 3, 4],
        )

        with _mocked_eviction_internals(manager) as internals:
            manager._evict_due_requests(SimpleNamespace(generation_requests=[request]))

        eviction_inputs = internals.execute.call_args.args[0]
        assert eviction_inputs[0].source_length == physical_confirmed
        # The logical position restores everything already evicted.
        assert eviction_inputs[0].logical_source_length == physical_confirmed + 100
        cache.resize.assert_called_once_with(1024 + manager.budget, None)

    def test_one_model_draft_co_compression_contract_is_accepted(self):
        # Construction accepts the separate draft manager, and the executor
        # call-site gate accepts the one-model MTP roundtrip (base-ctor
        # marking is asserted in the executor manager tests).
        draft_manager = _make_fake_v2(is_draft=True)
        TriAttention(
            _make_tri_config(budget=8),
            _make_fake_v2(),
            draft_kv_cache_manager=draft_manager,
        )

        from tensorrt_llm._torch.pyexecutor._util import validate_kv_cache_compression_with_spec
        from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

        validate_kv_cache_compression_with_spec(
            _make_tri_config(budget=8),
            MTPDecodingConfig(max_draft_len=1),
            draft_manager,
        )

    @pytest.mark.parametrize(
        "reserved_draft,expected_growth",
        [
            (0, 1),
            # The reserved draft width protects capacity regardless of any
            # step's actual draft length: growth is the cached constant.
            (6, 7),
        ],
    )
    def test_prepare_snapshots_fixed_linear_generation_growth(
        self, reserved_draft, expected_growth
    ):
        manager = _make_fake_v2()
        manager._kv_reserve_draft_tokens = reserved_draft
        manager.kv_cache_map = {
            7: SimpleNamespace(capacity=106, is_active=True),
        }
        triattention = TriAttention(_make_tri_config(budget=8), manager)
        batch = SimpleNamespace(
            context_requests=[],
            generation_requests=[_make_request(7, py_draft_tokens=[1, 2, 3])],
        )

        triattention.prepare_resources(batch)

        assert triattention._inflight_scheduled_batch is batch
        # Members of the prepared batch grow by the cached constant; others
        # by zero. The prepared batch itself is the identity early-out.
        assert triattention._inflight_generation_growth(SimpleNamespace(), 7) == expected_growth
        assert triattention._inflight_generation_growth(SimpleNamespace(), 99) == 0
        assert triattention._inflight_generation_growth(batch, 7) == 0


class TestFixedScoreMetadata:
    def test_union_forces_normalized_scores(self):
        # Union eviction always z-normalizes: False is coerced to True at construction.
        triattention = _make_triattention(budget=4, eviction_mode="union", normalize_scores=False)
        assert triattention.normalize_scores is True

    def test_execute_rejects_int32_overflowing_logical_source_lengths(self):
        # Round starts past the int32 metadata range fail loudly (in the host
        # metadata build) before any GPU work is enqueued.
        device = torch.device("cuda", torch.cuda.current_device())
        staging = _make_bare_staging(device, max_requests=1, staged_blocks_per_seq=8)
        gather = mock.Mock()
        manager = _make_staging_manager(
            torch.zeros(1, 2, 2, 12, dtype=torch.int32), gather, torch.cuda.Stream(device=device)
        )
        staging.kv_cache_manager = manager
        eviction_inputs = [
            _make_eviction_input(request_id=7, source_length=64, logical_source_length=2**31)
        ]

        with pytest.raises((RuntimeError, OverflowError, ValueError)):
            staging._execute_eviction_round(eviction_inputs)
        assert gather.call_count == 0

    def test_bulk_page_table_copy_snapshots_and_orders_consumers(self):
        """The bulk copy stages immutable host snapshots, and the next copy
        waits for the previous cohort's consumers.

        Both persistent V2 host inputs (the block-offset table and the
        index-mapper slot assignment) may mutate as soon as staging returns;
        the staged device tables must reflect the values at staging time. A
        subsequent bulk copy must also wait until the previous round's
        consumers (ordered by the round executor's completion event) are done.
        """
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
        staging = _make_bare_staging(device, max_requests=1, staged_blocks_per_seq=8)
        staging._staging_reuse_event.record(current_stream)
        manager = _make_staging_manager(host_table, gather, manager_stream)

        def stage_once():
            # Raises on any staging failure; success returns None.
            staging._stage_block_offsets(
                manager,
                [7],
                staging._block_offsets_host,
                staging._block_offsets_device,
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
        assert staging._block_offsets_host.shape == (1, 1, 2, 8)
        host_table[0, 0, 0, :5] = torch.tensor([13, 14, 15, 16, 17], dtype=torch.int32)
        selected_slot[0] = 1
        current_stream.synchronize()

        assert staging._block_offsets_device[0, 0, 0, :5].tolist() == [6, 8, 10, 12, 14]
        assert staging._block_offsets_device[0, 0, 1, :5].tolist() == [7, 9, 11, 13, 15]

        # Round 2: same contract on a re-staged cohort.
        host_table[0, 0, 0, :5] = torch.tensor([18, 19, 20, 21, 22], dtype=torch.int32)
        selected_slot[0] = 0
        with torch.cuda.stream(manager_stream):
            torch.cuda._sleep(50_000_000)
        stage_once()
        host_table[0, 0, 0, :5] = torch.tensor([23, 24, 25, 26, 27], dtype=torch.int32)
        selected_slot[0] = 1
        current_stream.synchronize()

        assert staging._block_offsets_device[0, 0, 0, :5].tolist() == [36, 38, 40, 42, 44]
        assert staging._block_offsets_device[0, 0, 1, :5].tolist() == [37, 39, 41, 43, 45]

        # Round 3: a delayed consumer read (snapshot) queued before the
        # round's completion ordering must complete before the next bulk
        # copy overwrites the device tables.
        selected_slot[0] = 0
        manager_stream.synchronize()
        snapshot = torch.empty_like(staging._block_offsets_device)
        torch.cuda._sleep(20_000_000)
        snapshot.copy_(staging._block_offsets_device)
        # The round executor's completion ordering: one event records the
        # consumers and the manager stream waits on it.
        staging._compaction_done_event.record(torch.cuda.current_stream(device))
        manager_stream.wait_event(staging._compaction_done_event)

        stage_once()
        current_stream.synchronize()

        assert snapshot[0, 0, 0, :5].tolist() == [36, 38, 40, 42, 44]
        assert staging._block_offsets_device[0, 0, 0, :5].tolist() == [46, 48, 50, 52, 54]

    def test_compaction_move_offsets_stage_keep_plus_tail_and_pad_rows(self):
        # The derived move offsets stage keep + tail moves per request
        # (keep_count=4 -> [6, 7]); padded rows past the cohort repeat the
        # final offset and contribute no moves. (The executor-call contract
        # itself is pinned by the overlap-tail and draft publication tests.)
        offsets_manager = TriAttention.__new__(TriAttention)
        offsets_manager._request_capacity = 8
        offsets_manager._keep_count = 4
        offsets_manager._swa_window = None
        offsets_manager._draft_protected_tail_capacity = None
        eviction_inputs = [
            _make_eviction_input(request_id=7, source_length=8, target_tail_length=2),
            _make_eviction_input(request_id=8, source_length=10, target_tail_length=3),
        ]
        dense, swa, draft = offsets_manager._compute_compaction_move_offsets(eviction_inputs)
        assert dense == [0, 6, 13, 13, 13, 13, 13, 13, 13]
        assert swa is None
        assert draft is None

    @requires_sm100
    def test_fused_score_spans_distinct_storages_and_block_tables(self):
        """ONE launch over layers in DISTINCT storages with DISTINCT block
        tables (the production V2 shape), checked against the Torch oracle,
        then relaunched after a round-start advance and a table rebind.
        (Single-request launches are pinned by the CuTe score oracle's
        request-count loop; the distinct-storages property is layer-axis.)"""
        pytest.importorskip("cutlass")
        from tensorrt_llm._torch.kv_cache_compression.triattention import triattention as module

        request_count = 8
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
        logical_source_lengths = round_device[:request_count].tolist()
        seq_lens = [seq_len - request % 2 for request in range(request_count)]
        layer_order = list(range(num_layers))
        tri = _make_cute_buffers(
            eviction_mode="per_head",
            layer_pools=pools,
            max_requests=max_requests,
            seq_len=seq_len,
            num_q_heads=num_q_heads,
            q_real=q_real,
            q_imag=q_imag,
            mlr_coef=mlr,
            freq_scale_sq=freq,
            omega=omega,
            offsets=offsets,
            decode_width=seq_len - prompt_len,
            keep_count=4,
            # One page-table slot per layer (distinct pools).
            layer_pool_ids=list(layer_order),
            normalize_scores=False,
        )
        source_lengths = torch.tensor(seq_lens, dtype=torch.int32, device=device)

        # Rounds stage through the production executor: the gather double
        # writes each layer's K page ids and the bulk copy encodes K/V rows.
        def gather_k_block_offsets(host_table, source, request_ids, num_blocks):
            assert request_ids == list(range(request_count))
            source[..., 0, :].zero_()
            source[:, :request_count, 0, :page_count].copy_(
                page_ids_3d[:, :request_count].to(torch.int32).cpu()
            )

        manager = _make_staging_manager(
            torch.zeros(num_layers, max_requests, 2, 8, dtype=torch.int32),
            gather_k_block_offsets,
            torch.cuda.Stream(device=device),
            num_slots=num_layers,
        )
        tri.kv_cache_manager = manager

        def prepared_cohort():
            return [
                _make_eviction_input(
                    request_id=request,
                    source_length=int(source_lengths[request]),
                    logical_source_length=int(round_device[request]),
                    prompt_length=prompt_len,
                )
                for request in range(request_count)
            ]

        def score_rectangle():
            # Test-side extraction of the decode-window rectangle from the
            # scratch (production reduces the scratch in-kernel).
            group = tri._num_q_heads // tri._num_kv_heads
            segments = request_count * tri._num_layers
            source = (
                tri._score_scratch[: tri._num_kv_heads * 8 * segments * tri._score_token_capacity]
                .view(
                    tri._num_kv_heads,
                    8,
                    request_count,
                    tri._num_layers,
                    tri._score_token_capacity,
                )[:, :group]
                .permute(2, 3, 0, 1, 4)
                .reshape(
                    request_count,
                    tri._num_layers,
                    tri._num_q_heads,
                    tri._score_token_capacity,
                )
            )
            columns = prompt_len + torch.arange(
                tri._selection_width_capacity, dtype=torch.int64, device=device
            ).view(1, 1, 1, -1)
            columns = columns.clamp_(max=tri._score_token_capacity - 1).expand(
                request_count,
                tri._num_layers,
                tri._num_q_heads,
                tri._selection_width_capacity,
            )
            return torch.gather(source, 3, columns)

        score_sentinel = -12345.0
        tri._score_scratch.fill_(score_sentinel)
        # The compact stage is stubbed to a no-op: this test owns the score
        # buffers only, never a staged move decision.
        with mock.patch.object(module, "compact"):
            tri._execute_eviction_round(prepared_cohort())
        fixed = score_rectangle()
        assert tri._decode_lengths_device.tolist() == [seq_len - prompt_len for seq_len in seq_lens]

        oracle = _torch_tri_score_oracle(
            pools,
            {layer: page_ids_3d[layer, :request_count] for layer in layer_order},
            seq_lens,
            logical_source_lengths,
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
                decode_length = seq_lens[request] - prompt_len
                segment = fixed[request, layer_slot, :, :decode_length]
                expected = oracle[request * num_layers + layer][
                    :, prompt_len : prompt_len + decode_length
                ]
                torch.testing.assert_close(segment, expected, rtol=5e-3, atol=5e-3)

        round_device.add_(17)
        page_ids_3d = page_ids_3d.roll(1, dims=2)
        source_lengths.copy_(
            torch.tensor(
                [seq_len - (request + 1) % 2 for request in range(request_count)],
                dtype=torch.int32,
                device=device,
            )
        )
        expected_second_widths = source_lengths - prompt_len
        tri._score_scratch.fill_(score_sentinel)
        tri._decode_lengths_device.fill_(-1)
        with mock.patch.object(module, "compact"):
            tri._execute_eviction_round(prepared_cohort())
        second_launch = score_rectangle()
        assert torch.equal(tri._decode_lengths_device, expected_second_widths)
        assert not torch.equal(second_launch, fixed)


class TestKernelMaskedSwa:
    @pytest.mark.parametrize("budget,fits_window", [(128, True), (127, False)])
    def test_layer_partition_uses_local_config_and_validates_window(self, budget, fits_window):
        mgr = _make_triattention()
        mgr.model_path = "/models/gpt-oss"
        mgr.budget = budget
        mgr._global_layers = [0, 1, 2, 3]
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
                with pytest.raises(ValueError, match="budget=127"):
                    mgr._attention_layer_partition()
                return
            dense, sliding, window = mgr._attention_layer_partition()

        load.assert_called_once_with(
            "/models/gpt-oss", trust_remote_code=True, local_files_only=True
        )
        assert dense == [1, 3]
        assert sliding == [0, 2]
        assert window == 128
