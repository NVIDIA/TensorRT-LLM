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

Config, construction, eviction lifecycle, page-table staging, and admission-sized
score state; the manager publishes evicted counts via
``LlmRequest.py_num_compressed_tokens``. Draft contracts live in
``test_triattention_draft_cocompaction.py``.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from conftest import make_bare_staging as _make_bare_staging
from conftest import make_fake_v2 as _make_fake_v2
from conftest import make_request as _make_request
from conftest import make_staging_manager as _make_staging_manager
from conftest import make_tri_config as _make_tri_config
from conftest import make_triattention as _make_triattention
from conftest import mocked_eviction_internals as _mocked_eviction_internals

# TriAttention lives in the kv_cache_compression package. It exposes only the
# compression manager -- no attention classes or KV-cache-manager subclass.
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
    TriAttention,
    _RequestState,
)

# Framework base class lives in pyexecutor.resource_manager; the factory lives
# in pyexecutor._util (next to _create_kv_cache_manager), matching #15106.
from tensorrt_llm._torch.pyexecutor._util import create_kv_cache_compression_manager


def _set_request_state(manager, request_id, *, confirmed_tokens=0, evicted_tokens=0):
    state = _RequestState(
        confirmed_tokens=confirmed_tokens,
        evicted_tokens=evicted_tokens,
    )
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
        # The factory contract is independent of GPU-owned persistent buffers.
        fake_v2 = _make_fake_v2(enable_block_reuse=False)
        cfg = _make_tri_config(budget=32, beta=16, eviction_mode="per_head")
        with mock.patch.object(TriAttention, "_initialize_eviction_state") as initialize:
            mgr = create_kv_cache_compression_manager(cfg, kv_cache_manager=fake_v2)
        assert isinstance(mgr, TriAttention)
        assert mgr.budget == 32
        assert mgr.beta == 16
        assert mgr.eviction_mode == "per_head"
        assert mgr.kv_cache_manager is fake_v2
        initialize.assert_called_once_with()


class TestTriAttentionClass:
    def test_request_init_and_finish_lifecycle(self):
        # Init reserves request-dependent capacity and tracks state. Finish
        # clears only request state; persistent runtime objects stay resident.
        manager = _make_fake_v2()
        manager.num_extra_kv_tokens = 4
        manager._kv_reserve_draft_tokens = 4
        with mock.patch.object(TriAttention, "_initialize_eviction_state"):
            triattention = TriAttention(_make_tri_config(budget=8), manager)
        with (
            mock.patch.object(triattention, "_validate_request_capacity") as validate,
            mock.patch.object(triattention, "_reserve_eviction_capacity") as reserve,
        ):
            request_11 = _make_request(11)
            request_12 = _make_request(12)
            triattention.on_request_init(request_11)
            triattention.on_request_init(request_12)

        assert triattention.adjusts_generation_kv_length is True
        assert manager.kv_compression_manages_history
        assert set(triattention._request_states) == {11, 12}
        assert validate.call_args_list == [mock.call(request_11), mock.call(request_12)]
        assert reserve.call_args_list == [mock.call(request_11), mock.call(request_12)]

        phase = object()
        triattention._phase = phase
        batch = SimpleNamespace()
        triattention._inflight_scheduled_batch = batch
        triattention.on_request_finish(_make_request(11))
        triattention.on_request_finish(_make_request(12))
        assert triattention._request_states == {}
        assert triattention._inflight_scheduled_batch is batch
        assert triattention._phase is phase

    def test_capacity_guard_uses_maximum_steady_eviction_peak(self):
        manager = _make_triattention(budget=10, beta=8)
        manager.kv_cache_manager.get_num_available_tokens = mock.Mock(return_value=17)

        with pytest.raises(ValueError, match="requires 19 tokens"):
            manager._validate_request_capacity(
                _make_request(7, py_prompt_len=0, py_max_new_tokens=100)
            )

        manager.kv_cache_manager.get_num_available_tokens.assert_called_once_with(
            token_num_upper_bound=18,
            max_num_draft_tokens=1,
        )

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
        # Identity requests (seq_len == prompt + budget) are the pre-launch
        # owner no-op: nothing launches and nothing is published.
        manager = _make_triattention(budget=4)
        manager.kv_cache_manager._stream = mock.Mock()
        request = _make_request(7, py_prompt_len=2)
        # seq_len == prompt + budget: the due filter must drop the request.
        cache = SimpleNamespace(
            capacity=6, history_length=0, is_active=True, resize=mock.Mock(return_value=True)
        )
        manager.kv_cache_manager.kv_cache_map = {7: cache}
        state = _set_request_state(manager, 7, confirmed_tokens=127)

        with _mocked_eviction_internals(manager) as internals:
            manager._evict_due_requests(SimpleNamespace(generation_requests=[request]))

        internals.execute.assert_not_called()
        assert request.py_num_compressed_tokens == 0
        assert state.evicted_tokens == 0
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
        with mock.patch.object(TriAttention, "_initialize_eviction_state"):
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
        _set_request_state(mgr, 7, confirmed_tokens=127)
        mgr.beta = 128
        mgr.budget = 4096
        return mgr, request, batch

    def test_suspended_cache_defers_that_request_pre_launch(self):
        # A suspended cache is a legal overlap-scheduler transient: that
        # request defers (pre-launch, no cadence mutation) while the rest of
        # the request group proceeds.
        manager, first_request, _ = self._make_due_decode_request(seq_len=1024 + 4096 + 1)
        second_request = _make_request(8, py_prompt_len=1024)
        manager.kv_cache_manager.kv_cache_map[8] = SimpleNamespace(is_active=False)
        first_state = manager._request_states[7]
        second_state = _set_request_state(manager, 8, confirmed_tokens=127)
        batch = SimpleNamespace(generation_requests=[first_request, second_request])

        with _mocked_eviction_internals(manager) as internals:
            manager._evict_due_requests(batch)

        # Only the active request launched; the suspended one deferred whole.
        eviction_inputs = internals.execute.call_args.args[0]
        assert [item.request.py_request_id for item in eviction_inputs] == [7]
        assert first_state.confirmed_tokens == 128
        assert second_state.confirmed_tokens == 127

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
        _set_request_state(manager, 7, confirmed_tokens=127, evicted_tokens=100)
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
        with mock.patch.object(TriAttention, "_initialize_eviction_state"):
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
        with mock.patch.object(TriAttention, "_initialize_eviction_state"):
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

    def test_bulk_page_table_copy_snapshots_on_current_stream(self):
        """Snapshot and order bulk page-table copies on the caller's current stream."""
        device = torch.device("cuda", torch.cuda.current_device())
        current_stream = torch.cuda.current_stream(device)
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
        manager = _make_staging_manager(host_table, gather, current_stream)

        def stage_once():
            # Raises on any staging failure; success returns None.
            with torch.cuda.stream(current_stream):
                staging._stage_block_offsets(
                    manager,
                    [7],
                    staging._block_offsets_host,
                    staging._block_offsets_device,
                )

        # Round 1: mutate the host table and the slot assignment right after
        # staging. The staged result must still reflect the gathered snapshot.
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

        # Round 2: same contract on a re-staged request group.
        host_table[0, 0, 0, :5] = torch.tensor([18, 19, 20, 21, 22], dtype=torch.int32)
        selected_slot[0] = 0
        stage_once()
        host_table[0, 0, 0, :5] = torch.tensor([23, 24, 25, 26, 27], dtype=torch.int32)
        selected_slot[0] = 1
        current_stream.synchronize()

        assert staging._block_offsets_device[0, 0, 0, :5].tolist() == [36, 38, 40, 42, 44]
        assert staging._block_offsets_device[0, 0, 1, :5].tolist() == [37, 39, 41, 43, 45]

        # Round 3: a consumer queued before restaging sees the prior table.
        selected_slot[0] = 0
        snapshot = torch.empty_like(staging._block_offsets_device)
        snapshot.copy_(staging._block_offsets_device)

        stage_once()
        current_stream.synchronize()

        assert snapshot[0, 0, 0, :5].tolist() == [36, 38, 40, 42, 44]
        assert staging._block_offsets_device[0, 0, 0, :5].tolist() == [46, 48, 50, 52, 54]


class TestKernelMaskedSwa:
    @pytest.mark.parametrize("budget,fits_window", [(128, True), (127, False)])
    def test_attention_layers_use_local_config_and_validate_window(self, budget, fits_window):
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
                    mgr._resolve_attention_layers()
                return
            dense, sliding, window = mgr._resolve_attention_layers()

        load.assert_called_once_with(
            "/models/gpt-oss", trust_remote_code=True, local_files_only=True
        )
        assert dense == [1, 3]
        assert sliding == [0, 2]
        assert window == 128
