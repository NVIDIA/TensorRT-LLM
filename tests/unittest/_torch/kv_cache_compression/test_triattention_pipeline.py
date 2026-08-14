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
from conftest import make_test_pretrained_config as _make_test_pretrained_config
from conftest import make_tri_config as _make_tri_config
from conftest import make_triattention as _make_triattention
from conftest import mocked_eviction_internals as _mocked_eviction_internals

# TriAttention lives in the kv_cache_compression package. It exposes only the
# compression manager -- no attention classes or KV-cache-manager subclass.
from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
    TriAttentionCompressionManager,
)

# Framework base class lives in pyexecutor.resource_manager; the factory lives
# in pyexecutor._util (next to _create_kv_cache_manager), matching #15106.
from tensorrt_llm._torch.pyexecutor._util import (
    create_kv_cache_compression_manager,
    validate_kv_cache_compression_compatibility,
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


def _make_hf_config(**values):
    """Expose the normalized Hugging Face text-config contract."""
    text_config = SimpleNamespace(to_dict=lambda: dict(values))
    return SimpleNamespace(get_text_config=lambda: text_config)


class TestConfigAndFactory:
    def test_factory_allows_block_reuse_and_propagates_config_fields(self):
        # The factory contract is independent of GPU-owned persistent buffers.
        fake_v2 = _make_fake_v2(enable_block_reuse=True)
        cfg = _make_tri_config(budget=32, beta=16, eviction_mode="per_head")
        validate_kv_cache_compression_compatibility(
            cfg,
            SimpleNamespace(enable_block_reuse=True),
            None,
        )
        with (
            mock.patch(
                "tensorrt_llm._torch.pyexecutor._util.is_sm_100f",
                return_value=True,
            ),
            mock.patch.object(
                TriAttentionCompressionManager, "_initialize_eviction_state"
            ) as initialize,
        ):
            mgr = create_kv_cache_compression_manager(
                cfg,
                kv_cache_manager=fake_v2,
                pretrained_config=_make_test_pretrained_config(),
            )
        assert isinstance(mgr, TriAttentionCompressionManager)
        assert mgr.budget == 32
        assert mgr.beta == 16
        assert mgr.eviction_mode == "per_head"
        assert mgr.kv_cache_manager is fake_v2
        assert fake_v2.kv_compression_manages_history
        assert cfg.changes_physical_kv_length
        assert cfg.supports_block_reuse()
        assert not cfg.supports_speculative_decoding()
        initialize.assert_called_once_with()


class TestTriAttentionCompressionManager:
    def test_loads_flat_pt(self, flat_calibration_pt):
        mgr = _make_triattention()
        mgr.calibration_path = flat_calibration_pt
        mgr.pretrained_config = None
        mgr._load_calibration()

        assert torch.equal(mgr._omega, torch.arange(4, dtype=torch.float32))
        assert torch.equal(mgr._freq_scale_sq, torch.ones(4))
        assert torch.equal(mgr._calibration_q_real, torch.zeros(2, 2, 4))
        assert torch.equal(mgr._calibration_q_imag, torch.zeros(2, 2, 4))
        assert torch.equal(mgr._calibration_mlr_coef, torch.ones(2, 2, 4))

    def test_loads_official_layout(self, tmp_path):
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
            mgr._load_calibration()

        assert mgr._calibration_q_real.shape == (num_layers, num_heads, freq_count)
        torch.testing.assert_close(
            mgr._calibration_q_real[1, 0].cpu(), torch.full((freq_count,), 10.0)
        )
        torch.testing.assert_close(
            mgr._calibration_q_imag[1, 0].cpu(), torch.full((freq_count,), 1.0)
        )
        torch.testing.assert_close(
            mgr._calibration_mlr_coef[1, 1].cpu(), torch.full((freq_count,), -8.0)
        )
        assert mgr._omega.numel() == freq_count
        idx = torch.arange(0, 2 * freq_count, 2, dtype=torch.float32)
        torch.testing.assert_close(mgr._omega.cpu(), 1.0 / (10000.0 ** (idx / (2 * freq_count))))
        assert torch.equal(mgr._freq_scale_sq.cpu(), torch.ones(freq_count))

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
        calibration_path = tmp_path / "official.pt"
        torch.save(
            {
                "metadata": {"sampled_heads": [(0, 0)]},
                "stats": {
                    "layer00_head00": {
                        "q_mean_real": torch.zeros(freq_count),
                        "q_mean_imag": torch.zeros(freq_count),
                        "q_abs_mean": torch.ones(freq_count),
                    }
                },
            },
            calibration_path,
        )
        mgr.calibration_path = str(calibration_path)

        from transformers import AutoConfig

        mgr.pretrained_config = AutoConfig.from_pretrained(plain)
        mgr._load_calibration()
        omega = mgr._omega
        freq_scale_sq = mgr._freq_scale_sq
        idx = torch.arange(0, 64, 2, dtype=torch.float32)
        torch.testing.assert_close(omega, (1.0 / (1000000.0 ** (idx / 64)))[:freq_count])
        assert torch.equal(freq_scale_sq, torch.ones(freq_count))

        # The executor's config loader clears rope_parameters for default
        # (unscaled) RoPE and keeps the canonical value on config.rope_theta.
        engine_cfg = AutoConfig.from_pretrained(plain)
        engine_cfg.rope_parameters = None
        engine_cfg.rope_theta = 1000000.0
        mgr.pretrained_config = engine_cfg
        mgr._load_calibration()
        torch.testing.assert_close(mgr._omega, omega)
        assert torch.equal(mgr._freq_scale_sq, torch.ones(freq_count))

        mgr.pretrained_config = AutoConfig.from_pretrained(yarn)
        mgr._load_calibration()
        omega_yarn = mgr._omega
        freq_scale_sq_yarn = mgr._freq_scale_sq
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
        manager = _make_triattention(budget=4, beta=4)
        manager.kv_cache_manager._stream = mock.Mock()
        request = _make_request(7, py_prompt_len=2)
        # seq_len == prompt + budget: the due filter must drop the request.
        cache = SimpleNamespace(
            capacity=6, history_length=0, is_active=True, resize=mock.Mock(return_value=True)
        )
        manager.kv_cache_manager.kv_cache_map = {7: cache}

        with _mocked_eviction_internals(manager) as internals:
            manager._evict_due_requests(SimpleNamespace(generation_requests=[request]))

        internals.execute.assert_not_called()
        assert request.py_num_compressed_tokens == 0
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
        with mock.patch.object(TriAttentionCompressionManager, "_initialize_eviction_state"):
            mgr = TriAttentionCompressionManager(
                _make_tri_config(budget=8),
                fake_v2,
                pretrained_config=_make_test_pretrained_config(),
            )
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
        mgr.beta = 128
        mgr.budget = 4096
        mgr._selection_width_capacity = mgr.budget + mgr.beta + 1
        return mgr, request, batch

    def test_suspended_cache_defers_that_request_pre_launch(self):
        # A suspended cache is a legal overlap-scheduler transient: that
        # request defers while the rest of the request group proceeds, then
        # catches up to the missed cadence boundary when it resumes.
        manager, first_request, _ = self._make_due_decode_request(seq_len=1024 + 4096 + 128)
        second_request = _make_request(8, py_prompt_len=1024)
        second_cache = SimpleNamespace(
            capacity=1024 + 4096 + 128,
            history_length=1024,
            is_active=False,
            resize=mock.Mock(return_value=True),
        )
        manager.kv_cache_manager.kv_cache_map[8] = second_cache
        batch = SimpleNamespace(generation_requests=[first_request, second_request])

        with _mocked_eviction_internals(manager) as internals:
            manager._evict_due_requests(batch)
            # Only the active request launched in the first round.
            eviction_requests = internals.execute.call_args.args[0]
            assert [item.request.py_request_id for item in eviction_requests] == [7]
            assert second_request.py_num_compressed_tokens == 0

            second_cache.is_active = True
            # Resumption executes one more token before the next final update.
            second_cache.capacity += 1
            manager._evict_due_requests(SimpleNamespace(generation_requests=[second_request]))

        resumed = internals.execute.call_args.args[0]
        assert [item.request.py_request_id for item in resumed] == [8]
        assert second_request.py_num_compressed_tokens == 129
        second_cache.resize.assert_called_once_with(1024 + 4096, None)

    def test_deferred_eviction_checks_the_compiled_selection_width(self):
        manager, request, batch = self._make_due_decode_request(seq_len=1024 + 4096 + 128 + 2)

        with _mocked_eviction_internals(manager) as internals:
            with pytest.raises(RuntimeError, match="selection width"):
                manager._evict_due_requests(batch)

        internals.execute.assert_not_called()

    # Accepted draft tokens may cross the same cadence boundary; they do not
    # change the fixed overlap reservation.
    @pytest.mark.parametrize("accepted", [0, 3])
    def test_overlap_tail_is_excluded_from_selection_and_compacted(self, accepted):
        confirmed = 1024 + 4096 + 128
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
        mgr.on_generation_step_begin(SimpleNamespace(generation_requests=[request]))
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
        assert launched.target_tail_length == tail
        assert request.py_num_compressed_tokens == confirmed - retained
        cache.resize.assert_called_once_with(retained + tail, None)
        draft_cache.resize.assert_called_once_with(retained + 1, None)

    def test_confirmed_length_comes_from_capacity_ledger_not_logical_length(self):
        # The due-branch source length must come from the physical capacity
        # ledger (capacity minus the protected tail), never the logical length.
        manager = _make_triattention(beta=128)
        compressed_tokens = manager.beta - manager.budget
        # Reachable second cadence boundary, within the compiled selection span.
        physical_confirmed = 1024 + manager.budget + manager.beta
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
            py_num_compressed_tokens=compressed_tokens,
            max_beam_num_tokens=physical_confirmed + compressed_tokens + 1,
            py_draft_tokens=[1, 2, 3, 4],
        )

        with _mocked_eviction_internals(manager) as internals:
            manager._evict_due_requests(SimpleNamespace(generation_requests=[request]))

        eviction_requests = internals.execute.call_args.args[0]
        assert eviction_requests[0].source_length == physical_confirmed
        assert request.py_num_compressed_tokens == (
            compressed_tokens + physical_confirmed - 1024 - manager.budget
        )
        cache.resize.assert_called_once_with(1024 + manager.budget, None)

    @pytest.mark.parametrize("spec_mode", ["mtp", "eagle3"])
    def test_one_model_draft_co_compression_is_accepted(self, spec_mode):
        draft_manager = _make_fake_v2(is_draft=True)
        with mock.patch.object(TriAttentionCompressionManager, "_initialize_eviction_state"):
            TriAttentionCompressionManager(
                _make_tri_config(budget=8),
                _make_fake_v2(),
                draft_kv_cache_manager=draft_manager,
                pretrained_config=_make_test_pretrained_config(),
            )

        from tensorrt_llm._torch.pyexecutor._util import validate_kv_cache_compression_compatibility
        from tensorrt_llm.llmapi.llm_args import Eagle3DecodingConfig, MTPDecodingConfig

        spec_config = (
            MTPDecodingConfig(max_draft_len=1)
            if spec_mode == "mtp"
            else Eagle3DecodingConfig(
                max_draft_len=1,
                speculative_model="draft",
                eagle3_one_model=True,
            )
        )

        validate_kv_cache_compression_compatibility(
            _make_tri_config(budget=8),
            SimpleNamespace(enable_block_reuse=False),
            spec_config,
        )


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
                staging._stage_block_offset_snapshot(
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

    def test_device_page_table_snapshot_delegates_to_manager(self):
        device = torch.device("cuda", torch.cuda.current_device())
        current_stream = torch.cuda.current_stream(device)
        host_table = torch.empty(
            1,
            1,
            2,
            8,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        gather = mock.Mock()
        manager = _make_staging_manager(host_table, gather, current_stream)
        manager.uses_device_page_table = True
        manager.materialize_block_offsets_snapshot = mock.Mock()
        staging = _make_bare_staging(device, max_requests=1, staged_blocks_per_seq=8)

        staging._stage_block_offset_snapshot(
            manager,
            [7],
            staging._block_offsets_host,
            staging._block_offsets_device,
        )

        manager.materialize_block_offsets_snapshot.assert_called_once_with(
            staging._block_offsets_device,
            [7],
            host_staging=staging._block_offsets_host,
            stream=current_stream,
        )
        gather.assert_not_called()


class TestKernelMaskedSwa:
    @pytest.mark.parametrize("budget,fits_window", [(128, True), (127, False)])
    def test_attention_layers_use_local_config_and_validate_window(self, budget, fits_window):
        mgr = _make_triattention()
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

        mgr.pretrained_config = config

        if not fits_window:
            # The decode budget must cover the kernel-masked SWA window.
            with pytest.raises(ValueError, match="budget=127"):
                mgr._resolve_attention_layers()
            return
        dense, sliding, window = mgr._resolve_attention_layers()

        assert dense == [1, 3]
        assert sliding == [0, 2]
        assert window == 128
