"""Tests for KV cache token estimation in KvCacheCreator._get_token_num_for_estimation.

Guards the ADP (Attention Data Parallelism) cache-block reduction: when
enable_attention_dp is True and tp_size > 1, _create_dummy_context_requests
produces tp_size duplicate requests, but the scheduler distributes them
1-per-rank.  Each rank's KV cache therefore only needs capacity for its own
share, not all copies.
"""

from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.py_executor_creator import (
    _should_defer_gms_shadow_cuda_graph_capture,
    _should_defer_gms_shadow_startup_warmup,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import (KVCacheManager,
                                                              ResourceManagerType)
from tensorrt_llm.llmapi.llm_args import LoadFormat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_request(num_input_tokens, beam_width=1):
    """Create a mock request with the fields _get_token_num_for_estimation reads."""
    req = Mock()
    req.input_token_ids = list(range(num_input_tokens))
    req.sampling_config.beam_width = beam_width
    return req


def _make_creator(
    tokens_per_block,
    dummy_reqs,
    enable_attention_dp,
    tp_size,
    batch_size=1,
    model_max_seq_len=1,
    max_cuda_graph_batch_size=1,
):
    """Build a minimal KvCacheCreator (bypasses __init__) wired up for
    _get_token_num_for_estimation only."""
    c = object.__new__(KvCacheCreator)

    c._tokens_per_block = tokens_per_block
    c._net_max_seq_len = 2048
    c._speculative_config = None
    c._dummy_reqs = dummy_reqs

    c._mapping = Mock(enable_attention_dp=enable_attention_dp, tp_size=tp_size, cp_config={})

    c._llm_args = Mock(disable_overlap_scheduler=True)

    c._model_engine = Mock(
        batch_size=batch_size,
        max_seq_len=model_max_seq_len,
        _max_cuda_graph_batch_size=max_cuda_graph_batch_size,
    )

    c._kv_cache_config = Mock(free_gpu_memory_fraction=0.9)

    return c


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_gpu():
    """Stub out CUDA memory queries and per-token KV size so the test runs on
    any machine and the memory cap never constrains the result."""
    huge = 100 * (1 << 30)
    with (
        patch("torch.cuda.mem_get_info", return_value=(huge, huge)),
        patch.object(KvCacheCreator, "_get_kv_size_per_token", return_value=1),
    ):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_adp_reduces_blocks_to_per_rank_share():
    """With ADP + tp_size duplicated requests the result must equal a single
    rank's share, not the sum across all duplicates."""
    tpb = 64
    tp = 4
    n_in = 128  # ceil((128+1)/64) = 3 blocks per request

    baseline = _make_creator(tpb, [_make_mock_request(n_in)], enable_attention_dp=False, tp_size=1)
    adp = _make_creator(
        tpb, [_make_mock_request(n_in) for _ in range(tp)], enable_attention_dp=True, tp_size=tp
    )

    assert adp._get_token_num_for_estimation() == baseline._get_token_num_for_estimation()


def test_without_adp_all_blocks_counted():
    """Without ADP every request's blocks contribute to the total."""
    tpb = 64
    n_in = 128  # 3 blocks each
    n_reqs = 4

    c = _make_creator(
        tpb, [_make_mock_request(n_in) for _ in range(n_reqs)], enable_attention_dp=False, tp_size=1
    )

    # 4 reqs * 3 blocks * 64 tokens/block = 768
    assert c._get_token_num_for_estimation() == n_reqs * 3 * tpb


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_adp_various_tp_sizes(tp_size):
    """ADP division must hold for several representative tp_size values."""
    tpb = 64
    n_in = 128  # 3 blocks per request

    c = _make_creator(
        tpb,
        [_make_mock_request(n_in) for _ in range(tp_size)],
        enable_attention_dp=True,
        tp_size=tp_size,
    )

    total = tp_size * 3
    expected_blocks = (total + tp_size - 1) // tp_size
    assert c._get_token_num_for_estimation() == expected_blocks * tpb


def test_regression_without_fix_would_overcount():
    """If the ADP ceil-division fix were removed, the returned
    value would be tp_size times too large.  This test guards that fix."""
    tpb = 64
    tp = 4
    n_in = 128

    c = _make_creator(
        tpb, [_make_mock_request(n_in) for _ in range(tp)], enable_attention_dp=True, tp_size=tp
    )

    result = c._get_token_num_for_estimation()

    correct = 3 * tpb  # 192  (per-rank share)
    wrong = tp * 3 * tpb  # 768  (all duplicates summed)
    assert result == correct
    assert result != wrong


def test_gms_shadow_calibration_accounts_resident_weights_and_local_overhead():
    gib = 1 << 30
    creator = object.__new__(KvCacheCreator)
    creator._gms_weight_bytes = Mock(return_value=6 * gib)
    creator._moe_workspace_bytes = Mock(return_value=4 * gib)

    available = creator._cal_gms_shadow_max_memory(
        torch_peak_memory=14 * gib,
        model_bytes=10 * gib,
        total_gpu_memory=100 * gib,
        fraction=0.85,
        temporary_kv_bytes=2 * gib,
        non_torch_extra_bytes=20 * gib,
    )

    # non_torch_extra is process-local; RO GMS weight mappings are not part of
    # that NVML number.  Only subtract MoE workspace to avoid double counting it.
    # local_non_kv = model 10 + activation 4 + GMS weights 6
    #              + MoE workspace 4 + other native extra 16 = 40 GiB.
    assert available == int((100 - 40) * gib * 0.85)


def test_gms_shadow_calibration_reserves_warmup_activation_with_vmm_kv():
    gib = 1 << 30
    creator = object.__new__(KvCacheCreator)
    creator._gms_weight_bytes = Mock(return_value=74 * gib)
    creator._moe_workspace_bytes = Mock(return_value=0)

    available = creator._cal_gms_shadow_max_memory(
        torch_peak_memory=3 * gib,
        model_bytes=1 * gib,
        total_gpu_memory=180 * gib,
        fraction=0.85,
        temporary_kv_bytes=9 * gib,
        non_torch_extra_bytes=13 * gib,
    )

    # VMM KV allocation is not reflected in torch's peak allocator stats, so
    # the 2 GiB torch activation peak must still be reserved for full-size KV.
    assert available == int((180 - 90) * gib * 0.85)


def test_gms_shadow_calibration_counts_local_extra_smaller_than_gms_weights():
    gib = 1 << 30
    creator = object.__new__(KvCacheCreator)
    creator._gms_weight_bytes = Mock(return_value=74 * gib)
    creator._moe_workspace_bytes = Mock(return_value=0)

    available = creator._cal_gms_shadow_max_memory(
        torch_peak_memory=1 * gib,
        model_bytes=1 * gib,
        total_gpu_memory=180 * gib,
        fraction=0.85,
        temporary_kv_bytes=0,
        non_torch_extra_bytes=13 * gib,
    )

    assert available == int((180 - 88) * gib * 0.85)


def test_gms_shadow_final_calibration_uses_process_local_non_torch_memory():
    gib = 1 << 30
    creator = object.__new__(KvCacheCreator)
    creator._mapping = Mock(cp_config={})
    creator._kv_cache_config = Mock(free_gpu_memory_fraction=0.85,
                                    max_gpu_total_bytes=0)
    creator._skip_est = False
    creator._profiling_stage_data = None
    creator._kv_cache_manager_cls = KVCacheManager
    creator._max_kv_tokens_in = None
    creator._max_gpu_total_bytes_in = 0
    creator._gms_shadow_torch_activation_reserve_bytes = 0
    creator._use_gms_shadow_kv_calibration = Mock(return_value=True)
    creator._cal_gms_shadow_max_memory = Mock(return_value=64 * gib)
    creator._get_kv_size_per_token = Mock(return_value=gib)
    creator._current_process_gpu_memory = Mock(return_value=25 * gib)

    with (
            patch("torch.cuda.empty_cache"),
            patch("torch.cuda.reset_peak_memory_stats"),
            patch("torch.cuda.mem_get_info", return_value=(80 * gib, 100 * gib)),
            patch("torch.cuda.memory_stats",
                  return_value={"allocated_bytes.all.current": 8 * gib}),
    ):
        creator.configure_kv_cache_capacity(py_executor=None)

    creator._cal_gms_shadow_max_memory.assert_called_once_with(
        torch_peak_memory=8 * gib,
        model_bytes=8 * gib,
        total_gpu_memory=100 * gib,
        fraction=0.85,
        temporary_kv_bytes=0,
        non_torch_extra_bytes=17 * gib,
        min_torch_activation_bytes=0,
    )
    assert creator._kv_cache_config.max_gpu_total_bytes == 64 * gib
    assert creator._kv_cache_config.max_tokens == 64


def test_gms_shadow_final_calibration_reuses_warmup_activation_reserve():
    gib = 1 << 30
    creator = object.__new__(KvCacheCreator)
    creator._mapping = Mock(cp_config={})
    creator._kv_cache_config = Mock(free_gpu_memory_fraction=0.85,
                                    max_gpu_total_bytes=0)
    creator._skip_est = False
    creator._profiling_stage_data = None
    creator._kv_cache_manager_cls = KVCacheManager
    creator._max_kv_tokens_in = None
    creator._max_gpu_total_bytes_in = 0
    creator._gms_shadow_torch_activation_reserve_bytes = 3 * gib
    creator._use_gms_shadow_kv_calibration = Mock(return_value=True)
    creator._cal_gms_shadow_max_memory = Mock(return_value=64 * gib)
    creator._get_kv_size_per_token = Mock(return_value=gib)
    creator._current_process_gpu_memory = Mock(return_value=25 * gib)

    with (
            patch("torch.cuda.empty_cache"),
            patch("torch.cuda.reset_peak_memory_stats"),
            patch("torch.cuda.mem_get_info", return_value=(80 * gib, 100 * gib)),
            patch("torch.cuda.memory_stats",
                  return_value={"allocated_bytes.all.current": 8 * gib}),
    ):
        creator.configure_kv_cache_capacity(py_executor=None)

    creator._cal_gms_shadow_max_memory.assert_called_once_with(
        torch_peak_memory=8 * gib,
        model_bytes=8 * gib,
        total_gpu_memory=100 * gib,
        fraction=0.85,
        temporary_kv_bytes=0,
        non_torch_extra_bytes=17 * gib,
        min_torch_activation_bytes=3 * gib,
    )
    assert creator._kv_cache_config.max_gpu_total_bytes == 64 * gib
    assert creator._kv_cache_config.max_tokens == 64


@pytest.mark.parametrize(
    ("gms_mode", "engine_id", "expected"),
    [("ro", "0", True), ("rw", "1", False), ("", "0", False), ("", "1", True)],
)
@pytest.mark.parametrize("load_format",
                         [LoadFormat.GMS, "GMS", "gms", "LoadFormat.GMS", 3])
def test_gms_shadow_calibration_only_applies_to_ro_or_non_primary_engine(
    monkeypatch, load_format, gms_mode, engine_id, expected
):
    creator = object.__new__(KvCacheCreator)
    creator._llm_args = Mock(load_format=load_format, gms_mode=gms_mode)

    monkeypatch.setenv("ENGINE_ID", engine_id)
    monkeypatch.delenv("TRTLLM_GMS_SHADOW_MEMORY_CALIBRATION", raising=False)

    assert creator._use_gms_shadow_kv_calibration() is expected


def test_gms_shadow_calibration_can_be_disabled(monkeypatch):
    creator = object.__new__(KvCacheCreator)
    creator._llm_args = Mock(load_format=LoadFormat.GMS, gms_mode="ro")

    monkeypatch.setenv("TRTLLM_GMS_SHADOW_MEMORY_CALIBRATION", "0")

    assert creator._use_gms_shadow_kv_calibration() is False


def test_gms_shadow_calibration_uses_mpi_worker_environment(monkeypatch):
    creator = object.__new__(KvCacheCreator)
    creator._llm_args = Mock(load_format="auto", gms_mode="auto")

    monkeypatch.setenv("ENGINE_ID", "1")
    monkeypatch.setenv("GMS_SOCKET_DIR", "/tmp/gms")
    monkeypatch.delenv("TRTLLM_GMS_SHADOW_MEMORY_CALIBRATION", raising=False)

    assert creator._use_gms_shadow_kv_calibration() is True


def test_gms_shadow_calibration_uses_llm_env_overrides(monkeypatch):
    creator = object.__new__(KvCacheCreator)
    creator._llm_args = Mock(load_format="auto",
                             gms_mode="auto",
                             env_overrides={
                                 "ENGINE_ID": "1",
                                 "GMS_SOCKET_DIR": "/tmp/gms",
                             })

    monkeypatch.delenv("ENGINE_ID", raising=False)
    monkeypatch.delenv("GMS_SOCKET_DIR", raising=False)
    monkeypatch.delenv("TRTLLM_GMS_SHADOW_MEMORY_CALIBRATION", raising=False)

    assert creator._use_gms_shadow_kv_calibration() is True


def test_gms_shadow_calibration_env_overrides_do_not_override_rw(monkeypatch):
    creator = object.__new__(KvCacheCreator)
    creator._llm_args = Mock(load_format=LoadFormat.GMS,
                             gms_mode="rw",
                             env_overrides={
                                 "ENGINE_ID": "1",
                                 "GMS_SOCKET_DIR": "/tmp/gms",
                             })

    monkeypatch.delenv("ENGINE_ID", raising=False)
    monkeypatch.delenv("GMS_SOCKET_DIR", raising=False)
    monkeypatch.delenv("TRTLLM_GMS_SHADOW_MEMORY_CALIBRATION", raising=False)

    assert creator._use_gms_shadow_kv_calibration() is False


def test_gms_shadow_calibration_worker_environment_does_not_override_rw(
        monkeypatch):
    creator = object.__new__(KvCacheCreator)
    creator._llm_args = Mock(load_format=LoadFormat.GMS, gms_mode="rw")

    monkeypatch.setenv("ENGINE_ID", "1")
    monkeypatch.setenv("GMS_SOCKET_DIR", "/tmp/gms")
    monkeypatch.delenv("TRTLLM_GMS_SHADOW_MEMORY_CALIBRATION", raising=False)

    assert creator._use_gms_shadow_kv_calibration() is False


def test_legacy_kv_manager_trusts_calibrated_max_gpu_total_bytes(monkeypatch):
    manager = object.__new__(KVCacheManager)
    monkeypatch.setattr(KVCacheManager, "get_cache_bytes_per_token",
                        lambda _: 1024)
    kv_cache_config = Mock(
        free_gpu_memory_fraction=0.85,
        host_cache_size=0,
        max_gpu_total_bytes=80 * 1024,
        max_tokens=None,
    )
    mapping = Mock(world_size=1)

    with patch("torch.cuda.mem_get_info",
               return_value=(10 * 1024, 100 * 1024)) as mem_get_info:
        blocks, secondary_blocks = manager.calculate_max_num_blocks(
            kv_cache_config,
            head_dim=0,
            tokens_per_block=16,
            mapping=mapping,
            dtype=None,
        )

    mem_get_info.assert_not_called()
    assert blocks == 5
    assert secondary_blocks == 0


def test_gms_shadow_calibration_runs_when_estimation_is_disabled():
    creator = object.__new__(KvCacheCreator)
    creator._skip_est = False
    creator._max_gpu_total_bytes_in = 0
    creator._kv_cache_config = Mock(max_gpu_total_bytes=0)
    creator._llm_args = Mock(load_format=LoadFormat.GMS,
                             gms_mode="ro",
                             env_overrides={"ENGINE_ID": "1"})
    creator._use_gms_shadow_kv_calibration = Mock(return_value=True)
    creator.configure_kv_cache_capacity = Mock()
    creator._should_create_separate_draft_kv_cache = Mock(return_value=False)
    creator._create_kv_cache_manager = Mock(return_value="kv")
    creator._model_engine = Mock()
    creator._kv_connector_manager = None
    creator._draft_model_engine = None
    resources = {}

    creator.build_managers(resources, estimating_kv_cache=False)

    creator.configure_kv_cache_capacity.assert_called_once_with()
    assert resources[ResourceManagerType.KV_CACHE_MANAGER] == "kv"
    assert resources[ResourceManagerType.DRAFT_KV_CACHE_MANAGER] is None


def test_gms_shadow_final_calibration_discards_warmup_derived_cap():
    creator = object.__new__(KvCacheCreator)
    creator._skip_est = False
    creator._max_gpu_total_bytes_in = 0
    creator._kv_cache_config = Mock(max_gpu_total_bytes=32)
    creator._llm_args = Mock(load_format=LoadFormat.GMS,
                             gms_mode="ro",
                             env_overrides={"ENGINE_ID": "1"})
    creator._use_gms_shadow_kv_calibration = Mock(return_value=True)

    def configure_kv_cache_capacity():
        assert creator._kv_cache_config.max_gpu_total_bytes == 0
        creator._kv_cache_config.max_gpu_total_bytes = 96

    creator.configure_kv_cache_capacity = Mock(side_effect=configure_kv_cache_capacity)
    creator._should_create_separate_draft_kv_cache = Mock(return_value=False)
    creator._create_kv_cache_manager = Mock(return_value="kv")
    creator._model_engine = Mock()
    creator._kv_connector_manager = None
    creator._draft_model_engine = None
    resources = {}

    creator.build_managers(resources, estimating_kv_cache=False)

    creator.configure_kv_cache_capacity.assert_called_once_with()
    assert creator._kv_cache_config.max_gpu_total_bytes == 96
    assert resources[ResourceManagerType.KV_CACHE_MANAGER] == "kv"
    assert resources[ResourceManagerType.DRAFT_KV_CACHE_MANAGER] is None


def test_gms_shadow_deferred_cuda_graph_capture_enabled_for_ro_shadow(
        monkeypatch):
    llm_args = Mock(load_format=LoadFormat.GMS,
                    gms_mode="ro",
                    cuda_graph_config=Mock(),
                    sleep_config=Mock())

    monkeypatch.delenv("TRTLLM_GMS_DEFER_SHADOW_CUDA_GRAPH_CAPTURE",
                       raising=False)

    assert _should_defer_gms_shadow_cuda_graph_capture(
        llm_args, estimating_kv_cache=True) is True


def test_gms_shadow_deferred_cuda_graph_capture_requires_estimation():
    llm_args = Mock(load_format=LoadFormat.GMS,
                    gms_mode="ro",
                    cuda_graph_config=Mock(),
                    sleep_config=Mock())

    assert _should_defer_gms_shadow_cuda_graph_capture(
        llm_args, estimating_kv_cache=False) is False


def test_gms_shadow_deferred_cuda_graph_capture_can_be_disabled(monkeypatch):
    llm_args = Mock(load_format=LoadFormat.GMS,
                    gms_mode="ro",
                    cuda_graph_config=Mock(),
                    sleep_config=Mock())

    monkeypatch.setenv("TRTLLM_GMS_DEFER_SHADOW_CUDA_GRAPH_CAPTURE", "0")

    assert _should_defer_gms_shadow_cuda_graph_capture(
        llm_args, estimating_kv_cache=True) is False


def test_gms_shadow_deferred_cuda_graph_capture_does_not_apply_to_rw():
    llm_args = Mock(load_format=LoadFormat.GMS,
                    gms_mode="rw",
                    cuda_graph_config=Mock(),
                    sleep_config=Mock(),
                    env_overrides={
                        "ENGINE_ID": "1",
                        "GMS_SOCKET_DIR": "/tmp/gms",
                    })

    assert _should_defer_gms_shadow_cuda_graph_capture(
        llm_args, estimating_kv_cache=True) is False


def test_gms_shadow_deferred_startup_warmup_enabled_for_ro_shadow(
        monkeypatch):
    llm_args = Mock(load_format=LoadFormat.GMS,
                    gms_mode="ro",
                    cuda_graph_config=Mock(),
                    sleep_config=Mock())

    monkeypatch.delenv("TRTLLM_GMS_DEFER_SHADOW_STARTUP_WARMUP",
                       raising=False)

    assert _should_defer_gms_shadow_startup_warmup(
        llm_args, estimating_kv_cache=True) is True


def test_gms_shadow_deferred_startup_warmup_can_be_disabled(monkeypatch):
    llm_args = Mock(load_format=LoadFormat.GMS,
                    gms_mode="ro",
                    cuda_graph_config=Mock(),
                    sleep_config=Mock())

    monkeypatch.setenv("TRTLLM_GMS_DEFER_SHADOW_STARTUP_WARMUP", "0")

    assert _should_defer_gms_shadow_startup_warmup(
        llm_args, estimating_kv_cache=True) is False


def test_deferred_warmup_skips_invalid_kv_cleanup_once():
    class DummyEngine:
        kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER

    engine = DummyEngine()
    kv_cache_manager = Mock()
    kv_cache_manager.check_invalid_values_in_kv_cache.return_value = False
    resource_manager = Mock()
    resource_manager.get_resource_manager.return_value = kv_cache_manager

    @PyTorchModelEngine.warmup_with_kv_cache_cleanup
    def deferred_warmup(self, resource_manager):
        self._skip_warmup_kv_cache_cleanup_once = True

    deferred_warmup(engine, resource_manager)

    kv_cache_manager.check_invalid_values_in_kv_cache.assert_not_called()
    assert not engine._skip_warmup_kv_cache_cleanup_once

    @PyTorchModelEngine.warmup_with_kv_cache_cleanup
    def real_warmup(self, resource_manager):
        return "warmed"

    assert real_warmup(engine, resource_manager) == "warmed"
    kv_cache_manager.check_invalid_values_in_kv_cache.assert_called_once_with(
        fill_with_zero=True)
