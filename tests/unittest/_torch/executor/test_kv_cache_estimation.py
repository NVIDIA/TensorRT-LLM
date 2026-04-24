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

    # local_non_kv = model 10 + activation 2 + GMS weights 6
    #              + MoE workspace 4 + non-GMS/native extra 10 = 32 GiB.
    assert available == int((100 - 32) * gib * 0.85)


@pytest.mark.parametrize(
    ("gms_mode", "engine_id", "expected"),
    [("ro", "0", True), ("rw", "1", False), ("", "0", False), ("", "1", True)],
)
def test_gms_shadow_calibration_only_applies_to_ro_or_non_primary_engine(
    monkeypatch, gms_mode, engine_id, expected
):
    creator = object.__new__(KvCacheCreator)
    creator._llm_args = Mock(load_format=LoadFormat.GMS, gms_mode=gms_mode)

    monkeypatch.setenv("ENGINE_ID", engine_id)
    monkeypatch.delenv("TRTLLM_GMS_SHADOW_MEMORY_CALIBRATION", raising=False)

    assert creator._use_gms_shadow_kv_calibration() is expected


def test_gms_shadow_calibration_can_be_disabled(monkeypatch):
    creator = object.__new__(KvCacheCreator)
    creator._llm_args = Mock(load_format=LoadFormat.GMS, gms_mode="ro")

    monkeypatch.setenv("TRTLLM_GMS_SHADOW_MEMORY_CALIBRATION", "0")

    assert creator._use_gms_shadow_kv_calibration() is False
