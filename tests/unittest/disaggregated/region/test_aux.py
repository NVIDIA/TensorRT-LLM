# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.auxiliary import (
    AuxBuffer,
    AuxBufferMeta,
    AuxSlot,
    _decode_draft_history,
    _encode_draft_history,
    build_aux_transfer_layout,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import DataType, KVCacheManagerV2

pytestmark = pytest.mark.cpu_only


def test_aux_buffer_meta_construction():
    meta = AuxBufferMeta(
        ptrs=np.array([0x1000, 0x2000], dtype=np.int64),
        size=np.array([512, 1024], dtype=np.int64),
        item_sizes=np.array([32, 64], dtype=np.int64),
        device="cpu",
    )
    assert meta.ptrs.tolist() == [0x1000, 0x2000]
    assert meta.size.tolist() == [512, 1024]
    assert meta.item_sizes.tolist() == [32, 64]
    assert meta.device == "cpu"

    # Test defaults
    meta2 = AuxBufferMeta(
        ptrs=np.array([0x1000], dtype=np.int64), size=np.array([512], dtype=np.int64)
    )
    assert len(meta2.item_sizes) == 0
    assert meta2.device == "cpu"


def test_aux_buffer_meta_to_from_dict():
    meta = AuxBufferMeta(
        ptrs=np.array([0x1000, 0x2000], dtype=np.int64),
        size=np.array([512, 1024], dtype=np.int64),
        item_sizes=np.array([32, 64], dtype=np.int64),
        device="cuda:0",
    )
    d = meta.to_dict()
    assert d == {
        "ptrs": [0x1000, 0x2000],
        "size": [512, 1024],
        "item_sizes": [32, 64],
        "device": "cuda:0",
    }
    restored = AuxBufferMeta.from_dict(d)
    np.testing.assert_array_equal(restored.ptrs, meta.ptrs)
    np.testing.assert_array_equal(restored.size, meta.size)
    np.testing.assert_array_equal(restored.item_sizes, meta.item_sizes)
    assert restored.device == meta.device


def test_aux_buffer_alloc_and_free_slot():
    buf = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=8, device="cpu")
    slot = buf.alloc_slot()
    assert isinstance(slot, AuxSlot)
    assert 0 <= slot.id < 4
    buf.free_slot(slot.id)

    # Can re-allocate after freeing
    slot2 = buf.alloc_slot()
    assert isinstance(slot2, AuxSlot)
    buf.free_slot(slot2.id)


def test_aux_buffer_alloc_full_raises():
    buf = AuxBuffer(max_slot_num=2, beam_width=1, max_draft_len=8, device="cpu")
    buf.alloc_slot()
    buf.alloc_slot()
    with pytest.raises(ValueError, match="No free auxiliary buffer slots"):
        buf.alloc_slot()


def test_aux_buffer_free_unallocated_raises():
    buf = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=8, device="cpu")
    with pytest.raises(ValueError, match="not currently allocated"):
        buf.free_slot(0)


def test_aux_buffer_meta_property():
    buf = AuxBuffer(max_slot_num=4, beam_width=2, max_draft_len=8, device="cpu")
    meta = buf.meta
    assert isinstance(meta, AuxBufferMeta)
    assert len(meta.ptrs) == 4
    assert len(meta.size) == 4
    assert len(meta.item_sizes) == 4
    assert meta.device == "cpu"
    # Verify sizes are positive
    assert all(s > 0 for s in meta.size)
    assert all(s > 0 for s in meta.item_sizes)


def test_fill_slot_get_slot_tokens_round_trip():
    """fill_slot then get_slot_tokens returns the same token data."""
    buf = AuxBuffer(max_slot_num=4, beam_width=2, max_draft_len=4, device="cpu")
    slot = buf.alloc_slot()

    mock_request = MagicMock()
    mock_request.get_last_tokens.return_value = [42, 7]
    mock_request.py_draft_tokens = [10, 20, 30]
    mock_request.prompt_len = 128
    mock_request.cached_tokens = 9
    mock_request.py_disaggregated_params = None

    buf.fill_slot(slot.id, mock_request)
    first_tokens, draft_tokens = buf.get_slot_tokens(slot.id)
    first_tokens_with_usage, draft_tokens_with_usage, (prompt_tokens, cached_tokens) = (
        buf.get_slot_data(slot.id)
    )

    print(
        f"[usage_check] aux_buffer get_slot_data: "
        f"prompt_tokens={prompt_tokens}, cached_tokens={cached_tokens}"
    )
    assert first_tokens == [42, 7]
    assert draft_tokens == [10, 20, 30]
    assert first_tokens_with_usage == [42, 7]
    assert draft_tokens_with_usage == [10, 20, 30]
    assert prompt_tokens == 128
    assert cached_tokens == 9


def test_fill_slot_unallocated_raises():
    """fill_slot on an unallocated slot raises ValueError."""
    buf = AuxBuffer(max_slot_num=4, beam_width=2, max_draft_len=4, device="cpu")
    mock_request = MagicMock()
    mock_request.get_last_tokens.return_value = [1]
    mock_request.py_draft_tokens = []

    with pytest.raises(ValueError, match="not currently allocated"):
        buf.fill_slot(0, mock_request)


def test_aux_buffer_zero_max_draft_len_round_trip():
    """AuxBuffer round-trips with max_draft_len=0.

    A ctx server without speculative_config builds an AuxBuffer with
    max_draft_len=0 (ctx/gen spec-config split). fill/get must round-trip
    with an empty draft-token list.
    """
    buf = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=0, device="cpu")
    slot = buf.alloc_slot()

    mock_request = MagicMock()
    mock_request.get_last_tokens.return_value = [42]
    mock_request.py_draft_tokens = []
    mock_request.prompt_len = 16
    mock_request.cached_tokens = 0
    mock_request.py_disaggregated_params = None

    buf.fill_slot(slot.id, mock_request)
    first_tokens, draft_tokens = buf.get_slot_tokens(slot.id)
    assert first_tokens == [42]
    assert draft_tokens == []

    # The draft buffer's per-slot item size is 0.
    assert buf.meta.item_sizes[1] == 0

    # Overfilling draft tokens must be rejected, not silently truncated.
    mock_request.py_draft_tokens = [1]
    with pytest.raises(ValueError, match="exceeds `max_draft_len`"):
        buf.fill_slot(slot.id, mock_request)


def test_aux_transfer_layout_ctx_no_spec_gen_sa():
    """Ctx/gen spec split: ctx max_draft_len=0, gen max_draft_len=2.

    A ctx server without speculative_config publishes a zero-size draft-token
    buffer while an SA/NGram gen server sizes it by max_draft_len. The
    empty-source entry must be dropped from the transfer layout (nothing to
    send), while the other buffers keep each side's own slot stride.
    """
    ctx = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=0, device="cpu")
    gen = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=2, device="cpu")
    layout = build_aux_transfer_layout(ctx.meta, gen.meta)

    # Buffer order: [first_tokens, draft_tokens, token_counts, prompt_token_counts];
    # the draft_tokens entry (index 1) is dropped.
    expected_keep = [0, 2, 3]
    np.testing.assert_array_equal(layout.src_item_sizes, ctx.meta.item_sizes[expected_keep])
    np.testing.assert_array_equal(layout.dst_item_sizes, gen.meta.item_sizes[expected_keep])
    np.testing.assert_array_equal(layout.src_base_ptrs, ctx.meta.ptrs[expected_keep])
    np.testing.assert_array_equal(layout.dst_base_ptrs, gen.meta.ptrs[expected_keep])


@pytest.mark.parametrize("dtype", ["torch.bfloat16", "torch.float8_e4m3fn"])
def test_draft_history_record_round_trips_the_storage_dtype(dtype):
    """Prefill and generation servers compare the decoded layout, dtype included."""
    history = {
        "valid_length": 89,
        "position": 89,
        "layout": {
            "num_layers": 5,
            "num_kv_heads": 16,
            "head_dim": 64,
            "dtype": dtype,
            "attention_backend": "VANILLA",
            "kv_factor": 2,
            "window_size": None,
        },
    }

    assert _decode_draft_history(_encode_draft_history(history)) == history


def _draft_manager(backend, *, head_dim=576, kv_factor=1, dtype=DataType.BF16):
    manager = object.__new__(KVCacheManagerV2)
    manager.draft_layer_ids = (93,)
    manager.layer_offsets = {93: 0}
    manager._draft_total_num_kv_heads = 1
    manager.num_kv_heads_per_layer = [1]
    manager.head_dim_per_layer = [head_dim]
    manager.dtype_per_layer = [dtype]
    manager.kv_factor_per_layer = [kv_factor]
    manager.draft_window_size = None
    manager.draft_attention_backend = backend
    return manager


@pytest.mark.parametrize("backend", ["VANILLA", "TRTLLM", "CUTEDSL"])
def test_standalone_draft_identity_is_the_storage_format(backend):
    """CTX and GEN may pick different drafter attention backends over one page format."""
    layout = _draft_manager(backend).get_draft_transfer_identity()

    assert layout["attention_backend"] == "VANILLA"
    history = {"valid_length": 1024, "position": 1024, "layout": layout}
    assert _decode_draft_history(_encode_draft_history(history)) == history


def test_embedded_dspark_draft_identity_keeps_its_backend():
    """Only embedded DSpark stores a different history, so it stays distinguishable."""
    layout = _draft_manager("DSv4", head_dim=512).get_draft_transfer_identity()

    assert layout["attention_backend"] == "DSv4"
    history = {"valid_length": 128, "position": 4096, "layout": layout}
    assert _decode_draft_history(_encode_draft_history(history)) == history


def test_draft_identity_names_the_storage_dtype_of_an_fp8_pool():
    """Peers with different bytes per token must not look alike."""
    bf16 = _draft_manager("CUTEDSL").get_draft_transfer_identity()
    fp8 = _draft_manager("CUTEDSL", dtype=DataType.FP8).get_draft_transfer_identity()

    assert bf16["dtype"] == "torch.bfloat16"
    assert fp8["dtype"] == "torch.float8_e4m3fn"
    assert bf16 != fp8
