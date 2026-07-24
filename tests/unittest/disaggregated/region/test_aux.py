from unittest.mock import MagicMock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.auxiliary import (
    AuxBuffer,
    AuxBufferMeta,
    AuxSlot,
    compute_aux_transfer_descs,
)


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
    """A ctx server without speculative_config builds an AuxBuffer with
    max_draft_len=0 (ctx/gen spec-config split). fill/get must round-trip
    with an empty draft-token list."""
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


def test_compute_aux_transfer_descs_symmetric():
    """Matched spec configs: every buffer transfers at its full item size,
    offsets follow each side's slot index."""
    src = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=4, device="cpu")
    dst = AuxBuffer(max_slot_num=8, beam_width=1, max_draft_len=4, device="cpu")
    src_ptrs, dst_ptrs, sizes = compute_aux_transfer_descs(src.meta, dst.meta, 2, 5)

    assert len(src_ptrs) == len(dst_ptrs) == len(sizes) == 4
    np.testing.assert_array_equal(sizes, src.meta.item_sizes)
    np.testing.assert_array_equal(src_ptrs, src.meta.ptrs + src.meta.item_sizes * 2)
    np.testing.assert_array_equal(dst_ptrs, dst.meta.ptrs + dst.meta.item_sizes * 5)


def test_compute_aux_transfer_descs_ctx_no_spec_gen_sa():
    """Ctx/gen spec split: ctx max_draft_len=0, gen max_draft_len=2.

    The zero-size draft-token entry must be dropped (nothing to send; a
    zero-byte RDMA descriptor is agent-dependent behavior), while the other
    buffers transfer at full size using each side's own slot stride."""
    ctx = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=0, device="cpu")
    gen = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=2, device="cpu")
    src_ptrs, dst_ptrs, sizes = compute_aux_transfer_descs(ctx.meta, gen.meta, 1, 3)

    # Buffer order: [first_tokens, draft_tokens, token_counts, prompt_token_counts];
    # the draft_tokens entry (index 1) is dropped.
    assert len(sizes) == 3
    expected_keep = [0, 2, 3]
    np.testing.assert_array_equal(sizes, ctx.meta.item_sizes[expected_keep])
    np.testing.assert_array_equal(
        src_ptrs, (ctx.meta.ptrs + ctx.meta.item_sizes * 1)[expected_keep]
    )
    np.testing.assert_array_equal(
        dst_ptrs, (gen.meta.ptrs + gen.meta.item_sizes * 3)[expected_keep]
    )


def test_compute_aux_transfer_descs_src_larger_than_dst_clamps():
    """Reverse asymmetry (src draft buffer wider than dst): the transfer size
    must clamp to the destination's item size so the write cannot overrun the
    destination slot."""
    src = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=4, device="cpu")
    dst = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=2, device="cpu")
    _, _, sizes = compute_aux_transfer_descs(src.meta, dst.meta, 0, 0)

    assert len(sizes) == 4
    # draft_tokens entry clamped to dst's 2-token item size.
    assert sizes[1] == dst.meta.item_sizes[1]
    np.testing.assert_array_equal(sizes, np.minimum(src.meta.item_sizes, dst.meta.item_sizes))
