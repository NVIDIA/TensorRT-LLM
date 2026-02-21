import pytest

from tensorrt_llm._torch.disaggregation.native.region.aux_ import AuxBuffer, AuxBufferMeta


def test_aux_buffer_meta_construction():
    meta = AuxBufferMeta(
        ptrs=[0x1000, 0x2000],
        size=[512, 1024],
        item_sizes=[32, 64],
        device="cpu",
    )
    assert meta.ptrs == [0x1000, 0x2000]
    assert meta.size == [512, 1024]
    assert meta.item_sizes == [32, 64]
    assert meta.device == "cpu"

    # Test defaults
    meta2 = AuxBufferMeta(ptrs=[0x1000], size=[512])
    assert meta2.item_sizes == []
    assert meta2.device == "cpu"


def test_aux_buffer_meta_to_from_dict():
    meta = AuxBufferMeta(
        ptrs=[0x1000, 0x2000],
        size=[512, 1024],
        item_sizes=[32, 64],
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
    assert restored.ptrs == meta.ptrs
    assert restored.size == meta.size
    assert restored.item_sizes == meta.item_sizes
    assert restored.device == meta.device


def test_aux_buffer_alloc_and_free_slot():
    buf = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=8, device="cpu")
    slot = buf.alloc_slot()
    assert isinstance(slot, int)
    assert 0 <= slot < 4
    buf.free_slot(slot)

    # Can re-allocate after freeing
    slot2 = buf.alloc_slot()
    assert isinstance(slot2, int)
    buf.free_slot(slot2)


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
    assert len(meta.ptrs) == 2  # first_tokens_buffer + draft_tokens_buffer
    assert len(meta.size) == 2
    assert len(meta.item_sizes) == 2
    assert meta.device == "cpu"
    # Verify sizes are positive
    assert all(s > 0 for s in meta.size)
    assert all(s > 0 for s in meta.item_sizes)
