import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.region.page import BUFFER_ENTRY_DTYPE, PoolDescriptor


def _make_pool_descriptor(base_address=0x10000, slot_bytes=256, num_slots=4):
    entries = np.array(
        [
            (0, 0, 0, 128),  # layer_id=0, role=0(KEY), offset=0, size=128
            (0, 1, 128, 128),  # layer_id=0, role=1(VALUE), offset=128, size=128
        ],
        dtype=BUFFER_ENTRY_DTYPE,
    )
    return PoolDescriptor(
        base_address=base_address,
        slot_bytes=slot_bytes,
        num_slots=num_slots,
        buffer_entries=entries,
    )


def test_pool_descriptor_construction():
    pd = _make_pool_descriptor()
    assert pd.base_address == 0x10000
    assert pd.slot_bytes == 256
    assert pd.num_slots == 4
    assert len(pd.buffer_entries) == 2


def test_pool_descriptor_pool_bytes():
    pd = _make_pool_descriptor(slot_bytes=256, num_slots=4)
    assert pd.pool_bytes == 1024

    pd2 = _make_pool_descriptor(slot_bytes=512, num_slots=8)
    assert pd2.pool_bytes == 4096


def test_pool_descriptor_get_slot_address():
    pd = _make_pool_descriptor(base_address=0x10000, slot_bytes=256, num_slots=4)
    assert pd.get_slot_address(0) == 0x10000
    assert pd.get_slot_address(1) == 0x10000 + 256
    assert pd.get_slot_address(3) == 0x10000 + 768


def test_pool_descriptor_slot_overflow_raises():
    pd = _make_pool_descriptor(num_slots=4)
    with pytest.raises(ValueError, match="slot_id .* >= num_slots"):
        pd.get_slot_address(4)
    with pytest.raises(ValueError, match="slot_id .* >= num_slots"):
        pd.get_slot_address(100)


def test_pool_descriptor_get_device_pointer():
    pd = _make_pool_descriptor(base_address=0x10000, slot_bytes=256, num_slots=4)
    # slot 0, layer 0, role 0 (KEY) → base + 0*256 + offset(0) = 0x10000
    assert pd.get_device_pointer(0, layer_id=0, role_enum=0) == 0x10000
    # slot 0, layer 0, role 1 (VALUE) → base + 0*256 + offset(128) = 0x10080
    assert pd.get_device_pointer(0, layer_id=0, role_enum=1) == 0x10000 + 128
    # slot 2, layer 0, role 0 (KEY) → base + 2*256 + offset(0)
    assert pd.get_device_pointer(2, layer_id=0, role_enum=0) == 0x10000 + 512


def test_pool_descriptor_get_device_pointer_not_found():
    pd = _make_pool_descriptor()
    with pytest.raises(ValueError, match="Buffer not found"):
        pd.get_device_pointer(0, layer_id=99, role_enum=0)
    with pytest.raises(ValueError, match="Buffer not found"):
        pd.get_device_pointer(0, layer_id=0, role_enum=99)
