import pytest

from tensorrt_llm._torch.disaggregation.base.region import (
    DataLayout,
    DataRole,
    IndexRange,
    KVRegionSpec,
    RegionSpec,
)


def test_index_range_valid():
    r = IndexRange(start=0, end=10)
    assert r.start == 0
    assert r.end == 10

    # end == start is valid (closed interval)
    r2 = IndexRange(start=5, end=5)
    assert r2.start == 5
    assert r2.end == 5


def test_index_range_invalid_negative():
    with pytest.raises(ValueError, match="must be >= 0"):
        IndexRange(start=-1, end=5)
    with pytest.raises(ValueError, match="must be >= 0"):
        IndexRange(start=0, end=-1)


def test_index_range_invalid_non_int():
    with pytest.raises(TypeError, match="must be integers"):
        IndexRange(start=1.5, end=5)
    with pytest.raises(TypeError, match="must be integers"):
        IndexRange(start=0, end="3")


def test_index_range_invalid_end_before_start():
    with pytest.raises(ValueError, match="end must be >= start"):
        IndexRange(start=10, end=5)


def test_data_role_flags():
    assert DataRole.KEY == 1
    assert DataRole.VALUE == 2
    combined = DataRole.KEY | DataRole.VALUE
    assert DataRole.KEY in combined
    assert DataRole.VALUE in combined


def test_data_layout_flags():
    assert DataLayout.HND == 1
    assert DataLayout.NHD == 2
    assert DataLayout.HND != DataLayout.NHD


def test_region_spec_construction():
    spec = RegionSpec()
    assert spec.layers is None

    spec2 = RegionSpec(layers=IndexRange(0, 31))
    assert spec2.layers.start == 0
    assert spec2.layers.end == 31


def test_kv_region_spec_defaults():
    spec = KVRegionSpec()
    assert spec.layers is None
    assert spec.role == DataRole.KEY | DataRole.VALUE
    assert spec.heads is None
    assert spec.tokens is None


def test_kv_region_spec_with_all_axes():
    spec = KVRegionSpec(
        layers=IndexRange(0, 15),
        role=DataRole.KEY,
        heads=IndexRange(0, 7),
        tokens=IndexRange(0, 127),
    )
    assert spec.layers == IndexRange(0, 15)
    assert spec.role == DataRole.KEY
    assert spec.heads == IndexRange(0, 7)
    assert spec.tokens == IndexRange(0, 127)
