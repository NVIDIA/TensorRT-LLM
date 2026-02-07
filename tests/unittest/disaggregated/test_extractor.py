import pytest
import torch

from tensorrt_llm._torch.disaggregation.base.region import MemRegionGroup, SpecRegion
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
    KVPoolAttrs,
    KVRegionExtractorV1,
    LayerGroupAttrs,
    MambaLayerGroupAttrs,
    PoolRole,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    CacheTypeCpp,
    DataType,
    KvCacheConfig,
    KVCacheManager,
    Mapping,
)


class DummyRankInfo:
    instance_name = "dummy"
    instance_rank = 0
    tp_size = 1
    tp_rank = 0
    pp_size = 1
    pp_rank = 0
    dp_size = 1
    dp_rank = 0
    cp_size = 1
    cp_rank = 0
    device_id = 0
    kv_heads_per_rank = 8
    tokens_per_block = 32
    dims_per_head = 16
    element_bytes = 2
    enable_attention_dp = False
    is_mla = False
    layer_num_per_pp = [1]

    @property
    def kv_factor(self) -> int:
        return 2 if not self.is_mla else 1


@pytest.mark.cuda
def test_extract():
    num_layers = 1
    num_kv_heads = 8
    head_dim = 16
    tokens_per_block = 32
    max_seq_len = 128
    max_batch_size = 2
    dtype = DataType.HALF
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1, gpus_per_node=1)
    kv_cache_config = KvCacheConfig(
        max_tokens=512,
        free_gpu_memory_fraction=0.1,
        max_attention_window=None,
        enable_block_reuse=False,
        event_buffer_max_size=0,
        onboard_blocks=0,
        host_cache_size=0,
        enable_partial_reuse=False,
        copy_on_partial_reuse=False,
        sink_token_length=0,
        max_util_for_resume=1,
    )
    kv_cache_type = CacheTypeCpp.SELF

    manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        kv_cache_type=kv_cache_type,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=mapping,
        dtype=dtype,
    )

    extractor = KVRegionExtractorV1(manager)
    region_ids = [0, 1]
    spec_region = extractor.extract(region_ids)

    assert isinstance(spec_region, SpecRegion)
    memory = spec_region.memory
    assert isinstance(memory, MemRegionGroup)
    assert len(memory.ptrs) == len(region_ids)
    assert memory.bytes_per_region > 0

    pool_ptrs = manager.get_unique_primary_pool()
    if hasattr(pool_ptrs, "__getitem__"):
        if hasattr(pool_ptrs[0], "data_ptr"):
            pool_base_ptr = int(pool_ptrs[0].data_ptr())
        else:
            pool_base_ptr = int(pool_ptrs[0])
    else:
        pool_base_ptr = (
            int(pool_ptrs.data_ptr()) if hasattr(pool_ptrs, "data_ptr") else int(pool_ptrs)
        )
    expected_block_bytes = memory.bytes_per_region
    expected_ptrs = [pool_base_ptr + block_id * expected_block_bytes for block_id in region_ids]
    assert list(memory.ptrs) == expected_ptrs

    manager.shutdown()


# ============== Mamba-related tests ==============


class MockMambaCache:
    """Mock MambaCacheManager.State for testing."""

    def __init__(
        self,
        num_local_layers: int,
        max_batch_size: int,
        conv_dim: int,
        d_conv: int,
        nheads: int,
        head_dim: int,
        d_state: int,
        dtype: torch.dtype = torch.float16,
    ):
        # conv_states shape: (num_local_layers, max_batch_size, conv_dim, d_conv - 1)
        self.conv = torch.zeros(
            (num_local_layers, max_batch_size, conv_dim, d_conv - 1),
            dtype=dtype,
            device="cpu",
        )
        # ssm_states shape: (num_local_layers, max_batch_size, nheads, head_dim, d_state)
        self.temporal = torch.zeros(
            (num_local_layers, max_batch_size, nheads, head_dim, d_state),
            dtype=dtype,
            device="cpu",
        )


class MockMambaHybridManager:
    """Mock MambaHybridCacheManager for testing Mamba-related extractor functions."""

    def __init__(
        self,
        num_local_layers: int = 4,
        max_batch_size: int = 8,
        conv_dim: int = 16,
        d_conv: int = 4,
        nheads: int = 8,
        head_dim: int = 64,
        d_state: int = 128,
        dtype: torch.dtype = torch.float16,
        global_layer_ids: list = None,
    ):
        self.mamba_cache = MockMambaCache(
            num_local_layers=num_local_layers,
            max_batch_size=max_batch_size,
            conv_dim=conv_dim,
            d_conv=d_conv,
            nheads=nheads,
            head_dim=head_dim,
            d_state=d_state,
            dtype=dtype,
        )
        # mamba_cache_index: request_id -> slot_id
        self.mamba_cache_index = {}
        # mamba_layer_offsets: {global_layer_idx: local_offset}
        if global_layer_ids is None:
            global_layer_ids = list(range(num_local_layers))
        self.mamba_layer_offsets = {
            global_id: local_idx for local_idx, global_id in enumerate(global_layer_ids)
        }


class MockRegularManager:
    """Mock regular KVCacheManager (without Mamba) for comparison."""

    def __init__(self):
        self.dtype = torch.float16
        # Has KV cache attributes but no Mamba attributes
        pass

    def get_unique_primary_pool(self):
        return [torch.zeros(1024, dtype=self.dtype)]


def test_is_mamba_hybrid_manager():
    """Test _is_mamba_hybrid_manager detection."""
    mamba_manager = MockMambaHybridManager()
    regular_manager = MockRegularManager()

    # MambaHybridManager should be detected
    assert KVRegionExtractorV1._is_mamba_hybrid_manager(mamba_manager) is True

    # Regular manager should not be detected as Mamba
    assert KVRegionExtractorV1._is_mamba_hybrid_manager(regular_manager) is False

    # Object without required attributes
    class EmptyObj:
        pass

    assert KVRegionExtractorV1._is_mamba_hybrid_manager(EmptyObj()) is False

    # Object with only mamba_cache but no mamba_cache_index
    class PartialMamba:
        mamba_cache = None

    assert KVRegionExtractorV1._is_mamba_hybrid_manager(PartialMamba()) is False


def test_attrs_from_mamba_manager():
    """Test _attrs_from_mamba_manager creates correct LayerGroupAttrs."""
    num_local_layers = 4
    max_batch_size = 8
    conv_dim = 16
    d_conv = 4
    nheads = 8
    head_dim = 64
    d_state = 128
    dtype = torch.float16
    global_layer_ids = [10, 11, 12, 13]  # Non-zero global layer IDs

    manager = MockMambaHybridManager(
        num_local_layers=num_local_layers,
        max_batch_size=max_batch_size,
        conv_dim=conv_dim,
        d_conv=d_conv,
        nheads=nheads,
        head_dim=head_dim,
        d_state=d_state,
        dtype=dtype,
        global_layer_ids=global_layer_ids,
    )

    attrs = KVRegionExtractorV1._attrs_from_mamba_manager(manager)

    # Check type
    assert isinstance(attrs, LayerGroupAttrs)

    # Check group_id (should be -1, set by caller)
    assert attrs.group_id == -1

    # Check pool base pointers (2 pools: Conv and SSM)
    assert len(attrs.pool_base_ptrs) == 2
    assert attrs.pool_base_ptrs[0] == int(manager.mamba_cache.conv.data_ptr())
    assert attrs.pool_base_ptrs[1] == int(manager.mamba_cache.temporal.data_ptr())

    # Check pool sizes
    assert len(attrs.pool_sizes) == 2
    conv_size = manager.mamba_cache.conv.element_size() * manager.mamba_cache.conv.numel()
    ssm_size = manager.mamba_cache.temporal.element_size() * manager.mamba_cache.temporal.numel()
    assert attrs.pool_sizes[0] == conv_size
    assert attrs.pool_sizes[1] == ssm_size

    # Check roles_to_pool_idx
    assert PoolRole.CONV_STATE in attrs.roles_to_pool_idx
    assert PoolRole.SSM_STATE in attrs.roles_to_pool_idx
    assert attrs.roles_to_pool_idx[PoolRole.CONV_STATE] == 0
    assert attrs.roles_to_pool_idx[PoolRole.SSM_STATE] == 1

    # Check block_bytes_per_pool (bytes per slot per layer)
    assert len(attrs.block_bytes_per_pool) == 2
    # Conv: (conv_dim, d_conv - 1) per slot per layer
    expected_conv_bytes = conv_dim * (d_conv - 1) * dtype.itemsize
    # SSM: (nheads, head_dim, d_state) per slot per layer
    expected_ssm_bytes = nheads * head_dim * d_state * dtype.itemsize
    assert attrs.block_bytes_per_pool[0] == expected_conv_bytes
    assert attrs.block_bytes_per_pool[1] == expected_ssm_bytes

    # Check global_layer_ids
    assert attrs.global_layer_ids == sorted(global_layer_ids)

    # Check kv_head_num_per_rank (nheads from SSM state)
    assert attrs.kv_head_num_per_rank == nheads

    # Check max_batch_size_per_pool
    assert attrs.max_batch_size_per_pool == max_batch_size


def test_extract_mamba():
    """Test extract method for Mamba states (SSM_STATE, CONV_STATE)."""
    num_local_layers = 4
    max_batch_size = 8
    conv_dim = 16
    d_conv = 4
    nheads = 8
    head_dim = 64
    d_state = 128
    dtype = torch.float16
    global_layer_ids = [0, 1, 2, 3]

    manager = MockMambaHybridManager(
        num_local_layers=num_local_layers,
        max_batch_size=max_batch_size,
        conv_dim=conv_dim,
        d_conv=d_conv,
        nheads=nheads,
        head_dim=head_dim,
        d_state=d_state,
        dtype=dtype,
        global_layer_ids=global_layer_ids,
    )

    # Create LayerGroupAttrs for Mamba
    mamba_attrs = KVRegionExtractorV1._attrs_from_mamba_manager(manager)
    mamba_attrs.group_id = 0

    # Create KVPoolAttrs with Mamba layer group
    kv_pool_attrs = KVPoolAttrs(
        layer_to_group_id={lid: 0 for lid in global_layer_ids},
        layer_group_attrs_list=[mamba_attrs],
    )

    extractor = KVRegionExtractorV1(kv_pool_attrs)

    # Test extracting SSM_STATE with slot_id=3
    slot_id = 3
    spec_region = extractor.extract(
        region_ids=[slot_id],
        layer_group_id=0,
        pool_role=PoolRole.SSM_STATE,
    )

    assert isinstance(spec_region, SpecRegion)
    memory = spec_region.memory
    assert isinstance(memory, MemRegionGroup)

    # Should return num_local_layers addresses (one per layer)
    assert len(memory.ptrs) == num_local_layers

    # Verify address calculation for each layer
    ssm_base_ptr = int(manager.mamba_cache.temporal.data_ptr())
    ssm_block_size = nheads * head_dim * d_state * dtype.itemsize
    for layer_idx in range(num_local_layers):
        expected_addr = ssm_base_ptr + (layer_idx * max_batch_size + slot_id) * ssm_block_size
        assert memory.ptrs[layer_idx] == expected_addr

    assert memory.bytes_per_region == ssm_block_size

    # Test extracting CONV_STATE with slot_id=5
    slot_id = 5
    spec_region = extractor.extract(
        region_ids=[slot_id],
        layer_group_id=0,
        pool_role=PoolRole.CONV_STATE,
    )

    memory = spec_region.memory
    assert len(memory.ptrs) == num_local_layers

    conv_base_ptr = int(manager.mamba_cache.conv.data_ptr())
    conv_block_size = conv_dim * (d_conv - 1) * dtype.itemsize
    for layer_idx in range(num_local_layers):
        expected_addr = conv_base_ptr + (layer_idx * max_batch_size + slot_id) * conv_block_size
        assert memory.ptrs[layer_idx] == expected_addr

    assert memory.bytes_per_region == conv_block_size


def test_extract_mamba_invalid_slot():
    """Test extract method with invalid slot_id returns empty region."""
    manager = MockMambaHybridManager(num_local_layers=4, max_batch_size=8)

    mamba_attrs = KVRegionExtractorV1._attrs_from_mamba_manager(manager)
    mamba_attrs.group_id = 0

    kv_pool_attrs = KVPoolAttrs(
        layer_to_group_id={i: 0 for i in range(4)},
        layer_group_attrs_list=[mamba_attrs],
    )

    extractor = KVRegionExtractorV1(kv_pool_attrs)

    # Test with negative slot_id
    spec_region = extractor.extract(
        region_ids=[-1],
        layer_group_id=0,
        pool_role=PoolRole.SSM_STATE,
    )

    memory = spec_region.memory
    assert len(memory.ptrs) == 0
    assert memory.bytes_per_region == 0


def test_extract_mamba_requires_single_slot():
    """Test that Mamba extract raises error if multiple slot_ids provided."""
    manager = MockMambaHybridManager(num_local_layers=4, max_batch_size=8)

    mamba_attrs = KVRegionExtractorV1._attrs_from_mamba_manager(manager)
    mamba_attrs.group_id = 0

    kv_pool_attrs = KVPoolAttrs(
        layer_to_group_id={i: 0 for i in range(4)},
        layer_group_attrs_list=[mamba_attrs],
    )

    extractor = KVRegionExtractorV1(kv_pool_attrs)

    # Test with multiple slot_ids (should raise ValueError)
    with pytest.raises(ValueError, match="Mamba extract expects exactly 1 slot_id"):
        extractor.extract(
            region_ids=[0, 1, 2],
            layer_group_id=0,
            pool_role=PoolRole.SSM_STATE,
        )


def test_layer_group_attrs_serialization():
    """Test LayerGroupAttrs to_dict and from_dict with Mamba pool roles."""
    attrs = MambaLayerGroupAttrs(
        group_id=1,
        pool_base_ptrs=[1000, 2000],
        pool_sizes=[4096, 8192],
        roles_to_pool_idx={PoolRole.CONV_STATE: 0, PoolRole.SSM_STATE: 1},
        block_bytes_per_pool=[128, 256],
        global_layer_ids=[0, 1, 2, 3],
        kv_head_num_per_rank=8,
        max_batch_size_per_pool=16,
    )

    # Test serialization
    attrs_dict = attrs.to_dict()
    assert attrs_dict["group_id"] == 1
    assert attrs_dict["roles_to_pool_idx"] == {"CONV_STATE": 0, "SSM_STATE": 1}
    assert attrs_dict["max_batch_size_per_pool"] == 16

    # Test deserialization
    restored_attrs = MambaLayerGroupAttrs.from_dict(attrs_dict)
    assert restored_attrs.group_id == 1
    assert restored_attrs.roles_to_pool_idx[PoolRole.CONV_STATE] == 0
    assert restored_attrs.roles_to_pool_idx[PoolRole.SSM_STATE] == 1
    assert restored_attrs.max_batch_size_per_pool == 16
    assert restored_attrs.global_layer_ids == [0, 1, 2, 3]


def test_mamba_layer_group_attrs_serialization():
    """Test MambaLayerGroupAttrs to_dict/from_dict with conv section info."""
    attrs = MambaLayerGroupAttrs(
        group_id=2,
        pool_base_ptrs=[3000, 4000],
        pool_sizes=[16384, 32768],
        roles_to_pool_idx={PoolRole.CONV_STATE: 0, PoolRole.SSM_STATE: 1},
        block_bytes_per_pool=[512, 1024],
        global_layer_ids=[0, 2, 4, 6],
        kv_head_num_per_rank=4,
        max_batch_size_per_pool=8,
        conv_section_bytes_per_rank=[64, 64, 128],
        d_inner_per_rank=128,
        ng_ds_per_rank=64,
        d_conv=4,
        conv_elem_size=2,
        ssm_elem_size=4,
        ssm_head_dim=16,
        ssm_d_state=32,
    )

    # Test serialization
    attrs_dict = attrs.to_dict()
    assert attrs_dict["_type"] == "mamba"
    assert attrs_dict["conv_section_bytes_per_rank"] == [64, 64, 128]
    assert attrs_dict["d_inner_per_rank"] == 128
    assert attrs_dict["d_conv"] == 4

    # Test deserialization via MambaLayerGroupAttrs.from_dict
    restored = MambaLayerGroupAttrs.from_dict(attrs_dict)
    assert isinstance(restored, MambaLayerGroupAttrs)
    assert restored.conv_section_bytes_per_rank == [64, 64, 128]
    assert restored.d_inner_per_rank == 128
    assert restored.ng_ds_per_rank == 64
    assert restored.d_conv == 4
    assert restored.group_id == 2
    assert restored.roles_to_pool_idx[PoolRole.CONV_STATE] == 0

    # Test polymorphic deserialization via LayerGroupAttrs.from_dict
    # When _type="mamba" is present, it should dispatch to MambaLayerGroupAttrs
    restored_poly = LayerGroupAttrs.from_dict(attrs_dict)
    assert isinstance(restored_poly, MambaLayerGroupAttrs)
    assert restored_poly.conv_section_bytes_per_rank == [64, 64, 128]


def test_kv_pool_attrs_with_mamba_layer_group_serialization():
    """Test KVPoolAttrs serialization with a mix of LayerGroupAttrs and MambaLayerGroupAttrs."""
    kv_group = LayerGroupAttrs(
        group_id=0,
        pool_base_ptrs=[1000],
        pool_sizes=[4096],
        roles_to_pool_idx={PoolRole.KV_CACHE: 0},
        block_bytes_per_pool=[128],
        global_layer_ids=[1, 3, 5],
        kv_head_num_per_rank=8,
    )
    mamba_group = MambaLayerGroupAttrs(
        group_id=1,
        pool_base_ptrs=[2000, 3000],
        pool_sizes=[8192, 16384],
        roles_to_pool_idx={PoolRole.CONV_STATE: 0, PoolRole.SSM_STATE: 1},
        block_bytes_per_pool=[256, 512],
        global_layer_ids=[0, 2, 4, 6],
        kv_head_num_per_rank=4,
        max_batch_size_per_pool=16,
        conv_section_bytes_per_rank=[32, 32, 64],
        d_conv=4,
    )

    pool_attrs = KVPoolAttrs(
        layer_to_group_id={1: 0, 3: 0, 5: 0, 0: 1, 2: 1, 4: 1, 6: 1},
        layer_group_attrs_list=[kv_group, mamba_group],
    )

    # Serialize
    d = pool_attrs.to_dict()

    # Deserialize
    restored = KVPoolAttrs.from_dict(d)
    assert len(restored.layer_group_attrs_list) == 2

    # First group should be a plain LayerGroupAttrs
    g0 = restored.layer_group_attrs_list[0]
    assert not isinstance(g0, MambaLayerGroupAttrs)
    assert g0.group_id == 0

    # Second group should be a MambaLayerGroupAttrs
    g1 = restored.layer_group_attrs_list[1]
    assert isinstance(g1, MambaLayerGroupAttrs)
    assert g1.conv_section_bytes_per_rank == [32, 32, 64]
    assert g1.d_conv == 4
