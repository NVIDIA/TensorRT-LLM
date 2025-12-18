import pytest

# Try to import the transfer agent binding module
try:
    import tensorrt_llm.tensorrt_llm_transfer_agent_binding as tab

    HAS_TRANSFER_AGENT = True
except ImportError:
    HAS_TRANSFER_AGENT = False

pytestmark = pytest.mark.skipif(
    not HAS_TRANSFER_AGENT,
    reason="Transfer agent bindings not available (tensorrt_llm_transfer_agent_binding)",
)


def test_memory_type_enum():
    """Test MemoryType enum values."""
    assert tab.MemoryType.DRAM is not None
    assert tab.MemoryType.VRAM is not None
    assert tab.MemoryType.BLK is not None
    assert tab.MemoryType.OBJ is not None
    assert tab.MemoryType.FILE is not None

    # Verify they are distinct
    assert tab.MemoryType.DRAM != tab.MemoryType.VRAM
    assert tab.MemoryType.VRAM != tab.MemoryType.BLK


def test_transfer_op_enum():
    """Test TransferOp enum values."""
    assert tab.TransferOp.READ is not None
    assert tab.TransferOp.WRITE is not None
    assert tab.TransferOp.READ != tab.TransferOp.WRITE


def test_transfer_state_enum():
    """Test TransferState enum values."""
    assert tab.TransferState.IN_PROGRESS is not None
    assert tab.TransferState.SUCCESS is not None
    assert tab.TransferState.FAILURE is not None

    # Verify they are distinct
    assert tab.TransferState.IN_PROGRESS != tab.TransferState.SUCCESS
    assert tab.TransferState.SUCCESS != tab.TransferState.FAILURE
    assert tab.TransferState.IN_PROGRESS != tab.TransferState.FAILURE


def test_memory_desc():
    """Test MemoryDesc class."""
    addr = 0x1000
    length = 4096
    device_id = 0

    desc = tab.MemoryDesc(addr, length, device_id)

    assert desc.addr == addr
    assert desc.len == length
    assert desc.device_id == device_id


def test_memory_desc_different_values():
    """Test MemoryDesc with different values."""
    test_cases = [
        (0x0, 1, 0),
        (0xFFFFFFFF, 65536, 1),
        (0x12345678, 1024, 7),
    ]

    for addr, length, device_id in test_cases:
        desc = tab.MemoryDesc(addr, length, device_id)
        assert desc.addr == addr
        assert desc.len == length
        assert desc.device_id == device_id


def test_memory_descs():
    """Test MemoryDescs class."""
    desc1 = tab.MemoryDesc(0x1000, 4096, 0)
    desc2 = tab.MemoryDesc(0x2000, 8192, 0)

    descs = tab.MemoryDescs(tab.MemoryType.VRAM, [desc1, desc2])

    assert descs.type == tab.MemoryType.VRAM
    assert len(descs.descs) == 2
    assert descs.descs[0].addr == 0x1000
    assert descs.descs[1].addr == 0x2000


def test_memory_descs_empty():
    """Test MemoryDescs with empty list."""
    descs = tab.MemoryDescs(tab.MemoryType.DRAM, [])
    assert descs.type == tab.MemoryType.DRAM
    assert len(descs.descs) == 0


def test_agent_desc_from_string():
    """Test AgentDesc from string."""
    test_data = "test_agent_descriptor"
    desc = tab.AgentDesc(test_data)
    assert desc.backend_agent_desc == test_data.encode()


def test_agent_desc_from_bytes():
    """Test AgentDesc from bytes."""
    test_data = b"test_binary_data\x00\x01\x02"
    desc = tab.AgentDesc(test_data)
    assert desc.backend_agent_desc == test_data


def test_base_agent_config_default():
    """Test BaseAgentConfig with default values."""
    config = tab.BaseAgentConfig()
    # Default values should be set
    assert config is not None


def test_base_agent_config_custom():
    """Test BaseAgentConfig with custom values."""
    name = "test_agent"
    use_prog_thread = True
    multi_thread = False
    use_listen_thread = True
    num_workers = 4

    config = tab.BaseAgentConfig(
        name=name,
        use_prog_thread=use_prog_thread,
        multi_thread=multi_thread,
        use_listen_thread=use_listen_thread,
        num_workers=num_workers,
    )

    assert config.name == name
    assert config.use_prog_thread == use_prog_thread
    assert config.multi_thread == multi_thread
    assert config.use_listen_thread == use_listen_thread
    assert config.num_workers == num_workers


def test_base_agent_config_readwrite():
    """Test BaseAgentConfig read/write properties."""
    config = tab.BaseAgentConfig()

    config.name = "modified_name"
    assert config.name == "modified_name"

    config.use_prog_thread = False
    assert config.use_prog_thread is False

    config.multi_thread = True
    assert config.multi_thread is True

    config.use_listen_thread = True
    assert config.use_listen_thread is True

    config.num_workers = 8
    assert config.num_workers == 8


def test_transfer_request():
    """Test TransferRequest class."""
    src_desc = tab.MemoryDesc(0x1000, 4096, 0)
    dst_desc = tab.MemoryDesc(0x2000, 4096, 1)

    src_descs = tab.MemoryDescs(tab.MemoryType.VRAM, [src_desc])
    dst_descs = tab.MemoryDescs(tab.MemoryType.VRAM, [dst_desc])

    remote_name = "remote_agent"

    request = tab.TransferRequest(tab.TransferOp.WRITE, src_descs, dst_descs, remote_name)

    assert request.op == tab.TransferOp.WRITE
    assert request.remote_name == remote_name
    assert request.src_descs.type == tab.MemoryType.VRAM
    assert request.dst_descs.type == tab.MemoryType.VRAM


def test_transfer_request_read_op():
    """Test TransferRequest with READ operation."""
    src_desc = tab.MemoryDesc(0x3000, 2048, 0)
    dst_desc = tab.MemoryDesc(0x4000, 2048, 0)

    src_descs = tab.MemoryDescs(tab.MemoryType.DRAM, [src_desc])
    dst_descs = tab.MemoryDescs(tab.MemoryType.DRAM, [dst_desc])

    request = tab.TransferRequest(tab.TransferOp.READ, src_descs, dst_descs, "another_remote")

    assert request.op == tab.TransferOp.READ
    assert request.remote_name == "another_remote"
