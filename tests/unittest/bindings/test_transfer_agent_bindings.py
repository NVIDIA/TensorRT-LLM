# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pytest

# Try to import the transfer agent binding module
try:
    import tensorrt_llm.tensorrt_llm_transfer_agent_binding as tab

    HAS_TRANSFER_AGENT = True
    # Check which backends are available (compile-time flags)
    HAS_NIXL = getattr(tab, "NIXL_ENABLED", False)
    HAS_MOONCAKE = getattr(tab, "MOONCAKE_ENABLED", False)
except ImportError:
    HAS_TRANSFER_AGENT = False
    HAS_NIXL = False
    HAS_MOONCAKE = False


def _is_mooncake_runtime_available():
    """Check if Mooncake runtime libraries are actually available.

    HAS_MOONCAKE only indicates compile-time support. At runtime,
    DynLibLoader::dlopen searches: LD_LIBRARY_PATH -> RUNPATH -> system paths.
    We mirror this by trying the library name first (covers LD_LIBRARY_PATH
    and dev builds), then falling back to the bundled path (installed wheels).
    """
    if not HAS_MOONCAKE:
        return False

    import ctypes
    import os

    wrapper_name = "libtensorrt_llm_mooncake_wrapper.so"

    # 1) Try by name: finds via LD_LIBRARY_PATH / system paths (dev workflow)
    try:
        ctypes.CDLL(wrapper_name)
        return True
    except OSError:
        pass

    # 2) Fallback: try bundled path (installed wheel)
    try:
        binding_dir = os.path.dirname(tab.__file__)
        wrapper_path = os.path.join(binding_dir, "libs", wrapper_name)
        ctypes.CDLL(wrapper_path)
        return True
    except (OSError, AttributeError, TypeError):
        pass

    return False


HAS_MOONCAKE_RUNTIME = _is_mooncake_runtime_available()

# Try to import torch for functional tests
try:
    import torch

    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

pytestmark = pytest.mark.skipif(
    not HAS_TRANSFER_AGENT,
    reason="Transfer agent bindings not available (tensorrt_llm_transfer_agent_binding)",
)

# =============================================================================
# Common Tests (independent of backend)
# =============================================================================


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


def test_agent_desc_serialize_returns_bytes():
    """Test that AgentDesc.serialize() returns bytes."""
    desc = tab.AgentDesc("some_blob")
    serialized = desc.serialize()
    assert isinstance(serialized, bytes)
    assert len(serialized) > 0


def test_agent_desc_serialize_deserialize_no_regions():
    """Test AgentDesc serialize/deserialize roundtrip without VMM regions."""
    original_blob = b"nixl_metadata_blob_content"
    desc = tab.AgentDesc(original_blob)
    serialized = desc.serialize()

    restored = tab.AgentDesc.deserialize(serialized)
    assert restored.backend_agent_desc == original_blob


def test_agent_desc_serialize_deserialize_roundtrip_string():
    """Test AgentDesc serialize/deserialize roundtrip with string input."""
    original_blob = "string_blob_data"
    desc = tab.AgentDesc(original_blob)
    serialized = desc.serialize()

    restored = tab.AgentDesc.deserialize(serialized)
    assert restored.backend_agent_desc == original_blob.encode()


def test_agent_desc_serialize_deserialize_binary_blob():
    """Test AgentDesc serialize/deserialize with binary data containing null bytes."""
    # NIXL blobs can contain arbitrary binary data including null bytes
    original_blob = bytes(range(256))
    desc = tab.AgentDesc(original_blob)
    serialized = desc.serialize()

    restored = tab.AgentDesc.deserialize(serialized)
    assert restored.backend_agent_desc == original_blob


def test_agent_desc_deserialize_accepts_bytes():
    """Test that AgentDesc.deserialize() accepts bytes input."""
    desc = tab.AgentDesc(b"test_data")
    serialized = desc.serialize()

    # deserialize should accept bytes
    restored = tab.AgentDesc.deserialize(serialized)
    assert restored.backend_agent_desc == b"test_data"


def test_agent_desc_serialize_deserialize_empty_blob():
    """Test AgentDesc serialize/deserialize with empty backend blob."""
    desc = tab.AgentDesc(b"")
    serialized = desc.serialize()

    restored = tab.AgentDesc.deserialize(serialized)
    assert restored.backend_agent_desc == b""


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
    enable_telemetry = True
    backend_params = {"key1": "value1", "key2": "value2"}

    config = tab.BaseAgentConfig(
        name=name,
        use_prog_thread=use_prog_thread,
        multi_thread=multi_thread,
        use_listen_thread=use_listen_thread,
        enable_telemetry=enable_telemetry,
        backend_params=backend_params,
    )

    assert config.name == name
    assert config.use_prog_thread == use_prog_thread
    assert config.multi_thread == multi_thread
    assert config.use_listen_thread == use_listen_thread
    assert config.enable_telemetry == enable_telemetry
    assert config.backend_params == backend_params


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

    config.enable_telemetry = True
    assert config.enable_telemetry is True

    config.backend_params = {"test_key": "test_value"}
    assert config.backend_params == {"test_key": "test_value"}


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


def test_backend_availability_flags():
    """Test that backend availability flags are exposed."""
    # These should always be defined (either True or False)
    assert hasattr(tab, "NIXL_ENABLED")
    assert hasattr(tab, "MOONCAKE_ENABLED")
    assert isinstance(tab.NIXL_ENABLED, bool)
    assert isinstance(tab.MOONCAKE_ENABLED, bool)


# =============================================================================
# NIXL-specific Tests
# =============================================================================


@pytest.mark.skipif(not HAS_NIXL, reason="NIXL backend not available")
class TestNixlTransferAgent:
    """Test cases for NixlTransferAgent."""

    def test_nixl_transfer_agent_class_exists(self):
        """Test that NixlTransferAgent class exists."""
        assert hasattr(tab, "NixlTransferAgent")

    def test_nixl_transfer_status_class_exists(self):
        """Test that NixlTransferStatus class exists."""
        assert hasattr(tab, "NixlTransferStatus")

    def test_nixl_transfer_agent_is_base_subclass(self):
        """Test that NixlTransferAgent is a subclass of BaseTransferAgent."""
        assert issubclass(tab.NixlTransferAgent, tab.BaseTransferAgent)

    def test_nixl_transfer_status_is_base_subclass(self):
        """Test that NixlTransferStatus is a subclass of TransferStatus."""
        assert issubclass(tab.NixlTransferStatus, tab.TransferStatus)

    def test_nixl_transfer_agent_has_required_methods(self):
        """Test that NixlTransferAgent has all required methods."""
        required_methods = [
            "register_memory",
            "deregister_memory",
            "load_remote_agent",
            "load_remote_agent_by_connection",
            "get_local_agent_desc",
            "get_local_connection_info",
            "invalidate_remote_agent",
            "submit_transfer_requests",
            "notify_sync_message",
            "get_notified_sync_messages",
            "check_remote_descs",
        ]
        for method in required_methods:
            assert hasattr(tab.NixlTransferAgent, method), f"Missing method: {method}"


# =============================================================================
# Mooncake-specific Tests
# =============================================================================


@pytest.mark.skipif(not HAS_MOONCAKE, reason="Mooncake backend not available")
class TestMooncakeTransferAgent:
    """Test cases for Mooncake transfer agent via make_transfer_agent factory.

    Note: MooncakeTransferAgent/MooncakeTransferStatus are not directly exposed
    as nanobind classes to avoid a hard load-time dependency on libtransfer_engine.so.
    Instead, agents are created via the make_transfer_agent("mooncake", ...) factory
    which uses dlopen for lazy loading.
    """

    def test_mooncake_enabled_flag(self):
        """Test that MOONCAKE_ENABLED flag is set."""
        assert tab.MOONCAKE_ENABLED is True

    def test_make_transfer_agent_factory_exists(self):
        """Test that the make_transfer_agent factory function exists."""
        assert hasattr(tab, "make_transfer_agent")


# =============================================================================
# Functional Tests - Data Transfer Validation
# =============================================================================


def _create_memory_descs_from_tensor(tensor, memory_type):
    """Helper to create MemoryDescs from a torch tensor."""
    addr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    device_id = tensor.device.index if tensor.is_cuda else 0
    desc = tab.MemoryDesc(addr, size, device_id)
    return tab.MemoryDescs(memory_type, [desc])


@pytest.mark.skipif(
    not (HAS_TORCH and HAS_CUDA),
    reason="Torch with CUDA support required for functional tests",
)
@pytest.mark.skipif(not HAS_NIXL, reason="NIXL backend not available")
class TestNixlFunctionalTransfer:
    """Functional tests for NIXL data transfer between two agents."""

    def test_nixl_write_transfer_gpu_tensor(self):
        """Test WRITE transfer of GPU tensor data between two NIXL agents."""
        device = torch.device("cuda:0")

        # Create source tensor with known data pattern
        src_tensor = torch.arange(1024, dtype=torch.float32, device=device)

        # Create destination tensor (zeros)
        dst_tensor = torch.zeros(1024, dtype=torch.float32, device=device)

        # Verify initial state
        assert not torch.equal(src_tensor, dst_tensor)

        # Create two agents
        config_a = tab.BaseAgentConfig(
            name="agent_a",
            use_prog_thread=True,
            use_listen_thread=False,
        )
        config_b = tab.BaseAgentConfig(
            name="agent_b",
            use_prog_thread=True,
            use_listen_thread=False,
        )

        agent_a = tab.NixlTransferAgent(config_a)
        agent_b = tab.NixlTransferAgent(config_b)

        # Register memory regions
        src_descs = _create_memory_descs_from_tensor(src_tensor, tab.MemoryType.VRAM)
        dst_descs = _create_memory_descs_from_tensor(dst_tensor, tab.MemoryType.VRAM)

        agent_a.register_memory(src_descs)
        agent_b.register_memory(dst_descs)

        # Exchange agent descriptors
        agent_a_desc = agent_a.get_local_agent_desc()
        agent_b_desc = agent_b.get_local_agent_desc()

        agent_a.load_remote_agent("agent_b", agent_b_desc)
        agent_b.load_remote_agent("agent_a", agent_a_desc)

        # Create transfer request: agent_a writes src_tensor to agent_b's dst_tensor
        request = tab.TransferRequest(
            tab.TransferOp.WRITE,
            src_descs,  # local source
            dst_descs,  # remote destination
            "agent_b",  # remote agent name
        )

        # Submit transfer and wait for completion
        status = agent_a.submit_transfer_requests(request)
        result = status.wait(timeout_ms=5000)

        assert result == tab.TransferState.SUCCESS, f"Transfer failed with state: {result}"

        # Synchronize CUDA to ensure transfer is complete
        torch.cuda.synchronize()

        # Verify data was transferred correctly
        assert torch.equal(src_tensor, dst_tensor), "Data mismatch after transfer"

        # Cleanup
        agent_a.deregister_memory(src_descs)
        agent_b.deregister_memory(dst_descs)

    def test_nixl_write_transfer_multiple_chunks(self):
        """Test WRITE transfer with multiple memory chunks."""
        device = torch.device("cuda:0")

        # Create multiple source tensors
        src_tensors = [
            torch.arange(i * 256, (i + 1) * 256, dtype=torch.float32, device=device)
            for i in range(4)
        ]

        # Create corresponding destination tensors
        dst_tensors = [torch.zeros(256, dtype=torch.float32, device=device) for _ in range(4)]

        # Create agents
        config_a = tab.BaseAgentConfig(
            name="agent_a", use_prog_thread=True, use_listen_thread=False
        )
        config_b = tab.BaseAgentConfig(
            name="agent_b", use_prog_thread=True, use_listen_thread=False
        )

        agent_a = tab.NixlTransferAgent(config_a)
        agent_b = tab.NixlTransferAgent(config_b)

        # Create memory descriptors for all chunks
        src_memory_descs = []
        dst_memory_descs = []
        for src, dst in zip(src_tensors, dst_tensors):
            src_memory_descs.append(
                tab.MemoryDesc(src.data_ptr(), src.numel() * src.element_size(), 0)
            )
            dst_memory_descs.append(
                tab.MemoryDesc(dst.data_ptr(), dst.numel() * dst.element_size(), 0)
            )

        src_descs = tab.MemoryDescs(tab.MemoryType.VRAM, src_memory_descs)
        dst_descs = tab.MemoryDescs(tab.MemoryType.VRAM, dst_memory_descs)

        # Register memory
        agent_a.register_memory(src_descs)
        agent_b.register_memory(dst_descs)

        # Exchange agent info
        agent_a.load_remote_agent("agent_b", agent_b.get_local_agent_desc())
        agent_b.load_remote_agent("agent_a", agent_a.get_local_agent_desc())

        # Transfer
        request = tab.TransferRequest(tab.TransferOp.WRITE, src_descs, dst_descs, "agent_b")
        status = agent_a.submit_transfer_requests(request)
        result = status.wait(timeout_ms=5000)

        assert result == tab.TransferState.SUCCESS

        torch.cuda.synchronize()

        # Verify all chunks
        for i, (src, dst) in enumerate(zip(src_tensors, dst_tensors)):
            assert torch.equal(src, dst), f"Data mismatch in chunk {i}"

        # Cleanup
        agent_a.deregister_memory(src_descs)
        agent_b.deregister_memory(dst_descs)

    def test_nixl_wait_in_progress_on_zero_timeout(self):
        """Test that wait(timeout_ms=0) returns IN_PROGRESS for a large in-flight transfer."""
        device = torch.device("cuda:0")

        # Use a large tensor to maximize chance of catching transfer in-flight
        num_elements = 10_000_000
        src_tensor = torch.arange(num_elements, dtype=torch.float32, device=device)
        dst_tensor = torch.zeros(num_elements, dtype=torch.float32, device=device)

        config_a = tab.BaseAgentConfig(
            name="agent_a", use_prog_thread=True, use_listen_thread=False
        )
        config_b = tab.BaseAgentConfig(
            name="agent_b", use_prog_thread=True, use_listen_thread=False
        )

        agent_a = tab.NixlTransferAgent(config_a)
        agent_b = tab.NixlTransferAgent(config_b)

        src_descs = _create_memory_descs_from_tensor(src_tensor, tab.MemoryType.VRAM)
        dst_descs = _create_memory_descs_from_tensor(dst_tensor, tab.MemoryType.VRAM)

        agent_a.register_memory(src_descs)
        agent_b.register_memory(dst_descs)

        agent_a.load_remote_agent("agent_b", agent_b.get_local_agent_desc())
        agent_b.load_remote_agent("agent_a", agent_a.get_local_agent_desc())

        request = tab.TransferRequest(tab.TransferOp.WRITE, src_descs, dst_descs, "agent_b")
        status = agent_a.submit_transfer_requests(request)

        # With timeout_ms=0, wait checks status once and returns immediately.
        result = status.wait(timeout_ms=0)
        assert result == tab.TransferState.IN_PROGRESS

        # Wait for the transfer to actually finish before cleanup
        final_result = status.wait(timeout_ms=5000)
        assert final_result == tab.TransferState.SUCCESS

        torch.cuda.synchronize()
        assert torch.equal(src_tensor, dst_tensor)

        agent_a.deregister_memory(src_descs)
        agent_b.deregister_memory(dst_descs)

    def test_nixl_wait_failure_on_invalidated_remote(self):
        """Test that submitting a transfer to an invalidated remote agent causes FAILURE."""
        device = torch.device("cuda:0")

        src_tensor = torch.arange(1024, dtype=torch.float32, device=device)
        dst_tensor = torch.zeros(1024, dtype=torch.float32, device=device)

        config_a = tab.BaseAgentConfig(
            name="agent_a", use_prog_thread=True, use_listen_thread=False
        )
        config_b = tab.BaseAgentConfig(
            name="agent_b", use_prog_thread=True, use_listen_thread=False
        )

        agent_a = tab.NixlTransferAgent(config_a)
        agent_b = tab.NixlTransferAgent(config_b)

        src_descs = _create_memory_descs_from_tensor(src_tensor, tab.MemoryType.VRAM)
        dst_descs = _create_memory_descs_from_tensor(dst_tensor, tab.MemoryType.VRAM)

        agent_a.register_memory(src_descs)
        agent_b.register_memory(dst_descs)

        agent_a.load_remote_agent("agent_b", agent_b.get_local_agent_desc())
        agent_b.load_remote_agent("agent_a", agent_a.get_local_agent_desc())

        # Invalidate the remote agent before submitting
        agent_a.invalidate_remote_agent("agent_b")

        request = tab.TransferRequest(tab.TransferOp.WRITE, src_descs, dst_descs, "agent_b")

        # Backend may either raise on submit or return a failed status
        try:
            status = agent_a.submit_transfer_requests(request)
        except Exception:
            pass  # Raising is acceptable behavior
        else:
            # If no exception, the transfer should fail when waited on
            result = status.wait(timeout_ms=5000)
            assert result == tab.TransferState.FAILURE, (
                f"Expected FAILURE after invalidation, got {result}"
            )

        agent_a.deregister_memory(src_descs)
        agent_b.deregister_memory(dst_descs)


@pytest.mark.skipif(
    not (HAS_TORCH and HAS_CUDA),
    reason="Torch with CUDA support required for functional tests",
)
@pytest.mark.skipif(
    not HAS_MOONCAKE_RUNTIME,
    reason="Mooncake runtime libraries not available (libtransfer_engine.so)",
)
class TestMooncakeFunctionalTransfer:
    """Functional tests for Mooncake data transfer between two agents."""

    def test_mooncake_write_transfer_gpu_tensor(self):
        """Test WRITE transfer of GPU tensor data between two Mooncake agents."""
        device = torch.device("cuda:0")

        # Create source tensor with known data pattern
        src_tensor = torch.arange(1024, dtype=torch.float32, device=device)

        # Create destination tensor (zeros)
        dst_tensor = torch.zeros(1024, dtype=torch.float32, device=device)

        # Verify initial state
        assert not torch.equal(src_tensor, dst_tensor)

        # Create two agents via factory (uses dlopen for lazy loading)
        config_a = tab.BaseAgentConfig(name="mooncake_agent_a", use_prog_thread=True)
        config_b = tab.BaseAgentConfig(name="mooncake_agent_b", use_prog_thread=True)
        agent_a = tab.make_transfer_agent("mooncake", config_a)
        agent_b = tab.make_transfer_agent("mooncake", config_b)
        # Register memory regions
        src_descs = _create_memory_descs_from_tensor(src_tensor, tab.MemoryType.VRAM)
        dst_descs = _create_memory_descs_from_tensor(dst_tensor, tab.MemoryType.VRAM)

        agent_a.register_memory(src_descs)
        agent_b.register_memory(dst_descs)
        agent_a_desc = agent_a.get_local_agent_desc()

        agent_b_desc = agent_b.get_local_agent_desc()

        agent_a.load_remote_agent("mooncake_agent_b", agent_b_desc)
        agent_b.load_remote_agent("mooncake_agent_a", agent_a_desc)

        request = tab.TransferRequest(
            tab.TransferOp.WRITE,
            src_descs,  # local source
            dst_descs,  # remote destination
            "mooncake_agent_b",  # remote agent name
        )

        # # Submit transfer and wait for completion
        status = agent_a.submit_transfer_requests(request)

        result = status.wait()
        assert result == tab.TransferState.SUCCESS, f"Transfer failed with state: {result}"

        # Synchronize CUDA to ensure transfer is complete
        torch.cuda.synchronize()

        # Verify data was transferred correctly
        assert torch.equal(src_tensor, dst_tensor), "Data mismatch after transfer"

        # Cleanup
        agent_a.deregister_memory(src_descs)
        agent_b.deregister_memory(dst_descs)

    def test_mooncake_write_transfer_multiple_chunks(self):
        """Test WRITE transfer with multiple memory chunks."""
        device = torch.device("cuda:0")

        # Create multiple source tensors
        src_tensors = [
            torch.arange(i * 256, (i + 1) * 256, dtype=torch.float32, device=device)
            for i in range(4)
        ]

        # Create corresponding destination tensors
        dst_tensors = [torch.zeros(256, dtype=torch.float32, device=device) for _ in range(4)]

        # Create agents via factory (uses dlopen for lazy loading)
        config_a = tab.BaseAgentConfig(name="mooncake_agent_a", use_prog_thread=True)
        config_b = tab.BaseAgentConfig(name="mooncake_agent_b", use_prog_thread=True)

        agent_a = tab.make_transfer_agent("mooncake", config_a)
        agent_b = tab.make_transfer_agent("mooncake", config_b)

        # Create memory descriptors for all chunks
        src_memory_descs = []
        dst_memory_descs = []
        for src, dst in zip(src_tensors, dst_tensors):
            src_memory_descs.append(
                tab.MemoryDesc(src.data_ptr(), src.numel() * src.element_size(), 0)
            )
            dst_memory_descs.append(
                tab.MemoryDesc(dst.data_ptr(), dst.numel() * dst.element_size(), 0)
            )

        src_descs = tab.MemoryDescs(tab.MemoryType.VRAM, src_memory_descs)
        dst_descs = tab.MemoryDescs(tab.MemoryType.VRAM, dst_memory_descs)

        # Register memory
        agent_a.register_memory(src_descs)
        agent_b.register_memory(dst_descs)

        # Exchange agent info
        agent_a.load_remote_agent("mooncake_agent_b", agent_b.get_local_agent_desc())
        agent_b.load_remote_agent("mooncake_agent_a", agent_a.get_local_agent_desc())

        # Transfer
        request = tab.TransferRequest(
            tab.TransferOp.WRITE, src_descs, dst_descs, "mooncake_agent_b"
        )
        status = agent_a.submit_transfer_requests(request)
        result = status.wait(timeout_ms=5000)

        assert result == tab.TransferState.SUCCESS

        torch.cuda.synchronize()

        # Verify all chunks
        for i, (src, dst) in enumerate(zip(src_tensors, dst_tensors)):
            assert torch.equal(src, dst), f"Data mismatch in chunk {i}"

        # Cleanup
        agent_a.deregister_memory(src_descs)
        agent_b.deregister_memory(dst_descs)

    def test_mooncake_wait_in_progress_on_zero_timeout(self):
        """Test that wait(timeout_ms=0) returns IN_PROGRESS for a large in-flight transfer."""
        device = torch.device("cuda:0")

        num_elements = 10_000_000
        src_tensor = torch.arange(num_elements, dtype=torch.float32, device=device)
        dst_tensor = torch.zeros(num_elements, dtype=torch.float32, device=device)

        config_a = tab.BaseAgentConfig(name="mooncake_agent_a", use_prog_thread=True)
        config_b = tab.BaseAgentConfig(name="mooncake_agent_b", use_prog_thread=True)

        agent_a = tab.make_transfer_agent("mooncake", config_a)
        agent_b = tab.make_transfer_agent("mooncake", config_b)

        src_descs = _create_memory_descs_from_tensor(src_tensor, tab.MemoryType.VRAM)
        dst_descs = _create_memory_descs_from_tensor(dst_tensor, tab.MemoryType.VRAM)

        agent_a.register_memory(src_descs)
        agent_b.register_memory(dst_descs)

        agent_a.load_remote_agent("mooncake_agent_b", agent_b.get_local_agent_desc())
        agent_b.load_remote_agent("mooncake_agent_a", agent_a.get_local_agent_desc())

        request = tab.TransferRequest(
            tab.TransferOp.WRITE, src_descs, dst_descs, "mooncake_agent_b"
        )
        status = agent_a.submit_transfer_requests(request)

        result = status.wait(timeout_ms=0)
        assert result == tab.TransferState.IN_PROGRESS

        final_result = status.wait(timeout_ms=5000)
        assert final_result == tab.TransferState.SUCCESS

        torch.cuda.synchronize()
        assert torch.equal(src_tensor, dst_tensor)

        agent_a.deregister_memory(src_descs)
        agent_b.deregister_memory(dst_descs)

    @pytest.mark.skip(reason="Mooncake invalidateRemoteAgent is not yet implemented")
    def test_mooncake_wait_failure_on_invalidated_remote(self):
        """Test that submitting a transfer to an invalidated remote agent causes FAILURE."""
        device = torch.device("cuda:0")

        src_tensor = torch.arange(1024, dtype=torch.float32, device=device)
        dst_tensor = torch.zeros(1024, dtype=torch.float32, device=device)

        config_a = tab.BaseAgentConfig(name="mooncake_agent_a", use_prog_thread=True)
        config_b = tab.BaseAgentConfig(name="mooncake_agent_b", use_prog_thread=True)

        agent_a = tab.make_transfer_agent("mooncake", config_a)
        agent_b = tab.make_transfer_agent("mooncake", config_b)

        src_descs = _create_memory_descs_from_tensor(src_tensor, tab.MemoryType.VRAM)
        dst_descs = _create_memory_descs_from_tensor(dst_tensor, tab.MemoryType.VRAM)

        agent_a.register_memory(src_descs)
        agent_b.register_memory(dst_descs)

        agent_a.load_remote_agent("mooncake_agent_b", agent_b.get_local_agent_desc())
        agent_b.load_remote_agent("mooncake_agent_a", agent_a.get_local_agent_desc())

        agent_a.invalidate_remote_agent("mooncake_agent_b")

        request = tab.TransferRequest(
            tab.TransferOp.WRITE, src_descs, dst_descs, "mooncake_agent_b"
        )

        # Backend may either raise on submit or return a failed status
        try:
            status = agent_a.submit_transfer_requests(request)
        except Exception:
            pass  # Raising is acceptable behavior
        else:
            # If no exception, the transfer should fail when waited on
            result = status.wait(timeout_ms=5000)
            assert result == tab.TransferState.FAILURE, (
                f"Expected FAILURE after invalidation, got {result}"
            )

        agent_a.deregister_memory(src_descs)
        agent_b.deregister_memory(dst_descs)
