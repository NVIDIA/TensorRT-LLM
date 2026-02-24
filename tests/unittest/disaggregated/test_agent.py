from dataclasses import dataclass, field
from unittest import TestCase
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.agent import (
    MemoryDescs,
    MemoryType,
    RegMemoryDescs,
    TransferOp,
    TransferRequest,
    TransferStatus,
)
from tensorrt_llm._torch.disaggregation.nixl.agent import NixlTransferAgent


class TestTransferStatus(TestCase):
    def test_mock_transfer_status(self):
        mock_transfer_status = Mock(spec=TransferStatus)
        mock_transfer_status.is_completed.return_value = True
        self.assertTrue(mock_transfer_status.is_completed())
        mock_transfer_status.is_completed.assert_called_once()
        mock_transfer_status.wait.return_value = True
        timeout_values = [None, 1000, 5000]
        for timeout in timeout_values:
            with self.subTest(timeout=timeout):
                result = mock_transfer_status.wait(timeout_ms=timeout)
                self.assertTrue(result)
                mock_transfer_status.wait.assert_called_with(timeout_ms=timeout)


def _convert_to_memory_descs(reg_descs: RegMemoryDescs) -> MemoryDescs:
    tuples = [(ptr, size, device_id) for (ptr, size, device_id, _) in reg_descs.descs]

    def _convert_memory_type(py_type: str) -> MemoryType:
        """Convert Python memory type string to C++ MemoryType."""
        type_map = {
            "DRAM": MemoryType.DRAM,
            "VRAM": MemoryType.VRAM,
            "GPU": MemoryType.VRAM,
            "BLK": MemoryType.BLK,
            "OBJ": MemoryType.OBJ,
            "FILE": MemoryType.FILE,
        }
        return type_map.get(py_type.upper(), MemoryType.VRAM)

    return MemoryDescs(_convert_memory_type(reg_descs.type), tuples)


@dataclass
class MemoryManager:
    allocated_memory: list[torch.Tensor] = field(default_factory=list)

    def allocate_memory(
        self, size: int, name: str, memory_type=MemoryType.VRAM, device_id: int = 0
    ) -> RegMemoryDescs:
        device = torch.device(f"cuda:{device_id}" if memory_type == MemoryType.VRAM else "cpu")

        # Allocate memory block using torch.Tensor and track it
        block = torch.zeros(size, dtype=torch.uint8, device=device)
        self.allocated_memory.append(block)

        # Return RegMemoryDescs with position arguments
        memory_descs = RegMemoryDescs(
            type=memory_type, descs=[(block.data_ptr(), block.numel(), device_id, name)]
        )
        return memory_descs

    def clear_memory(self):
        """Clear all tracked memory blocks."""
        self.allocated_memory.clear()


@pytest.fixture
def memory_manager():
    return MemoryManager()


@pytest.fixture(params=[256, 512])
def memory_size(request):
    return request.param


@pytest.fixture(params=["DRAM", "VRAM"])
def memory_type(request):
    return request.param


@pytest.fixture
def alloc(memory_manager, memory_size, memory_type):
    """Allocate memory for source and destination, based on the memory_size and memory_type parameters."""
    assert memory_size > 0, "Memory size must be a positive integer."
    if memory_type == "VRAM" and not torch.cuda.is_available():
        pytest.skip("CUDA not available for VRAM transfer tests")
    src_descs = memory_manager.allocate_memory(
        size=memory_size, name="src_mem", memory_type=memory_type
    )
    dst_descs = memory_manager.allocate_memory(
        size=memory_size, name="dst_mem", memory_type=memory_type
    )
    return src_descs, dst_descs


@pytest.fixture
def transfer_agent_src():
    return NixlTransferAgent(name="src_agent")


@pytest.fixture
def transfer_agent_dst():
    return NixlTransferAgent(name="dst_agent")


def test_transfer_between_agents(
    transfer_agent_src,
    transfer_agent_dst,
    memory_manager,
    alloc,
    memory_size,
    memory_type,
):
    """End-to-end test of data transfer between two agents with parameterized memory sizes and types."""
    # Debug log the parameters being tested
    logger.info(f"Testing with memory_size={memory_size}, memory_type={memory_type}")

    # Unpack source and destination memory descriptions
    memory_descs_src, memory_descs_dst = alloc

    # Fill source memory with sequential data for validation
    src_data = memory_manager.allocated_memory[0]
    assert memory_size > 0, "Memory size must be positive."
    tensor = torch.arange(memory_size, dtype=torch.uint8) % 10
    src_data.copy_(tensor)

    # Register memory with source and destination agents
    transfer_agent_src.register_memory(memory_descs_src)
    transfer_agent_dst.register_memory(memory_descs_dst)

    src_agent_desc = transfer_agent_src.get_local_agent_desc()
    transfer_agent_dst.load_remote_agent("src_agent", src_agent_desc)

    dst_agent_desc = transfer_agent_dst.get_local_agent_desc()
    transfer_agent_src.load_remote_agent("dst_agent", dst_agent_desc)

    # Create and submit the transfer request
    transfer_request = TransferRequest(
        op=TransferOp.WRITE,
        src_descs=_convert_to_memory_descs(memory_descs_src),
        dst_descs=_convert_to_memory_descs(memory_descs_dst),
        remote_name="dst_agent",
        sync_message=None,
    )
    transfer_status = transfer_agent_src.submit_transfer_requests(transfer_request)
    assert transfer_status.wait(timeout_ms=5000), "Transfer did not complete within timeout."

    # Validate transfer completion
    assert transfer_status.is_completed(), "Transfer did not complete successfully."

    # Validate that the destination data matches the source data
    dst_data = memory_manager.allocated_memory[1]
    assert torch.equal(dst_data, src_data), "Destination data does not match source data."

    # Clean up by deregistering memory and clearing allocations
    transfer_agent_src.deregister_memory(memory_descs_src)
    transfer_agent_dst.deregister_memory(memory_descs_dst)
    memory_manager.clear_memory()

    transfer_agent_src.invalidate_remote_agent("dst_agent")
    transfer_agent_dst.invalidate_remote_agent("src_agent")


if __name__ == "__main__":
    pytest.main()
