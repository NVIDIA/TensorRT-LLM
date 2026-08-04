import os
from dataclasses import dataclass, field
from unittest import TestCase
from unittest.mock import Mock, patch

import pytest
import torch

# Exclude IB (no fabric) and gdr_copy (UCX rcache SIGABRT at teardown).
os.environ.setdefault("UCX_TLS", "^ib,gdr_copy")

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

try:
    from tensorrt_llm._torch.disaggregation.nixl._agent_cpp import (
        BindingsNixlTransferAgent,
        BindingsNixlTransferStatus,
    )
    from tensorrt_llm.tensorrt_llm_transfer_agent_binding import TransferState

    _HAS_CPP_NIXL_BINDING = True
except Exception:  # pragma: no cover - binding unavailable in some envs
    _HAS_CPP_NIXL_BINDING = False

_AGENT_CPP_MODULE = "tensorrt_llm._torch.disaggregation.nixl._agent_cpp"


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
        self, size: int, name: str, memory_type: str = "VRAM", device_id: int = 0
    ) -> RegMemoryDescs:
        # `memory_type` is the string used by RegMemoryDescs (see base/agent.py:81);
        # do not compare against `MemoryType.VRAM`, which is overridden to a C++ enum
        # when the C++ binding is available.
        device = torch.device(f"cuda:{device_id}" if memory_type == "VRAM" else "cpu")

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


@pytest.mark.skipif(not _HAS_CPP_NIXL_BINDING, reason="nixl C++ transfer-agent binding unavailable")
class TestBindingsNixlTransferStatus(TestCase):
    """Cover BindingsNixlTransferStatus (#14137) with a mocked cpp_status.

    Verifies wait()/is_completed()/last_status()/last_status_str() without a
    GPU or a real NIXL agent.
    """

    def test_wait_success_returns_true(self):
        cpp = Mock()
        cpp.wait.return_value = TransferState.SUCCESS
        status = BindingsNixlTransferStatus(cpp, agent_name="testAgent")
        self.assertTrue(status.wait(timeout_ms=5000))
        cpp.wait.assert_called_once_with(5000)

    def test_wait_none_timeout_maps_to_minus_one(self):
        cpp = Mock()
        cpp.wait.return_value = TransferState.SUCCESS
        status = BindingsNixlTransferStatus(cpp, agent_name="a")
        self.assertTrue(status.wait())
        cpp.wait.assert_called_once_with(-1)

    def test_wait_failure_returns_false_and_logs(self):
        cpp = Mock()
        cpp.wait.return_value = TransferState.FAILURE
        status = BindingsNixlTransferStatus(cpp, agent_name="testAgent")
        with patch(f"{_AGENT_CPP_MODULE}.logger") as mlog:
            self.assertFalse(status.wait())
        mlog.error.assert_called_once()
        msg = mlog.error.call_args.args[0]
        self.assertIn("non-SUCCESS", msg)
        self.assertIn("testAgent", msg)

    def test_is_completed_passthrough(self):
        cpp = Mock()
        cpp.is_completed.return_value = True
        self.assertTrue(BindingsNixlTransferStatus(cpp).is_completed())

    def test_last_status_passthrough(self):
        cpp = Mock()
        cpp.get_last_status.return_value = 7
        cpp.get_last_status_str.return_value = "NIXL_ERR_INVALID_PARAM"
        status = BindingsNixlTransferStatus(cpp)
        self.assertEqual(status.last_status(), 7)
        self.assertEqual(status.last_status_str(), "NIXL_ERR_INVALID_PARAM")

    def test_last_status_unavailable_fallback(self):
        # cpp_status without get_last_status*/-> graceful sentinels, no raise.
        cpp = Mock(spec=[])
        status = BindingsNixlTransferStatus(cpp)
        self.assertEqual(status.last_status(), -1)
        self.assertEqual(status.last_status_str(), "<unavailable>")


@pytest.mark.skipif(not _HAS_CPP_NIXL_BINDING, reason="nixl C++ transfer-agent binding unavailable")
class TestBindingsNixlTransferAgentShutdown(TestCase):
    """Cover BindingsNixlTransferAgent.shutdown() (#14137) idempotency.

    shutdown() nulls _cpp_agent FIRST, so a second/re-entrant call is a no-op
    and submit-after-shutdown does not reach a torn-down agent. Tested via
    cls.__new__ + a mocked _cpp_agent to bypass the real-agent __init__.
    """

    @staticmethod
    def _agent_with_mock_cpp():
        agent = BindingsNixlTransferAgent.__new__(BindingsNixlTransferAgent)
        agent._cpp_agent = Mock()
        agent.name = "testAgent"
        return agent

    def test_shutdown_is_idempotent(self):
        agent = self._agent_with_mock_cpp()
        cpp = agent._cpp_agent
        agent.shutdown()
        agent.shutdown()  # _cpp_agent is None now -> early return, no double shutdown
        cpp.shutdown.assert_called_once()
        self.assertIsNone(agent._cpp_agent)

    def test_shutdown_without_init_is_noop(self):
        agent = BindingsNixlTransferAgent.__new__(BindingsNixlTransferAgent)  # never set _cpp_agent
        agent.shutdown()  # getattr(..., None) -> None -> return, must not raise

    def test_submit_after_shutdown_raises(self):
        agent = self._agent_with_mock_cpp()
        agent.shutdown()
        with self.assertRaises(Exception):
            agent.submit_transfer_requests(Mock())


if __name__ == "__main__":
    pytest.main()
