"""
Tests for cp_broadcast functionality in both MPIDist and TorchDist.

This module tests the context parallelism broadcast operation which is used
when CP (context parallelism) is enabled (e.g., in Helix parallelism).

For MPIDist tests, run with mpirun:
mpirun -n 2 python -m pytest tests/unittest/_torch/distributed/test_cp_broadcast.py -v

For TorchDist tests, see test_ops.py which uses Ray for distributed testing.
"""

import numpy as np
import pytest

from tensorrt_llm._torch.distributed import MPIDist
from tensorrt_llm.mapping import Mapping


def get_mpi_info():
    """Get MPI rank and world size, returns (0, 1) if MPI is not available."""
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return comm.Get_rank(), comm.Get_size()
    except ImportError:
        return 0, 1


def skip_if_not_mpi():
    """Skip test if not running under MPI with sufficient ranks."""
    rank, world_size = get_mpi_info()
    if world_size < 2:
        pytest.skip("Test requires at least 2 MPI ranks (run with mpirun -n 2)")


class TestMPIDistCpBroadcast:
    """Tests for MPIDist.cp_broadcast functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up MPI environment and mapping for each test."""
        skip_if_not_mpi()
        self.rank, self.world_size = get_mpi_info()

        # Set up mapping with CP enabled (cp_size = world_size, tp_size = 1)
        self.mapping = Mapping(
            world_size=self.world_size,
            rank=self.rank,
            tp_size=1,
            cp_size=self.world_size,
            pp_size=1,
        )
        self.dist = MPIDist(mapping=self.mapping)

    def test_broadcast_numpy_array(self):
        """Test broadcasting a numpy array via cp_broadcast."""
        root = 0
        shape = (64, 128)

        if self.mapping.cp_rank == root:
            # Root rank creates the data to broadcast
            data = np.random.randn(*shape).astype(np.float32)
        else:
            # Non-root ranks have empty/zero data
            data = np.zeros(shape, dtype=np.float32)

        # Store original data from root for verification
        from mpi4py import MPI

        expected = np.zeros(shape, dtype=np.float32)
        MPI.COMM_WORLD.Bcast(data if self.mapping.cp_rank == root else expected, root=root)
        if self.mapping.cp_rank == root:
            expected = data.copy()

        # Perform cp_broadcast
        result = self.dist.cp_broadcast(data, root=root)

        # Verify all ranks have the same data
        np.testing.assert_array_almost_equal(result, expected)

    def test_broadcast_python_dict(self):
        """Test broadcasting a Python dictionary via cp_broadcast."""
        root = 0

        if self.mapping.cp_rank == root:
            obj = {
                "model_name": "llama",
                "batch_size": 32,
                "tokens": [1, 2, 3, 4, 5],
                "config": {"hidden_size": 4096, "num_layers": 32},
            }
        else:
            obj = None

        result = self.dist.cp_broadcast(obj, root=root)

        # Verify all ranks received the correct object
        assert result["model_name"] == "llama"
        assert result["batch_size"] == 32
        assert result["tokens"] == [1, 2, 3, 4, 5]
        assert result["config"]["hidden_size"] == 4096
        assert result["config"]["num_layers"] == 32

    def test_broadcast_python_list(self):
        """Test broadcasting a Python list via cp_broadcast."""
        root = 0

        if self.mapping.cp_rank == root:
            obj = ["request1", "request2", {"id": 123, "data": [1, 2, 3]}]
        else:
            obj = None

        result = self.dist.cp_broadcast(obj, root=root)

        assert result == ["request1", "request2", {"id": 123, "data": [1, 2, 3]}]

    def test_broadcast_from_non_zero_root(self):
        """Test broadcasting from a non-zero root rank."""
        if self.world_size < 2:
            pytest.skip("Need at least 2 ranks to test non-zero root")

        root = 1  # Broadcast from rank 1

        if self.mapping.cp_rank == root:
            obj = {"source": "rank1", "value": 42}
        else:
            obj = None

        result = self.dist.cp_broadcast(obj, root=root)

        assert result["source"] == "rank1"
        assert result["value"] == 42

    def test_broadcast_large_object(self):
        """Test broadcasting a large object that may require chunking."""
        root = 0
        # Create a large list to test chunking behavior
        large_size = 100000

        if self.mapping.cp_rank == root:
            obj = list(range(large_size))
        else:
            obj = None

        result = self.dist.cp_broadcast(obj, root=root)

        assert len(result) == large_size
        assert result[0] == 0
        assert result[-1] == large_size - 1

    def test_broadcast_string(self):
        """Test broadcasting a simple string via cp_broadcast."""
        root = 0

        if self.mapping.cp_rank == root:
            obj = "Hello from root rank!"
        else:
            obj = None

        result = self.dist.cp_broadcast(obj, root=root)

        assert result == "Hello from root rank!"


# Additional integration-style test that can be run standalone
def test_mpi_cp_broadcast_integration():
    """
    Integration test for MPIDist cp_broadcast.
    """
    rank, world_size = get_mpi_info()
    if world_size < 2:
        pytest.skip("Test requires at least 2 MPI ranks")

    # Create mapping with CP enabled
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        tp_size=1,
        cp_size=world_size,
        pp_size=1,
    )
    dist = MPIDist(mapping=mapping)

    # Test 1: Broadcast dict
    if mapping.cp_rank == 0:
        payload = {"requests": [{"id": i} for i in range(10)]}
    else:
        payload = None

    result = dist.cp_broadcast(payload, root=0)
    assert len(result["requests"]) == 10
    assert result["requests"][0]["id"] == 0

    # Test 2: Broadcast numpy array
    shape = (32, 64)
    if mapping.cp_rank == 0:
        arr = np.ones(shape, dtype=np.float32) * (rank + 1)
    else:
        arr = np.zeros(shape, dtype=np.float32)

    result = dist.cp_broadcast(arr, root=0)
    expected_val = 1.0  # From rank 0
    np.testing.assert_array_almost_equal(result, np.ones(shape) * expected_val)


if __name__ == "__main__":
    # Allow running directly with mpirun
    pytest.main([__file__, "-v"])


class TestMPIDistTpCpBroadcast:
    """Tests for MPIDist.tp_cp_broadcast functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up MPI environment and mapping for each test."""
        skip_if_not_mpi()
        self.rank, self.world_size = get_mpi_info()

        # Set up mapping with both TP and CP enabled
        # For 2 ranks: tp_size=1, cp_size=2 (tp_cp_broadcast will only do cp_broadcast)
        self.mapping = Mapping(
            world_size=self.world_size,
            rank=self.rank,
            tp_size=1,
            cp_size=self.world_size,
            pp_size=1,
        )
        self.dist = MPIDist(mapping=self.mapping)

    def test_tp_cp_broadcast_python_dict(self):
        """Test broadcasting a Python dictionary via tp_cp_broadcast."""
        root = 0

        # Only rank 0 in both TP and CP groups should have the object
        if self.mapping.tp_rank == root and self.mapping.cp_rank == root:
            obj = {
                "model_name": "llama",
                "batch_size": 32,
                "tokens": [1, 2, 3, 4, 5],
            }
        else:
            obj = None

        result = self.dist.tp_cp_broadcast(obj, root=root)

        # Verify all ranks received the correct object
        assert result["model_name"] == "llama"
        assert result["batch_size"] == 32
        assert result["tokens"] == [1, 2, 3, 4, 5]

    def test_tp_cp_broadcast_python_list(self):
        """Test broadcasting a Python list via tp_cp_broadcast."""
        root = 0

        if self.mapping.tp_rank == root and self.mapping.cp_rank == root:
            obj = ["request1", "request2", {"id": 123, "data": [1, 2, 3]}]
        else:
            obj = None

        result = self.dist.tp_cp_broadcast(obj, root=root)

        assert result == ["request1", "request2", {"id": 123, "data": [1, 2, 3]}]
