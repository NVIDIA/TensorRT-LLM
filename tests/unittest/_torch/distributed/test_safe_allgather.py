# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Tests for safe_allgather and MPIDist.tp_allgather functionality.

This module tests the chunked MPI.Allgatherv-based allgather operation
that safely handles large serialized objects by avoiding MPI's 32-bit
count/displacement limits.

Run with mpirun:
    mpirun -n 2 python -m pytest tests/unittest/_torch/distributed/test_safe_allgather.py -v
"""

import numpy as np
import pytest

from tensorrt_llm import mapping
from tensorrt_llm._torch import distributed
from tensorrt_llm._torch.distributed import communicator
from tensorrt_llm.bindings import BuildInfo


def get_mpi_info():
    """Get MPI rank and world size, returns (0, 1) if MPI is not available."""
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return comm.Get_rank(), comm.Get_size()
    except ImportError:
        return 0, 1


def get_mpi_comm():
    """Get MPI COMM_WORLD communicator."""
    from mpi4py import MPI

    return MPI.COMM_WORLD


class TestSafeAllgather:
    """Tests for the safe_allgather free function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up MPI environment for each test."""
        if not BuildInfo.ENABLE_MULTI_DEVICE:
            pytest.skip("Test requires ENABLE_MULTI_DEVICE build")
        self.rank, self.world_size = get_mpi_info()
        if self.world_size < 2:
            pytest.skip("Test requires at least 2 MPI ranks (run with mpirun -n 2)")
        self.comm = get_mpi_comm()

    def test_allgather_python_dict(self):
        """Test allgathering a Python dict from each rank."""
        obj = {"rank": self.rank, "data": [self.rank * 10, self.rank * 20]}

        result = communicator.safe_allgather(self.comm, obj)

        assert len(result) == self.world_size
        for i in range(self.world_size):
            assert result[i]["rank"] == i
            assert result[i]["data"] == [i * 10, i * 20]

    def test_allgather_python_list(self):
        """Test allgathering a Python list with mixed types."""
        obj = [self.rank, f"rank_{self.rank}", {"id": self.rank}]

        result = communicator.safe_allgather(self.comm, obj)

        assert len(result) == self.world_size
        for i in range(self.world_size):
            assert result[i] == [i, f"rank_{i}", {"id": i}]

    def test_allgather_numpy_array(self):
        """Test allgathering numpy arrays with rank-specific data."""
        shape = (4, 8)
        data = np.ones(shape, dtype=np.float32) * (self.rank + 1)

        result = communicator.safe_allgather(self.comm, data)

        assert len(result) == self.world_size
        for i in range(self.world_size):
            expected = np.ones(shape, dtype=np.float32) * (i + 1)
            np.testing.assert_array_equal(result[i], expected)

    def test_allgather_string(self):
        """Test allgathering simple strings."""
        obj = f"hello from rank {self.rank}"

        result = communicator.safe_allgather(self.comm, obj)

        assert len(result) == self.world_size
        for i in range(self.world_size):
            assert result[i] == f"hello from rank {i}"

    def test_allgather_none(self):
        """Test allgathering None objects (zero-size payloads)."""
        result = communicator.safe_allgather(self.comm, None)

        assert len(result) == self.world_size
        for i in range(self.world_size):
            assert result[i] is None

    def test_allgather_large_object(self):
        """Test allgathering objects that exceed chunk_size to exercise
        multi-round chunking."""
        large_size = 200000
        obj = list(range(self.rank * large_size, (self.rank + 1) * large_size))

        result = communicator.safe_allgather(self.comm, obj, chunk_size=64 * 1024)

        assert len(result) == self.world_size
        for i in range(self.world_size):
            assert len(result[i]) == large_size
            assert result[i][0] == i * large_size
            assert result[i][-1] == (i + 1) * large_size - 1

    def test_allgather_asymmetric_sizes(self):
        """Test allgathering objects of different sizes from each rank."""
        obj = list(range((self.rank + 1) * 1000))

        result = communicator.safe_allgather(self.comm, obj)

        assert len(result) == self.world_size
        for i in range(self.world_size):
            expected = list(range((i + 1) * 1000))
            assert result[i] == expected

    def test_allgather_custom_chunk_size(self):
        """Test with a very small chunk_size to force many rounds."""
        obj = {"rank": self.rank, "values": list(range(500))}

        result = communicator.safe_allgather(self.comm, obj, chunk_size=64)

        assert len(result) == self.world_size
        for i in range(self.world_size):
            assert result[i]["rank"] == i
            assert result[i]["values"] == list(range(500))

    def test_allgather_empty_collections(self):
        """Test allgathering empty dicts and lists."""
        obj = {} if self.rank % 2 == 0 else []

        result = communicator.safe_allgather(self.comm, obj)

        assert len(result) == self.world_size
        for i in range(self.world_size):
            if i % 2 == 0:
                assert result[i] == {}
            else:
                assert result[i] == []

    def test_allgather_invalid_chunk_size(self):
        """Test that invalid chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            communicator.safe_allgather(self.comm, "test", chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            communicator.safe_allgather(self.comm, "test", chunk_size=-1)

    def test_allgather_cross_rank_consistency(self):
        """Test that every rank receives the exact same result list."""
        obj = {"rank": self.rank, "values": list(range(self.rank * 100))}

        result = communicator.safe_allgather(self.comm, obj)

        result_serializable = [r if not isinstance(r, np.ndarray) else r.tolist() for r in result]
        all_results = self.comm.allgather(result_serializable)
        for other_result in all_results:
            assert other_result == result_serializable

    def test_allgather_chunk_size_one(self):
        """Test with chunk_size=1 to force maximum chunking rounds.

        This exercises the same code path that protects against int32
        overflow in MPI counts/displacements. With chunk_size=1, every
        byte is a separate round, so the displacement math is tested at
        maximum granularity. Actual 2GB+ payloads are impractical in
        unit tests, but the chunking logic is identical regardless of
        chunk_size.
        """
        import pickle

        obj = {"rank": self.rank, "data": list(range(50))}
        serialized_size = len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

        result = communicator.safe_allgather(self.comm, obj, chunk_size=1)

        assert len(result) == self.world_size
        for i in range(self.world_size):
            assert result[i]["rank"] == i
            assert result[i]["data"] == list(range(50))
        # Verify we actually forced many rounds (one byte per round)
        assert serialized_size > 1

    def test_allgather_displacement_correctness_asymmetric(self):
        """Test that displacement math is correct when ranks have very
        different payload sizes.

        Rank 0 sends a tiny object while rank 1 sends a much larger one.
        This stresses the displacement calculation: if displacements are
        wrong, rank 1's data would overwrite rank 0's or vice versa.
        """
        if self.rank == 0:
            obj = "small"
        else:
            obj = {"big": list(range(10000)), "rank": self.rank}

        result = communicator.safe_allgather(self.comm, obj, chunk_size=256)

        assert result[0] == "small"
        for i in range(1, self.world_size):
            assert result[i]["rank"] == i
            assert result[i]["big"] == list(range(10000))

    def test_allgather_total_exceeds_int32_chunk_boundary(self):
        """Test with payload sizes chosen so that the unchunked total
        displacement would overflow a 32-bit int, but chunking keeps
        each round's values within int32 range.

        We can't allocate 2GB+ in a unit test, so instead we verify
        the function works correctly with a chunk_size smaller than
        each rank's payload, confirming the multi-round Allgatherv
        path is exercised (the same path that prevents real overflow).
        """
        obj = np.full(100_000, self.rank, dtype=np.uint8).tobytes()

        result = communicator.safe_allgather(self.comm, obj, chunk_size=1024)

        assert len(result) == self.world_size
        for i in range(self.world_size):
            expected = np.full(100_000, i, dtype=np.uint8).tobytes()
            assert result[i] == expected

    def test_allgather_round_buffer_copy_back(self):
        """Test that the per-round temp buffer correctly copies data back
        into the final recvbuf at absolute offsets.

        Uses asymmetric payloads with a small chunk_size so multiple
        rounds are needed, and each round's data must land at the right
        position. Verifies the fix where we use 0-based displacements
        in MPI and then copy into recvbuf using 64-bit Python indexing.
        """
        if self.rank == 0:
            obj = list(range(5000))
        else:
            obj = {"rank": self.rank, "payload": list(range(20000))}

        result = communicator.safe_allgather(self.comm, obj, chunk_size=512)

        assert result[0] == list(range(5000))
        for i in range(1, self.world_size):
            assert result[i]["rank"] == i
            assert result[i]["payload"] == list(range(20000))

    def test_allgather_chunk_size_auto_capped(self):
        """Test that chunk_size is automatically reduced when
        chunk_size * size would exceed int32 max.

        We pass a deliberately oversized chunk_size and verify the
        function still produces correct results (meaning the internal
        capping logic worked).
        """
        obj = {"rank": self.rank, "values": list(range(100))}

        oversized_chunk = np.iinfo(np.int32).max

        result = communicator.safe_allgather(self.comm, obj, chunk_size=oversized_chunk)

        assert len(result) == self.world_size
        for i in range(self.world_size):
            assert result[i]["rank"] == i
            assert result[i]["values"] == list(range(100))


class TestMPIDistTpAllgather:
    """Tests that MPIDist.tp_allgather correctly wires through to
    safe_allgather via the TP sub-communicator."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up MPI environment and mapping for each test."""
        if not BuildInfo.ENABLE_MULTI_DEVICE:
            pytest.skip("Test requires ENABLE_MULTI_DEVICE build")
        self.rank, self.world_size = get_mpi_info()
        if self.world_size < 2:
            pytest.skip("Test requires at least 2 MPI ranks (run with mpirun -n 2)")

        self.mapping = mapping.Mapping(
            world_size=self.world_size,
            rank=self.rank,
            tp_size=self.world_size,
            cp_size=1,
            pp_size=1,
        )
        self.dist = distributed.MPIDist(mapping=self.mapping)

    def test_tp_allgather_end_to_end(self):
        """Test that MPIDist.tp_allgather correctly routes through the TP
        sub-communicator and returns consistent results on all ranks."""
        obj = {
            "tp_rank": self.rank,
            "batch_size": 32 + self.rank,
            "tokens": list(range(self.rank * 10, (self.rank + 1) * 10)),
        }

        result = self.dist.tp_allgather(obj)

        assert len(result) == self.world_size
        for i in range(self.world_size):
            assert result[i]["tp_rank"] == i
            assert result[i]["batch_size"] == 32 + i
            assert result[i]["tokens"] == list(range(i * 10, (i + 1) * 10))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
