import unittest

import numpy as np
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import \
    HostMoeTensorSharer

pytestmark = pytest.mark.cpu_only


class TestHostMoeTensorSharer(unittest.TestCase):
    """Tests for HostMoeTensorSharer functionality"""

    def verify_tensor_data(self, tensor_data, expert_id, tensor_shape):
        """Verify tensor data matches the expected pattern for the given expert ID.

        Args:
            tensor_data: The tensor data to verify
            expert_id: The expert ID for which the data was generated
            tensor_shape: Expected shape of the tensor

        Returns:
            (bool, str): Tuple containing (success, error_message)
        """
        try:
            # Verify tensor shape
            if tensor_data.shape != tensor_shape:
                return False, f"Expert {expert_id}'s tensor has incorrect shape: {tensor_data.shape}, expected: {tensor_shape}"

            # Create expected data based on the pattern
            expected_data = np.ones(tensor_shape, dtype=np.float32)
            for i in range(tensor_shape[0]):
                for j in range(tensor_shape[1]):
                    expected_data[i, j] = expert_id * 1000 + i * 100 + j

            # Convert tensor data to numpy for comparison if needed
            numpy_tensor = tensor_data
            if isinstance(tensor_data, torch.Tensor):
                numpy_tensor = tensor_data.cpu().numpy()

            # Check if the data matches the expected pattern
            np.testing.assert_allclose(
                numpy_tensor,
                expected_data,
                rtol=1e-5,
                atol=1e-5,
                err_msg=
                f"Expert {expert_id}'s tensor data does not match expected values"
            )
            return True, ""
        except Exception as e:
            return False, str(e)

    def generate_tensor_data(self, expert_id, tensor_shape):
        """Generate tensor data for the given expert ID and shape.

        This function generates deterministic data based on expert ID and position indices.
        It follows the same pattern used when creating tensors for sharing.

        Args:
            expert_id: The expert ID for which to generate data
            tensor_shape: Shape of the tensor to generate

        Returns:
            torch.Tensor: Generated tensor
        """
        tensor_data = torch.ones(tensor_shape, dtype=torch.float32)
        for i in range(tensor_shape[0]):
            for j in range(tensor_shape[1]):
                tensor_data[i, j] = expert_id * 1000 + i * 100 + j
        return tensor_data

    def test_host_tensor_sharing_basic(self):
        """Basic test for host tensor sharing"""
        # Get MPI communication information
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        layer_id = 0

        # Test tensor parameters
        experts_per_rank = 2  # Each rank is responsible for 2 consecutive experts
        expert_count = size * experts_per_rank
        tensor_shape = (16, 32)  # Use 2D tensor for testing

        # Maximum supported ranks (can adjust as needed)
        max_ranks = 8
        if size > max_ranks:
            self.skipTest(f"This test supports up to {max_ranks} MPI processes")

        # Create shared communicator
        shared_comm = comm.Split_type(split_type=MPI.COMM_TYPE_SHARED)

        # Initialize HostMoeTensorSharer
        sharer = HostMoeTensorSharer(layer_id, expert_count, shared_comm)

        # Set shared memory base name
        shared_memory_base_name = "test_host_sharer"
        sharer.set_shared_memory_base_name(shared_memory_base_name)

        # Calculate the range of experts this rank is responsible for
        start_expert_id = rank * experts_per_rank
        end_expert_id = start_expert_id + experts_per_rank
        my_expert_ids = list(range(start_expert_id, end_expert_id))

        # Initialize empty list to store host tensor shapes
        sharer.host_tensor_shapes = []

        # Dictionary to store all retrieved tensors
        all_tensors = {}

        # Create and share tensors for experts assigned to this rank
        for expert_id in my_expert_ids:
            # Generate deterministic data using our helper function
            tensor_data = self.generate_tensor_data(expert_id, tensor_shape)

            # Share host tensor
            sharer.share_host_tensor_with_shape(expert_id, "weight",
                                                tensor_data)

        # Pre-register tensor shapes for experts handled by other ranks
        for expert_id in range(expert_count):
            if expert_id not in my_expert_ids:
                sharer.pre_register_host_tensor_with_shape(
                    expert_id, "weight", torch.float32, tensor_shape)

        sharer.finalize_layer_weights()

        # Ensure all processes have created and registered their tensors
        comm.Barrier()

        # Define callback function to retrieve all tensors
        def tensor_callback(expert_id, tensor_name, tensor_data):
            key = (expert_id, tensor_name)
            all_tensors[key] = tensor_data
            return True  # Continue processing

        # Finalize host tensor sharing with the callback
        sharer.finalize_host_tensor_sharing(tensor_callback)

        # Ensure finalization is complete across all ranks
        comm.Barrier()

        # Verify all expected tensors were retrieved
        expected_tensor_count = expert_count  # One tensor per expert
        self.assertEqual(
            len(all_tensors), expected_tensor_count,
            f"Expected {expected_tensor_count} tensors, but got {len(all_tensors)}"
        )

        # Track verification failures
        verification_failures = []

        # Check if all expert tensors are correctly shared and stored in all_tensors
        for expert_id in range(expert_count):
            key = (expert_id, "weight")

            # Verify tensor is in the collected tensors dictionary
            self.assertIn(
                key, all_tensors,
                f"Expert {expert_id}'s tensor was not retrieved by callback")

            # Get tensor data from our collected dictionary
            tensor_data = all_tensors[key]
            self.assertIsNotNone(
                tensor_data, f"Retrieved tensor for expert {expert_id} is None")

            # Every rank verifies every tensor's data
            success, error_message = self.verify_tensor_data(
                tensor_data, expert_id, tensor_shape)

            if not success:
                verification_failures.append(
                    f"Rank {rank} verification failure for expert {expert_id}: {error_message}"
                )
                print(
                    f"Rank {rank}: Failed to verify expert {expert_id}'s data: {error_message}"
                )
            else:
                print(
                    f"Rank {rank}: Successfully verified expert {expert_id}'s data"
                )

        # Ensure all ranks see all verification results
        # Gather verification failures from all ranks
        all_failures = comm.allgather(verification_failures)

        # Flatten the list of lists
        all_failures = [
            failure for failure_list in all_failures for failure in failure_list
        ]

        # If there are any failures, display them and fail the test
        if all_failures:
            failure_message = "\n".join(all_failures)
            self.fail(f"Tensor verification failed:\n{failure_message}")

        # Final sync point to ensure all verifications are complete
        comm.Barrier()
        print(
            f"Rank {rank}: All expert data verification completed successfully")

        sharer.pre_shutdown_cleanup()

        # Synchronize before cleanup
        comm.Barrier()

    def test_host_tensor_sharing_3d(self):
        """Share 3D expert tensors, e.g. TRTLLM-Gen BlockMajorK weights."""
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if size > 8:
            self.skipTest("This test supports up to 8 MPI processes")

        experts_per_rank = 2
        expert_count = size * experts_per_rank
        tensor_shape = (4, 8, 64)  # [K / block_k, Mn, block_k]
        numel = 4 * 8 * 64

        def expert_tensor(expert_id):
            return (torch.arange(numel, dtype=torch.float32) +
                    expert_id * numel).reshape(tensor_shape)

        shared_comm = comm.Split_type(split_type=MPI.COMM_TYPE_SHARED)
        sharer = HostMoeTensorSharer(0, expert_count, shared_comm)
        sharer.set_shared_memory_base_name("test_host_sharer_3d")

        my_expert_ids = range(rank * experts_per_rank,
                              (rank + 1) * experts_per_rank)
        for expert_id in range(expert_count):
            if expert_id in my_expert_ids:
                sharer.share_host_tensor_with_shape(expert_id, "weight",
                                                    expert_tensor(expert_id))
            else:
                sharer.pre_register_host_tensor_with_shape(
                    expert_id, "weight", torch.float32,
                    torch.Size(tensor_shape))

        sharer.finalize_layer_weights()
        comm.Barrier()

        received = {}

        def tensor_callback(expert_id, tensor_name, tensor_data):
            received[(expert_id, tensor_name)] = tensor_data
            return True

        sharer.finalize_host_tensor_sharing(tensor_callback)
        comm.Barrier()

        failures = []
        for expert_id in range(expert_count):
            t = received.get((expert_id, "weight"))
            if t is None:
                failures.append(f"rank {rank}: expert {expert_id} missing")
            elif tuple(t.shape) != tensor_shape:
                failures.append(
                    f"rank {rank}: expert {expert_id} shape {tuple(t.shape)}")
            elif not torch.equal(t.cpu(), expert_tensor(expert_id)):
                failures.append(f"rank {rank}: expert {expert_id} data differs")
        all_failures = [f for fs in comm.allgather(failures) for f in fs]
        if all_failures:
            self.fail("\n".join(all_failures))

        comm.Barrier()
        sharer.pre_shutdown_cleanup()
        comm.Barrier()


if __name__ == "__main__":
    # This file should be run with mpirun, for example:
    # mpirun -np 2 python -m unittest tests/unittest/_torch/moe/test_moe_host_sharer.py
    # Run tests using unittest
    unittest.main()
