import os
import unittest
from unittest.mock import MagicMock, patch

import torch
from mpi4py import MPI

from tensorrt_llm._torch.modules.fused_moe.moe_load_balancer import (
    MoeLoadBalancer, MoeLoadBalancerIterContext, SingleLayerMoeLoadBalancer,
    get_moe_load_balancer, moe_load_balancer_add_single_layer)


class TestMoeLoadBalancer(unittest.TestCase):
    """
    Test cases for the MoeLoadBalancer class.
    """

    def setUp(self):
        os.environ["TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK"] = "1"

    @patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer')
    def test_moe_load_balancer_init(self, mock_load_balancer_impl):
        """Test initialization of MoeLoadBalancer."""

        torch.cuda.set_device(0)

        # Setup
        ep_rank = 0
        ep_size = 4
        layer_updates_per_iter = 2

        # Exercise
        balancer = MoeLoadBalancer(ep_rank, ep_size, layer_updates_per_iter)

        # Verify
        mock_load_balancer_impl.assert_called_once_with(ep_rank, ep_size,
                                                        layer_updates_per_iter)
        self.assertEqual(balancer.ep_rank, ep_rank)
        self.assertEqual(balancer.ep_size, ep_size)
        self.assertEqual(balancer.layer_updates_per_iter,
                         layer_updates_per_iter)
        self.assertEqual(balancer.load_balancer_impl,
                         mock_load_balancer_impl.return_value)
        self.assertIsNone(balancer._previous_balancer)

    @patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer')
    def test_moe_load_balancer_add_layer(self, mock_load_balancer_impl):
        """Test adding a layer to MoeLoadBalancer."""

        torch.cuda.set_device(0)

        # Setup
        ep_rank = 0
        ep_size = 4
        layer_updates_per_iter = 2
        expert_count = 8
        top_k = 2
        slot_count_per_rank = 4

        mock_single_layer = MagicMock()
        mock_load_balancer_impl.return_value.add_layer.return_value = mock_single_layer

        # Exercise
        balancer = MoeLoadBalancer(ep_rank, ep_size, layer_updates_per_iter)
        result = balancer.add_layer(expert_count, top_k, slot_count_per_rank)

        # Verify
        mock_load_balancer_impl.return_value.add_layer.assert_called_once_with(
            expert_count, top_k, slot_count_per_rank)
        self.assertIsInstance(result, SingleLayerMoeLoadBalancer)
        self.assertEqual(result.single_layer_load_balancer_impl,
                         mock_single_layer)

    @patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer')
    def test_context_manager(self, mock_load_balancer_impl):
        """Test MoeLoadBalancer as a context manager."""

        torch.cuda.set_device(0)

        # Setup
        ep_rank = 0
        ep_size = 4
        layer_updates_per_iter = 2

        # Exercise & Verify
        # Before entering context
        self.assertIsNone(get_moe_load_balancer())

        with MoeLoadBalancer(ep_rank, ep_size,
                             layer_updates_per_iter) as balancer:
            # Inside context
            self.assertEqual(get_moe_load_balancer(), balancer)

        # After exiting context
        self.assertIsNone(get_moe_load_balancer())

    @patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer')
    def test_nested_context_managers(self, mock_load_balancer_impl):
        """Test nested MoeLoadBalancer context managers."""

        torch.cuda.set_device(0)

        # Setup
        outer_balancer = MoeLoadBalancer(0, 4, 2)
        inner_balancer = MoeLoadBalancer(1, 4, 2)

        # Exercise & Verify
        with outer_balancer:
            self.assertEqual(get_moe_load_balancer(), outer_balancer)

            with inner_balancer:
                self.assertEqual(get_moe_load_balancer(), inner_balancer)

            # After exiting inner context
            self.assertEqual(get_moe_load_balancer(), outer_balancer)

        # After exiting outer context
        self.assertIsNone(get_moe_load_balancer())

    @patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer')
    def test_moe_load_balancer_add_single_layer_with_context(
            self, mock_load_balancer_impl):
        """Test moe_load_balancer_add_single_layer with active context."""

        torch.cuda.set_device(0)

        # Setup
        ep_rank = 0
        ep_size = 4
        layer_updates_per_iter = 2
        expert_count = 8
        top_k = 2
        slot_count_per_rank = 4

        mock_single_layer = MagicMock()
        mock_load_balancer_impl.return_value.add_layer.return_value = mock_single_layer

        # Exercise
        with MoeLoadBalancer(ep_rank, ep_size, layer_updates_per_iter):
            result = moe_load_balancer_add_single_layer(expert_count, top_k,
                                                        slot_count_per_rank)

        # Verify
        mock_load_balancer_impl.return_value.add_layer.assert_called_once_with(
            expert_count, top_k, slot_count_per_rank)
        self.assertIsInstance(result, SingleLayerMoeLoadBalancer)
        self.assertEqual(result.single_layer_load_balancer_impl,
                         mock_single_layer)

    def test_moe_load_balancer_add_single_layer_without_context(self):
        """Test moe_load_balancer_add_single_layer without active context."""

        torch.cuda.set_device(0)

        # Exercise
        result = moe_load_balancer_add_single_layer(8, 2, 4)

        # Verify
        self.assertIsNone(result)

    @patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer')
    def test_exception_in_context(self, mock_load_balancer_impl):
        """Test handling of exceptions inside the context manager."""

        torch.cuda.set_device(0)

        # Setup
        class TestException(Exception):
            pass

        # Exercise & Verify
        try:
            with MoeLoadBalancer(0, 4, 2) as balancer:
                self.assertEqual(get_moe_load_balancer(), balancer)
                raise TestException("Test exception")
        except TestException:
            # Exception should be propagated
            pass
        else:
            self.fail("Expected TestException to be raised")

        # Verify the global state is cleaned up
        self.assertIsNone(get_moe_load_balancer())

    @patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer')
    def test_single_layer_moe_load_balancer_methods(self,
                                                    mock_load_balancer_impl):
        """Test methods of SingleLayerMoeLoadBalancer."""

        torch.cuda.set_device(0)

        # Setup
        mock_single_layer_impl = MagicMock()
        layer = SingleLayerMoeLoadBalancer(mock_single_layer_impl,
                                           MPI.COMM_WORLD,
                                           expert_count=4)

        # Mock out torch.ops.trtllm functions
        with patch('torch.ops.trtllm.moe_load_balance_wait_gpu_stage') as mock_wait, \
             patch('torch.ops.trtllm.moe_load_balance_set_cpu_stage') as mock_set_cpu, \
             patch('torch.ops.trtllm.moe_load_balance_statistic') as mock_statistic, \
             patch('torch.ops.trtllm.moe_load_balance_routing') as mock_route:

            # Exercise - test each method
            # add_weight_slot
            mock_weight = MagicMock()
            layer._add_weight_slot(1, "weight1", mock_weight)
            mock_single_layer_impl.add_single_weight_slot.assert_called_once_with(
                1, "weight1", mock_weight)

            # add_host_weight
            mock_host_weight = MagicMock()
            layer._add_host_weight(2, "weight2", mock_host_weight)
            mock_single_layer_impl.add_single_host_weight.assert_called_once_with(
                2, "weight2", mock_host_weight)

            # set_initial_weight_assignments
            initial_assignments = [0, 1, 2, 3]
            layer.set_initial_weight_assignments(initial_assignments)
            mock_single_layer_impl.set_initial_weight_assignments.assert_called_once_with(
                initial_assignments)

            # wait_for_gpu_stage
            mock_wait.return_value = torch.tensor([1])
            layer.start_wait_gpu_stage()
            layer.done_wait_gpu_stage()
            result = layer.statistic_flag_tensor
            mock_wait.assert_called_once_with(
                mock_single_layer_impl.get_pointer())
            self.assertEqual(result, mock_wait.return_value)

            # statistic
            mock_expert_ids = torch.tensor([[0, 1], [2, 3]])
            mock_enabled = torch.tensor([1])
            layer.statistic_flag_tensor = mock_enabled
            layer.update_statistic_with_global_ids(mock_expert_ids, True, False)
            mock_statistic.assert_called_once_with(
                mock_expert_ids, mock_enabled,
                mock_single_layer_impl.get_pointer(), True, False)

            # route
            mock_selected_experts = torch.tensor([[0, 1], [2, 3]])
            mock_route.return_value = torch.tensor([[0, 1], [2, 3]])
            result = layer.route(mock_selected_experts)
            assert torch.equal(result, mock_route.return_value)

            # set_cpu_stage
            layer.start_set_cpu_stage()
            layer.done_set_cpu_stage()
            mock_set_cpu.assert_called_once_with(
                mock_single_layer_impl.get_pointer())

    @patch('tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer')
    def test_moe_load_balancer_lifecycle_methods(self, mock_load_balancer_impl):
        """Test lifecycle methods of MoeLoadBalancer."""

        torch.cuda.set_device(0)

        # Setup
        balancer = MoeLoadBalancer(0, 4, 2)

        # Exercise - test each method
        # finalize_model
        balancer.finalize_model()
        mock_load_balancer_impl.return_value.finalize_model.assert_called_once()

        # set_warm_up_iter_count
        balancer.set_warm_up_iter_count(10)
        mock_load_balancer_impl.return_value.set_warm_up_iter_count.assert_called_once_with(
            10)

        balancer.set_iter_info(True, True)

        with MoeLoadBalancerIterContext(balancer):
            mock_load_balancer_impl.return_value.start_iter.assert_called_once_with(
                0, True, True)

        mock_load_balancer_impl.return_value.end_iter.assert_called_once_with(0)

        # shutdown
        balancer.shutdown()
        mock_load_balancer_impl.return_value.shutdown.assert_called_once()

    def test_real_statistic_kernel(self):
        """Test the real statistic kernel functionality."""

        torch.cuda.set_device(0)

        # Setup parameters
        ep_rank = 0
        ep_size = 2  # Small number of ranks for testing
        expert_count = 4
        top_k = 2
        slots_per_rank = 2  # Each rank has 2 slots, total 4 slots for 4 experts

        # Create a real MoeLoadBalancer
        balancer = MoeLoadBalancer(ep_rank, ep_size, 1)

        balancer.set_use_gpu_memcpy(True)

        # Add a layer with initial weight assignments
        # Each slot is assigned to exactly one expert initially
        layer = balancer.add_layer(expert_count, top_k, slots_per_rank)
        initial_assignments = [0, 1, 2, 3]  # Expert i is assigned to slot i
        layer.set_initial_weight_assignments(initial_assignments)

        # Finalize the model
        balancer.finalize_model()

        # enable statistic, disable weight update
        balancer.set_iter_info(True, False)

        # Create sample token data - each token selects 2 experts
        # 4 tokens, each selecting 2 experts
        gathered_raw_expert_ids = torch.tensor(
            [
                [0, 1],  # Token 0 selects experts 0 and 1
                [1, 2],  # Token 1 selects experts 1 and 2
                [2, 3],  # Token 2 selects experts 2 and 3
                [0, 3]  # Token 3 selects experts 0 and 3
            ],
            dtype=torch.int32,
            device="cuda")

        try:
            with MoeLoadBalancerIterContext(balancer):
                # Wait for GPU stage and get enabled flag
                layer.start_wait_gpu_stage()
                layer.done_wait_gpu_stage()

                # Run statistic - just test it runs without error
                layer.update_statistic_with_global_ids(gathered_raw_expert_ids,
                                                       True, True)

                # Set CPU stage to signal completion
                layer.start_set_cpu_stage()
                layer.done_set_cpu_stage()

            # Test passed if we got here without exceptions
            self.assertTrue(True, "Statistic kernel ran successfully")

        except Exception as e:
            self.fail(f"Statistic kernel test failed with exception: {e}")
        finally:
            # Clean up
            balancer.shutdown()

    def test_real_routing_kernel(self):
        """Test the real routing kernel functionality and verify results."""

        torch.cuda.set_device(0)

        # Setup parameters
        ep_rank = 0
        ep_size = 2
        expert_count = 4
        top_k = 2
        slots_per_rank = 2  # Total 4 slots (2 slots per rank)

        # Create a real MoeLoadBalancer
        balancer = MoeLoadBalancer(ep_rank, ep_size, 1)

        balancer.set_use_gpu_memcpy(True)

        # Add a layer with known initial weight assignments
        layer = balancer.add_layer(expert_count, top_k, slots_per_rank)

        # Set initial assignments: expert i is on slot i
        initial_assignments = [0, 1, 2, 3]
        layer.set_initial_weight_assignments(initial_assignments)

        # Finalize the model
        balancer.finalize_model()

        # enable statistic, disable weight update
        balancer.set_iter_info(True, False)

        # Create sample token data - tokens selecting different experts
        token_selected_experts = torch.tensor(
            [
                [0, 1],  # Token 0 selects experts 0 and 1
                [1, 2],  # Token 1 selects experts 1 and 2
                [2, 3],  # Token 2 selects experts 2 and 3
                [0, 3]  # Token 3 selects experts 0 and 3
            ],
            dtype=torch.int32,
            device="cuda")

        try:
            with MoeLoadBalancerIterContext(balancer):
                # Wait for GPU stage
                layer.start_wait_gpu_stage()
                layer.done_wait_gpu_stage()

                # Run routing
                routed_slots = layer.route(token_selected_experts)

                # Set CPU stage
                layer.start_set_cpu_stage()
                layer.done_set_cpu_stage()

            # Verify results - with our initial assignment, expert i should map to slot i
            expected_slots = torch.tensor(
                [
                    [0, 1],  # Experts 0,1 -> Slots 0,1
                    [1, 2],  # Experts 1,2 -> Slots 1,2
                    [2, 3],  # Experts 2,3 -> Slots 2,3
                    [0, 3]  # Experts 0,3 -> Slots 0,3
                ],
                dtype=torch.int32,
                device="cuda")

            self.assertTrue(
                torch.all(torch.eq(routed_slots, expected_slots)),
                f"Routing results don't match expected slots.\nExpected: {expected_slots}\nGot: {routed_slots}"
            )

        except Exception as e:
            self.fail(f"Routing kernel test failed with exception: {e}")
        finally:
            # Clean up
            balancer.shutdown()


if __name__ == '__main__':
    unittest.main()
