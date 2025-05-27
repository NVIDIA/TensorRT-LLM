#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest

import numpy as np
import torch

import tensorrt_llm.bindings.internal.runtime as _tbr


class TestMoePythonBindings(unittest.TestCase):

    def setUp(self):
        torch.cuda.set_device(0)
        # Common test parameters
        self.expert_count = 8
        self.top_k = 2
        self.ep_rank = 0  # Local rank
        self.ep_size = 2  # Total ranks
        self.slot_count_per_rank = 4
        self.layer_updates_per_iter = 1
        self.gathered_raw_expert_ids = torch.tensor(
            [
                [0, 1],  # Token 0 selects experts 0 and 1
                [1, 2],  # Token 1 selects experts 1 and 2
                [2, 3],  # Token 2 selects experts 2 and 3
                [0, 3]  # Token 3 selects experts 0 and 3
            ],
            dtype=torch.int32,
            device="cuda")

    def test_moe_weight_struct(self):
        """Test the Python binding of MoeWeight structure"""
        # Create a MoeWeight instance
        weight = _tbr.MoeWeight()

        # Verify structure field access
        weight.weight_ptr = 1024  # use a dummy value for pointer
        weight.height = 10
        weight.width = 1024
        weight.pitch = 1024

        # Verify __repr__ method
        repr_str = str(weight)
        self.assertIn("height=10", repr_str)
        self.assertIn("width=1024", repr_str)
        self.assertIn("pitch=1024", repr_str)

    def test_moe_load_balancer_creation(self):
        """Test MoeLoadBalancer creation"""
        # Create MoeLoadBalancer instance
        balancer = _tbr.MoeLoadBalancer(
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
            layer_updates_per_iter=self.layer_updates_per_iter)
        self.assertIsNotNone(balancer)

    def test_add_layer(self):
        """Test adding a layer to MoeLoadBalancer"""
        balancer = _tbr.MoeLoadBalancer(
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
            layer_updates_per_iter=self.layer_updates_per_iter)

        # Add a layer and verify return value type
        layer0 = balancer.add_layer(
            expert_count=self.expert_count,
            top_k=self.top_k,
            slot_count_per_rank=self.slot_count_per_rank)

        self.assertIsInstance(layer0, _tbr.SingleLayerMoeLoadBalancer)
        self.assertEqual(layer0.get_layer_id(), 0)

        # Add another layer and verify return value type
        layer1 = balancer.add_layer(
            expert_count=self.expert_count,
            top_k=self.top_k,
            slot_count_per_rank=self.slot_count_per_rank)
        self.assertIsInstance(layer1, _tbr.SingleLayerMoeLoadBalancer)
        self.assertEqual(layer1.get_layer_id(), 1)

    def _create_weight_buffers(self, weight_height, weight_width, num_experts,
                               num_slots):
        """Create weight buffers for testing"""
        # Create CPU weights
        host_weights = []
        for i in range(num_experts):
            # Create a unique weight for each expert
            host_data = np.ones(
                (weight_height, weight_width), dtype=np.float32) * (i + 1)
            host_buffer = torch.tensor(host_data,
                                       dtype=torch.float32).contiguous()
            host_weight = _tbr.MoeWeight()
            host_weight.weight_ptr = host_buffer.data_ptr()
            host_weight.height = weight_height
            host_weight.width = weight_width * 4  # float32 = 4 bytes
            host_weight.pitch = weight_width * 4  # pitch = width (contiguous memory)
            host_weights.append((host_buffer, host_weight))

        # Create GPU weight slots
        slot_weights = []
        for i in range(num_slots):
            # Create a GPU buffer for each slot
            cuda_buffer = torch.zeros((weight_height, weight_width),
                                      dtype=torch.float32,
                                      device='cuda').contiguous()
            gpu_weight = _tbr.MoeWeight()
            gpu_weight.weight_ptr = cuda_buffer.data_ptr()
            gpu_weight.height = weight_height
            gpu_weight.width = weight_width * 4  # float32 = 4 bytes
            gpu_weight.pitch = weight_width * 4  # pitch = width (contiguous memory)
            slot_weights.append((cuda_buffer, gpu_weight))

        return host_weights, slot_weights

    def test_single_layer_moe_load_balancer_operations(self):
        """Test operations of SingleLayerMoeLoadBalancer"""
        # Create MoeLoadBalancer instance
        balancer = _tbr.MoeLoadBalancer(
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
            layer_updates_per_iter=self.layer_updates_per_iter)

        # Add a layer
        layer = balancer.add_layer(expert_count=self.expert_count,
                                   top_k=self.top_k,
                                   slot_count_per_rank=self.slot_count_per_rank)

        # Create weight buffers for testing
        weight_height = 10
        weight_width = 1024
        host_weights, slot_weights = self._create_weight_buffers(
            weight_height, weight_width, self.expert_count,
            self.slot_count_per_rank)

        # Add weight slots
        for slot_id, (_, slot_weight) in enumerate(slot_weights):
            layer.add_single_weight_slot(slot_id, "test_weight", slot_weight)

        # Add host weights
        for expert_id, (_, host_weight) in enumerate(host_weights):
            layer.add_single_host_weight(expert_id, "test_weight", host_weight)

        # Create initial weight assignments
        initial_assignments = []
        for r in range(self.ep_size):
            expert_start = r * self.expert_count // self.ep_size
            for slot_id in range(self.slot_count_per_rank):
                expert_id = (expert_start + slot_id) % self.expert_count
                initial_assignments.append(expert_id)

        # Set initial weight assignments
        layer.set_initial_weight_assignments(initial_assignments)

        # Finalize model setup
        balancer.finalize_model()

        # Run one iteration
        balancer.set_warm_up_iter_count(1)
        balancer.start_iter(
            0, True, True)  # Iteration 0, enable statistics, enable updates

        enabled = torch.ops.trtllm.moe_load_balance_wait_gpu_stage(
            layer.get_pointer())
        torch.ops.trtllm.moe_load_balance_statistic(
            self.gathered_raw_expert_ids, enabled, layer.get_pointer(), True,
            True)
        torch.ops.trtllm.moe_load_balance_set_cpu_stage(layer.get_pointer())

        balancer.end_iter(0)

        # Run a second iteration
        balancer.start_iter(
            1, True, True)  # Iteration 1, enable statistics, enable updates
        enabled = torch.ops.trtllm.moe_load_balance_wait_gpu_stage(
            layer.get_pointer())
        torch.ops.trtllm.moe_load_balance_statistic(
            self.gathered_raw_expert_ids, enabled, layer.get_pointer(), True,
            True)
        torch.ops.trtllm.moe_load_balance_set_cpu_stage(layer.get_pointer())
        balancer.end_iter(1)

        # Shutdown the load balancer
        balancer.shutdown()

    def test_moe_load_balancer_multiple_layers(self):
        """Test MoeLoadBalancer with multiple layers"""
        # Create MoeLoadBalancer instance
        balancer = _tbr.MoeLoadBalancer(
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
            layer_updates_per_iter=self.layer_updates_per_iter)

        # Create initial weight assignments
        initial_assignments = []
        for r in range(self.ep_size):
            expert_start = r * self.expert_count // self.ep_size
            for slot_id in range(self.slot_count_per_rank):
                expert_id = (expert_start + slot_id) % self.expert_count
                initial_assignments.append(expert_id)

        # Add multiple layers
        num_layers = 3
        layers = []
        for _ in range(num_layers):
            layer = balancer.add_layer(
                expert_count=self.expert_count,
                top_k=self.top_k,
                slot_count_per_rank=self.slot_count_per_rank)
            layers.append(layer)
            layer.set_initial_weight_assignments(initial_assignments)

        # Verify that we got multiple different layers
        self.assertEqual(len(layers), num_layers)
        for i in range(num_layers):
            for j in range(i + 1, num_layers):
                self.assertIsNot(
                    layers[i], layers[j],
                    f"Layer {i} and {j} should be different objects")

        # Finalize model setup
        balancer.finalize_model()

        # Set warm-up iteration count
        balancer.set_warm_up_iter_count(2)

        # Run several iterations
        configs = [
            (0, True,
             True),  # Iteration 0: Enable statistics, enable updates (warm-up)
            (1, True,
             True),  # Iteration 1: Enable statistics, enable updates (warm-up)
            (2, True, True),  # Iteration 2: Enable statistics, enable updates
            (3, True, False),  # Iteration 3: Enable statistics, disable updates
            (4, False,
             False),  # Iteration 4: Disable statistics, disable updates
        ]

        for iter_id, enable_statistic, enable_update_weights in configs:
            balancer.start_iter(iter_id, enable_statistic,
                                enable_update_weights)
            for layer in layers:
                enabled = torch.ops.trtllm.moe_load_balance_wait_gpu_stage(
                    layer.get_pointer())
                torch.ops.trtllm.moe_load_balance_statistic(
                    self.gathered_raw_expert_ids, enabled, layer.get_pointer(),
                    enable_statistic, enable_update_weights)
                torch.ops.trtllm.moe_load_balance_set_cpu_stage(
                    layer.get_pointer())
            balancer.end_iter(iter_id)

        # Shutdown the load balancer
        balancer.shutdown()


if __name__ == '__main__':
    unittest.main()
