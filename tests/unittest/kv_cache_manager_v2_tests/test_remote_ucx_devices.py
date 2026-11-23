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
"""remote_cache_config.transport.ucx_devices: the {GPU} token and the NIXL backend parameter. No GPU needed."""

import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache import remote_donor
from tensorrt_llm._torch.pyexecutor.kv_cache.remote_donor import (
    make_donor,
    nixl_ucx_backend_params,
    resolve_ucx_devices,
)


def _visible(value=None):
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    if value is not None:
        env["CUDA_VISIBLE_DEVICES"] = value
    return mock.patch.dict(os.environ, env, clear=True)


@pytest.mark.cpu_only
class TestRemoteUcxDevices(unittest.TestCase):
    def test_unset_and_literal_values_pass_through(self):
        self.assertIsNone(resolve_ucx_devices(None, device=0))
        self.assertEqual(
            resolve_ucx_devices("rdma_vf_rail0:1,rdma_vf_rail2:1", device=3),
            "rdma_vf_rail0:1,rdma_vf_rail2:1",
        )

    def test_gpu_token_maps_the_device_through_cuda_visible_devices(self):
        # A TP2 CTX instance on GPUs 2,3: its second rank (CUDA device 1) is GPU 3 and uses rail 3.
        with _visible("2,3"):
            self.assertEqual(
                resolve_ucx_devices("rdma_vf_rail{GPU}:1", device=1), "rdma_vf_rail3:1"
            )
        with _visible():
            self.assertEqual(
                resolve_ucx_devices("rdma_vf_rail{GPU}:1", device=2), "rdma_vf_rail2:1"
            )

    def test_gpu_token_maps_a_uuid_through_the_pci_bus_id(self):
        with (
            _visible("GPU-1a2b,GPU-3c4d"),
            mock.patch.object(remote_donor, "_nvml_gpu_index", return_value=5) as nvml,
        ):
            self.assertEqual(
                resolve_ucx_devices("rdma_vf_rail{GPU}:1", device=1), "rdma_vf_rail5:1"
            )
        nvml.assert_called_once_with(1)

    def test_unresolvable_tokens_raise(self):
        cases = [
            ("rdma_vf_rail{GPU}:1", "0"),  # device 1 not visible
            ("rdma_vf_rail{RANK}:1", None),
        ]
        for devices, visible in cases:
            with self.subTest(devices=devices, visible=visible), _visible(visible):
                with self.assertRaises(ValueError):
                    resolve_ucx_devices(devices, device=1)


class TestNixlUcxBackendParams(unittest.TestCase):
    """NIXL 1.5.2's UCX backend reads `device_list` (names split on ", ", port 1 appended); `ucx_devices` is ignored."""

    @pytest.mark.cpu_only
    def test_devices_become_the_device_list_nixl_reads(self):
        self.assertEqual(nixl_ucx_backend_params(None), {})
        self.assertEqual(nixl_ucx_backend_params("mlx5_0:1"), {"device_list": "mlx5_0"})
        self.assertEqual(
            nixl_ucx_backend_params("rdma_vf_rail0:1, rdma_vf_rail2:1,mlx5_3"),
            {"device_list": "rdma_vf_rail0, rdma_vf_rail2, mlx5_3"},
        )

    @pytest.mark.cpu_only
    def test_a_port_other_than_1_raises(self):
        for devices in ("mlx5_0:2", "mlx5_0:1,", ":1"):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                nixl_ucx_backend_params(devices)

    # Not cpu_only: the NIXL donor registers its memory with a CUDA context current.
    def test_the_donor_agent_gets_the_device_list(self):
        """The donor's NIXL agent is created with `device_list` in its backend parameters."""
        configs = []

        class _Config:
            def __init__(self):
                self.backend_params = {}

        class _Agent:
            def __init__(self, config):
                configs.append(config)

            def register_memory(self, descs):
                pass

            def deregister_memory(self, descs):
                pass

            def get_local_agent_desc(self):
                return types.SimpleNamespace(serialize=lambda: b"desc")

        binding = types.ModuleType("tensorrt_llm.tensorrt_llm_transfer_agent_binding")
        binding.BaseAgentConfig = _Config
        binding.NixlTransferAgent = _Agent
        binding.MemoryDesc = lambda *args: args
        binding.MemoryDescs = lambda *args: args
        binding.MemoryType = types.SimpleNamespace(DRAM="DRAM")
        directory = tempfile.mkdtemp(prefix="kvcm2_ucx_devices_")
        try:
            donor = make_donor(
                directory,
                4 << 20,
                transport="nixl",
                chunk_bytes=2 << 20,
                ucx_devices="mlx5_0:1,mlx5_3:1",
            )
            with mock.patch.dict(sys.modules, {binding.__name__: binding}):
                donor.start()
                donor.close()
        finally:
            shutil.rmtree(directory, ignore_errors=True)
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].backend_params, {"device_list": "mlx5_0, mlx5_3"})

    @pytest.mark.cpu_only
    def test_the_remote_level_agent_gets_the_device_list(self):
        """The remote level's tier carries `device_list` to its NIXL agent's backend parameters."""
        from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
        from tensorrt_llm.llmapi.llm_args import RemoteCacheConfig

        manager = object.__new__(KVCacheManagerV2)
        manager.is_draft = False
        directory = tempfile.mkdtemp(prefix="kvcm2_ucx_devices_")
        try:
            tiers = []
            manager._add_remote_level(
                RemoteCacheConfig(
                    directory=directory,
                    level={"size": 1 << 30},
                    transport={"kind": "nixl", "ucx_devices": "mlx5_0:1,mlx5_3:1"},
                ),
                tiers,
            )
        finally:
            shutil.rmtree(directory, ignore_errors=True)
        self.assertEqual(len(tiers), 1)
        self.assertEqual(tiers[0].transfer.backend_params, {"device_list": "mlx5_0, mlx5_3"})


if __name__ == "__main__":
    unittest.main()
