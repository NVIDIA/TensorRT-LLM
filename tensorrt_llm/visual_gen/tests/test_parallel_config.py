# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch.distributed as dist

from visual_gen.configs.parallel import BaseParallelConfig, DiTParallelConfig, VAEParallelConfig


@pytest.fixture
def parallel_config():
    """Fixture that provides a fresh BaseParallelConfig instance for each test."""
    config = BaseParallelConfig()
    config.clear_instance()
    yield config
    config.clear_instance()  # Clean up after the test


def get_valid_configs():
    """Generate valid parallel configurations for testing.

    Args:
        max_gpus (int): Maximum number of GPUs to consider

    Returns:
        list: List of tuples containing (tp_size, ulysses_size, ring_size, dp_size, cfg_size)
    """
    configs = []

    # Generate all possible combinations that fit within max_gpus
    for tp_size in [1, 2, 4, 8]:
        for ulysses_size in [1, 2, 4, 8]:
            for ring_size in [1, 2, 4, 8]:
                for dp_size in [1, 2, 4, 8]:
                    for cfg_size in [1, 2]:
                        total_size = tp_size * ulysses_size * ring_size * dp_size * cfg_size
                        if not dist.is_initialized():
                            dist.init_process_group(backend="nccl")
                        if total_size == dist.get_world_size():
                            configs.append((tp_size, ulysses_size, ring_size, dp_size, cfg_size))

    # Clear any existing instances before yielding configurations
    BaseParallelConfig.clear_instance()
    VAEParallelConfig.clear_instance()
    DiTParallelConfig.clear_instance()

    return configs


@pytest.mark.parametrize("tp_size,ulysses_size,ring_size,dp_size,cfg_size", get_valid_configs())
def test_parallel_config_initialization(
    parallel_config: BaseParallelConfig, tp_size, ulysses_size, ring_size, dp_size, cfg_size
):
    """Test parallel configuration initialization with different combinations."""
    # Set configuration
    parallel_config.set_config(
        tp_size=tp_size, ulysses_size=ulysses_size, ring_size=ring_size, dp_size=dp_size, cfg_size=cfg_size
    )

    # Verify configuration values
    assert parallel_config.tp_size() == tp_size
    assert parallel_config.ulysses_size() == ulysses_size
    assert parallel_config.ring_size() == ring_size
    assert parallel_config.dp_size() == dp_size
    assert parallel_config.cfg_size() == cfg_size

    # Verify total parallel size
    total_size = tp_size * ulysses_size * ring_size * dp_size * cfg_size
    assert parallel_config.get_total_parallel_size() == total_size

    # Verify device mesh initialization
    device_mesh = parallel_config.device_mesh()
    assert device_mesh is not None

    # Verify process groups
    assert parallel_config.check_process_groups()


@pytest.mark.parametrize("tp_size,ulysses_size,ring_size,dp_size,cfg_size", get_valid_configs())
def test_parallel_config_ranks(
    parallel_config: BaseParallelConfig, tp_size, ulysses_size, ring_size, dp_size, cfg_size
):
    """Test rank retrieval for different parallel configurations."""
    # Skip test if not in distributed environment
    if not dist.is_initialized():
        pytest.skip("Test requires distributed environment")

    parallel_config.set_config(
        tp_size=tp_size, ulysses_size=ulysses_size, ring_size=ring_size, dp_size=dp_size, cfg_size=cfg_size
    )

    # Get the global rank of current process
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Verify total size matches world size
    sp_size = ulysses_size * ring_size
    total_size = dp_size * cfg_size * sp_size * tp_size
    assert total_size == world_size, f"Total parallel size {total_size} does not match world size {world_size}"

    # Calculate expected ranks based on global rank
    expected_ranks = {}
    remaining_rank = global_rank

    # Calculate dp rank
    expected_ranks["dp"] = remaining_rank // (cfg_size * sp_size * tp_size)
    remaining_rank %= cfg_size * sp_size * tp_size

    # Calculate cfg rank
    expected_ranks["cfg"] = remaining_rank // (sp_size * tp_size)
    remaining_rank %= sp_size * tp_size

    # Calculate ring rank
    expected_ranks["ring"] = remaining_rank // (ulysses_size * tp_size)
    remaining_rank %= ulysses_size * tp_size

    # Calculate ulysses rank
    expected_ranks["ulysses"] = remaining_rank // tp_size
    remaining_rank %= tp_size

    # Calculate tp rank
    expected_ranks["tp"] = remaining_rank

    # Get actual ranks
    actual_ranks = parallel_config.all_ranks()

    # Compare expected and actual ranks
    for dim in ["dp", "cfg", "ring", "ulysses", "tp"]:
        assert (
            actual_ranks[dim] == expected_ranks[dim]
        ), f"Rank mismatch for {dim}: expected {expected_ranks[dim]}, got {actual_ranks[dim]}"

    # Verify with device mesh if available
    device_mesh = parallel_config.device_mesh()
    if device_mesh is not None:
        for dim in device_mesh.mesh_dim_names:
            assert actual_ranks[dim] == device_mesh.get_local_rank(
                dim
            ), f"Rank mismatch with device mesh for {dim}: expected {device_mesh.get_local_rank(dim)}, got {actual_ranks[dim]}"


def test_invalid_configurations(parallel_config: BaseParallelConfig):
    """Test invalid parallel configurations."""
    # Test invalid tp_size
    with pytest.raises(ValueError):
        parallel_config.set_config(tp_size=0)

    # Test invalid cfg_size
    with pytest.raises(ValueError):
        parallel_config.set_config(cfg_size=3)


def test_singleton_pattern():
    """Test singleton pattern implementation."""
    config1 = BaseParallelConfig()
    config2 = BaseParallelConfig()
    assert config1 is config2

    vae_config1 = VAEParallelConfig()
    vae_config2 = VAEParallelConfig()
    assert vae_config1 is vae_config2

    dit_config1 = DiTParallelConfig()
    dit_config2 = DiTParallelConfig()
    assert dit_config1 is dit_config2

    # Different config types should be different instances
    assert config1 is not vae_config1
    assert vae_config1 is not dit_config1
