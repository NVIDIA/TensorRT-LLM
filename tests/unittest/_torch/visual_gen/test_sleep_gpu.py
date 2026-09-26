# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allocator regression tests; no model checkpoint required."""

import gc

import pytest
import torch

from tensorrt_llm._torch.visual_gen.sleep import PipelineSleepManager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["CPU", "PINNED"])
@pytest.mark.parametrize("release_cpu_backup", [False, True])
def test_sleep_preserves_values_addresses_and_unrelated_allocations(
    mode: str, release_cpu_backup: bool
) -> None:
    unrelated = torch.full((1024,), 17, dtype=torch.int32, device="cuda")
    manager = PipelineSleepManager(
        mode, torch.device("cuda:0"), release_cpu_backup=release_cpu_backup
    )
    with manager.loading():
        tensor = torch.full((8 * 1024 * 1024,), 42, dtype=torch.int32, device="cuda")
    pointer = tensor.data_ptr()
    for value in (42, 19, 63):
        tensor.fill_(value)
        manager.sleep()
        assert manager.is_sleeping
        assert bool((unrelated == 17).all())
        with pytest.raises(RuntimeError, match="asleep"):
            manager.ensure_awake()
        manager.wake_up()
        manager.ensure_awake()
        assert tensor.data_ptr() == pointer
        assert bool((tensor == value).all())
    manager.sleep()
    del tensor, manager
    gc.collect()
    torch.cuda.synchronize()
    assert bool((unrelated == 17).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["CPU", "PINNED"])
def test_sleep_isolates_two_allocation_pools(mode: str) -> None:
    first = PipelineSleepManager(mode, torch.device("cuda:0"), release_cpu_backup=True)
    with first.loading():
        first_tensor = torch.full((1024 * 1024,), 42, dtype=torch.int32, device="cuda")
    first_pointer = first_tensor.data_ptr()
    first.sleep()

    second = PipelineSleepManager(mode, torch.device("cuda:0"))
    with second.loading():
        second_tensor = torch.full((1024 * 1024,), 19, dtype=torch.int32, device="cuda")
    second_pointer = second_tensor.data_ptr()
    assert first._tag != second._tag
    assert first_pointer != second_pointer
    assert first.is_sleeping and not second.is_sleeping
    assert bool((second_tensor == 19).all())
    second.sleep()
    assert first.is_sleeping and second.is_sleeping

    first.wake_up()
    assert not first.is_sleeping and second.is_sleeping
    with pytest.raises(RuntimeError, match="asleep"):
        second.ensure_awake()
    assert first_tensor.data_ptr() == first_pointer
    assert bool((first_tensor == 42).all())
    first_tensor.fill_(73)
    first.sleep()

    second.wake_up()
    assert first.is_sleeping and not second.is_sleeping
    assert second_tensor.data_ptr() == second_pointer
    assert bool((second_tensor == 19).all())
    first.wake_up()
    assert first_tensor.data_ptr() == first_pointer
    assert bool((first_tensor == 73).all())
    assert bool((second_tensor == 19).all())

    first.sleep()
    second.sleep()
    del first_tensor, second_tensor, first, second
    gc.collect()
    torch.cuda.synchronize()
