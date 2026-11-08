#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for GPU clock locking functionality.

This test requires actual GPU hardware and appropriate permissions to lock GPU clocks.
It verifies that:
1. GPU clocks can be locked to frequencies specified in gpu_configs.yml
2. The actual clock frequencies match the configured values after locking
3. Clock unlocking works properly during cleanup
"""

import os
import time

import pynvml
import pytest
import yaml

from .gpu_clock_lock import GPUClockLock, GPUClockLockFailFastError
from .misc import clean_device_product_name


class TestGPUClockLockIntegration:
    """Integration tests for GPU clock locking with actual hardware."""

    @pytest.fixture
    def gpu_config(self):
        """Load GPU configurations from gpu_configs.yml."""
        config_path = os.path.join(os.path.dirname(__file__), "../../perf_configs/gpu_configs.yml")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def gpu_name(self):
        """Get the current GPU name."""
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            cleaned_name = clean_device_product_name(gpu_name)
            return cleaned_name
        finally:
            pynvml.nvmlShutdown()

    def test_clock_locking_sets_correct_frequencies(self, gpu_config, gpu_name):
        """Test that GPU clocks are locked to the correct frequencies from gpu_configs.yml.

        Note: This test requires root/sudo permissions to lock GPU clocks.
        Run with: sudo -E pytest tests/integration/defs/perf/test_gpu_clock_lock.py -v -s
        """
        # Skip test if GPU is not in config
        if gpu_name not in gpu_config.get("GPUs", {}):
            pytest.skip(f"GPU '{gpu_name}' not found in gpu_configs.yml")

        expected_sm_clk = gpu_config["GPUs"][gpu_name]["sm_clk"]
        expected_mem_clk = gpu_config["GPUs"][gpu_name]["mem_clk"]

        print(f"\nTesting GPU: {gpu_name}")
        print(f"Expected SM Clock: {expected_sm_clk} MHz")
        print(f"Expected Memory Clock: {expected_mem_clk} MHz")

        # Create GPU clock lock instance
        gpu_clock_lock = GPUClockLock(gpu_id="0", interval_ms=100)

        try:
            # Enter context manager - this locks the clocks
            with gpu_clock_lock:
                # Give a moment for clocks to stabilize
                time.sleep(0.5)

                # Query actual clock frequencies
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                # Get application clocks (the ones we set)
                actual_sm_clk = pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM)
                actual_mem_clk = pynvml.nvmlDeviceGetApplicationsClock(
                    handle, pynvml.NVML_CLOCK_MEM
                )

                # Get current running clocks (may differ slightly)
                current_sm_clk = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                current_mem_clk = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)

                pynvml.nvmlShutdown()

                print(f"Actual Application SM Clock: {actual_sm_clk} MHz")
                print(f"Actual Application Memory Clock: {actual_mem_clk} MHz")
                print(f"Current Running SM Clock: {current_sm_clk} MHz")
                print(f"Current Running Memory Clock: {current_mem_clk} MHz")

                # Verify application clocks match expected values
                assert actual_sm_clk == expected_sm_clk, (
                    f"SM clock mismatch: expected {expected_sm_clk} MHz, got {actual_sm_clk} MHz"
                )
                assert actual_mem_clk == expected_mem_clk, (
                    f"Memory clock mismatch: expected {expected_mem_clk} MHz, "
                    f"got {actual_mem_clk} MHz"
                )

                print("✓ Clock frequencies verified successfully!")

        except GPUClockLockFailFastError as e:
            if "Insufficient Permissions" in str(e):
                pytest.skip(
                    f"Insufficient permissions to lock GPU clocks. "
                    f"Run with: sudo -E pytest {__file__} -v -s"
                )
            raise
        finally:
            # Ensure cleanup happens even if test fails
            gpu_clock_lock.teardown()

    def test_clock_unlocking_restores_original_clocks(self, gpu_config, gpu_name):
        """Test that GPU clocks are restored to original values after unlocking."""
        # Skip test if GPU is not in config
        if gpu_name not in gpu_config.get("GPUs", {}):
            pytest.skip(f"GPU '{gpu_name}' not found in gpu_configs.yml")

        # Get original clocks before locking
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        original_sm_clk = pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM)
        original_mem_clk = pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_MEM)
        pynvml.nvmlShutdown()

        print(f"\nOriginal SM Clock: {original_sm_clk} MHz")
        print(f"Original Memory Clock: {original_mem_clk} MHz")

        # Create GPU clock lock and lock clocks
        gpu_clock_lock = GPUClockLock(gpu_id="0", interval_ms=100)

        try:
            with gpu_clock_lock:
                # Clocks are locked here
                print("Clocks locked...")
                time.sleep(0.5)

            # Exiting context manager should restore clocks
            time.sleep(0.5)

            # Verify clocks are restored
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            restored_sm_clk = pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM)
            restored_mem_clk = pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_MEM)
            pynvml.nvmlShutdown()

            print(f"Restored SM Clock: {restored_sm_clk} MHz")
            print(f"Restored Memory Clock: {restored_mem_clk} MHz")

            assert restored_sm_clk == original_sm_clk, (
                f"SM clock not restored: expected {original_sm_clk} MHz, got {restored_sm_clk} MHz"
            )
            assert restored_mem_clk == original_mem_clk, (
                f"Memory clock not restored: expected {original_mem_clk} MHz, "
                f"got {restored_mem_clk} MHz"
            )

            print("✓ Clocks restored successfully!")

        except GPUClockLockFailFastError as e:
            if "Insufficient Permissions" in str(e):
                pytest.skip(
                    f"Insufficient permissions to lock GPU clocks. "
                    f"Run with: sudo -E pytest {__file__} -v -s"
                )
            raise
        finally:
            gpu_clock_lock.teardown()

    def test_get_target_gpu_clocks_returns_config_values(self, gpu_config, gpu_name):
        """Test that get_target_gpu_clocks returns the correct values from gpu_configs.yml."""
        # Skip test if GPU is not in config
        if gpu_name not in gpu_config.get("GPUs", {}):
            pytest.skip(f"GPU '{gpu_name}' not found in gpu_configs.yml")

        expected_sm_clk = gpu_config["GPUs"][gpu_name]["sm_clk"]
        expected_mem_clk = gpu_config["GPUs"][gpu_name]["mem_clk"]

        # Create GPU clock lock instance
        gpu_clock_lock = GPUClockLock(gpu_id="0", interval_ms=100)

        try:
            # Get target clocks
            target_clocks = gpu_clock_lock.get_target_gpu_clocks()

            assert target_clocks is not None, "get_target_gpu_clocks returned None"

            target_sm_clk, target_mem_clk = target_clocks

            print(f"\nGPU: {gpu_name}")
            print(f"Target SM Clock: {target_sm_clk} MHz (expected: {expected_sm_clk} MHz)")
            print(f"Target Memory Clock: {target_mem_clk} MHz (expected: {expected_mem_clk} MHz)")

            assert target_sm_clk == expected_sm_clk, (
                f"Target SM clock mismatch: expected {expected_sm_clk} MHz, got {target_sm_clk} MHz"
            )
            assert target_mem_clk == expected_mem_clk, (
                f"Target memory clock mismatch: expected {expected_mem_clk} MHz, "
                f"got {target_mem_clk} MHz"
            )

            print("✓ Target clocks match configuration!")

        finally:
            gpu_clock_lock.teardown()

    def test_multi_gpu_clock_locking(self, gpu_config):
        """Test clock locking works with multiple GPUs."""
        # Get number of available GPUs
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()

        if gpu_count < 2:
            pytest.skip("Multi-GPU test requires at least 2 GPUs")

        print(f"\nTesting with {gpu_count} GPUs")

        # Create GPU IDs string
        gpu_ids = ",".join(str(i) for i in range(gpu_count))

        # Create GPU clock lock instance for all GPUs
        gpu_clock_lock = GPUClockLock(gpu_id=gpu_ids, interval_ms=100)

        try:
            with gpu_clock_lock:
                time.sleep(0.5)

                # Verify clocks for each GPU
                pynvml.nvmlInit()
                for gpu_idx in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                    gpu_name = clean_device_product_name(pynvml.nvmlDeviceGetName(handle))

                    if gpu_name not in gpu_config.get("GPUs", {}):
                        print(f"GPU {gpu_idx} ({gpu_name}) not in config, skipping")
                        continue

                    expected_sm_clk = gpu_config["GPUs"][gpu_name]["sm_clk"]
                    expected_mem_clk = gpu_config["GPUs"][gpu_name]["mem_clk"]

                    actual_sm_clk = pynvml.nvmlDeviceGetApplicationsClock(
                        handle, pynvml.NVML_CLOCK_SM
                    )
                    actual_mem_clk = pynvml.nvmlDeviceGetApplicationsClock(
                        handle, pynvml.NVML_CLOCK_MEM
                    )

                    print(f"GPU {gpu_idx} ({gpu_name}):")
                    print(f"  SM Clock: {actual_sm_clk} MHz (expected: {expected_sm_clk} MHz)")
                    print(
                        f"  Memory Clock: {actual_mem_clk} MHz (expected: {expected_mem_clk} MHz)"
                    )

                    assert actual_sm_clk == expected_sm_clk
                    assert actual_mem_clk == expected_mem_clk

                pynvml.nvmlShutdown()

                print("✓ All GPU clocks verified successfully!")

        except GPUClockLockFailFastError as e:
            if "Insufficient Permissions" in str(e):
                pytest.skip(
                    f"Insufficient permissions to lock GPU clocks. "
                    f"Run with: sudo -E pytest {__file__} -v -s"
                )
            raise
        finally:
            gpu_clock_lock.teardown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
