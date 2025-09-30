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
"""
Controls GPU clock settings(Not implemented yet) and monitors GPU status using pynvml.

The monitoring part creates a Python thread (not truly multiprocess) that polls the GPU and CPU states.
"""

import datetime
import os
import platform
import socket
# Std
import threading
import time

import psutil  # type: ignore
# Nvidia
import pynvml  # type: ignore
from defs.trt_test_alternative import print_info, print_warning

from .misc import clean_device_product_name


class InvalidGPUMonitoringResultError(RuntimeError):
    """GPU monitoring result is invalid, probably caused by clock frequency drops due to thermal issue."""


class GPUClockLockFailFastError(RuntimeError):
    """GPU clock locking has failed."""


class GPUState:

    def __init__(self, gpu_id, gpu_clock, mem_clock, timestamp, graphics_clk,
                 gpu_util, mem_util, encoder_util, decoder_util, gpu_temp,
                 mem_temp, fan_speed, perf_state, power_draw, process_num):
        self.gpu_id = gpu_id
        self.gpu_clock__MHz = gpu_clock
        self.memory_clock__MHz = mem_clock
        self.timestamp = timestamp
        self.graphics_clock__MHz = graphics_clk
        self.gpu_utilization__pct = gpu_util
        self.memory_utilization__pct = mem_util
        self.encoder_utilization__pct = encoder_util
        self.decoder_utilization__pct = decoder_util
        self.gpu_temperature__C = gpu_temp
        self.memory_temperature__C = mem_temp
        self.fan_speed__pct = fan_speed
        self.perf_state = perf_state
        self.power_draw__W = power_draw
        self.process_num = process_num


class GPUClockLock:

    def __init__(self, gpu_id, interval_ms):
        """
        Sets up clock values and tears down every run. At the end of the session call teardown to complete session and
        reset GPU clocks.

        Args:
            gpu_id (str): GPU identifier, either comma-separated UUIDs or comma-separated indices in string.
            interval_ms (float): Interval duration between monitoring samples.
        """
        # Initialize pynvml
        self._nvml_initialized = False
        self._gpu_handles = []

        # Input params.
        self._gpu_id = gpu_id
        self._gpu_id_list = [int(id) for id in gpu_id.split(",")]
        self._mobile_disable_clock_locking = False

        # Create GPU handles, one per GPU.
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            self._gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                for gpu_id in self._gpu_id_list
            ]
            print_info(f"Created GPU handles: {self._gpu_handles}")
        except pynvml.NVMLError as e:
            print_warning(f"Failed to initialize NVML: {e}")

        if self._gpu_handles is None:
            print_warning(
                "Unable to create GPU handles. GPU monitoring will be disabled."
            )
        else:
            print_info("GPU handles created successfully!")

        # Setup device properties.
        self._setup_properties()

        # Fields for monitoring thread.
        self._interval_ms = interval_ms
        self._is_monitoring = False
        self._state_data = []

    def get_os_properties(self):
        return self._os_properties

    def get_cpu_properties(self):
        return self._cpu_properties

    def get_gpu_properties(self):
        return self._gpu_properties

    def get_gpu_id(self):
        return self._gpu_id

    def get_driver_version(self):
        return self._nvidia_driver_version

    def get_device_count(self):
        return self._nvidia_device_count

    def get_ip_address(self):
        return self._ip_address

    def get_target_gpu_clocks(self):
        """
        Get the target GPU clocks (sm_clk and mem_clk) for the first GPU in the list.
        """
        # We don't set gpu clock currently, so let it return None.
        return None

    def __enter__(self):
        """
        Do all the steps needed at the start of a test case:
        - Lock gpu clock to target.
        - Start monitoring.
        """
        print_info("gpu clock lock enter!!!")
        if not self._nvml_initialized:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            self._gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                for gpu_id in self._gpu_id_list
            ]
            print_info(f"Reinitialized GPU handles: {self._gpu_handles}")
        self.start_monitor()
        return self

    def __exit__(self, *args):
        """
        Do all the steps needed at the end of a test case:
        - Stop monitoring.
        - Set gpu clock back to original state.
        - Validate gpu monitoring result.
        """
        self.stop_monitor()
        self.validate_gpu_monitoring_data()
        print_info("gpu clock lock exit!!!")

    def start_monitor(self):
        """Start GPU monitoring."""

        if self._gpu_handles is None:
            print_warning(
                "Unable to start GPU monitoring. GPU handles are not initialized."
            )
            return

        if self._is_monitoring:
            raise RuntimeError(
                "GPU monitoring is already in progress. Monitoring cannot be started!"
            )

        # Delete state_data
        self._state_data = []
        self._is_monitoring = True

        # Initialize thread
        self._thread = threading.Thread(
            target=self._monitoring_thread,
            name="LLM Test - GPUMonitor",
            kwargs={"interval_ms": self._interval_ms})
        self._thread.daemon = True
        self._thread.start()

    def stop_monitor(self):
        """Stop GPU monitoring."""
        if self._gpu_handles is None:
            return

        if not self._is_monitoring:
            raise RuntimeError(
                "GPU monitoring has not been started. Monitoring cannot be stopped!"
            )

        self._is_monitoring = False
        self._thread.join()

    def get_state_data(self):
        """
        Get all the gpu monitoring data since monitoring started.
        Seems like from our empirical data get_state_data() can return None.
        This might have something to do with thread failure if something were to happen to GPU monitoring thread.
        """
        return self._state_data

    def validate_gpu_monitoring_data(self, deviation_perc=0.07, num_entries=3):
        """
        Check that all the current monitoring data is within the given deviation_perc for the given number of
        consecutive entries in a row.

        The "num_entries" argument specifies the number of consecutive entries the monitoring data needs to be invalid
        before considering the entire dataset as invalid
        """

        if self._mobile_disable_clock_locking:
            print_info("Skipped gpu monitoring validation for mobile board")
            return

    def teardown(self):
        """
        Call when the session finishes. Reset GPU clocks back to its original state.
        """
        # Revert clocks back to normal if all tests have finished.
        # Set current clock value back to session entry clock.
        #self.release_clock()
        if self._nvml_initialized:
            pynvml.nvmlShutdown()
            self._nvml_initialized = False
        print_info("NVML shutdown.")

    def _gpu_poll_state(self):
        if not self._nvml_initialized:
            print_warning("NVML is not initialized. Skipping GPU polling.")
            return

        for gpu_idx, gpu_handle in enumerate(self._gpu_handles):
            try:
                sm_clk = pynvml.nvmlDeviceGetClockInfo(gpu_handle,
                                                       pynvml.NVML_CLOCK_SM)
                mem_clk = pynvml.nvmlDeviceGetClockInfo(gpu_handle,
                                                        pynvml.NVML_CLOCK_MEM)
                graphics_clk = pynvml.nvmlDeviceGetClockInfo(
                    gpu_handle, pynvml.NVML_CLOCK_GRAPHICS)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
                mem_util = pynvml.nvmlDeviceGetUtilizationRates(
                    gpu_handle).memory
                gpu_temp = pynvml.nvmlDeviceGetTemperature(
                    gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                perf_state = pynvml.nvmlDeviceGetPerformanceState(gpu_handle)
                power_draw = pynvml.nvmlDeviceGetPowerUsage(
                    gpu_handle) / 1000.0  # Convert from milliwatts to watts
                process_num = len(
                    pynvml.nvmlDeviceGetComputeRunningProcesses(gpu_handle))
                encoder_util = pynvml.nvmlDeviceGetEncoderUtilization(
                    gpu_handle)[0]  # Get encoder utilization percentage
                decoder_util = pynvml.nvmlDeviceGetDecoderUtilization(
                    gpu_handle)[0]  # Get decoder utilization percentage
                gpu_state = GPUState(
                    gpu_id=self._gpu_id_list[gpu_idx],
                    gpu_clock=sm_clk,
                    mem_clock=mem_clk,
                    timestamp=datetime.datetime.now(),
                    graphics_clk=graphics_clk,
                    gpu_util=gpu_util,
                    mem_util=mem_util,
                    gpu_temp=gpu_temp,
                    mem_temp=
                    None,  # Can't use pynvml to get memory temperature data
                    fan_speed=
                    None,  # Will hit not supported exception when call pynvml.nvmlDeviceGetFanSpeed()
                    perf_state=perf_state,
                    power_draw=power_draw,
                    process_num=process_num,
                    encoder_util=encoder_util,
                    decoder_util=decoder_util)
                self._state_data.append(gpu_state)
            except pynvml.NVMLError as e:
                print_warning(f"Error polling GPU state for GPU {gpu_idx}: {e}")

    def _monitoring_thread(self, interval_ms):
        """Actual thread that runs to monitor similar to perf_runner.monitor"""
        interval = interval_ms / 1000

        # Get the state of the object.
        while self._is_monitoring:
            self._gpu_poll_state()
            # Sleep the thread
            time.sleep(interval)

        # Final time for interpolation.
        self._gpu_poll_state()
        time.sleep(interval)
        self._gpu_poll_state()

    def _setup_properties(self):
        """Set up OS/CPU/GPU properties """
        try:
            self._os_properties = {
                "os_name": os.name,
                "platform": platform.system(),
                "platform_version": platform.version()
            }
        except Exception as e:
            self._os_properties = None
            print_warning("Unable to fetch os properties. Reason: {}".format(e))

        try:
            self._cpu_properties = {
                "cpu_count":
                os.cpu_count(),
                "cpu_freq":
                psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }
        except Exception as e:
            self._cpu_properties = None
            print_warning(
                "Unable to fetch cpu properties. Reason: {}".format(e))

        try:
            self._ip_address = socket.gethostbyname(socket.gethostname())
        except Exception as e:
            self._ip_address = None
            print_warning("Unable to fetch os IP address. Reason: {}".format(e))

        if self._gpu_handles is not None:
            self._nvidia_driver_version = pynvml.nvmlSystemGetDriverVersion()
            self._nvidia_device_count = pynvml.nvmlDeviceGetCount()
            self._gpu_properties = {
                "device_product_name":
                pynvml.nvmlDeviceGetName(self._gpu_handles[0]),
                "pci_device_id":
                pynvml.nvmlDeviceGetPciInfo(self._gpu_handles[0]).pciDeviceId
            }

            # Clean up the device product name because the product names have changed after driver updates.
            self._gpu_properties[
                "device_product_name"] = clean_device_product_name(
                    self._gpu_properties["device_product_name"])

            if "jetson" in self._gpu_properties[
                    "device_product_name"] or "p3710" in self._gpu_properties[
                        "device_product_name"]:
                self._mobile_disable_clock_locking = True

        else:
            self._nvidia_driver_version = None
            self._nvidia_device_count = None
            self._gpu_properties = None
