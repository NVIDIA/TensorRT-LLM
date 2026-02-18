# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from pynvml import (
        NVMLError,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetTotalEnergyConsumption,
        nvmlInit,
        nvmlShutdown,
    )

    has_nvml = True
except ImportError:
    has_nvml = False


class EnergyMonitor:
    def __init__(self, world_size):
        global has_nvml
        self._start_energies = None
        if has_nvml:
            try:
                nvmlInit()
                self._device_count = min(nvmlDeviceGetCount(), world_size)
                self._handles = [nvmlDeviceGetHandleByIndex(i) for i in range(self._device_count)]
            except NVMLError:
                has_nvml = False

    def start(self):
        if has_nvml:
            self._start_energies = [
                nvmlDeviceGetTotalEnergyConsumption(handle) for handle in self._handles
            ]

    def stop(self, statistics, world_size):
        if not has_nvml:
            return statistics.set_energy(None)

        total_energy = 0.0
        for handle, start_energy in zip(self._handles, self._start_energies):
            energy = (nvmlDeviceGetTotalEnergyConsumption(handle) - start_energy) / 1000.0
            total_energy += energy

        total_energy *= world_size / self._device_count
        nvmlShutdown()

        statistics.set_energy(total_energy)
