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

import logging
import os

try:
    from pynvml import (
        NVMLError,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetHandleByUUID,
        nvmlDeviceGetTotalEnergyConsumption,
        nvmlInit,
        nvmlShutdown,
    )

    has_nvml = True
except ImportError:
    has_nvml = False

logger = logging.getLogger(__name__)


class EnergyMonitor:
    def __init__(self, world_size):
        self._enabled = has_nvml
        self._world_size = world_size
        self._start_energies = None
        self._total_energy = None
        if self._enabled:
            try:
                nvmlInit()
                self._handles = self._get_gpu_handles(world_size)
                self._device_count = len(self._handles)
            except (NVMLError, ValueError) as e:
                logger.warning(f"Failed to initialize NVML: {e}")
                self._enabled = False

    @staticmethod
    def _get_gpu_handles(world_size):
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        device_ids = (
            [e.strip() for e in cuda_visible.split(",") if e.strip()] if cuda_visible else []
        )

        if not device_ids:
            count = min(nvmlDeviceGetCount(), world_size)
            return [nvmlDeviceGetHandleByIndex(i) for i in range(count)]

        handles = []
        for device_id in device_ids[:world_size]:
            if device_id.startswith(("GPU-", "MIG-")):
                handles.append(nvmlDeviceGetHandleByUUID(device_id))
            else:
                handles.append(nvmlDeviceGetHandleByIndex(int(device_id)))
        return handles

    def __enter__(self):
        if self._enabled:
            try:
                self._start_energies = [
                    nvmlDeviceGetTotalEnergyConsumption(handle) for handle in self._handles
                ]
            except NVMLError as e:
                logger.warning(f"Failed to read GPU energy on start: {e}")
                self._start_energies = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._enabled or self._start_energies is None:
            return False

        try:
            total_energy = 0.0
            for handle, start_energy in zip(self._handles, self._start_energies):
                energy = (nvmlDeviceGetTotalEnergyConsumption(handle) - start_energy) / 1000.0
                total_energy += energy
            self._total_energy = total_energy * self._world_size / self._device_count
        except NVMLError as e:
            logger.warning(f"Failed to read GPU energy on stop: {e}")
        finally:
            try:
                nvmlShutdown()
            except NVMLError:
                pass
        return False

    @property
    def total_energy(self):
        return self._total_energy
