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
import os
from multiprocessing import Event, Process, Queue

from tensorrt_llm.logger import logger
from tensorrt_llm.profiler import (MemUnitType, bytes_to_target_unit,
                                   device_memory_info, host_memory_info)


class MemoryMonitor:

    def __init__(self, query_interval=0.1):
        self.query_interval = query_interval  # second(s)
        self.mem_monitor_process = None
        # bytes
        self._peak_host_memory = 0
        self._peak_device_memory = 0

        self.pid = os.getpid()
        self.device_handles = {}

        self.signal_event = Event()  # Sending signal to subprocess
        self.peak_mem_queue = Queue()  # Receiving results from subprocess

    def start(self):
        self.mem_monitor_process = Process(target=self._upd_peak_memory_usage,
                                           args=(self.signal_event,
                                                 self.peak_mem_queue))
        self.mem_monitor_process.start()
        logger.debug("Launched memory monitor subprocess.")

    def kill(self):
        if self.mem_monitor_process is not None:
            self.mem_monitor_process.kill()
            logger.debug("Memory monitor subprocess is killed.")

    def stop(self):
        self.signal_event.set()
        logger.debug("Sent signal to stop memory monitor subprocess.")

        peak_mem_use = self.peak_mem_queue.get(timeout=10)

        self._peak_host_memory = max(self._peak_host_memory, peak_mem_use[0])
        self._peak_device_memory = max(self._peak_device_memory,
                                       peak_mem_use[1])

        self.mem_monitor_process.join(timeout=10)
        self.mem_monitor_process = None
        logger.debug("Memory monitor subprocess joined.")

    def _upd_peak_memory_usage(self, signal_event, peak_mem_queue):
        peak_host_used, peak_device_used = self.get_memory_usage()
        while not signal_event.is_set():
            host_used, device_used = self.get_memory_usage()
            peak_host_used = max(host_used, peak_host_used)
            peak_device_used = max(device_used, peak_device_used)
        peak_mem_queue.put((peak_host_used, peak_device_used))

    def get_memory_usage(self):
        host_used, _, _ = host_memory_info(self.pid)
        device_used, _, _ = device_memory_info()
        return host_used, device_used

    def get_peak_memory_usage(self, unit: MemUnitType = 'GiB'):
        return bytes_to_target_unit(self._peak_host_memory, unit), \
            bytes_to_target_unit(self._peak_device_memory, unit)
