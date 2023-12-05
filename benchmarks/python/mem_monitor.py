# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from multiprocessing import Event, Process, Queue

from tensorrt_llm.logger import logger
from tensorrt_llm.profiler import (MemUnitType, bytes_to_target_unit,
                                   device_memory_info)


class MemoryMonitor:

    def __init__(self, query_interval=0.1):
        self.query_interval = query_interval  # second(s)
        self.mem_monitor_process = None
        # bytes
        self._peak_host_memory = 0
        self._peak_device_memory = 0

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

        self._peak_device_memory = max(self._peak_device_memory,
                                       self.peak_mem_queue.get())

        self.mem_monitor_process.join()
        self.mem_monitor_process = None
        logger.debug("Memory monitor subprocess joined.")

    def _upd_peak_memory_usage(self, signal_event, peak_mem_queue):
        peak_used, _, _ = device_memory_info()
        while not signal_event.is_set():
            used, _, _ = device_memory_info()
            peak_used = max(used, peak_used)
        peak_mem_queue.put(peak_used)

    def get_peak_memory_usage(self, unit: MemUnitType = 'GiB'):
        return bytes_to_target_unit(self._peak_host_memory, unit), \
            bytes_to_target_unit(self._peak_device_memory, unit)
