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
Prints session data to session_properties.csv or session_data.yml for perf tests.
"""

import datetime
import socket

from . import data_export as dataexport
from .data import SessionData


class SessionDataWriter:

    def __init__(self, log_output_directory, output_formats, gpu_clock_lock):
        """
        Records information of the current session and prints it to session_properties.csv or session_data.yml during
        teardown.

        Args:
            log_output_directory (str): Output directory for perf payloads.
            output_formats (List(str)): Output format perf payload formats. Options: ["csv", "yaml"]
            gpu_clock_lock (GPUClockLock): The GPUClockLock instance for GPU clock locking and monitoring.
        """
        self._log_output_directory = log_output_directory
        self._output_formats = set(output_formats)
        self._gpu_clock_lock = gpu_clock_lock

        # Perf specific session properties and values to utilize
        self._session_data = SessionData()

        # Record session start time.
        self._session_data.start_timestamp = datetime.datetime.utcnow()

        # Populate session information
        self._session_data.os_properties = self._gpu_clock_lock.get_os_properties(
        )
        self._session_data.cpu_properties = self._gpu_clock_lock.get_cpu_properties(
        )
        self._session_data.gpu_properties = self._gpu_clock_lock.get_gpu_properties(
        )
        self._session_data.nvidia_driver_version = self._gpu_clock_lock.get_driver_version(
        )
        self._session_data.nvidia_device_count = self._gpu_clock_lock.get_device_count(
        )
        self._session_data.hostname = socket.gethostname()
        self._session_data.ip = self._gpu_clock_lock.get_ip_address()

    def teardown(self):
        """
        Called when the session finishes. Writes session info to session_properties.csv or session_data.yml.
        """
        # Record session end time.
        self._session_data.end_timestamp = datetime.datetime.utcnow()
        # Write session payload.
        self._write_session_perf_logs()

    def _write_session_perf_logs(self):
        """
        Write session data. Should only be called once at the end of the entire
        perf session, in otherwords, only during teardown().
        """
        # Output various log files depending on options.
        for fmt in self._output_formats:
            dataexport.write_session_properties(self._log_output_directory,
                                                [self._session_data],
                                                output=fmt)
