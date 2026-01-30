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
"""Performance logging utilities for KV cache transfer."""

import logging
import os
import sys
import threading
import time

from tensorrt_llm import logger

# CSV header for performance log files
_PERF_CSV_HEADER = (
    "timestamp,task_type,unique_rid,peer_rank,"
    "transfer_size_bytes,avg_segment_size_bytes,transfer_entry_count,"
    "prepare_args_latency_ms,queue_latency_ms,transfer_latency_ms,task_latency_ms,throughput_mbs"
)


class PerfTimer:
    """Timer for measuring the time of the KV cache transfer."""

    def __init__(self):
        self._push_start_times = dict()
        self._push_end_times = dict()
        self._start_transfer_times = dict()
        self._end_transfer_times = dict()
        self._tranfer_entry_counts = dict()
        self._transfer_sizes = dict()
        self._prepare_args_start_times = dict()
        self._prepare_args_end_times = dict()
        self._task_start_times = dict()
        self._task_end_times = dict()

    def record_task_start(self, peer_rank: int):
        self._task_start_times[peer_rank] = time.perf_counter()

    def record_task_end(self, peer_rank: int):
        self._task_end_times[peer_rank] = time.perf_counter()

    def record_prepare_args_start(self, peer_rank: int):
        self._prepare_args_start_times[peer_rank] = time.perf_counter()

    def record_prepare_args_end(self, peer_rank: int):
        self._prepare_args_end_times[peer_rank] = time.perf_counter()

    def record_push_start(self, peer_rank: int):
        self._push_start_times[peer_rank] = time.perf_counter()

    def record_push_end(self, peer_rank: int):
        self._push_end_times[peer_rank] = time.perf_counter()

    def get_prepare_args_latency(self, peer_rank: int) -> float:
        """Get prepare args latency in seconds."""
        if (
            self._prepare_args_start_times.get(peer_rank, None) is None
            or self._prepare_args_end_times.get(peer_rank, None) is None
        ):
            return 0.0
        return self._prepare_args_end_times[peer_rank] - self._prepare_args_start_times[peer_rank]

    def get_queue_latency(self, peer_rank: int) -> float:
        """Get queue latency in seconds."""
        if (
            self._push_start_times.get(peer_rank, None) is None
            or self._push_end_times.get(peer_rank, None) is None
        ):
            return 0.0
        return self._push_end_times[peer_rank] - self._push_start_times[peer_rank]

    def record_transfer_start(self, peer_rank: int):
        self._start_transfer_times[peer_rank] = time.perf_counter()

    def record_transfer_end(self, peer_rank: int):
        self._end_transfer_times[peer_rank] = time.perf_counter()

    def get_transfer_size(self, peer_rank: int) -> int:
        return self._transfer_sizes.get(peer_rank, 0)

    def get_transfer_latency(self, peer_rank: int) -> float:
        if (
            self._start_transfer_times.get(peer_rank, None) is None
            or self._end_transfer_times.get(peer_rank, None) is None
        ):
            return 0.0
        return self._end_transfer_times[peer_rank] - self._start_transfer_times[peer_rank]  # s

    def get_transfer_throughput(self, peer_rank: int) -> float:
        if (
            self._start_transfer_times.get(peer_rank, None) is None
            or self._end_transfer_times.get(peer_rank, None) is None
        ):
            return 0.0
        latency = self.get_transfer_latency(peer_rank)
        if latency == 0:
            return 0.0
        return (self._transfer_sizes[peer_rank] / latency) / (1024.0 * 1024.0)  # MB/s

    def record_transfer_sizes(self, peer_rank: int, size: int, count: int):
        self._transfer_sizes[peer_rank] = size
        self._tranfer_entry_counts[peer_rank] = count

    def get_average_segment_size(self, peer_rank: int) -> float:
        if self._tranfer_entry_counts.get(peer_rank, 0) == 0:
            return 0.0
        return self._transfer_sizes[peer_rank] / self._tranfer_entry_counts[peer_rank]

    def get_transfer_entry_count(self, peer_rank: int) -> int:
        return self._tranfer_entry_counts.get(peer_rank, 0)

    def get_task_latency(self, peer_rank: int) -> float:
        if self._task_end_times.get(peer_rank, None) is None:
            return 0.0
        return self._task_end_times[peer_rank] - self._task_start_times[peer_rank]


class PerfLogManager:
    """Singleton manager for KV transfer performance logging.

    Logic:
    - TLLM_ENABLE_CACHE_TRANSFER_PERF_INFO not set: no output
    - TLLM_ENABLE_CACHE_TRANSFER_PERF_INFO set, TLLM_KV_TRANSFER_PERF_LOG_FILE not set:
      logger.info to stdout
    - Both set: CSV output to {TLLM_KV_TRANSFER_PERF_LOG_FILE}_{instance_name}_{instance_rank}.csv
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._file_loggers = {}  # (instance_name, instance_rank) -> logger
        self._file_lock = threading.Lock()
        self._perf_enabled = os.getenv("TLLM_ENABLE_CACHE_TRANSFER_PERF_INFO", "0") == "1"
        self._log_file_base = os.getenv("TLLM_KV_TRANSFER_PERF_LOG_FILE")

    @property
    def enabled(self) -> bool:
        return self._perf_enabled

    @property
    def use_file(self) -> bool:
        return self._perf_enabled and self._log_file_base is not None

    def _get_or_create_file_logger(self, instance_name: str, instance_rank: int):
        """Get or create a file logger for the given instance."""
        key = (instance_name, instance_rank)
        if key in self._file_loggers:
            return self._file_loggers[key]

        with self._file_lock:
            if key in self._file_loggers:
                return self._file_loggers[key]

            # Create file path: {base}_{instance_name}_{instance_rank}.csv
            log_file = f"{self._log_file_base}_{instance_name}_{instance_rank}.csv"

            try:
                # Create directory if needed
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

                # Check if file exists (to decide whether to write header)
                write_header = not os.path.exists(log_file)

                file_logger = logging.getLogger(f"kv_transfer_perf_{instance_name}_{instance_rank}")
                file_logger.setLevel(logging.INFO)
                file_logger.propagate = False

                file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
                # Custom formatter with milliseconds
                formatter = logging.Formatter(
                    fmt="%(asctime)s.%(msecs)03d,%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
                )
                file_handler.setFormatter(formatter)
                file_logger.addHandler(file_handler)

                # Write CSV header if new file
                if write_header:
                    file_handler.stream.write(_PERF_CSV_HEADER + "\n")
                    file_handler.stream.flush()

                self._file_loggers[key] = file_logger
                return file_logger
            except Exception as e:
                sys.stderr.write(
                    f"[KV Transfer] Warning: Failed to create perf log file {log_file}: {e}\n"
                )
                return None

    def log(self, instance_name: str, instance_rank: int, csv_line: str, info_msg: str):
        """Log performance data.

        Args:
            instance_name: Instance name for file naming
            instance_rank: Instance rank for file naming
            csv_line: CSV formatted line (without timestamp, will be added by logger)
            info_msg: Human-readable message for logger.info output
        """
        if not self._perf_enabled:
            return

        if self._log_file_base:
            # Output to file only (no stdout)
            file_logger = self._get_or_create_file_logger(instance_name, instance_rank)
            if file_logger:
                file_logger.info(csv_line)
        else:
            # Output to stdout via logger.info
            logger.info(info_msg)


# Singleton instance
perf_log_manager = PerfLogManager()
