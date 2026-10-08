# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Process and system telemetry, including AMDGPU sysfs counters.

GPU identities are physical DRM/PCI identities, not HIP logical indices. This
avoids incorrect attribution when HIP_VISIBLE_DEVICES reorders the GPUs.
Missing counters stay null rather than masquerading as zero utilization.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path


def read_number(path: Path) -> int | None:
    """Read an optional integer sysfs counter."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def amd_gpu_snapshot(drm_root: Path = Path("/sys/class/drm")) -> list[dict]:
    """Read physical AMD GPU utilization and VRAM counters without shell commands."""
    devices = []
    for card in sorted(drm_root.glob("card[0-9]*")):
        if not card.name.removeprefix("card").isdigit():
            continue
        device = card / "device"
        try:
            vendor = (device / "vendor").read_text().strip().lower()
        except OSError:
            continue
        if vendor != "0x1002":
            continue
        used = read_number(device / "mem_info_vram_used")
        total = read_number(device / "mem_info_vram_total")
        devices.append(
            {
                "id": f"drm:{card.name}:{device.resolve().name}",
                "vendor": "AMD",
                "pci_address": device.resolve().name,
                "gpu_percent": read_number(device / "gpu_busy_percent"),
                "memory_busy_percent": read_number(device / "mem_busy_percent"),
                "vram_used_bytes": used,
                "vram_total_bytes": total,
                "vram_percent": 100 * used / total if used is not None and total else None,
                "source": "amdgpu-sysfs",
            }
        )
    return devices


class ResourceSampler:
    """Periodically sample CPU, RAM, GPU and VRAM; summaries cover all samples.

    Raw samples are bounded. Running statistics include discarded raw samples,
    so a long-running server does not lose its peak or average utilization.
    Process CPU percentage can exceed 100% (100% is one logical CPU).
    """

    def __init__(
        self,
        interval: float = 0.1,
        max_samples: int = 10000,
        drm_root: Path = Path("/sys/class/drm"),
    ) -> None:
        if interval <= 0 or max_samples <= 0:
            raise ValueError("interval and max_samples must be positive")
        self.interval = interval
        self.drm_root = drm_root
        self.samples: deque[dict] = deque(maxlen=max_samples)
        self.count = 0
        self._stats: dict[str, dict[str, float | int]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_cpu: tuple[int, int] | None = None
        self._last_process: tuple[float, float] | None = None
        self._origin = time.monotonic()
        self._nvml = None
        self._process = None
        self.warnings: list[str] = []
        try:
            import psutil
        except ImportError:
            self._psutil = None
        else:
            self._psutil = psutil
            self._process = psutil.Process()
            self._process.cpu_percent()
            psutil.cpu_percent()
        try:
            import pynvml
        except ImportError:
            pass
        else:
            try:
                pynvml.nvmlInit()
            except pynvml.NVMLError:
                pass
            else:
                self._nvml = pynvml

    def _proc_snapshot(self) -> dict[str, float | int | None]:
        now = time.monotonic()
        process_time = sum(os.times()[:2])
        process_percent = None
        if self._last_process is not None:
            previous_time, previous_wall = self._last_process
            elapsed = now - previous_wall
            if elapsed > 0:
                process_percent = 100 * (process_time - previous_time) / elapsed
        self._last_process = (process_time, now)
        cpu_percent = None
        memory: dict[str, int] = {}
        rss = None
        try:
            fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            ticks = [int(value) for value in fields]
            # Guest ticks are already included in user/nice; do not count twice.
            total, idle = sum(ticks[:8]), ticks[3] + ticks[4]
            if self._last_cpu is not None:
                delta = total - self._last_cpu[0]
                if delta > 0:
                    cpu_percent = 100 * (1 - (idle - self._last_cpu[1]) / delta)
            self._last_cpu = (total, idle)
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                memory[key] = int(value.split()[0]) * 1024
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        total_memory = memory.get("MemTotal")
        available = memory.get("MemAvailable")
        used = total_memory - available if total_memory and available is not None else None
        return {
            "cpu_system_percent": cpu_percent,
            "cpu_process_percent": process_percent,
            "ram_used_bytes": used,
            "ram_total_bytes": total_memory,
            "ram_percent": 100 * used / total_memory if used is not None and total_memory else None,
            "process_rss_bytes": rss,
            "process_ram_percent": 100 * rss / total_memory
            if rss is not None and total_memory
            else None,
        }

    def _host_snapshot(self) -> dict[str, float | int | None]:
        if self._psutil is None:
            return self._proc_snapshot()
        memory = self._psutil.virtual_memory()
        # Match MemAvailable-based accounting across psutil and /proc fallbacks.
        used = memory.total - memory.available
        return {
            "cpu_system_percent": self._psutil.cpu_percent(),
            "cpu_process_percent": self._process.cpu_percent(),
            "ram_used_bytes": used,
            "ram_total_bytes": memory.total,
            "ram_percent": 100 * used / memory.total,
            "process_rss_bytes": self._process.memory_info().rss,
            "process_ram_percent": 100 * self._process.memory_info().rss / memory.total,
        }

    def _nvidia_snapshot(self) -> list[dict]:
        if self._nvml is None:
            return []
        nvml = self._nvml
        devices = []
        try:
            count = nvml.nvmlDeviceGetCount()
            for index in range(count):
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                try:
                    utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                except nvml.NVMLError:
                    gpu_percent = memory_busy_percent = None
                else:
                    gpu_percent, memory_busy_percent = utilization.gpu, utilization.memory
                devices.append(
                    {
                        "id": f"nvml:{nvml.nvmlDeviceGetUUID(handle)}",
                        "vendor": "NVIDIA",
                        "gpu_percent": gpu_percent,
                        "memory_busy_percent": memory_busy_percent,
                        "vram_used_bytes": memory.used,
                        "vram_total_bytes": memory.total,
                        "vram_percent": 100 * memory.used / memory.total if memory.total else None,
                        "source": "nvml",
                    }
                )
        except nvml.NVMLError as error:
            message = f"NVML telemetry unavailable: {error}"
            if message not in self.warnings:
                self.warnings.append(message)
        return devices

    def sample(self) -> None:
        """Take one snapshot and update online min/mean/peak statistics."""
        host = self._host_snapshot()
        gpus = amd_gpu_snapshot(self.drm_root) + self._nvidia_snapshot()
        sample = {"elapsed_s": time.monotonic() - self._origin, "host": host, "gpus": gpus}
        self.samples.append(sample)
        self.count += 1
        metrics = dict(host)
        for gpu in gpus:
            for name, value in gpu.items():
                if name not in ("id", "vendor", "source", "pci_address"):
                    metrics[f"{gpu['id']}/{name}"] = value
        for name, value in metrics.items():
            if value is None:
                continue
            stats = self._stats.setdefault(
                name, {"count": 0, "sum": 0.0, "min": float(value), "peak": float(value)}
            )
            stats["count"] += 1
            stats["sum"] += value
            stats["min"] = min(stats["min"], value)
            stats["peak"] = max(stats["peak"], value)

    def start(self) -> None:
        """Start a single background sampler thread."""
        if self._thread is not None:
            raise RuntimeError("ResourceSampler has already started")
        self._origin = time.monotonic()
        self.sample()

        def run() -> None:
            while not self._stop.wait(self.interval):
                self.sample()

        self._thread = threading.Thread(target=run, name="trtllm-profile-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling, take a final snapshot, and release optional NVML state."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()
        if self._nvml is not None:
            self._nvml.nvmlShutdown()
            self._nvml = None

    def report(self) -> dict:
        """Return complete summaries and the bounded tail of raw samples."""
        return {
            "interval_s": self.interval,
            "sample_count": self.count,
            "discarded_raw_samples": self.count - len(self.samples),
            "summary": {
                name: {
                    "count": stats["count"],
                    "min": stats["min"],
                    "mean": stats["sum"] / stats["count"],
                    "peak": stats["peak"],
                }
                for name, stats in sorted(self._stats.items())
            },
            "samples": list(self.samples),
            "warnings": self.warnings,
        }
