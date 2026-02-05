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

import logging
import os
import platform
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_STATS_SERVER = "https://stats.tensorrt-llm.nvidia.com/v1/collect"
DEFAULT_HEARTBEAT_INTERVAL = 600  # 10 minutes
CONFIG_DIR = Path.home() / ".config" / "trtllm"
OPT_OUT_FILE = CONFIG_DIR / "do_not_track"


def is_usage_stats_enabled() -> bool:
    """Check if usage statistics collection is enabled."""
    # Check environment variable
    if os.getenv("TRTLLM_NO_USAGE_STATS", "0") == "1":
        return False

    # Check for opt-out file
    if OPT_OUT_FILE.exists():
        return False

    return True


def collect_system_info() -> Dict[str, Any]:
    """Collect system and hardware information."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }

    # GPU information via torch
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            if torch.cuda.device_count() > 0:
                info["gpu_type"] = torch.cuda.get_device_name(0)
                info["gpu_memory_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                )
            info["cuda_version"] = torch.version.cuda
    except Exception:
        # If torch is not available or GPU detection fails, continue without GPU info
        pass

    return info


def detect_cloud_provider() -> str:
    """Detect cloud provider via metadata endpoints.

    This function is designed to be ultra-safe and never block the main server.
    Uses aggressive timeouts and broad exception handling.
    """
    detectors = [
        ("aws", "http://169.254.169.254/latest/meta-data/", {}),
        (
            "gcp",
            "http://metadata.google.internal/computeMetadata/v1/",
            {"Metadata-Flavor": "Google"},
        ),
        (
            "azure",
            "http://169.254.169.254/metadata/instance?api-version=2021-02-01",
            {"Metadata": "true"},
        ),
        ("oci", "http://169.254.169.254/opc/v2/instance/", {}),
    ]

    for provider, endpoint, headers in detectors:
        try:
            response = requests.get(endpoint, timeout=0.5, headers=headers, allow_redirects=False)
            if response.status_code == 200:
                return provider
        except requests.RequestException:
            # Network error or timeout - try next provider
            continue
        except Exception:
            # Unexpected error but never crash
            continue

    return "unknown"


class UsageMessage:
    """Manages collection and reporting of usage statistics."""

    def __init__(self):
        self.uuid = str(uuid.uuid4())

    def collect_initial_data(self, llm_args: Any, pretrained_config: Optional[Any]) -> Dict:
        """Collect comprehensive initial report."""
        import tensorrt_llm

        VERSION = tensorrt_llm.__version__

        return {
            "uuid": self.uuid,
            "timestamp_ns": time.time_ns(),
            "trtllm_version": VERSION,
            "system_info": collect_system_info(),
            "cloud_provider": detect_cloud_provider(),
            "trtllm_config": self._extract_trtllm_config(llm_args),
            "model_info": self._extract_model_info(pretrained_config),
        }

    def collect_heartbeat_data(self) -> Dict:
        """Collect lightweight heartbeat."""
        return {
            "uuid": self.uuid,
            "timestamp_ns": time.time_ns(),
        }

    def _extract_trtllm_config(self, llm_args: Any) -> Dict:
        """Extract TensorRT-LLM configuration from llm_args."""
        config = {
            "backend": getattr(llm_args, "backend", None),
            "tensor_parallel_size": getattr(llm_args, "tensor_parallel_size", None),
            "pipeline_parallel_size": getattr(llm_args, "pipeline_parallel_size", None),
            "context_parallel_size": getattr(llm_args, "context_parallel_size", None),
            "dtype": getattr(llm_args, "dtype", None),
        }

        # KV cache config
        if hasattr(llm_args, "kv_cache_config"):
            kv_cfg = llm_args.kv_cache_config
            config["kv_cache"] = {
                "enable_block_reuse": getattr(kv_cfg, "enable_block_reuse", None),
                "max_tokens": getattr(kv_cfg, "max_tokens", None),
                "free_gpu_memory_fraction": getattr(kv_cfg, "free_gpu_memory_fraction", None),
            }

        # Quantization from model_kwargs
        if hasattr(llm_args, "model_kwargs") and llm_args.model_kwargs:
            config["quantization"] = llm_args.model_kwargs.get("quant_algo")

        return config

    def _extract_model_info(self, pretrained_config: Optional[Any]) -> Dict:
        """Extract model architecture info (no sensitive paths)."""
        if pretrained_config is None:
            return {"architecture": "unknown"}

        # Extract architecture from architectures list
        architecture = "unknown"
        if hasattr(pretrained_config, "architectures") and pretrained_config.architectures:
            architecture = pretrained_config.architectures[0]

        return {
            "architecture": architecture,
            "model_type": getattr(pretrained_config, "model_type", None),
            "num_layers": getattr(pretrained_config, "num_hidden_layers", None),
            "hidden_size": getattr(pretrained_config, "hidden_size", None),
            "num_attention_heads": getattr(pretrained_config, "num_attention_heads", None),
        }

    def report_once(self, data: Dict):
        """Send data to stats server.

        Ultra-safe: uses aggressive timeout and fails silently on any error.
        """
        try:
            endpoint = os.getenv("TRTLLM_USAGE_STATS_SERVER", DEFAULT_STATS_SERVER)
            response = requests.post(
                endpoint,
                json=data,
                timeout=2.0,
                headers={"Content-Type": "application/json"},
                allow_redirects=False,
            )
            response.raise_for_status()
        except Exception:
            # Silently fail - never crash the server
            pass


def report_usage(llm_args: Any, pretrained_config: Optional[Any] = None):
    """Start background usage statistics collection.

    Ultra-safe: wrapped in try/except to ensure it NEVER crashes the main server,
    even if thread creation fails.
    """
    try:
        if not is_usage_stats_enabled():
            return

        message = UsageMessage()

        # Start daemon thread
        thread = threading.Thread(
            target=_background_reporter,
            args=(message, llm_args, pretrained_config),
            daemon=True,
            name="trtllm-usage-stats",
        )
        thread.start()
    except Exception:
        # Never crash the main server - silently fail if thread can't start
        pass


def _background_reporter(message: UsageMessage, llm_args: Any, pretrained_config: Optional[Any]):
    """Background thread that sends reports.

    Ultra-safe: all operations wrapped to ensure no exceptions escape.
    Runs with aggressive timeouts to prevent blocking.
    """
    try:
        # Send initial report with timeout protection
        try:
            initial_data = message.collect_initial_data(llm_args, pretrained_config)
            message.report_once(initial_data)
        except Exception:
            # Continue to heartbeat even if initial report fails
            pass

        # Send periodic heartbeats
        interval = int(os.getenv("TRTLLM_USAGE_HEARTBEAT_INTERVAL", DEFAULT_HEARTBEAT_INTERVAL))
        max_heartbeats = 1000  # Safety limit: stop after ~7 days at 10min intervals
        heartbeat_count = 0

        while heartbeat_count < max_heartbeats:
            time.sleep(interval)
            heartbeat_count += 1
            try:
                heartbeat_data = message.collect_heartbeat_data()
                message.report_once(heartbeat_data)
            except Exception:
                # Continue to next heartbeat even on failure
                continue
    except Exception:
        # Thread exits silently on any unexpected error
        pass
