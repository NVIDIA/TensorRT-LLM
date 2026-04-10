#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Check TileIR availability (nvtriton, TileIR active, Blackwell GPU).

Standalone script -- only Python stdlib required (GPU checks use optional imports).
Outputs structured JSON to stdout.

Usage:
    python tileir_check.py          # Live check
    python tileir_check.py --mock   # Return mock data for testing
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys

# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_triton_installed() -> bool:
    """Check if any triton package is installed."""
    return importlib.util.find_spec("triton") is not None


def check_nvtriton_installed() -> bool:
    """Check if the installed triton is nvtriton (has TileIR backend).

    nvtriton is installed AS the 'triton' package. The reliable way to
    detect nvtriton is to check for the TileIR backend module.
    """
    try:
        return importlib.util.find_spec("triton.backends.tileir") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def check_tileir_active() -> bool:
    """Check if TileIR backend is currently active (ENABLE_TILE=1)."""
    try:
        import triton  # type: ignore[import-not-found]

        target = triton.runtime.driver.active.get_current_target()
        return target.backend == "tileir"
    except (ImportError, AttributeError):
        return False


def check_blackwell_gpu() -> tuple[bool, list[int]]:
    """Check if running on sm_100+ (Blackwell).

    Returns:
        Tuple of (is_blackwell, [major, minor] capability).
    """
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return cap[0] >= 10, list(cap)
        return False, [0, 0]
    except ImportError:
        return False, [0, 0]


def build_recommendation(
    *,
    triton_installed: bool,
    nvtriton_installed: bool,
    tileir_active: bool,
    blackwell_gpu: bool,
) -> str:
    """Build a human-readable recommendation string."""
    if not triton_installed:
        return "Install triton or nvtriton first"
    if not nvtriton_installed:
        return (
            "Install nvtriton (replaces triton package) for TileIR support; "
            "current configs will be ignored by standard triton"
        )
    if not tileir_active and not blackwell_gpu:
        return "nvtriton installed but no Blackwell GPU detected; TileIR requires sm_100+ hardware"
    if not tileir_active:
        return "Set TRITON_PTXAS_PATH and run with ENABLE_TILE=1 to activate TileIR"
    return "TileIR is active and ready"


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------


def check_tileir_availability() -> dict:
    """Run all TileIR availability checks.

    Returns:
        Dict with structured availability information.
    """
    has_triton = check_triton_installed()
    has_nvtriton = check_nvtriton_installed()
    is_active = check_tileir_active()
    has_blackwell, gpu_cap = check_blackwell_gpu()

    recommendation = build_recommendation(
        triton_installed=has_triton,
        nvtriton_installed=has_nvtriton,
        tileir_active=is_active,
        blackwell_gpu=has_blackwell,
    )

    return {
        "nvtriton_installed": has_nvtriton,
        "tileir_active": is_active,
        "blackwell_gpu": has_blackwell,
        "triton_installed": has_triton,
        "gpu_capability": gpu_cap,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------


def _mock_data() -> dict:
    """Return realistic mock data for testing without GPU/triton."""
    return {
        "nvtriton_installed": False,
        "tileir_active": False,
        "blackwell_gpu": True,
        "triton_installed": True,
        "gpu_capability": [10, 0],
        "recommendation": ("Set TRITON_PTXAS_PATH and run with ENABLE_TILE=1 to activate TileIR"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Check TileIR backend availability.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Return mock data for testing.",
    )
    args = parser.parse_args()

    if args.mock:
        data = _mock_data()
    else:
        data = check_tileir_availability()

    json.dump(data, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
