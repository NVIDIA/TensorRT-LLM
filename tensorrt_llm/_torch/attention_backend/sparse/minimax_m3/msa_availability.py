# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Availability checks for the MiniMax-M3 MSA sparse attention kernels.

The MSA kernels are provided by the fmha_sm100 package from the MSA git
submodule at 3rdparty/MSA and run only on the SM100 architecture family
(SM100 and SM103). These helpers gate backend selection so a request for the
MSA path fails early with a clear message on unsupported systems.
"""

from __future__ import annotations

from tensorrt_llm._utils import get_sm_version, is_sm_100f

from .msa_utils import msa_package_available

# fmha_sm100 runs on the SM100 architecture family (SM100 and SM103). Other
# architectures, including SM120, are not supported.
MSA_PACKAGE = "fmha_sm100"


def ensure_msa_available() -> None:
    """Raise RuntimeError if the MSA sparse attention path cannot run here."""
    if not msa_package_available():
        raise RuntimeError(
            f"MiniMax-M3 MSA sparse attention requires the {MSA_PACKAGE} kernels "
            "from the MSA git submodule at 3rdparty/MSA. Initialize it with "
            "'git submodule update --init --recursive'."
        )
    if not is_sm_100f():
        sm_version = get_sm_version()
        raise RuntimeError(
            "MiniMax-M3 MSA sparse attention requires an SM100 or SM103 device, "
            f"but the current device reports SM version {sm_version}."
        )


__all__ = [
    "MSA_PACKAGE",
    "ensure_msa_available",
]
