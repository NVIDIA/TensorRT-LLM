# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import socket


def get_diagnostic_host_identity() -> tuple[str, str]:
    """Return a stable node identity and the source used to derive it.

    ``HOSTNAME`` can be inherited from the container launch node in Slurm
    jobs, so prefer Slurm's execution-node identity and then the kernel
    hostname.
    """
    slurm_node = os.getenv("SLURMD_NODENAME")
    if slurm_node:
        return _sanitize_diagnostic_value(slurm_node), "slurm"

    try:
        hostname = socket.gethostname()
    except OSError:
        hostname = ""
    if hostname:
        return _sanitize_diagnostic_value(hostname), "socket"

    environment_hostname = os.getenv("HOSTNAME")
    if environment_hostname:
        return _sanitize_diagnostic_value(environment_hostname), "environment"
    return "unknown", "fallback"


def _sanitize_diagnostic_value(value: str) -> str:
    return "_".join(value.split())
