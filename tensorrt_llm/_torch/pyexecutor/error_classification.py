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
"""Error classification for fatal-error detection in PyExecutor.

This module is intentionally dependency-free (no CUDA, no C++ extensions)
so that it can be imported and tested in any environment.
"""

# Patterns that corrupt the CUDA context beyond recovery.
# Matched case-insensitively against the error message.
IMMEDIATE_FATAL_PATTERNS: list[str] = [
    "cudaerrorillegaladdress",
    "cudaerrorlaunchfailure",
    "device-side assert",
    "unrecoverable",
]

# Patterns that are serious but may be transient (e.g. a single OOM
# during a traffic spike).  These drain the error budget 5x faster
# than transient errors.
SEVERE_ERROR_PATTERNS: list[str] = [
    "cuda out of memory",
    "cuda error",
    "nccl error",
]


def classify_error(error_msg: str) -> str:
    """Classify an error message by severity.

    Args:
        error_msg: The error message string to classify.

    Returns:
        One of ``"immediate_fatal"``, ``"severe"``, or ``"transient"``.

        - **immediate_fatal**: CUDA context is corrupted (device-side
          assert, illegal address, launch failure).  No future CUDA call
          can succeed.
        - **severe**: The operation failed but the CUDA context is
          intact (e.g. OOM, NCCL timeout).  Recovery is possible if
          workload decreases.
        - **transient**: All other errors (bad input, timeout, etc.).
    """
    error_lower = error_msg.lower()
    for p in IMMEDIATE_FATAL_PATTERNS:
        if p in error_lower:
            return "immediate_fatal"
    for p in SEVERE_ERROR_PATTERNS:
        if p in error_lower:
            return "severe"
    return "transient"
