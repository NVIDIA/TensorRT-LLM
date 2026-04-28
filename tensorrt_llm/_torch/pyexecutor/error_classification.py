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
"""Error classification and error budget for fatal-error detection.

This module is intentionally dependency-free (no CUDA, no C++ extensions)
so that it can be imported and tested in any environment.
"""

import dataclasses
import time

# Patterns that corrupt the CUDA context beyond recovery.
# Matched case-insensitively against the error message.
IMMEDIATE_FATAL_PATTERNS: list[str] = [
    "cudaerrorillegaladdress",
    "cudaerrorlaunchfailure",
    "illegal memory access",
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


@dataclasses.dataclass
class ErrorBudget:
    """Token-bucket error budget for fatal-error promotion.

    Each error deducts a cost from the budget (0.1 for transient, 0.5
    for severe).  The budget recovers at ``recovery_rate`` per second
    of error-free wall time.  When exhausted, the error is promoted to
    fatal.  Immediate-fatal errors bypass the budget entirely.

    Attributes:
        budget: Current budget level (starts at 1.0, capped at 1.0).
        last_error_time: Monotonic timestamp of the last error.
        recovery_rate: Budget recovered per second of error-free time.
        cost: Cost per transient error (severe costs 5x this).
    """

    budget: float = 1.0
    last_error_time: float | None = None
    recovery_rate: float = 0.1
    cost: float = 0.1

    def consume(self, error_msg: str) -> bool:
        """Deduct from the budget and return True if exhausted.

        Args:
            error_msg: The error message to classify and budget.

        Returns:
            True if the error should be treated as fatal.
        """
        now = time.monotonic()
        classification = classify_error(error_msg)

        if classification == "immediate_fatal":
            return True

        if self.last_error_time is not None:
            elapsed = now - self.last_error_time
            self.budget = min(1.0, self.budget + elapsed * self.recovery_rate)
        self.last_error_time = now

        deduction = self.cost
        if classification == "severe":
            deduction *= 5

        self.budget -= deduction
        return self.budget < 1e-9
