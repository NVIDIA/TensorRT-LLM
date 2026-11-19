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
"""Configuration checks and environment variables of Wide-EP fault tolerance."""

from collections.abc import Sequence

# The rendezvous store's host:port. The authority hosts the store.
RENDEZVOUS_ENV = "TLLM_FAULT_TOLERANCE_RENDEZVOUS"
# The hex secret that every deployment process holds. It derives the hello token.
SHARED_SECRET_ENV = "TLLM_FAULT_TOLERANCE_SHARED_SECRET"
# The alternative that each error about an unmet fault tolerance requirement offers.
TURN_FAULT_TOLERANCE_OFF = "turn fault tolerance off (enable_wide_ep_fault_tolerance=False)"


def check_fault_tolerance_requirements(checks: Sequence[tuple[bool, str]]) -> None:
    """Raise ValueError naming every unmet ``(met, requirement)`` check."""
    missing = [requirement for met, requirement in checks if not met]
    if missing:
        raise ValueError(
            "Wide-EP fault tolerance requires "
            + "; ".join(missing)
            + f"; change the configuration or {TURN_FAULT_TOLERANCE_OFF}"
        )
