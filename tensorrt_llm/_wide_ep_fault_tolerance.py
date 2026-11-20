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
"""Configuration checks for Wide-EP fault tolerance, and the rank identity from its launcher."""

import os
from collections.abc import Sequence
from typing import NamedTuple, Optional

# Set for every rank under Wide-EP fault tolerance to
# "<rank>,<world size>,<GPUs per node>,<replacement>".
FAULT_TOLERANCE_RANK_ENV = "TLLM_FAULT_TOLERANCE_RANK"
# The rendezvous store's host:port. The authority hosts the store.
RENDEZVOUS_ENV = "TLLM_FAULT_TOLERANCE_RENDEZVOUS"
# The hex secret that every deployment process holds. It derives the hello token.
SHARED_SECRET_ENV = "TLLM_FAULT_TOLERANCE_SHARED_SECRET"  # nosec B105 - environment variable name, not a secret
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


class RankIdentity(NamedTuple):
    """A rank under Wide-EP fault tolerance, where each rank is its own MPI world."""

    rank: int
    world_size: int
    gpus_per_node: int
    replacement: bool

    def env_value(self) -> str:
        """Return the ``FAULT_TOLERANCE_RANK_ENV`` value that gives a process this identity."""
        return ",".join(str(int(field)) for field in self)


def wide_ep_fault_tolerance_rank() -> Optional[RankIdentity]:
    """Return the rank identity the node supervisor gave this process, if one started it."""
    value = os.environ.get(FAULT_TOLERANCE_RANK_ENV)
    if value is None:
        return None
    rank, world_size, gpus_per_node, replacement = (int(field) for field in value.split(","))
    return RankIdentity(rank, world_size, gpus_per_node, bool(replacement))
