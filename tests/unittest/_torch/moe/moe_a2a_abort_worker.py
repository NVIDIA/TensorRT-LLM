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
"""Pickle-stable entry points for MoE A2A abort MPI workers.

``test_moe_comm`` is registered with cloudpickle by value. Serializing its
abort workers directly can therefore walk their PyTorch globals and reach the
unpickleable ``torch.ops`` singleton. These thin entry points live in an
ordinary importable module, so mpi4py serializes them by reference and imports
the implementation only after deserialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _torch.modules.moe.test_moe_comm import CommTestConfig


def run_dispatch_abort_worker(
    config: CommTestConfig,
    missing_rank: int,
    abort_source: str,
) -> dict:
    """Import and run the dispatch worker after MPI deserialization."""
    from _torch.modules.moe.test_moe_comm import _worker_running_dispatch_abort

    return _worker_running_dispatch_abort(config, missing_rank, abort_source)


def run_combine_abort_worker(
    config: CommTestConfig,
    missing_rank: int,
    abort_source: str,
) -> dict:
    """Import and run the combine worker after MPI deserialization."""
    from _torch.modules.moe.test_moe_comm import _worker_running_combine_abort

    return _worker_running_combine_abort(config, missing_rank, abort_source)
