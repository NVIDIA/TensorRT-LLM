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
"""MPI pool session of Wide-EP fault tolerance, whose node supervisor starts the ranks but 0."""

import os
import subprocess  # nosec B404
import sys
from concurrent.futures import Future
from typing import Callable, Dict, List, Optional, TypeVar

from mpi4py.futures import MPIPoolExecutor

from .._wide_ep_fault_tolerance import FAULT_TOLERANCE_RANK_ENV, RankIdentity
from ._llmapi_node_supervisor import start_session_node_supervisor, stop_session_node_supervisor
from .mpi_session import MpiPoolSession, _mgmn_shutdown_grace_seconds, _run_on_each_worker

T = TypeVar("T")


class NodeSupervisedMpiPoolSession(MpiPoolSession):
    """Run rank 0 in the pool while a node supervisor starts and replaces the other ranks.

    Each rank is its own MPI world under Wide-EP fault tolerance, so the pool holds rank 0 only.
    """

    def __init__(self, n_workers: int, env_overrides: Optional[Dict[str, str]] = None):
        """Start the ``n_workers`` ranks of one node.

        Args:
            n_workers: The number of ranks.
            env_overrides: Extra environment variables of every rank.
        """
        self._node_supervisor: Optional[subprocess.Popen] = None
        super().__init__(n_workers, env_overrides=env_overrides)

    def submit(self, task: Callable[..., T], *args, **kwargs) -> List[Future[T]]:
        """Run ``task`` on rank 0 and return its future."""
        return [self.mpi_pool.submit(_run_on_each_worker, task, *args, **kwargs)]

    def shutdown(self, wait=True):
        # A dead pool can never be joined, as in MpiPoolSession.shutdown.
        wait = wait and not getattr(self, "_pool_dead", False)
        super().shutdown(wait)
        self._stop_node_supervisor(_mgmn_shutdown_grace_seconds() if wait else None)

    def abort(self):
        self._stop_node_supervisor(None)
        super().abort()

    def _create_pool(self, env: Dict[str, str], python_args: List[str]) -> MPIPoolExecutor:
        self._node_supervisor = start_session_node_supervisor(self.n_workers, {**os.environ, **env})
        # One node of n_workers ranks, as the node supervisor's.
        env[FAULT_TOLERANCE_RANK_ENV] = RankIdentity(
            0, self.n_workers, self.n_workers, False
        ).env_value()
        try:
            return MPIPoolExecutor(max_workers=1, path=sys.path, env=env, python_args=python_args)
        except BaseException:
            self._stop_node_supervisor(None)
            raise

    def _stop_node_supervisor(self, grace_s: Optional[float]) -> None:
        """Stop the node supervisor, if any, after waiting ``grace_s`` seconds for it to exit."""
        if self._node_supervisor is not None:
            stop_session_node_supervisor(self._node_supervisor, grace_s)
            self._node_supervisor = None
