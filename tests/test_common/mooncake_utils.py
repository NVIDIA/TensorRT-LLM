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
"""Running a Mooncake master for a test, on ports picked for it.

A deployment names its master's ports, so `running_master` takes them as
given. A test has no pair to name and wants any free one, on a machine whose
other ports belong to whatever else runs there. A port is reserved here by
binding it and letting go, so another process can still take it before the
master binds, and the master then exits before it accepts connections. Each
attempt picks a fresh pair.
"""

import contextlib
from typing import Iterator

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.master import (
    LaunchedMaster,
    running_master,
)
from tensorrt_llm._utils import get_free_ports

__all__ = ["running_master_on_free_ports"]

PORT_ATTEMPTS = 5


def _start_master(
    stack: contextlib.ExitStack, run_dir: str, attempts: int, kwargs: dict
) -> LaunchedMaster:
    """Enter `running_master` on `stack`, on the first pair of ports it keeps.

    A failed start leaves nothing of the master behind, so every attempt
    begins from the state the first one did. The last failure is raised as it
    is, since it names the port that was taken.
    """
    for attempt in range(1, attempts + 1):
        rpc_port, metrics_port = get_free_ports(2)
        try:
            return stack.enter_context(
                running_master(run_dir, rpc_port=rpc_port, metrics_port=metrics_port, **kwargs)
            )
        except (RuntimeError, TimeoutError):
            if attempt == attempts:
                raise
            print(
                f"mooncake-store: the master did not come up on rpc={rpc_port} "
                f"metrics={metrics_port} (attempt {attempt} of {attempts}); "
                "retrying on another pair of ports."
            )
    raise AssertionError("unreachable: the last attempt either returns or raises")


@contextlib.contextmanager
def running_master_on_free_ports(
    run_dir: str, *, attempts: int = PORT_ATTEMPTS, **kwargs
) -> Iterator[LaunchedMaster]:
    """Run a master on free ports for the body's duration.

    Args:
        run_dir: Passed through to `running_master`.
        attempts: Pairs of ports to try.
        kwargs: Everything else `running_master` takes, except the two ports.
    """
    with contextlib.ExitStack() as stack:
        yield _start_master(stack, run_dir, attempts, kwargs)
