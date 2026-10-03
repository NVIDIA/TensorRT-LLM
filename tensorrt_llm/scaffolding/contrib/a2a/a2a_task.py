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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Union

from tensorrt_llm.scaffolding.task import Task

if TYPE_CHECKING:
    from tensorrt_llm.scaffolding.controller import Controller


@dataclass
class A2ASendTask(Task):
    """Send a natural-language message to a remote A2A agent and collect its reply."""

    # the name of the remote agent (matches the agent card's name) to route to
    agent_name: Optional[str] = field(default=None)
    # the natural-language message to send to the remote agent
    message: Optional[str] = field(default=None)

    worker_tag: Union[str, "Controller.WorkerTag"] = None

    # result field, filled in by the worker with the agent's textual response
    output_str: Optional[str] = None

    @staticmethod
    def create_a2a_task(
        agent_name: str, message: str, worker_tag: Union[str, "Controller.WorkerTag"] = None
    ) -> "A2ASendTask":
        task = A2ASendTask()
        task.agent_name = agent_name
        task.message = message
        task.worker_tag = worker_tag
        return task


@dataclass
class A2AListTask(Task):
    """Discover the remote A2A agents reachable by the worker."""

    worker_tag: Union[str, "Controller.WorkerTag"] = None

    # result field, filled in by the worker with a list of AgentInfo objects
    result_agents: Optional[List[Any]] = None

    @staticmethod
    def create_a2a_task(worker_tag: Union[str, "Controller.WorkerTag"] = None) -> "A2AListTask":
        task = A2AListTask()
        task.worker_tag = worker_tag
        return task
