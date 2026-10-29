# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility for the CuTe DSL 4.7/4.8 task execution APIs."""

import inspect

from cutlass.experimental.task_scheduling.memory import ResourceContext
from cutlass.experimental.task_scheduling.task import Task


_TASK_ACCEPTS_CONTEXT = (
    "context" in inspect.signature(Task._run_task_body_impl).parameters
)


def task_context_kwargs(
    context: ResourceContext | None,
) -> dict[str, ResourceContext | None]:
    """Supply the explicit execution context only on the 4.7 task API.

    DSL 4.8 removed this argument from task execution and stage-info methods:
    ``init_variables`` now stores it on the task. Keep the 4.7 propagation,
    including ``None``, while allowing both APIs to call PrimTS overrides.
    Signature detection happens once at import; this helper runs at trace time.
    """
    return {"context": context} if _TASK_ACCEPTS_CONTEXT else {}
