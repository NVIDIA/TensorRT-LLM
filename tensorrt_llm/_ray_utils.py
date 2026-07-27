# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import functools
from contextlib import contextmanager
from typing import Callable, Optional

try:
    import ray
except ImportError:
    import tensorrt_llm.ray_stub as ray


@contextmanager
def unwrap_ray_errors():
    try:
        yield
    except ray.exceptions.RayTaskError as e:
        raise e.as_instanceof_cause() from e


def control_action_decorator(func: Optional[Callable] = None,
                             *,
                             drain: bool = True) -> Callable:
    """Wrap a method in the ``control_action`` context manager.

    Supports both bare and parameterized forms::

        @control_action_decorator                  # drain=True (default)
        def shutdown(self): ...

        @control_action_decorator(drain=False)     # non-draining variant
        def update_weights_via_ipc_zmq(self): ...
    """

    def decorator(f: Callable) -> Callable:

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            with self.engine.control_action(drain=drain):
                return f(self, *args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
