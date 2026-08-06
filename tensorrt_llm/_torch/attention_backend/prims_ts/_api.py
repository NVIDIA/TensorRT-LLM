# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Local compatibility for FlashInfer's optional API instrumentation."""

from collections.abc import Callable
from typing import TypeVar, overload

_CallableT = TypeVar("_CallableT", bound=Callable[..., object])


@overload
def flashinfer_api(func: _CallableT, *, trace: object | None = None) -> _CallableT: ...


@overload
def flashinfer_api(
    func: None = None, *, trace: object | None = None
) -> Callable[[_CallableT], _CallableT]: ...


def flashinfer_api(
    func: _CallableT | None = None, *, trace: object | None = None
) -> _CallableT | Callable[[_CallableT], _CallableT]:
    """Preserve the upstream decorator ABI without FlashInfer instrumentation.

    TensorRT-LLM does not vendor FlashInfer's API logging and trace-template
    framework. Returning the original callable preserves the zero-overhead
    behavior used by FlashInfer when instrumentation is disabled.
    """

    del trace
    if func is not None:
        return func

    def decorator(wrapped: _CallableT) -> _CallableT:
        return wrapped

    return decorator
