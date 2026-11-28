# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reserve a CPython data-stack chunk for an eager Kimi forward.

CPython 3.11+ maps a larger chunk when a frame outgrows the current one. Reserving a
large frame at forward entry leaves room in that chunk for the nested CuTe launches.
This avoids repeated mappings and UCX unmap callbacks during disaggregated warmup.
"""

from collections.abc import Callable
from struct import calcsize
from typing import ParamSpec, TypeVar

import torch

_Params = ParamSpec("_Params")
_Result = TypeVar("_Result")
_EXTRA_STACK_SLOTS = 512 * 1024 // calcsize("P")


def _call_in_large_frame(
    fn: Callable[_Params, _Result], /, *args: _Params.args, **kwargs: _Params.kwargs
) -> _Result:
    return fn(*args, **kwargs)


# CPython includes the operand stack's maximum depth in the frame allocation.
_call_in_large_frame.__code__ = _call_in_large_frame.__code__.replace(
    co_stacksize=_call_in_large_frame.__code__.co_stacksize + _EXTRA_STACK_SLOTS
)


def run_on_large_datastack_chunk(
    fn: Callable[_Params, _Result], /, *args: _Params.args, **kwargs: _Params.kwargs
) -> _Result:
    """Call the forward on a large host frame in eager mode and directly during tracing."""
    if torch.compiler.is_compiling():
        return fn(*args, **kwargs)
    return _call_in_large_frame(fn, *args, **kwargs)
