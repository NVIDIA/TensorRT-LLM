# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from functools import wraps as _wraps

from tensorrt_llm._utils import mpi_disabled as _mpi_disabled

if _mpi_disabled():
    raise RuntimeError(
        "Ray requested (TLLM_DISABLE_MPI=1), but not installed. Please install Ray."
    )


def remote(*args, **kwargs):

    def decorator(func):
        # Returns a function that always raises.
        # Decorated class depends on ray, but ray is not installed.
        @_wraps(func)
        def stub_checker(*_, **__):
            raise RuntimeError(
                f'Ray not installed, so the remote function / actor "{func.__name__}" is not available.'
            )

        return stub_checker

    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return decorator(args[0])

    return decorator


def __getattr__(name):
    raise RuntimeError(
        f'Ray not installed, so "ray.{name}" is unavailable. Please install Ray.'
    )
