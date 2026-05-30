# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from contextlib import contextmanager


class CudaGraphState:
    """A singleton class used to broadcast the state during cuda graph capture."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create a new instance if it doesn't exist
            cls._instance = super().__new__(cls)
        return cls._instance

    # Indicates the warm-up phase of cuda graph capture when
    # the graph is executed with representative inputs.
    WARM_UP: bool = False

    def begin_warm_up():
        if CudaGraphState.WARM_UP:
            raise ValueError("Already in a warm-up state")
        CudaGraphState.WARM_UP = True

    def end_warm_up():
        if not CudaGraphState.WARM_UP:
            raise ValueError("Not in warm-up state")
        CudaGraphState.WARM_UP = False

    def in_warm_up() -> bool:
        return CudaGraphState.WARM_UP


cuda_graph_state = CudaGraphState


@contextmanager
def CudaGraphWarmUpPhase():
    cuda_graph_state.begin_warm_up()
    try:
        yield
    finally:
        cuda_graph_state.end_warm_up()
