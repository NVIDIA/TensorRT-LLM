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

    # Indicates that captured-graph wrappers must short-circuit to eager.
    # Set by ad_executor.maybe_pad_for_cuda_graph under attention-DP mixed mode
    # so all ranks read kwargs (e.g. batch_info_host slot 14) consistently
    # instead of using stale capture-time scalar kernel args. See
    # BypassCapturedGraphs() below.
    BYPASS: bool = False

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

    def begin_bypass():
        if CudaGraphState.BYPASS:
            raise ValueError("Already in a bypass state")
        CudaGraphState.BYPASS = True

    def end_bypass():
        if not CudaGraphState.BYPASS:
            raise ValueError("Not in bypass state")
        CudaGraphState.BYPASS = False

    def in_bypass() -> bool:
        return CudaGraphState.BYPASS


cuda_graph_state = CudaGraphState


@contextmanager
def CudaGraphWarmUpPhase():
    cuda_graph_state.begin_warm_up()
    try:
        yield
    finally:
        cuda_graph_state.end_warm_up()


@contextmanager
def BypassCapturedGraphs():
    """Force every CapturedGraph wrapper inside this scope to short-circuit to eager.

    Used by ``ad_executor.maybe_pad_for_cuda_graph`` under attention-DP mixed mode:
    when the cross-rank ``tp_allgather`` vote says some ranks must run eager (e.g.
    one rank is in prefill while others are in decode), all ranks enter this
    context for the call so captured graphs whose shapes happen to match are
    bypassed too. Otherwise the captured kernel-launch args (notably the
    ``int(batch_info_host[14].item())`` baked at capture time, where slot 14
    holds ``max_dp_num_tokens`` per ``BatchInfo``) would diverge from the eager
    ranks' fresh reads, corrupting the ``MoeAlltoAll`` collective.
    """
    cuda_graph_state.begin_bypass()
    try:
        yield
    finally:
        cuda_graph_state.end_bypass()
