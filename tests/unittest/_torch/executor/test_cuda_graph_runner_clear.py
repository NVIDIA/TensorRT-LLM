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
"""CUDAGraphRunner.clear() must release the C++ MoE workspaces first.

Guards a SIGSEGV that needs no GPU to prevent but cannot be reproduced without
one: FusedMoeRunner allocates its workspace under isCapturing (moeOp.cpp
getWorkspaceInfo -- the only thop unit that does), so the tensor comes from the
graph's private memory pool, and caches it in mStreamWorkspaces. That map hangs
off MoERunner.runner_dict, a class attribute, so it outlives the executor that
captured the graph. With KV-cache-size estimation on, that executor is torn
down between the two warmups; if the workspace is still held when
empty_cache() erases the pool, the next clear_all_workspaces() frees it against
a dead pool and faults inside the caching allocator's free_block().

Reproducing the fault needs the real allocator layout -- in pure torch a held
block keeps cudaMalloc_count above zero, so the pool is never erased and the
crash cannot be staged. The ordering is checkable without any of that, which is
what this does.
"""

from unittest import mock

from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as cgr


class _FakeGraph:
    def __init__(self, order):
        self._order = order

    def reset(self):
        self._order.append("graph.reset")


def test_clear_releases_moe_workspaces_before_erasing_the_pool():
    order = []

    runner = object.__new__(cgr.CUDAGraphRunner)
    runner.graphs = {("k",): _FakeGraph(order)}
    runner.graph_outputs = {}
    runner.graph_metadata = {}
    runner.padding_dummy_requests = {}
    runner.memory_pool = object()

    with mock.patch(
        "tensorrt_llm._torch.custom_ops.torch_custom_ops.MoERunner.clear_all_workspaces",
        side_effect=lambda: order.append("clear_all_workspaces"),
    ):
        with mock.patch("torch.cuda.empty_cache", side_effect=lambda: order.append("empty_cache")):
            runner.clear()

    assert "clear_all_workspaces" in order, (
        "clear() never released the C++ MoE workspaces; whoever frees them "
        "next does it against a pool this method has already erased"
    )
    # empty_cache() is the deadline: that is where release_cached_blocks()
    # erases the PrivatePool. graph.reset() only drops its use_count, so
    # landing between reset() and empty_cache() would also be correct.
    assert order.index("clear_all_workspaces") < order.index("empty_cache"), (
        f"MoE workspaces must be released before the pool is erased; got {order}"
    )
