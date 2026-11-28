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
"""Eager decoder-layer NCCL-window scopes around the process's first window request.

The record of a window request never resets, so the first request must happen in this
module's own MPI workers.
"""

import pickle
import sys

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer
from tensorrt_llm._torch.nccl_window_tensor_scope import (
    install_eager_nccl_window_tensor_scopes,
    nccl_window_tensor_scope,
)
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)

_SKIP = "skip"
_NUM_LAYERS = 2
_NUM_FORWARDS = 2
# Above the NCCL_SYMMETRIC registration threshold for two ranks, so every allreduce copies its
# input into a window buffer and returns a window-backed output.
_SHAPE = (64, 4096)


@torch.inference_mode()
def run_window_allreduce_layers(tensor_parallel_size: int) -> dict[str, object] | str:
    """Run two forwards of eager decoder layers that each make a window-backed allreduce."""
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=tensor_parallel_size, tp_size=tensor_parallel_size, rank=rank)

    # cloudpickle pickles this module by value, and a direct reference to torch.ops in this
    # function would make it pickle torch.ops itself.
    def has_requested_window_buffers() -> bool:
        return torch.ops.trtllm.has_requested_nccl_window_buffers()

    def is_window_buffer(tensor: torch.Tensor) -> bool:
        return torch.ops.trtllm.is_nccl_window_buffer(tensor, mapping.tp_group)

    class AllReduceLayer(DecoderLayer):
        """Allreduce whose window-backed output is a temporary of the layer call."""

        def __init__(self) -> None:
            super().__init__()
            self.allreduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL_SYMMETRIC)
            self.window_ptrs: list[int] = []
            self.window_backed: list[bool] = []
            self.requested_after_allreduce: list[bool] = []

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            reduced = self.allreduce(hidden_states)
            self.window_ptrs.append(reduced.data_ptr())
            self.window_backed.append(is_window_buffer(reduced))
            self.requested_after_allreduce.append(has_requested_window_buffers())
            return reduced * 0.5

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(AllReduceLayer() for _ in range(_NUM_LAYERS))

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                hidden_states = layer(hidden_states)
            return hidden_states

    model = Model().cuda()
    handles = install_eager_nccl_window_tensor_scopes(model)
    requested_before = has_requested_window_buffers()
    outputs = []
    try:
        for _ in range(_NUM_FORWARDS):
            hidden_states = torch.full(_SHAPE, rank + 1.0, dtype=torch.bfloat16, device="cuda")
            # The enclosing scope of a model forward, as the model engine installs it.
            with nccl_window_tensor_scope(hidden_states) as scope:
                outputs.append(scope.escape(model(hidden_states)).float().cpu())
    finally:
        for handle in handles:
            handle.remove()

    layers = model.layers
    if not all(all(layer.window_backed) for layer in layers):
        # NCCL windows (or NVLink for NCCL_SYMMETRIC registration) are not available here.
        return _SKIP
    return {
        "requested_before": requested_before,
        "requested_in_first_layer_call": layers[0].requested_after_allreduce[0],
        "window_ptrs": [ptr for layer in layers for ptr in layer.window_ptrs],
        "outputs": outputs,
    }


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_first_window_request_in_eager_layer_keeps_buffer_reuse(
    mpi_pool_executor: MPIPoolExecutor,
) -> None:
    """The layer call that makes the first window request releases its window temporaries.

    The next layer, and every layer of the next forward, then reuse that call's window buffer,
    as they do when every layer call is scoped.
    """
    tensor_parallel_size = mpi_pool_executor.num_workers
    results = list(
        mpi_pool_executor.map(
            run_window_allreduce_layers,
            *zip(*[(tensor_parallel_size,)] * tensor_parallel_size),
        )
    )
    if any(result == _SKIP for result in results):
        pytest.skip("NCCL window buffers are not available")

    expected = float(tensor_parallel_size * (tensor_parallel_size + 1)) / 4
    for _ in range(_NUM_LAYERS - 1):
        expected = expected * tensor_parallel_size / 2
    for result in results:
        assert result["requested_before"] is False
        assert result["requested_in_first_layer_call"] is True
        assert len(set(result["window_ptrs"])) == 1, result["window_ptrs"]
        for output in result["outputs"]:
            torch.testing.assert_close(output, torch.full(_SHAPE, expected))
