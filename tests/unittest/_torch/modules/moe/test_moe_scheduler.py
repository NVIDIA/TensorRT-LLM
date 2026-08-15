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

"""Focused tests for MoE scheduler chunk execution."""

from types import MethodType

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe.moe_scheduler import ExternalCommMoEScheduler


class _DummyMoe:
    use_dp = False
    enable_alltoall = False
    aux_stream = None
    backend = object()
    repeat_idx = 0
    repeat_count = 1

    @staticmethod
    def split_chunk(num_tokens: int, num_chunks: int) -> list[int]:
        quotient, remainder = divmod(num_tokens, num_chunks)
        return [quotient + (chunk_idx < remainder) for chunk_idx in range(num_chunks)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_multichunk_forward_is_cuda_graph_capturable() -> None:
    """Host-only chunk bookkeeping must not become a CUDA tensor."""
    scheduler = ExternalCommMoEScheduler(_DummyMoe())

    def forward_chunk(_self, x_chunk, _router_logits, *_args, **_kwargs):
        return x_chunk + 1

    scheduler._forward_chunk_impl = MethodType(forward_chunk, scheduler)
    x = torch.arange(8, dtype=torch.float32, device="cuda").reshape(4, 2)
    router_logits = torch.zeros((4, 2), dtype=torch.float32, device="cuda")

    def forward() -> torch.Tensor:
        return scheduler._forward_multiple_chunks(
            x,
            router_logits,
            num_chunks=2,
            output_dtype=None,
            all_rank_num_tokens=[4],
            use_dp_padding=False,
        )

    with torch.device("cuda"):
        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = forward()
        graph.replay()
        torch.cuda.synchronize()

    torch.testing.assert_close(output, x + 1)
