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
"""``rubin_l2_prefetch``: byte ranges of dense tensors, and the prefetch kernel leaves memory
untouched eagerly, on a side stream and under CUDA graph replay."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 GPU",
)


def _mod():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm import (
        rubin_l2_prefetch,
    )

    return rubin_l2_prefetch


def test_tensor_range():
    m = _mod()
    w = torch.empty(256, 384, dtype=torch.float8_e4m3fn, device="cuda")
    assert m.tensor_range(w) == (w.data_ptr(), 256 * 384)
    # DeepGEMM-packed scale: int32 [N, words] with strides (1, N), dense in memory.
    packed = torch.empty(3, 256, dtype=torch.int32, device="cuda").t()
    assert m.tensor_range(packed) == (packed.data_ptr(), 3 * 256 * 4)
    # sizes are rounded inwards to 16 bytes
    odd = torch.empty(37, dtype=torch.bfloat16, device="cuda")
    addr, nbytes = m.tensor_range(odd)
    assert addr == odd.data_ptr() and nbytes == 64
    # a strided slice is not dense
    with pytest.raises(ValueError):
        m.tensor_range(w[:, :128])


def test_launch_ctas():
    m = _mod()
    assert m.launch_ctas([(0, 16)]) == 1
    assert m.launch_ctas([(0, 32 * m.CHUNK_BYTES)]) == 1
    assert m.launch_ctas([(0, 32 * m.CHUNK_BYTES + 16)]) == 2
    assert m.launch_ctas([(0, 1 << 30)]) == m.DEFAULT_MAX_CTAS
    assert m.launch_ctas([(0, 1 << 30)], max_ctas=7) == 7


def _buffers():
    gen = torch.Generator(device="cuda").manual_seed(0)
    sizes = (16, 4096 + 48, 3 << 20, (5 << 20) + 16)
    return [
        torch.randint(-1000, 1000, (n // 4,), generator=gen, device="cuda", dtype=torch.int32)
        for n in sizes
    ]


def test_prefetch_eager_and_side_stream():
    m = _mod()
    bufs = _buffers()
    refs = [b.clone() for b in bufs]
    m.run_k3_l2_prefetch([m.tensor_range(b) for b in bufs])
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        m.run_k3_l2_prefetch([m.tensor_range(b) for b in bufs], max_ctas=3)
    torch.cuda.current_stream().wait_stream(side)
    # empty ranges launch nothing
    m.run_k3_l2_prefetch([(bufs[0].data_ptr(), 0)])
    torch.cuda.synchronize()
    for b, r in zip(bufs, refs):
        assert torch.equal(b, r)
    with pytest.raises(ValueError):
        m.run_k3_l2_prefetch([(bufs[0].data_ptr() + 4, 16)])
    with pytest.raises(ValueError):
        m.run_k3_l2_prefetch([(bufs[0].data_ptr(), 16)] * (m.MAX_RANGES + 1))


def test_prefetch_cuda_graph_side_stream():
    """Captured on a forked stream next to two kernels on the main stream."""
    m = _mod()
    bufs = _buffers()
    refs = [b.clone() for b in bufs]
    x = torch.randn(1 << 16, device="cuda")
    y = torch.empty_like(x)
    main, side = torch.cuda.Stream(), torch.cuda.Stream()
    fork, join = torch.cuda.Event(), torch.cuda.Event()

    def body():
        torch.mul(x, 2.0, out=y)
        fork.record(main)
        with torch.cuda.stream(side):
            side.wait_event(fork)
            m.run_k3_l2_prefetch([m.tensor_range(b) for b in bufs])
            join.record(side)
        torch.add(y, 1.0, out=y)
        main.wait_event(join)

    with torch.cuda.stream(main):
        body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=main):
        body()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, x * 2.0 + 1.0)
    for b, r in zip(bufs, refs):
        assert torch.equal(b, r)
