# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""maybe_execute_in_parallel: the tensors fn1 returns stay valid for the calling stream.

fn1 runs on aux_stream, so its outputs are allocated in aux_stream order, while the caller
reads them on its own stream. Once the caller drops them, the allocator must not give their
memory to new aux_stream work until the caller's queued reads have run: the caching allocator
returns a freed block to the pool of the stream it was allocated on, and cudaMallocAsync frees
it in that stream's order. Under CUDA graph capture the outputs are not recorded; dropping them
inside a capture must neither grow the graph pool nor fail the capture. A compiled caller must
still trace as one graph. The tests need no sanitizer. They run under the process's allocator backend, and
test_this_file_under_cuda_malloc_async reruns them under cudaMallocAsync.
"""

import os
import subprocess
import sys

import pytest
import torch

from tensorrt_llm._torch.modules.multi_stream_utils import (
    maybe_execute_in_parallel,
    with_multi_stream,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# About 25 ms at 2 GHz: the calling stream is still sleeping when the host has dropped fn1's
# output and queued the next aux_stream fill, which takes microseconds.
_SLEEP_CYCLES = 50_000_000


def _wrap(out: torch.Tensor, shape: str):
    if shape == "tuple_list":
        return (None, [out])
    if shape == "dict":
        return {"out": out, "none": None}
    return out


def _unwrap(result, shape: str) -> torch.Tensor:
    if shape == "tuple_list":
        return result[1][0]
    if shape == "dict":
        return result["out"]
    return result


@pytest.mark.parametrize("shape", ["tensor", "tuple_list", "dict"])
def test_aux_outputs_outlive_the_calling_streams_reads(shape: str) -> None:
    numel = 1 << 20
    aux_stream = torch.cuda.Stream()
    event0, event1 = torch.cuda.Event(), torch.cuda.Event()

    def fn1():
        return _wrap(torch.full((numel,), 1.0, device="cuda"), shape)

    torch.cuda.synchronize()
    # A free block of this size left in aux_stream's pool by earlier work could take the fill
    # below instead of fn1's block, and hide a missing record.
    torch.cuda.empty_cache()
    with with_multi_stream(True):
        _, result1 = maybe_execute_in_parallel(lambda: None, fn1, event0, event1, aux_stream)
    out = _unwrap(result1, shape)
    torch.cuda._sleep(_SLEEP_CYCLES)
    # Queued behind the sleep: this read of fn1's output runs after everything below.
    copy = out.clone()
    del out, result1
    # The next aux_stream allocation of the same size, filled at once (aux_stream is idle).
    # Without the record it takes fn1's freed memory and overwrites it before the read above.
    with torch.cuda.stream(aux_stream):
        torch.full((numel,), 2.0, device="cuda")
    torch.cuda.synchronize()
    assert torch.equal(copy, torch.ones_like(copy))


def test_aux_output_over_memory_torch_did_not_allocate() -> None:
    # The native allocator skips such memory in record_stream; cudaMallocAsync raises on it.
    cudart = pytest.importorskip("cuda.bindings.runtime")
    numel = 1024
    err, ptr = cudart.cudaMalloc(numel * 4)
    assert err == cudart.cudaError_t.cudaSuccess

    class _Buffer:
        def __init__(self):
            self.__cuda_array_interface__ = {
                "shape": (numel,),
                "typestr": "<f4",
                "data": (int(ptr), False),
                "strides": None,
                "version": 3,
            }

    try:
        foreign = torch.as_tensor(_Buffer(), device="cuda")
        aux_stream = torch.cuda.Stream()
        event0, event1 = torch.cuda.Event(), torch.cuda.Event()
        with with_multi_stream(True):
            _, out = maybe_execute_in_parallel(
                lambda: None, lambda: foreign.fill_(3.0), event0, event1, aux_stream
            )
        torch.cuda.synchronize()
        assert out.data_ptr() == int(ptr)
        assert torch.equal(out, torch.full_like(out, 3.0))
    finally:
        torch.cuda.synchronize()
        cudart.cudaFree(ptr)


def test_graph_capture_with_aux_outputs() -> None:
    aux_stream = torch.cuda.Stream()
    event0, event1 = torch.cuda.Event(), torch.cuda.Event()
    x = torch.randn(4096, device="cuda")

    def fn0():
        return x * 2

    def fn1():
        return x + 1, [x - 1], {"y": x * 3}

    graph = torch.cuda.CUDAGraph()
    with with_multi_stream(True):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            maybe_execute_in_parallel(fn0, fn1, event0, event1, aux_stream)
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(graph):
            r0, (a, [b], c) = maybe_execute_in_parallel(fn0, fn1, event0, event1, aux_stream)
            out = r0 + a + b + c["y"]
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, x * 2 + (x + 1) + (x - 1) + x * 3)


def test_aux_outputs_dropped_inside_graph_capture() -> None:
    layers, numel = 8, 1 << 20
    aux_stream = torch.cuda.Stream()
    event0, event1 = torch.cuda.Event(), torch.cuda.Event()
    # Values below 2^24: the fp32 sums are exact.
    x = (torch.arange(numel, device="cuda") % 4096).to(torch.float32)
    addresses = []

    def layer_stack():
        acc = torch.zeros_like(x)
        for layer in range(layers):
            _, out = maybe_execute_in_parallel(
                lambda: None, lambda layer=layer: x + layer, event0, event1, aux_stream
            )
            addresses.append(out.data_ptr())
            acc = acc + out
            del out
        # cudaMallocAsync frees the outputs in aux_stream order: the capture must join it.
        torch.cuda.current_stream().wait_stream(aux_stream)
        return acc

    graph = torch.cuda.CUDAGraph()
    with with_multi_stream(True):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            layer_stack()
        torch.cuda.current_stream().wait_stream(side)
        addresses.clear()
        with torch.cuda.graph(graph):
            acc = layer_stack()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(acc, x * layers + sum(range(layers)))
    if torch.cuda.memory.get_allocator_backend() == "native":
        # A recorded block freed during capture is held until the capture ends.
        assert len(set(addresses)) == 1, f"{len(set(addresses))} blocks for {layers} outputs"


def test_compiled_caller_traces_fullgraph() -> None:
    # Dynamo traces neither the capture query nor record_stream, so a compiled caller must take
    # the path without the record and still give one graph.
    aux_stream = torch.cuda.Stream()
    event0, event1 = torch.cuda.Event(), torch.cuda.Event()

    def caller(x):
        a, b = maybe_execute_in_parallel(lambda: x * 2, lambda: x + 1, event0, event1, aux_stream)
        return a + b

    torch._dynamo.reset()
    compiled = torch.compile(caller, backend="eager", fullgraph=True)
    x = torch.randn(16, device="cuda")
    with with_multi_stream(True):
        out = compiled(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, 3 * x + 1)


def test_this_file_under_cuda_malloc_async() -> None:
    if torch.cuda.memory.get_allocator_backend() == "cudaMallocAsync":
        pytest.skip("this process already runs under cudaMallocAsync")
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", __file__],
        capture_output=True,
        text=True,
        env=env,
        timeout=1200,
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-3000:]
