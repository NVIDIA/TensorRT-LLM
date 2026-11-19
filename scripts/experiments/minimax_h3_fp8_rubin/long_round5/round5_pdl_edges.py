# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate launch PDL through capture edge data; no candidate kernel executes."""

from types import ModuleType


def prove(
    torch: ModuleType,
    cuda: ModuleType,
    module: ModuleType,
    function: object,
    *,
    blocks: int = 848,
    threads: int = 256,
) -> list[dict]:
    def integer(value: int | bytes) -> int:
        return int.from_bytes(value, "little") if isinstance(value, bytes) else int(value)

    raw = torch.zeros((1, 9), device="cuda", dtype=torch.bfloat16)
    scale = torch.ones((1,), device="cuda", dtype=torch.float32)
    factor = torch.ones_like(scale)
    rows = []
    for enabled in (False, True):
        owners = [torch.full((521,), 165, device="cuda", dtype=torch.uint8) for _ in range(2)]
        outputs = [owner[256:265].view(torch.float8_e4m3fn).reshape(1, 9) for owner in owners]
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        try:
            with torch.cuda.graph(graph):
                for output in outputs:
                    module.out_config(raw, scale, factor, output, 0, enabled, blocks, threads)
            assert not graph._has_graph_exec
            handle = cuda.CUgraph(graph.raw_cuda_graph())
            status, _, count = cuda.cuGraphGetNodes(handle, 0)
            assert int(status) == 0 and int(count) == 2
            status, nodes, count = cuda.cuGraphGetNodes(handle, count)
            assert int(status) == 0 and len(nodes) == int(count) == 2
            for node in nodes:
                status, params = cuda.cuGraphKernelNodeGetParams(node)
                assert int(status) == 0 and int(params.func) == int(function)
                assert [params.gridDimX, params.gridDimY, params.gridDimZ] == [blocks, 1, 1]
                assert [params.blockDimX, params.blockDimY, params.blockDimZ] == [
                    threads,
                    1,
                    1,
                ] and params.sharedMemBytes == 0
            status, _, _, _, count = cuda.cuGraphGetEdges(handle, 0)
            assert int(status) == 0 and int(count) == 1
            status, starts, ends, edges, count = cuda.cuGraphGetEdges(handle, count)
            assert int(status) == 0 and len(edges) == int(count) == 1
            assert (
                int(starts[0]) in [int(n) for n in nodes]
                and int(ends[0]) in [int(n) for n in nodes]
                and int(starts[0]) != int(ends[0])
            )
            edge = edges[0]
            observed = dict(
                type=integer(edge.type),
                from_port=integer(edge.from_port),
                to_port=integer(edge.to_port),
            )
            kind = int(
                cuda.CUgraphDependencyType.CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC
                if enabled
                else cuda.CUgraphDependencyType.CU_GRAPH_DEPENDENCY_TYPE_DEFAULT
            )
            port = int(cuda.CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC) if enabled else 0
            assert observed == dict(type=kind, from_port=port, to_port=0)
            assert all(bool((owner == 165).all()) for owner in owners)
            rows.append(
                dict(
                    PDL=enabled,
                    edge=observed,
                    exact_runtime_functions=True,
                    grid=[blocks, 1, 1],
                    block=[threads, 1, 1],
                    poison_unchanged=True,
                    never_instantiated_or_replayed=True,
                    candidate_kernel_launches=0,
                )
            )
        finally:
            graph.reset()
    return rows
