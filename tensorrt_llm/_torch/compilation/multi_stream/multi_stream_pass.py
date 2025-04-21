from operator import getitem

import torch
from torch.fx import GraphModule, Node

from .range_config import get_range


def multi_stream_pass(gm: GraphModule):

    cpu_node_parent = {}
    record_nodes = []
    set_stream_nodes = []
    side_stream_end_nodes = []
    aux_stream = [torch.cuda.Stream()]
    scheduled_nodes = set()

    node_to_idx = {}

    for idx, node in enumerate(gm.graph.nodes):
        node_to_idx[node] = idx
        if node.op == "call_function" and node.target == getitem:
            cpu_node_parent[node] = node.args[0]
        else:
            cpu_node_parent[node] = node

    for node in gm.graph.nodes:
        start_node, path_top_nodes, aux_stream_num = get_range(
            node, cpu_node_parent)
        if start_node is not None:
            assert isinstance(path_top_nodes, set)
            # Current only support 1 aux stream
            assert aux_stream_num == 1
            dangling_nodes = set()
            path = [set(), set()]
            if len(path_top_nodes) != 2:
                continue
            start_idx = node_to_idx[start_node]

            path_top_nodes = list(path_top_nodes)

            def find_parallel_path():
                if path_top_nodes[0] is None:
                    idx = 1
                elif path_top_nodes[1] is None:
                    idx = 0
                else:
                    idx = 0 if node_to_idx[path_top_nodes[0]] < node_to_idx[
                        path_top_nodes[1]] else 1
                process_node = path_top_nodes[idx]
                if process_node in dangling_nodes:
                    dangling_nodes.remove(process_node)
                path[idx].add((node_to_idx[process_node], process_node))

                umet_dep = set()
                for arg in process_node.args:

                    def handle_arg(arg):
                        if (isinstance(arg, Node)
                                and node_to_idx[arg] >= start_idx):
                            path[idx].add((node_to_idx[arg], arg))
                            path[idx].add((node_to_idx[cpu_node_parent[arg]],
                                           cpu_node_parent[arg]))
                            umet_dep.add(cpu_node_parent[arg])

                    # We assume the inputs are not modified by inplace ops.
                    # To-Do: add inplace op support to find the real dependent nodes
                    if isinstance(arg, list):
                        for i in arg:
                            handle_arg(i)
                    else:
                        handle_arg(arg)

                if start_node in umet_dep:
                    umet_dep.remove(start_node)

                last_node = None
                for node in umet_dep:
                    if node.op == "call_function" and node.target == torch.ops.aten.t.default:
                        arg_node = node.args[0]
                        if arg_node.op == "placeholder":
                            path[idx].add((node_to_idx[node], node))
                        continue
                    if last_node is None:
                        last_node = node
                    else:
                        if node_to_idx[node] > node_to_idx[last_node]:
                            last_node = node

                if last_node is not None:
                    umet_dep.remove(last_node)

                path_top_nodes[idx] = last_node

                if process_node in path_top_nodes:
                    path_top_nodes.remove(process_node)
                dangling_nodes.update(umet_dep)

                if all([i == None for i in path_top_nodes]):
                    return
                else:
                    find_parallel_path()

            find_parallel_path()

            path = [sorted(list(i)) for i in path]

            for p in path:
                dangling_nodes.difference_update([i[1] for i in p])
                if (start_idx, start_node) in p:
                    p.remove((start_idx, start_node))
                while len(p) > 0 and p[0][1].target == getitem:
                    p.pop(0)

                prev_idx = p[0][0] - 1
                for idx, node in p:
                    assert idx == prev_idx + 1
                    prev_idx = idx

            if len(dangling_nodes) == 0:
                if path[0][0][1] > path[1][0][1]:
                    path[0], path[1] = path[1], path[0]
                flattened_path = [i[1] for j in path for i in j]
                if scheduled_nodes.isdisjoint(flattened_path):
                    record_nodes.append(path[0][0][1])
                    set_stream_nodes.append(path[1][0][1])
                    side_stream_end_nodes.append(path[1][-1][1])
                    scheduled_nodes.update(flattened_path)

    graph = gm.graph
    for r, s, e in zip(record_nodes, set_stream_nodes, side_stream_end_nodes):
        with graph.inserting_before(r):
            graph.create_node("call_function",
                              torch.ops.trtllm.record_event.default,
                              args=(0, ))

        with graph.inserting_before(s):
            graph.create_node("call_function",
                              torch.ops.trtllm.set_stream.default,
                              args=(aux_stream[0].cuda_stream, ))
            graph.create_node("call_function",
                              torch.ops.trtllm.wait_event.default,
                              args=(0, ))

        with graph.inserting_after(e):
            graph.create_node("call_function",
                              torch.ops.trtllm.wait_event.default,
                              args=(1, ))
            graph.create_node("call_function",
                              torch.ops.trtllm.set_stream.default,
                              args=(-1, ))
            graph.create_node("call_function",
                              torch.ops.trtllm.record_event.default,
                              args=(1, ))

    for node in graph.nodes:
        if node.op != "placeholder":
            with graph.inserting_before(node):
                graph.create_node("call_function",
                                  torch.ops.trtllm.get_current_stream.default)
            break

    return gm, aux_stream
