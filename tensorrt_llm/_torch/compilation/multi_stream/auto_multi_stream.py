import time
from dataclasses import dataclass, field
from operator import getitem
from queue import PriorityQueue
from typing import Dict, List

import torch
from torch.fx import Graph, GraphModule, Node

from tensorrt_llm.logger import logger

from ..utils import inplace_info


def is_symint_node(node: Node) -> bool:
    if node is not None and 'val' in node.meta:
        # This is a symint call that happens on host. No need to count time on stream.
        if isinstance(node.meta['val'], torch.SymInt):
            return True
    return False


def estimate_time(node: Node) -> int:
    if node is None:
        return 0
    if is_symint_node(node):
        # This is a symint call that happens on host. No need to count time on stream.
        return 0

    # Add cost model for ops that need special handling.
    # We can start with rough estimation and refine it later.

    no_cost_ops = {
        getitem, torch.ops.aten.view.default, torch.ops.aten.view.dtype,
        torch.ops.aten.alias.default, torch.ops.aten.empty.memory_format,
        torch.ops.aten.permute.default
    }

    moe_ops = {
        torch.ops.trtllm.fp4_block_scale_moe_runner.default,
        torch.ops.trtllm.fused_moe.default,
    }

    gemm_ops = {
        torch.ops.aten.mm.default,
        torch.ops.trtllm.nvfp4_gemm.default,
        torch.ops.trtllm.fp8_batched_gemm_trtllmgen.default,
        torch.ops.trtllm.w4a8_mxfp4_fp8_gemm.default,
        torch.ops.trtllm.finegrained_mixed_dtype_gemm.default,
        torch.ops.trtllm.bmm_out.default,
        torch.ops.trtllm.cublas_scaled_mm.default,
        torch.ops.trtllm.cublas_mm.default,
        torch.ops.trtllm.dsv3_router_gemm_op.default,
        torch.ops.trtllm.dsv3_fused_a_gemm_op.default,
        torch.ops.trtllm.fp4_gemm.default,
        torch.ops.trtllm.fp4_bmm.default,
        torch.ops.trtllm.fp8_block_scaling_gemm.default,
        torch.ops.trtllm.matmul_to_ub.default,
    }

    # These ops are not counted in the time estimation.
    if node.op == "call_function" and node.target in no_cost_ops:
        return 0

    # Add estimation below. With accurate estimation, the stream assignment
    # can give the best performance. But it is hard to get accurate estimation.
    #
    # So currently, these estimations are not accurate. They just make sure the key path
    # is correctly scheduled. Adjust the estimation or add new ones
    # if the stream assignment is not desired.

    MOE_OP_COST = 20
    GEMM_OP_COST = 10
    DEFAULT_OP_COST = 1

    # Adjust MOE weight to make the router -> MOE key path
    if node.op == "call_function" and node.target in moe_ops:
        return MOE_OP_COST

    # GEMM ops
    if node.op == "call_function" and node.target in gemm_ops:
        return GEMM_OP_COST

    # Refine the estimation of time for nodes.
    return DEFAULT_OP_COST


@dataclass
class Stream:
    # Stream id
    id: int

    # Nodes running on the stream
    nodes: List['MultiStreamNode'] = field(init=False, default_factory=list)

    # Current elapsed time of the stream
    current_time: int = field(init=False, default=0)


class MultiStreamNode:

    def __init__(self, node: Node, in_edges: Dict[Node, 'MultiStreamNode']):
        # The node in the original graph
        self.node = node

        # The distance to the exit of DAG
        self.distance = 0

        # Weight for the node which represents the computation cost
        self.weight = estimate_time(node)

        # The in edges of the node
        self.in_edges = in_edges

        # The out edges of the node
        self.out_edges = []

        # end time of the node
        self.end_time = 0

        # Assigned stream for the node
        self.stream = None

        # wait on events
        self.wait_on = []

        # trigger event
        self.event = None


class MultiStreamDAG:

    def __init__(self, gm: GraphModule):
        self.gm = gm
        self.node_to_id = {}
        self.node_in_degrees = {}
        self.output_nodes = []
        self.placeholders = []
        self.nodes = {}
        self.in_degrees = {}
        self.work_list = []
        self.entry_node = None
        self.exit_node = None

        self.create_dag_from_gm(gm)
        assert self.entry_node is not None
        assert self.exit_node is not None

    def create_dag_from_gm(self, gm: GraphModule) -> None:
        """
        Create a DAG from the graph module.
        """
        # Create node to id mapping
        for node in gm.graph.nodes:
            self.node_to_id[node] = len(self.node_to_id)

        # Fake entry node.
        # All nodes without in edges will be connected to this node.
        self.entry_node = MultiStreamNode(None, dict())

        latest_inplace_stat = {}
        inplace_map = inplace_info()

        def flatten_args(args):
            """Recursively flatten nested arguments into a flat list."""
            args_new = []
            stack = list(args)
            while stack:
                arg = stack.pop()
                if isinstance(arg, dict):
                    stack.extend(arg.values())
                elif isinstance(arg, (list, tuple)):
                    stack.extend(arg)
                else:
                    args_new.append(arg)
            return args_new

        # Pop all the placeholders from gm
        # We know that the node is already in topological order
        for node in gm.graph.nodes:
            # We assume that all the placeholders are already synced with the base stream
            if node.op == "placeholder":
                self.placeholders.append(node)
                continue

            args = flatten_args([a for a in node.args] +
                                [a for a in node.kwargs.values()])

            in_edges = dict()
            for arg in args:
                if arg in latest_inplace_stat:
                    in_edges[arg] = latest_inplace_stat[arg]
                elif isinstance(arg, torch.fx.Node) and arg.op != "placeholder":
                    in_edges[arg] = self.nodes[arg]

            # For node without in edge, connect it to the entry
            if len(in_edges) == 0:
                in_edges[None] = self.entry_node

            vertex = MultiStreamNode(node, in_edges)
            if node.op == "output":
                self.exit_node = vertex
                vertex.distance = 0
            self.nodes[node] = vertex
            self.in_degrees[vertex] = len(in_edges)
            if node.op == "call_function":
                func = node.target
                if func in inplace_map:
                    for inplace_arg in inplace_map[func].values():
                        # At this stage, all inplace op must be using kwargs for all params
                        assert inplace_arg in node.kwargs
                        latest_inplace_stat[node.kwargs[inplace_arg]] = vertex

            for edge in in_edges.values():
                edge.out_edges.append(vertex)
        self.compute_distance()

    def compute_distance(self) -> None:
        """
        Compute the distance to the exit node for each node.
        """
        # Reverse topological sort to compute distance to exit node
        work_list = [self.exit_node]
        out_degrees = {
            node: len(node.out_edges)
            for node in self.nodes.values()
        }
        out_degrees[self.entry_node] = len(self.entry_node.out_edges)

        while len(work_list) > 0:
            node = work_list.pop()
            for in_edge in node.in_edges.values():
                out_degrees[in_edge] -= 1
                in_edge.distance = max(in_edge.distance,
                                       node.weight + node.distance)
                if out_degrees[in_edge] == 0:
                    work_list.append(in_edge)

    def assign_streams(self, max_num_streams: int) -> int:
        """
        Assign streams to the nodes in the DAG.
        Return the number of events created.
        """
        worklist = PriorityQueue()
        num_nodes = len(self.node_to_id)

        # When accessing node, the distance to the exit node is main priority.
        # The node with largest distance means currently this is the bottleneck of the whole graph.
        def calc_priority(node_id: int, distance: int) -> int:
            # We keep the node order by default.
            # It also gives deterministic order for priority queue.
            return (-distance) * num_nodes + node_id

        streams = [Stream(i) for i in range(max_num_streams)]

        def pick_stream(start_time, node) -> Stream:
            if node.weight == 0:
                # This is a symint node or a getitem node.
                # It always assigns to the stream that produce the node.
                for n in node.in_edges.values():
                    if is_symint_node(n.node):
                        continue
                    return n.stream
                return streams[0]

            closest_stream = None
            least_time = float('inf')
            for st in streams:
                if st.current_time <= start_time:
                    return st
                else:
                    if st.current_time < least_time:
                        least_time = st.current_time
                        closest_stream = st
            return closest_stream

        # We just start from the out_edges of the entry node. Entry node is just a fake node
        # For entry, we assign to the primary stream.
        self.entry_node.stream = streams[0]
        streams[0].nodes.append(self.entry_node)
        for out_edge in self.entry_node.out_edges:
            worklist.put((calc_priority(self.node_to_id[out_edge.node],
                                        out_edge.distance), out_edge))

        sync_event_id = 0

        while not worklist.empty():
            _, node = worklist.get()
            assert node.stream is None

            # Get when current node can start.
            # Start time is the max of the end time of all the in edges.
            start_time = max(
                [in_edge.end_time for in_edge in node.in_edges.values()])
            node.stream = pick_stream(start_time, node)
            node.end_time = max(start_time,
                                node.stream.current_time) + node.weight
            node.stream.current_time = node.end_time
            node.stream.nodes.append(node)

            for in_edge_tensor, in_edge in node.in_edges.items():
                if in_edge.stream != node.stream and not is_symint_node(
                        in_edge.node):
                    if in_edge.event is None:
                        in_edge.event = sync_event_id
                        sync_event_id += 1
                    node.wait_on.append((in_edge, in_edge_tensor))

            # Now, for any in edge running on different stream, we need to create a sync event.
            for out_edge in node.out_edges:
                self.in_degrees[out_edge] -= 1
                if self.in_degrees[out_edge] == 0:
                    worklist.put((calc_priority(self.node_to_id[out_edge.node],
                                                out_edge.distance), out_edge))
        self.streams = streams
        return sync_event_id

    def create_new_graph(self) -> Graph:
        """
        Create new graph with the nodes assigned to the streams.
        """
        # Now each node should have been assigned a stream. We will now create a new graph and insert all nodes
        # As torch need to create node for switching stream, need to group nodes as much as possible.
        remap = {}
        new_graph = Graph()

        for st in self.streams:
            logger.debug(f"{len(st.nodes)} nodes running on stream {st.id}")

        # First, push all placeholders to the new graph.
        for placeholder in self.placeholders:
            remap[placeholder] = new_graph.node_copy(placeholder,
                                                     lambda n: remap[n])

        # Then, we will push all the nodes into the new graph.
        # Build in_degrees again as we need to check whether a stream is ready to run.
        self.in_degrees = {
            node: len(node.in_edges)
            for node in self.nodes.values()
        }
        self.in_degrees[self.entry_node] = 0

        stream_pos = [0] * len(self.streams)

        def has_more_nodes() -> bool:
            for st in self.streams:
                if len(st.nodes) > stream_pos[st.id]:
                    return True
            return False

        last_stream = 0

        # The nodes in stream are already in topological order.
        while has_more_nodes():
            for st in self.streams:
                if len(st.nodes) == stream_pos[st.id]:
                    continue
                node = st.nodes[stream_pos[st.id]]
                if self.in_degrees[node] != 0:
                    # This stream is not ready to run now.
                    continue

                # Any time the stream is changed, set the stream.
                if node.stream.id != last_stream:
                    # Change stream
                    new_graph.create_node("call_function",
                                          torch.ops.trtllm.set_stream,
                                          args=(node.stream.id, ))
                    last_stream = node.stream.id

                for _ in range(stream_pos[st.id], len(st.nodes)):
                    node = st.nodes[stream_pos[st.id]]
                    if self.in_degrees[node] != 0:
                        break
                    for out_edge in node.out_edges:
                        self.in_degrees[out_edge] -= 1
                    stream_pos[st.id] += 1
                    # It could be the fake entry node.
                    if node.node is not None:
                        # Wait on all the events that the node is waiting on.
                        for wait in node.wait_on:
                            new_graph.create_node("call_function",
                                                  torch.ops.trtllm.wait_event,
                                                  args=(wait[0].event, ))
                        remap[node.node] = new_graph.node_copy(
                            node.node, lambda n: remap[n])
                        for wait in node.wait_on:
                            # wait[1] is the actual tensor that the op is waiting on.
                            # Need to record stream for that tensor.
                            if wait[1] is None:
                                continue
                            new_graph.create_node(
                                "call_function",
                                torch.ops.trtllm.record_stream,
                                args=(remap[wait[1]], st.id))
                    if node.event is not None:
                        new_graph.create_node("call_function",
                                              torch.ops.trtllm.record_event,
                                              args=(node.event, ))

                # After each handling, start again to make sure primary stream is pushed first.
                break
        return new_graph

    def optimize(self, max_num_streams: int) -> int:
        """
        Run multistream optimize for MultiStreamDAG. The graph module that used to create the DAG will be updated.
        Return the number of events created.
        """
        num_events = self.assign_streams(max_num_streams)
        new_graph = self.create_new_graph()
        self.gm.graph = new_graph
        return num_events


def multi_stream_schedule(gm: GraphModule, max_num_streams: int) -> int:
    """
    Schedule the graph module for multi stream execution.
    gm is the graph module to be scheduled. The gm will be updated by this function.
    max_num_streams is the maximum number of streams to be used. The scheduler may not use all the streams.
    Return the number of events created.
    """
    dag = MultiStreamDAG(gm)
    return dag.optimize(max_num_streams)


# Following code is for debug purpose. Use print_dag_to_dot to print a MultiStreamDAG to dot file.


def dump_dag_as_dot(dag: MultiStreamDAG, max_num_nodes: int = 500) -> None:
    COLORS = [
        "red", "chocolate", "cyan", "gold", "coral", "green", "blue", "orange",
        "purple", "brown"
    ]
    filename = f"dag_{int(time.time())}.dot"
    with open(filename, 'w') as f:
        f.write("digraph G {\n")
        f.write(
            f"id_entry [label=\"node=entry, distance={dag.entry_node.distance}\"]\n"
        )
        cnt = 0
        for node in dag.nodes.values():
            color = "white" if node.stream is None else COLORS[node.stream.id]
            f.write(
                f"id_{dag.node_to_id[node.node]} [label=\"node={node.node}, "
                f"distance={node.distance}, weight={node.weight}\", "
                f"color={color}, shape=oval]\n")
            for in_edge in node.in_edges.values():
                id = str(dag.node_to_id[
                    in_edge.node]) if in_edge.node is not None else "entry"
                f.write(f"id_{id} -> id_{dag.node_to_id[node.node]}\n")
            if cnt > max_num_nodes:
                break
            cnt += 1
        f.write("}\n")
        f.flush()
