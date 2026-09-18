import json
import os
import time
from dataclasses import dataclass, field
from operator import getitem
from queue import PriorityQueue
from typing import Dict, List, Optional

import torch
from torch._higher_order_ops.auto_functionalize import (
    auto_functionalized, auto_functionalized_v2)
from torch.fx import Graph, GraphModule, Node

from tensorrt_llm.logger import logger

from ..utils import inplace_info


def is_symint_node(node: Node) -> bool:
    if node is not None and 'val' in node.meta:
        # This is a symint call that happens on host. No need to count time on stream.
        if isinstance(node.meta['val'], torch.SymInt):
            return True
    return False


# Cost model: rough relative device time of one op at small token counts
# (roughly microseconds of a decode step). Only the ratios matter: they decide
# the critical path (priority) and which stream an op lands on.
MOE_MODULE_COST = 60  # moe_custom_op: routing + quantize + sort + expert GEMMs
MOE_OP_COST = 20  # fused expert GEMMs (all local experts in one launch)
GROUPED_GEMM_COST = 12  # one grouped GEMM of a two-launch expert path
GEMM_OP_COST = 10
COMM_OP_COST = 10  # all-reduce / all-to-all: latency bound, exposed
MOE_SORT_COST = 6
ROUTING_OP_COST = 3
QUANT_OP_COST = 2
NORM_OP_COST = 2
DEFAULT_OP_COST = 1

_NO_COST_OPS = {
    getitem, torch.ops.aten.view.default, torch.ops.aten.view.dtype,
    torch.ops.aten.alias.default, torch.ops.aten.empty.memory_format,
    torch.ops.aten.permute.default
}

_ATEN_GEMM_OPS = {torch.ops.aten.mm.default}

# trtllm ops by class name: many are registered only when their backend (CuTe
# DSL, cuda.tile, FlashInfer, ...) is available, so they are matched by op name.
_TRTLLM_OP_COSTS = (
    # The whole MoE forward behind one op (torch.compile wraps the routing
    # kernel, activation quantize, moe_sort, output memset and the expert GEMMs
    # in moe_custom_op). It has to dominate the shared-expert chain that runs
    # next to it: costed as a single fused GEMM it let the scheduler queue the
    # shared-expert gate GEMM behind it on the same stream, and the join then
    # waited for that GEMM after the experts had finished.
    (MOE_MODULE_COST, ("moe_custom_op", )),
    (MOE_OP_COST, (
        "fp4_block_scale_moe_runner",
        "fp8_block_scale_moe_runner",
        "mxfp8_block_scale_moe_runner",
        "fused_moe",
        "cute_dsl_mxfp8_fused_fc12_moe_inplace_rubin",
        "cute_dsl_megamoe_nvfp4_blackwell",
    )),
    (GROUPED_GEMM_COST, (
        "cute_dsl_nvfp4_grouped_gemm_blackwell",
        "cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin",
        "cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_locality_domain_inplace_rubin",
        "cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell",
        "cute_dsl_nvfp4_grouped_gemm_finalize_locality_domain_inplace_rubin",
        "cute_dsl_bf16_gather_grouped_gemm_swiglu_rubin",
        "cute_dsl_bf16_gather_grouped_gemm_swiglu_locality_domain_inplace_rubin",
        "cute_dsl_bf16_grouped_gemm_finalize_inplace_rubin",
        "cute_dsl_bf16_grouped_gemm_finalize_locality_domain_inplace_rubin",
    )),
    (GEMM_OP_COST, (
        "nvfp4_gemm",
        "fp8_batched_gemm_trtllmgen",
        "w4a8_mxfp4_fp8_gemm",
        "finegrained_mixed_dtype_gemm",
        "bmm_out",
        "cublas_scaled_mm",
        "cublas_mm",
        "dsv3_router_gemm_op",
        "dsv3_fused_a_gemm_op",
        "fp4_gemm",
        "fp4_bmm",
        "fp8_block_scaling_gemm",
        "matmul_to_ub",
        "cute_dsl_bf16_gemm_rubin",
        "cute_dsl_bf16_gemm_blackwell",
        "cute_dsl_bf16_gemm_locality_domain_inplace_rubin",
        "cute_dsl_nvfp4_gemm_blackwell",
        "cute_dsl_nvfp4_gemm_inplace_rubin",
        "cute_dsl_nvfp4_gemm_locality_domain_inplace_rubin",
        "cute_dsl_nvfp4_dense_gemm_swiglu_blackwell",
        "cute_dsl_nvfp4_dense_gemm_gelu_blackwell",
        "cute_dsl_fp8_bmm_rubin",
        "cute_dsl_fp8_bmm_blackwell",
        "cute_dsl_bf16_bmm_rubin",
        "cute_dsl_bf16_bmm_blackwell",
        "cute_dsl_bf16_bmm_locality_domain_inplace_rubin",
    )),
    (COMM_OP_COST, (
        "allreduce",
        "mnnvl_fusion_allreduce",
        "moe_finalize_allreduce",
        "userbuffers_allreduce_finalize",
        "allgather",
        "reducescatter",
        "mnnvl_moe_alltoallv",
        "mnnvl_moe_alltoallv_combine",
    )),
    (MOE_SORT_COST, (
        "moe_sort",
    )),
    (ROUTING_OP_COST, (
        "renorm_moe_routing_op",
        "default_moe_routing_op",
        "noaux_tc_op",
        "moe_permute_op",
        "moe_permute",
        "moe_finalize_scale_op",
        "moe_unpermute_inplace",
    )),
    (QUANT_OP_COST, (
        "mxfp8_quantize",
        "fp4_quantize",
        "fp8_quantize",
        "fp8_quantize_1x128",
        "quantize_e4m3_per_tensor",
    )),
    (NORM_OP_COST, (
        "cuda_tile_rms_norm",
        "cuda_tile_rms_norm_fuse_residual_",
        "flashinfer_rmsnorm",
        "flashinfer_fused_add_rmsnorm",
        "flashinfer_fused_add_rmsnorm_quant",
        "fused_qk_norm_rope",
    )),
)

_TRTLLM_OP_COSTS_BY_NAME: Dict[str, int] = {
    name: cost
    for cost, names in _TRTLLM_OP_COSTS
    for name in names
}


def trtllm_op_name(target) -> Optional[str]:
    """``name`` for ``torch.ops.trtllm.<name>`` overloads (C++ or Python custom
    ops), None for anything else. Resolved from the op schema so the lookup does
    not depend on which backends happened to register their ops."""
    schema = getattr(target, "_schema", None)
    if schema is None:
        return None
    namespace, _, name = schema.name.partition("::")
    return name if namespace == "trtllm" else None


def effective_target(node: Node):
    """The op a node runs: unwrap the auto_functionalize HOP so a mutating op
    that is not registered in inplace_info() is still costed as itself."""
    target = node.target
    if target in (auto_functionalized, auto_functionalized_v2) and node.args:
        return node.args[0]
    return target


def estimate_time(node: Node) -> int:
    if node is None:
        return 0
    if is_symint_node(node):
        # This is a symint call that happens on host. No need to count time on stream.
        return 0
    if node.op != "call_function":
        return DEFAULT_OP_COST
    target = effective_target(node)
    if target in _NO_COST_OPS:
        return 0
    if target in _ATEN_GEMM_OPS:
        return GEMM_OP_COST
    name = trtllm_op_name(target)
    if name is not None:
        return _TRTLLM_OP_COSTS_BY_NAME.get(name, DEFAULT_OP_COST)
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
                        args = flatten_args([node.kwargs[inplace_arg]])
                        for arg in args:
                            latest_inplace_stat[arg] = vertex

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


def _dump_schedule(dag: "MultiStreamDAG", directory: str) -> None:
    """Write the piece's nodes (target, cost estimate, stream, edges) as JSON so
    the schedule can be inspected and re-simulated offline.
    Enabled by TLLM_MULTI_STREAM_DUMP=<dir>; one file per scheduled piece and
    process."""
    try:
        os.makedirs(directory, exist_ok=True)
        nodes = []
        for node, v in dag.nodes.items():
            target = str(node.target)
            if "auto_functionalized" in target and node.args:
                # Record the mutating op wrapped by the functionalization HOP.
                target = f"{target}[{node.args[0]}]"
            val = node.meta.get("val") if isinstance(node.meta, dict) else None
            shape = None
            if isinstance(val, torch.Tensor):
                shape = [str(d) for d in val.shape] + [str(val.dtype)]
            elif isinstance(val, (tuple, list)):
                shape = [[str(d) for d in t.shape] + [str(t.dtype)]
                         if isinstance(t, torch.Tensor) else str(t) for t in val]
            nodes.append({
                "name": node.name,
                "op": node.op,
                "target": target,
                "shape": shape,
                "weight": v.weight,
                "distance": v.distance,
                "stream": None if v.stream is None else v.stream.id,
                "end_time": v.end_time,
                "in_edges": [e.node.name for e in v.in_edges.values() if e.node is not None],
                "waits_on": [e.node.name for e, _ in v.wait_on if e.node is not None],
            })
        fn = os.path.join(
            directory,
            f"piece_{os.getpid()}_{_dump_schedule.counter}.json")
        _dump_schedule.counter += 1
        with open(fn, "w") as f:
            json.dump({"streams": [len(st.nodes) for st in dag.streams],
                       "nodes": nodes}, f)
        logger.debug(f"multi-stream schedule dumped to {fn}")
    except Exception as e:  # never let a debug dump break compilation
        logger.warning(f"multi-stream schedule dump failed: {e}")


_dump_schedule.counter = 0


def multi_stream_schedule(gm: GraphModule, max_num_streams: int) -> int:
    """
    Schedule the graph module for multi stream execution.
    gm is the graph module to be scheduled. The gm will be updated by this function.
    max_num_streams is the maximum number of streams to be used. The scheduler may not use all the streams.
    Return the number of events created.
    """
    dag = MultiStreamDAG(gm)
    num_events = dag.assign_streams(max_num_streams)
    dump_dir = os.environ.get("TLLM_MULTI_STREAM_DUMP")
    if dump_dir:
        _dump_schedule(dag, dump_dir)
    new_graph = dag.create_new_graph()
    dag.gm.graph = new_graph
    return num_events


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
