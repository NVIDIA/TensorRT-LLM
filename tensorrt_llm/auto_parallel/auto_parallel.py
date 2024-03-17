import gc
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tensorrt as trt
import torch
from filelock import FileLock

from tensorrt_llm.functional import DimRange, Tensor
from tensorrt_llm.logger import logger
from tensorrt_llm.network import Network, net_guard

from .config import AutoParallelConfig
from .device_mesh import LogicalDeviceMesh, PhysicalDeviceMesh
from .node_graph import NodeGraph
from .parallelization import ParallelConfig, parallelize
from .pipeline_graph import PipelineGraph
from .simplifier import GraphConfig, Simplifier, StageType
from .utils import current_flags


def to_network(graph: PipelineGraph, network: Network):
    logger.debug("Converting graph to network")
    trt_network = graph.as_trt()
    trt_network.name = network.trt_network.name
    new_network = Network()
    new_network._init(trt_network)
    new_network._dtype = network._dtype
    new_network._plugin_config = network._plugin_config
    new_network._unfilled_weights = graph._unfilled_weights
    new_network._auto_parallel_config = graph._auto_parallel_config
    with net_guard(network):
        for i in range(trt_network.num_inputs):
            input = trt_network.get_input(i)
            tensor = Tensor(is_network_input=False)
            if input.name in network._inputs:
                profiles = network._inputs[input.name].profiles
            elif len(network._inputs) == 0:
                profiles = []
            else:
                shape = input.shape
                num_profiles = len(list(network._inputs.values())[0].profiles)
                profile = DimRange(shape, [None] * len(shape))
                profiles = [profile] * num_profiles
            tensor.profiles = profiles
            tensor.trt_tensor = input
            new_network._inputs[input.name] = tensor
    return new_network


def find_solution(
    node_graph: NodeGraph,
    graph_config: GraphConfig,
    lmesh: LogicalDeviceMesh,
    memory_budget: int,
    flags: list,
    device: int,
    dump_path: str,
) -> ParallelConfig:
    torch.cuda.set_device(device)
    with current_flags(*flags):
        cost_graph = node_graph.get_cost_graph(lmesh)
        num_stages = graph_config.num_stages
        if num_stages == 1:
            stage_types = [None]
        elif num_stages == 2:
            stage_types = [StageType.START, StageType.END]
        else:
            stage_types = [StageType.START, StageType.BLOCK, StageType.END]

        best_config, best_solution = None, None
        for stage_type in stage_types:
            if stage_type is not None:
                node_graph.set_slowest_stage(stage_type, graph_config)
            solution = node_graph.find_solution(
                cost_graph,
                memory_budget,
            )
            cost = solution.total_cost
            if best_config is None or cost < best_config.cost:
                best_config = ParallelConfig()
                best_config.graph_config = graph_config
                best_config.lmesh = lmesh
                best_config.cost = cost
                best_config.graph_strategy = solution.node_best_strategy
                best_config.stage_type = stage_type
                best_solution = solution
        if dump_path is not None:
            lock = FileLock(f"{dump_path}/path.lock", thread_local=False)
            vlz_name = f"{dump_path}/solution."
            if graph_config.num_micro_batches != 1:
                vlz_name += f"mbs{graph_config.num_micro_batches}."
            if graph_config.num_stages != 1:
                vlz_name += f"stages{graph_config.num_stages}."
            vlz_name += lmesh.cluster_key
            with lock:
                node_graph.visualize_solution(
                    best_solution,
                    vlz_name,
                    ignore_shape_io=True,
                )
        return best_config


def infer_builder_flags(network):
    fp16_enabled = False
    bf16_enabled = False
    int8_enabled = False
    fp8_enabled = False

    def check_dtype(tensor):
        nonlocal fp16_enabled
        nonlocal bf16_enabled
        nonlocal int8_enabled
        nonlocal fp8_enabled
        if tensor.dtype == trt.DataType.HALF:
            fp16_enabled = True
        elif tensor.dtype == trt.DataType.BF16:
            bf16_enabled = True
        elif tensor.dtype == trt.DataType.INT8:
            int8_enabled = True
        elif tensor.dtype == trt.DataType.FP8:
            fp8_enabled = True

    trt_network = network.trt_network
    for i in range(trt_network.num_inputs):
        input = trt_network.get_input(i)
        check_dtype(input)
    for i in range(trt_network.num_layers):
        layer = trt_network.get_layer(i)
        for j in range(layer.num_outputs):
            output = layer.get_output(j)
            check_dtype(output)

    builder_flags = 0
    if fp16_enabled:
        builder_flags |= 1 << int(trt.BuilderFlag.FP16)
        builder_flags |= 1 << int(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    if bf16_enabled:
        builder_flags |= 1 << int(trt.BuilderFlag.BF16)
        builder_flags |= 1 << int(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    if int8_enabled:
        builder_flags |= 1 << int(trt.BuilderFlag.INT8)
    if fp8_enabled:
        builder_flags |= 1 << int(trt.BuilderFlag.FP8)
        builder_flags |= 1 << int(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    return builder_flags


def auto_parallel(network: Network, config: AutoParallelConfig):
    debug_mode = config.debug_mode
    memory_budget = config.get_cluster_info(
    ).memory_budget_per_device * 1024 * 1024 * 1024
    enable_pipeline_parallelism = config.enable_pipeline_parallelism
    if config.world_size < config.gpus_per_node:
        num_hosts = 1
        num_devices_per_host = config.world_size
    else:
        assert config.world_size % config.gpus_per_node == 0
        num_hosts = config.world_size // config.gpus_per_node
        num_devices_per_host = config.gpus_per_node
    parallel_config_cache = config.parallel_config_cache
    dump_path = config.dump_path if debug_mode else None
    fill_weights = config.fill_weights

    if num_hosts == 1 and num_devices_per_host == 1:
        return [network]

    if dump_path is not None:
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)

    builder_flags = config.builder_flags or infer_builder_flags(network)
    flags = [builder_flags, network.strongly_typed]
    with current_flags(*flags):
        simplifier = Simplifier(network, config)
        network_hash = simplifier.get_network_hash()

        best_config = None
        if parallel_config_cache is not None and Path(
                parallel_config_cache).exists():
            parallel_config = ParallelConfig.from_file(parallel_config_cache)
            if (ParallelConfig.VERSION == parallel_config.version
                    and network_hash == parallel_config.network_hash
                    and config == parallel_config.auto_parallel_config):
                logger.info(
                    f"use cache of parallel config from {parallel_config_cache}"
                )
                best_config = parallel_config

        if best_config is None:
            num_devices = num_hosts * num_devices_per_host
            phy_ids = [[
                i + j * num_devices_per_host
                for i in range(num_devices_per_host)
            ] for j in range(num_hosts)]
            phy_mesh = PhysicalDeviceMesh(phy_ids, config)
            if enable_pipeline_parallelism:
                num_micro_batches_list = simplifier.list_all_num_micro_batches()
            else:
                num_micro_batches_list = [1]

            jobs = []
            for num_micro_batches in num_micro_batches_list:
                simplifier.infer_shapes(num_micro_batches)
                if enable_pipeline_parallelism:
                    pipeline_configs = phy_mesh.list_all_pipeline_configs()
                else:
                    pipeline_configs = [(1, num_devices)]
                for num_stages, num_devices_per_stage in pipeline_configs:
                    # TODO: add fallback path that allows num_micro_batches >= num_stages
                    #       if no solution satisfies memory budget
                    if num_micro_batches < num_stages:
                        continue
                    simplified_graph, graph_config = simplifier.simplify_graph(
                        phy_mesh,
                        num_stages,
                        num_devices_per_stage,
                    )
                    if simplified_graph is None:
                        continue
                    node_graph = NodeGraph(simplified_graph)
                    node_graph.assign_cost_weights(graph_config)
                    lmeshes = graph_config.stage_phy_meshes[
                        0].get_logical_meshes()
                    for lmesh in lmeshes:
                        jobs.append(
                            (node_graph, graph_config, lmesh, memory_budget *
                             (num_devices / num_devices_per_stage)))

            try:
                with ThreadPoolExecutor() as executor:
                    best_config = sorted(
                        executor.map(
                            lambda x: find_solution(
                                *x,
                                flags,
                                torch.cuda.current_device(),
                                dump_path,
                            ),
                            jobs,
                        ),
                        key=lambda x: x.cost,
                    )[0]
            finally:
                phy_mesh.close()

            if parallel_config_cache is not None:
                best_config.network_hash = network_hash
                best_config.auto_parallel_config = config
                best_config.save(parallel_config_cache)

        new_graphs = parallelize(simplifier, best_config)

    networks = [to_network(new_graph, network) for new_graph in new_graphs]
    if debug_mode and fill_weights:
        networks[0]._fill_weights()

    gc.collect()
    torch.cuda.empty_cache()

    return networks
