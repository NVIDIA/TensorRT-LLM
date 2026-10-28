# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model Streamer helpers that `ModelLoader` calls for ``checkpoint_io_policy="model_streamer"``.

Run:ai Model Streamer reads each checkpoint byte once per node and shares it
among the ranks of the node, so large checkpoints load faster. A model
describes the tensors it needs and how to load them in
``plan_model_streamer_load()``, which returns a `ModelStreamerPlan`.
"""

import json
import os
import socket
import time
from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from strenum import StrEnum
from torch import nn

from tensorrt_llm.logger import logger


@dataclass
class ModelStreamerPlan:
    """Tensor names and functions provided by an LLM model to work with ModelStreamer.

    An LLM model should return this from its function
    ``plan_model_streamer_load(checkpoint_keys)``.
    `load_weights_with_model_streamer` calls this function to get a
    `ModelStreamerPlan` to stream tensors.
    """

    # Some tensors need to be processed together by the LLM model, e.g.
    # it may collect tensor "gate_proj" and "up_proj" to create a quantized
    # "gate_up_proj".
    # So we have `groups` where each element is a (tensor_names, func) pair
    # to define this logic for the current rank.
    # The streaming code will call ``func`` when the required tensors have
    # arrived.
    # Notes:
    # - A tensor can appear in two groups. The streaming code
    #   handles that.
    # - Only put tensors that the current rank needs.
    groups: list[tuple[Collection[str], Callable[[dict[str, torch.Tensor]], None]]]
    # A function to assign a batch index to each tensor name so we can batch
    # tensor processing to avoid OOM during streaming.
    # Note this function is global: it should work on all tensors of the LLM
    # model regardless of the current rank.
    batch_index: Optional[Callable[[str], int]] = None
    # Called after streaming is done to conduct ModelStreamer-specific
    # finalization.
    # Note the model loading code still calls an LLM model's general post load
    # function post_load_weights() after that.
    finish: Optional[Callable[[], None]] = None


class ModelStreamerMetricNames(StrEnum):
    """Names of the metrics that `load_weights_with_model_streamer` logs after a load.

    They stay out of `ModelLoader.metrics`, so the startup metrics do not change.
    Each member is a ``str``, so it works directly as a key of the metrics dict
    that `run_model_streamer()` returns.
    """

    # Time of streaming all batches, in seconds.
    MODEL_STREAMER_SECONDS = "model_streamer_seconds"
    # Number of batches streamed, one `SafetensorsStreamer.stream_files()` call each.
    MODEL_STREAMER_BATCHES = "model_streamer_batches"
    # Tensors and bytes streamed for all ranks on this node.
    MODEL_STREAMER_NODE_TENSORS = "model_streamer_node_tensors"
    MODEL_STREAMER_NODE_BYTES = "model_streamer_node_bytes"
    # Tensors that this rank needs, a subset of the node's tensors.
    MODEL_STREAMER_RANK_TENSORS = "model_streamer_rank_tensors"


def get_weight_source_map(checkpoint_dir: str) -> dict[str, str]:
    """Map each tensor name in a safetensors checkpoint to the file that holds it.

    Example return:
        {
            "model.embed_tokens.weight": "/ckpt/model-00001-of-00002.safetensors",
            "lm_head.weight": "/ckpt/model-00002-of-00002.safetensors",
            ...
        }
    """
    directory = Path(checkpoint_dir)
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        with index.open() as index_file:
            weight_source_file_map = {
                key: str(directory / name)
                for key, name in json.load(index_file)["weight_map"].items()
            }
    else:
        weight_source_file_map = {}
        for path in sorted(str(path) for path in directory.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as file:
                keys = set(file.keys())
            overlap = keys.intersection(weight_source_file_map)
            if overlap:
                raise ValueError(f"Duplicate safetensors keys: {sorted(overlap)[:10]}")
            weight_source_file_map.update(dict.fromkeys(keys, path))
    paths = set(weight_source_file_map.values())
    if not paths or any(not os.path.isfile(path) for path in paths):
        raise FileNotFoundError(f"No complete safetensors checkpoint in {checkpoint_dir}")
    return weight_source_file_map


def gather_tensor_names_from_rank_to_node(
    local_keys: Collection[str],
    *,
    rank: int,
    world_size: int,
) -> tuple[set[str], int]:
    """Combine the tensor names that the ranks on this node need.

    ModelStreamer will stream all tensors on this node together, which is tested
    to be faster than:
    - Each rank uses ModelStreamer to stream its own weights independently.
    - All ranks participate in streaming all the tensors of an LLM.

    Returns:
        The names that the ranks on this node need, and the number of ranks
        on this node.

    Raises:
        RuntimeError: If no gathered entry has this rank's hostname, or if
            `ensure_model_streamer_torch_distributed()` finds a process group
            or an MPI world that does not match ``rank`` and ``world_size``.
    """
    if world_size == 1:
        return set(local_keys), 1

    # Imported here so that only Model Streamer loads pay for them.
    from tensorrt_llm._torch.distributed.initialization import (
        ensure_model_streamer_torch_distributed,
    )
    from tensorrt_llm._utils import mpi_comm, mpi_disabled

    ensure_model_streamer_torch_distributed(rank, world_size)
    hostname = socket.gethostname()
    local = (hostname, tuple(sorted(local_keys)))
    if mpi_disabled():
        import torch.distributed as dist

        # Ray workers have no MPI; their process group already serves TRT-LLM.
        gathered: list[tuple[str, tuple[str, ...]] | None] = [None] * world_size
        dist.all_gather_object(gathered, local)
    else:
        # A collective on the default process group would make NCCL keep that
        # communicator's GPU buffers for the rest of the run.
        gathered = mpi_comm().allgather(local)
    peers = [entry for entry in gathered if entry is not None and entry[0] == hostname]
    if not peers:
        raise RuntimeError(f"No Model Streamer peers found on {hostname}")
    node_keys = set().union(*(entry[1] for entry in peers))
    return node_keys, len(peers)


def stream_batched_tensors(
    weight_source_file_map: Mapping[str, str],
    tensor_batches: Sequence[set[str]],
    consume_tensor: Callable[[str, torch.Tensor], None],
    *,
    distributed: bool,
) -> dict[str, float]:
    """Stream the batched tensors and give each arrived tensor to a callback.

    Args:
        weight_source_file_map: tensor name -> source safetensor file.
        tensor_batches: batches of tensors to stream.
        consume_tensor: Called once for each streamed tensor. The tensor is a
            view into a buffer that the streamer reuses, so the callback must
            copy each tensor that it keeps on the current CUDA stream before it
            returns.
        distributed: Whether the ranks on this node participate in the reads together.

    Returns:
        time and data transmission metrics, keyed by `ModelStreamerMetricNames`

    Raises:
        - If the streamer does not return every named tensor.
        - If the streamer returns a tensor that is not required.
        - If the streamer returns a tensor twice.
    """
    try:
        from runai_model_streamer import SafetensorsStreamer
    except ImportError as error:
        raise ImportError(
            "checkpoint_io_policy='model_streamer' requires a runai-model-streamer wheel with "
            "SafetensorsStreamer.stream_files(tensor_names=...)"
        ) from error

    selected_tensors: int = 0
    selected_bytes: int = 0
    streamed_batches: int = 0
    start_time: float = time.perf_counter()
    with SafetensorsStreamer() as streamer:
        for cur_batch in tensor_batches:
            if not cur_batch:
                continue
            seen: set[str] = set()
            cur_files: list[str] = sorted({weight_source_file_map[name] for name in cur_batch})
            # ModelStreamer starts streaming on the current batch of tensors
            # owned=False yields views into the streamer's buffers, so a tensor
            # that no group on this rank keeps costs no copy.
            streamer.stream_files(
                cur_files,
                device="cuda",
                is_distributed=distributed,
                tensor_names=cur_batch,
                owned=False,
            )
            # process each streamed tensor
            for name, tensor in streamer.get_tensors():
                if name not in cur_batch or name in seen:
                    raise ValueError(f"Unexpected or duplicate Model Streamer tensor: {name}")
                seen.add(name)
                selected_bytes += tensor.numel() * tensor.element_size()
                consume_tensor(name, tensor)
                # No torch.cuda.synchronize() is needed here. consume_tensor()
                # queues its copy on the current CUDA stream before it returns.
                # The streamer writes this buffer next with a copy on the same
                # stream or with an NCCL broadcast, which waits for this stream.
            if seen != cur_batch:
                raise RuntimeError(
                    f"Model Streamer omitted tensors: {sorted(cur_batch - seen)[:10]}"
                )
            selected_tensors += len(cur_batch)
            streamed_batches += 1
    return {
        ModelStreamerMetricNames.MODEL_STREAMER_SECONDS: time.perf_counter() - start_time,
        ModelStreamerMetricNames.MODEL_STREAMER_NODE_TENSORS: float(selected_tensors),
        ModelStreamerMetricNames.MODEL_STREAMER_NODE_BYTES: float(selected_bytes),
        ModelStreamerMetricNames.MODEL_STREAMER_BATCHES: float(streamed_batches),
    }


def run_model_streamer(
    plan: ModelStreamerPlan,
    weight_source_file_map: Mapping[str, str],
    *,
    rank: int,
    world_size: int,
) -> dict[str, float]:
    """Use ModelStreamer to stream tensors efficiently according to ``plan``.

    `load_weights_with_model_streamer` calls this with the plan that the LLM
    model returned.

    The function gathers the names of the tensor each rank needs from ``plan``,
    merges into a node-local tensor name set, batches those names and calls
    ModelStreamer to stream tensors in each batch.

    Whenever a tensor group defined in ``plan.groups`` is finished streaming,
    it calls that group's callback function (provided by the LLM class) to
    process the tensors.

    After all tensors are streamed, call `plan.finish()`.

    Args:
        plan: tensor name groups and callback functions provided by the LLM
            class.
        weight_source_file_map: tensor name -> source safetensor file.
        rank: The rank of this process, ``Mapping.rank``.
        world_size: The number of ranks, ``Mapping.world_size``.

    Returns:
        Time and data transmission metrics, keyed by `ModelStreamerMetricNames`.

    Raises:
        - If a checkpoint lacks the tensor required by ``plan``.
        - If a group in ``plan.groups`` is empty.
        - If ModelStreamer failed to deliver required tensors.
    """

    # plan.groups: list of (tensor names, tensor load func) pairs
    if any(not names for names, _ in plan.groups):
        raise ValueError("Every Model Streamer load group must name at least one tensor")
    # local_names: all tensor names involved in plan.groups
    local_names = set().union(*(names for names, _ in plan.groups))
    missing = local_names - weight_source_file_map.keys()
    if missing:
        raise KeyError(f"Model Streamer checkpoint is missing tensors: {sorted(missing)[:10]}")
    # node_local_tensor_names: set[str], ranks_on_node: int
    node_local_tensor_names, ranks_on_node = gather_tensor_names_from_rank_to_node(
        local_names, rank=rank, world_size=world_size
    )
    # batch_idx_to_tensors_map: batch index -> the tensor names in that batch
    batch_idx_to_tensors_map: dict[int, set[str]] = defaultdict(set)
    for tensor_name in node_local_tensor_names:
        index = 0 if plan.batch_index is None else plan.batch_index(tensor_name)
        batch_idx_to_tensors_map[index].add(tensor_name)
    # batches: list[set[str]], divide node-local tensors into batches for streaming
    batches = [batch_idx_to_tensors_map[index] for index in sorted(batch_idx_to_tensors_map)]

    # For each group: the tensors that have not arrived yet
    # Initialize it to be all tensor names of that group
    waiting: list[set[str]] = [set(names) for names, _ in plan.groups]
    # For each group: the tensor that have arrived
    # Initialize it to be empty for each group
    arrived: list[dict[str, torch.Tensor] | None] = [{} for _ in plan.groups]
    tensor_name_to_group_idx: dict[str, list[int]] = defaultdict(list)
    for index, (names, _) in enumerate(plan.groups):
        for name in names:
            tensor_name_to_group_idx[name].append(index)

    def consume(tensor_name: str, tensor: torch.Tensor) -> None:
        """Called by the ModelStreamer streaming loop that feeds one tensor at a time."""
        groups_having_the_tensor: list[int] | None = tensor_name_to_group_idx.get(tensor_name)
        if not groups_having_the_tensor:
            # Only another rank on this node needs this tensor.
            return
        # ModelStreamer reuses the buffer that holds the tensor for next streaming iteration,
        # so we have to make a copy here.
        kept = tensor.clone()
        for group_index in groups_having_the_tensor:
            arrived[group_index][tensor_name] = kept
            waiting[group_index].discard(tensor_name)
            if not waiting[group_index]:
                # All tensors inside this group has arrived.
                # Call the function to process those tensors
                load_those_tensors = plan.groups[group_index][1]
                load_those_tensors(arrived[group_index])
                # clear up the used dict to free up some memory
                arrived[group_index] = None

    metrics = stream_batched_tensors(
        weight_source_file_map, batches, consume, distributed=ranks_on_node > 1
    )
    incomplete: list[list[str]] = [sorted(names)[:3] for names in waiting if names]
    if incomplete:
        raise RuntimeError(
            f"{len(incomplete)} Model Streamer load groups did not complete, "
            f"for example groups waiting for {incomplete[:3]}"
        )
    if plan.finish is not None:
        plan.finish()
    metrics[ModelStreamerMetricNames.MODEL_STREAMER_RANK_TENSORS] = float(len(local_names))
    return metrics


def check_model_streamer_support(model: nn.Module) -> None:
    """Raise when ``model`` cannot plan a Model Streamer load.

    Raises:
        - ValueError: the model has no ``plan_model_streamer_load()``.
    """
    if not callable(getattr(model, "plan_model_streamer_load", None)):
        raise ValueError(
            f"{type(model).__name__} does not support "
            "checkpoint_io_policy='model_streamer': it has no plan_model_streamer_load()"
        )


def load_weights_with_model_streamer(
    model: nn.Module, checkpoint_dir: str, *, rank: int, world_size: int
) -> None:
    """Stream the checkpoint tensors that ``model`` needs into it with Model Streamer.

    The model's ``plan_model_streamer_load()`` names the tensors this rank
    needs and how to load them. Only the target model is streamed; a separate
    draft checkpoint is read with the regular checkpoint reader. Logs the
    Model Streamer metrics (`ModelStreamerMetricNames`).

    Args:
        model: model with CUDA memory, to fill
        checkpoint_dir: checkpoint directory of the target model
        rank: rank of this process, ``Mapping.rank``
        world_size: number of ranks, ``Mapping.world_size``
    """
    parameter = next(iter(model.parameters()), None)
    if parameter is not None and parameter.device.type == "cuda":
        # Model Streamer writes to the current CUDA device, so make it the
        # device that holds the model's weights.
        torch.cuda.set_device(parameter.device)
    weight_source_file_map: dict[str, str] = get_weight_source_map(checkpoint_dir)
    plan: ModelStreamerPlan = model.plan_model_streamer_load(set(weight_source_file_map))
    streaming_metrics = run_model_streamer(
        plan, weight_source_file_map, rank=rank, world_size=world_size
    )
    if torch.cuda.is_available():
        # Finish the queued copies, so that post-load hooks read the loaded
        # weights and the caller's load timing includes them.
        torch.cuda.synchronize()
    metrics_text = ", ".join(f"{name}={value:.4f}" for name, value in streaming_metrics.items())
    logger.info(f"Model Streamer metrics for rank {rank}: {metrics_text}")
