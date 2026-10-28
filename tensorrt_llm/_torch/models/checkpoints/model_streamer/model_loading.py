# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model Streamer helpers that `ModelLoader` calls for ``checkpoint_io_policy="model_streamer"``.

Run:ai Model Streamer reads each checkpoint byte once per node and shares it
among the ranks of the node, so large checkpoints load faster. A model
describes the tensors it needs and how to load them in
``plan_model_streamer_load()``, which returns a `ModelStreamerPlan`.
"""

import contextlib
import json
import os
import socket
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from packaging.version import Version
from safetensors import safe_open
from strenum import StrEnum
from torch import nn

from tensorrt_llm._torch.distributed.initialization import model_streamer_torch_distributed
from tensorrt_llm._utils import mpi_comm, mpi_disabled
from tensorrt_llm.logger import logger

# The first release whose SafetensorsStreamer.stream_files() takes the
# tensor_names and owned arguments.
_MIN_MODEL_STREAMER_VERSION = Version("0.17.0")


@dataclass
class ModelStreamerPlan:
    """The tensors that a model needs on this rank, and how to load them.

    A model returns this from ``plan_model_streamer_load(checkpoint_keys)``.
    `load_weights_with_model_streamer()` streams the named tensors and calls the
    load functions.

    Attributes:
        groups: ``(tensor names, load function)`` pairs. A model loads some
            tensors together, for example ``gate_proj`` and ``up_proj`` into one
            fused ``gate_up_proj``. The streaming calls the load function of a
            group once, with a dict of all the tensors of the group, as soon as
            they have all arrived. Name only the tensors that this rank needs.
            A tensor can be in several groups. These groups then get the same
            tensor object, so a load function must not change its tensors in
            place. A group keeps its tensors in memory until all of them arrive,
            so put the tensors of a group in one batch. The load functions run
            while the ranks stream, so they must not run collectives.
        batch_index: Gives the batch that streams a tensor, to limit the memory
            that the streaming uses. Batches stream in increasing index order.
            The ranks of a node stream their tensors together, so this function
            must give each tensor name the same index on every rank, whichever
            tensors the rank needs. ``None`` streams all tensors in one batch.
        finish: Called once after all tensors are streamed and the Model
            Streamer process group has ended, for steps that only the Model
            Streamer load needs. The model loader still calls the model's
            ``post_load_weights()`` after it.
    """

    groups: list[tuple[Collection[str], Callable[[dict[str, torch.Tensor]], None]]]
    batch_index: Optional[Callable[[str], int]] = None
    finish: Optional[Callable[[], None]] = None


class ModelStreamerMetricNames(StrEnum):
    """Names of the metrics that `load_weights_with_model_streamer` logs after a load.

    They stay out of `ModelLoader.metrics`, so the startup metrics do not change.
    Each member is a ``str``, so it works directly as a key of the metrics dict
    that `_run_model_streamer()` returns.
    """

    # Time of streaming all batches, in seconds.
    SECONDS = "model_streamer_seconds"
    # Number of batches streamed, one `SafetensorsStreamer.stream_files()` call each.
    BATCHES = "model_streamer_batches"
    # Tensors and bytes streamed for all ranks on this node.
    NODE_TENSORS = "model_streamer_node_tensors"
    NODE_BYTES = "model_streamer_node_bytes"
    # Tensors that this rank needs, a subset of the node's tensors.
    RANK_TENSORS = "model_streamer_rank_tensors"


def _get_weight_source_map(checkpoint_dir: str) -> dict[str, str]:
    """Map each tensor name in a safetensors checkpoint to the file that holds it.

    When ``model.safetensors.index.json`` exists, its ``weight_map`` is the only
    source, as in Hugging Face Transformers. Otherwise the function reads the
    keys of every ``*.safetensors`` file, and skips ``consolidated*`` files when
    there are other shards, as the native reader does.

    Example return:
        {
            "model.embed_tokens.weight": "/ckpt/model-00001-of-00002.safetensors",
            "lm_head.weight": "/ckpt/model-00002-of-00002.safetensors",
            ...
        }

    Raises:
        ValueError: If two files without an index hold the same tensor name.
        FileNotFoundError: If the checkpoint has no safetensors file, or the
            index names a file that does not exist.
    """
    directory = Path(checkpoint_dir)
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        with index.open() as index_file:
            weight_map = json.load(index_file)["weight_map"]
        paths = {name: str(directory / name) for name in set(weight_map.values())}
        weight_source_file_map = {key: paths[name] for key, name in weight_map.items()}
    else:
        all_shards = sorted(directory.glob("*.safetensors"))
        shards = [path for path in all_shards if "consolidated" not in path.name] or all_shards
        weight_source_file_map = {}
        for path in map(str, shards):
            with safe_open(path, framework="pt", device="cpu") as file:
                keys = set(file.keys())
            overlap = keys.intersection(weight_source_file_map)
            if overlap:
                raise ValueError(
                    f"Duplicate safetensors keys ({len(overlap)}): {sorted(overlap)[:10]}"
                )
            weight_source_file_map.update(dict.fromkeys(keys, path))
    paths = set(weight_source_file_map.values())
    if not paths or any(not os.path.isfile(path) for path in paths):
        raise FileNotFoundError(f"No complete safetensors checkpoint in {checkpoint_dir}")
    return weight_source_file_map


def _gather_node_tensor_names(
    local_names: Collection[str],
    error: Optional[Exception],
    *,
    world_size: int,
) -> tuple[set[str], int, bool]:
    """Gather the tensor names that the ranks of this node need, and every rank's planning error.

    Model Streamer streams the tensors of a node together, which is tested
    to be faster than:
    - Each rank uses Model Streamer to stream its own weights independently.
    - All ranks participate in streaming all the tensors of an LLM.

    Every rank must call this function, also a rank whose planning failed, so
    that no rank waits for a failed rank in a later collective.

    Args:
        local_names: The tensor names that this rank needs.
        error: The error of this rank's planning, or ``None``.
        world_size: The number of ranks, ``Mapping.world_size``.

    Returns:
        The names that the ranks of this node need, the largest number of
        ranks on one node, and whether the ranks are on more than one node.
        The last two values are the same on every rank.

    Raises:
        Exception: ``error`` itself, on the rank that failed.
        RuntimeError: On the other ranks, when a rank failed.
    """
    if world_size == 1:
        if error is not None:
            raise error
        return set(local_names), 1, False

    hostname = socket.gethostname()
    message = None if error is None else f"{type(error).__name__}: {error}"
    local = (hostname, tuple(sorted(local_names)), message)
    if mpi_disabled():
        # Ray workers have no MPI; their process group already serves TRT-LLM.
        gathered: list[Optional[tuple[str, tuple[str, ...], Optional[str]]]] = [None] * world_size
        dist.all_gather_object(gathered, local)
    else:
        # Under MPI, gather over MPI: the Model Streamer process group does not
        # exist yet. A group that another component created outlives the load,
        # and a collective on it would make NCCL keep the communicator's GPU
        # buffers for the rest of the run.
        gathered = mpi_comm().allgather(local)
    if error is not None:
        raise error
    failed = [(rank, entry[2]) for rank, entry in enumerate(gathered) if entry[2] is not None]
    if failed:
        rank, rank_error = failed[0]
        raise RuntimeError(
            f"{len(failed)} rank(s) failed to plan the Model Streamer load, "
            f"for example rank {rank}: {rank_error}"
        )
    ranks_per_node = Counter(entry[0] for entry in gathered)
    node_names = set().union(*(entry[1] for entry in gathered if entry[0] == hostname))
    return node_names, max(ranks_per_node.values()), len(ranks_per_node) > 1


def _stream_batched_tensors(
    weight_source_file_map: Mapping[str, str],
    tensor_batches: Sequence[set[str]],
    consume_tensor: Callable[[str, torch.Tensor], None],
    *,
    distributed: bool,
) -> dict[str, float | int]:
    """Stream the batched tensors and give each arrived tensor to a callback.

    Args:
        weight_source_file_map: tensor name -> source safetensor file.
        tensor_batches: batches of tensors to stream. No batch is empty.
        consume_tensor: Called once for each streamed tensor. The tensor is a
            view into a buffer that the streamer reuses, so the callback must
            copy each tensor that it keeps on the current CUDA stream before it
            returns.
        distributed: Whether the ranks on this node share the reads. Every rank
            must pass the same value.

    Returns:
        time and data transmission metrics, keyed by `ModelStreamerMetricNames`

    Raises:
        RuntimeError: If ``distributed`` is true but the streamer reads alone on
            this rank, or if the streamer does not return every named tensor.
        ValueError: If the streamer returns a tensor that is not in the batch,
            or a tensor twice.
    """
    from runai_model_streamer import SafetensorsStreamer

    # Model Streamer sets RUNAI_STREAMER_PROCESS_GROUP_SIZE for this process
    # when it is unset and never removes it. Remove it after the load, so that
    # a later load with another number of ranks per node counts again.
    group_size_was_set = "RUNAI_STREAMER_PROCESS_GROUP_SIZE" in os.environ
    node_tensors: int = 0
    node_bytes: int = 0
    start_time: float = time.perf_counter()
    try:
        with SafetensorsStreamer() as streamer:
            for cur_batch in tensor_batches:
                seen: set[str] = set()
                cur_files: list[str] = sorted({weight_source_file_map[name] for name in cur_batch})
                # owned=False yields views into the streamer's buffers, so a tensor
                # that no group on this rank keeps costs no copy.
                streamer.stream_files(
                    cur_files,
                    device="cuda",
                    is_distributed=distributed,
                    tensor_names=cur_batch,
                    owned=False,
                )
                # On each call, the streamer decides again whether the ranks share
                # the reads. When they do not, it logs the reason only at INFO level,
                # and each rank reads every tensor of its node.
                if distributed and not streamer.file_streamer.is_distributed:
                    raise RuntimeError(
                        "Model Streamer does not share the reads of this node among its ranks. "
                        "This happens when RUNAI_STREAMER_DIST=auto is set and the process "
                        "group backend is not exactly 'nccl', or when the free GPU memory of "
                        "a rank is less than twice the largest tensor of a batch. "
                        "Set RUNAI_STREAMER_LOG_TO_STDERR=1 to see the reason that it logs, or "
                        "set RUNAI_STREAMER_DIST=0 to read without sharing."
                    )
                # process each streamed tensor
                for name, tensor in streamer.get_tensors():
                    if name not in cur_batch or name in seen:
                        raise ValueError(f"Unexpected or duplicate Model Streamer tensor: {name}")
                    seen.add(name)
                    node_bytes += tensor.numel() * tensor.element_size()
                    consume_tensor(name, tensor)
                    # No torch.cuda.synchronize() is needed here. consume_tensor()
                    # queues its copy on the current CUDA stream before it returns.
                    # The streamer writes this buffer next with a copy on the same
                    # stream or with an NCCL broadcast, which waits for this stream.
                if seen != cur_batch:
                    omitted = cur_batch - seen
                    raise RuntimeError(
                        f"Model Streamer omitted {len(omitted)} tensors: {sorted(omitted)[:10]}"
                    )
                node_tensors += len(cur_batch)
    finally:
        if not group_size_was_set:
            os.environ.pop("RUNAI_STREAMER_PROCESS_GROUP_SIZE", None)
    return {
        ModelStreamerMetricNames.SECONDS: time.perf_counter() - start_time,
        ModelStreamerMetricNames.NODE_TENSORS: node_tensors,
        ModelStreamerMetricNames.NODE_BYTES: node_bytes,
        ModelStreamerMetricNames.BATCHES: len(tensor_batches),
    }


def _run_model_streamer(
    model: nn.Module,
    checkpoint_dir: str,
    *,
    rank: int,
    world_size: int,
) -> dict[str, float | int]:
    """Stream the checkpoint tensors that ``model`` plans to load, and load them.

    `load_weights_with_model_streamer` calls this.

    Each rank maps the checkpoint's tensors to their files and gets a
    `ModelStreamerPlan` from the model's ``plan_model_streamer_load()``. The
    ranks then gather the tensor names that each node needs. Each node streams
    its names in batches, and the ranks of a node share the reads over
    ``torch.distributed`` when some node has more than one rank and
    ``RUNAI_STREAMER_DIST`` is not ``0``. Without sharing, each rank streams only
    its own tensors.

    When all tensors of a group in ``plan.groups`` have arrived, the function
    calls the load function of that group. After all tensors are streamed and
    the process group has ended, it calls ``plan.finish()``.

    Args:
        model: model with CUDA memory, to fill.
        checkpoint_dir: checkpoint directory of the model.
        rank: The rank of this process, ``Mapping.rank``.
        world_size: The number of ranks, ``Mapping.world_size``.

    Returns:
        Time and data transmission metrics, keyed by `ModelStreamerMetricNames`.

    Raises:
        TypeError: If a group in ``plan.groups`` gives its tensor names as one
            ``str``.
        ValueError: If a group in ``plan.groups`` names no tensor.
        KeyError: If the checkpoint lacks a tensor that the plan names.
        RuntimeError: If another rank fails to plan its load, if
            ``RUNAI_STREAMER_DIST_GLOBAL=1`` is set for ranks on several nodes,
            or if `model_streamer_torch_distributed()` cannot provide a process
            group for ``rank`` and ``world_size``.
        Exception: The rank whose checkpoint index or
            ``plan_model_streamer_load()`` fails raises that error.
    """
    planning_error: Optional[Exception] = None
    local_names: set[str] = set()
    try:
        weight_source_file_map = _get_weight_source_map(checkpoint_dir)
        plan: ModelStreamerPlan = model.plan_model_streamer_load(set(weight_source_file_map))
        for names, _ in plan.groups:
            if isinstance(names, str):
                raise TypeError(
                    f"A Model Streamer load group names its tensors as one str: {names!r}"
                )
            if not names:
                raise ValueError("Every Model Streamer load group must name at least one tensor")
        # local_names: all tensor names involved in plan.groups
        local_names = set().union(*(names for names, _ in plan.groups))
        missing = local_names - weight_source_file_map.keys()
        if missing:
            raise KeyError(
                f"Model Streamer checkpoint is missing {len(missing)} tensors: "
                f"{sorted(missing)[:10]}"
            )
    except Exception as error:
        # Any planning error, also one from the model's own plan: this rank must
        # still join the gather below, which raises the error on every rank.
        planning_error = error
    node_tensor_names, max_ranks_per_node, ranks_span_nodes = _gather_node_tensor_names(
        local_names, planning_error, world_size=world_size
    )
    # Each node streams only the tensors that its own ranks need. In Model
    # Streamer's global mode, the ranks of all nodes would share the reads in one
    # group although the nodes stream different tensors.
    if ranks_span_nodes and int(os.environ.get("RUNAI_STREAMER_DIST_GLOBAL", "0")) == 1:
        raise RuntimeError(
            "checkpoint_io_policy='model_streamer' shares reads only among the ranks of a node. "
            "Unset RUNAI_STREAMER_DIST_GLOBAL to load on several nodes."
        )
    # Every rank takes part in Model Streamer's group setup when some node has
    # more than one rank, also a rank that is alone on its node. A rank that
    # skipped it would leave the ranks of other nodes waiting.
    shares_reads = max_ranks_per_node > 1 and os.environ.get("RUNAI_STREAMER_DIST") != "0"
    streamed_names = node_tensor_names if shares_reads else local_names
    # batch_names: batch index -> the tensor names in that batch
    batch_names: dict[int, set[str]] = defaultdict(set)
    for tensor_name in sorted(streamed_names):
        index = 0 if plan.batch_index is None else plan.batch_index(tensor_name)
        batch_names[index].add(tensor_name)
    batches = [batch_names[index] for index in sorted(batch_names)]

    # For each group: the names of the tensors that have not arrived yet.
    pending_names: list[set[str]] = [set(names) for names, _ in plan.groups]
    # For each group: the tensors that have arrived.
    arrived_tensors: list[Optional[dict[str, torch.Tensor]]] = [{} for _ in plan.groups]
    tensor_name_to_group_indices: dict[str, list[int]] = defaultdict(list)
    for group_index, names in enumerate(pending_names):
        for name in names:
            tensor_name_to_group_indices[name].append(group_index)

    def consume(tensor_name: str, tensor: torch.Tensor) -> None:
        """Called by the Model Streamer streaming loop that feeds one tensor at a time."""
        group_indices = tensor_name_to_group_indices.get(tensor_name)
        if not group_indices:
            # Only another rank on this node needs this tensor.
            return
        # Model Streamer reuses the buffer that holds the tensor for the next
        # streaming iteration, so we have to make a copy here.
        kept = tensor.clone()
        for group_index in group_indices:
            arrived_tensors[group_index][tensor_name] = kept
            pending_names[group_index].discard(tensor_name)
            if not pending_names[group_index]:
                # All tensors of this group have arrived, so load them.
                load_group = plan.groups[group_index][1]
                load_group(arrived_tensors[group_index])
                # Free the tensors that the group no longer needs.
                arrived_tensors[group_index] = None

    # Only the streaming needs the process group. A group that the context
    # creates ends with it, so that the process group state after the load is
    # the same as before it.
    with model_streamer_torch_distributed(rank, world_size):
        metrics = _stream_batched_tensors(
            weight_source_file_map, batches, consume, distributed=shares_reads
        )
    # _stream_batched_tensors() raises when it does not deliver a named tensor.
    assert not any(pending_names), "a Model Streamer load group did not complete"
    if plan.finish is not None:
        plan.finish()
    metrics[ModelStreamerMetricNames.RANK_TENSORS] = len(local_names)
    return metrics


def check_model_streamer_support(model: nn.Module) -> None:
    """Raise when ``model`` cannot plan a Model Streamer load, or Model Streamer is missing.

    Raises:
        ValueError: The model has no ``plan_model_streamer_load()``.
        ImportError: runai-model-streamer is not installed, or it is older than
            the release that this loader needs.
    """
    if not callable(getattr(model, "plan_model_streamer_load", None)):
        raise ValueError(
            f"{type(model).__name__} does not support "
            "checkpoint_io_policy='model_streamer': it has no plan_model_streamer_load()"
        )
    try:
        installed = Version(version("runai-model-streamer"))
    except PackageNotFoundError:
        installed = None
    if installed is None or installed < _MIN_MODEL_STREAMER_VERSION:
        found = "it is not installed" if installed is None else f"found {installed}"
        raise ImportError(
            "checkpoint_io_policy='model_streamer' requires runai-model-streamer "
            f">={_MIN_MODEL_STREAMER_VERSION}, but {found}"
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
    # Model Streamer writes to the current CUDA device, so make it the device
    # that holds the model's weights during the load.
    device = (
        torch.cuda.device(parameter.device)
        if parameter is not None and parameter.device.type == "cuda"
        else contextlib.nullcontext()
    )
    with device:
        streaming_metrics = _run_model_streamer(
            model, checkpoint_dir, rank=rank, world_size=world_size
        )
        if torch.cuda.is_available():
            # Finish the queued copies, so that post-load hooks read the loaded
            # weights and the caller's load timing includes them.
            torch.cuda.synchronize()
    metrics_text = ", ".join(
        f"{name}={value:.4f}" if isinstance(value, float) else f"{name}={value}"
        for name, value in streaming_metrics.items()
    )
    logger.info(f"Model Streamer metrics for rank {rank}: {metrics_text}")
