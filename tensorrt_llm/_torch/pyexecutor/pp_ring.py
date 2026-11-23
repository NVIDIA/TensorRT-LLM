# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Depth of the pipeline-parallel micro-batch ring and the sample-state relay lag.

``PyExecutor._executor_loop_pp`` keeps ``num_micro_batches`` micro-batches in a
ring. With the synchronous (inline) sample-state relay the first rank, at
iteration ``t``, receives the sample state of the micro-batch it launched at
iteration ``t + 1 - lag`` from the last rank before it schedules ``t + 1``, so at
most ``lag`` micro-batches are in flight: the ring is a closed loop of ``lag``
jobs over ``pp_size`` stages. The default ``lag`` is ``pp_size``, one job per
stage, which runs the pipeline at the pace of the slowest micro-batch in flight
whenever micro-batch costs vary.

A disaggregated server at ``pipeline_parallel_size > 1`` deepens the ring to
``max(pp_size, DISAGG_RING_DEPTH)`` slots and keeps the synchronous relay:
every non-last rank relays the micro-batch launched ``depth - 1`` iterations
earlier instead of ``pp_size - 1``, so the first rank can run up to
``depth - 1`` micro-batches ahead of the last stage. Every other executor
(``pipeline_parallel_size == 1``, aggregated serving) keeps the default ring and
offsets. The asynchronous relay (``TLLM_PP_ASYNC_BROADCAST_SAMPLE_STATE=1``)
keeps its own window.

Capacity is not scaled with the ring: the capacity scheduler, the sequence-slot
pool and the KV cache manager still admit ``max_batch_size * pp_size`` requests.
A deeper ring spreads them over more micro-batches; it pays when a micro-batch
holds few requests (long context chunks), not when every micro-batch is full.
"""

from tensorrt_llm.logger import logger

#: Ring slots of a disaggregated pipeline-parallel server (at least ``pp_size``).
DISAGG_RING_DEPTH = 8


def resolve_num_micro_batches(
    pp_size: int,
    async_relay: bool,
    min_async_micro_batches: int,
    disaggregated: bool = False,
) -> int:
    """Number of micro-batch slots of the PP ring.

    Args:
        pp_size: pipeline-parallel size of the executor.
        async_relay: whether the sample-state relay runs on its background thread.
        min_async_micro_batches: the slot count the asynchronous relay needs.
        disaggregated: whether the executor takes part in disaggregated serving.

    Returns:
        ``max(pp_size, min_async_micro_batches)`` with the asynchronous relay;
        ``max(pp_size, DISAGG_RING_DEPTH)`` with the synchronous relay on a
        disaggregated executor at ``pp_size > 1``; ``pp_size`` otherwise (the
        default ring).
    """
    if async_relay:
        return max(pp_size, min_async_micro_batches)
    if pp_size <= 1 or not disaggregated:
        return pp_size
    depth = max(pp_size, DISAGG_RING_DEPTH)
    if depth != pp_size:
        logger.info(
            f"PP micro-batch ring of {depth} slots over {pp_size} stages (disaggregated "
            "server); sample states are relayed synchronously."
        )
    return depth


def relay_lag(pp_size: int, num_micro_batches: int, async_relay: bool) -> int:
    """Iterations from launching a micro-batch to relaying its sample state on the first rank.

    The asynchronous relay keeps the stock lag of ``pp_size`` (its large slot count
    only absorbs the relay thread's delivery delay); the synchronous relay lags by
    the ring depth (``resolve_num_micro_batches``).

    Args:
        pp_size: pipeline-parallel size.
        num_micro_batches: slot count of the ring.
        async_relay: whether the relay runs on its background thread.
    """
    return pp_size if async_relay else num_micro_batches


def executed_microbatch_offset(is_last_pp_rank: bool, lag: int) -> int:
    """Offset from the current slot to the slot whose sample state this rank relays now.

    The relay chain runs ``(pp_size - 1) -> 0 -> 1 -> ... -> (pp_size - 2)``: the
    last rank relays the micro-batch it ran in the previous iteration, every other
    rank the micro-batch launched ``lag - 1`` iterations earlier.

    Args:
        is_last_pp_rank: whether this rank is the last pipeline stage.
        lag: the relay lag (``relay_lag``).
    """
    return -1 if is_last_pp_rank else 1 - lag


def last_rank_due_relay_slot(microbatch_id: int, lag: int, num_micro_batches: int) -> int:
    """Slot of the last rank's relay send whose matching receive the first rank posts now.

    The first rank receives the sample state of micro-batch ``m`` at iteration
    ``m + lag - 1``; the last rank sent it at iteration ``m + 1``. At iteration ``t``
    (slot ``microbatch_id``) the due send is that of micro-batch ``t + 1 - lag``.

    Args:
        microbatch_id: the current slot.
        lag: the relay lag (``relay_lag``).
        num_micro_batches: slot count of the ring.
    """
    return (microbatch_id + 1 - lag) % num_micro_batches
