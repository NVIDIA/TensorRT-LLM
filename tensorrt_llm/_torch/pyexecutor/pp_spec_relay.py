# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compact sample-state relay of a pipeline-parallel executor with a one-model speculative sampler.

The PP loop relays each micro-batch's host sample state from the last rank to
the other ranks (``PyExecutor._ring_broadcast_sample_state``). With
``SpecSampler`` that host state is the whole slot-indexed store: ``new_tokens``
``[max_draft_len + 1, max_num_sequences, 1]``, ``new_tokens_lens``
``[max_num_sequences]`` and ``next_draft_tokens``
``[max_num_sequences, max_total_draft_tokens]``, all int32, whatever the number
of requests in the micro-batch. Readers only use the rows of the micro-batch's
own requests.

On a disaggregated server at ``pipeline_parallel_size > 1`` the last rank sends
only those rows, as one flat int32 buffer (``encode_compact_spec_relay``). The
message then stays small enough for the transport's eager protocol, so a
non-last rank's send completes without waiting for the next rank to post its
receive. Every rank decodes both formats (``expand_compact_spec_relay``), and
non-last ranks forward the received object unchanged.

The expanded host state has the sender's store shape. Each request's row sits at
the sender's sequence slot, so every value ``SpecSampler.update_requests`` reads
equals the value it reads from the full store. Rows of other slots are zero
instead of stale; no reader uses them. Every other executor sends the full store.
"""

import dataclasses
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch

from tensorrt_llm.logger import logger

from ..speculative.spec_sampler_base import SampleStateTensorsSpec
from .llm_request import LlmRequestState

# First element of a compact wire tuple; bump the version on any layout change.
WIRE_TAG = "tllm.pp_spec_relay.v1"
# Leading int32 columns of a row: the sender's sequence slot, then new_tokens_lens.
_SLOT_COL = 0
_LEN_COL = 1
_FIXED_COLS = 2

# (tag, max_accepted_path_len, num_slots, draft_width, has_verify, request ids, rows)
CompactSpecRelayWire = Tuple[str, int, int, int, bool, Tuple[int, ...], bytes]

# Fields of a SpecSampler host state. A host carrying any other instance attribute
# (one set after construction) is sent in full, since the wire would drop it.
_SPEC_FIELDS = frozenset(field.name for field in dataclasses.fields(SampleStateTensorsSpec))

_warned_reorder = False


def is_compact_spec_relay(obj: Any) -> bool:
    """Whether ``obj`` is a compact wire made by ``encode_compact_spec_relay``."""
    return type(obj) is tuple and len(obj) == 7 and type(obj[0]) is str and obj[0] == WIRE_TAG


def relay_payload_bytes(wire: Any) -> int:
    """Bytes of the sample-state buffers one relay message carries.

    A compact wire counts its rows buffer plus 8 bytes per request id; any other
    host state counts the bytes of its tensor fields. Pickle framing and the
    ``py_result`` diffs are left out, so both forms cost the same few microseconds
    to measure (a pickle of the full store would cost the full form more per send
    and bias a comparison run with the diagnostics on).

    Args:
        wire: the sample state a rank relays (compact wire or host state).
    """
    if is_compact_spec_relay(wire):
        return len(wire[6]) + 8 * len(wire[5])
    attributes = getattr(wire, "__dict__", None)
    if attributes is None:
        return 0
    return sum(
        value.numel() * value.element_size()
        for value in attributes.values()
        if isinstance(value, torch.Tensor)
    )


def _is_cpu_int32(tensor: Any, dim: int) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.device.type == "cpu"
        and tensor.dtype == torch.int32
        and tensor.dim() == dim
    )


def encode_compact_spec_relay(host: Any, requests: Sequence[Any]) -> Optional[CompactSpecRelayWire]:
    """The rows of ``requests`` of a ``SpecSampler`` host state, as a compact wire.

    Args:
        host: the micro-batch's host sample state (synchronized by the caller).
        requests: the micro-batch's sampled requests (``SampleState.requests``),
            in order; each row is taken at the request's ``py_seq_slot``.

    Returns:
        The wire, or None when ``host`` is not exactly a ``SpecSampler`` host state
        this format carries in full (another sampler's state, log probs, an
        unexpected dtype, device or shape, an attribute beyond its fields, or a
        request without a unique valid slot): the caller then sends ``host``
        itself.
    """
    if (
        type(host) is not SampleStateTensorsSpec
        or host.log_probs is not None
        or vars(host).keys() != _SPEC_FIELDS
    ):
        return None
    new_tokens = host.new_tokens
    new_tokens_lens = host.new_tokens_lens
    next_draft_tokens = host.next_draft_tokens
    verify_lens = host.verify_lens
    if not (
        _is_cpu_int32(new_tokens, 3)
        and _is_cpu_int32(new_tokens_lens, 1)
        and _is_cpu_int32(next_draft_tokens, 2)
    ):
        return None
    path_len, num_slots, beam_width = new_tokens.shape
    if (
        beam_width != 1
        or new_tokens_lens.shape[0] != num_slots
        or next_draft_tokens.shape[0] != num_slots
    ):
        return None
    draft_width = next_draft_tokens.shape[1]
    num_rows = len(requests)
    has_verify = verify_lens is not None
    if has_verify and not (_is_cpu_int32(verify_lens, 1) and verify_lens.shape[0] == num_rows):
        return None
    slots = [request.py_seq_slot for request in requests]
    if (
        any(
            not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < num_slots
            for slot in slots
        )
        or len(set(slots)) != num_rows
    ):
        return None

    width = _FIXED_COLS + path_len + draft_width + int(has_verify)
    rows = np.empty((num_rows, width), dtype=np.int32)
    if num_rows:
        index = np.asarray(slots, dtype=np.int64)
        rows[:, _SLOT_COL] = index
        rows[:, _LEN_COL] = new_tokens_lens.numpy()[index]
        tokens_end = _FIXED_COLS + path_len
        rows[:, _FIXED_COLS:tokens_end] = new_tokens.numpy()[:, index, 0].T
        rows[:, tokens_end : tokens_end + draft_width] = next_draft_tokens.numpy()[index]
        if has_verify:
            rows[:, -1] = verify_lens.numpy()
    request_ids = tuple(int(request.py_request_id) for request in requests)
    return (
        WIRE_TAG,
        int(path_len),
        int(num_slots),
        int(draft_width),
        has_verify,
        request_ids,
        rows.tobytes(),
    )


def expand_compact_spec_relay(
    wire: CompactSpecRelayWire, requests: Sequence[Any]
) -> SampleStateTensorsSpec:
    """The host sample state a compact wire stands for, for this rank's ``requests``.

    The store tensors have the sender's shape and hold each wire row at the
    sender's slot (zeros elsewhere); ``verify_lens`` follows the order of
    ``requests``, matched by request id.

    Args:
        wire: a wire made by ``encode_compact_spec_relay``.
        requests: this rank's requests of the same micro-batch
            (``SampleState.requests``).

    Raises:
        RuntimeError: a request ``SpecSampler.update_requests`` reads (not
            finished, with a slot) has no row of the same id at its own slot,
            or the request order differs and a request id is ambiguous.
        ValueError: the wire is malformed.
    """
    global _warned_reorder
    tag, path_len, num_slots, draft_width, has_verify, request_ids, rows_bytes = wire
    if tag != WIRE_TAG:
        raise ValueError(f"not a compact sample-state relay wire: {tag!r}")
    num_rows = len(request_ids)
    width = _FIXED_COLS + path_len + draft_width + int(has_verify)
    rows = np.frombuffer(rows_bytes, dtype=np.int32)
    if rows.size != num_rows * width:
        raise ValueError(
            f"compact sample-state relay wire holds {rows.size} values, "
            f"expected {num_rows} rows of {width}"
        )
    rows = rows.reshape(num_rows, width)

    new_tokens = np.zeros((path_len, num_slots, 1), dtype=np.int32)
    new_tokens_lens = np.zeros((num_slots,), dtype=np.int32)
    next_draft_tokens = np.zeros((num_slots, draft_width), dtype=np.int32)
    sender_slots = rows[:, _SLOT_COL].astype(np.int64)
    if num_rows:
        tokens_end = _FIXED_COLS + path_len
        new_tokens[:, sender_slots, 0] = rows[:, _FIXED_COLS:tokens_end].T
        new_tokens_lens[sender_slots] = rows[:, _LEN_COL]
        next_draft_tokens[sender_slots] = rows[:, tokens_end : tokens_end + draft_width]

    if tuple(int(request.py_request_id) for request in requests) == request_ids:
        row_of = list(range(num_rows))
    else:
        index = {}
        for row, request_id in enumerate(request_ids):
            if request_id in index:
                raise RuntimeError(
                    "compact sample-state relay: the request order differs from the "
                    f"sender's and request id {request_id} is not unique"
                )
            index[request_id] = row
        row_of = [index.get(int(request.py_request_id)) for request in requests]
        if not _warned_reorder:
            _warned_reorder = True
            logger.warning(
                "compact sample-state relay: the request order differs from the "
                "sender's; matching rows by request id"
            )

    verify_lens = np.zeros((len(requests),), dtype=np.int32) if has_verify else None
    for i, (request, row) in enumerate(zip(requests, row_of)):
        if row is not None and verify_lens is not None:
            verify_lens[i] = rows[row, -1]
        slot = request.py_seq_slot
        if request.state == LlmRequestState.GENERATION_COMPLETE or slot is None:
            continue
        if row is None or int(sender_slots[row]) != slot:
            raise RuntimeError(
                f"compact sample-state relay: request {request.py_request_id} at slot "
                f"{slot} has no row of its own at that slot in the sender's state"
            )

    return SampleStateTensorsSpec(
        new_tokens=torch.from_numpy(new_tokens),
        new_tokens_lens=torch.from_numpy(new_tokens_lens),
        next_draft_tokens=torch.from_numpy(next_draft_tokens),
        verify_lens=None if verify_lens is None else torch.from_numpy(verify_lens),
    )
