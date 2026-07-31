r"""Build a trace JSON augmented with per-request offline upper-bound metrics.

The output mirrors the input ``*.trace.json`` schema exactly, except every
assistant ``message`` event that produced a scored request gets the
``optimal_*`` upper-bound fields copied from the corresponding request
record. Downstream tools can then view per-event UB metrics inline without
joining against the separate ``*.cachehit.json`` summary.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

ANNOTATION_FIELDS = (
    "optimal_cache_hit_tokens",
    "optimal_cache_miss_tokens",
    "optimal_cache_hit_rate",
    "optimal_cache_block_hit_rate",
)


def build_annotated_trace(
    trace_data: Dict[str, Any],
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a deep copy of *trace_data* with hit-rate fields added.

    Args:
        trace_data: Parsed contents of a ``*.trace.json`` file.
        record: Output from :func:`compute_cache_hit_upper_bound` for the
            same trace. Each entry in ``record["requests"]`` is matched to
            an event by its ``event_index``.

    Returns:
        A new JSON-serializable dict in the same schema as the input
        trace, with the four annotation fields attached to scored
        assistant events.
    """
    annotated = copy.deepcopy(trace_data)
    events = annotated.get("events")
    if not isinstance(events, list):
        return annotated
    for request in record.get("requests", []):
        event_index = request.get("event_index")
        if not isinstance(event_index, int):
            continue
        if event_index < 0 or event_index >= len(events):
            continue
        event = events[event_index]
        if not isinstance(event, dict):
            continue
        for field in ANNOTATION_FIELDS:
            if field in request:
                event[field] = request[field]
    return annotated
