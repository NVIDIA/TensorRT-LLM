# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""History policy for VisualGen client-latency regression checks."""

from collections.abc import Mapping

VISUAL_GEN_METRIC_CONTRACT = "v2_client_latency"


def transform_visual_gen_history(record: Mapping[str, object]) -> dict[str, object] | None:
    """Select compatible timing data and discard inherited baseline values.

    Only records carrying the current timing contract enter regression inputs.
    Historical client results need source-verified metadata backfill before
    reuse; dates and commit identities alone do not establish the timing window.

    Even a client-timing row may have inherited a server-window baseline from
    CI. Drop baseline values before selecting the latest baseline/threshold
    record, while preserving threshold overrides and the original DB record.

    Args:
        record: A historical perf result from the database.

    Returns:
        A copy with compatible regression inputs, or None for excluded timing
        contracts. Records from other runtimes are copied unchanged.
    """
    if record.get("s_runtime") != "visual_gen":
        return dict(record)
    if record.get("s_visual_gen_metric_contract") != VISUAL_GEN_METRIC_CONTRACT:
        return None
    return {key: value for key, value in record.items() if not key.startswith("d_baseline_")}
