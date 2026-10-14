# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""History policy for VisualGen client-latency regression checks."""

import json
from collections.abc import Mapping
from pathlib import Path

VISUAL_GEN_METRIC_CONTRACT = "v2_client_latency"
_VERIFIED_CLIENT_COMMITS = frozenset(
    json.loads(Path(__file__).with_name("visual_gen_client_timing_commits.json").read_text())[
        "commits"
    ]
)


def transform_visual_gen_history(record: Mapping[str, object]) -> dict[str, object] | None:
    """Select compatible timing data and discard inherited baseline values.

    Explicit source metadata identifies new uploads. A source-verified commit
    snapshot admits existing unversioned client results, including releases.
    Neither record dates nor an unavailable/shallow Git history establish the
    timing window. Unverified records are excluded from all regression inputs.

    Even a client-timing row may have inherited a server-window baseline from
    CI. Drop baseline values before selecting the latest baseline/threshold
    record, while preserving threshold overrides and the original DB record.
    """
    if record.get("s_runtime") != "visual_gen":
        return dict(record)
    contract = record.get("s_visual_gen_metric_contract")
    compatible = (
        contract == VISUAL_GEN_METRIC_CONTRACT
        if contract
        else record.get("s_commit") in _VERIFIED_CLIENT_COMMITS
    )
    if not compatible:
        return None
    return {key: value for key, value in record.items() if not key.startswith("d_baseline_")}
