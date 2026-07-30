# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure coordination helpers for paired perf-sanity agreement experiments."""

import re
from typing import Optional, Set

_AGREEMENT_ARM_MARKER_RE = re.compile(
    r"PYTHON_AGREEMENT_AB_ARM_(START|END) "
    r"server_idx=(\d+) role=(CTX_\d+) process_id=(\d+)"
)


def format_agreement_arm_marker(
    transition: str,
    server_idx: int,
    role: str,
    process_id: str,
) -> str:
    """Return a stable marker that scopes evidence to one CTX service lifetime."""
    if transition not in ("START", "END"):
        raise ValueError(f"Unsupported agreement-arm transition: {transition}")
    return (
        f"PYTHON_AGREEMENT_AB_ARM_{transition} "
        f"server_idx={server_idx} role={role} process_id={process_id}"
    )


def expected_disagg_lifecycle_roles(
    num_ctx_servers: int,
    ctx_world_size: int,
    num_gen_servers: int,
    gen_world_size: int,
) -> Set[str]:
    """Return every pytest role expected at a disaggregated lifecycle barrier."""
    roles = {
        f"CTX_{server_idx}.{process_id}"
        for server_idx in range(num_ctx_servers)
        for process_id in range(ctx_world_size)
    }
    roles.update(
        f"GEN_{server_idx}.{process_id}"
        for server_idx in range(num_gen_servers)
        for process_id in range(gen_world_size)
    )
    roles.update(("DISAGG_SERVER.0", "BENCHMARK.0"))
    return roles


def extract_agreement_arm_log(
    log_text: str,
    server_idx: int,
    expected_ctx_roles: int,
    role: str = "CTX_0",
) -> Optional[str]:
    """Extract one exact arm from a cumulative outer CTX log.

    Returns ``None`` while the complete marker set has not yet become visible.
    Raises ``ValueError`` for duplicate or inconsistent marker sets.
    """
    all_markers = list(_AGREEMENT_ARM_MARKER_RE.finditer(log_text))
    start_markers = {}
    end_markers = {}
    for marker in all_markers:
        transition, marker_server_idx, marker_role, process_id = marker.groups()
        if int(marker_server_idx) != server_idx or marker_role != role:
            continue
        markers = start_markers if transition == "START" else end_markers
        if process_id in markers:
            raise ValueError(
                "Duplicate agreement-arm marker: "
                f"transition={transition}, server_idx={server_idx}, "
                f"role={role}, process_id={process_id}"
            )
        markers[process_id] = marker

    expected_process_ids = {str(process_id) for process_id in range(expected_ctx_roles)}
    observed_process_ids = set(start_markers).union(end_markers)
    if not observed_process_ids.issubset(expected_process_ids):
        raise ValueError(
            "Agreement-arm markers contain unexpected CTX process IDs: "
            f"observed={sorted(observed_process_ids)}, "
            f"expected={sorted(expected_process_ids)}"
        )
    if set(start_markers) != expected_process_ids or set(end_markers) != expected_process_ids:
        return None

    for process_id in expected_process_ids:
        if end_markers[process_id].start() <= start_markers[process_id].end():
            raise ValueError(
                "Agreement-arm END marker precedes its START marker: "
                f"server_idx={server_idx}, role={role}, process_id={process_id}"
            )
    start_offset = min(marker.end() for marker in start_markers.values())
    next_arm_offsets = [
        marker.start()
        for marker in all_markers
        if marker.group(1) == "START"
        and int(marker.group(2)) > server_idx
        and marker.group(3) == role
    ]
    # Include the cumulative tail so evidence forwarded after END remains in
    # scope. The caller keeps every role at a post-verification barrier, so a
    # later arm cannot start while this tail is being polled.
    end_offset = min(next_arm_offsets) if next_arm_offsets else len(log_text)
    return log_text[start_offset:end_offset]
