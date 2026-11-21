# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What an engine's log says of its block reuse."""

from __future__ import annotations

from typing import Any, Mapping

from .harness import workload_io

#: The head of what the engine logs when it turns block reuse off for a hybrid
#: model with no snapshot policy: "Disabling KV cache block reuse for the hybrid
#: Mamba model because no Mamba state snapshot policy is configured. ..." The
#: engine turns reuse off for other reasons too (some MLA configurations), with
#: other messages; this one is the reason an options file can fix.
REUSE_DISABLED_MARKER = "Disabling KV cache block reuse for the hybrid Mamba model"


def reuse_disabled_detail(output: str, *, options: str | None) -> str | None:
    """Why a gate fails when its engine turned block reuse off (``options``: the unit's file).

    ``None`` when the log does not say so. The engine logs it only for an
    engine asked for block reuse.
    """
    at = output.find(REUSE_DISABLED_MARKER)
    if at < 0:
        return None
    end = output.find("\n", at)
    line = output[output.rfind("\n", 0, at) + 1 : end if end >= 0 else None].strip()
    where = f"the unit's options file {options}" if options else "an options file for the unit"
    return (
        f"the unit asks for KV-cache block reuse and the engine turned it off: "
        f"{line[:240]!r}. Set `kv_cache_config.mamba_state_config` (a positive "
        f"`periodic_snapshot_interval` or snapshot offsets) in {where} "
        f"(accuracy.llm_api_options); the gates take it from there. Or turn "
        f"`kv_cache_reuse` off for the unit"
    )


#: The harness's record of what one engine reused (``checks/harness/reuse_probe.py``).
REUSE_MARKER = workload_io.REUSE_MARKER


def parse_reuse_records(output: str) -> list[dict[str, Any]]:
    """Every ``MV2FLOW_REUSE`` record in ``output``, in order."""
    return workload_io.parse_marked(output, REUSE_MARKER)


def reuse_failure(output: str, llm: Mapping[str, Any], *, options: str | None) -> str | None:
    """Why a gate fails on what its engine reused; ``None`` when it does not.

    The engine turned block reuse off (:func:`reuse_disabled_detail`), or
    ``llm``, the options the gate sent, asks for block reuse and the reuse
    probe read no cached token: one prompt sent twice, the second computed
    from the start. A probe that could not run, or whose responses reported
    no ``cached_tokens``, observed nothing and fails nothing.
    """
    reuse_off = reuse_disabled_detail(output, options=options)
    if reuse_off:
        return reuse_off
    kv = llm.get("kv_cache_config")
    if not (isinstance(kv, Mapping) and kv.get("enable_block_reuse")):
        return None
    records = parse_reuse_records(output)
    probe = records[-1].get("probe") if records else None
    if not isinstance(probe, Mapping):
        return None
    reused = probe.get("reused")
    if isinstance(reused, bool) or not isinstance(reused, (int, float)) or int(reused) != 0:
        return None
    return (
        f"the engine reused nothing: the reuse probe sent one {probe.get('tokens')}-token prompt "
        "twice after completion on the same cache, but cached_tokens was 0 on the second response. "
        "Check the declared snapshot policy and engine's actual block-reuse configuration"
    )


__all__ = [
    "REUSE_DISABLED_MARKER",
    "REUSE_MARKER",
    "parse_reuse_records",
    "reuse_disabled_detail",
    "reuse_failure",
]
