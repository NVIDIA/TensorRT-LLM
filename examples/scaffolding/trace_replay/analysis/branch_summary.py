r"""Per-branch / per-sub-agent rollups over per-request cache-hit records.

Three groupings are produced from the request list:

* ``by_branch_root``  – first element of ``branch_path`` (or ``"root"`` if the
  request ran at the top scope). Useful for "Researcher 0 vs 1 vs 2" splits.
* ``by_branch_depth`` – ``len(branch_path)``. Useful for "ToT depth 1 vs 3".
* ``by_system_prompt`` – first ``system_prompt_id`` UUID seen on the
  conversation that owns the request. Useful for "all Researcher turns vs all
  Compressor turns" without needing extra trace fields.

All rollups are pure functions of the per-request list — no cache state,
no event ordering required.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

_NO_SYSTEM_KEY = "(no system)"
_ROOT_KEY = "root"


def compute_branch_rollups(
    requests: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Return all three rollups for *requests*."""
    return {
        "by_branch_root": _rollup(requests, _branch_root_key),
        "by_branch_depth": _rollup(requests, _branch_depth_key, sort_numeric=True),
        "by_system_prompt": _rollup(requests, _system_prompt_key),
    }


def _branch_root_key(req: Dict[str, Any]) -> str:
    branch_path = req.get("branch_path") or []
    if not branch_path:
        return _ROOT_KEY
    return f"branch:{branch_path[0]}"


def _branch_depth_key(req: Dict[str, Any]) -> str:
    branch_path = req.get("branch_path") or []
    return f"depth:{len(branch_path)}"


def _system_prompt_key(req: Dict[str, Any]) -> str:
    seed: Optional[str] = req.get("system_prompt_id_seed")
    if not seed:
        return _NO_SYSTEM_KEY
    return f"uuid:{seed}"


def _rollup(
    requests: Sequence[Dict[str, Any]],
    key_fn: Callable[[Dict[str, Any]], str],
    *,
    sort_numeric: bool = False,
) -> List[Dict[str, Any]]:
    """Group *requests* by ``key_fn``, return one summary dict per group."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for req in requests:
        grouped.setdefault(key_fn(req), []).append(req)

    summaries = [_group_summary(key, group) for key, group in grouped.items()]
    if sort_numeric:
        summaries.sort(key=lambda s: _numeric_suffix(s["group_key"]))
    else:
        summaries.sort(key=lambda s: s["group_key"])
    return summaries


def _group_summary(group_key: str, group: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_prompt_tokens = sum(r["prompt_tokens"] for r in group)
    total_hit_blocks = sum(r["optimal_cache_hit_blocks"] for r in group)
    total_miss_blocks = sum(r["optimal_cache_miss_blocks"] for r in group)
    total_hit_tokens = sum(r["optimal_cache_hit_tokens"] for r in group)
    total_miss_tokens = sum(r["optimal_cache_miss_tokens"] for r in group)
    block_total = total_hit_blocks + total_miss_blocks
    return {
        "group_key": group_key,
        "request_count": len(group),
        "total_prompt_tokens": total_prompt_tokens,
        "optimal_total_cache_hit_blocks": total_hit_blocks,
        "optimal_total_cache_miss_blocks": total_miss_blocks,
        "optimal_total_cache_hit_tokens": total_hit_tokens,
        "optimal_total_cache_miss_tokens": total_miss_tokens,
        "optimal_overall_cache_hit_rate": _safe_div(total_hit_tokens, total_prompt_tokens),
        "optimal_overall_cache_block_hit_rate": _safe_div(total_hit_blocks, block_total),
    }


def merge_rollup_arrays(
    rollups: Sequence[List[Dict[str, Any]]],
    *,
    sort_numeric: bool = False,
) -> List[Dict[str, Any]]:
    """Merge per-trace rollup arrays of the same kind into one dataset rollup.

    Groups are matched by ``group_key``; numeric totals are summed and rates
    recomputed from the sums.
    """
    aggregated: Dict[str, Dict[str, int]] = {}
    for trace_rollup in rollups:
        for entry in trace_rollup:
            key = entry["group_key"]
            slot = aggregated.setdefault(
                key,
                {
                    "request_count": 0,
                    "total_prompt_tokens": 0,
                    "optimal_total_cache_hit_blocks": 0,
                    "optimal_total_cache_miss_blocks": 0,
                    "optimal_total_cache_hit_tokens": 0,
                    "optimal_total_cache_miss_tokens": 0,
                },
            )
            for field in slot:
                slot[field] += entry[field]

    summaries: List[Dict[str, Any]] = []
    for key, slot in aggregated.items():
        block_total = (
            slot["optimal_total_cache_hit_blocks"] + slot["optimal_total_cache_miss_blocks"]
        )
        summaries.append(
            {
                "group_key": key,
                **slot,
                "optimal_overall_cache_hit_rate": _safe_div(
                    slot["optimal_total_cache_hit_tokens"], slot["total_prompt_tokens"]
                ),
                "optimal_overall_cache_block_hit_rate": _safe_div(
                    slot["optimal_total_cache_hit_blocks"], block_total
                ),
            }
        )
    if sort_numeric:
        summaries.sort(key=lambda s: _numeric_suffix(s["group_key"]))
    else:
        summaries.sort(key=lambda s: s["group_key"])
    return summaries


def _numeric_suffix(group_key: str) -> int:
    """Sort key for ``depth:N`` / ``branch:N`` style identifiers."""
    _, _, tail = group_key.partition(":")
    try:
        return int(tail)
    except ValueError:
        return -1


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0
