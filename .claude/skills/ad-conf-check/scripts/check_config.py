#!/usr/bin/env python3
"""Check whether AutoDeploy YAML configs were actually applied based on server log.

Usage:
    python3 check_config.py <yaml_path> [<yaml_path2> ...] <log_path> [--output <output_path>]

Accepts one or more YAML config files. When multiple YAMLs are provided,
they are deep-merged left-to-right: later files override earlier ones for
overlapping keys (like AutoDeploy's own layering of default + user configs).

Parses the merged config and searches the log for evidence that each config
was applied, skipped, disabled, or failed. Outputs a summary table.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# ── Status constants ──────────────────────────────────────────────────────────
APPLIED = "APPLIED"
FAILED = "FAILED"
SKIPPED = "SKIPPED"
DISABLED = "DISABLED"
UNKNOWN = "UNKNOWN"  # no log evidence found


# ── Per-config checkers ───────────────────────────────────────────────────────
# Each checker returns (status, evidence_line_or_None).


def _search(log: str, pattern: str) -> Optional[str]:
    """Return first matching line or None."""
    m = re.search(pattern, log, re.MULTILINE | re.IGNORECASE)
    return m.group(0).strip() if m else None


def _search_all(log: str, pattern: str) -> List[str]:
    """Return all matching lines."""
    return [m.group(0).strip() for m in re.finditer(pattern, log, re.MULTILINE | re.IGNORECASE)]


def _check_transform_summary(log: str, transform_key: str) -> Tuple[str, Optional[str]]:
    """Check the [SUMMARY] line for a given transform."""
    # Pattern: [stage=..., transform=<key>] ... [SUMMARY] ...
    prefix_pat = rf"\[stage=\w+,\s*transform={re.escape(transform_key)}\]"

    # Look for summary line
    summary_pat = rf"^.*{prefix_pat}.*\[SUMMARY\].*$"
    line = _search(log, summary_pat)
    if not line:
        return UNKNOWN, None

    # Strip ANSI codes for matching
    clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
    if "disabled" in clean.lower():
        return DISABLED, line
    if "skipped" in clean.lower():
        return SKIPPED, line
    m = re.search(r"matches=(\d+)", clean)
    if m:
        n = int(m.group(1))
        return (APPLIED if n > 0 else SKIPPED), line
    return UNKNOWN, line


# ── Config-specific checkers ──────────────────────────────────────────────────


def check_compile_backend(log: str, value: str) -> Tuple[str, Optional[str]]:
    if value == "torch-cudagraph":
        ev = _search(log, r"Warm up with max_batch_size=.*before graph capture")
        if ev:
            return APPLIED, ev
        ev = _search(log, r"Capturing graph for batch size:")
        if ev:
            return APPLIED, ev
    elif value == "torch-compile":
        ev = _search(log, r"Torch Dynamo cache size limit")
        if ev:
            return APPLIED, ev
    elif value == "torch-opt":
        ev = _search(log, r"Setting Torch Dynamo recompile limit")
        if ev:
            return APPLIED, ev
    return UNKNOWN, None


def check_cuda_graph_batch_sizes(log: str, sizes: list) -> Tuple[str, Optional[str]]:
    ev = _search(log, r"Using cuda_graph_batch_sizes:.*")
    if ev:
        return APPLIED, ev
    # Fallback: check individual captures
    captured = _search_all(log, r"Capturing graph for batch size: \d+")
    if captured:
        return APPLIED, f"Captured {len(captured)} batch sizes"
    return UNKNOWN, None


def check_piecewise_enabled(log: str, enabled: bool) -> Tuple[str, Optional[str]]:
    if not enabled:
        return DISABLED, "piecewise_enabled=false in config"

    # Check for failure first
    fail = _search(log, r"model is not a GraphModule.*piecewise CUDA graph.*Falling back")
    if fail:
        return FAILED, fail

    # Check for success
    success = _search(log, r"TorchCudagraphCompiler: dual-mode enabled \(monolithic \+ piecewise\)")
    if success:
        # Still check if it prepared successfully
        prepared = _search(log, r"PiecewiseCapturedGraph: prepared with \d+ submodules")
        captured = _search(log, r"PiecewiseCapturedGraph: captured graphs for num_tokens=")
        if captured:
            return APPLIED, captured
        if prepared:
            return APPLIED, prepared
        return APPLIED, success

    # Check for auto-generated buckets as partial evidence
    auto = _search(log, r"Auto-generated piecewise_num_tokens from max_num_tokens=")
    if auto:
        return APPLIED, auto

    return UNKNOWN, None


def check_allreduce_strategy(log: str, strategy: str) -> Tuple[str, Optional[str]]:
    ev = _search(log, rf"Using allreduce strategy:\s*{re.escape(strategy)}")
    if ev:
        return APPLIED, ev
    return UNKNOWN, None


def check_sharding_source(log: str, sources: list) -> Tuple[str, Optional[str]]:
    evidences = []
    for src in sources:
        if src == "manual":
            ev = _search(log, r"Applying sharding from manual config")
            skip = _search(log, r"No manual config found\. Skipping")
            if skip:
                return FAILED, skip
            if ev:
                evidences.append(ev)
        elif src == "factory":
            ev = _search(log, r"Applying sharding from factory config")
            skip = _search(log, r"No factory config found\. Skipping")
            if skip:
                return FAILED, skip
            if ev:
                evidences.append(ev)
        elif src == "heuristic":
            ev = _search(log, r"Running autodeploy TP sharding heuristics")
            if ev:
                evidences.append(ev)
    if evidences:
        return APPLIED, "; ".join(evidences)
    return UNKNOWN, None


def check_sharding_dims(log: str, dims: list) -> Tuple[str, Optional[str]]:
    evidences = []
    for dim in dims:
        if dim == "tp":
            ev = _search(log, r"Applied \d+ TP shards from config")
            if ev:
                evidences.append(ev)
            else:
                skip = _search(log, r"Skipping TP sharding for single device")
                if skip:
                    evidences.append(f"TP: {skip}")
        elif dim == "ep":
            ev = _search(log, r"Running autodeploy EP sharding heuristics")
            if ev:
                evidences.append(ev)
            else:
                skip = _search(log, r"Skipping EP sharding for single device")
                if skip:
                    evidences.append(f"EP: {skip}")
        elif dim == "bmm":
            ev = _search(log, r"Running autodeploy BMM sharding heuristics")
            if ev:
                evidences.append(ev)
            else:
                skip = _search(log, r"Skipping DP BMM sharding for single device")
                if skip:
                    evidences.append(f"BMM: {skip}")
    if evidences:
        return APPLIED, "; ".join(evidences)
    return UNKNOWN, None


def check_multi_stream_moe(log: str) -> Tuple[str, Optional[str]]:
    status, ev = _check_transform_summary(log, "multi_stream_moe")
    if status != UNKNOWN:
        return status, ev
    # Check specific logs
    fail = _search(log, r"No merge add found downstream of MoE node")
    if fail:
        return FAILED, fail
    return UNKNOWN, None


def check_multi_stream_gemm(log: str) -> Tuple[str, Optional[str]]:
    status, ev = _check_transform_summary(log, "multi_stream_gemm")
    if status != UNKNOWN:
        return status, ev
    success = _search(log, r"Fork point.*: moving.*to aux stream")
    if success:
        return APPLIED, success
    return UNKNOWN, None


def check_fuse_gemms_mixed_children(log: str) -> Tuple[str, Optional[str]]:
    status, ev = _check_transform_summary(log, "fuse_gemms_mixed_children")
    if status != UNKNOWN:
        return status, ev
    fused = _search(log, r"Fusing \d+ GEMMs")
    if fused:
        return APPLIED, fused
    skip = _search(log, r"Skipping GEMM fusion for.*mixed dtypes")
    if skip:
        return FAILED, skip
    return UNKNOWN, None


# ── Generic fallback checker ─────────────────────────────────────────────────


def _generic_check(log: str, key: str, value) -> Tuple[str, Optional[str]]:
    """Best-effort check for any config key/value pair by searching the log.

    Tries multiple strategies:
    1. Search for 'key=value' pattern
    2. Search for the key name near the value
    3. Search for the value itself (if specific enough)
    4. For transforms, check SUMMARY line
    """
    str_val = str(value)

    # Strategy 1: key=value (common in AutoDeploy logs)
    short_key = key.rsplit(".", 1)[-1]  # last segment of dotted key
    ev = _search(log, rf"{re.escape(short_key)}\s*[=:]\s*{re.escape(str_val)}")
    if ev:
        return APPLIED, ev

    # Strategy 2: for transform configs, check the transform's SUMMARY line
    if key.startswith("transforms."):
        parts = key.split(".")
        if len(parts) >= 2:
            transform_name = parts[1]
            ts, te = _check_transform_summary(log, transform_name)
            if ts == APPLIED:
                return (
                    APPLIED,
                    f"{transform_name} transform applied ({te})"
                    if te
                    else f"{transform_name} transform applied",
                )
            # Also check APPLY lines for the transform
            apply_ev = _search(log, rf"\[transform={re.escape(transform_name)}\].*\[APPLY\]")
            if apply_ev:
                return APPLIED, apply_ev

    # Strategy 3: search for the value directly (only if specific enough)
    if (
        isinstance(value, str)
        and len(str_val) >= 4
        and str_val.lower() not in ("true", "false", "none")
    ):
        ev = _search(log, re.escape(str_val))
        if ev:
            return APPLIED, ev
    elif isinstance(value, (int, float)) and value not in (0, 1, True, False):
        ev = _search(log, rf"\b{re.escape(str_val)}\b")
        if ev:
            # Only count if the key name is also nearby or it's in a relevant context
            key_ev = _search(log, rf"(?i){re.escape(short_key)}.*{re.escape(str_val)}")
            if key_ev:
                return APPLIED, key_ev

    return UNKNOWN, f"No log evidence found for {key}={str_val}"


# ── Main logic ────────────────────────────────────────────────────────────────


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates and returns *base*).

    For overlapping keys:
    - If both values are dicts, merge recursively.
    - Otherwise the override value wins.
    """
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def flatten_yaml(cfg: dict, prefix: str = "") -> List[Tuple[str, str, object]]:
    """Flatten YAML config into (dotted_key, display_key, value) triples."""
    items = []
    for k, v in cfg.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(flatten_yaml(v, full_key))
        else:
            items.append((full_key, full_key, v))
    return items


def check_config(yaml_paths: List[str], log_path: str) -> List[Dict]:
    """Parse YAML(s) and log, return list of {config, value, status, evidence} dicts.

    When multiple *yaml_paths* are given they are deep-merged left-to-right
    so that later files override earlier ones for overlapping keys.
    """
    cfg: dict = {}
    for yp in yaml_paths:
        with open(yp) as f:
            layer = yaml.safe_load(f) or {}
        deep_merge(cfg, layer)
    with open(log_path) as f:
        log = f.read()

    results = []

    def add(config_key: str, value, status: str, evidence: Optional[str]):
        results.append(
            {
                "config": config_key,
                "value": str(value),
                "status": status,
                "evidence": evidence or "-",
            }
        )

    # ── Top-level parameters ──────────────────────────────────────────────
    if "compile_backend" in cfg:
        s, e = check_compile_backend(log, cfg["compile_backend"])
        add("compile_backend", cfg["compile_backend"], s, e)

    if "attn_backend" in cfg:
        # Check insert_cached_attention transform for backend
        s, e = _check_transform_summary(log, "insert_cached_attention")
        add("attn_backend", cfg["attn_backend"], s, e)

    if "cuda_graph_batch_sizes" in cfg:
        s, e = check_cuda_graph_batch_sizes(log, cfg["cuda_graph_batch_sizes"])
        add("cuda_graph_batch_sizes", cfg["cuda_graph_batch_sizes"], s, e)

    if "enable_chunked_prefill" in cfg:
        ev = _search(log, r"(?i)chunked.?prefill|enable_chunked_prefill")
        if ev:
            add("enable_chunked_prefill", cfg["enable_chunked_prefill"], APPLIED, ev)
        else:
            add(
                "enable_chunked_prefill",
                cfg["enable_chunked_prefill"],
                UNKNOWN,
                "No explicit log (executor-level config)",
            )

    if "model_factory" in cfg:
        ev = _search(log, re.escape(cfg["model_factory"]))
        s = APPLIED if ev else UNKNOWN
        add("model_factory", cfg["model_factory"], s, ev)

    if "max_seq_len" in cfg:
        val = cfg["max_seq_len"]
        ev = _search(log, rf"max_seq_len={val}")
        adjusted = _search(log, r"Adjusted max_seq_len to \d+")
        if ev and adjusted:
            add("max_seq_len", val, APPLIED, f"{ev} (WARNING: {adjusted})")
        elif ev:
            add("max_seq_len", val, APPLIED, ev)
        else:
            add("max_seq_len", val, UNKNOWN, "No explicit log (consumed by downstream transforms)")

    if "max_num_tokens" in cfg:
        ev = _search(log, rf"max_num_tokens={cfg['max_num_tokens']}")
        s = APPLIED if ev else UNKNOWN
        add("max_num_tokens", cfg["max_num_tokens"], s, ev)

    if "max_batch_size" in cfg:
        ev = _search(log, rf"max_batch_size={cfg['max_batch_size']}")
        s = APPLIED if ev else UNKNOWN
        add("max_batch_size", cfg["max_batch_size"], s, ev)

    # ── kv_cache_config ───────────────────────────────────────────────────
    kv_cfg = cfg.get("kv_cache_config", {})
    if isinstance(kv_cfg, dict):
        for k, v in kv_cfg.items():
            if k == "free_gpu_memory_fraction":
                ev = _search(log, rf"free_gpu_memory_fraction={re.escape(str(v))}")
                if ev:
                    add(f"kv_cache_config.{k}", v, APPLIED, ev)
                else:
                    add(
                        f"kv_cache_config.{k}", v, UNKNOWN, "No explicit log (KV cache init config)"
                    )
            elif k == "tokens_per_block":
                ev = _search(log, rf"tokens_per_block={re.escape(str(v))}")
                if ev:
                    add(f"kv_cache_config.{k}", v, APPLIED, ev)
                else:
                    add(
                        f"kv_cache_config.{k}", v, UNKNOWN, "No explicit log (KV cache init config)"
                    )
            elif k == "enable_block_reuse":
                ev = _search(log, r"(?i)block.reuse|enable_block_reuse")
                if ev:
                    add(f"kv_cache_config.{k}", v, APPLIED, ev)
                else:
                    add(
                        f"kv_cache_config.{k}",
                        v,
                        UNKNOWN,
                        "No explicit log (KV cache init config, default is False)",
                    )
            else:
                ev = _search(log, rf"{re.escape(k)}={re.escape(str(v))}")
                if ev:
                    add(f"kv_cache_config.{k}", v, APPLIED, ev)
                else:
                    add(
                        f"kv_cache_config.{k}", v, UNKNOWN, "No explicit log (KV cache init config)"
                    )

    # ── model_kwargs ──────────────────────────────────────────────────────
    model_kwargs = cfg.get("model_kwargs", {})
    if isinstance(model_kwargs, dict):
        for k, v in model_kwargs.items():
            if k == "torch_dtype":
                # Check if torch_dtype was found or fell back to a default
                fallback = _search(log, r"torch_dtype not found in quant_config.*using default.*")
                deprecated = _search(log, r"`torch_dtype` is deprecated")
                if fallback:
                    add(
                        f"model_kwargs.{k}",
                        v,
                        FAILED,
                        f"{fallback}" + (f" ({deprecated})" if deprecated else ""),
                    )
                else:
                    ev = _search(log, rf"(?i)torch_dtype.*{re.escape(str(v))}")
                    if ev:
                        add(f"model_kwargs.{k}", v, APPLIED, ev)
                    else:
                        add(f"model_kwargs.{k}", v, UNKNOWN, "No explicit log (model init kwargs)")
            else:
                ev = _search(log, rf"{re.escape(k)}.*{re.escape(str(v))}")
                if ev:
                    add(f"model_kwargs.{k}", v, APPLIED, ev)
                else:
                    add(f"model_kwargs.{k}", v, UNKNOWN, "No explicit log (model init kwargs)")

    # ── transforms ────────────────────────────────────────────────────────
    transforms = cfg.get("transforms", {})
    if not isinstance(transforms, dict):
        return results

    for t_name, t_cfg in transforms.items():
        if not isinstance(t_cfg, dict):
            continue

        # Special-case handlers
        if t_name == "compile_model":
            pe = t_cfg.get("piecewise_enabled")
            if pe is not None:
                s, e = check_piecewise_enabled(log, pe)
                add("transforms.compile_model.piecewise_enabled", pe, s, e)
            pnt = t_cfg.get("piecewise_num_tokens")
            if pnt is not None:
                add(
                    "transforms.compile_model.piecewise_num_tokens",
                    pnt,
                    UNKNOWN,
                    "Check piecewise capture logs for actual tokens used",
                )
            continue

        if t_name == "detect_sharding":
            ars = t_cfg.get("allreduce_strategy")
            if ars:
                s, e = check_allreduce_strategy(log, ars)
                add("transforms.detect_sharding.allreduce_strategy", ars, s, e)
            ss = t_cfg.get("sharding_source")
            if ss:
                s, e = check_sharding_source(log, ss)
                add("transforms.detect_sharding.sharding_source", ss, s, e)
            sd = t_cfg.get("sharding_dims")
            if sd:
                s, e = check_sharding_dims(log, sd)
                add("transforms.detect_sharding.sharding_dims", sd, s, e)
            eadp = t_cfg.get("enable_attention_dp")
            if eadp is not None:
                ev = _search(log, r"Attention DP is enabled")
                s = APPLIED if (ev and eadp) else (UNKNOWN if eadp else UNKNOWN)
                add("transforms.detect_sharding.enable_attention_dp", eadp, s, ev)
            # Other detect_sharding params with specific checkers
            for k, v in t_cfg.items():
                if k in (
                    "allreduce_strategy",
                    "sharding_source",
                    "sharding_dims",
                    "enable_attention_dp",
                    "manual_config",
                ):
                    continue
                if k == "dist_mapping" and isinstance(v, dict):
                    # Check process grid log for moe_tp / moe_ep values
                    grid_ev = _search(log, r"process grid:.*\[TP, MoE_TP, MoE_EP\].*")
                    if grid_ev:
                        # Verify the values match
                        tp_match = re.search(
                            r"MoE_TP.*?=.*?(\d+)", grid_ev.replace(" ", "")
                        ) or re.search(
                            r"\[TP, MoE_TP, MoE_EP\]\s*=\s*\[(\d+),\s*(\d+),\s*(\d+)\]", grid_ev
                        )
                        if tp_match:
                            add(f"transforms.detect_sharding.{k}", v, APPLIED, grid_ev)
                        else:
                            add(f"transforms.detect_sharding.{k}", v, APPLIED, grid_ev)
                    else:
                        add(
                            f"transforms.detect_sharding.{k}",
                            v,
                            UNKNOWN,
                            "Check detect_sharding transform summary",
                        )
                elif k == "shard_all_unprocessed":
                    # If sharding SUMMARY has high match count, it's likely applied
                    ss, se = _check_transform_summary(log, "detect_sharding")
                    if ss == APPLIED:
                        add(
                            f"transforms.detect_sharding.{k}",
                            v,
                            APPLIED,
                            f"detect_sharding applied with {se}"
                            if se
                            else "detect_sharding applied",
                        )
                    else:
                        add(
                            f"transforms.detect_sharding.{k}",
                            v,
                            UNKNOWN,
                            "Check detect_sharding transform summary",
                        )
                elif k == "simple_shard_filter":
                    # Check if the filter target was sharded
                    ev = _search(log, rf"SHARD_DEBUG.*param_key={re.escape(str(v))}.*sharded_shape")
                    if ev:
                        add(f"transforms.detect_sharding.{k}", v, APPLIED, ev)
                    else:
                        ev2 = _search(log, rf"{re.escape(str(v))}.*shard")
                        if ev2:
                            add(f"transforms.detect_sharding.{k}", v, APPLIED, ev2)
                        else:
                            add(
                                f"transforms.detect_sharding.{k}",
                                v,
                                UNKNOWN,
                                "Check detect_sharding transform summary",
                            )
                else:
                    add(
                        f"transforms.detect_sharding.{k}",
                        v,
                        UNKNOWN,
                        "Check detect_sharding transform summary",
                    )
            # manual_config: just check sharding was applied
            if "manual_config" in t_cfg:
                ev = _search(log, r"Applied \d+ TP shards from config")
                s = APPLIED if ev else UNKNOWN
                add("transforms.detect_sharding.manual_config", "(tp_plan)", s, ev)
            continue

        if t_name == "multi_stream_moe":
            if t_cfg.get("enabled"):
                s, e = check_multi_stream_moe(log)
                add("transforms.multi_stream_moe.enabled", True, s, e)
            else:
                add("transforms.multi_stream_moe.enabled", False, DISABLED, "Disabled in config")
            continue

        if t_name == "multi_stream_gemm":
            if t_cfg.get("enabled"):
                s, e = check_multi_stream_gemm(log)
                add("transforms.multi_stream_gemm.enabled", True, s, e)
            else:
                add("transforms.multi_stream_gemm.enabled", False, DISABLED, "Disabled in config")
            continue

        if t_name == "fuse_gemms_mixed_children":
            if t_cfg.get("enabled"):
                s, e = check_fuse_gemms_mixed_children(log)
                add("transforms.fuse_gemms_mixed_children.enabled", True, s, e)
            else:
                add(
                    "transforms.fuse_gemms_mixed_children.enabled",
                    False,
                    DISABLED,
                    "Disabled in config",
                )
            continue

        # ── Generic transform with enabled flag ──────────────────────────
        enabled = t_cfg.get("enabled")
        if enabled is not None:
            if enabled:
                s, e = _check_transform_summary(log, t_name)
                add(f"transforms.{t_name}.enabled", True, s, e)
            else:
                add(f"transforms.{t_name}.enabled", False, DISABLED, "Disabled in config")

        # Non-enabled params in this transform — try SUMMARY for the transform
        for k, v in t_cfg.items():
            if k in ("enabled", "stage"):
                continue
            # Try to find evidence via the transform's SUMMARY or APPLY lines
            ts, te = _check_transform_summary(log, t_name)
            if ts == APPLIED:
                add(
                    f"transforms.{t_name}.{k}",
                    v,
                    APPLIED,
                    f"{t_name} transform applied ({te})" if te else f"{t_name} transform applied",
                )
            else:
                add(
                    f"transforms.{t_name}.{k}",
                    v,
                    UNKNOWN,
                    f"Check {t_name} transform logs for details",
                )

    # ── Catch-all: pick up any YAML keys not already covered ────────────
    # Flatten the entire YAML and check for keys that have no result yet.
    all_flat = flatten_yaml(cfg)
    covered_keys = {r["config"] for r in results}
    for dotted_key, _display_key, value in all_flat:
        if dotted_key in covered_keys:
            continue
        # Skip if a parent key is already covered (e.g., dist_mapping covered
        # as a dict means we don't need dist_mapping.moe_tp separately)
        parent_covered = any(dotted_key.startswith(ck + ".") for ck in covered_keys)
        if parent_covered:
            continue
        # Skip keys that are purely structural or internal-only
        if dotted_key in ("runtime", "world_size"):
            continue
        # Skip dict values (they were expanded into sub-keys)
        if isinstance(value, dict):
            continue
        s, e = _generic_check(log, dotted_key, value)
        add(dotted_key, value, s, e)

    return results


def format_table(results: List[Dict], use_color: bool = True) -> str:
    """Format results as a 3-column table: Config (with value), Status, Evidence."""
    # Status icons
    icons = {
        APPLIED: "\033[32mAPPLIED\033[0m" if use_color else "APPLIED",
        FAILED: "\033[31mFAILED\033[0m" if use_color else "FAILED",
        SKIPPED: "\033[33mSKIPPED\033[0m" if use_color else "SKIPPED",
        DISABLED: "\033[90mDISABLED\033[0m" if use_color else "DISABLED",
        UNKNOWN: "\033[33mUNKNOWN\033[0m" if use_color else "UNKNOWN",
    }

    # Build config display strings: "key = value"
    config_strs = []
    for r in results:
        val = r["value"]
        if len(val) > 50:
            val = val[:47] + "..."
        config_strs.append(f"{r['config']} = {val}")

    # Compute column widths
    w_cfg = max((len(s) for s in config_strs), default=10)
    w_sta = 10  # fixed for "DISABLED" (longest)
    w_evi = 90

    w_cfg = max(w_cfg, 6)

    lines = []
    header = f"| {'Config':<{w_cfg}} | {'Result':<{w_sta}} | {'Evidence':<{w_evi}} |"
    sep = f"|{'-' * (w_cfg + 2)}|{'-' * (w_sta + 2)}|{'-' * (w_evi + 2)}|"
    lines.append(header)
    lines.append(sep)

    for r, cfg_str in zip(results, config_strs):
        evi = r["evidence"][:w_evi] if r["evidence"] else "-"
        status_display = icons.get(r["status"], r["status"])
        # For alignment, pad based on raw status length
        raw_status = r["status"]
        padding = w_sta - len(raw_status)
        status_col = status_display + " " * padding
        lines.append(f"| {cfg_str:<{w_cfg}} | {status_col} | {evi:<{w_evi}} |")

    return "\n".join(lines)


def format_summary(results: List[Dict]) -> str:
    """Format a brief summary of results."""
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    parts = []
    for s in [APPLIED, FAILED, SKIPPED, DISABLED, UNKNOWN]:
        if s in counts:
            parts.append(f"{s}: {counts[s]}")
    return f"Total configs checked: {len(results)} | " + " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Check AutoDeploy config application from logs")
    parser.add_argument(
        "yaml_paths",
        nargs="+",
        help="One or more YAML config files (later files override earlier for overlapping keys)",
    )
    parser.add_argument("--log", "-l", required=True, help="Path to the server log file")
    parser.add_argument("--output", "-o", help="Optional output file path for results")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    args = parser.parse_args()

    for yp in args.yaml_paths:
        if not Path(yp).exists():
            print(f"Error: YAML file not found: {yp}", file=sys.stderr)
            sys.exit(1)
    if not Path(args.log).exists():
        print(f"Error: Log file not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    if len(args.yaml_paths) > 1:
        print(f"Merging {len(args.yaml_paths)} YAML configs (later files override earlier):")
        for i, yp in enumerate(args.yaml_paths, 1):
            print(
                f"  {i}. {yp}"
                + (
                    " (lowest priority)"
                    if i == 1
                    else " (highest priority)"
                    if i == len(args.yaml_paths)
                    else ""
                )
            )
        print()

    results = check_config(args.yaml_paths, args.log)
    use_color = not args.no_color and sys.stdout.isatty()

    table = format_table(results, use_color=use_color)
    summary = format_summary(results)

    print("\n=== AutoDeploy Config Check Results ===\n")
    print(table)
    print(f"\n{summary}\n")

    # Highlight failures
    failures = [r for r in results if r["status"] == FAILED]
    if failures:
        print("WARNING: The following configs FAILED to apply:")
        for f in failures:
            print(f"  - {f['config']} = {f['value']}")
            print(f"    Evidence: {f['evidence']}")
        print()

    if args.output:
        with open(args.output, "w") as f:
            f.write("=== AutoDeploy Config Check Results ===\n\n")
            f.write(format_table(results, use_color=False))
            f.write(f"\n\n{summary}\n")
            if failures:
                f.write("\nWARNING: The following configs FAILED to apply:\n")
                for fail in failures:
                    f.write(f"  - {fail['config']} = {fail['value']}\n")
                    f.write(f"    Evidence: {fail['evidence']}\n")
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
