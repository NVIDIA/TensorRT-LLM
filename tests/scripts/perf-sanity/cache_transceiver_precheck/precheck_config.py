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
"""Config resolution for the disagg cache-transceiver precheck.

Pure-stdlib (+pyyaml) module: NO torch / tensorrt_llm imports, so it can be
unit-tested on a CPU-only machine and imported by the launch tooling.

The precheck mirrors the *exact* disaggregated perf-sanity test configuration:
instance counts and per-side parallelism come from the same yaml
(`tests/scripts/perf-sanity/disaggregated/*.yaml`) the real test runs with,
and the transceiver is built from the yaml's own `cache_transceiver_config`
block, so config parity is by construction rather than by copying values.
"""

import json
import os

# Optional per-yaml overrides live under a `cache_transceiver_precheck:` block.
PRECHECK_DEFAULTS = {
    # Request lengths to transfer. None -> derived: [1024, benchmark ISL],
    # clamped to cache_transceiver_config.max_tokens_in_buffer and to
    # max_request_length below.
    "request_lengths": None,
    # Cap for the DERIVED lengths only (long-ISL tests would otherwise move
    # multi-GB per request in the precheck); explicit request_lengths in the
    # yaml's cache_transceiver_precheck block are used as-is.
    "max_request_length": 32768,
    # Measured requests per (peer, request_length) after warmup.
    "num_requests": 2,
    "warmup_requests": 1,
    # Upper bound on concurrently in-flight transfer pairs per chunk.
    "max_concurrent_pairs": 8,
    # signal.alarm / hang-detector budget for one chunk of transfers.
    "chunk_timeout_s": 180,
    # How long the gen side waits for the ctx rendezvous files (covers ctx
    # KV-pool allocation + NIXL/UCX transceiver bring-up).
    "rendezvous_timeout_s": 600,
    # Verify the received bytes against the deterministic fill pattern.
    "verify_data": True,
}

# Fallback KV shape when the model directory cannot be resolved: the precheck
# still exercises the exact network path, just with a synthetic cache shape.
FALLBACK_KV_SHAPE = {
    "num_layers": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "is_mla": False,
    "source": "fallback",
}


def _side_cfg(cfg, role):
    worker_config = cfg.get("worker_config", {}) or {}
    side = worker_config.get(role, {}) or {}
    if not side:
        raise ValueError(f"worker_config.{role} missing in disagg yaml")
    return side


def _parallel(side):
    tp = int(side.get("tensor_parallel_size", 1))
    pp = int(side.get("pipeline_parallel_size", 1))
    cp = int(side.get("context_parallel_size", 1))
    adp = bool(side.get("enable_attention_dp", False))
    world = tp * pp * cp
    # With attention DP the request-level data parallel width is the attention
    # TP width (each dp rank owns whole requests / full KV heads).
    dp_size = tp if adp else 1
    return {
        "tp": tp,
        "pp": pp,
        "cp": cp,
        "enable_attention_dp": adp,
        "world_size": world,
        "dp_size": dp_size,
    }


def _spec_nextn(side):
    spec = side.get("speculative_config") or {}
    decoding_type = str(spec.get("decoding_type", "")).upper()
    if decoding_type == "MTP":
        return int(spec.get("num_nextn_predict_layers", 0) or 0)
    return 0


def resolve_plan(cfg, benchmark_mode="e2e"):
    """Build the shared precheck plan both roles must agree on.

    `cfg` is the parsed disagg perf-sanity yaml. Returns a plain dict; the
    `fingerprint` field is exchanged over the rendezvous channel so a ctx/gen
    disagreement (e.g. mismatched yamls) fails fast with a clear error.
    """
    hardware = cfg.get("hardware", {}) or {}
    benchmark = cfg.get("benchmark", {}) or {}

    yaml_mode = str(benchmark.get("mode", ""))
    if benchmark_mode == "gen_only" and "gen_only_no_context" in yaml_mode:
        # No ctx workers at launch -> no KV transfer in the real test either.
        return {"skip": True, "skip_reason": "gen_only_no_context mode has no KV transfer"}

    num_ctx_servers = int(hardware.get("num_ctx_servers", 0) or 0)
    num_gen_servers = int(hardware.get("num_gen_servers", 0) or 0)
    if num_ctx_servers < 1 or num_gen_servers < 1:
        raise ValueError(
            f"hardware.num_ctx_servers/num_gen_servers must be >= 1, got "
            f"{num_ctx_servers}/{num_gen_servers}"
        )
    gpus_per_node = int(hardware.get("gpus_per_node", 0) or 0)
    if gpus_per_node < 1:
        raise ValueError("hardware.gpus_per_node is required")

    ctx_side = _side_cfg(cfg, "ctx")
    gen_side = _side_cfg(cfg, "gen")
    ctx = _parallel(ctx_side)
    gen = _parallel(gen_side)

    ctx_xcvr = ctx_side.get("cache_transceiver_config") or {}
    gen_xcvr = gen_side.get("cache_transceiver_config") or {}
    if not ctx_xcvr.get("backend") and not gen_xcvr.get("backend"):
        return {"skip": True, "skip_reason": "no cache_transceiver_config.backend in yaml"}
    if ctx_xcvr.get("backend") != gen_xcvr.get("backend"):
        raise ValueError(
            f"ctx/gen cache_transceiver_config.backend mismatch: "
            f"{ctx_xcvr.get('backend')} vs {gen_xcvr.get('backend')}"
        )

    ctx_nextn = _spec_nextn(ctx_side)
    gen_nextn = _spec_nextn(gen_side)

    knobs = dict(PRECHECK_DEFAULTS)
    knobs.update(cfg.get("cache_transceiver_precheck", {}) or {})

    tokens_per_block = int(
        (ctx_side.get("kv_cache_config") or {}).get("tokens_per_block", 32) or 32
    )
    max_tokens_in_buffer = ctx_xcvr.get("max_tokens_in_buffer") or gen_xcvr.get(
        "max_tokens_in_buffer"
    )

    req_lens = knobs["request_lengths"]
    if not req_lens:
        isl = int(benchmark.get("input_length", 1024) or 1024)
        req_lens = [1024, min(isl, int(knobs["max_request_length"]))]
    if max_tokens_in_buffer:
        req_lens = [min(int(r), int(max_tokens_in_buffer)) for r in req_lens]
    req_lens = sorted({max(int(r), tokens_per_block) for r in req_lens})

    # Transfer pairs: with attention DP each dp rank owns whole requests, so
    # cover every dp rank on both sides; without DP a single request already
    # involves every rank of the instance (KV is sharded across TP/PP).
    n_pairs = max(ctx["dp_size"], gen["dp_size"], 1)
    chunk_size = max(1, min(n_pairs, int(knobs["max_concurrent_pairs"])))

    plan = {
        "skip": False,
        "num_ctx_servers": num_ctx_servers,
        "num_gen_servers": num_gen_servers,
        "gpus_per_node": gpus_per_node,
        "ctx": ctx,
        "gen": gen,
        "ctx_cache_transceiver_config": ctx_xcvr,
        "gen_cache_transceiver_config": gen_xcvr,
        "ctx_num_nextn_predict_layers": ctx_nextn,
        "gen_num_nextn_predict_layers": gen_nextn,
        "ctx_kv_dtype": str((ctx_side.get("kv_cache_config") or {}).get("dtype", "auto")),
        "gen_kv_dtype": str((gen_side.get("kv_cache_config") or {}).get("dtype", "auto")),
        "tokens_per_block": tokens_per_block,
        "request_lengths": req_lens,
        "num_requests": int(knobs["num_requests"]),
        "warmup_requests": int(knobs["warmup_requests"]),
        "n_pairs": n_pairs,
        "chunk_size": chunk_size,
        "chunk_timeout_s": int(knobs["chunk_timeout_s"]),
        "rendezvous_timeout_s": int(knobs["rendezvous_timeout_s"]),
        "verify_data": bool(knobs["verify_data"]),
    }
    plan["fingerprint"] = plan_fingerprint(plan)
    return plan


def plan_fingerprint(plan):
    """Stable string both sides must agree on before transferring."""
    keys = (
        "num_ctx_servers",
        "num_gen_servers",
        "ctx",
        "gen",
        "ctx_cache_transceiver_config",
        "gen_cache_transceiver_config",
        "ctx_num_nextn_predict_layers",
        "gen_num_nextn_predict_layers",
        "tokens_per_block",
        "request_lengths",
        "num_requests",
        "warmup_requests",
        "n_pairs",
        "chunk_size",
    )
    return json.dumps({k: plan[k] for k in keys}, sort_keys=True)


def side_plan(plan, role):
    """Per-role view: this role's parallelism + transceiver/kv config."""
    other = "gen" if role == "ctx" else "ctx"
    return {
        "role": role,
        "parallel": plan[role],
        "peer_parallel": plan[other],
        "cache_transceiver_config": plan[f"{role}_cache_transceiver_config"],
        "kv_dtype": plan[f"{role}_kv_dtype"],
        "num_nextn_predict_layers": plan[f"{role}_num_nextn_predict_layers"],
        "num_peers": plan["num_gen_servers" if role == "ctx" else "num_ctx_servers"],
    }


def pair_participates(plan, role, tp_rank, pair_idx):
    """Whether this rank takes part in transfer pair `pair_idx`.

    Attention-DP side: pair k belongs to dp rank k % dp_size (every pp stage
    of that dp rank participates). Non-DP side: KV is sharded across the whole
    instance, so every rank participates in every pair.
    """
    side = plan[role]
    if not side["enable_attention_dp"]:
        return True
    return tp_rank == pair_idx % side["dp_size"]


def owned_pairs(plan, role, tp_rank, chunk_pairs):
    return [k for k in chunk_pairs if pair_participates(plan, role, tp_rank, k)]


def max_owned_per_chunk(plan, role):
    """Max concurrently owned pairs per rank (KV pool sizing)."""
    side = plan[role]
    chunk = plan["chunk_size"]
    if not side["enable_attention_dp"]:
        return chunk
    return (chunk + side["dp_size"] - 1) // side["dp_size"]


def chunks(plan):
    """Pair indices grouped into concurrency-bounded chunks."""
    pairs = list(range(plan["n_pairs"]))
    size = plan["chunk_size"]
    return [pairs[i : i + size] for i in range(0, len(pairs), size)]


# --------------------------------------------------------------------------- #
# Model KV shape resolution
# --------------------------------------------------------------------------- #
def _load_model_path_dict(llm_src):
    """Import MODEL_PATH_DICT from tests/integration/defs/perf/_model_paths.py."""
    path = os.path.join(llm_src, "tests", "integration", "defs", "perf", "_model_paths.py")
    namespace = {}
    with open(path) as f:
        exec(compile(f.read(), path, "exec"), namespace)  # noqa: S102 - repo-local constants file
    return namespace["MODEL_PATH_DICT"]


def resolve_model_dir(cfg, llm_src=None, llm_models_root=None):
    """Resolve the local model directory the same way test_perf_sanity does.

    metadata.model_name -> MODEL_PATH_DICT -> $LLM_MODELS_ROOT/<relpath>,
    falling back to metadata.model_dir_name under LLM_MODELS_ROOT. Returns
    None when nothing resolvable exists (precheck then uses FALLBACK_KV_SHAPE).
    """
    llm_models_root = llm_models_root or os.environ.get("LLM_MODELS_ROOT", "")
    if not llm_models_root:
        return None
    metadata = cfg.get("metadata", {}) or {}
    candidates = []
    model_name = metadata.get("model_name", "")
    if llm_src and model_name:
        try:
            rel = _load_model_path_dict(llm_src).get(model_name)
            if rel:
                candidates.append(os.path.join(llm_models_root, rel))
        except (OSError, KeyError, SyntaxError):
            pass
    if metadata.get("model_dir_name"):
        candidates.append(os.path.join(llm_models_root, metadata["model_dir_name"]))
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "config.json")):
            return cand
    return None


def model_kv_shape(model_dir):
    """KV cache shape (per-token layout) from the model's config.json.

    Handles MLA checkpoints (kv_lora_rank present: one latent 'head' of
    kv_lora_rank + qk_rope_head_dim) and GQA/MHA. `num_layers` excludes the
    MTP nextn layers -- those are added by the KV cache manager via
    spec_config, mirroring real serving.
    """
    if not model_dir:
        return dict(FALLBACK_KV_SHAPE)
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            hf_cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return dict(FALLBACK_KV_SHAPE)
    if isinstance(hf_cfg.get("text_config"), dict):
        hf_cfg = hf_cfg["text_config"]

    num_layers = hf_cfg.get("num_hidden_layers")
    if num_layers is None:
        return dict(FALLBACK_KV_SHAPE)

    if hf_cfg.get("kv_lora_rank"):  # MLA (DeepSeek-family, Kimi K2, ...)
        return {
            "num_layers": int(num_layers),
            "num_kv_heads": 1,
            "head_dim": int(hf_cfg["kv_lora_rank"]) + int(hf_cfg.get("qk_rope_head_dim", 0)),
            "is_mla": True,
            "source": "config.json (MLA)",
        }

    num_heads = hf_cfg.get("num_attention_heads", 1)
    num_kv_heads = hf_cfg.get("num_key_value_heads", num_heads)
    head_dim = hf_cfg.get("head_dim")
    if not head_dim:
        hidden = hf_cfg.get("hidden_size")
        head_dim = hidden // num_heads if hidden and num_heads else 128
    return {
        "num_layers": int(num_layers),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
        "is_mla": False,
        "source": "config.json",
    }
