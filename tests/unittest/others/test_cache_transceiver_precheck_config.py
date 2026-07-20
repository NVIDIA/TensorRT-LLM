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
"""CPU-only tests for the disagg cache-transceiver precheck config resolution.

Target: tests/scripts/perf-sanity/cache_transceiver_precheck/precheck_config.py
"""

import json
import os
import sys

import pytest

_PRECHECK_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "tests",
    "scripts",
    "perf-sanity",
    "cache_transceiver_precheck",
)
sys.path.insert(0, os.path.abspath(_PRECHECK_DIR))

import precheck_config as pcfg  # noqa: E402


def _disagg_yaml(ctx_extra=None, gen_extra=None, **overrides):
    """Minimal disagg perf-sanity yaml shaped like the checked-in configs."""
    ctx = {
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "context_parallel_size": 1,
        "enable_attention_dp": True,
        "kv_cache_config": {"dtype": "fp8"},
        "cache_transceiver_config": {"max_tokens_in_buffer": 16384, "backend": "NIXL"},
        "speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 1},
    }
    gen = {
        "tensor_parallel_size": 16,
        "pipeline_parallel_size": 1,
        "context_parallel_size": 1,
        "enable_attention_dp": True,
        "kv_cache_config": {"dtype": "fp8"},
        "cache_transceiver_config": {"max_tokens_in_buffer": 16384, "backend": "NIXL"},
        "speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 1},
    }
    ctx.update(ctx_extra or {})
    gen.update(gen_extra or {})
    cfg = {
        "metadata": {"model_name": "deepseek_r1_0528_fp4_v2"},
        "benchmark": {"mode": "e2e", "input_length": 8192, "output_length": 1024},
        "hardware": {"gpus_per_node": 4, "num_ctx_servers": 1, "num_gen_servers": 1},
        "worker_config": {"ctx": ctx, "gen": gen},
    }
    cfg.update(overrides)
    return cfg


def test_resolve_plan_adp_asymmetric():
    plan = pcfg.resolve_plan(_disagg_yaml())
    assert not plan["skip"]
    assert plan["ctx"] == {
        "tp": 4,
        "pp": 1,
        "cp": 1,
        "enable_attention_dp": True,
        "world_size": 4,
        "dp_size": 4,
    }
    assert plan["gen"]["world_size"] == 16 and plan["gen"]["dp_size"] == 16
    # Cover every gen dp rank.
    assert plan["n_pairs"] == 16
    assert plan["chunk_size"] == 8
    assert plan["request_lengths"] == [1024, 8192]
    assert plan["ctx_num_nextn_predict_layers"] == 1
    assert plan["ctx_cache_transceiver_config"]["backend"] == "NIXL"


def test_request_lengths_clamped_by_buffer_and_cap():
    cfg = _disagg_yaml(benchmark={"mode": "e2e", "input_length": 131072})
    cfg["worker_config"]["ctx"]["cache_transceiver_config"]["max_tokens_in_buffer"] = 131104
    cfg["worker_config"]["gen"]["cache_transceiver_config"]["max_tokens_in_buffer"] = 131104
    plan = pcfg.resolve_plan(cfg)
    # Derived ISL is capped by max_request_length (default 32768).
    assert plan["request_lengths"] == [1024, 32768]

    cfg["worker_config"]["ctx"]["cache_transceiver_config"]["max_tokens_in_buffer"] = 4096
    cfg["worker_config"]["gen"]["cache_transceiver_config"]["max_tokens_in_buffer"] = 4096
    plan = pcfg.resolve_plan(cfg)
    assert plan["request_lengths"] == [1024, 4096]

    # Explicit yaml override is used as-is (not capped).
    cfg["cache_transceiver_precheck"] = {"request_lengths": [64000]}
    cfg["worker_config"]["ctx"]["cache_transceiver_config"]["max_tokens_in_buffer"] = 131104
    cfg["worker_config"]["gen"]["cache_transceiver_config"]["max_tokens_in_buffer"] = 131104
    plan = pcfg.resolve_plan(cfg)
    assert plan["request_lengths"] == [64000]


def test_gen_only_no_context_skips():
    cfg = _disagg_yaml(benchmark={"mode": "gen_only_no_context", "input_length": 1024})
    plan = pcfg.resolve_plan(cfg, benchmark_mode="gen_only")
    assert plan["skip"]
    # e2e over the same yaml still runs (ctx servers are launched there).
    assert not pcfg.resolve_plan(cfg, benchmark_mode="e2e")["skip"]


def test_backend_mismatch_raises():
    cfg = _disagg_yaml(
        gen_extra={"cache_transceiver_config": {"backend": "UCX", "max_tokens_in_buffer": 16384}}
    )
    with pytest.raises(ValueError, match="backend mismatch"):
        pcfg.resolve_plan(cfg)


def test_pair_participation_and_chunks():
    plan = pcfg.resolve_plan(_disagg_yaml())
    # ADP ctx (dp4): pair k belongs to tp_rank k % 4.
    assert pcfg.pair_participates(plan, "ctx", 1, 5)
    assert not pcfg.pair_participates(plan, "ctx", 0, 5)
    # ADP gen (dp16): 1:1.
    assert pcfg.pair_participates(plan, "gen", 5, 5)
    assert not pcfg.pair_participates(plan, "gen", 4, 5)
    assert pcfg.chunks(plan) == [list(range(8)), list(range(8, 16))]
    # ctx rank owns 2 pairs per chunk of 8; gen rank owns at most 1.
    assert pcfg.max_owned_per_chunk(plan, "ctx") == 2
    assert pcfg.max_owned_per_chunk(plan, "gen") == 1

    # Non-ADP side participates everywhere and owns the whole chunk.
    plan_pp = pcfg.resolve_plan(
        _disagg_yaml(
            ctx_extra={
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 8,
                "enable_attention_dp": False,
            }
        )
    )
    assert plan_pp["ctx"]["dp_size"] == 1 and plan_pp["n_pairs"] == 16
    assert pcfg.pair_participates(plan_pp, "ctx", 0, 11)
    assert pcfg.max_owned_per_chunk(plan_pp, "ctx") == plan_pp["chunk_size"]


def test_fingerprint_role_agnostic():
    plan_a = pcfg.resolve_plan(_disagg_yaml())
    plan_b = pcfg.resolve_plan(_disagg_yaml())
    assert plan_a["fingerprint"] == plan_b["fingerprint"]
    changed = _disagg_yaml()
    changed["worker_config"]["gen"]["tensor_parallel_size"] = 8
    assert pcfg.resolve_plan(changed)["fingerprint"] != plan_a["fingerprint"]


def test_model_kv_shape_mla_and_gqa(tmp_path):
    mla = tmp_path / "mla"
    mla.mkdir()
    (mla / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 61,
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
                "num_attention_heads": 128,
            }
        )
    )
    shape = pcfg.model_kv_shape(str(mla))
    assert shape == {
        "num_layers": 61,
        "num_kv_heads": 1,
        "head_dim": 576,
        "is_mla": True,
        "source": "config.json (MLA)",
    }

    gqa = tmp_path / "gqa"
    gqa.mkdir()
    (gqa / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
            }
        )
    )
    shape = pcfg.model_kv_shape(str(gqa))
    assert shape["num_kv_heads"] == 8 and shape["head_dim"] == 128 and not shape["is_mla"]

    # Unresolvable model dir -> synthetic fallback (precheck still runs).
    assert pcfg.model_kv_shape(None)["source"] == "fallback"
    assert pcfg.model_kv_shape(str(tmp_path / "missing"))["source"] == "fallback"


def test_side_plan_views():
    plan = pcfg.resolve_plan(_disagg_yaml())
    ctx_view = pcfg.side_plan(plan, "ctx")
    gen_view = pcfg.side_plan(plan, "gen")
    assert ctx_view["parallel"]["world_size"] == 4
    assert ctx_view["peer_parallel"]["world_size"] == 16
    assert ctx_view["num_peers"] == 1 and gen_view["num_peers"] == 1
    assert gen_view["cache_transceiver_config"]["max_tokens_in_buffer"] == 16384
