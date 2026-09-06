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
import shlex
import subprocess
import sys
import time
import types

import pytest

__extra_import_path__ = ["~/tests/scripts/perf-sanity/cache_transceiver_precheck"]
import precheck_config as pcfg
import run_precheck as rp

_PRECHECK_DIR = os.path.dirname(os.path.abspath(pcfg.__file__))


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
    assert plan["wave_size"] == 8
    assert plan["request_lengths"] == [1024, 8192]
    assert plan["ctx_num_nextn_predict_layers"] == 1
    assert plan["ctx_cache_transceiver_config"]["backend"] == "NIXL"


def test_spec_nextn_max_draft_len_fallback():
    # Checked-in yamls spell the MTP depth either way (num_nextn_predict_layers
    # is MTPDecodingConfig's deprecated alias of max_draft_len); both must
    # resolve to the same spec-layer count.
    cfg = _disagg_yaml(
        ctx_extra={"speculative_config": {"decoding_type": "MTP", "max_draft_len": 3}},
        gen_extra={
            "speculative_config": {
                "decoding_type": "MTP",
                "num_nextn_predict_layers": 3,
                "max_draft_len": 3,
            }
        },
    )
    plan = pcfg.resolve_plan(cfg)
    assert plan["ctx_num_nextn_predict_layers"] == 3
    assert plan["gen_num_nextn_predict_layers"] == 3

    # Non-MTP speculation never contributes MTP KV layers, even with a
    # max_draft_len present.
    cfg = _disagg_yaml(
        ctx_extra={"speculative_config": {"decoding_type": "Eagle", "max_draft_len": 3}},
        gen_extra={"speculative_config": {"decoding_type": "Eagle", "max_draft_len": 3}},
    )
    plan = pcfg.resolve_plan(cfg)
    assert plan["ctx_num_nextn_predict_layers"] == 0
    assert plan["gen_num_nextn_predict_layers"] == 0


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


def test_pair_participation_and_waves():
    plan = pcfg.resolve_plan(_disagg_yaml())
    # ADP ctx (dp4): pair k belongs to tp_rank k % 4.
    assert pcfg.pair_participates(plan, "ctx", 1, 5)
    assert not pcfg.pair_participates(plan, "ctx", 0, 5)
    # ADP gen (dp16): 1:1.
    assert pcfg.pair_participates(plan, "gen", 5, 5)
    assert not pcfg.pair_participates(plan, "gen", 4, 5)
    assert pcfg.waves(plan) == [list(range(8)), list(range(8, 16))]
    # ctx rank owns 2 pairs per wave of 8; gen rank owns at most 1.
    assert pcfg.max_owned_per_wave(plan, "ctx") == 2
    assert pcfg.max_owned_per_wave(plan, "gen") == 1

    # Non-ADP side participates everywhere and owns the whole wave.
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
    assert pcfg.max_owned_per_wave(plan_pp, "ctx") == plan_pp["wave_size"]


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
        "vocab_size": None,
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
    assert ctx_view["num_peers"] == 1 and gen_view["num_peers"] == 1
    assert gen_view["cache_transceiver_config"]["max_tokens_in_buffer"] == 16384


class TestControlWireFormat:
    """run_precheck's HMAC-JSON control frames (importable without torch)."""

    def test_roundtrip(self):
        key = b"\x01" * 32
        msg = ["go", {"li": 0, "rep": 1, "wave": 2}]
        assert rp.unpack_msg(rp.pack_msg(msg, key), key) == msg

    def test_tampered_frame_rejected(self):
        key = b"\x01" * 32
        raw = rp.pack_msg(["hello", {}], key)
        bad = raw[:-1] + bytes([raw[-1] ^ 0xFF])
        with pytest.raises(rp._TransferError):
            rp.unpack_msg(bad, key)

    def test_wrong_key_rejected(self):
        raw = rp.pack_msg(["hello", {}], b"\x01" * 32)
        with pytest.raises(rp._TransferError):
            rp.unpack_msg(raw, b"\x02" * 32)

    def test_short_frame_rejected(self):
        with pytest.raises(rp._TransferError):
            rp.unpack_msg(b"tiny", b"\x01" * 32)

    def test_addr_file_owner_only(self, tmp_path):
        path = str(tmp_path / "rendezvous" / "ctx0_gen0.addr")
        rp.write_addr(path, {"host": "h", "port": 1, "key": "aa"})
        assert (os.stat(path).st_mode & 0o777) == 0o600
        with open(path) as f:
            assert json.load(f)["key"] == "aa"


def test_use_kv_cache_manager_v2_flags():
    # Absent -> "auto" (the driver resolves it against the model's
    # get_preferred_kv_cache_manager_version at runtime, like serving).
    plan = pcfg.resolve_plan(_disagg_yaml())
    assert plan["ctx_use_kv_cache_manager_v2"] == "auto"
    assert plan["gen_use_kv_cache_manager_v2"] == "auto"
    assert pcfg.side_plan(plan, "ctx")["use_kv_cache_manager_v2"] == "auto"

    # Explicit yaml values win, per side.
    plan = pcfg.resolve_plan(
        _disagg_yaml(
            ctx_extra={"kv_cache_config": {"dtype": "fp8", "use_kv_cache_manager_v2": False}},
            gen_extra={"kv_cache_config": {"dtype": "fp8", "use_kv_cache_manager_v2": True}},
        )
    )
    assert plan["ctx_use_kv_cache_manager_v2"] is False
    assert plan["gen_use_kv_cache_manager_v2"] is True
    assert pcfg.side_plan(plan, "gen")["use_kv_cache_manager_v2"] is True


def test_resolve_model_prefs_auto_requires_registered_model(monkeypatch):
    monkeypatch.setattr(rp, "load_internal_apis", lambda: types.SimpleNamespace())
    monkeypatch.setattr(rp, "_lookup_model_cls", lambda _model_dir: (None, None))
    cache_cfg = types.SimpleNamespace(transceiver_runtime="PYTHON")

    with pytest.raises(RuntimeError, match="refusing to assume V1"):
        rp.resolve_model_prefs(None, {"use_kv_cache_manager_v2": "auto"}, cache_cfg)


def test_resolve_model_prefs_auto_propagates_model_preference_failure(monkeypatch):
    class FailingModel:
        @classmethod
        def get_preferred_kv_cache_manager_version(cls, _pretrained_config):
            raise RuntimeError("model hook failed")

    def resolve_v2(_shim, model_cls, pretrained_config):
        return model_cls.get_preferred_kv_cache_manager_version(pretrained_config)

    api = types.SimpleNamespace(
        TorchLlmArgs=lambda **kwargs: types.SimpleNamespace(**kwargs),
        resolve_kv_cache_manager_v2_auto=resolve_v2,
    )
    monkeypatch.setattr(rp, "load_internal_apis", lambda: api)
    monkeypatch.setattr(rp, "_lookup_model_cls", lambda _model_dir: (FailingModel, object()))
    cache_cfg = types.SimpleNamespace(transceiver_runtime="PYTHON")
    side = {
        "use_kv_cache_manager_v2": "auto",
        "parallel": {"tp": 2, "pp": 1, "cp": 1},
    }

    with pytest.raises(RuntimeError, match="V2 'auto' resolution failed.*refusing to assume V1"):
        rp.resolve_model_prefs("/model", side, cache_cfg)


def test_resolve_model_prefs_passes_model_metadata_to_resolver(monkeypatch):
    captured = []

    class Model:
        pass

    def resolve_v2(shim, model_cls, pretrained_config):
        captured.append((shim, model_cls, pretrained_config))
        return True

    api = types.SimpleNamespace(
        TorchLlmArgs=lambda **kwargs: types.SimpleNamespace(**kwargs),
        MTPDecodingConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        resolve_kv_cache_manager_v2_auto=resolve_v2,
    )
    hf_view = object()
    monkeypatch.setattr(rp, "load_internal_apis", lambda: api)
    monkeypatch.setattr(rp, "_lookup_model_cls", lambda _model_dir: (Model, hf_view))
    cache_cfg = types.SimpleNamespace(transceiver_runtime="PYTHON")
    side = {
        "use_kv_cache_manager_v2": "auto",
        "parallel": {"tp": 4, "pp": 2, "cp": 3},
        "num_nextn_predict_layers": 3,
    }

    assert rp.resolve_model_prefs("/model", side, cache_cfg)
    resolver_args, model_cls, pretrained_config = captured.pop()
    assert resolver_args.model == "/model"
    assert resolver_args.tensor_parallel_size == 4
    assert resolver_args.pipeline_parallel_size == 2
    assert resolver_args.context_parallel_size == 3
    assert resolver_args.kv_cache_config == {"use_kv_cache_manager_v2": "auto"}
    assert resolver_args.cache_transceiver_config is cache_cfg
    assert resolver_args.speculative_config.num_nextn_predict_layers == 3
    assert model_cls is Model
    assert pretrained_config is hf_view


def test_resolve_model_prefs_auto_propagates_resolver_failure(monkeypatch):
    class Model:
        pass

    def fail_resolver(_shim, _model_cls, _pretrained_config):
        raise RuntimeError("resolver failed")

    api = types.SimpleNamespace(
        TorchLlmArgs=lambda **kwargs: types.SimpleNamespace(**kwargs),
        resolve_kv_cache_manager_v2_auto=fail_resolver,
    )
    monkeypatch.setattr(rp, "load_internal_apis", lambda: api)
    monkeypatch.setattr(rp, "_lookup_model_cls", lambda _model_dir: (Model, object()))
    cache_cfg = types.SimpleNamespace(transceiver_runtime="PYTHON")
    side = {
        "use_kv_cache_manager_v2": "auto",
        "parallel": {"tp": 1, "pp": 1, "cp": 1},
    }

    with pytest.raises(RuntimeError, match="V2 'auto' resolution failed.*refusing to assume V1"):
        rp.resolve_model_prefs("/model", side, cache_cfg)


def test_resolve_model_prefs_explicit_v1_does_not_require_model(monkeypatch):
    monkeypatch.setattr(rp, "load_internal_apis", lambda: types.SimpleNamespace())
    monkeypatch.setattr(rp, "_lookup_model_cls", lambda _model_dir: (None, None))
    cache_cfg = types.SimpleNamespace(transceiver_runtime="PYTHON")

    assert not rp.resolve_model_prefs(None, {"use_kv_cache_manager_v2": False}, cache_cfg)


def test_resolve_model_prefs_runtime_auto_failure_is_not_silently_demoted(monkeypatch):
    def fail_runtime_resolver(*_args):
        raise RuntimeError("model runtime hook failed")

    api = types.SimpleNamespace(resolve_transceiver_runtime_auto=fail_runtime_resolver)
    monkeypatch.setattr(rp, "load_internal_apis", lambda: api)
    monkeypatch.setattr(rp, "_lookup_model_cls", lambda _model_dir: (type("Model", (), {}), None))
    cache_cfg = types.SimpleNamespace(transceiver_runtime="auto")

    with pytest.raises(RuntimeError, match="may differ from serving"):
        rp.resolve_model_prefs(
            "/model",
            {"use_kv_cache_manager_v2": False},
            cache_cfg,
        )


def test_model_kv_shape_vocab_size(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 129280,
            }
        )
    )
    assert pcfg.model_kv_shape(str(model_dir))["vocab_size"] == 129280


class TestRendezvousStaleness:
    """wait_for_addr must skip addr files stamped by a previous run."""

    def test_same_job_accepted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        p = str(tmp_path / "rendezvous" / "ctx0_gen0.addr")
        rp.write_addr(p, {"host": "h", "port": 1, "key": "aa"})
        got = rp.wait_for_addr(p, timeout_s=2)
        assert got["job"] == "12345" and got["port"] == 1

    def test_stale_job_skipped_until_timeout(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "11111")
        p = str(tmp_path / "rendezvous" / "ctx0_gen0.addr")
        rp.write_addr(p, {"host": "h", "port": 1, "key": "aa"})  # stamped 11111
        monkeypatch.setenv("SLURM_JOB_ID", "22222")  # new run
        with pytest.raises(rp._Timeout):
            rp.wait_for_addr(p, timeout_s=2)

    def test_no_job_id_accepts_any(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "11111")
        p = str(tmp_path / "rendezvous" / "ctx0_gen0.addr")
        rp.write_addr(p, {"host": "h", "port": 1, "key": "aa"})
        monkeypatch.delenv("SLURM_JOB_ID")  # manual non-slurm run
        assert rp.wait_for_addr(p, timeout_s=2)["port"] == 1

    def test_write_addr_replaces_stale_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "11111")
        p = str(tmp_path / "rendezvous" / "ctx0_gen0.addr")
        rp.write_addr(p, {"host": "old", "port": 1, "key": "aa"})
        monkeypatch.setenv("SLURM_JOB_ID", "22222")
        rp.write_addr(p, {"host": "new", "port": 2, "key": "bb"})
        got = rp.wait_for_addr(p, timeout_s=2)
        assert got["host"] == "new" and got["job"] == "22222"


def test_wireup_timeout_derivation():
    plan = pcfg.resolve_plan(_disagg_yaml())  # ctx dep4 -> gen dep16
    assert plan["wireup_timeout_s"] == 600
    plan = pcfg.resolve_plan(_disagg_yaml(gen_extra={"tensor_parallel_size": 4}))
    assert plan["wireup_timeout_s"] == 600
    plan = pcfg.resolve_plan(_disagg_yaml(cache_transceiver_precheck={"wireup_timeout_s": 42}))
    assert plan["wireup_timeout_s"] == 42


def test_timeout_budget_is_bounded_and_layered():
    one_peer = pcfg.timeout_budget(_disagg_yaml(), max_world=16)
    assert one_peer == {
        "wireup_timeout_s": 600,
        "longest_phase_timeout_s": 900,
        "watchdog_timeout_s": 960,
        "step_timeout_s": 1200,
    }

    cfg = _disagg_yaml(hardware={"gpus_per_node": 4, "num_ctx_servers": 12, "num_gen_servers": 1})
    multi_peer = pcfg.timeout_budget(cfg, max_world=32)
    # Even 12 serialized ctx peers retain the 20-minute default health limit.
    # Progress refreshes protect legitimate per-peer phases without scaling
    # the external backstop with topology.
    assert multi_peer["longest_phase_timeout_s"] == 900
    assert multi_peer["step_timeout_s"] == 1200

    cfg["cache_transceiver_precheck"] = {"step_timeout_s": 900}
    with pytest.raises(ValueError, match="must exceed the longest phase watchdog"):
        pcfg.timeout_budget(cfg, max_world=32)

    cfg["cache_transceiver_precheck"] = {"step_timeout_s": 1800}
    assert pcfg.timeout_budget(cfg, max_world=32)["step_timeout_s"] == 1800

    cfg["cache_transceiver_precheck"] = {"step_timeout_s": 1801}
    with pytest.raises(ValueError, match="must not exceed the global health-check limit"):
        pcfg.timeout_budget(cfg, max_world=32)


@pytest.mark.parametrize("model_root", ["/models with spaces", "/models/o'hare"])
def test_precheck_commands_propagate_model_root(monkeypatch, model_root):
    # CI provides the model root inside its inbound pytest command, not in the
    # environment of the Python process that generates the launch script.
    monkeypatch.delenv("LLM_MODELS_ROOT", raising=False)
    monkeypatch.delenv("TRTLLM_DISAGG_CT_PRECHECK", raising=False)
    lines = pcfg.precheck_prefix_lines(
        {},
        "e2e",
        "$config",
        "unset UCX_TLS &&",
        max_world=8,
        llm_models_root=model_root,
    )
    shell_script = "\n".join(
        [
            "CTX_WORKER_ENV_VARS=",
            "GEN_WORKER_ENV_VARS=",
            "PYTEST_COMMON_VARS=",
            "llmSrcNode=/repo",
            "testOutputDir=/tmp/output",
            "config=/tmp/config.yaml",
            *lines,
            "printf '%s\\n' \"$pytestCommandCTXPrecheck\"",
            "printf '%s\\n' \"$pytestCommandGENPrecheck\"",
        ]
    )

    result = subprocess.run(
        ["bash"], input=shell_script, capture_output=True, check=True, text=True
    )
    commands = result.stdout.splitlines()

    assert len(commands) == 2
    for command in commands:
        tokens = shlex.split(command)
        assignment = f"LLM_MODELS_ROOT={model_root}"
        assert assignment in tokens
        assert tokens.index(assignment) < tokens.index("python3")


def test_precheck_commands_split_pytest_common_vars(monkeypatch):
    # $PYTEST_COMMON_VARS is spliced unquoted on purpose: bash word splitting
    # must yield separate K=V env-assignment tokens ahead of the executable.
    # Values containing spaces are unsupported by design — this pins the
    # expected splitting behavior.
    monkeypatch.delenv("LLM_MODELS_ROOT", raising=False)
    monkeypatch.delenv("TRTLLM_DISAGG_CT_PRECHECK", raising=False)
    lines = pcfg.precheck_prefix_lines(
        {},
        "e2e",
        "$config",
        "unset UCX_TLS &&",
        max_world=8,
        llm_models_root="/models",
    )
    shell_script = "\n".join(
        [
            "CTX_WORKER_ENV_VARS=",
            "GEN_WORKER_ENV_VARS=",
            'PYTEST_COMMON_VARS="FOO=1 BAR=two"',
            "llmSrcNode=/repo",
            "testOutputDir=/tmp/output",
            "config=/tmp/config.yaml",
            *lines,
            "printf '%s\\n' \"$pytestCommandCTXPrecheck\"",
        ]
    )

    result = subprocess.run(
        ["bash"], input=shell_script, capture_output=True, check=True, text=True
    )
    tokens = shlex.split(result.stdout.splitlines()[0])

    python_index = tokens.index("python3")
    for assignment in ("FOO=1", "BAR=two"):
        assert tokens.index(assignment) < python_index


def _enabled_line(cfg):
    lines = pcfg.precheck_prefix_lines(
        cfg,
        "e2e",
        "$c",
        "unset &&",
        max_world=8,
        llm_models_root="/models",
    )
    return next(x for x in lines if x.startswith("export ctPrecheckEnabled"))


@pytest.mark.parametrize(
    "model_root",
    (
        "/models with spaces",
        "/models/it's",
        "/models/$HOME/$(must-not-run)",
    ),
)
def test_precheck_commands_export_model_root_safely(model_root, monkeypatch):
    monkeypatch.delenv("TRTLLM_DISAGG_CT_PRECHECK", raising=False)
    lines = pcfg.precheck_prefix_lines(
        _disagg_yaml(cache_transceiver_precheck={"enabled": True}),
        "e2e",
        "$config",
        "unset UCX_TLS &&",
        max_world=8,
        llm_models_root=model_root,
    )

    commands = [line for line in lines if "pytestCommand" in line]
    assert len(commands) == 2
    assert all("python3" in line for line in commands)

    script = "\n".join(lines) + '\nprintf "%s" "$LLM_MODELS_ROOT"\n'
    result = subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout == model_root


def test_disabled_precheck_does_not_require_or_export_model_root(monkeypatch):
    monkeypatch.delenv("TRTLLM_DISAGG_CT_PRECHECK", raising=False)

    lines = pcfg.precheck_prefix_lines(
        _disagg_yaml(cache_transceiver_precheck={"enabled": False}),
        "e2e",
        "$config",
        "unset UCX_TLS &&",
        max_world=8,
    )

    assert "export ctPrecheckEnabled=0" in lines
    assert not any(line.startswith("export LLM_MODELS_ROOT=") for line in lines)


@pytest.mark.parametrize("llm_models_root", [None, ""])
def test_enabled_precheck_requires_model_root(monkeypatch, llm_models_root):
    monkeypatch.delenv("TRTLLM_DISAGG_CT_PRECHECK", raising=False)

    with pytest.raises(ValueError, match="requires LLM_MODELS_ROOT"):
        pcfg.precheck_prefix_lines(
            _disagg_yaml(cache_transceiver_precheck={"enabled": True}),
            "e2e",
            "$config",
            "unset UCX_TLS &&",
            max_world=8,
            llm_models_root=llm_models_root,
        )


def test_precheck_enabled_helper(monkeypatch):
    # submit.py consults this helper to decide whether a missing model root is
    # fatal — it must mirror the policy encoded in ctPrecheckEnabled.
    monkeypatch.delenv("TRTLLM_DISAGG_CT_PRECHECK", raising=False)
    assert pcfg.PRECHECK_DEFAULTS["enabled"] is True
    assert pcfg.precheck_enabled({}) is True
    assert pcfg.precheck_enabled({"cache_transceiver_precheck": {"enabled": False}}) is False
    monkeypatch.setenv("TRTLLM_DISAGG_CT_PRECHECK", "0")
    assert pcfg.precheck_enabled({}) is False
    monkeypatch.setenv("TRTLLM_DISAGG_CT_PRECHECK", "true")
    assert pcfg.precheck_enabled({"cache_transceiver_precheck": {"enabled": False}}) is True


def test_skip_waived_case_overrides_force_enable(monkeypatch):
    monkeypatch.setenv("TRTLLM_DISAGG_CT_PRECHECK", "1")
    lines = pcfg.precheck_prefix_lines(
        {},
        "e2e",
        "$c",
        "unset &&",
        max_world=8,
        skip_precheck=True,
    )
    assert next(x for x in lines if x.startswith("export ctPrecheckEnabled")).endswith("=0")


def test_precheck_env_kill_switch_truthy(monkeypatch):
    """The TRTLLM_DISAGG_CT_PRECHECK kill switch parses the usual boolean spellings.

    So a force-enable like =true is not silently read as "off", and anything
    ambiguous is rejected instead of guessed at.
    """
    cfg = {"cache_transceiver_precheck": {"enabled": True}}
    monkeypatch.delenv("TRTLLM_DISAGG_CT_PRECHECK", raising=False)
    assert _enabled_line(cfg).endswith("=1")  # yaml opt-in
    assert _enabled_line({}).endswith("=1")  # on by default
    for v in ("1", "true", "on", "YES", " True "):
        monkeypatch.setenv("TRTLLM_DISAGG_CT_PRECHECK", v)
        assert _enabled_line(cfg).endswith("=1"), v
    for v in ("0", "false", "off", "no"):
        monkeypatch.setenv("TRTLLM_DISAGG_CT_PRECHECK", v)
        assert _enabled_line(cfg).endswith("=0"), v
    # env overrides yaml either way (kill switch): yaml opt-out but env force-on
    monkeypatch.setenv("TRTLLM_DISAGG_CT_PRECHECK", "true")
    assert _enabled_line({"cache_transceiver_precheck": {"enabled": False}}).endswith("=1")
    monkeypatch.setenv("TRTLLM_DISAGG_CT_PRECHECK", "maybe")
    with pytest.raises(ValueError):
        _enabled_line(cfg)


def test_gate_library_content(tmp_path):
    """The gate library loads from next to the draft, with an in-repo fallback.

    It falls back to the in-repo copy for an external draft, strips blank
    lines, and errors if truly absent.
    """
    dd = tmp_path / "disaggregated"
    dd.mkdir()
    (dd / "slurm_ct_precheck_gate.sh").write_text(
        "run_cache_transceiver_precheck() { :; }\n\n\nrun_cache_transceiver_precheck\n"
    )
    draft = str(dd / "slurm_launch_draft.sh")
    got = pcfg.gate_library_content(draft, str(tmp_path))
    assert "run_cache_transceiver_precheck()" in got
    assert got.endswith("\n") and "\n\n" not in got  # blank lines stripped

    # external draft -> fall back to <llm_src>/jenkins/.../slurm_ct_precheck_gate.sh
    repo = tmp_path / "repo"
    gate2 = repo / "jenkins" / "scripts" / "perf" / "disaggregated" / "slurm_ct_precheck_gate.sh"
    gate2.parent.mkdir(parents=True)
    gate2.write_text("echo hi\n")
    assert pcfg.gate_library_content("/nowhere/draft.sh", str(repo)) == "echo hi\n"

    with pytest.raises(FileNotFoundError):
        pcfg.gate_library_content("/nowhere/draft.sh", str(tmp_path / "empty"))


@pytest.mark.parametrize(
    ("exit_code", "expected_verdict"),
    ((124, "EXTERNAL_TIMEOUT"), (137, "EXTERNAL_KILL")),
)
def test_gate_records_external_timeout_verdict(tmp_path, exit_code, expected_verdict):
    gate = os.path.join(
        os.path.dirname(_PRECHECK_DIR),
        "..",
        "..",
        "..",
        "jenkins",
        "scripts",
        "perf",
        "disaggregated",
        "slurm_ct_precheck_gate.sh",
    )
    gate = os.path.abspath(gate)
    shell_script = f"""
source {shlex.quote(gate)}
timeout() {{ return {exit_code}; }}
sleep() {{ :; }}
cleanup_on_failure() {{ :; }}
ctPrecheckEnabled=1
ctPrecheckTimeout=1200
testOutputDir={shlex.quote(str(tmp_path / "output"))}
jobWorkspace={shlex.quote(str(tmp_path / "workspace"))}
mkdir -p "$jobWorkspace"
numGenServers=1
numCtxServers=1
nodesPerGenServer=1
nodesPerCtxServer=1
gpusPerNodePerGenServer=1
gpusPerNodePerCtxServer=1
genNodeLists=(gen-node)
ctxNodeLists=(ctx-node)
srunArgs=()
pytestCommandGENPrecheck=gen-command
pytestCommandCTXPrecheck=ctx-command
precheckRunScript=/unused
run_cache_transceiver_precheck
"""
    subprocess.run(["bash"], input=shell_script, capture_output=True, check=True, text=True)

    status_dir = tmp_path / "output" / "cache_transceiver_precheck" / "status"
    for name in ("gen_0", "ctx_0"):
        verdict = (status_dir / f"{name}.status").read_text()
        assert verdict.startswith(f"{expected_verdict} {name}:")
        if exit_code == 124:
            assert "1200s total-runtime backstop" in verdict
        else:
            assert "possibly timeout -k escalation" in verdict
    junit = (tmp_path / "workspace" / "results-ct-precheck.xml").read_text()
    assert expected_verdict in junit
    assert "NO_STATUS" not in junit


def test_gate_scopes_precheck_launch_environment(tmp_path):
    gate = os.path.join(
        os.path.dirname(_PRECHECK_DIR),
        "..",
        "..",
        "..",
        "jenkins",
        "scripts",
        "perf",
        "disaggregated",
        "slurm_ct_precheck_gate.sh",
    )
    gate = os.path.abspath(gate)
    shell_script = f"""
source {shlex.quote(gate)}
timeout() {{
    case "$DISAGG_SERVING_TYPE:$pytestCommand" in
        "GEN_PRECHECK_0:gen-command --server-idx 0"|\
        "CTX_PRECHECK_0:ctx-command --server-idx 0") return 0 ;;
        *) return 99 ;;
    esac
}}
sleep() {{ :; }}
cleanup_on_failure() {{ return 98; }}
ctPrecheckEnabled=1
ctPrecheckTimeout=1200
testOutputDir={shlex.quote(str(tmp_path / "output"))}
jobWorkspace={shlex.quote(str(tmp_path / "workspace"))}
mkdir -p "$jobWorkspace"
numGenServers=1
numCtxServers=1
nodesPerGenServer=1
nodesPerCtxServer=1
gpusPerNodePerGenServer=1
gpusPerNodePerCtxServer=1
genNodeLists=(gen-node)
ctxNodeLists=(ctx-node)
srunArgs=()
pytestCommandGENPrecheck=gen-command
pytestCommandCTXPrecheck=ctx-command
precheckRunScript=/unused
DISAGG_SERVING_TYPE=REAL_PERF_PARENT
pytestCommand=real-perf-command
run_cache_transceiver_precheck
printf '%s\n%s\n' "$DISAGG_SERVING_TYPE" "$pytestCommand"
"""
    result = subprocess.run(
        ["bash"], input=shell_script, capture_output=True, check=True, text=True
    )
    assert result.stdout.splitlines()[-2:] == ["REAL_PERF_PARENT", "real-perf-command"]


def test_rid_tags_dense_within_session():
    """Rids must be dense within a (ctx, gen) session.

    The C++ notification tag is rid & 0xFFF, so dense rids keep tags from
    aliasing across reps/lengths.
    """
    plan = pcfg.resolve_plan(_disagg_yaml())  # n_pairs=16
    total_reps = plan["warmup_requests"] + plan["num_requests"]
    n_pairs = plan["n_pairs"]

    def session_rids(ctx_idx, gen_idx):
        out = []
        for li in range(2):
            for rep in range(total_reps):
                for pair in range(n_pairs):
                    seq = (li * total_reps + rep) * n_pairs + pair
                    out.append(rp.make_rid(ctx_idx, gen_idx, 2, seq))
        return out

    a = session_rids(0, 0)
    b = session_rids(1, 0)
    assert len(set(a)) == len(a) and len(set(b)) == len(b)
    assert not (set(a) & set(b))  # globally unique across sessions
    tags = [r & 0xFFF for r in a]
    assert len(set(tags)) == len(tags)  # no tag aliasing within a session


class TestMultiPeerOrchestration:
    """CPU-only end-to-end runs of the multi-peer session protocol.

    Exercises the exact multi-instance logic of the hardware "B" topology:
    real ZMQ sockets + HMAC frames + StatusRecorder + rendezvous files via
    the real PrecheckRunner/_serve_gen_peers/_drive_ctx_peers, with only the
    GPU transfer methods stubbed out.
    """

    class _FakeComm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def bcast(self, obj, root=0):
            return obj

        def gather(self, obj, root=0):
            return [obj]

        def allgather(self, obj):
            return [obj]

    class _FakeParams:
        first_gen_tokens = [0]
        req_id = 1
        opaque_state = b"op"
        draft_tokens = None
        ctx_dp_rank = 0
        disagg_info_endpoint = None

    def _mk_runner(
        self,
        role,
        server_idx,
        plan,
        work_dir,
        monkeypatch,
        fail_ctx=False,
        wave_delay_s=0,
    ):
        import types

        # PrecheckRunner.__init__ imports mpi4py only to ensure MPI init.
        monkeypatch.setitem(sys.modules, "mpi4py", types.SimpleNamespace(MPI=None))
        # The gen side converts wire params through tensorrt_llm bindings;
        # identity is fine here (params_to_wire is covered separately).
        monkeypatch.setattr(rp, "params_from_wire", lambda d: d)

        args = types.SimpleNamespace(server_idx=server_idx, work_dir=work_dir)
        side = pcfg.side_plan(plan, role)
        runner = rp.PrecheckRunner(args, plan, side, self._FakeComm())

        calls = {"waves": 0}

        def ctx_run_wave(peer_idx, li, req_len, rep, wave):
            if fail_ctx:
                raise rp._TransferError("injected ctx failure")
            if wave_delay_s:
                time.sleep(wave_delay_s)
            calls["waves"] += 1
            return {p: self._FakeParams() for p in wave}, {}

        runner.ctx_run_wave = ctx_run_wave
        runner.ctx_finish_wave = lambda reqs: None
        runner.gen_run_wave = lambda peer_idx, li, req_len, rep, wave, params: (True, "")
        runner._calls = calls
        return runner

    def _run(
        self,
        tmp_path,
        monkeypatch,
        fail_peer_idx=None,
        fail_ctx_idx=None,
        num_ctx_servers=2,
        first_ctx_wave_delay_s=0,
        peer_progress_timeout_s=None,
    ):
        import threading

        monkeypatch.setenv("SLURM_JOB_ID", "777")
        # Publish loopback in the addr files: the real node hostname may not
        # resolve in sandboxed/CI environments, and everything is one process.
        monkeypatch.setenv("SLURMD_NODENAME", "127.0.0.1")
        cfg = _disagg_yaml(
            hardware={
                "gpus_per_node": 4,
                "num_ctx_servers": num_ctx_servers,
                "num_gen_servers": 1,
            },
            cache_transceiver_precheck={
                "request_lengths": [32],
                "num_requests": 1,
                "warmup_requests": 1,
                "rendezvous_timeout_s": 30,
                "wave_timeout_s": 30,
                "wireup_timeout_s": 0,
            },
        )
        plan = pcfg.resolve_plan(cfg)
        if peer_progress_timeout_s is not None:
            plan["peer_progress_timeout_s"] = peer_progress_timeout_s
            monkeypatch.setattr(rp, "CONTROL_POLL_INTERVAL_MS", 50)
        work = str(tmp_path)
        noop = lambda *a, **k: None  # noqa: E731 - signal.alarm needs main thread

        gen = self._mk_runner("gen", 0, plan, work, monkeypatch)
        ctxs = [
            self._mk_runner(
                "ctx",
                i,
                plan,
                work,
                monkeypatch,
                fail_ctx=(fail_ctx_idx is not None and i == fail_ctx_idx),
                wave_delay_s=first_ctx_wave_delay_s if i == 0 else 0,
            )
            for i in range(num_ctx_servers)
        ]

        if fail_peer_idx is not None:
            real_gen_run_peer = rp.gen_run_peer

            def fail_before_transfer(runner, peer_idx, arm, disarm):
                if peer_idx != fail_peer_idx:
                    return real_gen_run_peer(runner, peer_idx, arm, disarm)
                sock, key = rp._gen_open_session(runner, peer_idx, arm)
                reason = "injected pre-transfer failure"
                # Publish fail-fast before the peer handles our abort, so both
                # sides deterministically classify this as the same safe,
                # pre-dispatch failure rather than an ownership-fatal abort.
                rp.raise_abort_flag(runner.work_dir, f"ctx_{peer_idx} TRANSFER_ERROR: {reason}")
                try:
                    runner._leader_send_recv(sock, ("abort", reason), key)
                finally:
                    sock.close(linger=0)
                raise rp._TransferError(reason)

            monkeypatch.setattr(rp, "gen_run_peer", fail_before_transfer)

        failures = []

        def rec(peer, exc):
            failures.append((peer, type(exc).__name__))

        threads = [
            threading.Thread(
                target=rp._serve_gen_peers, args=(c, plan, noop, noop, rec), daemon=True
            )
            for c in ctxs
        ]
        for t in threads:
            t.start()

        def gen_arm(what, publish_progress=True, **kwargs):
            if publish_progress:
                rp.publish_peer_progress(gen, what)

        try:
            rp._drive_ctx_peers(
                gen, gen_arm, noop, rp._make_peer_failure_recorder(gen, noop, {"what": "test"})
            )
        finally:
            # Always join every peer, even when the driver raises. Asserting
            # inside the loop can itself strand later peers and trip CI's
            # pytest-threadleak hook.
            for thread in threads:
                thread.join(timeout=60)
            leaked = [thread.name for thread in threads if thread.is_alive()]
            assert not leaked, f"ctx serve threads wedged: {leaked}"
        return plan, gen, ctxs, failures

    def test_two_ctx_full_pass(self, tmp_path, monkeypatch):
        plan, gen, ctxs, failures = self._run(tmp_path, monkeypatch)
        assert not failures
        # gen recorded a PASS per (peer, req_len)
        assert {(c["peer"], c["status"]) for c in gen.recorder.cases} == {
            ("ctx_0", "PASS"),
            ("ctx_1", "PASS"),
        }
        # every ctx served the full schedule (reps x waves) and got its
        # deferred done (PASS recorded only after done/bye completes)
        total_waves = len(pcfg.waves(plan)) * (plan["warmup_requests"] + plan["num_requests"])
        for c in ctxs:
            assert c._calls["waves"] == total_waves
            assert [x["status"] for x in c.recorder.cases] == ["PASS"]

    def test_four_ctx_full_pass(self, tmp_path, monkeypatch):
        # ctx_0's four delayed waves take longer than the one-second queued
        # wait budget. gen phase progress must refresh ctx_1..ctx_3 rather than
        # letting their hello watchdog expire cumulatively behind ctx_0.
        plan, gen, ctxs, failures = self._run(
            tmp_path,
            monkeypatch,
            num_ctx_servers=4,
            first_ctx_wave_delay_s=0.3,
            peer_progress_timeout_s=1,
        )
        assert not failures
        assert [c["peer"] for c in gen.recorder.cases] == [
            "ctx_0",
            "ctx_1",
            "ctx_2",
            "ctx_3",
        ]
        assert all(c["status"] == "PASS" for c in gen.recorder.cases)
        assert all([case["status"] for case in ctx.recorder.cases] == ["PASS"] for ctx in ctxs)

    def test_one_ctx_four_gen_full_pass(self, tmp_path, monkeypatch):
        """Queued gen instances refresh from the active ctx's progress."""
        import threading

        monkeypatch.setenv("SLURM_JOB_ID", "778")
        monkeypatch.setenv("SLURMD_NODENAME", "127.0.0.1")
        monkeypatch.setattr(rp, "CONTROL_POLL_INTERVAL_MS", 50)
        cfg = _disagg_yaml(
            hardware={
                "gpus_per_node": 4,
                "num_ctx_servers": 1,
                "num_gen_servers": 4,
            },
            cache_transceiver_precheck={
                "request_lengths": [32],
                "num_requests": 1,
                "warmup_requests": 1,
                "rendezvous_timeout_s": 30,
                "wave_timeout_s": 30,
                "wireup_timeout_s": 0,
            },
        )
        plan = pcfg.resolve_plan(cfg)
        # One ctx session takes four 0.3s waves, so later gen instances wait
        # longer than this budget and need ctx progress to avoid false timeout.
        plan["peer_progress_timeout_s"] = 1
        work = str(tmp_path)
        noop = lambda *args, **kwargs: None  # noqa: E731 - signal.alarm needs main thread

        ctx = self._mk_runner(
            "ctx",
            0,
            plan,
            work,
            monkeypatch,
            wave_delay_s=0.3,
        )
        gens = [self._mk_runner("gen", i, plan, work, monkeypatch) for i in range(4)]
        peer_failures = []
        thread_errors = []

        def record_ctx_peer_failure(peer, exc):
            peer_failures.append((peer, type(exc).__name__))

        def progress_arm(runner):
            def arm(what, publish_progress=True, **kwargs):
                if publish_progress:
                    rp.publish_peer_progress(runner, what)

            return arm

        def run_ctx():
            try:
                rp._serve_gen_peers(
                    ctx,
                    plan,
                    progress_arm(ctx),
                    noop,
                    record_ctx_peer_failure,
                )
            except Exception as exc:  # noqa: BLE001 - surface thread failures in the test
                thread_errors.append(("ctx", exc))

        def run_gen(gen):
            try:
                rp._drive_ctx_peers(
                    gen,
                    progress_arm(gen),
                    noop,
                    rp._make_peer_failure_recorder(gen, noop, {"what": "test"}),
                )
            except Exception as exc:  # noqa: BLE001 - surface thread failures in the test
                thread_errors.append((f"gen_{gen.server_idx}", exc))

        threads = [threading.Thread(target=run_ctx, name="ctx_0", daemon=True)]
        threads.extend(
            threading.Thread(
                target=run_gen,
                args=(gen,),
                name=f"gen_{gen.server_idx}",
                daemon=True,
            )
            for gen in gens
        )
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        leaked = [thread.name for thread in threads if thread.is_alive()]
        assert not leaked, f"orchestration threads wedged: {leaked}"
        assert not thread_errors
        assert not peer_failures
        assert [case["status"] for case in ctx.recorder.cases] == ["PASS"] * 4
        assert all([case["status"] for case in gen.recorder.cases] == ["PASS"] for gen in gens)

    def test_ctx_failure_last_peer(self, tmp_path, monkeypatch):
        # The failing pair is driven LAST: the earlier healthy peer already
        # completed, so there is nothing left to fail-fast/skip.
        plan, gen, ctxs, failures = self._run(tmp_path, monkeypatch, fail_peer_idx=1)
        # gen side: healthy peer unaffected, failing peer gets a clear verdict
        by_peer = {c["peer"]: c["status"] for c in gen.recorder.cases}
        assert by_peer == {"ctx_0": "PASS", "ctx_1": "TRANSFER_ERROR"}
        assert not failures
        # ctx_0 served its full schedule and got the deferred done
        assert [c["status"] for c in ctxs[0].recorder.cases] == ["PASS"]
        assert [c["status"] for c in ctxs[1].recorder.cases] == ["SKIP"]

    def test_fail_fast_skips_remaining(self, tmp_path, monkeypatch):
        # The FIRST-driven pair fails: the remaining pair must be skipped
        # (not tested against a fabric already known bad), and told to abort
        # so it tears down promptly instead of waiting out its handshake alarm.
        plan, gen, ctxs, failures = self._run(tmp_path, monkeypatch, fail_peer_idx=0)
        by_peer = {c["peer"]: c["status"] for c in gen.recorder.cases}
        assert by_peer == {"ctx_0": "TRANSFER_ERROR", "ctx_1": "SKIP"}
        assert not failures
        # ctx_1 never ran a single transfer wave: fail-fast reached it first.
        assert ctxs[1]._calls["waves"] == 0
        # ctx_1 recorded a non-failing SKIP (its driver aborted the session).
        assert [c["status"] for c in ctxs[1].recorder.cases] == ["SKIP"]
        # the shared, job-stamped abort flag was dropped.
        assert rp.abort_flag_reason(str(tmp_path)) is not None
        # SKIP does not count toward the overall verdict; only ctx_0 failed.
        assert [c["peer"] for c in gen.recorder.failed_cases()] == ["ctx_0"]

    def test_abort_flag_stale_job_ignored(self, tmp_path, monkeypatch):
        # A flag left by a previous run (different SLURM_JOB_ID) in a reused
        # work dir must not fail-fast a fresh run -- same staleness rule as
        # addr files.
        monkeypatch.setenv("SLURM_JOB_ID", "111")
        rp.raise_abort_flag(str(tmp_path), "old run failure")
        assert rp.abort_flag_reason(str(tmp_path)) == "old run failure"
        monkeypatch.setenv("SLURM_JOB_ID", "222")
        assert rp.abort_flag_reason(str(tmp_path)) is None


def test_ctx_run_wave_missing_params_broadcast(tmp_path, monkeypatch):
    """#4 regression: the "missing context_phase_params" verdict must be broadcast.

    It is computed only on the instance leader (only it holds the gathered
    params) -- without the broadcast, a NON-leader rank keeps reason=None,
    returns, and enters the next collective while the leader raises,
    deadlocking the step until the watchdog SIGKILLs it (misreported as
    TIMEOUT).

    Run the real ctx_run_wave on a non-leader rank: with the leader's verdict
    delivered via bcast the rank must raise; with a clean (None) broadcast it
    must return normally.
    """
    import types

    monkeypatch.setitem(sys.modules, "mpi4py", types.SimpleNamespace(MPI=None))
    # ctx_run_wave imports tensorrt_llm only for logger.info, never reached with
    # no owned pairs; a stub keeps the test pure-CPU.
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm",
        types.SimpleNamespace(logger=types.SimpleNamespace(info=lambda *a, **k: None)),
    )

    class _NonLeaderComm:
        def __init__(self, bcast_ret):
            self._bcast_ret = bcast_ret

        def Get_rank(self):
            return 1  # non-leader (leader is rank 0)

        def Get_size(self):
            return 2

        def allgather(self, obj):
            return ["", ""]  # no local send error on any rank

        def gather(self, obj, root=0):
            return None  # only the leader receives the gathered params

        def bcast(self, obj, root=0):
            return self._bcast_ret  # the leader's verdict reaching this rank

    plan = pcfg.resolve_plan(
        _disagg_yaml(
            cache_transceiver_precheck={
                "request_lengths": [32],
                "num_requests": 1,
                "warmup_requests": 1,
                "wireup_timeout_s": 0,
            }
        )
    )
    side = pcfg.side_plan(plan, "ctx")
    args = types.SimpleNamespace(server_idx=0, work_dir=str(tmp_path))

    def _mk(bcast_ret):
        r = rp.PrecheckRunner(args, plan, side, _NonLeaderComm(bcast_ret))
        r.mapping = types.SimpleNamespace(pp_rank=1, tp_rank=1)  # unused non-leader
        r._owned = lambda wave: []  # skip the GPU send path; verdict arrives via bcast
        return r

    # leader broadcast a missing-params verdict -> the non-leader raises it too
    with pytest.raises(rp._TransferError, match="missing context_phase_params"):
        _mk("missing context_phase_params for pairs [0]").ctx_run_wave(
            peer_idx=0, li=0, req_len=32, rep=0, wave=[0]
        )

    # leader broadcast None (all good) -> the non-leader returns cleanly
    params, reqs = _mk(None).ctx_run_wave(peer_idx=0, li=0, req_len=32, rep=0, wave=[0])
    assert params == {} and reqs == {}


def test_status_env_snapshot_excludes_nixl(tmp_path, monkeypatch):
    """NIXL_* is not captured.

    The only such variable seen in practice is NIXL_VERSION, a stale
    NGC-base-image marker that misstates the version of the actually-linked
    library.
    """
    monkeypatch.setenv("NIXL_VERSION", "1.0.0")
    monkeypatch.setenv("NIXL_PLUGIN_DIR", "/opt/x")
    monkeypatch.setenv("UCX_TLS", "rc,cuda_copy")
    rec = rp.StatusRecorder(str(tmp_path), "gen", 0, is_leader=True)
    assert not any(k.startswith("NIXL_") for k in rec.env)
    assert rec.env["UCX_TLS"] == "rc,cuda_copy"  # behavioral vars still captured


def test_sparse_attention_model_uses_simplified_mla_pool(tmp_path):
    """DeepSeek V4 / DSA can't be modeled as a single KV pool.

    The precheck is a NETWORK check, so it falls back to a simple MLA-flavored
    stand-in pool (real layer count, one latent head, is_mla=True) and still
    runs -- never skips.
    """
    d = tmp_path / "v4"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "num_hidden_layers": 43,
                "num_attention_heads": 64,
                "num_key_value_heads": 1,
                "head_dim": 512,
                "index_head_dim": 128,
                "index_n_heads": 64,
                "index_topk": 512,
                "sliding_window": 128,
                "vocab_size": 129280,
            }
        )
    )
    shape = pcfg.model_kv_shape(str(d))
    assert shape.get("simplified") and "sparse" in shape["simplified"]
    assert shape["is_mla"] is True and shape["num_kv_heads"] == 1
    assert shape["num_layers"] == 43  # real layer count preserved
    assert shape["head_dim"] == 512 and shape["vocab_size"] == 129280
    # a plain MLA model is modeled normally (no simplified marker)
    m = tmp_path / "mla"
    m.mkdir()
    (m / "config.json").write_text(
        json.dumps({"num_hidden_layers": 4, "kv_lora_rank": 512, "qk_rope_head_dim": 64})
    )
    assert not pcfg.model_kv_shape(str(m)).get("simplified")


def test_python_transceiver_bandwidth_csv(tmp_path):
    """Bandwidth from the Python transceiver's perf_logger CSVs.

    PerfLogManager names them "<instanceUuid>_<rank>.csv" (it gives
    TRTLLM_KVCACHE_TIME_OUTPUT_PATH top priority); the parser identifies them
    by header columns. Median over KVSendTask throughput_mbs (MiB/s -> GB/s),
    receiver rows ignored.
    """
    header = (
        "timestamp,task_type,unique_rid,peer_rank,transfer_size_bytes,"
        "avg_segment_size_bytes,transfer_entry_count,prepare_args_latency_ms,"
        "queue_latency_ms,transfer_latency_ms,task_latency_ms,throughput_mbs"
    )
    # two ctx ranks, each its own perf file; KVSendTask rows carry throughput
    (tmp_path / "cd93dae6-9d75-4b0e-8a89-2c9e2f0f1a2b_0.csv").write_text(
        header + "\n"
        "t,KVSendTask,1,0,1000,,,0,0,0,0,102400.00\n"
        "t,AuxSendTask,1,0,10,,,0,0,0,0,1.00\n"  # tiny metadata, ignored
    )
    (tmp_path / "cd93dae6-9d75-4b0e-8a89-2c9e2f0f1a2b_1.csv").write_text(
        header + "\nt,KVSendTask,2,1,1000,,,0,0,0,0,204800.00\n"
    )
    # a receiver file (no throughput) must not contribute
    (tmp_path / "5d7b1f80-aaaa-bbbb-cccc-ddddeeeeffff_8.csv").write_text(
        header + "\nt,KVRecvTask,3,0,,,,,,,5.0,\n"
    )
    bw = rp.parse_python_bandwidth_gbps(str(tmp_path))
    # median(102400, 204800) = 153600 MiB/s -> GB/s (*1024^2/1e9)
    assert abs(bw - 153600 * 1024 * 1024 / 1e9) < 1e-9
    # no perf files -> None
    assert rp.parse_python_bandwidth_gbps(str(tmp_path / "empty")) is None
