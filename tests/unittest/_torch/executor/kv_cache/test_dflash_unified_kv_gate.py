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
"""Selection of the unified target+DFlash KV cache layout."""

import json
import os
from types import SimpleNamespace

import click
import pytest
import torch
import yaml

from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.llmapi import DFlashDecodingConfig
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.quantization.mode import QuantMode

pytestmark = pytest.mark.cpu_only

_SHARED_DRAFT_KV_ENV = "TRTLLM_DISAGG_SHARED_DRAFT_KV"


@pytest.fixture(autouse=True)
def _shared_draft_kv_allowed(monkeypatch):
    # Tests of the layout rules assume a launcher that allows sharing.
    monkeypatch.setenv(_SHARED_DRAFT_KV_ENV, "1")


def _model_config(*, kv_heads, head_dim, kv_quant_algo=None):
    pretrained = SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=32,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        hidden_size=32 * head_dim,
        torch_dtype=torch.bfloat16,
    )
    quant_mode = QuantMode.from_quant_algo(kv_cache_quant_algo=kv_quant_algo)
    return SimpleNamespace(
        pretrained_config=pretrained,
        quant_config=SimpleNamespace(quant_mode=quant_mode),
    )


def _creator(*, is_disagg, target, draft, tp_size=1, draft_windows=None):
    creator = object.__new__(KvCacheCreator)
    creator._speculative_config = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(
            is_dflash=lambda: True, is_dspark=lambda: False, is_external_drafter=lambda: True
        ),
        attention_backend="TRTLLM",
        _allow_separate_draft_kv_cache=True,
        draft_len_schedule=None,
    )
    creator._draft_model_engine = None
    creator._draft_config = draft
    creator._mapping = SimpleNamespace(
        pp_size=1,
        cp_size=1,
        tp_size=tp_size,
        enable_attention_dp=False,
        pp_layers=lambda num_layers: list(range(num_layers)),
    )
    creator._max_seq_len = 4096
    creator._kv_cache_config = SimpleNamespace()
    creator._get_draft_max_attention_window = lambda max_seq_len, kv_cache_config: draft_windows
    creator._llm_args = SimpleNamespace(attn_backend="TRTLLM", kv_cache_compression_config=None)
    creator._model_engine = SimpleNamespace(
        model=SimpleNamespace(model_config=target), dtype=torch.bfloat16
    )
    creator._kv_cache_manager_cls = KVCacheManagerV2
    creator._kv_cache_config = SimpleNamespace(max_attention_window=None)
    creator._unification_max_seq_len = 4096
    creator._is_disagg = is_disagg
    creator._get_draft_kv_model_config = lambda: draft
    return creator


@pytest.mark.parametrize("is_disagg", [False, True])
def test_matching_draft_layout_unifies(is_disagg):
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=8, head_dim=128)

    assert _creator(is_disagg=is_disagg, target=target, draft=draft)._should_unify_dflash_kv_cache()


@pytest.mark.parametrize("is_disagg", [False, True])
def test_mismatched_draft_kv_heads_unify(is_disagg: bool) -> None:
    # Each transfer view carries its own KV-head count, so the draft's heads are resharded on their own.
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=4, head_dim=128)

    assert _creator(is_disagg=is_disagg, target=target, draft=draft)._should_unify_dflash_kv_cache()


def test_nvfp4_with_mismatched_draft_kv_heads_unifies_only_in_aggregated_serving() -> None:
    # NVFP4 block-scale views are not resharded, so an NVFP4 side still needs equal heads.
    target = _model_config(kv_heads=8, head_dim=128, kv_quant_algo=QuantAlgo.NVFP4)
    draft = _model_config(kv_heads=4, head_dim=128)

    assert _creator(is_disagg=False, target=target, draft=draft)._should_unify_dflash_kv_cache()
    assert not _creator(is_disagg=True, target=target, draft=draft)._should_unify_dflash_kv_cache()


@pytest.mark.parametrize("is_disagg", [False, True])
def test_mismatched_draft_head_dim_unifies(is_disagg: bool) -> None:
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=8, head_dim=64)

    assert _creator(is_disagg=is_disagg, target=target, draft=draft)._should_unify_dflash_kv_cache()


@pytest.mark.parametrize("is_disagg", [False, True])
def test_fp8_target_with_bf16_draft_unifies(is_disagg: bool) -> None:
    target = _model_config(kv_heads=8, head_dim=128, kv_quant_algo=QuantAlgo.FP8)
    draft = _model_config(kv_heads=8, head_dim=128)

    assert _creator(is_disagg=is_disagg, target=target, draft=draft)._should_unify_dflash_kv_cache()


def test_layout_compares_rank_local_heads():
    # 8 target heads and 2 draft heads differ per rank at every TP size; at
    # TP4 the draft is duplicated to 1 head per rank while the target has 2.
    # Per-view head counts let the two still transfer together.
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=2, head_dim=128)
    creator = _creator(is_disagg=True, target=target, draft=draft, tp_size=4)

    assert creator._dflash_kv_layout(target, torch.bfloat16)[0] == 2
    assert creator._dflash_kv_layout(draft, torch.bfloat16)[0] == 1
    assert creator._should_unify_dflash_kv_cache()


@pytest.mark.parametrize("is_disagg", [False, True])
@pytest.mark.parametrize("attention_dp", [False, True])
def test_attention_dp_keeps_every_head_on_each_rank(is_disagg, attention_dp):
    # At TP4 a 4-head target and a 1-head draft both hold one head per rank.
    # Under attention DP each rank keeps every head (4 and 1); the layouts differ
    # but still transfer together, since each view carries its own head count.
    target = _model_config(kv_heads=4, head_dim=128)
    draft = _model_config(kv_heads=1, head_dim=128)
    creator = _creator(is_disagg=is_disagg, target=target, draft=draft, tp_size=4)
    creator._mapping.enable_attention_dp = attention_dp

    heads = (
        creator._dflash_kv_layout(target, torch.bfloat16)[0],
        creator._dflash_kv_layout(draft, torch.bfloat16)[0],
    )
    assert heads == ((4, 1) if attention_dp else (1, 1))
    assert creator._should_unify_dflash_kv_cache()


@pytest.mark.parametrize("is_disagg", [False, True])
@pytest.mark.parametrize("per_layer_side", ["target", "draft"])
def test_per_layer_kv_heads_keep_separate_cache(is_disagg, per_layer_side):
    uniform = _model_config(kv_heads=8, head_dim=128)
    per_layer = _model_config(kv_heads=[8, 4], head_dim=128)
    target, draft = (per_layer, uniform) if per_layer_side == "target" else (uniform, per_layer)

    creator = _creator(is_disagg=is_disagg, target=target, draft=draft)

    assert not creator._should_unify_dflash_kv_cache()
    assert creator._build_unified_dflash_layer_specs(kv_cache_config=None) == ()


class _StopLauncher(Exception):
    pass


def _launch_from_disagg_config(monkeypatch, tmp_path, context_servers, generation_servers):
    """Run the real disaggregated_mpi_worker up to the rank-communicator split."""
    from tensorrt_llm.commands import serve

    config = tmp_path / "disagg.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "context_servers": {"num_instances": 1, **context_servers},
                "generation_servers": {"num_instances": 1, **generation_servers},
            }
        )
    )

    def stop(*args, **kwargs):
        raise _StopLauncher

    monkeypatch.setattr(
        serve._command_telemetry, "apply_disaggregated_telemetry_config", lambda *a, **k: None
    )
    monkeypatch.setattr(serve, "set_cuda_device", lambda: None)
    monkeypatch.setattr("tensorrt_llm.llmapi.disagg_utils.split_world_comm", stop)
    monkeypatch.delenv("TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT", raising=False)
    serve.disaggregated_mpi_worker.callback(
        config_file=str(config), log_level="info", telemetry=False
    )


def test_dflash_on_generation_worker_only_keeps_separate_cache(monkeypatch, tmp_path):
    # The launcher rejects the variable when only one section sets speculative_config.
    with pytest.raises(click.UsageError, match="both context_servers and generation_servers"):
        _launch_from_disagg_config(
            monkeypatch,
            tmp_path,
            context_servers={},
            generation_servers={"speculative_config": {"decoding_type": "DFlash"}},
        )
    monkeypatch.delenv(_SHARED_DRAFT_KV_ENV)
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=8, head_dim=128)

    for _role in ("context", "generation"):
        creator = _creator(is_disagg=True, target=target, draft=draft)
        assert not creator._should_unify_dflash_kv_cache()


def test_dflash_with_different_tp_shares_the_cache(monkeypatch, tmp_path):
    # Context and generation may differ in TP: heads are resharded per view.
    spec = {"decoding_type": "DFlash", "attention_backend": "TRTLLM"}
    with pytest.raises(_StopLauncher):
        _launch_from_disagg_config(
            monkeypatch,
            tmp_path,
            context_servers={"tensor_parallel_size": 4, "speculative_config": spec},
            generation_servers={"tensor_parallel_size": 8, "speculative_config": spec},
        )
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=4, head_dim=128)

    for tp_size in (4, 8):
        creator = _creator(is_disagg=True, target=target, draft=draft, tp_size=tp_size)
        assert creator._should_unify_dflash_kv_cache()


@pytest.mark.parametrize("is_disagg", [False, True])
def test_draft_layers_with_different_windows_keep_separate_cache(is_disagg):
    layout = _model_config(kv_heads=8, head_dim=128)

    creator = _creator(is_disagg=is_disagg, target=layout, draft=layout, draft_windows=[128, 4096])

    assert not creator._should_unify_dflash_kv_cache()
    assert creator._build_unified_dflash_layer_specs(kv_cache_config=None) == ()


@pytest.mark.parametrize("is_disagg", [False, True])
def test_draft_layers_with_one_window_still_unify(is_disagg):
    layout = _model_config(kv_heads=8, head_dim=128)

    creator = _creator(is_disagg=is_disagg, target=layout, draft=layout, draft_windows=[128, 128])

    assert creator._should_unify_dflash_kv_cache()


def test_shared_cache_requested_with_unsupported_layout_fails_at_startup():
    layout = _model_config(kv_heads=8, head_dim=128)
    creator = _creator(is_disagg=True, target=layout, draft=layout, draft_windows=[128, 4096])

    with pytest.raises(ValueError, match="different attention windows"):
        creator._check_shared_draft_kv_request()


@pytest.mark.parametrize(
    "context_servers,generation_servers,match",
    [
        (
            {"speculative_config": {"decoding_type": "DFlash", "attention_backend": "VANILLA"}},
            {"speculative_config": {"decoding_type": "DFlash", "attention_backend": "VANILLA"}},
            "attention_backend AUTO or TRTLLM",
        ),
        (
            {"speculative_config": {"decoding_type": "DFlash", "attention_backend": "FA4"}},
            {"speculative_config": {"decoding_type": "DFlash", "attention_backend": "TRTLLM"}},
            "attention_backend AUTO or TRTLLM",
        ),
    ],
    ids=["vanilla_dflash_backend", "fa4_dflash_backend"],
)
def test_launcher_rejects_shared_cache_for_ineligible_config(
    monkeypatch, tmp_path, context_servers, generation_servers, match
):
    spec = {"decoding_type": "DFlash", "attention_backend": "TRTLLM"}
    context_servers = {"speculative_config": spec, **context_servers}
    generation_servers = {"speculative_config": spec, **generation_servers}

    with pytest.raises(click.UsageError, match=match):
        _launch_from_disagg_config(monkeypatch, tmp_path, context_servers, generation_servers)


@pytest.mark.parametrize("attention_backend", [None, "AUTO", "TRTLLM"])
def test_launcher_shares_draft_kv_for_dflash_in_both_sections(
    monkeypatch, tmp_path, attention_backend
):
    """AUTO, the default, resolves to TRTLLM for windowed drafters, so it shares like TRTLLM."""
    spec = {"decoding_type": "DFlash"}
    if attention_backend is not None:
        spec["attention_backend"] = attention_backend
    monkeypatch.delenv(_SHARED_DRAFT_KV_ENV)

    with pytest.raises(_StopLauncher):
        _launch_from_disagg_config(
            monkeypatch, tmp_path, {"speculative_config": spec}, {"speculative_config": spec}
        )
    assert os.environ[_SHARED_DRAFT_KV_ENV] == "1"


@pytest.mark.parametrize(
    "context_attention_dp,generation_attention_dp", [(True, True), (True, False)]
)
def test_launcher_accepts_eligible_shared_cache_config(
    monkeypatch, tmp_path, context_attention_dp, generation_attention_dp
):
    spec = {"decoding_type": "DFlash", "attention_backend": "TRTLLM"}

    with pytest.raises(_StopLauncher):
        _launch_from_disagg_config(
            monkeypatch,
            tmp_path,
            {"speculative_config": spec, "enable_attention_dp": context_attention_dp},
            {"speculative_config": spec, "enable_attention_dp": generation_attention_dp},
        )


@pytest.mark.parametrize(
    ("shared", "windows", "refused"),
    [("0", [2048], True), ("0", None, False), ("1", [2048], False)],
)
def test_disagg_separate_windowed_draft_pool_is_refused(monkeypatch, shared, windows, refused):
    """Unshared, a transferred request's drafter would read blocks the pool retired behind the window."""
    monkeypatch.setenv(_SHARED_DRAFT_KV_ENV, shared)
    layout = _model_config(kv_heads=8, head_dim=128)
    creator = _creator(is_disagg=True, target=layout, draft=layout, draft_windows=windows)
    creator._speculative_config = DFlashDecodingConfig(max_draft_len=7, attention_backend="TRTLLM")
    creator._sparse_attention_config = None
    if refused:
        with pytest.raises(ValueError, match="TRTLLM_DISAGG_SHARED_DRAFT_KV=1 on every worker"):
            creator._check_shared_draft_kv_request()
    else:
        creator._check_shared_draft_kv_request()


@pytest.mark.parametrize("is_disagg", [False, True])
def test_rank_pins_auto_to_unify_like_trtllm(tmp_path, monkeypatch, is_disagg):
    """An AUTO the frontend could not probe is pinned on the rank, so it unifies like TRTLLM."""
    from tensorrt_llm._torch.pyexecutor.py_executor_creator import (
        _pin_dflash_auto_attention_backend,
    )
    from tensorrt_llm._torch.speculative import dflash

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dflash, "dflash_trtllm_gen_unavailability_reason", lambda: None)
    windowed = {"layer_types": ["sliding_attention"] * 2, "sliding_window": 16}
    (tmp_path / "config.json").write_text(json.dumps(windowed))
    layout = _model_config(kv_heads=8, head_dim=128)
    creator = _creator(is_disagg=is_disagg, target=layout, draft=layout, draft_windows=[16])
    creator._speculative_config = DFlashDecodingConfig(
        max_draft_len=7, speculative_model=str(tmp_path)
    )
    _pin_dflash_auto_attention_backend(creator._speculative_config)
    assert creator._speculative_config.attention_backend == "TRTLLM"
    assert creator._unified_drafter_attention_backend() == "TRTLLM"
    creator._check_shared_draft_kv_request()  # TRTLLM_DISAGG_SHARED_DRAFT_KV=1 (autouse)
    assert creator._should_unify_dflash_kv_cache()
