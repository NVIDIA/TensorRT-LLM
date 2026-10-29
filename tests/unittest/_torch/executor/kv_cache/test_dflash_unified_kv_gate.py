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

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from tensorrt_llm._torch.model_config import KVCacheLayerSpec
from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings import DataType
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


def _creator(*, is_disagg, target, draft, tp_size=1):
    creator = object.__new__(KvCacheCreator)
    creator._speculative_config = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(is_dflash=lambda: True, is_external_drafter=lambda: True),
        attention_backend="TRTLLM",
        _allow_separate_draft_kv_cache=True,
        draft_len_schedule=None,
    )
    creator._draft_config = draft
    creator._mapping = SimpleNamespace(
        pp_size=1,
        cp_size=1,
        tp_size=tp_size,
        enable_attention_dp=False,
        pp_layers=lambda num_layers: range(num_layers),
    )
    creator._llm_args = SimpleNamespace(attn_backend="TRTLLM", kv_cache_compression_config=None)
    creator._model_engine = SimpleNamespace(
        model=SimpleNamespace(model_config=target, config=target.pretrained_config)
    )
    creator._kv_cache_manager_cls = KVCacheManagerV2
    creator._kv_cache_config = SimpleNamespace(max_attention_window=None)
    creator._tokens_per_block = 32
    creator._unification_max_seq_len = 4096
    creator._max_seq_len = 4096
    creator._is_disagg = is_disagg
    creator._get_draft_kv_model_config = lambda: draft
    return creator


@pytest.mark.parametrize("is_disagg", [False, True])
def test_matching_draft_layout_unifies(is_disagg):
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=8, head_dim=128)

    creator = _creator(is_disagg=is_disagg, target=target, draft=draft)

    assert creator._should_unify_dflash_kv_cache()
    specs = creator._build_unified_dflash_layer_specs(creator._kv_cache_config)
    assert [spec.global_layer_id for spec in specs] == [0, 1, 2, 3]
    assert [spec.dtype for spec in specs] == [DataType.BF16] * 4


@pytest.mark.parametrize("is_disagg", [False, True])
@pytest.mark.parametrize(
    ("tokens_per_block", "draft_tokens_per_block"),
    [(8, 32), (16, 32), (32, 32), (64, 64), (128, 32)],
)
def test_unification_requires_supported_draft_page_size(
    is_disagg: bool, tokens_per_block: int, draft_tokens_per_block: int
) -> None:
    creator = _creator(
        is_disagg=is_disagg,
        target=_model_config(kv_heads=8, head_dim=128),
        draft=_model_config(kv_heads=8, head_dim=128),
    )
    creator._tokens_per_block = tokens_per_block

    assert creator._should_unify_dflash_kv_cache() is (tokens_per_block in (16, 32, 64))
    assert creator._get_draft_tokens_per_block() == draft_tokens_per_block


def test_mismatched_draft_kv_heads_unify_only_in_aggregated_serving() -> None:
    target = _model_config(kv_heads=8, head_dim=128)
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

    creator = _creator(is_disagg=is_disagg, target=target, draft=draft)

    assert creator._should_unify_dflash_kv_cache()
    specs = creator._build_unified_dflash_layer_specs(creator._kv_cache_config)
    assert [spec.dtype for spec in specs] == [DataType.FP8] * 2 + [DataType.BF16] * 2


@pytest.mark.parametrize("is_disagg", [False, True])
def test_variable_model_geometry_keeps_separate_cache(is_disagg: bool) -> None:
    target = _model_config(kv_heads=8, head_dim=128)
    target.kv_cache_layer_specs = (
        KVCacheLayerSpec(head_dim=128, num_kv_heads=8, num_q_heads=32),
        KVCacheLayerSpec(head_dim=256, num_kv_heads=4, num_q_heads=32),
    )
    draft = _model_config(kv_heads=8, head_dim=128)

    assert not _creator(
        is_disagg=is_disagg, target=target, draft=draft
    )._should_unify_dflash_kv_cache()


@pytest.mark.parametrize("is_disagg", [False, True])
def test_retained_token_mixed_precision_keeps_separate_cache(is_disagg: bool) -> None:
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=8, head_dim=128)
    creator = _creator(is_disagg=is_disagg, target=target, draft=draft)
    creator._kv_cache_config.mixed_precision_kv = object()

    assert not creator._should_unify_dflash_kv_cache()


def test_layout_compares_rank_local_heads():
    # 8 target heads and 2 draft heads differ per rank at every TP size; at
    # TP4 the draft is duplicated to 1 head per rank while the target has 2.
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=2, head_dim=128)
    creator = _creator(is_disagg=True, target=target, draft=draft, tp_size=4)

    assert creator._dflash_kv_layout(target, torch.bfloat16)[0] == 2
    assert creator._dflash_kv_layout(draft, torch.bfloat16)[0] == 1
    assert not creator._should_unify_dflash_kv_cache()


@pytest.mark.parametrize("is_disagg", [False, True])
def test_attention_dp_unifies_with_unsharded_heads(is_disagg):
    # Under attention DP each rank keeps every KV head, so the TP4 layouts of
    # an 8-head target and a 2-head draft are not resharded to 2 and 1.
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=8, head_dim=128)
    creator = _creator(is_disagg=is_disagg, target=target, draft=draft, tp_size=4)
    creator._mapping.enable_attention_dp = True

    assert creator._dflash_kv_layout(target, torch.bfloat16)[0] == 8
    assert (
        creator._dflash_kv_layout(_model_config(kv_heads=2, head_dim=128), torch.bfloat16)[0] == 2
    )
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
    with pytest.raises(_StopLauncher):
        serve.disaggregated_mpi_worker.callback(
            config_file=str(config), log_level="info", telemetry=False
        )


def test_dflash_on_generation_worker_only_keeps_separate_cache(monkeypatch, tmp_path):
    _launch_from_disagg_config(
        monkeypatch,
        tmp_path,
        context_servers={},
        generation_servers={
            "speculative_config": {"decoding_type": "DFlash", "attention_backend": "TRTLLM"}
        },
    )
    target = _model_config(kv_heads=8, head_dim=128)
    draft = _model_config(kv_heads=8, head_dim=128)

    for _role in ("context", "generation"):
        creator = _creator(is_disagg=True, target=target, draft=draft)
        assert not creator._should_unify_dflash_kv_cache()


def test_dflash_with_different_tp_keeps_separate_cache(monkeypatch, tmp_path):
    spec = {"decoding_type": "DFlash", "attention_backend": "TRTLLM"}
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
        assert not creator._should_unify_dflash_kv_cache()


@pytest.mark.parametrize(
    "context_backend,generation_backend,shared",
    [
        ("TRTLLM", "TRTLLM", True),
        ("VANILLA", "TRTLLM", False),
        ("TRTLLM", "VANILLA", False),
        ("FA4", "TRTLLM", False),
        ("TRTLLM", "FA4", False),
        (None, "TRTLLM", False),
        ("TRTLLM", None, False),
    ],
)
def test_launcher_requires_trtllm_draft_attention_on_both_peers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    context_backend: str | None,
    generation_backend: str | None,
    shared: bool,
) -> None:
    specs = []
    for backend in (context_backend, generation_backend):
        spec = {"decoding_type": "DFlash"}
        if backend is not None:
            spec["attention_backend"] = backend
        specs.append(spec)
    _launch_from_disagg_config(
        monkeypatch,
        tmp_path,
        context_servers={"speculative_config": specs[0]},
        generation_servers={"speculative_config": specs[1]},
    )

    assert os.environ[_SHARED_DRAFT_KV_ENV] == ("1" if shared else "0")
    for backend in (context_backend, generation_backend):
        creator = _creator(
            is_disagg=True,
            target=_model_config(kv_heads=8, head_dim=128),
            draft=_model_config(kv_heads=8, head_dim=128),
        )
        creator._speculative_config.attention_backend = backend or "VANILLA"
        assert creator._should_unify_dflash_kv_cache() is shared


@pytest.mark.parametrize("peer_index", [0, 1], ids=["context", "generation"])
@pytest.mark.parametrize(
    "server_overrides,spec_overrides,shared",
    [
        ({"attn_backend": "VANILLA"}, {}, False),
        ({}, {"draft_len_schedule": {8: 15}}, False),
        ({}, {"max_concurrency": 8}, False),
        ({"kv_cache_config": {"tokens_per_block": 128}}, {}, False),
        ({"kv_cache_config": {"tokens_per_block": 16}}, {}, True),
        ({"kv_cache_config": {"tokens_per_block": 64}}, {}, True),
        ({}, {"draft_len_schedule": None, "max_concurrency": None}, True),
    ],
    ids=["target-backend", "schedule", "concurrency", "page-128", "page-16", "page-64", "nulls"],
)
def test_launcher_checks_all_peer_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    peer_index: int,
    server_overrides: dict,
    spec_overrides: dict,
    shared: bool,
) -> None:
    servers = [
        {
            "speculative_config": {
                "decoding_type": "DFlash",
                "attention_backend": "TRTLLM",
                "max_draft_len": 15,
            }
        }
        for _ in range(2)
    ]
    servers[peer_index].update(server_overrides)
    servers[peer_index]["speculative_config"].update(spec_overrides)
    _launch_from_disagg_config(monkeypatch, tmp_path, servers[0], servers[1])

    assert os.environ[_SHARED_DRAFT_KV_ENV] == ("1" if shared else "0")
    # The unchanged, locally eligible peer must follow the common decision.
    creator = _creator(
        is_disagg=True,
        target=_model_config(kv_heads=8, head_dim=128),
        draft=_model_config(kv_heads=8, head_dim=128),
    )
    assert creator._should_unify_dflash_kv_cache() is shared
