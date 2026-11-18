# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 SM107 ULL drafter's FP8 context cache on CPU: the drafter's declaration and FP8
store conversion, both the same as the context servers' (the family drafter of the tp1 target),
the draft layers' page geometry and transfer IDs in the target's manager, and the gate deployment,
which shares the draft KV by those declarations alone."""

import pathlib
from types import SimpleNamespace as NS

import pytest
import torch
import yaml

from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import Fp4MlaV2CacheLayoutPolicy
from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.dspark import drafter as family_drafter
from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.dspark import primitives as family_primitives
from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__helix16_moetp16ep1 import (  # noqa: E501
    modeling as dflash,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__helix16_moetp16ep1 import (  # noqa: E501
    modeling as drafter,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    Role,
)
from tensorrt_llm.bindings import DataType

pytestmark = pytest.mark.cpu_only

_REPO = pathlib.Path(dflash.__file__).resolve().parents[6]
_GATE = (
    _REPO / "tests/scripts/perf-sanity/disaggregated/"
    "vr200_kimi-k3-nvfp4-dspark_gsm8k_ctx12_dwdp12_gen1_helix16_moetp16ep1_bs4_dspark7.yaml"
)
TARGET_LAYERS, DRAFT_LAYERS, LATENT, PAGE = 93, 5, 512 + 64, 128


def _draft_spec(layer_id: int, dtype=DataType.FP8) -> KVCacheLayerSpec:
    return KVCacheLayerSpec(
        global_layer_id=layer_id,
        owner=KVCacheLayerOwner.DRAFT,
        dtype=dtype,
        num_kv_heads=1,
        head_dim=LATENT,
        buffer_roles=(Role.KEY,),
    )


def test_drafter_declares_an_fp8_context() -> None:
    # Read by the KV cache creator (draft_kv_dtype), which then builds the appended draft layers
    # in FP8; the worker reads it as context_kv_fp8.
    gen = drafter.K3MLADSparkForCausalLM
    assert gen.draft_kv_dtype is torch.float8_e4m3fn
    assert object.__new__(gen).context_kv_fp8 is True
    assert dflash.DRAFT_FP8_KV_SCALE == 1.0


def test_context_servers_declare_and_write_the_same_fp8_context() -> None:
    # The context servers' drafter (tp1) declares the same pool dtype, so both sides append FP8
    # draft layers of one geometry and the transfer handshake sees one draft dtype.
    ctx, gen = family_drafter.K3DSparkDrafter, drafter.K3MLADSparkForCausalLM
    assert ctx.draft_kv_dtype is gen.draft_kv_dtype
    assert family_primitives.DRAFT_FP8_KV_SCALE == dflash.DRAFT_FP8_KV_SCALE
    assert family_primitives.DRAFT_FP8_MAX == dflash.DRAFT_FP8_MAX
    x = torch.linspace(-1000.0, 1000.0, 4097, dtype=torch.float32).to(torch.bfloat16)
    assert torch.equal(
        dflash.to_draft_fp8(x).view(torch.uint8),
        family_primitives.to_draft_fp8(x).view(torch.uint8),
    )


def test_fp8_store_conversion_saturates() -> None:
    x = torch.tensor([0.0, 1.5, -2.25, 448.0, 1000.0, -1000.0, 1e-3], dtype=torch.bfloat16)

    y = dflash.to_draft_fp8(x)

    assert y.dtype == torch.float8_e4m3fn
    ref = x.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    assert torch.isfinite(ref.float()).all()
    assert torch.equal(y.view(torch.uint8), ref.view(torch.uint8))
    assert y.float()[4].item() == 448.0 and y.float()[5].item() == -448.0


def test_draft_page_geometry() -> None:
    fp8, bf16 = _draft_spec(TARGET_LAYERS), _draft_spec(TARGET_LAYERS, DataType.BF16)

    # One latent per token and layer: 576 B in FP8, half the BF16 bytes.
    assert fp8.bytes_per_token() == LATENT == bf16.bytes_per_token() // 2
    assert PAGE * fp8.bytes_per_token() == 73_728
    assert DRAFT_LAYERS * PAGE * fp8.bytes_per_token() == 368_640


def test_draft_layer_transfer_ids() -> None:
    # The drafter's layers follow the target's 93 in the manager; each maps to 2 * global id,
    # clear of the target's latent (2 * L) and HP-ring (2 * L + 1) IDs.
    draft_local = range(TARGET_LAYERS, TARGET_LAYERS + DRAFT_LAYERS)
    manager = NS(
        pp_layers=list(range(TARGET_LAYERS + DRAFT_LAYERS)),
        _appended_draft_layer_specs={i: _draft_spec(i) for i in draft_local},
        _cache_manager_layer_ids=[],
        _fp4_mla_compact_to_local=[],
        _fp8_mla_manager_layer_ids=[],
        _hp_manager_layer_ids=[],
    )

    ids = [
        Fp4MlaV2CacheLayoutPolicy._fp4_mla_disagg_global_layer_id(manager, i) for i in draft_local
    ]

    assert ids == [186, 188, 190, 192, 194]


def test_gate_shares_the_draft_kv_by_declaration() -> None:
    config = yaml.safe_load(_GATE.read_text())

    for role in ("ctx", "gen"):
        spec = config["worker_config"][role]["speculative_config"]
        assert spec["decoding_type"] == "DSpark"
        assert not {"context_kv_dtype", "context_kv_only"} & set(spec)
        assert config["worker_config"][role]["kv_cache_config"]["use_kv_cache_manager_v2"] is True
    for name in ("worker_env_var", "ctx_worker_env_var", "gen_worker_env_var"):
        env = config["environment"].get(name) or ""
        assert not any(v.startswith("TRTLLM_DISAGG_SHARED_DRAFT_KV=") for v in env.split())
