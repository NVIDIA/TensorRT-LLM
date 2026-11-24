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
"""The family KDA + DSA cache manager of the Kimi K3.1 Rubin targets: its constructor arguments,
the arguments a target binds, its overrides of the shared KDA hybrid cache manager, its draft
layers, and the family KDA metadata's snapshot plan.

The shared manager loads the Kimi K3 package's KDA modules to seed replay caches in one launch
and to probe for mid-chunk snapshot support. The K3.1 manager seeds with plain copies from
stacked views of its pools, layer by layer where the views do not stack, and probes this
family's KDA prefill instead. CPU tests: the managers carry only the attributes the overrides
read.
"""

import functools
import importlib
import importlib.util
import os
import re
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch
from _kda_ops_util import bare

pytest.importorskip("fla")  # the shared kimi_kda package imports fla at load time

import tensorrt_llm
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm import (
    rubin_kda_dsa_cache_manager as dsa_cache_manager,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm import rubin_kda_metadata
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_dsa_cache_manager import (
    KV_CACHE_ARGS_KEY,
    RubinKdaDsaHybridCacheManager,
    RubinKdaReplayState,
    bind_kv_cache_args,
    select_rubin_kv_cache_manager_cls,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_metadata import (
    KDA_PREFILL_CHUNK_SIZE,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.params import DSAParams
from tensorrt_llm._torch.modules.kimi_kda import cache_manager as shared_cache_manager

# CPU test (l0_cpu); a GPU host other than sm_107 skips it.
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability(0) != (10, 7),
        reason="Kimi K3.1 Rubin: sm_107 or CPU only",
    ),
]

# The package the shared manager loads its K3 KDA modules from.
_K3_PACKAGE = shared_cache_manager._K3_KDA_PACKAGE.rsplit(".", 1)[0]
_NUM_SPEC = 2
_SECTION = 8
_WINDOW = 3  # committed convolution window (kernel size 4)
_SLOTS = 4
_LAYERS = 2
_CONV_ROLES = ("kda_conv_q", "kda_conv_k", "kda_conv_v")
_DRAFT_ROLES = ("kda_k_cache", "kda_g_cache", "kda_v_cache", "kda_beta_cache")


#: A manager without V2 pools, carrying the given attributes.
_manager = functools.partial(bare, RubinKdaDsaHybridCacheManager)


def _generation_manager() -> RubinKdaDsaHybridCacheManager:
    """What ``_initialize_spec_state`` reads on a generation server with two draft tokens."""
    spec_config = SimpleNamespace(tokens_per_gen_step=_NUM_SPEC + 1)
    return _manager(
        _requested_num_spec=_NUM_SPEC,
        spec_config=spec_config,
        kda_context_only=False,
        _mamba_ssm_stochastic_rounding=False,
        _requested_draft_records=False,
        _state_layout=SimpleNamespace(
            spec_config=spec_config,
            mamba_pp_layers=list(range(_LAYERS)),
            conv_state_layout="q_k_v",
            conv_section_dims=(_SECTION,) * 3,
        ),
    )


def _bind(state: RubinKdaReplayState) -> None:
    """Bind per-layer conv states and replay buffers as the V2 pool binding does."""
    generator = torch.Generator().manual_seed(0)
    state.prev_num_accepted_tokens = torch.ones(_SLOTS, dtype=torch.int32)
    state._conv_section_dims = (_SECTION,) * 3
    state._committed_window = _WINDOW
    state._conv_states = tuple(
        torch.randn(_SLOTS, 3 * _SECTION, _WINDOW, generator=generator).to(torch.bfloat16)
        for _ in range(_LAYERS)
    )
    state.pool_layers = [
        {
            **{
                name: torch.full((_SLOTS, _SECTION, _WINDOW + _NUM_SPEC), 7.0)
                for name in _CONV_ROLES
            },
            **{name: torch.full((_SLOTS, _NUM_SPEC, 2, 4), 7.0) for name in _DRAFT_ROLES},
        }
        for _ in range(_LAYERS)
    ]


@pytest.fixture
def k3_imports(monkeypatch):
    """Modules of the K3 package that the code under test asks importlib for."""
    requested = []
    import_module = importlib.import_module

    def spy(name, package=None):
        resolved = importlib.util.resolve_name(name, package) if name.startswith(".") else name
        if resolved == _K3_PACKAGE or resolved.startswith(_K3_PACKAGE + "."):
            requested.append(resolved)
        return import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", spy)
    return requested


#: What the DEP16 targets bind (zeroed pools, FP8 latent KV and index keys, an FP32 KDA state).
_TARGET_ARGS = dict(
    zero_gpu_pools=True,
    kv_dtype="fp8",
    indexer_k_dtype="fp8",
    state_dtype=torch.float32,
    fp8_kv_layers=None,
)


@pytest.mark.parametrize("composite", [False, True], ids=["text", "composite"])
def test_engine_passes_what_the_target_bound(composite) -> None:
    """A target binds its arguments to its model config when it is constructed;
    ``_create_kv_cache_manager`` then builds this manager with them, beside the DSA config and the
    text config (nested in a composite Kimi K3.1 config or flattened)."""
    text_config = SimpleNamespace(model_type="kimi_linear")
    sparse_attention_config = SimpleNamespace(algorithm="dsa")
    model_config = SimpleNamespace(
        pretrained_config=(
            SimpleNamespace(model_type="kimi_k31", text_config=text_config)
            if composite
            else text_config
        ),
        sparse_attention_config=sparse_attention_config,
        extra_attrs={"kv_cache_dtype": "fp8"},
    )
    bind_kv_cache_args(model_config, **_TARGET_ARGS)
    assert model_config.extra_attrs["kv_cache_dtype"] == "fp8"
    assert model_config.extra_attrs[KV_CACHE_ARGS_KEY] == _TARGET_ARGS
    assert RubinKdaDsaHybridCacheManager.extra_init_kwargs(model_config) == {
        "sparse_attention_config": sparse_attention_config,
        "pretrained_config": text_config,
        **_TARGET_ARGS,
        "draft_window": True,
    }
    # model_kwargs: {text_config: {kimi_k31_draft_window: false}} keeps the drafter's whole history.
    text_config.kimi_k31_draft_window = False
    assert RubinKdaDsaHybridCacheManager.extra_init_kwargs(model_config)["draft_window"] is False
    text_config.kimi_k31_draft_window = "off"
    with pytest.raises(ValueError, match="kimi_k31_draft_window must be true or false"):
        RubinKdaDsaHybridCacheManager.extra_init_kwargs(model_config)
    del text_config.kimi_k31_draft_window
    # A copy: the engine's kwargs do not alias the bound record.
    kwargs = RubinKdaDsaHybridCacheManager.extra_init_kwargs(model_config)
    kwargs["kv_dtype"] = "nvfp4"
    assert model_config.extra_attrs[KV_CACHE_ARGS_KEY]["kv_dtype"] == "fp8"


@pytest.mark.parametrize(
    "change, error, match",
    [
        ({"kv_dtype": "fp16"}, ValueError, "kv_dtype must be one of"),
        ({"kv_dtype": None}, ValueError, "kv_dtype must be one of"),
        ({"indexer_k_dtype": "bf16"}, ValueError, "indexer_k_dtype must be 'fp8' or 'fp4'"),
        ({"state_dtype": "float32"}, TypeError, "state_dtype must be a torch dtype"),
    ],
)
def test_a_target_binds_only_names_the_manager_knows(change, error, match) -> None:
    model_config = SimpleNamespace(extra_attrs={})
    with pytest.raises(error, match=match):
        bind_kv_cache_args(model_config, **{**_TARGET_ARGS, **change})
    assert KV_CACHE_ARGS_KEY not in model_config.extra_attrs


def _dsa_sparse_config(indexer_k_dtype: str = "fp8") -> SimpleNamespace:
    """A DSA config whose sparse parameters are real ``DSAParams``."""
    params = DSAParams(
        index_n_heads=32, index_head_dim=128, index_topk=2048, indexer_k_dtype=indexer_k_dtype
    )
    return SimpleNamespace(algorithm="dsa", to_sparse_params=lambda **kwargs: params)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            dict(kv_dtype="fp8", dtype="NVFP4"),
            r"stores fp8 latent KV; the engine asks for .*NVFP4",
        ),
        (dict(kv_dtype="nvfp4", dtype="FP8"), r"stores nvfp4 latent KV; the engine asks for .*FP8"),
        (
            dict(indexer_k_dtype="fp8", sparse_indexer_k_dtype="fp4"),
            "stores fp8 index keys; the DSA config asks for fp4",
        ),
        (
            dict(indexer_k_dtype="fp4", sparse_indexer_k_dtype="fp8"),
            "stores fp4 index keys; the DSA config asks for fp8",
        ),
    ],
)
def test_a_target_cache_refuses_another_storage_before_allocating(monkeypatch, kwargs, match):
    """The KV cache and index-K dtypes are the engine's (``kv_cache_config.dtype`` and the DSA
    config); a manager whose target was written for others fails before it allocates a pool."""

    def allocate(self, *args, **kwargs):
        raise AssertionError("the manager allocated its pools")

    monkeypatch.setattr(shared_cache_manager.KimiK3HybridCacheManager, "__init__", allocate)
    kwargs = dict(kwargs)
    sparse_config = _dsa_sparse_config(kwargs.pop("sparse_indexer_k_dtype", "fp8"))
    dtype = getattr(dsa_cache_manager.DataType, kwargs.pop("dtype", "FP8"))
    with pytest.raises(ValueError, match=match):
        RubinKdaDsaHybridCacheManager(
            sparse_attention_config=sparse_config,
            pretrained_config=None,
            layer_mask=[False, True],
            dtype=dtype,
            **kwargs,
        )


@pytest.mark.parametrize(
    "state_dtype, engine_state_dtype, refused",
    [
        (torch.float32, torch.bfloat16, True),
        (torch.bfloat16, torch.float32, True),
        (torch.float32, torch.float32, False),
        (None, torch.bfloat16, False),
    ],
)
def test_a_target_cache_refuses_another_kda_state(
    monkeypatch, state_dtype, engine_state_dtype, refused
):
    """The KDA state dtype is the engine's (``kv_cache_config.mamba_ssm_cache_dtype``), resolved by
    the base construction; the manager compares it right after. Past it, this stub base has no
    local layer, which the next check refuses."""

    def base_init(self, *args, **kwargs):
        self.ssm_state_dtype = engine_state_dtype
        self.num_local_layers = 0

    monkeypatch.setattr(shared_cache_manager.KimiK3HybridCacheManager, "__init__", base_init)
    expected = (
        pytest.raises(ValueError, match=r"keeps a torch\.(float32|bfloat16) KDA state")
        if refused
        else pytest.raises(ValueError, match="DSA requires at least one local MLA layer")
    )
    with expected:
        RubinKdaDsaHybridCacheManager(
            sparse_attention_config=_dsa_sparse_config(),
            pretrained_config=None,
            layer_mask=[False, True],
            dtype=dsa_cache_manager.DataType.FP8,
            state_dtype=state_dtype,
        )


@pytest.mark.parametrize("fp8_layers", [(), (3,)], ids=["nvfp4", "fp8_layer"])
@pytest.mark.parametrize("num_draft_layers", [0, 3], ids=["no_draft_cache", "draft_cache"])
def test_attention_bytes_per_token_include_the_indexer_k(
    monkeypatch, num_draft_layers, fp8_layers
) -> None:
    """The attention cache's bytes per token are every local layer's ``Role.ALL`` bytes, which
    hold the indexer K of the MLA layers, with or without a unified standalone draft cache. With
    one, the base sums the target layers' data roles and the draft layers' bytes: no indexer K.
    An FP8 KV layer counts its latent once, at one byte per value and without block scales."""
    roles = dsa_cache_manager.Role
    mla_layers, num_target_layers = (1, 3), 4  # layers 0 and 2 are KDA layers
    key_bytes, indexer_bytes, draft_bytes = 128, 132, 128
    fp8_key_bytes = 576  # one 576-wide latent per token
    num_layers = num_target_layers + num_draft_layers

    def base_layer_bytes(self, local_layer_idx, data_role):
        if local_layer_idx >= num_target_layers:
            return draft_bytes
        return key_bytes if local_layer_idx in mla_layers else 0

    def base_attention_bytes(self):
        target_bytes = sum(
            self.get_layer_bytes_per_token(local_layer_idx, roles.KEY)
            for local_layer_idx in range(num_target_layers)
        )
        return target_bytes + sum(self._draft_layer_sizes)

    base = shared_cache_manager.KimiK3HybridCacheManager
    for name, method in (
        ("get_layer_bytes_per_token", base_layer_bytes),
        ("_attention_cache_bytes_per_token", base_attention_bytes),
    ):
        monkeypatch.setattr(base, name, method, raising=False)
    manager = _manager(
        pp_layers=list(range(num_layers)),
        num_local_layers=num_layers,
        _indexer_layer_ids=frozenset(mla_layers),
        _indexer_bytes_per_token=lambda: indexer_bytes,
        _draft_layer_sizes=[draft_bytes] * num_draft_layers,
        _fp8_mla_global_layer_ids=frozenset(fp8_layers),
        cp_as_tp_layer_ids=frozenset(),
        num_kv_heads_per_layer=[1] * num_layers,
        head_dim_per_layer=[fp8_key_bytes] * num_layers,
    )
    all_bytes = sum(manager.get_layer_bytes_per_token(i, roles.ALL) for i in range(num_layers))
    mla_bytes = sum(
        (fp8_key_bytes if layer in fp8_layers else key_bytes) + indexer_bytes
        for layer in mla_layers
    )
    assert all_bytes == mla_bytes + num_draft_layers * draft_bytes
    assert manager._attention_cache_bytes_per_token() == all_bytes


def _k31_text_config():
    """The production K3.1 text config's KV cache shape: 93 layers, DSA (MLA) layers zero-based
    3, 7, ..., 87, 91 and 92, a 512 + 64 latent."""
    full_attn_layers = [*range(4, 89, 4), 92, 93]
    return SimpleNamespace(
        model_type="kimi_linear",
        architectures=["KimiLinearForCausalLM"],
        num_hidden_layers=93,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        linear_attn_config={"full_attn_layers": full_attn_layers},
        dsa_config={"dsa_layers": full_attn_layers},
    )


@pytest.mark.parametrize(
    "fp8_kv_layers, num_fp8_layers",
    [(None, 2), ((91, 92), 2), ((), 0), ((87,), 1)],
    ids=["default", "explicit", "none", "one"],
)
@pytest.mark.parametrize("composite", [False, True], ids=["text_config", "composite_config"])
def test_fp8_kv_layers_static_estimate(
    monkeypatch, fp8_kv_layers, num_fp8_layers, composite
) -> None:
    """The base estimate sizes every MLA layer as a latent-only NVFP4 page (a DSA config does not
    take the FP4 MLA formula); each FP8 KV layer adds 576 - 288 - 36 = 252 B per token on top of
    the indexer K, read from the text config of a composite config too. The selection is what the
    target bound."""
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    base_slope, index_bytes = 10_000, 24 * 68
    monkeypatch.setattr(
        shared_cache_manager.KimiK3HybridCacheManager,
        "get_cache_size_per_token",
        classmethod(lambda cls, *args, **kwargs: (base_slope, 0)),
    )
    monkeypatch.setattr(
        dsa_cache_manager, "get_indexer_k_cache_size_per_token", lambda *args: index_bytes
    )
    text_config = _k31_text_config()
    pretrained_config = (
        SimpleNamespace(model_type="kimi_k31", text_config=text_config)
        if composite
        else text_config
    )
    model_config = SimpleNamespace(
        pretrained_config=pretrained_config,
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4),
        extra_attrs={},
    )
    bind_kv_cache_args(
        model_config,
        zero_gpu_pools=False,
        kv_dtype="nvfp4",
        indexer_k_dtype="fp8",
        state_dtype=torch.float32,
        fp8_kv_layers=fp8_kv_layers,
    )
    slope, fixed = RubinKdaDsaHybridCacheManager.get_cache_size_per_token(
        model_config,
        Mapping(world_size=1, tp_size=1, pp_size=1, rank=0),
        max_batch_size=8,
        kv_cache_config=KvCacheConfig(max_tokens=1024, enable_block_reuse=False),
        tokens_per_block=64,
        max_seq_len=4096,
    )
    assert (slope, fixed) == (base_slope + index_bytes + num_fp8_layers * 252, 0)


@pytest.mark.parametrize(
    "kv_cache_quant_algo, bytes_per_token",
    [("NVFP4", 24 * (288 + 36 + 68) + 2 * 252), ("FP8", 24 * (576 + 68))],
    ids=["nvfp4", "fp8"],
)
def test_the_static_estimate_follows_the_kv_cache_dtype(
    monkeypatch, kv_cache_quant_algo, bytes_per_token
) -> None:
    """The 24 DSA layers of the production geometry, through the generic attention estimate that
    the hybrid one takes for them: the latent at the KV cache's dtype (half a byte and an E4M3
    scale per 16 values with NVFP4, a byte with FP8), the index keys, and the extra bytes of the
    FP8 KV layers with the NVFP4 cache only. 9,912 and 15,456 B per token; with the production
    drafter's five FP8 latents (2,880 B) a server logs 12,792 and 18,336."""
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    def attention_estimate(cls, model_config, mapping, **kwargs):
        return KVCacheManager.get_cache_size_per_token(model_config, mapping, num_layers=24), 0

    monkeypatch.setattr(
        shared_cache_manager.KimiK3HybridCacheManager,
        "get_cache_size_per_token",
        classmethod(attention_estimate),
    )
    monkeypatch.setattr(
        dsa_cache_manager, "get_indexer_k_cache_size_per_token", lambda *args: 24 * 68
    )
    text_config = _k31_text_config()
    text_config.num_attention_heads = 96
    model_config = SimpleNamespace(
        pretrained_config=text_config,
        quant_config=QuantConfig(kv_cache_quant_algo=getattr(QuantAlgo, kv_cache_quant_algo)),
        extra_attrs={},
    )
    bind_kv_cache_args(
        model_config,
        zero_gpu_pools=False,
        kv_dtype=kv_cache_quant_algo.lower(),
        indexer_k_dtype="fp8",
        state_dtype=torch.float32,
        fp8_kv_layers=None,
    )
    slope, fixed = RubinKdaDsaHybridCacheManager.get_cache_size_per_token(
        model_config,
        Mapping(world_size=1, tp_size=1, pp_size=1, rank=0),
        max_batch_size=8,
        kv_cache_config=KvCacheConfig(max_tokens=1024, enable_block_reuse=False),
        tokens_per_block=64,
        max_seq_len=4096,
    )
    assert (slope, fixed) == (bytes_per_token, 0)


def test_generation_server_builds_the_k31_replay_state() -> None:
    manager = _generation_manager()
    state = manager._initialize_spec_state()
    assert type(state) is RubinKdaReplayState
    assert manager._kda_replay is state
    assert state.num_speculative_tokens == _NUM_SPEC


def test_context_server_and_target_without_drafts_keep_the_shared_states() -> None:
    context = _generation_manager()
    context.kda_context_only = True
    assert context._initialize_spec_state() is None
    no_drafts = _generation_manager()
    no_drafts._requested_num_spec = None
    no_drafts.spec_config = None
    state = no_drafts._initialize_spec_state()
    assert isinstance(state, shared_cache_manager.KDAIntermediateState)


def test_unstackable_cpu_views_are_seeded_layer_by_layer(monkeypatch, k3_imports) -> None:
    """The replay state plans its own stacked seeding, never the shared batched one. CPU views do
    not stack, so it plans nothing, and the per-layer copies write the committed conv window and
    clear the rest of the replay caches of the seeded slots only."""

    def batched_seed_plan(self):
        raise AssertionError("the K3.1 replay state planned the shared batched seeding")

    monkeypatch.setattr(
        shared_cache_manager.KDAReplayState, "_plan_batched_seed", batched_seed_plan
    )
    state = _generation_manager()._initialize_spec_state()
    _bind(state)
    slots = torch.tensor([1, 3])
    others = torch.tensor([0, 2])

    state.seed_transferred_slots(slots)

    assert state._get_batched_seed_plan() is None
    assert state.prev_num_accepted_tokens.tolist() == [1, 0, 1, 0]
    for layer, conv_state in enumerate(state._conv_states):
        replay = state._layer_cache_fields(layer)
        for section, name in enumerate(_CONV_ROLES):
            buffer = replay[name]
            committed = conv_state[:, section * _SECTION : (section + 1) * _SECTION].float()
            assert torch.equal(buffer[slots, :, :_WINDOW], committed[slots])
            assert not buffer[slots, :, _WINDOW:].any()
            assert torch.all(buffer[others] == 7.0)
        for name in _DRAFT_ROLES:
            assert not replay[name][slots].any()
            assert torch.all(replay[name][others] == 7.0)
    assert k3_imports == []


@pytest.mark.parametrize(
    ("flag", "block_reuse", "is_draft", "snapshot_pages", "prefill", "expected"),
    [
        (True, True, False, True, True, KDA_PREFILL_CHUNK_SIZE),
        (False, True, False, True, True, None),
        (True, False, False, True, True, None),
        (True, True, True, True, True, None),
        (True, True, False, False, True, None),
        (True, True, False, True, False, None),
    ],
    ids=["on", "flag_off", "no_block_reuse", "draft", "no_snapshot_pages", "no_prefill"],
)
def test_midchunk_snapshots_follow_this_family_prefill(
    monkeypatch, k3_imports, flag, block_reuse, is_draft, snapshot_pages, prefill, expected
) -> None:
    """The shared gate (``TRTLLM_KDA_MIDCHUNK_SNAPSHOT=1``, block reuse, not a draft manager,
    pending SSM snapshot pages), then this family's prefill decides; the step alignment is the
    family KDA metadata's chunk size."""
    monkeypatch.setattr(shared_cache_manager, "_KDA_MIDCHUNK_SNAPSHOT", flag)
    kv_cache_attributes = {"acquire_ssm_snapshot_slot": lambda self: None} if snapshot_pages else {}
    monkeypatch.setattr(dsa_cache_manager, "_KVCache", type("KVCache", (), kv_cache_attributes))
    monkeypatch.setattr(dsa_cache_manager, "_kda_prefill_stores_snapshots", lambda: prefill)
    manager = _manager(enable_block_reuse=block_reuse, is_draft=is_draft)
    assert manager._get_kda_midchunk_snapshot_alignment() == expected
    assert k3_imports == []


# Runs in a fresh interpreter, where TRTLLM_KDA_MIDCHUNK_SNAPSHOT=1 (the K3.1 context servers'
# setting) reaches the shared manager at import, as in a server process.
_FRESH_PROCESS_CHECK = textwrap.dedent(
    """
    import sys
    from types import SimpleNamespace

    import torch

    try:
        from cutlass.cutlass_dsl.cutlass import CuTeDSL
    except ImportError:
        CuTeDSL = None
    pipeline = None if CuTeDSL is None else CuTeDSL._get_pipeline

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm import (
        rubin_kda_dsa_cache_manager as k31,
    )
    from tensorrt_llm._torch.modules.kimi_kda import cache_manager as shared

    assert shared._KDA_MIDCHUNK_SNAPSHOT

    # Context server: block reuse, a V2 backend with pending SSM snapshot pages.
    k31._KVCache = type("KVCache", (), {"acquire_ssm_snapshot_slot": lambda self: None})
    context = object.__new__(k31.RubinKdaDsaHybridCacheManager)
    context.enable_block_reuse = True
    context.is_draft = False
    alignment = context._get_kda_midchunk_snapshot_alignment()
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_metadata import (
        KDA_PREFILL_CHUNK_SIZE,
    )

    stores_snapshots = k31._kda_prefill_stores_snapshots()
    assert alignment == (KDA_PREFILL_CHUNK_SIZE if stores_snapshots else None), alignment

    # Generation server: the replay state seeds transferred slots, on the GPU when there is one.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec_config = SimpleNamespace(tokens_per_gen_step=3)
    generation = object.__new__(k31.RubinKdaDsaHybridCacheManager)
    generation._requested_num_spec = 2
    generation.spec_config = spec_config
    generation.kda_context_only = False
    generation._mamba_ssm_stochastic_rounding = False
    generation._requested_draft_records = False
    generation._state_layout = SimpleNamespace(
        spec_config=spec_config,
        mamba_pp_layers=[0, 1],
        conv_state_layout="q_k_v",
        conv_section_dims=(8, 8, 8),
    )
    state = generation._initialize_spec_state()
    assert type(state) is k31.RubinKdaReplayState and generation._kda_replay is state
    state.prev_num_accepted_tokens = torch.ones(4, dtype=torch.int32, device=device)
    state._conv_section_dims = (8, 8, 8)
    state._committed_window = 3
    state._conv_states = tuple(torch.randn(4, 24, 3, device=device) for _ in range(2))
    state.pool_layers = [
        {name: torch.zeros(4, 8, 5, device=device) for name in ("kda_conv_q", "kda_conv_k", "kda_conv_v")}
        for _ in range(2)
    ]
    state.seed_transferred_slots(torch.tensor([1, 3], device=device))

    k3 = shared._K3_KDA_PACKAGE.rsplit(".", 1)[0]
    loaded = sorted(name for name in sys.modules if name == k3 or name.startswith(k3 + "."))
    assert not loaded, f"K3 package modules loaded: {loaded}"
    assert CuTeDSL is None or CuTeDSL._get_pipeline is pipeline, "CuTeDSL._get_pipeline patched"
    print("k31 manager check passed")
    """
)


def test_a_k31_server_process_never_loads_the_k3_package() -> None:
    """Both server roles' manager paths (snapshot probe, replay state, seeding) leave the K3
    package unloaded and the CuTe DSL compile pipeline unpatched."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(tensorrt_llm.__file__)))
    python_path = os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")]))
    child_env = dict(os.environ, TRTLLM_KDA_MIDCHUNK_SNAPSHOT="1", PYTHONPATH=python_path)
    result = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_CHECK],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "k31 manager check passed" in result.stdout


_TOKENS_PER_BLOCK = 32


def _cache_dtype(nvfp4: bool):
    data_type = dsa_cache_manager.DataType
    return data_type.NVFP4 if nvfp4 else data_type.FP8


@pytest.mark.parametrize("nvfp4", [True, False], ids=["nvfp4", "fp8"])
def test_nvfp4_snapshot_points_sit_on_cache_blocks(nvfp4) -> None:
    """The shared snapshot planning, with snapshots 40 tokens into the prompt and 0 and 70 tokens
    before its end: with an NVFP4 cache every point is rounded down to a cache block and the one
    inside the first block is dropped; with another cache dtype the points stay."""
    manager = _manager(
        dtype=_cache_dtype(nvfp4),
        tokens_per_block=_TOKENS_PER_BLOCK,
        enable_block_reuse=True,
        kv_cache_config=SimpleNamespace(
            mamba_state_config=SimpleNamespace(
                periodic_snapshot_interval=None,
                additional_snapshot_offsets_from_start=[40],
                additional_snapshot_offsets_from_end=[0, 70],
            ),
            block_reuse_config=SimpleNamespace(enable_branch_snapshot=False),
        ),
        _kda_midchunk_snapshot_alignment=None,
    )
    request = SimpleNamespace(prompt_len=100, context_current_position=0, is_dummy_request=False)

    manager.prepare_expect_snapshot_points([request])

    assert request.expect_snapshot_points == ([32, 96] if nvfp4 else [30, 40, 100])


class _HistoryCache:
    """The V2 cache calls of a context update; as in V2, the history never moves back."""

    def __init__(self, history_length: int, is_active: bool = True) -> None:
        self.history_length = history_length
        self.is_active = is_active
        self.resized_to = []

    def resize(self, capacity, history_length) -> bool:
        assert capacity is None
        if history_length < self.history_length:
            raise ValueError("History length cannot be decreased")
        self.resized_to.append(history_length)
        self.history_length = history_length
        return True


@pytest.mark.parametrize("nvfp4", [True, False], ids=["nvfp4", "fp8"])
def test_nvfp4_history_follows_every_context_chunk(monkeypatch, nvfp4) -> None:
    """With an NVFP4 cache the context update first commits a pending mid-chunk snapshot, then
    moves every active cache's history to its context position before the shared update runs;
    with another cache dtype the shared update runs alone."""
    events = []

    def commit_midchunk(self, request, snapshot):
        events.append(("midchunk", request.py_request_id))
        self.kv_cache_map[request.py_request_id].resize(None, snapshot.point)

    def shared_update(self, scheduled_batch):
        histories = {rid: cache.history_length for rid, cache in self.kv_cache_map.items()}
        events.append(("shared", histories, sorted(self._kda_midchunk_snapshots)))

    base = shared_cache_manager.KimiK3HybridCacheManager
    monkeypatch.setattr(base, "_commit_kda_midchunk_snapshot", commit_midchunk)
    monkeypatch.setattr(base, "update_context_resources", shared_update)
    caches = {1: _HistoryCache(128), 2: _HistoryCache(0), 3: _HistoryCache(64, is_active=False)}
    manager = _manager(
        dtype=_cache_dtype(nvfp4),
        kv_cache_map=caches,
        _kda_midchunk_snapshots={1: SimpleNamespace(point=160)},
    )
    # Request 4 holds no cache.
    requests = [
        SimpleNamespace(py_request_id=rid, context_current_position=192) for rid in range(1, 5)
    ]

    manager.update_context_resources(SimpleNamespace(context_requests=requests))

    if nvfp4:
        assert events == [("midchunk", 1), ("shared", {1: 192, 2: 192, 3: 64}, [])]
        assert caches[1].resized_to == [160, 192]
    else:
        assert events == [("shared", {1: 128, 2: 0, 3: 64}, [1])]
        assert not any(cache.resized_to for cache in caches.values())


#: GPU pool groups of a DEP16 context server with the FP8 KV cache, as the KV cache manager V2
#: reports them: the drafter, the DSA layers (latent KV and index keys of 24 layers in 64-token
#: pages) and the KDA state.
_FP8_POOL_STATS = (
    SimpleNamespace(total=55887, slot_sizes=[184320]),
    SimpleNamespace(total=55940, slot_sizes=[884736, 104448]),
    SimpleNamespace(total=49, slot_sizes=[27131904, 1907712]),
)
#: Each pool group's segment of the line.
_POOL_SEGMENT = r"pool_group_id=(\d+), num_slots=(\d+), slot_size=\[([\d, ]*)\], bytes=(\d+)"


@pytest.mark.parametrize(
    "dtype, estimating, logged",
    [("FP8", False, True), ("FP8", True, False), ("NVFP4", False, False), ("BF16", False, False)],
    ids=["fp8", "fp8_estimate", "nvfp4", "bf16"],
)
def test_an_fp8_latent_cache_logs_its_pools(monkeypatch, dtype, estimating, logged) -> None:
    """The FP8 latent cache that serves logs the GPU slots of every pool group on one line, a
    segment per group in pool-group order; the cache built for the size estimate and the caches of
    other dtypes log nothing."""
    lines = []
    monkeypatch.setattr(dsa_cache_manager.logger, "info", lines.append)
    manager = _manager(
        dtype=getattr(dsa_cache_manager.DataType, dtype),
        is_estimating_kv_cache=estimating,
        impl=SimpleNamespace(get_storage_statistics=lambda level: list(_FP8_POOL_STATS)),
    )
    manager._log_fp8_latent_cache_pools()
    if not logged:
        assert lines == []
        return
    assert lines == [
        "Kimi K3.1 KV cache pools (FP8 latent KV): "
        "pool_group_id=0, num_slots=55887, slot_size=[184320], bytes=10301091840; "
        "pool_group_id=1, num_slots=55940, slot_size=[884736, 104448], bytes=55334952960; "
        "pool_group_id=2, num_slots=49, slot_size=[27131904, 1907712], bytes=1422941184"
    ]
    assert [int(group) for group, *_ in re.findall(_POOL_SEGMENT, lines[0])] == [0, 1, 2]


def test_pools_a_cache_cannot_report_are_a_warning(monkeypatch) -> None:
    """The pool line is a log line only: a cache whose pools cannot be read still starts."""

    def unavailable(level):
        raise RuntimeError("no storage statistics")

    infos, warnings = [], []
    monkeypatch.setattr(dsa_cache_manager.logger, "info", infos.append)
    monkeypatch.setattr(dsa_cache_manager.logger, "warning", warnings.append)
    manager = _manager(
        dtype=dsa_cache_manager.DataType.FP8,
        is_estimating_kv_cache=False,
        impl=SimpleNamespace(get_storage_statistics=unavailable),
    )
    manager._log_fp8_latent_cache_pools()
    assert infos == []
    assert warnings == ["Kimi K3.1 KV cache pools unavailable: no storage statistics"]


@pytest.mark.parametrize(
    "state_dtype, heads, num_kda_layers, estimating, line",
    [
        (
            torch.float32,
            96,
            69,
            False,
            "69 local KDA layers of [96, 128, 128], 434110464 B of state per slot",
        ),
        (
            torch.float32,
            12,
            69,
            False,
            "69 local KDA layers of [12, 128, 128], 54263808 B of state per slot",
        ),
        (torch.float32, 96, 69, True, None),
        (torch.float32, 96, 0, False, None),
        (torch.bfloat16, 96, 69, False, None),
    ],
    ids=["dep16", "tep8_helix8", "fp32_estimate", "no_local_kda_layers", "bf16"],
)
def test_an_fp32_kda_state_is_logged(
    monkeypatch, state_dtype, heads, num_kda_layers, estimating, line
) -> None:
    """``kv_cache_config.mamba_ssm_cache_dtype: float32``: the cache that serves logs its KDA state,
    the 69 local KDA layers of a DEP16 rank (96 heads) or of a TEP8 or helix8 rank (12 heads) and the
    bytes of their FP32 states in a slot. The cache built for the size estimate, a rank without
    KDA layers and a BF16 state log nothing."""
    lines = []
    monkeypatch.setattr(dsa_cache_manager.logger, "info", lines.append)
    manager = _manager(
        ssm_state_dtype=state_dtype,
        is_estimating_kv_cache=estimating,
        local_num_mamba_layers=num_kda_layers,
        ssm_state_shape=[heads, 128, 128] if num_kda_layers else [],
        ssm_bytes=heads * 128 * 128 * 4 if num_kda_layers else 0,
    )
    manager._log_fp32_kda_state()
    assert lines == ([] if line is None else [f"Kimi K3.1 KDA recurrent state: float32, {line}"])


def _draft_manager(fp8_layers=frozenset({3})) -> RubinKdaDsaHybridCacheManager:
    """Four model layers (MLA 1 and 3) and a unified drafter's two layers after them.

    The draft layers' global ids follow the rank's last layer; on an earlier pipeline stage they
    can repeat a model layer's id, which the draft layer 5 (global id 3) stands for here.
    """
    return _manager(
        pp_layers=[0, 1, 2, 3, 4, 3],
        cp_as_tp_layer_ids=frozenset({4, 5}),
        _indexer_layer_ids=frozenset({1, 3}),
        _indexer_bytes_per_token=lambda: 132,
        _fp8_mla_global_layer_ids=frozenset(fp8_layers),
        num_kv_heads_per_layer=[1] * 6,
        head_dim_per_layer=[576] * 6,
    )


def test_draft_layers_never_take_the_fp8_latent(monkeypatch) -> None:
    """An FP8 KV layer is a model MLA layer: a draft layer keeps the drafter's own dtype, roles and
    bytes, even where its global id repeats a selected layer's."""
    roles = dsa_cache_manager.Role
    base = shared_cache_manager.KimiK3HybridCacheManager
    monkeypatch.setattr(base, "_get_layer_dtype_by_local_id", lambda self, local: "draft_dtype")
    monkeypatch.setattr(base, "_get_buffer_roles_for_layer", lambda self, local: ["draft_roles"])
    monkeypatch.setattr(base, "get_layer_bytes_per_token", lambda self, local, role: 1152)
    manager = _draft_manager()
    assert manager._local_fp8_latent_layers() == [3]
    assert [manager._is_fp8_latent_layer(local) for local in range(6)] == [False] * 3 + [
        True,
        False,
        False,
    ]
    assert manager._get_layer_dtype_by_local_id(3) == dsa_cache_manager.DataType.FP8
    assert manager._get_buffer_roles_for_layer(3) == [roles.KEY]
    for draft_local in (4, 5):
        assert manager._get_layer_dtype_by_local_id(draft_local) == "draft_dtype"
        assert manager._get_buffer_roles_for_layer(draft_local) == ["draft_roles"]
    # The FP8 layer counts its 576-wide latent once; the draft layers count the base's bytes.
    assert manager.get_layer_bytes_per_token(3, roles.KEY) == 576
    assert manager.get_layer_bytes_per_token(4, roles.KEY) == 1152


def test_draft_layers_own_no_index_keys(monkeypatch) -> None:
    """Index keys belong to the model's MLA layers; a draft layer that follows the last layer owns
    none, so it has no INDEX_KEY buffer and no index bytes."""
    roles = dsa_cache_manager.Role
    monkeypatch.setattr(
        shared_cache_manager.KimiK3HybridCacheManager,
        "get_layer_bytes_per_token",
        lambda self, local, role: 0 if role == roles.INDEX_KEY else 1152,
    )
    # The V2 buffer descriptor, as a record (a compiled class on the GPU image).
    monkeypatch.setattr(
        dsa_cache_manager, "BufferConfig", lambda role, size: SimpleNamespace(role=role, size=size)
    )
    manager = _draft_manager(fp8_layers=())
    manager.pp_layers = [0, 1, 2, 3, 4, 5]
    manager.layer_offsets = {layer: local for local, layer in enumerate(manager.pp_layers)}
    assert manager.indexer_k_cache_local_layer_mask == [False, True, False, True, False, False]
    assert [manager.get_layer_bytes_per_token(local, roles.INDEX_KEY) for local in range(6)] == [
        0,
        132,
        0,
        132,
        0,
        0,
    ]
    buffers = manager._extra_buffers_per_layer(tokens_per_block=64)
    assert sorted(buffers) == [1, 3]
    assert [(buffer.role, buffer.size) for local in (1, 3) for buffer in buffers[local]] == [
        (roles.INDEX_KEY, 132 * 64)
    ] * 2


def _selector_model_config(algorithm="dsa", hybrid=True, num_mamba_layers=69, bound=True):
    model_config = SimpleNamespace(
        sparse_attention_config=SimpleNamespace(algorithm=algorithm),
        pretrained_config=SimpleNamespace(hybrid=hybrid),
        get_num_mamba_layers=lambda: num_mamba_layers,
        extra_attrs={},
    )
    if bound:
        bind_kv_cache_args(model_config, **_TARGET_ARGS)
    return model_config


@pytest.fixture
def selector(monkeypatch):
    """``select_rubin_kv_cache_manager_cls`` with the engine helpers it imports stubbed (the
    executor modules import the model zoo): hybrid configs carry ``hybrid``; the transceiver route
    and the TRTLLM_USE_PY_MAMBA switch are set per test."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import common

    state = SimpleNamespace(route=("NIXL", "PYTHON"), py_mamba=False)
    package = "tensorrt_llm._torch.pyexecutor"
    monkeypatch.setitem(
        sys.modules,
        f"{package}._util",
        SimpleNamespace(_resolve_disagg_transceiver_route=lambda config: state.route),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{package}.config_utils",
        SimpleNamespace(is_hybrid_linear=lambda config: config.hybrid),
    )
    monkeypatch.setattr(common, "use_py_mamba_cache_manager", lambda: state.py_mamba)
    return state


def _kv_cache_config(v2=True, reuse=False, rebalance=False):
    return SimpleNamespace(
        use_kv_cache_manager_v2=v2, enable_block_reuse=reuse, enable_kv_pool_rebalance=rebalance
    )


def test_the_selector_returns_the_family_manager(selector) -> None:
    assert (
        select_rubin_kv_cache_manager_cls(_selector_model_config(), _kv_cache_config())
        is RubinKdaDsaHybridCacheManager
    )
    assert (
        select_rubin_kv_cache_manager_cls(
            _selector_model_config(), _kv_cache_config(reuse=True), is_disagg=True
        )
        is RubinKdaDsaHybridCacheManager
    )


@pytest.mark.parametrize(
    "model_config",
    [
        _selector_model_config(algorithm=None, bound=False),
        _selector_model_config(hybrid=False, bound=False),
        _selector_model_config(num_mamba_layers=0, bound=False),
    ],
    ids=["no_dsa", "not_hybrid", "no_kda_layers"],
)
def test_the_selector_leaves_other_models_to_the_engine(selector, model_config) -> None:
    """Other models bind nothing (a drafter's config, for one)."""
    assert select_rubin_kv_cache_manager_cls(model_config, _kv_cache_config()) is None


@pytest.mark.parametrize("extra_attrs", [{}, None, {"kv_cache_dtype": "fp8"}])
def test_a_config_without_bound_arguments_is_refused(monkeypatch, selector, extra_attrs) -> None:
    """A target that forgets ``bind_kv_cache_args`` would get no pool zero and no dtype checks: the
    selector, the static estimate (before the base estimate runs) and the engine's constructor
    arguments refuse its config."""

    def estimate(cls, *args, **kwargs):
        raise AssertionError("the base estimate ran")

    monkeypatch.setattr(
        shared_cache_manager.KimiK3HybridCacheManager,
        "get_cache_size_per_token",
        classmethod(estimate),
    )
    model_config = _selector_model_config(bound=False)
    model_config.extra_attrs = extra_attrs
    with pytest.raises(ValueError, match="bind_kv_cache_args"):
        select_rubin_kv_cache_manager_cls(model_config, _kv_cache_config())
    with pytest.raises(ValueError, match="bind_kv_cache_args"):
        RubinKdaDsaHybridCacheManager.extra_init_kwargs(model_config)
    with pytest.raises(ValueError, match="bind_kv_cache_args"):
        RubinKdaDsaHybridCacheManager.get_cache_size_per_token(
            model_config, None, max_batch_size=8, kv_cache_config=None
        )


@pytest.mark.parametrize(
    "kv_cache_config, is_disagg, route, py_mamba, match",
    [
        (_kv_cache_config(v2=None), False, None, False, "requires use_kv_cache_manager_v2=True"),
        (_kv_cache_config(), True, ("UCX", "PYTHON"), False, "backend='NIXL'"),
        (_kv_cache_config(), True, ("NIXL", "CPP"), False, "transceiver_runtime='PYTHON'"),
        (
            _kv_cache_config(reuse=True, rebalance=True),
            False,
            None,
            False,
            "not compatible with enable_kv_pool_rebalance",
        ),
        (_kv_cache_config(), False, None, True, "unset TRTLLM_USE_PY_MAMBA"),
    ],
    ids=["no_v2", "disagg_ucx", "disagg_cpp", "reuse_rebalance", "py_mamba"],
)
def test_the_selector_refuses_what_the_manager_cannot_run(
    selector, kv_cache_config, is_disagg, route, py_mamba, match
) -> None:
    selector.route = route or selector.route
    selector.py_mamba = py_mamba
    with pytest.raises(ValueError, match=match):
        select_rubin_kv_cache_manager_cls(
            _selector_model_config(), kv_cache_config, is_disagg=is_disagg
        )


# The family KDA metadata's mid-chunk snapshot plan


def test_the_snapshot_plan_addresses_each_sequence() -> None:
    """``(context index, slot, offset)`` snapshots of a three-sequence step: the plan names each
    snapshot's slot and first packed token after its point, and per sequence the chunk, the pool
    row and the staged row (sequences first, snapshots after them), -1 where none."""
    plan = rubin_kda_metadata.build_kda_snapshot_plan(
        [(2, 9, 128), (0, 4, 64)], [192, 100, 256], torch.device("cpu")
    )
    assert plan.num_snapshots == 2
    assert plan.slots.tolist() == [4, 9]
    assert plan.token_indices.tolist() == [64, 192 + 100 + 128]
    assert plan.chunks.tolist() == [1, -1, 2]
    assert plan.pool_rows.tolist() == [4, -1, 9]
    assert plan.staged_rows.tolist() == [3, -1, 4]
    assert rubin_kda_metadata.build_kda_snapshot_plan([], [64], torch.device("cpu")) is None


@pytest.mark.parametrize(
    "snapshots, match",
    [
        ([(0, 1, 32)], "must be a multiple of 64"),
        ([(0, 1, 128)], "inside the 128-token step"),
        ([(0, 1, 0)], "inside the 128-token step"),
        ([(1, 1, 64)], "Invalid KDA snapshot sequence index 1"),
        ([(0, 1, 64), (0, 2, 64)], "Invalid KDA snapshot sequence index 0"),
    ],
)
def test_the_snapshot_plan_takes_points_on_the_prefill_grid(snapshots, match) -> None:
    with pytest.raises(ValueError, match=match):
        rubin_kda_metadata.build_kda_snapshot_plan(snapshots, [128], torch.device("cpu"))


def test_the_metadata_plans_the_snapshots_the_manager_hands_out(monkeypatch) -> None:
    """After the engine's preparation, the step's context requests ask the cache manager for their
    snapshots; a manager without them, or a step without request ids, plans none."""
    monkeypatch.setattr(
        rubin_kda_metadata.KimiK3MambaMetadata, "prepare", lambda self, attn_metadata: None
    )
    metadata = object.__new__(rubin_kda_metadata.RubinKdaMetadata)
    metadata._kda_schedule = torch.zeros(2, 4, dtype=torch.int32)
    asked = []

    def get_snapshots(request_ids):
        asked.append(list(request_ids))
        return [(1, 5, 64)]

    attn_metadata = SimpleNamespace(
        seq_lens=torch.tensor([64, 192, 1, 1]),
        num_contexts=2,
        request_ids=[11, 12, 13, 14],
        kv_cache_manager=SimpleNamespace(get_kda_midchunk_snapshots=get_snapshots),
    )
    metadata.prepare(attn_metadata)
    assert asked == [[11, 12]]
    assert metadata.kda_snapshot.slots.tolist() == [5]
    assert metadata.kda_snapshot.token_indices.tolist() == [64 + 64]
    for manager, request_ids in (
        (SimpleNamespace(), [11, 12, 13, 14]),
        (attn_metadata.kv_cache_manager, None),
    ):
        attn_metadata.kv_cache_manager, attn_metadata.request_ids = manager, request_ids
        metadata.prepare(attn_metadata)
        assert metadata.kda_snapshot is None
