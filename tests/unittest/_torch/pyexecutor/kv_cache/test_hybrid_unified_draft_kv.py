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
"""A standalone drafter's KV layers in a hybrid (recurrent + MLA) target manager."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import Fp4MlaV2CacheLayoutPolicy
from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import mamba_cache_manager_v2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
    _insert_draft_pool_ratio,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.quantization.mode import QuantMode
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    GpuCacheTierConfig,
    KVCacheManagerConfig,
    LayerId,
    SsmLayerConfig,
)

pytestmark = pytest.mark.cpu_only

_NUM_TARGET_LAYERS = 6
_NUM_DRAFT_LAYERS = 2


class _K3LikeManager(MambaHybridCacheManagerV2):
    accepts_appended_draft_layers = True


def _mla_config(*, num_layers, kv_quant_algo=None):
    pretrained = SimpleNamespace(
        num_hidden_layers=num_layers,
        num_attention_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        hidden_size=7168,
        torch_dtype=torch.bfloat16,
    )
    return SimpleNamespace(
        pretrained_config=pretrained,
        quant_config=SimpleNamespace(
            quant_mode=QuantMode.from_quant_algo(kv_cache_quant_algo=kv_quant_algo)
        ),
    )


def _creator(*, manager_cls=_K3LikeManager, reads_pages=True, tp_size=16, is_disagg=False):
    """A Kimi K3-like creator: NVFP4 MLA target, FP8 MLA drafter."""
    target = _mla_config(num_layers=_NUM_TARGET_LAYERS, kv_quant_algo=QuantAlgo.NVFP4)
    draft = _mla_config(num_layers=_NUM_DRAFT_LAYERS, kv_quant_algo=QuantAlgo.FP8)
    creator = object.__new__(KvCacheCreator)
    creator._speculative_config = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(
            is_dflash=lambda: False, is_dspark=lambda: True, is_external_drafter=lambda: True
        ),
        attention_backend="CUTEDSL",
        draft_is_embedded_in_target=False,
        _allow_separate_draft_kv_cache=True,
        draft_len_schedule=None,
    )
    creator._draft_config = draft
    creator._mapping = SimpleNamespace(
        pp_size=1,
        cp_size=1,
        tp_size=tp_size,
        enable_attention_dp=True,
        pp_layers=lambda num_layers: list(range(num_layers)),
    )
    creator._tokens_per_block = 128
    creator._max_seq_len = 4096
    creator._unification_max_seq_len = 4096
    creator._kv_cache_config = SimpleNamespace(max_attention_window=None)
    creator._get_draft_max_attention_window = lambda max_seq_len, kv_cache_config: None
    creator._llm_args = SimpleNamespace(attn_backend="TRTLLM", kv_cache_compression_config=None)
    creator._model_engine = SimpleNamespace(
        model=SimpleNamespace(
            model_config=target,
            config=SimpleNamespace(torch_dtype=torch.bfloat16),
            draft_model=SimpleNamespace(
                reads_managed_ctx_pages=lambda tokens_per_block: reads_pages
            ),
        )
    )
    creator._kv_cache_manager_cls = manager_cls
    creator._is_disagg = is_disagg
    creator._get_draft_kv_model_config = lambda: draft
    return creator


@pytest.fixture(autouse=True)
def _shared_draft_kv_auto(monkeypatch):
    monkeypatch.setenv("TRTLLM_DISAGG_SHARED_DRAFT_KV", "auto")


def test_k3_like_drafter_shares_the_hybrid_manager():
    creator = _creator()

    assert creator._dflash_unify_blocker() is None
    assert creator._should_unify_dflash_kv_cache()
    assert not creator._should_create_separate_draft_kv_cache()


def test_hybrid_manager_without_opt_in_keeps_separate_cache():
    creator = _creator(manager_cls=MambaHybridCacheManagerV2)

    assert "cannot hold draft layers" in creator._dflash_unify_blocker()


def test_drafter_not_reading_manager_pages_keeps_separate_cache():
    creator = _creator(reads_pages=False)

    assert "does not read paged draft KV" in creator._dflash_unify_blocker()


def test_hybrid_target_gets_only_the_draft_layer_specs():
    specs = _creator()._build_unified_dflash_layer_specs(SimpleNamespace(max_attention_window=None))

    assert [spec.global_layer_id for spec in specs] == [
        _NUM_TARGET_LAYERS + i for i in range(_NUM_DRAFT_LAYERS)
    ]
    for spec in specs:
        assert spec.owner is KVCacheLayerOwner.DRAFT
        assert (spec.dtype, spec.num_kv_heads, spec.head_dim) == (DataType.FP8, 1, 576)
        assert spec.buffer_roles == (Role.KEY,)


def test_hybrid_cache_size_per_token_adds_the_draft_layer_bytes(monkeypatch):
    estimate_kwargs = []

    def fake_estimate(*args, **kwargs):
        estimate_kwargs.append(kwargs)
        return 100, 7

    monkeypatch.setattr(mamba_cache_manager_v2, "_estimate_mamba_hybrid_cache_cost", fake_estimate)
    specs = (_draft_spec(_NUM_TARGET_LAYERS), _draft_spec(_NUM_TARGET_LAYERS + 1))
    common = dict(max_batch_size=4, kv_cache_config=KvCacheConfig(), tokens_per_block=32)

    without_draft = MambaHybridCacheManagerV2.get_cache_size_per_token(
        SimpleNamespace(), Mapping(), **common
    )
    with_draft = MambaHybridCacheManagerV2.get_cache_size_per_token(
        SimpleNamespace(), Mapping(), layer_specs=specs, shared_draft_block_width=16, **common
    )

    draft_bytes = sum(spec.bytes_per_token() for spec in specs)
    assert draft_bytes > 0
    assert with_draft == (without_draft[0] + draft_bytes, without_draft[1])
    for kwargs in estimate_kwargs:
        assert "layer_specs" not in kwargs
        assert "shared_draft_block_width" not in kwargs


def test_plain_target_still_gets_target_and_draft_specs():
    creator = _creator(manager_cls=KVCacheManagerV2)
    specs = creator._build_unified_dflash_layer_specs(SimpleNamespace(max_attention_window=[4096]))

    owners = [spec.owner for spec in specs]
    assert (
        owners
        == [KVCacheLayerOwner.TARGET] * _NUM_TARGET_LAYERS
        + [KVCacheLayerOwner.DRAFT] * _NUM_DRAFT_LAYERS
    )


def test_k3_like_drafter_shares_the_hybrid_manager_in_disagg():
    # NVFP4 target pages and FP8 draft pages: both K-only, so both transfer.
    creator = _creator(is_disagg=True)

    assert creator._dflash_unify_blocker() is None
    assert creator._should_unify_dflash_kv_cache()


@pytest.mark.parametrize(
    ("target", "draft", "compatible"),
    [
        # K-only latents move whole on every rank, whatever each side's dtype.
        ((1, 576, DataType.NVFP4, True), (1, 576, DataType.FP8, True), True),
        ((1, 576, DataType.NVFP4, True), (1, 576, DataType.BF16, True), True),
        # Head-sliced NVFP4 views need an identical layout.
        ((8, 128, DataType.NVFP4, False), (8, 128, DataType.BF16, False), False),
        ((8, 128, DataType.FP8, False), (8, 128, DataType.BF16, False), True),
    ],
)
def test_disagg_layout_compatibility(target, draft, compatible):
    assert KvCacheCreator._dflash_disagg_layouts_compatible(target, draft) is compatible


def _draft_spec(layer_id, dtype=DataType.FP8):
    return KVCacheLayerSpec(
        global_layer_id=layer_id,
        owner=KVCacheLayerOwner.DRAFT,
        dtype=dtype,
        num_kv_heads=1,
        head_dim=576,
        buffer_roles=(Role.KEY,),
    )


def test_append_draft_layers_follows_the_target_layers():
    specs = (_draft_spec(4), _draft_spec(5))

    mamba_mask, attention_mask, heads = MambaHybridCacheManagerV2._append_draft_layers(
        specs, [True, False, True, False], [False, True, False, True], 1, k_only=True
    )

    assert mamba_mask == [True, False, True, False, False, False]
    assert attention_mask == [False, True, False, True, True, True]
    assert heads == [1, 1, 1, 1, 1, 1]


@pytest.mark.parametrize(
    "specs, match",
    [
        ((_draft_spec(5),), "after the target layers"),
        (
            (
                KVCacheLayerSpec(
                    global_layer_id=4,
                    owner=KVCacheLayerOwner.TARGET,
                    dtype=DataType.FP8,
                    num_kv_heads=1,
                    head_dim=576,
                    buffer_roles=(Role.KEY,),
                ),
            ),
            "only draft-owned layers",
        ),
    ],
)
def test_append_draft_layers_rejects_other_layouts(specs, match):
    with pytest.raises(ValueError, match=match):
        MambaHybridCacheManagerV2._append_draft_layers(
            specs, [True, False, True, False], [False, True, False, True], 1, k_only=True
        )


def test_draft_pool_ratio_holds_the_scarcest_target_token_capacity():
    # Lifecycles in registration order: recurrent state, FP4 MLA, FP8 MLA,
    # draft, HP ring.
    block_bytes = [1000, 400, 50, 300, 50]
    full_history = [False, True, True, True, False]
    target_ratio = [0.45, 0.4, 0.1, 0.05]

    ratio = _insert_draft_pool_ratio(target_ratio, block_bytes, full_history, 3)

    assert sum(ratio) == pytest.approx(1.0)
    # Blocks per lifecycle are proportional to share / block bytes; the FP4
    # lifecycle is the scarcer full-history one (0.4/400 < 0.1/50).
    assert ratio[3] / block_bytes[3] == pytest.approx(ratio[1] / block_bytes[1])
    assert [ratio[i] for i in (0, 1, 2, 4)] == pytest.approx(
        [share * (1 - ratio[3]) for share in target_ratio]
    )


def test_draft_pool_ratio_needs_one_share_per_target_lifecycle():
    with pytest.raises(ValueError, match="2 entries for 3 target layer groups"):
        _insert_draft_pool_ratio([0.5, 0.5], [1, 1, 1, 1], [True] * 4, 1)


# Kimi K3 with a DSpark drafter, one representative layer per layer group in
# registration order: recurrent state, FP4 MLA, FP8 MLA, draft, HP ring. The
# FP8 MLA and draft layers are windowed just past max_seq_len, so each is a
# full-history layer group of its own.
_K3_MAX_SEQ_LEN = 996672
_K3_FP4_BLOCK_BYTES = 22 * 128 * 360
_K3_FP8_BLOCK_BYTES = 2 * 128 * 576
_K3_DRAFT_BLOCK_BYTES = 128 * 576
_K3_TARGET_POOL_RATIO = [0.14, 0.635, 0.205, 0.02]


def _k3_draft_pool_ratio_config(pool_ratio):
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager._appended_draft_layer_specs = {3: None}
    manager.max_seq_len = _K3_MAX_SEQ_LEN
    config = KVCacheManagerConfig(
        tokens_per_block=128,
        cache_tiers=[GpuCacheTierConfig(quota=96 << 30)],
        layers=[
            SsmLayerConfig(layer_id=LayerId(0), buffers=[BufferConfig(role="ssm", size=64 << 20)]),
            AttentionLayerConfig(
                layer_id=LayerId(1), buffers=[BufferConfig(role="key", size=_K3_FP4_BLOCK_BYTES)]
            ),
            AttentionLayerConfig(
                layer_id=LayerId(2),
                buffers=[BufferConfig(role="key", size=_K3_FP8_BLOCK_BYTES)],
                sliding_window_size=_K3_MAX_SEQ_LEN + 2,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(3),
                buffers=[BufferConfig(role="key", size=_K3_DRAFT_BLOCK_BYTES)],
                sliding_window_size=_K3_MAX_SEQ_LEN + 4,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(4),
                buffers=[BufferConfig(role=Role.MLA_HP_TAIL, size=22 * 19 * 576 * 2)],
                sliding_window_size=19,
            ),
        ],
        initial_pool_ratio=pool_ratio,
        # SSM layers require it; MambaHybridCacheManagerV2._build_cache_config sets it.
        commit_min_snapshot=True,
    )
    return manager, config


def test_k3_target_pool_ratio_gets_the_draft_share_inserted():
    manager, config = _k3_draft_pool_ratio_config(_K3_TARGET_POOL_RATIO)

    resolved = manager._with_draft_pool_ratio(config).initial_pool_ratio

    assert len(resolved) == 5 and sum(resolved) == pytest.approx(1.0)
    draft = resolved[3]
    assert 0 < draft < 1
    assert [resolved[i] for i in (0, 1, 2, 4)] == pytest.approx(
        [share * (1 - draft) for share in _K3_TARGET_POOL_RATIO]
    )
    # Token parity with the scarcer full-history target group (FP4 MLA here).
    assert draft / _K3_DRAFT_BLOCK_BYTES == pytest.approx(
        min(resolved[1] / _K3_FP4_BLOCK_BYTES, resolved[2] / _K3_FP8_BLOCK_BYTES)
    )


def test_k3_pool_ratio_with_one_share_per_layer_group_is_used_as_given():
    manager, config = _k3_draft_pool_ratio_config([0.1, 0.5, 0.2, 0.18, 0.02])
    assert manager._with_draft_pool_ratio(config) is config


def test_k3_pool_ratio_of_another_length_is_rejected():
    manager, config = _k3_draft_pool_ratio_config([0.5, 0.5])
    with pytest.raises(ValueError, match="one per target layer group \\(4\\)"):
        manager._with_draft_pool_ratio(config)


def test_base_manager_hooks_read_appended_draft_specs():
    manager = object.__new__(KVCacheManagerV2)
    manager.layer_specs = None
    manager.dtype = DataType.NVFP4
    manager.kv_factor = 1
    manager.pp_layers = [0, 1, 6]
    manager.layer_offsets = {0: 0, 1: 1, 6: 2}
    manager._legacy_draft_layer_ids = set()
    manager._appended_draft_layer_specs = {2: _draft_spec(6)}

    assert manager.get_layer_owner(6) is KVCacheLayerOwner.DRAFT
    assert manager.get_layer_owner(1) is KVCacheLayerOwner.TARGET
    assert manager.draft_layer_ids == (6,)
    assert manager.get_draft_kv_cache_view() is manager
    assert manager._get_layer_dtype_by_local_id(2) == DataType.FP8
    assert manager._get_layer_dtype_by_local_id(1) == DataType.NVFP4
    assert manager._get_buffer_roles_for_layer(2) == [Role.KEY]
    assert manager.get_layer_bytes_per_token(2, Role.ALL) == 576
    assert manager.get_layer_cache_spec(6) == _draft_spec(6)


def test_fp4_mla_transfer_ids_of_appended_draft_layers():
    # Target layer 0 is recurrent, 1 is FP4 MLA (HP ring at virtual layer 4),
    # 2 is an FP8 fallback layer, local 3 is draft layer 6.
    manager = SimpleNamespace(
        pp_layers=[0, 1, 2, 6],
        _appended_draft_layer_specs={3: _draft_spec(6)},
        _cache_manager_layer_ids=[1],
        _fp4_mla_compact_to_local=[1],
        _fp8_mla_manager_layer_ids=[2],
        _hp_manager_layer_ids=[4],
    )
    ids = [
        Fp4MlaV2CacheLayoutPolicy._fp4_mla_disagg_global_layer_id(manager, layer)
        for layer in range(5)
    ]

    assert ids == [None, 2, 4, 12, 3]


@pytest.mark.parametrize("reads_helix_pages", [True, False])
def test_helix_target_shares_only_with_a_helix_sharding_drafter(reads_helix_pages):
    # Kimi K3 helix8 generation: TP1 x CP8, no attention DP.
    creator = _creator(tp_size=1, is_disagg=True)
    creator._mapping.cp_size = 8
    creator._mapping.enable_attention_dp = False
    creator._mapping.has_cp_helix = lambda: True
    creator._model_engine.model.draft_model.reads_helix_sharded_ctx_pages = reads_helix_pages

    blocker = creator._dflash_unify_blocker()

    if reads_helix_pages:
        assert blocker is None
    else:
        assert "does not read a helix-sharded context" in blocker
