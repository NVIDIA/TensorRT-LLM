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
"""The role layouts, layer names and quantization declared by each KV cache manager."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import (
    Fp4MlaKVCacheManagerV2,
    Fp4MlaV2CacheLayoutPolicy,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
    CSA2CacheManager,
    CSA2CacheRole,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
    COMPRESS_BLOCK_SCALE_ROLE,
    DeepseekV4CacheManager,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import DeepseekV4AttentionType
from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import DSACacheManagerV2
from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.cache_manager import (
    Glm5NextCacheManager,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._torch.modules.fla.cache_manager import Qwen35HybridCacheManager
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
from tensorrt_llm._torch.modules.mamba.cache_manager import NemotronHybridCacheManager
from tensorrt_llm._torch.modules.qwen4_exp.cache_manager import (
    PLE_CONV_STATE,
    PLE_NGRAM_CONTEXT,
    QSA_INDEX_POSITION,
    QSA_SIDE_PLANES,
    Qwen4ExpHybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import MambaRole
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.page_table import (
    ByHead,
    Own,
    PageTableSpec,
    Replicated,
    RingSlots,
    Sectioned,
    layer_ids_of,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.runtime.kv_cache_manager_v2 import DataRole, LayerId

pytestmark = pytest.mark.cpu_only

KV_ROLES = (Role.KEY, Role.VALUE, Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE)
STATE_ROLES = (MambaRole.CONV_STATE, MambaRole.SSM_STATE)


def _spec_for(manager, roles) -> PageTableSpec:
    spec = manager.page_table_spec()
    assert not spec.untransferred_roles & spec.derived_roles
    for role in roles:
        assert spec.layout_of(str(role)) is not None, role
    for role in spec.untransferred_roles | spec.derived_roles:
        assert spec.layout_of(role) == Own(), role
    return spec


def _stub(manager_cls, **attributes):
    """A manager of ``manager_cls`` holding only ``attributes``, which its spec reads; a
    K/V cache of FP16 over model layer 0 unless they say otherwise."""
    manager = object.__new__(manager_cls)
    manager.dtype = DataType.HALF
    manager.kv_factor = 2
    manager.pp_layers = [0]
    manager.impl = SimpleNamespace(
        layer_grouping=[[LayerId(0)]],
        buffer_grouping=[[SimpleNamespace(layer_id=LayerId(0), role=Role.KEY)]],
    )
    for name, value in attributes.items():
        setattr(manager, name, value)
    return manager


def _hybrid(manager_cls, **attributes):
    """A hybrid manager stub with one recurrent layer: conv state of three 32-channel sections
    and SSM state of 4 heads of 8 x 8, both FP16."""
    state = dict(
        _attention_cache_layout_policy=None,
        local_num_mamba_layers=1,
        conv_state_shape=[96, 3],
        conv_section_dims=[32, 32, 32],
        ssm_state_shape=[4, 8, 8],
        conv_state_dtype=torch.float16,
        ssm_state_dtype=torch.float16,
    )
    return _stub(manager_cls, **{**state, **attributes})


@pytest.mark.parametrize(
    "dtype, quant_mode", [(DataType.HALF, None), (DataType.FP8, "fp8"), (DataType.NVFP4, "nvfp4")]
)
def test_base_manager_splits_kv_by_head(dtype, quant_mode):
    manager = _stub(KVCacheManagerV2, dtype=dtype)

    spec = _spec_for(manager, KV_ROLES)

    assert all(spec.layout_of(str(role)) == ByHead() for role in KV_ROLES)
    assert spec.quant_mode == quant_mode
    assert spec.layer_ids is None
    assert layer_ids_of(manager, spec) == {0: (0, 0)}


def test_base_manager_holds_an_mla_cache_s_one_latent_head_on_every_rank():
    manager = _stub(KVCacheManagerV2, kv_factor=1)

    spec = _spec_for(manager, (Role.KEY, Role.KEY_BLOCK_SCALE))

    assert spec.layout_of(str(Role.KEY)) == ByHead(total=1)


def _mixed_precision_buffers(layer_grouping):
    return [
        [SimpleNamespace(layer_id=layer, role=Role.KEY) for layer in layers]
        for layers in layer_grouping
    ]


def test_base_manager_names_mixed_precision_layers():
    """Model layers 4 and 5 on this pipeline rank, their high layers internal layers 2 and 3."""
    layer_grouping = [[LayerId(0), LayerId(1)], [LayerId(2), LayerId(3)]]
    manager = _stub(
        KVCacheManagerV2,
        dtype=DataType.BF16,
        pp_layers=[4, 5],
        mixed_precision_low_dtype=DataType.NVFP4,
        mixed_precision_high_dtype=DataType.BF16,
        mixed_precision_high_layers={0: 2, 1: 3},
        impl=SimpleNamespace(
            layer_grouping=layer_grouping,
            buffer_grouping=_mixed_precision_buffers(layer_grouping),
        ),
    )

    spec = _spec_for(manager, KV_ROLES)

    # Low layers take global layers 2 * L, their high layers 2 * L + 1.
    assert dict(spec.layer_ids) == {0: (8, 4), 1: (10, 5), 2: (9, 4), 3: (11, 5)}
    assert spec.mixed_precision is None
    # Layers it keeps unmixed hold the manager's own type.
    assert spec.quant_mode is None


def test_virtual_layer_manager_uses_default_layer_names():
    """Model layer 3 keeps its sliding-window and compressed caches in internal layers 0 and 1."""
    swa, compress = DeepseekV4AttentionType.SWA, DeepseekV4AttentionType.COMPRESS
    layer_grouping = [[LayerId(0)], [LayerId(1)]]
    manager = _stub(
        DeepseekV4CacheManager,
        dtype=DataType.BF16,
        kv_factor=1,
        _disagg_ignored_roles=frozenset(),
        use_fp8_ds_mla=False,
        _use_nvfp4_compress=False,
        pp_layers=[3],
        _layer_attn_to_layer_id={(3, swa): 0, (3, compress): 1},
        impl=SimpleNamespace(
            layer_grouping=layer_grouping,
            buffer_grouping=_mixed_precision_buffers(layer_grouping),
        ),
    )
    num_types = len(DeepseekV4AttentionType)

    spec = _spec_for(manager, ())

    # Model layer x attention types + attention type, as the transfer encodes them.
    assert spec.layer_ids is None
    assert layer_ids_of(manager, spec) == {
        0: (3 * num_types + swa.value, 3),
        1: (3 * num_types + compress.value, 3),
    }


def test_latent_manager_uses_each_explicit_layers_own_heads():
    layer_specs = (
        SimpleNamespace(global_layer_id=0, kv_factor=1),
        SimpleNamespace(global_layer_id=1, kv_factor=2),
    )
    manager = _stub(KVCacheManagerV2, kv_factor=1, layer_specs=layer_specs)
    assert manager.page_table_spec().layout_of(str(Role.KEY)) == ByHead()


def test_virtual_layer_manager_names_mixed_precision_layers():
    """Model layer 3 keeps its sliding-window and compressed caches in internal layers 0 and 1,
    the high layer of its local layer in internal layer 2."""
    swa, compress = DeepseekV4AttentionType.SWA, DeepseekV4AttentionType.COMPRESS
    layer_grouping = [[LayerId(0)], [LayerId(1)], [LayerId(2)]]
    manager = _stub(
        DeepseekV4CacheManager,
        dtype=DataType.BF16,
        kv_factor=1,
        _disagg_ignored_roles=frozenset(),
        use_fp8_ds_mla=False,
        _use_nvfp4_compress=False,
        pp_layers=[3],
        _layer_attn_to_layer_id={(3, swa): 0, (3, compress): 1},
        mixed_precision_low_dtype=DataType.NVFP4,
        mixed_precision_high_dtype=DataType.FP8,
        mixed_precision_high_layers={0: 2},
        impl=SimpleNamespace(
            layer_grouping=layer_grouping,
            buffer_grouping=_mixed_precision_buffers(layer_grouping),
        ),
    )
    num_types = len(DeepseekV4AttentionType)

    spec = _spec_for(manager, ())

    assert dict(spec.layer_ids) == {
        0: (2 * (3 * num_types + swa.value), 3),
        1: (2 * (3 * num_types + compress.value), 3),
        2: (2 * 3 + 1, 3),
    }
    assert spec.mixed_precision is None
    assert spec.quant_mode is None


class _SplitLayerStorage:
    """Storage over model layers 4 and 5 whose layer 0 the reuse policies split over two cache
    groups, its high layer internal layer 2; like the runtime's, its deprecated layer grouping
    refuses to list a split layer."""

    buffer_grouping = [
        [SimpleNamespace(layer_id=LayerId(layer), role=Role.KEY) for layer in (0, 1)],
        [SimpleNamespace(layer_id=LayerId(2), role=Role.KEY)],
        [SimpleNamespace(layer_id=LayerId(0), role=Role.INDEX_KEY)],
    ]

    @property
    def layer_grouping(self):
        raise RuntimeError("layer 0 is split over cache groups")


def test_mixed_precision_layer_ids_follow_the_buffers_of_a_split_layer():
    manager = _stub(
        DSACacheManagerV2,
        dtype=DataType.BF16,
        kv_factor=1,
        use_fp8_ds_mla=False,
        pp_layers=[4, 5],
        mixed_precision_low_dtype=DataType.NVFP4,
        mixed_precision_high_dtype=DataType.BF16,
        mixed_precision_high_layers={0: 2},
        impl=_SplitLayerStorage(),
    )

    spec = manager.page_table_spec()

    # A layer's global layer is its own, whichever cache groups hold its buffers.
    assert dict(spec.layer_ids) == {0: (8, 4), 1: (10, 5), 2: (9, 4)}


@pytest.mark.parametrize(
    "dtype, fp8_ds_mla, quant_mode",
    [
        (DataType.BF16, False, None),
        (DataType.FP8, False, "fp8"),
        (DataType.NVFP4, False, "nvfp4"),
        (DataType.FP8, True, "fp8_ds_mla"),
    ],
)
def test_dsa_index_keys_are_alike_on_every_rank(dtype, fp8_ds_mla, quant_mode):
    manager = _stub(DSACacheManagerV2, dtype=dtype, kv_factor=1, use_fp8_ds_mla=fp8_ds_mla)

    spec = _spec_for(manager, (Role.KEY, Role.INDEX_KEY))

    assert spec.layout_of(str(Role.KEY)) == ByHead(total=1)
    assert spec.layout_of(str(Role.INDEX_KEY)) == Replicated()
    # Pages carrying their own scales are not plain FP8.
    assert spec.quant_mode == quant_mode


@pytest.mark.parametrize("storage, token_major", [("HND", False), ("NHD", True)])
def test_minimax_m3_lays_kv_out_as_its_backend_writes_it(storage, token_major):
    manager = _stub(MiniMaxM3KVCacheManagerV2, _main_kv_storage_layout=storage)

    spec = _spec_for(manager, (*KV_ROLES, Role.INDEX_KEY))

    assert spec.layout_of(str(Role.KEY)) == ByHead(token_major=token_major)
    assert spec.layout_of(str(Role.INDEX_KEY)) == Replicated()


@pytest.mark.parametrize("bounded_replay", [False, True])
def test_csa2_caches_are_alike_on_every_rank(bounded_replay):
    manager = _stub(CSA2CacheManager, bounded_replay_on_generation=bounded_replay)
    roles = tuple(cache.role for cache in CSA2CacheRole)

    spec = _spec_for(manager, roles)

    assert all(spec.layout_of(str(role)) == Replicated() for role in roles)
    # The persistent caches are packed records whatever the staging type.
    assert spec.quant_mode == "csa2_packed"
    assert spec.request_rebuilt_roles == (
        {str(CSA2CacheRole.SWA.role)} if bounded_replay else set()
    )


@pytest.mark.parametrize(
    "dtype, fp8_ds_mla, nvfp4_compress, quant_mode",
    [
        (DataType.BF16, False, False, None),
        (DataType.FP8, False, False, "fp8"),
        (DataType.FP8, True, False, "fp8_ds_mla"),
        # NVFP4 KV compresses the compressed cache and keeps the others in FP8.
        (DataType.FP8, False, True, "nvfp4"),
    ],
)
def test_deepseek_v4_splits_its_attention_caches_by_head(
    dtype, fp8_ds_mla, nvfp4_compress, quant_mode
):
    padding = DataRole("deepseek_v4_test_alignment_padding")
    manager = _stub(
        DeepseekV4CacheManager,
        dtype=dtype,
        kv_factor=1,
        _disagg_ignored_roles=frozenset({padding}),
        use_fp8_ds_mla=fp8_ds_mla,
        _use_nvfp4_compress=nvfp4_compress,
    )
    head_split = (COMPRESS_BLOCK_SCALE_ROLE, *(t.role for t in DeepseekV4AttentionType))

    spec = _spec_for(manager, (*head_split, padding))

    # Each holds MLA latents of one head, which every tensor rank holds whole.
    assert all(spec.layout_of(str(role)) == ByHead(total=1) for role in head_split)
    assert spec.untransferred_roles == frozenset({str(padding)})
    assert spec.quant_mode == quant_mode


def test_hybrid_manager_sections_conv_state_and_splits_ssm_state_by_head():
    conv, ssm = STATE_ROLES

    spec = _spec_for(_hybrid(MambaHybridCacheManagerV2), (*KV_ROLES, conv, ssm))

    assert spec.layout_of(str(conv)) == Sectioned((192, 192, 192), dtype="float16")
    assert spec.layout_of(str(ssm)) == ByHead(head_bytes=128, dtype="float16")


def test_hybrid_manager_without_recurrent_layers_lays_out_its_kv_alone():
    manager = _hybrid(MambaHybridCacheManagerV2)
    manager.local_num_mamba_layers = 0

    spec = _spec_for(manager, KV_ROLES)

    conv, ssm = STATE_ROLES
    assert spec.layout_of(str(conv)) is None and spec.layout_of(str(ssm)) is None


@pytest.mark.parametrize(
    "manager_cls",
    [Qwen35HybridCacheManager, NemotronHybridCacheManager, Qwen4ExpHybridCacheManager],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_model_replay_state_never_moves(manager_cls, enabled):
    replay_role = DataRole("test_replay")
    region = SimpleNamespace(roles=frozenset({replay_role})) if enabled else None
    manager = _hybrid(manager_cls, _replay_region=region)
    roles = (*STATE_ROLES, *KV_ROLES, *((replay_role,) if enabled else ()))
    spec = _spec_for(manager, roles)

    assert (str(replay_role) in spec.untransferred_roles) == enabled


def test_qwen4_exp_side_caches_are_alike_on_every_rank():
    roles = (PLE_CONV_STATE, PLE_NGRAM_CONTEXT, QSA_INDEX_POSITION, QSA_SIDE_PLANES)

    spec = _spec_for(
        _hybrid(Qwen4ExpHybridCacheManager), (*KV_ROLES, *STATE_ROLES, *roles, Role.INDEX_KEY)
    )

    assert all(spec.layout_of(str(role)) == Replicated() for role in (*roles, Role.INDEX_KEY))


@pytest.mark.parametrize("manager_cls", [KimiK3HybridCacheManager, Glm5NextCacheManager])
@pytest.mark.parametrize("kda_replay", [False, True])
def test_kda_replay_caches_never_move(manager_cls, kda_replay):
    replay_role = DataRole("kda_test_replay")
    replay = SimpleNamespace(roles=frozenset({replay_role}))
    manager = _hybrid(manager_cls, _kda_replay_region=replay if kda_replay else None)
    # Without replay the manager registers no replay cache.
    roles = (*KV_ROLES, *STATE_ROLES, *((replay_role,) if kda_replay else ()))
    if manager_cls is Glm5NextCacheManager:
        roles += (Role.INDEX_KEY,)

    spec = _spec_for(manager, roles)

    assert (str(replay_role) in spec.untransferred_roles) == kda_replay


def _fp4_mla_manager(hp_pool_size=19):
    """An FP4 MLA manager over model layers 0 and 1: latent caches in internal layers 0 and 1,
    their high-precision tails in 2 and 3, one layer group each."""

    def buffers(layers, role):
        return [SimpleNamespace(layer_id=LayerId(layer), role=role) for layer in layers]

    return _stub(
        Fp4MlaKVCacheManagerV2,
        dtype=DataType.NVFP4,
        kv_factor=1,
        pp_layers=[0, 1],
        head_dim_per_layer=[96, 96],
        _cache_manager_layer_ids=[LayerId(0), LayerId(1)],
        _hp_manager_layer_ids=[LayerId(2), LayerId(3)],
        _fp8_mla_manager_layer_ids=[],
        _fp4_mla_compact_to_local=[0, 1],
        _fp4_mla_hp_pool_size=hp_pool_size,
        impl=SimpleNamespace(
            layer_grouping=[[LayerId(0), LayerId(1)], [LayerId(2), LayerId(3)]],
            buffer_grouping=[buffers((0, 1), Role.KEY), buffers((2, 3), Role.MLA_HP_TAIL)],
        ),
    )


def test_fp4_mla_lays_latents_token_major_and_its_tail_as_a_ring():
    manager = _fp4_mla_manager()
    roles = (
        Role.KEY,
        Role.KEY_BLOCK_SCALE,
        Role.MLA_HP_TAIL,
        Role.MLA_V_SCALE,
        Role.MLA_V_PACKED,
    )

    spec = _spec_for(manager, roles)

    assert spec.layout_of(str(Role.KEY)) == ByHead(token_major=True, total=1)
    assert spec.layout_of(str(Role.MLA_HP_TAIL)) == RingSlots(19, dtype="bfloat16")
    # V scales and packed V derive from K and K scales on the generation worker, so they never
    # move, though the default lays out every role the spec does not name token-major.
    assert spec.derived_roles == frozenset({str(Role.MLA_V_SCALE), str(Role.MLA_V_PACKED)})
    assert spec.layout_of(str(Role.MLA_V_SCALE)) == Own()
    assert spec.quant_mode == "nvfp4"
    # Latent caches take global layers 2 * L, their tails 2 * L + 1.
    assert dict(spec.layer_ids) == {0: (0, 0), 1: (2, 1), 2: (1, 0), 3: (3, 1)}


def test_fp4_mla_refuses_a_layer_it_does_not_name():
    manager = _fp4_mla_manager()
    manager.impl.buffer_grouping.append([SimpleNamespace(layer_id=LayerId(7), role=Role.KEY)])

    with pytest.raises(ValueError, match="internal layer 7"):
        manager.page_table_spec()


def test_hybrid_fp4_mla_lays_its_attention_roles_and_layer_names_over_its_state():
    """Recurrent layers 0 and 2 and FP4 MLA layers 1 and 3, whose tails are internal layers 4
    and 5; one layer group for the state, the latents and the tails each."""

    def buffers(layers, role):
        return [SimpleNamespace(layer_id=LayerId(layer), role=role) for layer in layers]

    manager = _hybrid(
        MambaHybridCacheManagerV2,
        kv_factor=1,
        local_num_mamba_layers=2,
        pp_layers=[0, 1, 2, 3],
        _mamba_layer_mask=[True, False, True, False],
        head_dim_per_layer=[96] * 4,
        _cache_manager_layer_ids=[LayerId(1), LayerId(3)],
        _hp_manager_layer_ids=[LayerId(4), LayerId(5)],
        _fp8_mla_manager_layer_ids=[],
        _fp4_mla_compact_to_local=[1, 3],
        _fp4_mla_hp_pool_size=19,
        impl=SimpleNamespace(
            layer_grouping=[
                [LayerId(0), LayerId(2)],
                [LayerId(1), LayerId(3)],
                [LayerId(4), LayerId(5)],
            ],
            buffer_grouping=[
                buffers((0, 2), MambaRole.SSM_STATE),
                buffers((1, 3), Role.KEY),
                buffers((4, 5), Role.MLA_HP_TAIL),
            ],
        ),
    )
    manager._attention_cache_layout_policy = Fp4MlaV2CacheLayoutPolicy
    Fp4MlaV2CacheLayoutPolicy.install_manager_capabilities(manager)
    roles = (
        *STATE_ROLES,
        Role.KEY,
        Role.KEY_BLOCK_SCALE,
        Role.MLA_HP_TAIL,
        Role.MLA_V_SCALE,
        Role.MLA_V_PACKED,
    )

    spec = _spec_for(manager, roles)

    assert isinstance(spec.layout_of(str(MambaRole.CONV_STATE)), Sectioned)
    assert spec.layout_of(str(Role.KEY)) == ByHead(token_major=True, total=1)
    assert spec.layout_of(str(Role.MLA_V_SCALE)) == Own()
    # Recurrent layers and latent caches take global layers 2 * L, the tails 2 * L + 1.
    assert dict(spec.layer_ids) == {
        0: (0, 0),
        2: (4, 2),
        1: (2, 1),
        3: (6, 3),
        4: (3, 1),
        5: (7, 3),
    }
