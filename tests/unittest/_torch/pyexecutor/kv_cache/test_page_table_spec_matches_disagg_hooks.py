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
"""Each KV cache manager's page_table_spec says what its disaggregation hooks say.

A manager states how its roles spread over ranks twice: in the hooks the disaggregated transfer
reads (``get_disagg_role_mapper_kinds``, ``get_disagg_role_layouts``,
``get_disagg_ignored_roles``, ``get_disagg_global_layer_ids``) and in its ``page_table_spec``.
These tests hold the two to the same layout for every role the manager registers."""

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
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
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
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.runtime.kv_cache_manager_v2 import DataRole, LayerId

pytestmark = pytest.mark.cpu_only

KV_ROLES = (Role.KEY, Role.VALUE, Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE)
STATE_ROLES = (MambaRole.CONV_STATE, MambaRole.SSM_STATE)


def assert_spec_matches_disagg_hooks(manager, roles, layer_groups: int = 0) -> PageTableSpec:
    """Check that ``manager``'s spec lays out each of ``roles`` as its disaggregation hooks do,
    leaves the roles they ignore, and gives the layers of its first ``layer_groups`` layer groups
    the global layer ids they do; return the spec."""
    spec = manager.page_table_spec()
    kinds = manager.get_disagg_role_mapper_kinds()
    geometry = manager.get_disagg_role_layouts()
    ignored = {str(role) for role in manager.get_disagg_ignored_roles()}
    assert set(spec.untransferred_roles | spec.derived_roles) == ignored
    for role in roles:
        if str(role) in ignored:
            # Whatever else lays it out, a role that never moves is the rank's own share.
            assert spec.layout_of(str(role)) == Own(), role
            continue
        layout = spec.layout_of(str(role))
        kind = kinds.get(role, kinds[Role.ALL])
        shape = geometry.get(role)
        if kind == MapperKind.REPLICATED and shape is not None and shape.ring_slot_bytes:
            assert isinstance(layout, RingSlots), role
            assert layout.dtype == "bfloat16", role
        elif kind == MapperKind.REPLICATED:
            assert layout == Replicated(), role
        elif kind == MapperKind.SECTIONED:
            assert isinstance(layout, Sectioned), role
            assert layout.section_bytes == tuple(shape.section_bytes), role
        else:
            assert isinstance(layout, ByHead), role
            assert layout.token_major == (kind == MapperKind.NHD), role
            assert layout.head_bytes == (None if shape is None else shape.bytes_per_head), role
    for group in range(layer_groups):
        layers = manager.impl.buffer_grouping[group]
        internal = list(dict.fromkeys(int(buffer.layer_id) for buffer in layers))
        names = [spec.layer_ids[layer] for layer in internal]
        assert [g for g, _ in names] == manager.get_disagg_global_layer_ids(group)
    return spec


def _stub(manager_cls, **attributes):
    """A manager of ``manager_cls`` holding only ``attributes``, which its spec and hooks read; a
    K/V cache of FP16 unless they say otherwise."""
    manager = object.__new__(manager_cls)
    manager.dtype = DataType.HALF
    manager.kv_factor = 2
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

    spec = assert_spec_matches_disagg_hooks(manager, KV_ROLES)

    assert all(spec.layout_of(str(role)) == ByHead() for role in KV_ROLES)
    assert spec.quant_mode == quant_mode
    assert spec.layer_ids is None


def test_base_manager_holds_an_mla_cache_s_one_latent_head_on_every_rank():
    manager = _stub(KVCacheManagerV2, kv_factor=1)

    spec = assert_spec_matches_disagg_hooks(manager, (Role.KEY, Role.KEY_BLOCK_SCALE))

    assert spec.layout_of(str(Role.KEY)) == ByHead(total=1)


def _mixed_precision_buffers(layer_grouping):
    return [
        [SimpleNamespace(layer_id=layer, role=Role.KEY) for layer in layers]
        for layers in layer_grouping
    ]


def test_base_manager_names_mixed_precision_layers_as_its_hooks_do():
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
    # As _add_mixed_precision_layers rebinds the hook.
    manager.get_disagg_global_layer_ids = manager._mixed_precision_global_layer_ids

    spec = assert_spec_matches_disagg_hooks(manager, KV_ROLES, layer_groups=2)

    # Low layers take global layers 2 * L, their high layers 2 * L + 1.
    assert dict(spec.layer_ids) == {0: (8, 4), 1: (10, 5), 2: (9, 4), 3: (11, 5)}
    assert spec.mixed_precision == ("nvfp4", "bf16")
    # Layers it keeps unmixed hold the manager's own type.
    assert spec.quant_mode is None


def test_virtual_layer_manager_names_mixed_precision_layers_as_its_hooks_do():
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
    manager.get_disagg_global_layer_ids = manager._mixed_precision_global_layer_ids
    num_types = len(DeepseekV4AttentionType)

    spec = assert_spec_matches_disagg_hooks(manager, (), layer_groups=3)

    assert dict(spec.layer_ids) == {
        0: (2 * (3 * num_types + swa.value), 3),
        1: (2 * (3 * num_types + compress.value), 3),
        2: (2 * 3 + 1, 3),
    }
    # Its own pool already holds the mixed-precision low format; its high layers hold FP8.
    assert spec.mixed_precision == ("nvfp4", "fp8")
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
        (DataType.FP8, True, "fp8_ds_mla"),
    ],
)
def test_dsa_index_keys_are_alike_on_every_rank(dtype, fp8_ds_mla, quant_mode):
    manager = _stub(DSACacheManagerV2, dtype=dtype, kv_factor=1, use_fp8_ds_mla=fp8_ds_mla)

    spec = assert_spec_matches_disagg_hooks(manager, (Role.KEY, Role.INDEX_KEY))

    assert spec.layout_of(str(Role.KEY)) == ByHead(total=1)
    assert spec.layout_of(str(Role.INDEX_KEY)) == Replicated()
    # Pages carrying their own scales are not plain FP8.
    assert spec.quant_mode == quant_mode


@pytest.mark.parametrize("storage, token_major", [(MapperKind.HND, False), (MapperKind.NHD, True)])
def test_minimax_m3_lays_kv_out_as_its_backend_writes_it(storage, token_major):
    manager = _stub(MiniMaxM3KVCacheManagerV2, _main_kv_mapper_kind=storage)

    spec = assert_spec_matches_disagg_hooks(manager, (*KV_ROLES, Role.INDEX_KEY))

    assert spec.layout_of(str(Role.KEY)) == ByHead(token_major=token_major)
    assert spec.layout_of(str(Role.INDEX_KEY)) == Replicated()


@pytest.mark.parametrize("bounded_replay", [False, True])
def test_csa2_caches_are_alike_on_every_rank(bounded_replay):
    manager = _stub(CSA2CacheManager, bounded_replay_on_generation=bounded_replay)
    roles = tuple(cache.role for cache in CSA2CacheRole)

    spec = assert_spec_matches_disagg_hooks(manager, roles)

    assert all(spec.layout_of(str(role)) == Replicated() for role in roles)
    # The sliding-window caches a generation worker rebuilds under bounded replay are the ones
    # the hooks keep in transfer views of their own.
    for role in roles:
        partitioned = manager.get_disagg_pool_view_partition(LayerId(0), role) is not None
        assert (str(role) in spec.request_rebuilt_roles) == partitioned


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

    spec = assert_spec_matches_disagg_hooks(manager, (*head_split, padding))

    # Each holds MLA latents of one head, which every tensor rank holds whole.
    assert all(spec.layout_of(str(role)) == ByHead(total=1) for role in head_split)
    assert spec.untransferred_roles == frozenset({str(padding)})
    assert spec.quant_mode == quant_mode


def test_hybrid_manager_sections_conv_state_and_splits_ssm_state_by_head():
    conv, ssm = STATE_ROLES

    spec = assert_spec_matches_disagg_hooks(
        _hybrid(MambaHybridCacheManagerV2), (*KV_ROLES, conv, ssm)
    )

    assert spec.layout_of(str(conv)) == Sectioned((192, 192, 192), dtype="float16")
    assert spec.layout_of(str(ssm)) == ByHead(head_bytes=128, dtype="float16")


def test_hybrid_manager_without_recurrent_layers_lays_out_its_kv_alone():
    manager = _hybrid(MambaHybridCacheManagerV2)
    manager.local_num_mamba_layers = 0

    spec = assert_spec_matches_disagg_hooks(manager, KV_ROLES)

    conv, ssm = STATE_ROLES
    assert spec.layout_of(str(conv)) is None and spec.layout_of(str(ssm)) is None


def test_qwen4_exp_side_caches_are_alike_on_every_rank():
    roles = (PLE_CONV_STATE, PLE_NGRAM_CONTEXT, QSA_INDEX_POSITION, QSA_SIDE_PLANES)

    spec = assert_spec_matches_disagg_hooks(
        _hybrid(Qwen4ExpHybridCacheManager), (*KV_ROLES, *STATE_ROLES, *roles, Role.INDEX_KEY)
    )

    assert all(spec.layout_of(str(role)) == Replicated() for role in (*roles, Role.INDEX_KEY))


@pytest.mark.parametrize("manager_cls", [KimiK3HybridCacheManager, Glm5NextCacheManager])
@pytest.mark.parametrize("kda_replay", [False, True])
def test_kda_replay_caches_never_move(manager_cls, kda_replay):
    replay_role = DataRole("kda_test_replay")
    replay = SimpleNamespace(disagg_ignored_roles=lambda layout: frozenset({replay_role}))
    manager = _hybrid(manager_cls, _kda_replay=replay if kda_replay else None, _state_layout=None)
    # Without replay the manager registers no replay cache.
    roles = (*KV_ROLES, *STATE_ROLES, *((replay_role,) if kda_replay else ()))
    if manager_cls is Glm5NextCacheManager:
        roles += (Role.INDEX_KEY,)

    spec = assert_spec_matches_disagg_hooks(manager, roles)

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

    spec = assert_spec_matches_disagg_hooks(manager, roles, layer_groups=2)

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

    spec = assert_spec_matches_disagg_hooks(manager, roles, layer_groups=3)

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
