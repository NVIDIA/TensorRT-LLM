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
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
    _insert_draft_pool_ratio,
)
from tensorrt_llm.bindings import DataType
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


def _draft_pool_ratio_manager_and_config(pool_ratio):
    # Layer groups: target attention (256 B/block), recurrent state, draft (128 B/block).
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager._appended_draft_layer_specs = {2: None}
    manager.max_seq_len = 1024
    manager._pool_ratio_groups = []
    config = KVCacheManagerConfig(
        tokens_per_block=32,
        cache_tiers=[GpuCacheTierConfig(quota=64 << 20)],
        layers=[
            AttentionLayerConfig(layer_id=LayerId(0), buffers=[BufferConfig(role="key", size=256)]),
            SsmLayerConfig(layer_id=LayerId(1), buffers=[BufferConfig(role="ssm", size=1024)]),
            AttentionLayerConfig(
                layer_id=LayerId(2),
                buffers=[BufferConfig(role="key", size=128)],
                sliding_window_size=4096,
            ),
        ],
        initial_pool_ratio=pool_ratio,
        # SSM layers require it; MambaHybridCacheManagerV2._build_cache_config sets it.
        commit_min_snapshot=True,
    )
    return manager, config


def test_draft_pool_ratio_full_list_is_checked_against_token_parity():
    manager, config = _draft_pool_ratio_manager_and_config([0.6, 0.4])
    parity = manager._with_draft_pool_ratio(config).initial_pool_ratio
    assert parity == pytest.approx([0.6 / 1.3, 0.4 / 1.3, 0.3 / 1.3])

    manager, config = _draft_pool_ratio_manager_and_config(parity)
    assert manager._with_draft_pool_ratio(config) is config
    # The native grouping is checked against this order once built.
    assert [group for group, _ in manager._pool_ratio_groups] == [0, 1, 2]

    # Draft entry placed first instead of last: rejected, naming the draft group.
    manager, config = _draft_pool_ratio_manager_and_config(list(reversed(parity)))
    with pytest.raises(ValueError, match=r"draft layer group \(group 2\)"):
        manager._with_draft_pool_ratio(config)


def test_draft_pool_ratio_arity_error_names_the_layer_group_order():
    manager, config = _draft_pool_ratio_manager_and_config([0.25] * 4)
    with pytest.raises(
        ValueError,
        match=r"Layer groups in order: 0: attention \(layer 0 .*1: ssm .*2: draft \(layer 2 ",
    ):
        manager._with_draft_pool_ratio(config)


# Kimi K3 with a DSpark drafter, one representative layer per layer group in
# registration order. Bytes per 128-token block: 22 FP4 MLA layers, 2 FP8 MLA
# layers, 1 FP8 draft layer, 22 HP rings of 16 + 3 BF16 tokens.
_K3_MAX_SEQ_LEN = 996672
_K3_FP4_BLOCK_BYTES = 22 * 128 * 360
_K3_FP8_BLOCK_BYTES = 2 * 128 * 576
_K3_GPU_QUOTA = 96 << 30
# The PR order [ssm, fp4_mla, fp8_mla, mla_hp_tail] (draft inserted) and the
# same shares in hot pool-group order [fp8_mla, mla_hp_tail, fp4_mla, ssm].
_K3_LAYER_GROUP_ORDER = [0.14, 0.635, 0.205, 0.02]
_K3_POOL_GROUP_ORDER = [0.205, 0.02, 0.635, 0.14]


def _k3_pool_ratio_manager_and_config(pool_ratio):
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager._appended_draft_layer_specs = {3: None}
    manager.max_seq_len = _K3_MAX_SEQ_LEN
    manager._pool_ratio_groups = []
    manager._attention_cache_layout_policy = Fp4MlaV2CacheLayoutPolicy
    manager._cache_manager_layer_ids = [LayerId(1)]
    manager._fp8_mla_manager_layer_ids = [LayerId(2)]
    config = KVCacheManagerConfig(
        tokens_per_block=128,
        cache_tiers=[GpuCacheTierConfig(quota=_K3_GPU_QUOTA)],
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
                buffers=[BufferConfig(role="key", size=128 * 576)],
                sliding_window_size=_K3_MAX_SEQ_LEN + 4,
            ),
            AttentionLayerConfig(
                layer_id=LayerId(4),
                buffers=[BufferConfig(role=Role.MLA_HP_TAIL, size=22 * 19 * 576 * 2)],
                sliding_window_size=19,
            ),
        ],
        initial_pool_ratio=pool_ratio,
        commit_min_snapshot=True,
    )
    return manager, config


def _capture_log(monkeypatch, level):
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        mamba_cache_manager_v2 as module,
    )

    messages = []
    monkeypatch.setattr(module.logger, level, messages.append)
    return messages


def test_k3_pool_ratio_in_layer_group_order_logs_the_resolved_split(monkeypatch):
    infos = _capture_log(monkeypatch, "info")
    warnings = _capture_log(monkeypatch, "warning")
    manager, config = _k3_pool_ratio_manager_and_config(_K3_LAYER_GROUP_ORDER)

    resolved = manager._with_draft_pool_ratio(config).initial_pool_ratio

    assert len(resolved) == 5 and sum(resolved) == pytest.approx(1.0)
    assert [resolved[i] / (1 - resolved[3]) for i in (0, 1, 2, 4)] == pytest.approx(
        _K3_LAYER_GROUP_ORDER
    )
    assert warnings == []
    (line,) = [m for m in infos if "pool_ratio by layer group" in m]
    assert "0: ssm=" in line and "1: fp4_mla=" in line and "2: fp8_mla=" in line
    assert "3: draft=" in line and "(inserted)" in line and "4: mla_hp_tail=" in line


def test_k3_pool_ratio_in_pool_group_order_is_rejected():
    manager, config = _k3_pool_ratio_manager_and_config(_K3_POOL_GROUP_ORDER)

    with pytest.raises(
        ValueError,
        match=r"layer group 1 \(fp4_mla, pool_ratio\[1\] = .*while layer group 2 \(fp8_mla\)",
    ):
        manager._with_draft_pool_ratio(config)


def test_pool_ratio_group_at_floor_without_a_richer_group_warns(monkeypatch):
    warnings = _capture_log(monkeypatch, "warning")
    # Two full-history groups (the second windowed past max_seq_len) share a
    # floor of 10 blocks; the first gets 8, the second 16: a small quota, not
    # a misordered list.
    MambaHybridCacheManagerV2._check_pool_ratio_floors(
        ratio=[0.5, 0.5],
        block_bytes=[100, 50],
        keys=[(None, 0, None), (1100, 0, None)],
        full_history=[True, True],
        names=["attention", "attention"],
        positions=[0, 1],
        draft_group=None,
        gpu_quota=1600,
        tokens_per_block=128,
        groups="0: attention, 1: attention",
    )

    (warning,) = warnings
    assert "layer group 0 (attention, pool_ratio[0] = 0.5000) gets 8 blocks" in warning
    assert "floor of 10" in warning


def test_layer_group_names_are_recorded_without_a_pool_ratio():
    manager, config = _k3_pool_ratio_manager_and_config(None)

    assert manager._with_draft_pool_ratio(config) is config
    assert manager._layer_group_names == ["ssm", "fp4_mla", "fp8_mla", "draft", "mla_hp_tail"]
    assert manager._pool_ratio_groups == []


def test_pool_group_slots_are_logged_per_cache_level(monkeypatch):
    infos = _capture_log(monkeypatch, "info")
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager._layer_group_names = ["ssm", "fp4_mla", "fp8_mla"]
    level_stats = [
        [
            SimpleNamespace(total=100, slot_sizes=[1000, 24]),
            SimpleNamespace(total=7, slot_sizes=[64]),
        ],
        [SimpleNamespace(total=3, slot_sizes=[2048])],
    ]
    level_pool_groups = [[1, 0, 0], [0, 0, 0, 0]]
    manager.impl = SimpleNamespace(
        cache_tier_list=["GPU_MEM", "HOST_MEM"],
        get_life_cycle_pool_group_indices=lambda level: level_pool_groups[level],
    )
    monkeypatch.setattr(manager, "_get_storage_statistics", lambda level: level_stats[level])

    manager._log_pool_group_slots()

    gpu, host = infos
    assert gpu.startswith("KV cache slots by pool group, cache level 0 (")
    assert "0: fp4_mla+fp8_mla num_slots=100 slot_bytes=1024 bytes=102400" in gpu
    assert "1: ssm num_slots=7 slot_bytes=64 bytes=448" in gpu
    assert "0: ssm+fp4_mla+fp8_mla+layer_group_3 num_slots=3 slot_bytes=2048" in host


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


@pytest.mark.parametrize(
    "backend, tokens_per_block, reads",
    [
        ("CUTEDSL", 128, True),
        ("CUTEDSL", 64, True),
        ("CUTEDSL", 256, False),
        ("TRTLLM", 64, True),
        ("TRTLLM", 128, False),
        ("VANILLA", 128, True),
    ],
)
def test_mla_dspark_drafter_reads_manager_pages(backend, tokens_per_block, reads):
    from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM

    drafter = object.__new__(MLADSparkForCausalLM)
    drafter.dflash_attention_backend = backend

    assert drafter.reads_managed_ctx_pages(tokens_per_block) is reads


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


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_helix_shard_partitions_positions_by_page(cp_size):
    from tensorrt_llm._torch.speculative.dflash_attention import HelixDraftShard

    page = 4
    positions = torch.arange(3 * page * cp_size + 3)
    for cp_rank in range(cp_size):
        shard = HelixDraftShard(cp_size, cp_rank, page)
        local, owned = shard.local_slots(positions)
        mine = positions[(positions // page) % cp_size == cp_rank]
        assert torch.equal(positions[owned], mine)
        # A rank's positions fill its local slots in order, without gaps.
        assert torch.equal(local[owned], torch.arange(len(mine)))
        for length in range(len(positions) + 1):
            expected = int((mine < length).sum())
            assert int(shard.local_lengths(torch.tensor(length))) == expected


def test_helix_block_fixup_is_rank_local():
    from tensorrt_llm._torch.models.modeling_dspark import _build_mla_block_fixup
    from tensorrt_llm._torch.speculative.dflash_attention import HelixDraftShard

    page, block_size = 4, 3
    shard = HelixDraftShard(2, 1, page)
    ctx_len = torch.tensor([6, 1])
    table = torch.tensor([[7, 8, 9], [4, 5, 6]])

    fixup = _build_mla_block_fixup(ctx_len, table, block_size, page, shard)

    # Request 0: positions 6, 7 (global page 1, rank 1's local page 0) and 8
    # (rank 0's); request 1: positions 1-3, all on rank 0's page 0.
    assert fixup.owned.view(2, block_size).tolist() == [[True, True, False], [False] * 3]
    assert fixup.pages[0, :2].tolist() == [7, 7]
    assert fixup.offsets[0, :2].tolist() == [2, 3]
    assert fixup.seq_lens_i32.tolist() == [4, 0]
    assert fixup.kv_bounds_i32.tolist() == [4] * block_size + [0] * block_size
