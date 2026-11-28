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
its overrides of the shared KDA hybrid cache manager, its draft layers, and the family KDA
metadata's snapshot plan.

The K3.1 replay state keeps the replay caches in the recurrent-state slot, so every KDA layer
stays in one layer group. CPU tests: the managers carry only the attributes the overrides read.
"""

import functools
from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = [".", "../.."]
from _k31_utils import cpu_or_sm
from _kda_ops_util import bare  # noqa: E402 — needs the path declared above

pytest.importorskip("fla")  # the shared kimi_kda package imports fla at load time

from tensorrt_llm._torch.attention.backends.sparse.dsa.params import DSAParams
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm import (
    rubin_kda_dsa_cache_manager as dsa_cache_manager,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm import rubin_kda_metadata
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_cache_manager import (
    K31KDAReplayState,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_dsa_cache_manager import (
    RubinKdaDsaHybridCacheManager,
)
from tensorrt_llm._torch.modules.kimi_kda import cache_manager as shared_cache_manager
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    MambaRole,
    MambaStateLayout,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, LayerId, SsmLayerConfig

# CPU test (l0_cpu); a GPU host other than sm_107 skips it.
pytestmark = [pytest.mark.cpu_only, cpu_or_sm(107)]

_NUM_SPEC = 2
_SECTION = 8
_WINDOW = 3  # committed convolution window (kernel size 4)
_SLOTS = 4
_LAYERS = 2


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


def test_attention_bytes_per_token_include_the_indexer_k(monkeypatch) -> None:
    """The attention cache's bytes per token are the model layers' ``Role.ALL`` bytes, which hold
    the indexer K of the MLA layers, plus the standalone draft layers' bytes, with a unified
    standalone draft cache."""
    roles = dsa_cache_manager.Role
    mla_layers, num_target_layers, num_draft_layers = (1, 3), 4, 3  # layers 0 and 2 are KDA layers
    key_bytes, indexer_bytes, draft_bytes = 576, 132, 576
    num_layers = num_target_layers + num_draft_layers

    def base_layer_bytes(self, local_layer_idx, data_role):
        if local_layer_idx >= num_target_layers:
            return draft_bytes
        return key_bytes if local_layer_idx in mla_layers else 0

    monkeypatch.setattr(
        shared_cache_manager.KimiK3HybridCacheManager,
        "get_layer_bytes_per_token",
        base_layer_bytes,
        raising=False,
    )
    manager = _manager(
        pp_layers=list(range(num_layers)),
        num_local_layers=num_layers,
        _indexer_layer_ids=frozenset(mla_layers),
        _indexer_bytes_per_token=lambda: indexer_bytes,
        _draft_layer_sizes=[draft_bytes] * num_draft_layers,
        _unified_draft_layer_specs=dict.fromkeys(range(num_target_layers, num_layers)),
        num_kv_heads_per_layer=[1] * num_layers,
        head_dim_per_layer=[key_bytes] * num_layers,
    )
    all_bytes = sum(manager.get_layer_bytes_per_token(i, roles.ALL) for i in range(num_layers))
    assert (
        all_bytes == len(mla_layers) * (key_bytes + indexer_bytes) + num_draft_layers * draft_bytes
    )
    assert [
        manager.get_layer_bytes_per_token(i, roles.INDEX_KEY) for i in range(num_target_layers)
    ] == [
        0,
        indexer_bytes,
        0,
        indexer_bytes,
    ]
    assert manager._attention_cache_bytes_per_token() == all_bytes


def test_generation_server_builds_the_k31_replay_state() -> None:
    manager = _generation_manager()
    state = manager._initialize_spec_state()
    assert type(state) is K31KDAReplayState and state.replay_shares_state_slot
    assert manager._kda_replay is state
    assert state.num_speculative_tokens == _NUM_SPEC


def _kda_state_layout(layers=(0, 1)) -> MambaStateLayout:
    """The recurrent-state layout of two local KDA layers with two draft tokens."""
    return MambaStateLayout(
        layer_mask=(True,) * len(layers),
        pp_layers=layers,
        mamba_pp_layers=layers,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        conv_state_shape=(3 * _SECTION, _WINDOW),
        ssm_state_shape=(2, 4, 4),
        conv_state_dtype=torch.float32,
        ssm_state_dtype=torch.float32,
        n_groups_per_rank=2,
        max_batch_size=3,
        state_index_capacity=_SLOTS,
        slot_capacity=_SLOTS + 1,
        spec_config=SimpleNamespace(tokens_per_gen_step=_NUM_SPEC + 1),
        conv_section_dims=(_SECTION,) * 3,
        conv_state_layout="q_k_v",
    )


@pytest.mark.parametrize(
    "replay_state, one_group",
    [(lambda: _generation_manager()._initialize_spec_state(), True)],
    ids=["manager_replay_state"],
)
def test_the_replay_keeps_every_kda_layer_in_one_layer_group(replay_state, one_group) -> None:
    """KV cache manager V2 puts each layer in one layer group: a layer whose buffers carry two
    reuse policies fails the page-table setup ("Layer N has buffers in 2 layer groups"). The
    manager's replay state keeps the replay caches in the recurrent-state slot, as the Rubin verify
    kernels address them: they are buffers of each KDA layer's SSM config, and no PRIVATE
    replay-state region lands on the KDA layers."""
    from tensorrt_llm._torch.modules.kimi_kda.replay_region import KDAReplayRegion
    from tensorrt_llm._torch.pyexecutor.kv_cache.regions import lay_out_regions

    state = replay_state()
    layout = _kda_state_layout()
    layers = layout.mamba_pp_layers
    region = KDAReplayRegion(state, layout, {layer: layer for layer in layers}, _SLOTS, 0.8)
    configs = [
        SsmLayerConfig(
            layer_id=LayerId(layer),
            buffers=[
                BufferConfig(role=MambaRole.SSM_STATE, size=128),
                BufferConfig(role=MambaRole.CONV_STATE, size=64),
                *region.shared_buffers(layer),
            ],
        )
        for layer in layers
    ]
    region_configs, _ = lay_out_regions(region.declare_regions(), configs)

    policies = {
        layer: {
            buffer.reuse_policy if buffer.reuse_policy is not None else config.reuse_policy
            for buffer in config.buffers
        }
        for layer, config in zip(layers, configs)
    }
    for region_config in region_configs:
        for buffer in region_config.buffers:
            policies[int(buffer.layer_id)].add(region_config.reuse_policy)
    assert all(len(layer_policies) == 1 for layer_policies in policies.values()) == one_group
    if one_group:
        assert region_configs == []
        in_slot = {buffer.role for config in configs for buffer in config.buffers}
        assert region.roles and region.roles <= in_slot


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
