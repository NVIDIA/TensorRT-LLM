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
"""Mamba state-cache sizing when B/C groups replicate across TP ranks.

With ``tp_size > n_groups`` a group's B/C rows live on every rank whose heads
belong to it, so the per-rank conv state is *not* ``conv_dim // tp_size``: the
``d_inner`` section is sharded while the two grouped sections keep their full
``tp_ngroups * d_state`` width. All sizes must therefore come from
:class:`Mamba2TpShard` rather than from plain integer division.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.mamba.mamba2_tp import Mamba2TpShard
from tensorrt_llm._torch.pyexecutor._util import _create_kv_cache_manager
from tensorrt_llm._torch.pyexecutor.config_utils import MambaKVCacheParams
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    CppMambaHybridCacheManager,
    MambaHybridCacheManagerV2,
    MixedMambaHybridCacheManager,
    PythonMambaCacheManager,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# Model used by the Python-manager tests: 8 heads x 16 channels grouped into
# 2 B/C groups. tp=4 replicates each group over 2 ranks, tp=2 splits them.
_NUM_HEADS = 8
_HEAD_DIM = 16
_N_GROUPS = 2
_D_STATE = 8
_D_CONV = 4


def _python_manager(tp_size: int, rank: int, **extra) -> PythonMambaCacheManager:
    return PythonMambaCacheManager(
        d_state=_D_STATE,
        d_conv=_D_CONV,
        num_heads=_NUM_HEADS,
        n_groups=_N_GROUPS,
        head_dim=_HEAD_DIM,
        num_layers=1,
        max_batch_size=2,
        spec_state_size=2,
        mapping=Mapping(world_size=tp_size, tp_size=tp_size, rank=rank),
        dtype=torch.float16,
        ssm_cache_dtype=torch.float16,
        **extra,
    )


@skip_no_cuda
@pytest.mark.parametrize("rank", range(4))
def test_python_manager_replicated_group_shapes(rank):
    """tp=4 > n_groups=2: each rank holds one whole group's B/C next to its heads."""
    shard = Mamba2TpShard(
        tp_size=4,
        nheads=_NUM_HEADS,
        n_groups=_N_GROUPS,
        head_dim=_HEAD_DIM,
        d_state=_D_STATE,
    )
    # Guard the premise: this model/TP pair really is a replicated layout.
    assert shard.replicated and shard.tp_ngroups == 1

    mgr = _python_manager(tp_size=4, rank=rank)
    cache = mgr.mamba_layer_cache(0)

    # conv rows = tp_d_inner (32) + 2 * tp_grouped_state_dim (8), NOT 144 // 4.
    assert tuple(cache.conv.shape[1:]) == (48, 3)
    assert tuple(cache.temporal.shape[1:]) == (2, 16, 8)
    assert mgr.conv_section_dims == [32, 8, 8]


@skip_no_cuda
def test_python_manager_rejects_disagg_with_replicated_groups():
    """The Python state pool is the third disaggregated route (Mixed manager,
    Python transceiver): `disaggregation/resource/kv_extractor.py` reads
    `conv_section_dims` off it, and a TP-mismatched transfer would treat the
    replicated B/C sections as shards."""
    with pytest.raises(ValueError, match="replicated groups"):
        _python_manager(tp_size=4, rank=0, is_disagg=True)


@skip_no_cuda
def test_python_manager_allows_replicated_groups_without_disagg():
    """Aggregate serving never moves state across ranks, so it is unaffected."""
    mgr = _python_manager(tp_size=4, rank=0, is_disagg=False)

    assert mgr.conv_section_dims == [32, 8, 8]


@skip_no_cuda
def test_python_manager_allows_disagg_without_replication():
    """tp=2 <= n_groups=2 shards every conv section, which disagg can express."""
    mgr = _python_manager(tp_size=2, rank=0, is_disagg=True)

    assert mgr.conv_section_dims == [64, 8, 8]


@skip_no_cuda
def test_python_manager_even_groups_unchanged():
    """tp=2 <= n_groups=2 keeps the historical `// tp_size` layout bit-identical."""
    mgr = _python_manager(tp_size=2, rank=0)
    cache = mgr.mamba_layer_cache(0)

    assert tuple(cache.conv.shape[1:]) == (80, 3)
    assert tuple(cache.temporal.shape[1:]) == (4, 16, 8)
    assert mgr.conv_section_dims == [64, 8, 8]


def _v2_manager(rank: int, **extra):
    """V2 hybrid manager for a model whose single B/C group replicates at tp=4."""
    return MambaHybridCacheManagerV2(
        16,  # mamba_d_state
        4,  # mamba_d_conv
        8,  # mamba_num_heads
        1,  # mamba_n_groups
        16,  # mamba_head_dim
        2,  # mamba_num_layers
        [True, True],  # mamba_layer_mask
        torch.float16,  # mamba_cache_dtype
        torch.float16,  # mamba_ssm_cache_dtype
        # max_tokens is only there to give the V2 page pool a quota; it does
        # not take part in the recurrent-state sizing under test.
        KvCacheConfig(max_tokens=512),
        CacheTypeCpp.SELF,
        num_layers=0,
        num_kv_heads=1,
        head_dim=16,
        tokens_per_block=32,
        max_seq_len=64,
        max_batch_size=1,
        mapping=Mapping(world_size=4, tp_size=4, rank=rank),
        conv_state_layout="x_b_c",
        **extra,
    )


@skip_no_cuda
@pytest.mark.parametrize("rank", range(4))
def test_v2_manager_replicated_group_shapes(rank):
    """The V2 hybrid manager sizes its recurrent pages from the same shard."""
    mgr = _v2_manager(rank)

    assert mgr.conv_state_shape == [64, 3]
    assert mgr.ssm_state_shape == [2, 16, 16]
    assert mgr._n_groups_per_rank == 1
    assert mgr.conv_section_dims == [32, 16, 16]


@skip_no_cuda
def test_v2_manager_rejects_disagg_with_replicated_groups():
    """Same layout as the test above, which the pre-replication guard (a bare
    `grouped_state_dim % tp_size` check) would have waved through: n_groups=1
    and d_state=16 divide tp=4 cleanly even though every group is replicated."""
    with pytest.raises(ValueError, match="replicated groups"):
        _v2_manager(0, is_disagg=True)


def test_states_bytes_per_layer_replicated():
    """The memory estimate must budget the replicated B/C rows, not conv_dim // tp."""
    params = MambaKVCacheParams(
        state_size=8,
        conv_kernel=4,
        num_heads=8,
        n_groups=1,
        head_dim=16,
        mamba_layer_mask=[True],
        target_full_attention_layer_mask=[False],
        num_mamba_layers=1,
        num_draft_layers=0,
        dtype=torch.float16,
        mamba_ssm_cache_dtype=None,
    )

    # tp=4 > n_groups=1: conv rows = tp_d_inner (32) + 2 * d_state (8).
    conv_bytes = (32 + 2 * 8) * 3 * 2
    ssm_bytes = 2 * 16 * 8 * 2
    mapping = Mapping(world_size=4, tp_size=4, rank=0)
    assert params.get_states_bytes_per_layer(mapping) == conv_bytes + ssm_bytes


def test_shard_rejects_illegal_tp_in_manager():
    """An illegal TP degree is rejected by the shard, with the legal list attached."""
    with pytest.raises(ValueError, match="Valid tp_size values"):
        _python_manager(tp_size=3, rank=0)


def _cpp_manager(is_disagg: bool):
    """Cpp hybrid manager for a model whose single B/C group replicates at tp=4.

    8 heads x 16 channels, 1 group, d_state 8 -> per rank 2 heads (32 channels)
    plus a full replica of the group's 8 B and 8 C rows.
    """
    return CppMambaHybridCacheManager(
        8,  # mamba_d_state
        4,  # mamba_d_conv
        8,  # mamba_num_heads
        1,  # mamba_n_groups
        16,  # mamba_head_dim
        1,  # mamba_num_layers
        [True],  # mamba_layer_mask
        torch.float16,  # mamba_cache_dtype
        torch.float16,  # mamba_ssm_cache_dtype
        KvCacheConfig(max_tokens=512),
        CacheTypeCpp.SELF,
        num_layers=0,
        num_kv_heads=1,
        head_dim=16,
        tokens_per_block=32,
        max_seq_len=64,
        max_batch_size=1,
        mapping=Mapping(world_size=4, tp_size=4, rank=0),
        is_disagg=is_disagg,
    )


@skip_no_cuda
def test_cpp_manager_rejects_disagg_with_replicated_groups():
    """The C++ TP-mismatch split rebuilds conv sections from the GLOBAL dims by
    dividing by tp_size, which is not the per-rank layout once B/C replicate."""
    with pytest.raises(ValueError, match="replicated groups"):
        _cpp_manager(is_disagg=True)


@skip_no_cuda
def test_cpp_manager_allows_replicated_groups_without_disagg():
    """Aggregate serving has no cross-rank state transfer, so it is unaffected."""
    mgr = _cpp_manager(is_disagg=False)

    assert mgr.conv_state_shape == [48, 3]
    assert mgr.ssm_state_shape == [2, 16, 8]


def _mixed_manager(is_disagg: bool):
    """Mixed hybrid manager over the same replicated model as `_cpp_manager`.

    Its Mamba side is `PythonMambaCacheManager`, the pool the disaggregated
    `kv_extractor` reads `conv_section_dims` from.
    """
    return MixedMambaHybridCacheManager(
        8,  # mamba_d_state
        4,  # mamba_d_conv
        8,  # mamba_num_heads
        1,  # mamba_n_groups
        16,  # mamba_head_dim
        1,  # mamba_num_layers
        [True],  # mamba_layer_mask
        torch.float16,  # mamba_cache_dtype
        torch.float16,  # mamba_ssm_cache_dtype
        # The Mixed manager keeps the two pools independent and asserts that
        # block reuse is off.
        KvCacheConfig(max_tokens=512, enable_block_reuse=False),
        CacheTypeCpp.SELF,
        num_layers=0,
        layer_mask=[False],
        num_kv_heads=1,
        head_dim=16,
        tokens_per_block=32,
        max_seq_len=64,
        max_batch_size=1,
        mapping=Mapping(world_size=4, tp_size=4, rank=0),
        is_disagg=is_disagg,
    )


@skip_no_cuda
def test_mixed_manager_rejects_disagg_with_replicated_groups():
    """The third disaggregated route: Python transceiver / MIXED preference."""
    with pytest.raises(ValueError, match="replicated groups"):
        _mixed_manager(is_disagg=True)


@skip_no_cuda
def test_mixed_manager_allows_replicated_groups_without_disagg():
    """Aggregate serving keeps working with the replicated layout."""
    mgr = _mixed_manager(is_disagg=False)

    assert mgr._impl.conv_section_dims == [32, 8, 8]


@pytest.mark.parametrize(
    "manager_cls",
    [CppMambaHybridCacheManager, MixedMambaHybridCacheManager],
    ids=["cpp", "mixed"],
)
def test_factory_forwards_is_disagg_to_hybrid_manager(monkeypatch, manager_cls):
    """The guard is worthless unless the factory actually hands the manager the
    flag; it used to be passed only to the V2 manager, then only to V2 and Cpp,
    leaving the Mixed manager (Python transceiver / TLLM_MAMBA_MANAGER_PREFERENCE
    =MIXED) able to publish replicated conv sections to a disagg transfer."""
    captured = {}

    class RecordingManager(manager_cls):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    config = SimpleNamespace(
        architectures=["NemotronHForCausalLM"],
        hybrid_override_pattern="M*",
        vocab_size=131072,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
    )
    mamba_params = MambaKVCacheParams(
        state_size=8,
        conv_kernel=4,
        num_heads=4,
        n_groups=1,
        head_dim=8,
        mamba_layer_mask=[True, False],
        target_full_attention_layer_mask=[False, True],
        num_mamba_layers=1,
        num_draft_layers=0,
        dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.bfloat16,
    )
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor._util.get_sm_version", lambda: 90)
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor._util.extract_mamba_kv_cache_params",
        lambda *args, **kwargs: mamba_params,
    )
    _create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=RecordingManager,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        kv_cache_config=KvCacheConfig(),
        tokens_per_block=32,
        max_seq_len=2048,
        max_batch_size=4,
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=256,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=SimpleNamespace(pretrained_config=config, quant_config=None),
        dtype=torch.bfloat16,
        is_draft=False,
        is_disagg=True,
    )

    assert captured["is_disagg"] is True
