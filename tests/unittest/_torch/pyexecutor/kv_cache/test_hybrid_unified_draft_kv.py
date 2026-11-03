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

from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.quantization.mode import QuantMode

pytestmark = pytest.mark.cpu_only

_NUM_TARGET_LAYERS = 6
_NUM_DRAFT_LAYERS = 2


class _K3LikeManager(MambaHybridCacheManagerV2):
    _accepts_appended_draft_layers = True


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
