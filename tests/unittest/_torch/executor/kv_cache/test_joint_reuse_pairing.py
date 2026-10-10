# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for the target/draft joint-reuse pairing predicate."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
    DeepseekV4CacheManager,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import DSACacheManagerV2
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa.cache_manager import (
    QSAMambaHybridCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2

pytestmark = pytest.mark.cpu_only


def _make_creator(manager_cls, pp_size: int = 1) -> KvCacheCreator:
    """Minimal creator exercising only ``_joint_reuse_supported``."""
    c = object.__new__(KvCacheCreator)
    c._is_kv_cache_manager_v2 = True
    c._kv_cache_manager_cls = manager_cls
    c._mapping = SimpleNamespace(pp_size=pp_size)
    c._speculative_config = SimpleNamespace(max_draft_len=3)
    return c


@pytest.fixture
def pairs(monkeypatch):
    """Make ``draft_prompt_lookahead`` report a paired-capable mode."""
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor._util.draft_prompt_lookahead",
        lambda _cfg: 1,
    )


@pytest.mark.parametrize(
    "manager_cls,expects_pairing",
    [
        pytest.param(KVCacheManagerV2, True, id="base"),
        pytest.param(DSACacheManagerV2, True, id="dsa"),
        pytest.param(MiniMaxM3KVCacheManagerV2, True, id="minimax_m3"),
        pytest.param(MambaHybridCacheManagerV2, True, id="mamba_hybrid"),
        pytest.param(DeepseekV4CacheManager, False, id="deepseek_v4"),
        pytest.param(QSAMambaHybridCacheManagerV2, False, id="qsa"),
    ],
)
def test_every_v2_manager_pins_its_pairing_decision(manager_cls, expects_pairing, pairs):
    """No subclass may inherit -- or lose -- pairing silently.

    The decision reads the *draft* attribute: a separate one-model draft pool
    holds only the speculation layers, so it can support the reuse-match-backoff
    protocol when the pool holding the model's own layers cannot. That is why
    the hybrid Mamba manager pairs despite declining the protocol itself.
    """
    assert _make_creator(manager_cls)._joint_reuse_supported() is expects_pairing


def test_v1_kv_cache_manager_never_pairs(pairs):
    """The V2 gate short-circuits before any manager attribute is consulted."""
    creator = _make_creator(KVCacheManagerV2)
    creator._is_kv_cache_manager_v2 = False
    assert creator._joint_reuse_supported() is False


def test_mode_without_prompt_lookahead_never_pairs(monkeypatch):
    """Pairing needs a draft prompt lookahead, however supportive the pool."""
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor._util.draft_prompt_lookahead",
        lambda _cfg: None,
    )
    assert _make_creator(KVCacheManagerV2)._joint_reuse_supported() is False


def test_pairing_is_independent_of_pipeline_parallelism(pairs):
    """PP must not gate pairing.

    ``get_layer_masks`` returns an all-False mamba mask for any draft pool with
    draft layers, so ``local_num_mamba_layers`` is 0 on every rank -- including
    a non-last rank whose ``pp_layers`` fell back to layer 0. Gating on pp_size
    would only strip the optimization from non-Mamba managers.
    """
    assert _make_creator(KVCacheManagerV2, pp_size=4)._joint_reuse_supported() is True
