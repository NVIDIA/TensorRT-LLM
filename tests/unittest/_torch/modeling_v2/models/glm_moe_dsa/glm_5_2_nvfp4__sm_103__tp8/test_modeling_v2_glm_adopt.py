# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.2 tp8 target's class swaps side by side: the eight built-in classes the target
adopts, what each adoption refuses, and the state the target classes may declare.

Host-only: the objects are created without their constructors. What each adoption attaches is
covered by the layer, attention and speculative-worker tests of this target.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
    attention,
    layers,
    spec_worker,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.indexer import DeepseekV4Indexer
from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import DSATrtllmAttention
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer
from tensorrt_llm._torch.attention.backends.sparse.dsa.module import DSAHooks
from tensorrt_llm._torch.models.modeling_deepseekv3 import (
    DeepseekV3DecoderLayer,
    Deepseekv3MoE,
    DeepseekV3MTP,
    DeepseekV32Attention,
)
from tensorrt_llm._torch.models.modeling_exaone_moe import ExaoneMoeSparseMoEBlock
from tensorrt_llm._torch.speculative.eagle3 import MTPEagleWorker
from tensorrt_llm._torch.speculative.mtp_dynamic_tree import MTPEagleDynamicTreeWorker

pytestmark = pytest.mark.cpu_only

# The target and the family code it adopts with (``shared``).
TARGET_PACKAGE = layers.__name__.rpartition(".")[0].rpartition(".")[0]


def _adopt_moe(moe):
    return layers.GlmMoE.adopt(moe, None, 0)


# (target class, the built-in class it adopts, the adoption)
SWAPS = [
    (attention.GlmMLA, DeepseekV32Attention, attention.GlmMLA.adopt),
    (attention.GlmDsaHooks, DSAHooks, attention.GlmDsaHooks.adopt),
    (attention.GlmDsaBackend, DSATrtllmAttention, attention.GlmDsaBackend.adopt),
    (attention.GlmIndexer, Indexer, attention.GlmIndexer.adopt),
    (layers.GlmDecoderLayer, DeepseekV3DecoderLayer, layers.GlmDecoderLayer.adopt),
    (layers.GlmMtpLayer, DeepseekV3MTP, layers.GlmMtpLayer.adopt),
    (layers.GlmMoE, Deepseekv3MoE, _adopt_moe),
    (spec_worker.GlmMtpWorker, MTPEagleWorker, spec_worker.GlmMtpWorker.adopt),
]
SWAP_IDS = [cls.__name__ for cls, _, _ in SWAPS]

# Every attribute the target's code reads that adoption may leave unset, with its default.
CLASS_DEFAULTS = {
    layers.GlmDecoderLayer: {
        "post_attn_hop": None,
        "_into_hop": None,
        "_hop_routing": None,
        "post_moe_hop": None,
        "deferred_norm_hop": None,
        "_expert_prefetch_link": None,
    },
    layers.GlmMtpLayer: {
        "post_attn_hop": None,
        "_into_hop": None,
        "_hop_routing": None,
        "post_moe_hop": None,
        "deferred_norm_hop": None,
        "_expert_prefetch_link": None,
        "draft_front": None,
        "draft_live_row_hop": None,
        "_draft_live_rows": None,
        "_draft_live_rows_compacted": False,
    },
    layers.GlmMoE: {"router_gemm": None, "_routed_into_hop": None},
    attention.GlmMLA: {"_attn_tail": None, "_tail_prefetch": False},
    attention.GlmIndexer: {"_topk_bridge": None},
    spec_worker.GlmMtpWorker: {"_draft_loop_ops": None},
}


def _bare(cls):
    obj = cls.__new__(cls)
    if isinstance(obj, nn.Module):
        nn.Module.__init__(obj)
    return obj


def _target_classes(cls):
    return [klass for klass in cls.__mro__ if klass.__module__.startswith(TARGET_PACKAGE)]


@pytest.mark.parametrize("cls, builtin, adopt", SWAPS, ids=SWAP_IDS)
def test_each_target_class_extends_only_its_builtin_class(cls, builtin, adopt):
    assert issubclass(cls, builtin)
    added = [klass for klass in cls.__mro__ if klass not in builtin.__mro__]
    assert added == _target_classes(cls), added


@pytest.mark.parametrize("cls, builtin, adopt", SWAPS, ids=SWAP_IDS)
def test_adoption_refuses_a_subclass_of_the_builtin_class(cls, builtin, adopt):
    subclass = type(f"_Derived{builtin.__name__}", (builtin,), {})
    obj = _bare(subclass)
    with pytest.raises(TypeError, match=subclass.__name__):
        adopt(obj)
    assert type(obj) is subclass


@pytest.mark.parametrize("cls, builtin, adopt", SWAPS, ids=SWAP_IDS)
def test_adoption_refuses_an_adopted_object(cls, builtin, adopt):
    obj = _bare(cls)
    with pytest.raises(TypeError, match=cls.__name__):
        adopt(obj)
    assert type(obj) is cls


@pytest.mark.parametrize(
    "upstream, adopt",
    [
        (ExaoneMoeSparseMoEBlock, _adopt_moe),
        (DeepseekV4Indexer, attention.GlmIndexer.adopt),
        (MTPEagleDynamicTreeWorker, spec_worker.GlmMtpWorker.adopt),
    ],
    ids=lambda value: getattr(value, "__name__", None),
)
def test_adoption_refuses_the_upstream_subclasses(upstream, adopt):
    obj = _bare(upstream)
    with pytest.raises(TypeError, match=upstream.__name__):
        adopt(obj)
    assert type(obj) is upstream


@pytest.mark.parametrize("cls, builtin, adopt", SWAPS, ids=SWAP_IDS)
def test_the_target_classes_keep_the_builtin_layout(cls, builtin, adopt):
    for klass in _target_classes(cls):
        own = vars(klass)
        assert "__init__" not in own and "__new__" not in own and "__slots__" not in own, klass
        for name, value in own.items():
            assert not isinstance(value, (torch.Tensor, nn.Module)), (klass, name)


@pytest.mark.parametrize("cls", list(CLASS_DEFAULTS), ids=lambda cls: cls.__name__)
def test_the_class_level_defaults(cls):
    builtin = next(builtin for target, builtin, _ in SWAPS if target is cls)
    for name, default in CLASS_DEFAULTS[cls].items():
        assert getattr(cls, name) is default, name
        assert not hasattr(builtin, name), name
