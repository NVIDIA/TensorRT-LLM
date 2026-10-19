# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep8 target's replicated embedding and step-fold paths: what the shell installs when.

Host-only. ``replicated_embedding`` is installed by the shell's
``load_weights`` after the built-in load (and after the replicated experts);
``verify_fold`` / ``path_walk_tail`` by its drafter load hooks
after ``draft_head`` and ``drafter_worker``. Without them none of these is imported or
installed. The re-classed embedding serves only small integer calls from the
table and hands every other call to the built-in forward.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    runtime as _runtime,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    spec_worker as drafter_worker,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.draft_head import (  # noqa: E501
    head as draft_head,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_block_decode import (  # noqa: E501
    block_decode as draft,
)
from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
    MuseSparkOpenOmniForConditionalGeneration,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _no_runtime_hooks(monkeypatch):
    """The executor hooks pin host staging blocks; these shells are built without them."""
    monkeypatch.setattr(_runtime, "make_runtime_hooks", lambda model: None)


_SHELL = modeling.ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep
_PACKAGE = modeling.__name__.rsplit(".", 1)[0]
_SERVED = (
    "preattn_prep",
    "drafter_block_conv",
    "drafter_block_decode",
    "draft_head",
    "post_attention_hop",
    "projection_gathers",
    "replicated_experts",
    "swa_paged_decode",
    "dense_gemm_prefetch",
    "drafter_fc_shard",
    "fused_accept",
    "drafter_fixed_gemms",
)


# --------------------------------------------------------------------------
# replicated_embedding at weight load
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "paths, expected",
    [
        ((), ["load"]),
        (("preattn_prep", "projection_gathers"), ["load"]),
        (("replicated_embedding",), ["load", "replicated_embedding"]),
        (
            ("replicated_experts", "replicated_embedding"),
            ["load", "ring_buddy", "replicated_embedding"],
        ),
        (("replicated_experts",), ["load", "ring_buddy"]),
    ],
)
def test_load_weights_installs_replicated_embedding_after_the_builtin_load_and_ring_buddy(
    monkeypatch, paths, expected
):
    calls = []
    ring = types.ModuleType("ring_buddy")
    ring.BLOCK, ring.STATIC_FROM = 32, 28
    ring.install = lambda *a, **k: calls.append("ring_buddy")
    replicated = types.ModuleType("replicated_embedding")
    replicated.install = lambda text_model, mapping: calls.append("replicated_embedding")
    for parent, name, module in (
        ("kernels.replicated_experts", "ring_buddy", ring),
        ("kernels.replicated_embedding", "embedding", replicated),
    ):
        monkeypatch.setitem(sys.modules, f"{_PACKAGE}.{parent}.{name}", module)
        monkeypatch.setattr(
            importlib.import_module(f"{_PACKAGE}.{parent}"), name, module, raising=False
        )
    monkeypatch.setattr(
        modeling._weights, "load", lambda model, weights, mapper: calls.append("load") or "loaded"
    )
    model = object.__new__(_SHELL)
    model.__dict__["_plan"] = dispatch.plan_of(*paths)
    model.__dict__["llm"] = SimpleNamespace(model=object())
    model.__dict__["model_config"] = SimpleNamespace(mapping=object())
    assert _SHELL.load_weights(model, {}) == "loaded"
    assert calls == expected


# --------------------------------------------------------------------------
# verify_fold / path_walk_tail at drafter load
# --------------------------------------------------------------------------
@pytest.fixture
def build(monkeypatch):
    """Shells constructed through their own constructor with the built-in one stubbed."""
    events = []
    drafter, worker = nn.Module(), object()

    def builtin_init(self, model_config, *args, **kwargs):
        if hasattr(self, "llm"):
            return
        nn.Module.__init__(self)
        self.llm = SimpleNamespace(model=nn.Module(), draft_model=drafter, spec_worker=worker)

    monkeypatch.setattr(MuseSparkOpenOmniForConditionalGeneration, "__init__", builtin_init)
    monkeypatch.setattr(
        MuseSparkOpenOmniForConditionalGeneration,
        "load_draft_weights",
        lambda self, *a, **k: events.append("built-in draft load") or "loaded",
    )
    monkeypatch.setattr(
        modeling.MuseSparkOpenTepModelingV2,
        "adopt",
        classmethod(lambda cls, model, plan=dispatch.NO_PATHS, compute_capability=None: None),
    )
    monkeypatch.setattr(draft, "install", lambda *a: events.append("drafter_blocks"))
    monkeypatch.setattr(draft_head, "install", lambda *a: events.append("draft_head"))
    monkeypatch.setattr(drafter_worker, "install", lambda *a: events.append("drafter_worker"))
    fold = types.ModuleType("spec_loop_fold")
    fold.install = lambda spec_worker, verify, walk: events.append(
        ("verify_fold", spec_worker, verify, walk)
    )
    monkeypatch.setitem(sys.modules, f"{_PACKAGE}.kernels.verify_fold.fold", fold)
    monkeypatch.setattr(
        importlib.import_module(f"{_PACKAGE}.kernels.verify_fold"), "fold", fold, raising=False
    )

    def _build(*paths):
        plan = dispatch.plan_of(*paths)
        monkeypatch.setattr(modeling._dispatch, "compose", lambda *args, **kwargs: plan)
        events.clear()
        return _SHELL(model_config=SimpleNamespace(mapping="the mapping"))

    _build.events = events
    _build.worker = worker
    return _build


def test_verify_fold_installs_last_on_the_worker(build):
    shell = build(*_SERVED, "verify_fold", "path_walk_tail")
    assert shell.load_draft_weights("weights") == "loaded"
    assert build.events == [
        "built-in draft load",
        "drafter_blocks",
        "draft_head",
        "drafter_worker",
        ("verify_fold", build.worker, True, True),
    ]


def test_without_a_verify_fold_id_the_fold_is_not_installed(build):
    shell = build(*_SERVED)
    shell.load_draft_weights("weights")
    assert build.events == ["built-in draft load", "drafter_blocks", "draft_head", "drafter_worker"]


def test_replicated_embedding_alone_installs_nothing_on_the_worker(build):
    shell = build("replicated_embedding")
    shell.load_draft_weights("weights")
    assert build.events == ["built-in draft load"]


# --------------------------------------------------------------------------
# The re-classed embedding's dispatch
# --------------------------------------------------------------------------
def test_the_replicated_embedding_serves_small_integer_calls_only(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_embedding import (  # noqa: E501
        embedding as replicated_embedding,
    )
    from tensorrt_llm._torch.modules.embedding import Embedding

    served, builtin = [], []
    monkeypatch.setattr(
        replicated_embedding, "replicated_embedding_gather", lambda t, i: served.append(i) or "rows"
    )
    monkeypatch.setattr(Embedding, "forward", lambda self, i: builtin.append(i) or "built-in")
    module = nn.Module.__new__(replicated_embedding.ReplicatedEmbedding)
    nn.Module.__init__(module)
    # A host table here, so host ids are "on the table's device".
    module.__dict__[replicated_embedding.TABLE_ATTR] = torch.zeros(4, 2, dtype=torch.bfloat16)

    def ids(n, dtype=torch.int32, device="cpu"):
        return torch.zeros(n, dtype=dtype, device=device)

    assert replicated_embedding.MAX_ROWS == 576
    assert module.forward(ids(8)) == "rows"
    assert module.forward(ids(16, torch.int64)) == "rows"
    assert module.forward(ids(512)) == "rows"
    assert module.forward(ids(576, torch.int64)) == "rows"
    assert module.forward(ids(577)) == "built-in"
    assert module.forward(ids(0)) == "built-in"
    assert module.forward(ids(8, device="meta")) == "built-in"
    assert module.forward(ids(8, torch.float32)) == "built-in"
    assert module.forward(ids(8, torch.bool)) == "built-in"
    del module.__dict__[replicated_embedding.TABLE_ATTR]
    assert module.forward(ids(8)) == "built-in"
    assert len(served) == 4 and len(builtin) == 6


def _bf16(bits):
    """A bf16 tensor holding the given 16-bit patterns."""
    return torch.tensor([b - 0x10000 if b >= 0x8000 else b for b in bits], dtype=torch.int16).view(
        torch.bfloat16
    )


def test_the_replicated_embedding_value_census_counts_subnormal_non_finite_and_negative_zero():
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_embedding import (  # noqa: E501
        embedding as replicated_embedding,
    )

    bits = [
        0x0001,  # smallest subnormal
        0x807F,  # largest negative subnormal
        0x7FC0,  # NaN
        0xFFC1,  # negative NaN with a payload
        0x7F80,  # +inf
        0xFF80,  # -inf
        0x8000,  # -0.0
        0x0000,  # +0.0
        0x3F80,  # 1.0
        0x0080,  # smallest normal
        0x7F7F,  # largest finite
    ]
    table = _bf16(bits * 3).view(3, len(bits))
    assert replicated_embedding._value_census(table) == (2 * 3, 4 * 3, 1 * 3)


@pytest.mark.parametrize(
    "bits, refusal",
    [
        (0x0001, "subnormal"),
        (0x7FC0, "NaN or infinite"),
        (0xFFC1, "NaN or infinite"),
        (0x7F80, "NaN or infinite"),
        (0xFF80, "NaN or infinite"),
        (0x8000, None),
        (0x3F80, None),
    ],
)
def test_replicated_embedding_refuses_a_table_whose_all_reduce_bits_a_gather_cannot_reproduce(
    monkeypatch, bits, refusal
):
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
        dispatch as tp_agreement,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_embedding import (  # noqa: E501
        embedding as replicated_embedding,
    )
    from tensorrt_llm._torch.modules.embedding import Embedding

    class _BothRanksAlike:
        """The TP group of two: the peer gathered the same table, so it reports the same."""

        def get(self, mapping):
            return self

        def tp_allgather(self, obj, small_payload=False):
            return [obj, obj]

    monkeypatch.setattr(tp_agreement, "Distributed", _BothRanksAlike())
    for name in ("synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (0, 0))
    monkeypatch.setattr(replicated_embedding, "unsupported_reason", lambda embedding, mapping: None)
    shard = torch.full((2, 4), 0.5, dtype=torch.bfloat16)
    other = shard.clone()
    other.view(torch.int16)[1, 2] = _bf16([bits]).view(torch.int16)[0]
    table = torch.cat([shard, other])  # rank 0's shard, then rank 1's
    monkeypatch.setattr(replicated_embedding, "_gather_table", lambda shard, mapping, out: table)
    embedding = nn.Module.__new__(Embedding)
    nn.Module.__init__(embedding)
    embedding.weight = nn.Parameter(shard, requires_grad=False)
    text_model = SimpleNamespace(embed_tokens=embedding)
    mapping = SimpleNamespace(tp_rank=0, tp_size=2)
    if refusal is None:
        replicated_embedding.install(text_model, mapping)
        assert type(embedding) is replicated_embedding.ReplicatedEmbedding
        assert replicated_embedding.table_of(embedding) is table
        return
    with pytest.raises(ValueError, match=refusal):
        replicated_embedding.install(text_model, mapping)
    assert type(embedding) is Embedding
    assert replicated_embedding.TABLE_ATTR not in embedding.__dict__


@pytest.mark.parametrize("stage", ["layout", "table"])
def test_replicated_embedding_agrees_each_refusal_over_the_tp_group(monkeypatch, stage):
    # This rank could install; one peer refuses (its layout before the table's gather, or
    # its rows of the gathered table after it): every rank raises the peer's reason, the
    # layout refusal before the gather (a collective), the table refusal before the module
    # is re-classed.
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
        dispatch as tp_agreement,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_embedding import (  # noqa: E501
        embedding as replicated_embedding,
    )
    from tensorrt_llm._torch.modules.embedding import Embedding

    peer = {
        "layout": "[musespark_open] replicated_embedding cannot run here: this rank's shard is not rows "
        "[rank * width, (rank + 1) * width)",
        "table": "[musespark_open] replicated_embedding cannot run here: the gathered table does not hold "
        "this rank's shard at its rows",
    }[stage]
    refusing_call = 1 if stage == "layout" else 2

    class _Group:
        def __init__(self):
            self.gathered = []

        def get(self, mapping):
            return self

        def tp_allgather(self, obj, small_payload=False):
            self.gathered.append(obj)
            last = peer if len(self.gathered) == refusing_call else None
            return [obj] + [None] * 6 + [last]

    group = _Group()
    monkeypatch.setattr(tp_agreement, "Distributed", group)
    for name in ("synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (0, 0))
    monkeypatch.setattr(replicated_embedding, "unsupported_reason", lambda embedding, mapping: None)
    shard = torch.full((2, 4), 0.5, dtype=torch.bfloat16)
    gathers = []

    def gather_table(shard, mapping, out):
        gathers.append(mapping)
        return torch.cat([shard] * 8)

    monkeypatch.setattr(replicated_embedding, "_gather_table", gather_table)
    embedding = nn.Module.__new__(Embedding)
    nn.Module.__init__(embedding)
    embedding.weight = nn.Parameter(shard, requires_grad=False)
    text_model = SimpleNamespace(embed_tokens=embedding)
    with pytest.raises(ValueError) as refused:
        replicated_embedding.install(text_model, SimpleNamespace(tp_rank=0, tp_size=8))
    assert str(refused.value) == f"{peer} (on TP ranks [7] of 8)"
    assert len(gathers) == refusing_call - 1
    assert group.gathered == [None] * refusing_call
    assert type(embedding) is Embedding
    assert replicated_embedding.TABLE_ATTR not in embedding.__dict__
