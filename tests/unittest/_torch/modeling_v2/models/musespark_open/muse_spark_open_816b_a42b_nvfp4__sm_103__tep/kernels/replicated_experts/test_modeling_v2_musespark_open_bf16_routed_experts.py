# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep8 target's BF16 routed experts (``bf16_routed_experts``): what reaches ``ring_buddy``.

Host-only. The expert call itself (bitwise the packed FlashInfer call) is
``moe/test_modeling_v2_bf16_routed_experts.py``.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    weights as target_weights,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_experts import (  # noqa: E501
    ring_buddy,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe import (  # noqa: E501
    bf16_routed_experts as op_module,
)

pytestmark = pytest.mark.cpu_only

#: A composition with the persistent experts and without the BF16 routed experts.
_WITH_PERSISTENT = (
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
    "boundary_early_release",
    "finalize_transport",
    "deferred_input_prep",
    "sampler_store",
    "persistent_experts",
    "moe_prologue",
)


def test_the_bf16_routed_experts_build_on_no_other_path():
    """Over the ring-buddy stack with ``replicated_experts``, else over the owner-only window."""
    assert dispatch.closure({"bf16_routed_experts"}) == {"bf16_routed_experts"}
    dispatch.Plan(frozenset({"bf16_routed_experts"}))


_TP4 = types.SimpleNamespace(tp_size=4)


@pytest.mark.parametrize(
    "paths, expected",
    [
        (("replicated_experts",), {}),
        (("replicated_experts", "persistent_experts"), {"persistent": True}),
        (("replicated_experts", "bf16_routed_experts"), {"bf16_unpacked": True}),
        (("persistent_experts",), {"persistent": True, "windows": 1}),
        (("bf16_routed_experts",), {"bf16_unpacked": True, "windows": 1}),
        (_WITH_PERSISTENT, {"persistent": True}),
        (_WITH_PERSISTENT + ("bf16_routed_experts",), {"persistent": True, "bf16_unpacked": True}),
    ],
)
def test_bf16_routing_reaches_ring_buddy_install_only_when_planned(monkeypatch, paths, expected):
    """With ``bf16_routed_experts`` the install gets ``bf16_unpacked=True``; without it the
    keyword is not passed at all, so the install is the packed-routing one. Without
    ``replicated_experts`` the install is the owner-only window's (``windows=1``)."""
    calls = []
    fake = types.ModuleType("ring_buddy")
    fake.BLOCK, fake.STATIC_FROM = 32, 28

    def install(text_model, mapping, balance=True, static_from=32, **kwargs):
        calls.append(kwargs)

    fake.install = install
    package = modeling.__name__.rsplit(".", 1)[0]
    monkeypatch.setitem(sys.modules, f"{package}.kernels.replicated_experts.ring_buddy", fake)
    monkeypatch.setattr(
        importlib.import_module(f"{package}.kernels.replicated_experts"),
        "ring_buddy",
        fake,
        raising=False,
    )
    target_weights.install_loaded_features(object(), _TP4, dispatch.plan_of(*paths))
    assert calls == [expected]


class _OpBackend:
    """The two converters ``_bf16_call`` uses, and a packed call that records itself."""

    def __init__(self):
        self.packed = []

    @staticmethod
    def cvt_routing_method_type(value):
        return 100 + int(value)

    @staticmethod
    def cvt_activation_type(value):
        return 200 + int(value)

    def run_bf16_moe(self, **kwargs):
        self.packed.append(kwargs)
        return "packed"


def _backend(op_backend):
    return types.SimpleNamespace(
        op_backend=op_backend,
        intermediate_size_per_partition=4096,
        quant_method=types.SimpleNamespace(use_shuffled_weight=True, weight_layout=1),
        activation_type=0,
    )


def _stacks(window=2 * ring_buddy.BLOCK):
    """Stand-in stacks: only their leading dim (the call's window) is read."""
    return {"w3_w1_weight": torch.empty(window, 1), "w2_weight": torch.empty(window, 1)}


def test_unpacked_bf16_call_hands_the_entry_both_tensors(monkeypatch):
    seen = []
    monkeypatch.setattr(ring_buddy, "_act_type", lambda backend: 0)
    monkeypatch.setattr(
        type(op_module.bf16_routed_experts),
        "__call__",
        lambda self, *args: seen.append(args) or "unpacked",
    )
    op_backend = _OpBackend()
    routing = types.SimpleNamespace(routing_method_type=1)
    call = ring_buddy._bf16_call(_backend(op_backend), routing, unpacked=True)
    stacks = _stacks()
    x = torch.zeros((8, 4096), dtype=torch.bfloat16)
    ids = torch.zeros((8, 8), dtype=torch.int32)
    weights = torch.zeros((8, 8), dtype=torch.bfloat16)
    assert call(stacks, x, ids, weights) == "unpacked"
    assert op_backend.packed == []
    (args,) = seen
    assert args[0] is x and args[3] is ids and args[4] is weights
    assert args[1] is stacks["w3_w1_weight"] and args[2] is stacks["w2_weight"]
    # intermediate, the 64-expert window, FlashInfer's enum values, the layout, the geometry
    assert args[5:] == (4096, 0, 2 * ring_buddy.BLOCK, 101, 200, True, 1, 256, 8)
    # A row that is not plain BF16 (the NVFP4 layers' quantized input) is not this call's.
    assert call(stacks, x.float(), ids, weights) is None
    # The owner-only window of a tep4 rank: the 64-expert own block, the same geometry.
    seen.clear()
    assert call(_stacks(64), x, ids, weights) == "unpacked"
    assert seen[0][5:8] == (4096, 0, 64)


def test_packed_bf16_call_is_unchanged_without_bf16_routing(monkeypatch):
    monkeypatch.setattr(ring_buddy, "_act_type", lambda backend: 0)
    monkeypatch.setattr(
        type(op_module.bf16_routed_experts),
        "__call__",
        lambda self, *args: pytest.fail(
            "the packed call must not reach the unpacked-routing entry"
        ),
    )
    op_backend = _OpBackend()
    call = ring_buddy._bf16_call(_backend(op_backend), types.SimpleNamespace(routing_method_type=1))
    x = torch.zeros((8, 4096), dtype=torch.bfloat16)
    ids = torch.zeros((8, 8), dtype=torch.int32)
    weights = torch.zeros((8, 8), dtype=torch.bfloat16)
    assert call(_stacks(), x, ids, weights) == "packed"
    (kwargs,) = op_backend.packed
    assert kwargs["topk_ids"] is ids and kwargs["topk_weights"] is weights
    assert kwargs["routing_method_type"] == 1 and kwargs["gated_act_type"] == 0
    assert kwargs["do_finalize"] is False
