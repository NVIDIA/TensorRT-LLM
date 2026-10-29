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
"""CPU tests of the dense DWDP collector, checks, blob layout and checksums.

The toy decoder layers below imitate the Kimi-K3 tensors the collector has to handle: the fused
``qkvg_proj.weight`` buffer with ``q/k/v/g_proj.weight`` row-view buffers, the plain
``_bfa_proj_weight`` attribute with ``f_a_proj`` / ``b_proj`` Parameter row views and pad rows no
view covers, the ``_mtp_conv_weights`` tuple, ``_A_log_f32`` sharing ``A_log``'s storage, tensors
under the threshold, excluded scratch names (one of them aliasing an anchor), an MLA layer and the
MoE-dense roots.  Aliases that are not plain row views ride along: ``_gate_b_t`` is a transposed
view of ``f_b_proj``'s storage (the low-rank gate's layout), and ``_bfa_views`` holds a stride-0
view and an empty view of the ``_bfa`` storage.  Weights hold small integers so every toy forward
is exact in fp32 and can be compared bit for bit (``test_dense_repoint.py`` reuses the toy).
"""

import os
import re
import struct
import subprocess
import sys
import weakref
from pathlib import Path

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.modules.dwdp.dense import layout as dense_layout
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.layout import (
    assign_blob_offsets,
    collect_dense_layout,
    compute_checksums,
    row_pieces,
    tensor_checksum,
    write_blob,
)
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    CHECKSUM_MULTIPLIER,
    POOLED_SLACK_BYTES,
    ROW_ALIGN,
    BindingKind,
    Checksum,
    DenseLayout,
    DenseLayoutError,
    SegmentKind,
)

pytestmark = pytest.mark.cpu_only

ATTN = SegmentKind.ATTN
MOE = SegmentKind.MOE

#: The model tables.
ATTR_ALLOW = frozenset(
    {
        "_bfa_proj_weight",
        "_qkvg_proj_weight",
        "_gate_b_t",
        "_packed_conv_weight",
        "_mtp_conv_weights",
        "_w_q_t",
        "_w_k_t",
        "_w_v_t",
        "_A_log_f32",
        "_dt_bias_f32",
        "_onorm_w_f32",
    }
)
EXCLUDE = frozenset(
    {
        "_o_dense",
        "_fused_q_placeholder",
        "softmax_stats_tensor",
        "temp_softmax_stats_tensor",
        "e_score_correction_bias",
    }
)

#: Toy threshold: the toy weights are 8-64 KiB, the small tensors at most 2 KiB.
TOY_MIN_BYTES = 8192
NUM_RANKS = 4
HIDDEN = 256
ROWS = 64
F_ROWS = 16
B_ROWS = 2
TOKENS = 8
#: Decoder layer kinds: layer 0 stays replicated, layer 3 is an MLA layer.
LAYER_KINDS = ("dense", "kda", "kda", "mla", "kda", "kda")
COVERED = (1, 2, 3, 4, 5)

FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16


# ---------------------------------------------------------------------------
# Toy Kimi-K3 decoder layers
# ---------------------------------------------------------------------------


def _ints(generator, shape, dtype):
    """Small integers in [-3, 3], exactly representable in every moved dtype."""
    return torch.randint(-3, 4, shape, generator=generator).to(torch.float32).to(dtype)


def _bytes(generator, shape):
    return torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)


def _squash(x):
    """Fold back into [-3, 3]; exact on the integer-valued fp32 tensors of the toy."""
    return torch.remainder(x, 7.0) - 3.0


def _project(x, weight, scale=None):
    out = x @ weight.float().t()
    if scale is not None:
        out = out + scale.float().sum()
    return _squash(out)


def _parameter(tensor):
    return nn.Parameter(tensor, requires_grad=False)


class ToyFp8Linear(nn.Module):
    """Stands in for ``_Fp8BlockScaleWeightReadLinear``: weight and scale buffers."""

    def __init__(self, weight, scale):
        super().__init__()
        self.out_features = weight.shape[0]
        self._weights_transformed = True
        self._fused_projection = None
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("weight_scale", scale, persistent=False)

    def forward(self, x):
        return _project(x, self.weight, self.weight_scale)


class ToyNorm(nn.Module):
    def __init__(self, generator, size, dtype=BF16):
        super().__init__()
        self.weight = _parameter(_ints(generator, (size,), dtype))


class ToyLinear(nn.Linear):
    def __init__(self, generator, in_features, out_features):
        super().__init__(in_features, out_features, bias=False, dtype=BF16)
        self.weight.requires_grad_(False)
        self.weight.data.copy_(_ints(generator, (out_features, in_features), BF16))


class ToyKDA(nn.Module):
    """KDA mixer after load: fused FP8 qkvg, fused BF16 [f_a | b], decode constants."""

    def __init__(self, generator):
        super().__init__()
        g = generator
        self.A_log = _parameter(_ints(g, (16,), torch.float32))
        self.dt_bias = _parameter(_ints(g, (64,), torch.float32))
        for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
            setattr(self, name, ToyFp8Linear(_ints(g, (ROWS, HIDDEN), FP8), _bytes(g, (ROWS, 128))))
        self.o_proj = ToyFp8Linear(_ints(g, (HIDDEN, 4 * ROWS), FP8), _bytes(g, (HIDDEN, 32)))
        self.f_a_proj = ToyLinear(g, HIDDEN, F_ROWS)
        self.b_proj = ToyLinear(g, HIDDEN, B_ROWS)
        self.f_b_proj = ToyLinear(g, F_ROWS, HIDDEN)
        self.q_conv1d = ToyNorm(g, 4 * HIDDEN)
        self.o_norm = ToyNorm(g, HIDDEN)
        self._finalize(g)

    def _finalize(self, g):
        """Mirror ``finalize_decode_weights`` and the FP8 fusion of the K3 loader."""
        projections = [self.q_proj, self.k_proj, self.v_proj, self.g_proj]
        fused = torch.cat([module.weight for module in projections])
        self.qkvg_proj = ToyFp8Linear(fused, _bytes(g, (4 * ROWS, 64)))
        for index, module in enumerate(projections):
            module._fused_projection = (weakref.ref(self.qkvg_proj), index * ROWS)
            module.weight = fused[index * ROWS : (index + 1) * ROWS]
        bfa = torch.cat(
            [self.f_a_proj.weight.data, self.b_proj.weight.data, torch.zeros(6, HIDDEN, dtype=BF16)]
        )
        self.f_a_proj.weight.data = bfa[:F_ROWS]
        self.b_proj.weight.data = bfa[F_ROWS : F_ROWS + B_ROWS]
        self._bfa_proj_weight = bfa
        # Names in neither model table that alias the _bfa storage: a stride-0 view and an empty
        # view past the pad rows.
        self._bfa_views = [bfa[:1].expand(4, -1), bfa[bfa.shape[0] :]]
        self._qkvg_proj_weight = None
        # The low-rank gate's [f_b; g_b] bmm operand: a transposed view of f_b_proj's storage.
        self._gate_b_t = self.f_b_proj.weight.data.view(2, HIDDEN // 2, F_ROWS).transpose(1, 2)
        self._mtp_conv_weights = tuple(_ints(g, (4, 1024), BF16) for _ in range(3))
        self._packed_conv_weight = _ints(g, (3, HIDDEN), BF16)
        self._w_q_t = _ints(g, (4, 128), BF16)
        self._A_log_f32 = self.A_log.detach().float().contiguous()
        self._dt_bias_f32 = self.dt_bias.detach().float().contiguous()
        self._o_dense = self.o_proj.weight[64:192]
        self._projection_fork_event = object()

    def forward(self, x):
        heads = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x), self.g_proj(x)], -1)
        heads = _squash(heads + self.qkvg_proj(x))
        f_a = _project(x, self.f_a_proj.weight)
        beta = _project(x, self.b_proj.weight)
        bfa = _project(x, self._bfa_proj_weight)
        f_b = _project(f_a, self.f_b_proj.weight)
        gates = torch.bmm(f_a.expand(2, -1, -1), self._gate_b_t.float())
        gates = _squash(gates.transpose(0, 1).reshape(f_a.shape[0], -1))
        views = sum(_project(x, view).sum(-1, keepdim=True) for view in self._bfa_views)
        constants = (
            sum(weight.float().sum() for weight in self._mtp_conv_weights)
            + self._A_log_f32.sum()
            + self._dt_bias_f32.sum()
            + self._packed_conv_weight.float().sum()
            + self._o_dense.float().sum()
        )
        mixed = heads + f_b + gates + bfa.sum(-1, keepdim=True) + beta.sum(-1, keepdim=True)
        return self.o_proj(_squash(mixed + views + constants + self.o_norm.weight.float()))


class ToyAttentionBackend:
    """Not an ``nn.Module``: the walk never enters it (like MLA's ``mha`` / ``mqa``)."""

    def __init__(self, generator):
        self.workspace = _ints(generator, (4096,), BF16)


class ToyMLA(nn.Module):
    """MLA attention after load, with the BF16 absorb copies and excluded scratch tensors."""

    def __init__(self, generator):
        super().__init__()
        g = generator
        self.kv_a_proj_with_mqa = ToyFp8Linear(_ints(g, (128, HIDDEN), FP8), _bytes(g, (128, 64)))
        self.q_b_proj = ToyFp8Linear(_ints(g, (HIDDEN, 128), FP8), _bytes(g, (HIDDEN, 32)))
        self.kv_b_proj = ToyFp8Linear(_ints(g, (HIDDEN, 64), FP8), _bytes(g, (HIDDEN, 32)))
        self.o_proj = ToyFp8Linear(_ints(g, (HIDDEN, HIDDEN), FP8), _bytes(g, (HIDDEN, 32)))
        self.k_b_proj_trans = _parameter(_ints(g, (8, 64, 64), FP8))
        self.k_b_proj_trans_dequant = _parameter(_ints(g, (8, 64, 64), BF16))
        self.v_b_proj_scale = _parameter(self.kv_b_proj.weight_scale[:128])
        self._fused_q_placeholder = _ints(g, (64, HIDDEN), BF16)
        self.softmax_stats_tensor = torch.zeros(4096, dtype=torch.float32)
        self.mha = ToyAttentionBackend(g)

    def forward(self, x):
        latent = self.kv_a_proj_with_mqa(x)
        query = self.q_b_proj(latent)
        key_value = self.kv_b_proj(latent[:, :64])
        absorb = (
            self.k_b_proj_trans.float().sum()
            + self.k_b_proj_trans_dequant.float().sum()
            + self.v_b_proj_scale.float().sum()
        )
        return self.o_proj(_squash(query + key_value + absorb))


class ToyGate(nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.weight = _parameter(_ints(generator, (32, HIDDEN), torch.float32))
        self.e_score_correction_bias = _parameter(_ints(generator, (32,), torch.float32))


class ToySharedMLP(nn.Module):
    def __init__(self, generator):
        super().__init__()
        g = generator
        self.gate_up_proj = ToyFp8Linear(_ints(g, (HIDDEN, HIDDEN), FP8), _bytes(g, (HIDDEN, 32)))
        self.down_proj = ToyFp8Linear(_ints(g, (HIDDEN, 128), FP8), _bytes(g, (HIDDEN, 32)))

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(_squash(gate * up))


class ToyRoutedExperts(nn.Module):
    """Managed by expert DWDP; never a dense root."""

    def __init__(self, generator):
        super().__init__()
        self.weight = _parameter(_ints(generator, (4, HIDDEN, HIDDEN), BF16))


class ToyMoE(nn.Module):
    def __init__(self, generator):
        super().__init__()
        g = generator
        self.gate = ToyGate(g)
        self.shared_experts = ToySharedMLP(g)
        self.routed_expert_down_proj = ToyFp8Linear(
            _ints(g, (128, HIDDEN), FP8), _bytes(g, (128, 16))
        )
        self.routed_expert_up_proj = ToyFp8Linear(
            _ints(g, (HIDDEN, 128), FP8), _bytes(g, (HIDDEN, 8))
        )
        self.routed_expert_norm = ToyNorm(g, 128)
        self.routed_experts = ToyRoutedExperts(g)

    def forward(self, x):
        logits = _squash(x @ self.gate.weight.t() + self.gate.e_score_correction_bias)
        latent = self.routed_expert_down_proj(x)
        latent = _squash(
            latent * self.routed_expert_norm.weight.float() + logits.sum(-1, keepdim=True)
        )
        expert = _squash(latent + self.routed_experts.weight[0, 0, :128].float())
        return _squash(self.routed_expert_up_proj(expert) + self.shared_experts(x))


class ToyDecoderLayer(nn.Module):
    """Decoder layer with the ``dense_dwdp_roots`` model seam."""

    def __init__(self, layer_idx, kind, generator):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_moe = kind != "dense"
        self.is_kda = kind != "mla"
        self.input_layernorm = ToyNorm(generator, HIDDEN)
        if self.is_kda:
            self.linear_attn = ToyKDA(generator)
        else:
            self.self_attn = ToyMLA(generator)
        if self.is_moe:
            self.block_sparse_moe = ToyMoE(generator)
        else:
            self.mlp = ToySharedMLP(generator)

    def dense_dwdp_roots(self, segment):
        if not self.is_moe:
            return []
        if segment == "attn":
            if self.is_kda:
                return [("linear_attn", self.linear_attn)]
            return [("self_attn", self.self_attn)]
        moe = self.block_sparse_moe
        return [
            (f"block_sparse_moe.{name}", getattr(moe, name))
            for name in (
                "gate",
                "shared_experts",
                "routed_expert_down_proj",
                "routed_expert_up_proj",
                "routed_expert_norm",
            )
        ]

    def forward(self, x):
        hidden = _squash(x + self.input_layernorm.weight.float())
        attention = self.linear_attn(hidden) if self.is_kda else self.self_attn(hidden)
        hidden = _squash(x + attention)
        mlp = self.block_sparse_moe(hidden) if self.is_moe else self.mlp(hidden)
        return _squash(hidden + mlp)


class ToyModel(nn.Module):
    def __init__(self, seed=0, kinds=LAYER_KINDS):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.layers = nn.ModuleList(
            ToyDecoderLayer(index, kind, generator) for index, kind in enumerate(kinds)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def toy_input(seed=1234):
    return _ints(torch.Generator().manual_seed(seed), (TOKENS, HIDDEN), torch.float32)


def make_config(**overrides):
    return DenseConfig(enabled=True, **{"min_bytes": TOY_MIN_BYTES, **overrides})


def collect(model, config=None, *, num_ranks=NUM_RANKS):
    return collect_dense_layout(
        list(enumerate(model.layers)),
        config or make_config(),
        num_ranks=num_ranks,
        device=torch.device("cpu"),
        attr_allow=ATTR_ALLOW,
        exclude=EXCLUDE,
    )


def anchor_paths(layout, layer, segment):
    return [
        [binding.qualified_name for binding in anchor.bindings]
        for anchor in layout.segment(layer, segment).anchors
    ]


def storage_ptr(tensor):
    return tensor.untyped_storage().data_ptr()


#: Anchors of a toy KDA layer in walk order: the mixer's own attributes (sorted by name), then
#: the children in registration order (``qkvg_proj`` last, as the K3 loader adds it).  Each entry
#: lists the anchor's bindings in walk order.
KDA_ANCHORS = [
    [
        "linear_attn._bfa_proj_weight",
        "linear_attn._bfa_views[0]",
        "linear_attn._bfa_views[1]",
        "linear_attn.f_a_proj.weight",
        "linear_attn.b_proj.weight",
    ],
    ["linear_attn._gate_b_t", "linear_attn.f_b_proj.weight"],
    ["linear_attn._mtp_conv_weights[0]"],
    ["linear_attn._mtp_conv_weights[1]"],
    ["linear_attn._mtp_conv_weights[2]"],
    [
        "linear_attn.q_proj.weight",
        "linear_attn.k_proj.weight",
        "linear_attn.v_proj.weight",
        "linear_attn.g_proj.weight",
        "linear_attn.qkvg_proj.weight",
    ],
    ["linear_attn.q_proj.weight_scale"],
    ["linear_attn.k_proj.weight_scale"],
    ["linear_attn.v_proj.weight_scale"],
    ["linear_attn.g_proj.weight_scale"],
    ["linear_attn._o_dense", "linear_attn.o_proj.weight"],
    ["linear_attn.o_proj.weight_scale"],
    ["linear_attn.qkvg_proj.weight_scale"],
]

MLA_ANCHORS = [
    ["self_attn.k_b_proj_trans"],
    ["self_attn.k_b_proj_trans_dequant"],
    ["self_attn.v_b_proj_scale", "self_attn.kv_b_proj.weight_scale"],
    ["self_attn.kv_a_proj_with_mqa.weight"],
    ["self_attn.kv_a_proj_with_mqa.weight_scale"],
    ["self_attn.q_b_proj.weight"],
    ["self_attn.q_b_proj.weight_scale"],
    ["self_attn.kv_b_proj.weight"],
    ["self_attn.o_proj.weight"],
    ["self_attn.o_proj.weight_scale"],
]

MOE_ANCHORS = [
    ["block_sparse_moe.gate.weight"],
    ["block_sparse_moe.shared_experts.gate_up_proj.weight"],
    ["block_sparse_moe.shared_experts.gate_up_proj.weight_scale"],
    ["block_sparse_moe.shared_experts.down_proj.weight"],
    ["block_sparse_moe.shared_experts.down_proj.weight_scale"],
    ["block_sparse_moe.routed_expert_down_proj.weight"],
    ["block_sparse_moe.routed_expert_up_proj.weight"],
]


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def test_collects_k3_like_anchors_and_bindings():
    model = ToyModel()
    layout = collect(model)

    assert layout.layers == COVERED
    assert layout.warnings == []
    assert list(layout.segments) == [(layer, kind) for layer in COVERED for kind in (ATTN, MOE)]
    for layer in COVERED:
        expected = MLA_ANCHORS if LAYER_KINDS[layer] == "mla" else KDA_ANCHORS
        assert anchor_paths(layout, layer, ATTN) == expected
        assert anchor_paths(layout, layer, MOE) == MOE_ANCHORS

    kda = model.layers[1].linear_attn
    anchors = {
        anchor.bindings[0].qualified_name: anchor for anchor in layout.segment(1, ATTN).anchors
    }
    qkvg = anchors["linear_attn.q_proj.weight"]
    assert qkvg.nbytes == 4 * ROWS * HIDDEN
    assert [binding.byte_offset for binding in qkvg.bindings] == [0, 16384, 32768, 49152, 0]
    assert [binding.kind for binding in qkvg.bindings] == [BindingKind.BUFFER] * 5
    # The whole _bfa storage moves, including the pad rows no Parameter view covers.
    bfa = anchors["linear_attn._bfa_proj_weight"]
    assert bfa.nbytes == 24 * HIDDEN * 2
    assert [(binding.kind, binding.byte_offset) for binding in bfa.bindings] == [
        (BindingKind.ATTRIBUTE, 0),
        (BindingKind.SEQUENCE_ITEM, 0),
        (BindingKind.SEQUENCE_ITEM, bfa.nbytes),
        (BindingKind.PARAMETER, 0),
        (BindingKind.PARAMETER, F_ROWS * HIDDEN * 2),
    ]
    assert bfa.bindings[3].original.data_ptr() == kda.f_a_proj.weight.data_ptr()
    # Bindings that are not row views keep their geometry: stride 0, empty, transposed.
    stride_0, empty = bfa.bindings[1:3]
    assert (stride_0.size, stride_0.stride) == ((4, HIDDEN), (0, 1))
    assert stride_0.span_bytes == 2 * HIDDEN
    assert (empty.size, empty.span_bytes) == ((0, HIDDEN), 0)
    gate_b_t = anchors["linear_attn._gate_b_t"].bindings[0]
    assert (gate_b_t.size, gate_b_t.stride) == (
        (2, F_ROWS, HIDDEN // 2),
        (HIDDEN // 2 * F_ROWS, 1, F_ROWS),
    )
    assert not gate_b_t.original.is_contiguous()
    # The excluded scratch name that aliases o_proj is a binding of o_proj's anchor.
    o_dense = anchors["linear_attn._o_dense"].bindings[0]
    assert (o_dense.size, o_dense.stride, o_dense.byte_offset) == ((128, 256), (256, 1), 16384)
    mtp = anchors["linear_attn._mtp_conv_weights[1]"].bindings[0]
    assert (mtp.kind, mtp.name, mtp.index) == (BindingKind.SEQUENCE_ITEM, "_mtp_conv_weights", 1)
    # Sources are flat byte views of the whole original storage.
    for anchor in layout.anchors():
        assert anchor.source.dtype == torch.uint8 and anchor.source.numel() == anchor.nbytes
        assert storage_ptr(anchor.source) == storage_ptr(anchor.bindings[0].original)
        for binding in anchor.bindings:
            assert binding.owner_ref() is not None and binding.checksum is None
    # Checksums are left to compute_checksums.
    assert all(segment.checksum is None for segment in layout.segments.values())


def test_small_scratch_and_unwalked_tensors_stay_in_place():
    model = ToyModel()
    layout = collect(model)
    moved = {storage_ptr(anchor.source) for anchor in layout.anchors()}
    layer = model.layers[1]
    kda = layer.linear_attn
    mla = model.layers[3].self_attn
    moe = layer.block_sparse_moe
    stays = [
        kda.A_log,
        kda._A_log_f32,
        kda.dt_bias,
        kda._packed_conv_weight,
        kda._w_q_t,
        kda.q_conv1d.weight,
        kda.o_norm.weight,
        layer.input_layernorm.weight,
        mla._fused_q_placeholder,
        mla.softmax_stats_tensor,
        mla.mha.workspace,
        moe.gate.e_score_correction_bias,
        moe.routed_expert_down_proj.weight_scale,
        moe.routed_expert_up_proj.weight_scale,
        moe.routed_expert_norm.weight,
        moe.routed_experts.weight,
        model.layers[0].linear_attn.qkvg_proj.weight,
        model.layers[0].mlp.gate_up_proj.weight,
    ]
    assert not moved & {storage_ptr(tensor) for tensor in stays}
    # _A_log_f32 shares A_log's storage (``.float()`` of an fp32 tensor returns itself).
    assert storage_ptr(kda._A_log_f32) == storage_ptr(kda.A_log)


def test_zero_threshold_moves_small_aliases_together():
    model = ToyModel()
    kda = model.layers[1].linear_attn
    kda._w_k_t = torch.empty(0, dtype=BF16)
    layout = collect(model, make_config(min_bytes=0))

    anchors = {
        anchor.bindings[0].qualified_name: anchor for anchor in layout.segment(1, ATTN).anchors
    }
    a_log = anchors["linear_attn.A_log"]
    assert [binding.qualified_name for binding in a_log.bindings] == [
        "linear_attn.A_log",
        "linear_attn._A_log_f32",
    ]
    assert [binding.kind for binding in a_log.bindings] == [
        BindingKind.PARAMETER,
        BindingKind.ATTRIBUTE,
    ]
    names = {binding.qualified_name for anchor in layout.anchors() for binding in anchor.bindings}
    assert "linear_attn._w_k_t" not in names  # zero-byte storages never anchor
    assert "block_sparse_moe.gate.e_score_correction_bias" not in names  # excluded
    assert "block_sparse_moe.routed_expert_norm.weight" in names
    assert "self_attn._fused_q_placeholder" not in names
    assert layout.warnings == []


def test_attention_only_segments_ignore_moe_roots():
    layout = collect(ToyModel(), make_config(segments=("attn",)))
    assert list(layout.segments) == [(layer, ATTN) for layer in COVERED]
    assert layout.segment_kinds == (ATTN,)
    assert set(layout.slot_specs) == {ATTN}


def test_other_device_types_are_ignored():
    model = ToyModel()
    baseline = collect(model).digest
    model.layers[2].linear_attn._w_q_t = torch.empty(1 << 20, dtype=BF16, device="meta")
    assert collect(model).digest == baseline


def test_module_reached_twice_is_walked_once():
    model = ToyModel()
    layer = model.layers[2]
    layer.linear_attn.o_proj_alias = layer.linear_attn.o_proj
    moe = layer.block_sparse_moe
    roots = layer.dense_dwdp_roots("moe")
    object.__setattr__(
        layer,
        "dense_dwdp_roots",
        lambda segment: (
            [*roots, ("block_sparse_moe.gate_again", moe.gate)]
            if segment == "moe"
            else [("linear_attn", layer.linear_attn)]
        ),
    )
    layout = collect(model)
    assert anchor_paths(layout, 2, ATTN) == KDA_ANCHORS
    assert anchor_paths(layout, 2, MOE) == MOE_ANCHORS
    keys = [
        (id(binding.owner_ref()), binding.kind, binding.name, binding.index)
        for anchor in layout.anchors()
        for binding in anchor.bindings
    ]
    assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# Determinism and layout
# ---------------------------------------------------------------------------


def _reinsert_attributes_reversed(module):
    """Re-insert the tensor and sequence attributes of ``module`` in reverse name order."""
    attributes = vars(module)
    names = [
        name for name, value in attributes.items() if isinstance(value, (torch.Tensor, tuple, list))
    ]
    for name in sorted(names, reverse=True):
        attributes[name] = attributes.pop(name)


def test_two_independent_builds_have_the_same_digest():
    first = collect(ToyModel(seed=1))
    second = collect(ToyModel(seed=2))
    assert first.digest == second.digest
    assert first.digest_input() == second.digest_input()
    assert collect(ToyModel(seed=1)).digest == first.digest
    # The walk visits plain attributes by name, so the order a rank set them in does not matter.
    reordered = ToyModel(seed=1)
    for module in reordered.modules():
        _reinsert_attributes_reversed(module)
    names = list(vars(reordered.layers[1].linear_attn))
    assert names.index("_mtp_conv_weights") < names.index("_bfa_proj_weight")
    assert collect(reordered).digest == first.digest
    # The digest covers the switches and the shapes.
    assert collect(ToyModel(seed=1), num_ranks=8).digest != first.digest
    assert collect(ToyModel(seed=1), make_config(min_bytes=4096)).digest != first.digest
    assert collect(ToyModel(seed=1), make_config(store_chunk_mb=1)).digest != first.digest
    assert collect(ToyModel(seed=1), make_config(segments=("attn",))).digest != first.digest
    reshaped = ToyModel(seed=1)
    reshaped.layers[4].block_sparse_moe.gate.weight.data = torch.zeros(64, HIDDEN)
    assert collect(reshaped).digest != first.digest


def test_digest_is_the_same_in_another_process():
    """Ranks are separate processes, so nothing process-specific may reach the layout."""
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
        "import test_dense_layout as toy\n"
        "print(toy.collect(toy.ToyModel(seed=3)).digest)\n"
    )
    # The child sees this process's GPUs: importing tensorrt_llm reads the device capability.
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONHASHSEED": "12345"},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().splitlines()[-1] == collect(ToyModel(seed=4)).digest


def test_layout_is_4k_aligned():
    layout = collect(ToyModel())
    store_offset = 0
    for segment in layout.segments.values():
        assert segment.store_offset == store_offset
        store_offset += segment.row_bytes
        assert segment.row_bytes % ROW_ALIGN == 0 and segment.store_offset % ROW_ALIGN == 0
        assert segment.chunk_offset % ROW_ALIGN == 0
        assert segment.row_bytes * NUM_RANKS >= segment.blob_bytes
        end = 0
        for index, anchor in enumerate(segment.anchors):
            assert anchor.index == index and (anchor.layer, anchor.segment) == segment.key
            assert anchor.blob_offset % ROW_ALIGN == 0
            assert anchor.blob_offset == -(-end // ROW_ALIGN) * ROW_ALIGN
            end = anchor.blob_end
            for binding in anchor.bindings:
                start = anchor.blob_offset + binding.byte_offset
                assert start % binding.dtype.itemsize == 0
                assert binding.byte_offset + binding.span_bytes <= anchor.nbytes
        assert segment.blob_bytes == end
    for kind, spec in layout.slot_specs.items():
        widest = max(layout.segment(layer, kind).row_bytes for layer in layout.layers)
        assert (spec.row_bytes, spec.num_ranks) == (widest, NUM_RANKS)
        assert spec.alloc_bytes == NUM_RANKS * widest + ROW_ALIGN
    assert layout.moved_bytes == sum(anchor.nbytes for anchor in layout.anchors())


def test_assign_blob_offsets():
    assert assign_blob_offsets([]) == ([], 0)
    assert assign_blob_offsets([1, 4096, 5000, 1]) == ([0, 4096, 8192, 16384], 16385)
    assert assign_blob_offsets([8192, 3]) == ([0, 8192], 8195)
    with pytest.raises(ValueError):
        assign_blob_offsets([4096, 0])


# ---------------------------------------------------------------------------
# Storages left in place (warnings; strict raises)
# ---------------------------------------------------------------------------


def _paths(layout):
    return {binding.qualified_name for anchor in layout.anchors() for binding in anchor.bindings}


@pytest.mark.parametrize(
    "alias, reported",
    [
        # A registered buffer, i.e. an eligible reference from the attention segment.
        ("router_copy", "layers.2.linear_attn.router_copy"),
        # An excluded scratch name never anchors a storage, but it shares one all the same.
        ("_o_dense", "layers.2.block_sparse_moe.gate.weight"),
    ],
)
def test_storage_shared_across_segments_stays(alias, reported):
    model = ToyModel()
    layer = model.layers[2]
    gate = layer.block_sparse_moe.gate
    if alias == "router_copy":
        layer.linear_attn.register_buffer(alias, gate.weight.data, persistent=False)
    else:
        setattr(layer.linear_attn, alias, gate.weight[:8])

    layout = collect(model)
    assert len(layout.warnings) == 1
    assert f"storage of {reported} " in layout.warnings[0]
    assert "layers.2/attn, layers.2/moe" in layout.warnings[0]
    assert storage_ptr(gate.weight) not in {storage_ptr(a.source) for a in layout.anchors()}
    assert anchor_paths(layout, 2, MOE) == MOE_ANCHORS[1:]
    assert anchor_paths(layout, 1, MOE) == MOE_ANCHORS

    with pytest.raises(DenseLayoutError, match=re.escape(reported)):
        collect(model, make_config(strict=True))


def test_storage_shared_across_layers_stays():
    model = ToyModel()
    model.layers[4].block_sparse_moe.gate.weight = model.layers[1].block_sparse_moe.gate.weight

    layout = collect(model)
    assert len(layout.warnings) == 1
    assert "layers.1/moe, layers.4/moe" in layout.warnings[0]
    for layer in (1, 4):
        assert anchor_paths(layout, layer, MOE) == MOE_ANCHORS[1:]
    assert anchor_paths(layout, 2, MOE) == MOE_ANCHORS

    with pytest.raises(DenseLayoutError, match="layers.1.block_sparse_moe.gate.weight"):
        collect(model, make_config(strict=True))


def test_pooled_storage_stays():
    model = ToyModel()
    down_proj = model.layers[5].block_sparse_moe.shared_experts.down_proj
    pool = torch.zeros(POOLED_SLACK_BYTES + (2 << 20), dtype=FP8)
    view = pool[: HIDDEN * 128].view(HIDDEN, 128)
    view.copy_(down_proj.weight)
    down_proj.weight = view
    # A storage with less than POOLED_SLACK_BYTES of unreferenced bytes still moves whole.
    up_proj = model.layers[5].block_sparse_moe.routed_expert_up_proj
    roomy = torch.zeros(HIDDEN * 128 + (POOLED_SLACK_BYTES // 2), dtype=FP8)
    roomy[: HIDDEN * 128].view(HIDDEN, 128).copy_(up_proj.weight)
    up_proj.weight = roomy[: HIDDEN * 128].view(HIDDEN, 128)

    layout = collect(model)
    assert len(layout.warnings) == 1 and "pooled" in layout.warnings[0]
    assert "layers.5.block_sparse_moe.shared_experts.down_proj.weight" in layout.warnings[0]
    paths = anchor_paths(layout, 5, MOE)
    assert ["block_sparse_moe.shared_experts.down_proj.weight"] not in paths
    roomy_anchor = layout.segment(5, MOE).anchors[paths.index(MOE_ANCHORS[-1])]
    assert roomy_anchor.nbytes == roomy.untyped_storage().nbytes()

    with pytest.raises(DenseLayoutError, match="pooled"):
        collect(model, make_config(strict=True))


def test_pooled_check_takes_the_union_of_every_reference(monkeypatch):
    """At a slack of toy scale, every reference counts, and only all of them together.

    The q/k/v/g row views reach the fused qkvg storage before ``qkvg_proj.weight`` does, and the
    excluded ``_o_dense`` (rows 64-191) reaches the ``o_proj`` storage before ``o_proj.weight``
    does.
    """
    monkeypatch.setattr(dense_layout, "POOLED_SLACK_BYTES", ROW_ALIGN)
    layout = collect(ToyModel())
    assert layout.warnings == []
    for layer in COVERED:
        expected = MLA_ANCHORS if LAYER_KINDS[layer] == "mla" else KDA_ANCHORS
        assert anchor_paths(layout, layer, ATTN) == expected
        assert anchor_paths(layout, layer, MOE) == MOE_ANCHORS

    # o_proj.weight views the first HIDDEN rows of a storage with HIDDEN // 2 more rows.
    model = ToyModel()
    kda = model.layers[2].linear_attn
    rows = torch.cat([kda.o_proj.weight, torch.zeros(HIDDEN // 2, 4 * ROWS, dtype=FP8)])
    kda.o_proj.weight = rows[:HIDDEN]
    kda._o_dense = None
    layout = collect(model)
    assert len(layout.warnings) == 1
    assert "storage of layers.2.linear_attn.o_proj.weight " in layout.warnings[0]
    assert "looks pooled" in layout.warnings[0]
    # An excluded alias over the extra rows completes the union, so the whole storage moves.
    kda._o_dense = rows[HIDDEN:]
    layout = collect(model, make_config(strict=True))
    assert anchor_paths(layout, 2, ATTN) == KDA_ANCHORS
    o_proj = next(
        anchor
        for anchor in layout.segment(2, ATTN).anchors
        if anchor.path == "layers.2.linear_attn._o_dense"
    )
    assert o_proj.nbytes == rows.untyped_storage().nbytes()


def test_unknown_large_attribute_warns_and_strict_raises():
    model = ToyModel()
    kda = model.layers[2].linear_attn
    kda._decode_cache = torch.zeros(4096, dtype=torch.float32)

    layout = collect(model)
    assert len(layout.warnings) == 1
    assert "layers.2.linear_attn._decode_cache" in layout.warnings[0]
    assert "linear_attn._decode_cache" not in _paths(layout)
    with pytest.raises(DenseLayoutError, match="_decode_cache"):
        collect(model, make_config(strict=True))
    del kda._decode_cache

    # Each item of an unknown tuple is reported on its own, and none moves.
    kda._decode_tables = (torch.zeros(4096), torch.zeros(4096))
    layout = collect(model)
    assert len(layout.warnings) == 2
    for index, warning in enumerate(layout.warnings):
        assert f"layers.2.linear_attn._decode_tables[{index}] " in warning
        assert f"linear_attn._decode_tables[{index}]" not in _paths(layout)
    with pytest.raises(DenseLayoutError, match=re.escape("_decode_tables[0]")):
        collect(model, make_config(strict=True))
    del kda._decode_tables

    # Small unknown attributes are not reported.
    kda._small_cache = torch.zeros(16)
    assert collect(model, make_config(strict=True)).warnings == []


def test_unknown_attribute_aliasing_an_anchor_is_repointed_not_reported():
    model = ToyModel()
    kda = model.layers[2].linear_attn
    kda._decode_qkvg_view = kda.qkvg_proj.weight[ROWS:]
    layout = collect(model, make_config(strict=True))
    assert layout.warnings == []
    qkvg = next(
        anchor
        for anchor in layout.segment(2, ATTN).anchors
        if "linear_attn.qkvg_proj.weight" in [b.qualified_name for b in anchor.bindings]
    )
    alias = qkvg.bindings[0]
    assert (alias.qualified_name, alias.kind) == (
        "linear_attn._decode_qkvg_view",
        BindingKind.ATTRIBUTE,
    )
    assert alias.byte_offset == ROWS * HIDDEN


# ---------------------------------------------------------------------------
# Errors regardless of STRICT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strict", [False, True])
def test_untransformed_weights_raise(strict):
    model = ToyModel()
    model.layers[4].linear_attn.o_proj._weights_transformed = False
    with pytest.raises(DenseLayoutError, match="_weights_transformed=False"):
        collect(model, make_config(strict=strict))


def test_unmovable_dtype_raises():
    model = ToyModel()
    model.layers[1].linear_attn.register_buffer(
        "rope_cache", torch.zeros(2048, dtype=torch.float64), persistent=False
    )
    with pytest.raises(DenseLayoutError, match="rope_cache has dtype torch.float64"):
        collect(model)


class _MarkedTensor(torch.Tensor):
    pass


def test_tensor_subclass_binding_raises():
    model = ToyModel()
    kda = model.layers[1].linear_attn
    kda._marked_bfa = kda._bfa_proj_weight.as_subclass(_MarkedTensor)
    with pytest.raises(DenseLayoutError, match="_marked_bfa is a _MarkedTensor"):
        collect(model)


class _NoRoots(nn.Module):
    pass


def test_layer_errors():
    model = ToyModel()
    with pytest.raises(DenseLayoutError, match="does not provide dense_dwdp_roots"):
        collect_dense_layout(
            [*enumerate(model.layers), (len(model.layers), _NoRoots())],
            make_config(),
            num_ranks=NUM_RANKS,
            device=torch.device("cpu"),
            attr_allow=ATTR_ALLOW,
            exclude=EXCLUDE,
        )

    partial = ToyModel()
    layer = partial.layers[3]
    object.__setattr__(
        layer,
        "dense_dwdp_roots",
        lambda segment: [("self_attn", layer.self_attn)] if segment == "attn" else [],
    )
    with pytest.raises(DenseLayoutError, match="decoder layer 3 has dense roots for"):
        collect(partial)
    assert collect(partial, make_config(segments=("attn",))).layers == COVERED

    with pytest.raises(DenseLayoutError, match="no decoder layer has dense roots"):
        collect(ToyModel(kinds=("dense", "dense")))
    with pytest.raises(DenseLayoutError, match="nothing to move"):
        collect(ToyModel(), make_config(min_bytes=1 << 30))
    with pytest.raises(ValueError, match="strictly ascending"):
        collect_dense_layout(
            list(enumerate(model.layers))[::-1],
            make_config(),
            num_ranks=NUM_RANKS,
            device=torch.device("cpu"),
            attr_allow=ATTR_ALLOW,
            exclude=EXCLUDE,
        )

    bad_root = ToyModel()
    object.__setattr__(bad_root.layers[1], "dense_dwdp_roots", lambda segment: ["linear_attn"])
    with pytest.raises(DenseLayoutError, match="expected a \\(path, nn.Module\\) pair"):
        collect(bad_root)


# ---------------------------------------------------------------------------
# Checksums, rows and blobs
# ---------------------------------------------------------------------------


def _reference_checksum(tensor, word_offset=0):
    data = bytes(tensor.detach().reshape(-1).view(torch.uint8).tolist())
    data += bytes(-len(data) % 4)
    word_sum = weighted = 0
    for index, (word,) in enumerate(struct.iter_unpack("<i", data)):
        word_sum += word
        weighted += word * ((word_offset + index) * CHECKSUM_MULTIPLIER % (1 << 32))
    return Checksum(word_sum % (1 << 64), weighted % (1 << 64))


@pytest.mark.parametrize("small_chunks", [False, True])
def test_tensor_checksum_matches_reference(monkeypatch, small_chunks):
    if small_chunks:
        monkeypatch.setattr(dense_layout, "_CHUNK_WORDS", 64)
        monkeypatch.setattr(dense_layout, "_PARTIAL_WORDS", 16)
    generator = torch.Generator().manual_seed(7)
    raw = torch.randint(0, 256, (5003,), generator=generator, dtype=torch.uint8)
    cases = [
        (raw, 0),
        (raw[:1], 3),
        (raw[3:2050], 11),  # misaligned start, odd length
        (raw[:4096].view(torch.int32), (1 << 32) - 5),  # positions wrap the 32-bit hash
        (raw[:4096].view(torch.float32).reshape(32, 32), 1 << 40),
        (raw[2:4002].view(BF16).reshape(20, 100), 0),  # bf16 at a 2-byte storage offset
        (raw[:4000].view(FP8).reshape(40, 100), 99),
        (torch.full((4096,), -(1 << 31), dtype=torch.int32), 0),  # extreme words
        (torch.full((4096,), (1 << 31) - 1, dtype=torch.int32), 7),
        (torch.tensor(2.5), 0),  # 0-dim
    ]
    for tensor, word_offset in cases:
        assert tensor_checksum(tensor, word_offset=word_offset) == _reference_checksum(
            tensor, word_offset
        )
    assert tensor_checksum(torch.empty(0)) == Checksum(0, 0)

    # Sums of disjoint ranges with a common origin merge.
    whole = raw[:4096]
    merged = tensor_checksum(whole[:1024]).merge(tensor_checksum(whole[1024:], word_offset=256))
    assert merged == tensor_checksum(whole)

    with pytest.raises(ValueError, match="contiguous"):
        tensor_checksum(raw[:4096].view(64, 64).t())
    with pytest.raises(ValueError, match="non-negative"):
        tensor_checksum(raw, word_offset=-1)


def test_compute_checksums():
    layout = collect(ToyModel())
    compute_checksums(layout)
    for segment in layout.segments.values():
        blob = torch.full((NUM_RANKS * segment.row_bytes,), 0x5A, dtype=torch.uint8)
        write_blob(segment, blob, num_ranks=NUM_RANKS)
        assert segment.checksum == tensor_checksum(blob)
        assert segment.checksum == _reference_checksum(blob)
        for anchor in segment.anchors:
            for binding in anchor.bindings:
                assert binding.checksum == tensor_checksum(binding.original.contiguous())
    assert len(layout.checksums()) == len(layout.segments)

    anchor = next(layout.anchors())
    anchor.drop_originals()
    with pytest.raises(DenseLayoutError, match="already dropped"):
        compute_checksums(layout)


def test_rows_reassemble_the_blob():
    layout = collect(ToyModel())
    for segment in layout.segments.values():
        width = segment.row_bytes
        blob = torch.empty(NUM_RANKS * width + 100, dtype=torch.uint8).fill_(0xEE)
        write_blob(segment, blob, num_ranks=NUM_RANKS)
        assert torch.all(blob[NUM_RANKS * width :] == 0xEE)  # past N * S is untouched
        for anchor in segment.anchors:
            assert torch.equal(blob[anchor.blob_offset : anchor.blob_end], anchor.source)
        for rank in range(NUM_RANKS):
            row = torch.zeros(width, dtype=torch.uint8)
            previous_end = 0
            for piece in row_pieces(segment, rank, num_ranks=NUM_RANKS):
                assert piece.row_offset >= previous_end and piece.nbytes > 0
                previous_end = piece.row_offset + piece.nbytes
                source = segment.anchors[piece.anchor_index].source
                row[piece.row_offset : previous_end] = source[
                    piece.anchor_offset : piece.anchor_offset + piece.nbytes
                ]
            assert torch.equal(row, blob[rank * width : (rank + 1) * width])
    with pytest.raises(ValueError, match="outside"):
        row_pieces(segment, NUM_RANKS, num_ranks=NUM_RANKS)
    with pytest.raises(ValueError, match="cannot hold"):
        row_pieces(layout.segment(1, ATTN), 0, num_ranks=1)


def test_write_blob_validation():
    layout = collect(ToyModel())
    segment = layout.segment(1, ATTN)
    total = NUM_RANKS * segment.row_bytes
    with pytest.raises(ValueError, match="holds"):
        write_blob(segment, torch.zeros(total - 1, dtype=torch.uint8), num_ranks=NUM_RANKS)
    with pytest.raises(ValueError, match="flat uint8"):
        write_blob(segment, torch.zeros(total, dtype=torch.int8), num_ranks=NUM_RANKS)
    with pytest.raises(ValueError, match="flat uint8"):
        write_blob(segment, torch.zeros(2, total, dtype=torch.uint8), num_ranks=NUM_RANKS)
    out = torch.full((total,), 7, dtype=torch.uint8)
    segment.anchors[-1].drop_originals()
    with pytest.raises(DenseLayoutError, match="already dropped"):
        write_blob(segment, out, num_ranks=NUM_RANKS)
    assert torch.all(out == 7)  # nothing written before the check


def test_layout_is_a_dense_layout():
    layout = collect(ToyModel())
    assert isinstance(layout, DenseLayout)
    assert layout.num_ranks == NUM_RANKS
    assert [layout.slot_index(ATTN, layer) for layer in COVERED] == [0, 1, 0, 1, 0]
    assert [layout.slot_index(MOE, layer) for layer in COVERED] == [0, 1, 2, 0, 1]
