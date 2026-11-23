# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's geometry, derived from the checkpoint's own `config.json`, index and tensors.

CPU only: no weights are read into a tensor, no kernel is called. These are the
arithmetic facts the manifest and the forward both stand on, checked against the
checkpoint the target serves rather than against a copy of the numbers -- the
config for the geometry, the safetensors headers for the declared shapes.

The five layouts are the ones `dims` documents; a sixth (a head count that does
not divide the attention group, an expert count that does not divide the EP
group) must raise, because the manifest would otherwise silently drop rows.
"""

import json
import os
import struct
from types import SimpleNamespace

import pytest

# The checkpoint stand-in this directory's tests share; pytest puts the test
# file's own directory on the path.
from _target import CKPT, core, requires_checkpoint

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.dims import (  # noqa: E501
    THETA_SPARSE,
    THETA_SWA,
    dims,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.weights import (  # noqa: E501
    MODEL_WEIGHTS,
)

pytestmark = requires_checkpoint


def test_layer_pattern_from_the_checkpoint():
    d = dims(core())
    assert d.num_layers == 42
    # Two initial sparse layers, then ten repetitions of three windowed and one
    # sparse layer, closing on a sparse layer 41.
    assert d.sparse_layers == (0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41)
    assert sum(1 for g in d.layers if not g.sparse) == 30
    assert [g.sparse for g in d.layers[:4]] == [True, True, False, False]
    assert [g.sparse for g in d.layers[2:6]] == [False, False, False, True]


def test_rope_base_per_layer_type():
    d = dims(core())
    assert d.layers[0].theta == THETA_SPARSE
    assert d.layers[2].theta == THETA_SWA
    assert {g.theta for g in d.layers} == {THETA_SPARSE, THETA_SWA}
    assert d.mrope_section == (12, 10, 10)


def test_mla_geometry():
    d = dims(core())
    assert (d.nope, d.rope_dim, d.v_dim, d.kv_lora, d.q_lora) == (192, 64, 256, 512, 1024)
    assert d.qk_dim == 256
    assert d.lat_dim == 576
    # The sparse branch's effective softmax scale is the checkpoint's 1/sqrt(256).
    assert d.q_scaling == 1.0


def test_engram_layers_and_table_rows():
    d = dims(core())
    assert d.engram_layers == (1, 11, 21, 31)
    assert d.engram_num_heads == 16
    assert d.engram_head_dim == 96
    assert d.engram_max_ngram == 3
    for layer in d.engram_layers:
        sizes = d.engram_head_vocab_sizes[d.engram_ids[d.engram_layers.index(layer)]]
        rows = d.engram_table_rows[layer]
        offsets = d.engram_head_offsets[layer]
        assert rows == sum(sizes)
        assert offsets[0] == 0
        assert offsets == [sum(sizes[:k]) for k in range(len(sizes))]
        assert offsets[-1] + sizes[-1] == rows
    assert d.engram_table_rows[1] == 164994380


def test_engram_table_rows_match_the_index():
    index_path = os.path.join(CKPT, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        pytest.skip("no safetensors index beside the checkpoint")
    with open(index_path) as f:
        index = json.load(f)
    d = dims(core())
    for layer in d.engram_layers:
        key = f"model.layers.{layer}.engram_layer.multi_head_embedding.embedding.weight"
        assert key in index["weight_map"], key
    # The index does not carry shapes; the row count is the sum of the layer's own
    # head moduli, which the loader compares against the tensor it reads.


@pytest.mark.parametrize(
    "tp, ep, moe_tp, attn_dp, world, expected_heads, expected_experts",
    [
        (2, 2, 1, False, 2, (16, 32), 256),  # tep2: the disagg deployment's context engine
        (4, 1, 4, False, 4, (8, 16), 512),  # tp4: no EP, all experts on a quarter of the width
        (4, 4, 1, False, 4, (8, 16), 128),  # tep4
        (4, 4, 1, True, 4, (32, 64), 128),  # dep4: attention whole on every rank
        (8, 8, 1, False, 8, (4, 8), 64),  # tep8, two nodes
    ],
)
def test_member_layouts(tp, ep, moe_tp, attn_dp, world, expected_heads, expected_experts):
    d = dims(core(tp=tp, ep=ep, attn_dp=attn_dp, world=world, moe_tp=moe_tp))
    assert (d.heads_sparse, d.heads_swa) == (32, 64)
    assert (d.layers[0].heads_local, d.layers[2].heads_local) == expected_heads
    assert d.local_experts == expected_experts
    assert d.expert_offset in (0, 128, 256, 384, 64, 192, 320, 448)
    assert d.attn_tp == (1 if attn_dp else tp)
    # The MoE TP split falls on the experts' intermediate; the latent -- their
    # input width -- is whole on every rank, at both MoE TP widths.
    assert d.moe_inter == d.latent // moe_tp
    assert d.latent == 1536


def test_moe_and_dense_widths():
    d = dims(core())
    assert (d.num_experts, d.top_k, d.n_group, d.topk_group) == (512, 16, 1, 1)
    assert d.routed_scale == pytest.approx(3.998)
    assert d.swiglu_limit == pytest.approx(10.0)
    assert d.latent == 1536
    assert d.first_dense == 1
    assert d.hidden == 3072


def test_indexer_geometry():
    d = dims(core())
    assert (d.index_topk, d.index_kpool, d.index_n_heads, d.index_head_dim) == (2048, 4, 16, 128)
    assert d.index_group_topk == 512


def test_vision_geometry():
    """The tower's block of `vision_config`, which the manifest's `vis_*` shapes and the
    placeholder arithmetic both read."""
    d = dims(core())
    v = d.vision
    assert (v.depth, v.hidden, v.heads, v.head_dim) == (27, 1152, 16, 72)
    assert (v.inter, v.merge, v.patch, v.channels) == (4304, 2, 16, 3)
    assert (v.out_hidden, v.merge_dim, v.pixel_width) == (3072, 4608, 768)
    assert v.eps == pytest.approx(1e-5)
    # The position grid is (max_image_size / patch) squared -- the ceiling a frame has to
    # fit -- and the config's max_vit_seq_len repeats it: 16384 patch rows per call.
    assert (v.position_edge, v.position_edge**2) == (128, 16384)
    # `img_max_token_num` is the checkpoint's inference-time clamp on the processor, not a
    # tower limit; it is carried here for the placeholder check, not enforced by the tower.
    assert v.img_max_token_num == 4096
    # The tower is whole on every rank, and every layout derives the same geometry.
    assert dims(core(tp=8, ep=8, world=8)).vision == v
    assert dims(core(attn_dp=True, world=4)).vision == v


def test_undividable_head_count_is_refused():
    with pytest.raises(AssertionError):
        dims(core(tp=7, ep=1, world=7))


def test_undividable_expert_count_is_refused():
    with pytest.raises(AssertionError):
        dims(core(tp=4, ep=3, world=4))


def _checkpoint_tensor_shapes(keys) -> dict[str, tuple[int, ...]]:
    """The checkpoint's own shapes for `keys`, from the safetensors headers.

    The index gives each tensor's shard, not its shape; the shard's own header
    (JSON after its first eight bytes) does. One read per shard, cached here.
    """
    with open(os.path.join(CKPT, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    headers: dict[str, dict] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for key in keys:
        shard = weight_map[key]
        if shard not in headers:
            with open(os.path.join(CKPT, shard), "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                headers[shard] = json.loads(f.read(header_len))
        shapes[key] = tuple(headers[shard][key]["shape"])
    return shapes


@pytest.mark.parametrize(
    "moe_tp, tp, ep, member",
    [
        (4, 4, 1, "tp4"),  # MoE TP 4: every expert on every rank, a quarter of the width
        (1, 4, 4, "tep4"),  # MoE EP 4 / MoE TP 1: each expert's whole intermediate
    ],
)
def test_routed_expert_pair_matches_the_checkpoint(moe_tp, tp, ep, member):
    """The latent's two projections are whole on every rank, at both MoE TP widths.

    `routed_expert_down_proj` (hidden -> latent) and `routed_expert_up_proj`
    (latent -> hidden) are the weights around the routed experts whose widths the
    MoE TP split never touches: it falls on the experts' own intermediate. So the
    manifest declares the checkpoint tensor's own shape for the pair on every
    rank -- the same pair at tp4's `moe_tp` 4 and tep4's 1 -- and reads it with no
    cut and no transform. A declaration of `(hidden, moe_inter)` would name the
    per-rank intermediate as the K of a matmul whose input is the whole latent,
    and the loader's shape check refuses the read: the tp4 engine met exactly that
    at layer 1, `(3072, 384)` declared against the checkpoint's `(3072, 1536)`.
    """
    c = core(tp=tp, ep=ep, moe_tp=moe_tp)
    d = dims(c)
    # The manifest reads the model core's `dims`; the stand-in above is the pair
    # `dims` itself reads (the checkpoint's config and one member's Mapping).
    manifest = MODEL_WEIGHTS.manifest(SimpleNamespace(dims=d, model_config=c.model_config))
    entries = {entry.name: entry for entry in MODEL_WEIGHTS.WEIGHTS}
    pairs = {
        "pre_expert": "routed_expert_down_proj",
        "post_expert": "routed_expert_up_proj",
    }
    assert set(pairs) <= set(entries)
    assert d.moe_inter == d.latent // moe_tp

    moe_layers = range(d.first_dense, d.num_layers)
    checkpoint_shapes = _checkpoint_tensor_shapes(
        [
            f"model.layers.{i}.mlp.{ckpt_name}.weight"
            for i in moe_layers
            for ckpt_name in pairs.values()
        ]
    )

    for i in moe_layers:
        for name, ckpt_name in pairs.items():
            ckpt_key = f"model.layers.{i}.mlp.{ckpt_name}.weight"
            param_key = f"l{i}_{name}"
            assert param_key in manifest, param_key
            sources = manifest[param_key]
            assert len(sources) == 1, (param_key, sources)
            src_key, index, cut, transform = sources[0]
            assert src_key == ckpt_key, (param_key, src_key)
            assert index is None and cut is None and transform is None, (
                param_key,
                index,
                cut,
                transform,
            )
            declared = entries[name].shape(d, i)
            assert declared == checkpoint_shapes[ckpt_key], (
                member,
                param_key,
                declared,
                checkpoint_shapes[ckpt_key],
            )
