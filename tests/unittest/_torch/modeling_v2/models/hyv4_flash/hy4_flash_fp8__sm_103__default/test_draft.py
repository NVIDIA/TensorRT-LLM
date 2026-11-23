# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFly2 drafter's geometry and weight table, held against the checkpoint.

CPU only: no weight is read and no kernel is called. The drafter's table is
declared from `draft_dims`, and every declared shape, cut and dtype is checked
against the checkpoint's own `model.safetensors.index.json` and the safetensors
headers it points at -- so a table that covers the wrong tensors, cuts a rank's
share at the wrong row, or declares bf16 where the checkpoint stores fp8 fails
here rather than on a GPU as a silent numerical error.

Two widths are not published anywhere in `config.json` (the FFN's 9216 and the
correction MLP's 3072; the reference implementations stream them off the first
tensor they read), so they are `draft_dims`'s derivation and these tests hold
them against the checkpoint's own headers.

What this module cannot see: whether the forward that consumes the table is
right, whether the engine's block metadata reaches it, and whether a draft is
ever accepted. Those belong to the drafter's forward and to the acceptance gate.
"""

import json
import math
import os
import struct

import pytest
import torch

# The checkpoint stand-in this directory's tests share; pytest puts the test
# file's own directory on the path.
from _target import CKPT, core, requires_checkpoint

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.dims import (  # noqa: E501
    DRAFT_INDEX_ALIGN,
    draft_dims,
    round_up,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.spec.dfly_block_prepare import (  # noqa: E501
    BLOCK_WIDTH,
    MASK_TOKEN_ID,
    PAGE_ROWS,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.weights import (  # noqa: E501
    DRAFT_WEIGHTS,
    MODEL_WEIGHTS,
    draft_host,
)

pytestmark = requires_checkpoint

#: safetensors spells dtypes its own way.
DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def index():
    """The checkpoint's `model.safetensors.index.json` weight map."""
    with open(os.path.join(CKPT, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def headers(keys):
    """`{checkpoint key: (dtype, shape)}` for `keys`, from the safetensors headers.

    A safetensors header is JSON behind an 8-byte length; reading it gives every
    tensor's dtype and shape without touching a byte of weight data.
    """
    by_shard: dict[str, list[str]] = {}
    weight_map = index()
    for key in keys:
        by_shard.setdefault(weight_map[key], []).append(key)
    out = {}
    for shard, shard_keys in by_shard.items():
        with open(os.path.join(CKPT, shard), "rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(length))
        for key in shard_keys:
            entry = header[key]
            out[key] = (DTYPES[entry["dtype"]], tuple(entry["shape"]))
    return out


def dfly_keys():
    """Every `model.dfly_layers.*` key the checkpoint publishes."""
    return sorted(k for k in index() if k.startswith("model.dfly_layers."))


def manifest_rows(c=None):
    """`(dims, [(param key, checkpoint key, cut, transform), ...])` for the drafter."""
    stand_in = c or core()
    # The engine's `ModelConfig.torch_dtype` is a `torch.dtype`, which is what
    # `declare` allocates with; the family's checkpoint stand-in keeps the JSON
    # spelling, so hand over what the engine hands over.
    stand_in.model_config.torch_dtype = torch.bfloat16
    host = draft_host(stand_in)
    rows = [
        (param, ckpt_key, cut, transform)
        for param, sources in DRAFT_WEIGHTS.manifest(host).items()
        for ckpt_key, _, cut, transform in sources
    ]
    return host.dims, rows


def _cut(shape, cut):
    """The checkpoint shape with this rank's slices applied."""
    out = list(shape)
    for axis, part in enumerate(cut or ()):
        start = 0 if part.start is None else part.start
        stop = out[axis] if part.stop is None else part.stop
        out[axis] = stop - start
    return tuple(out)


# --- geometry ------------------------------------------------------------------


def test_draft_geometry_from_the_checkpoint():
    d = draft_dims(core())
    assert d.num_layers == 5
    # The five draft layers reuse the five sparse target layers the config names.
    assert d.target_kv_layers == (25, 29, 33, 37, 41)
    assert d.target_index_layers == d.target_kv_layers
    assert d.index_topk == 2048
    assert d.context_local_window is None
    assert d.mask_token_id == 120009
    assert d.mtp_fusion == "slot0"
    assert d.hidden_correction is True


def test_draft_attention_is_the_target_sparse_layer():
    d = draft_dims(core())
    assert (d.heads, d.heads_local) == (32, 8)
    assert (d.nope, d.rope_dim, d.v_dim, d.kv_lora, d.q_lora) == (192, 64, 256, 512, 1024)
    assert d.qk_dim == 256
    assert d.lat_dim == 576
    # A draft layer absorbs its queries as the target's sparse layers do, so the
    # effective softmax scale is the checkpoint's own 1/sqrt(256).
    assert d.q_scaling == 1.0


def test_draft_widths_the_config_does_not_publish():
    d = draft_dims(core())
    assert d.mlp_inter == 9216
    assert d.correction_inter == 3072
    assert d.fusion_in == 6144
    # Both are multiples of the hidden size; the checkpoint's headers below are
    # what makes them facts rather than a pattern.
    assert d.mlp_inter % d.hidden == 0
    assert d.correction_inter % d.hidden == 0
    assert d.fusion_in == 2 * d.hidden


def test_draft_block_and_index_widths():
    d = draft_dims(core())
    # Six rows per request: the five drafts and the anchor in column 0.
    assert d.block_width == 6
    assert d.block_width == BLOCK_WIDTH
    assert d.mask_token_id == MASK_TOKEN_ID
    # A row attends that target layer's 2048 picks plus its own block's slots.
    assert d.picked_rows == 2054
    assert d.index_width == 2176
    assert d.index_width == round_up(d.picked_rows, DRAFT_INDEX_ALIGN)
    assert d.index_width % DRAFT_INDEX_ALIGN == 0
    assert PAGE_ROWS == 32


@pytest.mark.parametrize("tp, heads_local", [(4, 8), (8, 4)])
def test_draft_head_split_follows_the_attention_group(tp, heads_local):
    d = draft_dims(core(tp=tp, ep=tp, world=tp))
    assert d.heads_local == heads_local
    # The head cut and the block-FP8 scale cut must both land on a boundary.
    assert d.heads_local * d.qk_dim % 128 == 0
    assert d.heads_local * d.v_dim % 128 == 0


def test_a_split_kv_and_index_layer_is_refused():
    """A draft layer reads one layer's latent pool and that same layer's picks."""
    c = core()
    c.model_config.pretrained_config.dfly_target_index_layer = [25, 29, 33, 37, 40]
    with pytest.raises(AssertionError):
        draft_dims(c)


def test_a_target_layer_that_is_not_sparse_is_refused():
    """Only a sparse target layer has the pooled picks a draft row attends."""
    c = core()
    c.model_config.pretrained_config.dfly_target_kv_layers = [25, 29, 33, 37, 26]
    c.model_config.pretrained_config.dfly_target_index_layer = [25, 29, 33, 37, 26]
    with pytest.raises(AssertionError):
        draft_dims(c)


# --- the weight table ----------------------------------------------------------


def test_the_table_covers_exactly_the_drafters_tensors():
    _, rows = manifest_rows()
    covered = sorted({ckpt for _, ckpt, _, _ in rows})
    assert covered == dfly_keys()
    assert len(covered) == 117
    # Five layers of 21: an MLA layer without the indexer the drafter does not
    # own, plus the FFN and the two sandwich norms.
    assert len([k for k in covered if k.split(".")[2].isdigit()]) == 5 * 21
    singletons = sorted(k for k in covered if not k.split(".")[2].isdigit())
    assert singletons == [
        "model.dfly_layers.confidence_head.proj.bias",
        "model.dfly_layers.confidence_head.proj.weight",
        "model.dfly_layers.final_norm.weight",
        "model.dfly_layers.hidden_correction.down_proj.weight",
        "model.dfly_layers.hidden_correction.embed_norm.weight",
        "model.dfly_layers.hidden_correction.gate_proj.weight",
        "model.dfly_layers.hidden_correction.hidden_norm.weight",
        "model.dfly_layers.hidden_correction.up_proj.weight",
        "model.dfly_layers.mtp_fusion.eh_proj.weight",
        "model.dfly_layers.mtp_fusion.enorm.weight",
        "model.dfly_layers.mtp_fusion.gate",
        "model.dfly_layers.mtp_fusion.hnorm.weight",
    ]


def test_the_manifest_covers_every_declared_parameter():
    host = draft_host(core())
    manifest = DRAFT_WEIGHTS.manifest(host)
    _, rows = manifest_rows()
    assert set(manifest) == {p for p, _, _, _ in rows}
    # Five layers of 22 (kv_b_proj's two operands count twice) and the twelve
    # singletons -- 122 destinations over the 117 checkpoint keys.
    assert len(manifest) == 5 * 22 + 12
    assert {"l0_q_b_proj", "l4_mlp_down", "final_norm", "corr_down"} <= set(manifest)
    # One checkpoint key, two destinations: kv_b_proj's K and V operands.
    kv = [p for p, ckpt, _, _ in rows if ckpt.endswith("self_attn.kv_b_proj.weight")]
    assert len(kv) == 5 * 2
    assert all(t is not None for p, ckpt, _, t in rows if ckpt.endswith("kv_b_proj.weight"))


def test_declared_shapes_cuts_and_dtypes_match_the_checkpoint():
    """Every declared shape, cut and dtype, against the checkpoint's own headers.

    A row with no transform must be *exactly* the checkpoint's tensor cut to this
    rank's share. A relayout (`kv_b_proj`, whose one tensor becomes the K and V
    operands by reshape, split and transpose) cannot be compared by shape, and it
    is not one destination either: the operands it makes together conserve the
    cut source's element count, so that is what is checked.
    """
    d, rows = manifest_rows()
    w = DRAFT_WEIGHTS.declare(d)
    checkpoint = headers({ckpt for _, ckpt, _, _ in rows})

    by_ckpt: dict[str, list] = {}
    for param, ckpt, cut, transform in rows:
        by_ckpt.setdefault(ckpt, []).append((param, cut, transform))

    for ckpt, group in by_ckpt.items():
        src_dtype, src_shape = checkpoint[ckpt]
        for param, cut, transform in group:
            dst = w[param]
            if transform is None:
                expected = _cut(src_shape, cut)
                assert tuple(dst.shape) == expected, (param, ckpt, tuple(dst.shape), expected)
                assert dst.dtype == src_dtype, (param, ckpt, dst.dtype, src_dtype)
        relayout = [(p, cut) for p, cut, transform in group if transform is not None]
        if relayout:
            expected = _cut(src_shape, relayout[0][1])
            total = sum(w[param].numel() for param, _ in relayout)
            assert total == math.prod(expected), (ckpt, total, expected, [p for p, _ in relayout])


def test_the_drafters_block_fp8_set_is_the_cores():
    """The same four expansions the core stores in FP8 are FP8 in the drafter."""
    d, rows = manifest_rows()
    w = DRAFT_WEIGHTS.declare(d)
    fp8 = {k.split("_", 1)[1] for k, p in w.items() if p.dtype == torch.float8_e4m3fn}
    assert fp8 == {"q_b_proj", "linear_gate", "o_proj", "mlp_gate", "mlp_up", "mlp_down"}
    assert sum(1 for p in w.values() if p.dtype == torch.float8_e4m3fn) == 5 * 6
    # The draft sink is bf16 in the checkpoint, where the core's is fp32. The
    # parameter is fp32 all the same: the attention op reads the sink as a raw
    # fp32 buffer and refuses any other dtype, so the read upcasts it.
    sink = headers({"model.dfly_layers.0.self_attn.learnable_sink_param"})
    checkpoint_dtype = sink["model.dfly_layers.0.self_attn.learnable_sink_param"][0]
    assert checkpoint_dtype == torch.bfloat16, checkpoint_dtype
    assert w["l0_sink"].dtype == torch.float32


def test_the_drafters_table_and_the_cores_partition_the_checkpoint():
    """Neither unit loads a key the other owns, and the two together leave nothing."""
    keys = {k: None for k in index()}
    core_left = MODEL_WEIGHTS.expected_unconsumed(None, keys)
    draft_left = DRAFT_WEIGHTS.expected_unconsumed(None, keys)
    assert dfly_keys() and set(dfly_keys()) <= core_left
    # Everything that is not the drafter's is either the core's or the tower's.
    assert draft_left == set(keys) - set(dfly_keys())
    assert dfly_keys()[0] not in draft_left
