# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Inkling-2-Small TP4/EP4 weight table against the checkpoint's tensor layout.

The layout below restates the released checkpoint's text-decoder tensors (names, shapes, dtypes),
as its safetensors headers declare them. Every rank's transforms run on meta tensors of those
shapes, so the table's declared shares, its sources and the checkpoint cannot drift apart
unnoticed.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.inkling2.config import ExpertFormat, InklingDims
from tensorrt_llm._torch.modeling_v2.models.inkling2.inkling_2_small_nvfp4__sm_103__tep4 import (
    weights as w,
)

pytestmark = pytest.mark.cpu_only

U8, F8, BF16, F32 = torch.uint8, torch.float8_e4m3fn, torch.bfloat16, torch.float32
GLOBAL = tuple(range(5, 48, 6))


def _dims() -> InklingDims:
    text = SimpleNamespace(
        num_hidden_layers=48,
        local_layer_ids=[i for i in range(48) if i not in GLOBAL],
        hidden_size=4608,
        vocab_size=201024,
        unpadded_vocab_size=200058,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=128,
        d_rel=16,
        rel_extent=1024,
        sliding_window_size=1024,
        sconv_kernel_size=4,
        n_layers_per_attn_res_block=6,
        n_routed_experts=384,
        num_experts_per_tok=8,
        intermediate_size=2304,
        route_scale=8.0,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        final_logit_softcapping=30.0,
        qk_l2_clip_threshold=22.627416997969522,
        rel_l2_clip_threshold=8.0,
        log_scaling_alpha=0.1,
        log_scaling_n_floor=8192,
        rms_norm_eps=1e-6,
    )
    groups = [
        {
            "compute_dtype": "nvfp4",
            "tensors": [f"model.llm.layers.{i}.mlp.experts.w13_weight" for i in range(40)],
        },
        {
            "compute_dtype": "mxfp8",
            "tensors": [f"model.llm.layers.{i}.mlp.experts.w13_weight" for i in range(40, 48)],
        },
    ]
    return InklingDims.from_config(text, {"model": {"parameter_groups": groups}})


def checkpoint_layout(d: InklingDims) -> dict:
    """name -> (shape, dtype) of the released checkpoint's text-decoder tensors."""
    h, e, i = d.hidden_size, d.num_experts, d.expert_intermediate_size
    out = {
        "model.llm.embed.weight": ((d.vocab_size, h), BF16),
        "model.llm.embed_norm.weight": ((h,), BF16),
        "model.llm.norm.weight": ((h,), BF16),
        "model.llm.unembed.weight": ((d.vocab_size, h), BF16),
        "model.llm.attn_res.output_query": ((h,), F32),
    }
    for layer in range(d.num_layers):
        p, a = f"model.llm.layers.{layer}", f"model.llm.layers.{layer}.attn"
        out[f"model.llm.attn_res.mlp_queries.{layer}"] = ((h,), F32)
        if layer > 0:
            out[f"model.llm.attn_res.attn_queries.{layer - 1}"] = (
                (3, h) if d.is_global(layer) else (h,),
                F32,
            )
        out.update({
            f"{a}.wq_du.weight": ((6144, h), BF16),
            f"{a}.wk_dv.weight": ((1024, h), BF16),
            f"{a}.wv_dv.weight": ((1024, h), BF16),
            f"{a}.wr_du.weight": ((768, h), BF16),
            f"{a}.wo_ud.weight": ((h, 6144), BF16),
            f"{a}.gate_attn.weight": ((6144, h), BF16),
            f"{a}.attn_sink_proj.weight": ((48, h), BF16),
            f"{a}.q_l2_clip.scale": ((48, 128), BF16),
            f"{a}.rel_l2_clip.scale": ((48, 16), BF16),
            f"{a}.rel_logits_proj.proj": ((16, 1024), BF16),
            f"{p}.attn_sconv.weight": ((h, 1, 4), BF16),
            f"{p}.mlp_sconv.weight": ((h, 1, 4), BF16),
            f"{p}.mlp_norm.weight": ((h,), BF16),
            f"{p}.mlp.gate.weight": ((e, h), BF16),
            f"{p}.mlp.gate.bias": ((e,), F32),
        })  # fmt: skip
        if d.is_global(layer):
            for s in ("k", "v"):
                out[f"{a}.{s}_sconv.static_w"] = ((4, 1024), BF16)
                out[f"{a}.{s}_sconv.weight_proj"] = ((32, h), BF16)
            for n in ("q", "k", "v"):
                out[f"{p}.{n}_input_norm.weight"] = ((h,), BF16)
        else:
            out[f"{a}.k_sconv.weight"] = ((1024, 1, 4), BF16)
            out[f"{a}.v_sconv.weight"] = ((1024, 1, 4), BF16)
            out[f"{p}.attn_norm.weight"] = ((h,), BF16)
        x = f"{p}.mlp.experts"
        if d.expert_formats[layer] is ExpertFormat.NVFP4:
            out.update({
                f"{x}.w13_weight": ((e, 2 * i, h // 2), U8),
                f"{x}.w13_weight_scale": ((e, 2 * i, h // 16), F8),
                f"{x}.w13_weight_amax": ((e,), F32),
                f"{x}.w13_weight_outer_scale_denominator": ((1,), F32),
                f"{x}.w2_weight": ((e, h, i // 2), U8),
                f"{x}.w2_weight_scale": ((e, h, i // 16), F8),
                f"{x}.w2_weight_amax": ((e,), F32),
                f"{x}.w2_weight_outer_scale_denominator": ((1,), F32),
            })  # fmt: skip
        else:
            out.update({
                f"{x}.w13_weight": ((e, 2 * i, h), F8),
                f"{x}.w13_weight_scale": ((e, 2 * i, h // 32), U8),
                f"{x}.w2_weight": ((e, h, i), F8),
                f"{x}.w2_weight_scale": ((e, h, i // 32), U8),
            })  # fmt: skip
    return out


def _mapping(rank: int):
    return SimpleNamespace(
        pp_size=1,
        cp_size=1,
        enable_attention_dp=False,
        moe_tp_size=1,
        moe_ep_size=4,
        tp_size=4,
        tp_rank=rank,
        moe_ep_rank=rank,
    )


@pytest.mark.parametrize("rank", range(4))
def test_every_entry_transforms_to_its_declared_share(rank):
    d = _dims()
    s = w.shares_of(d, _mapping(rank))
    layout = checkpoint_layout(d)
    used = set()
    for layer, entry in w.all_entries(s):
        flags = entry.expert_rows or (False,) * len(entry.sources)
        srcs = []
        for name, rows in zip(entry.sources, flags):
            assert name in layout, f"layer {layer} {entry.name}: {name} is not a checkpoint tensor"
            shape, dtype = layout[name]
            if rows:
                shape = (s.local_experts, *shape[1:])
            srcs.append(torch.empty(shape, dtype=dtype, device="meta"))
            assert name not in used, f"{name} is read twice"
            used.add(name)
        out = entry.transform(tuple(srcs), s)
        if out.dtype != entry.dtype and out.dtype == U8 and entry.dtype.itemsize == 1:
            out = out.view(entry.dtype)
        assert tuple(out.shape) == tuple(entry.shape), (
            layer,
            entry.name,
            tuple(out.shape),
            entry.shape,
        )
        assert out.dtype == entry.dtype, (layer, entry.name, out.dtype, entry.dtype)
    used.add("model.llm.unembed.weight")
    assert used == set(layout), sorted(set(layout) ^ used)[:8]


def test_shares_split_heads_and_experts_evenly():
    s = w.shares_of(_dims(), _mapping(2))
    assert (s.heads, s.kv_heads, s.q_width, s.kv_width, s.r_width) == (12, 2, 1536, 256, 192)
    assert (s.local_experts, s.expert_offset, s.experts()) == (96, 192, slice(192, 288))
    assert s.rows(6144) == slice(3072, 4608) and s.rows(32) == slice(16, 24)


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(os.environ.get("LLM_MODELS_ROOT", ""), "Inkling-2-Small")),
    reason="needs the Inkling-2-Small checkpoint under LLM_MODELS_ROOT",
)
def test_layout_matches_the_checkpoint_headers():
    import json
    import struct

    root = os.path.join(os.environ["LLM_MODELS_ROOT"], "Inkling-2-Small")
    index = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
    headers = {}
    for f in sorted(set(index.values())):
        with open(os.path.join(root, f), "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            headers.update(json.loads(fh.read(n)))
    names = {"BF16": BF16, "F32": F32, "U8": U8, "F8_E4M3": F8}
    actual = {
        k: (tuple(v["shape"]), names[v["dtype"]])
        for k, v in headers.items()
        if k.startswith("model.llm.")
    }
    assert actual == {k: (tuple(v[0]), v[1]) for k, v in checkpoint_layout(_dims()).items()}
