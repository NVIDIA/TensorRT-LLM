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
from tensorrt_llm._torch.modeling_v2.models.inkling2.ops.moe.inkling2_nvfp4_experts import (
    unswizzle_block_scales,
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
    """name -> (shape, dtype) of the checkpoint's text-decoder tensors for geometry `d`.

    For `_dims()` these are the released checkpoint's shapes.
    """
    h, e, i = d.hidden_size, d.num_experts, d.expert_intermediate_size
    q, kv, r = d.num_heads * d.head_dim, d.num_kv_heads * d.head_dim, d.num_heads * d.d_rel
    k = d.conv_kernel_size
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
            f"{a}.wq_du.weight": ((q, h), BF16),
            f"{a}.wk_dv.weight": ((kv, h), BF16),
            f"{a}.wv_dv.weight": ((kv, h), BF16),
            f"{a}.wr_du.weight": ((r, h), BF16),
            f"{a}.wo_ud.weight": ((h, q), BF16),
            f"{a}.gate_attn.weight": ((q, h), BF16),
            f"{a}.attn_sink_proj.weight": ((d.num_heads, h), BF16),
            f"{a}.q_l2_clip.scale": ((d.num_heads, d.head_dim), BF16),
            f"{a}.rel_l2_clip.scale": ((d.num_heads, d.d_rel), BF16),
            f"{a}.rel_logits_proj.proj": ((d.d_rel, d.rel_extent_of(layer)), BF16),
            f"{p}.attn_sconv.weight": ((h, 1, k), BF16),
            f"{p}.mlp_sconv.weight": ((h, 1, k), BF16),
            f"{p}.mlp_norm.weight": ((h,), BF16),
            f"{p}.mlp.gate.weight": ((e, h), BF16),
            f"{p}.mlp.gate.bias": ((e,), F32),
        })  # fmt: skip
        if d.is_global(layer):
            for s in ("k", "v"):
                out[f"{a}.{s}_sconv.static_w"] = ((k, kv), BF16)
                out[f"{a}.{s}_sconv.weight_proj"] = ((d.num_kv_heads * k, h), BF16)
            for n in ("q", "k", "v"):
                out[f"{p}.{n}_input_norm.weight"] = ((h,), BF16)
        else:
            out[f"{a}.k_sconv.weight"] = ((kv, 1, k), BF16)
            out[f"{a}.v_sconv.weight"] = ((kv, 1, k), BF16)
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


def test_transforms_put_values_where_the_kernels_read_them():
    """Real tensors whose values name their source row, against expectations built by hand."""
    s = w.shares_of(_dims(), _mapping(2))
    # A conv weight stores the oldest tap first; the kernels read lag 0, the newest input, first.
    taps = torch.arange(8, dtype=F32).reshape(2, 1, 4)
    assert w._conv_taps((taps,), s).tolist() == [[3.0, 7.0], [2.0, 6.0], [1.0, 5.0], [0.0, 4.0]]
    # A sliding layer's k/v conv: this rank's 256 of the 1024 channels, reversed the same way.
    kv_taps = torch.arange(1024 * 4, dtype=F32).reshape(1024, 1, 4)
    out = w._kv_conv_taps((kv_taps,), s)
    assert out.shape == (4, 256)
    assert all(
        out[lag, c] == kv_taps[512 + c, 0, 3 - lag] for lag in range(4) for c in (0, 77, 255)
    )
    # A global layer's static taps are lag-major already: this rank's channels.
    static = torch.arange(4 * 1024, dtype=F32).reshape(4, 1024)
    assert torch.equal(w._static_taps((static,), s), static[:, 512:768])
    # Stacked projections: each source's rows of this rank, in the order the entry names them.
    sizes = (6144, 768, 6144, 48)  # wq, wr, gate, sink
    srcs = tuple(torch.arange(n, dtype=F32)[:, None] + 100_000 * k for k, n in enumerate(sizes))
    stacked = w._cat_rows(srcs, s)[:, 0].long().tolist()
    rank_rows = [(k, range(n // 2, 3 * n // 4)) for k, n in enumerate(sizes)]  # rank 2 of 4
    assert stacked == [100_000 * k + r for k, rows in rank_rows for r in rows]
    # A global layer's k or v rows, then its dynamic conv's tap-projection rows.
    proj = torch.arange(1024, dtype=F32)[:, None]
    tap_proj = torch.arange(32, dtype=F32)[:, None] + 100_000
    out = w._kv_with_taps((proj, tap_proj), s)[:, 0].long().tolist()
    assert out == list(range(512, 768)) + [100_000 + r for r in range(16, 24)]
    # The output projection keeps this rank's input columns.
    wo = torch.arange(6144, dtype=F32)[None, :].expand(2, 6144)
    assert torch.equal(w._o_cols((wo,), s), wo[:, 3072:4608])
    # The NVFP4 global scale: this rank's experts' amax over the outer denominator.
    amax = torch.arange(384, dtype=F32)
    alpha = w._nvfp4_alpha((amax, torch.tensor([2688.0])), s)
    assert torch.equal(alpha, amax[192:288] / 2688.0)


def test_nvfp4_fc1_weight_and_scale_rows_move_together():
    """Kernel row k of the FC1 weight and of its block scales is the same checkpoint row."""
    i, h = 2304, 4608
    ids = torch.arange(2 * i)
    # Each checkpoint row carries its own index in its first two bytes.
    tag = torch.stack([ids % 256, ids // 256], dim=-1).to(U8)
    w13 = torch.cat([tag, torch.zeros(2 * i, h // 2 - 2, dtype=U8)], dim=-1)[None]
    w13_sf = torch.cat([tag, torch.zeros(2 * i, h // 16 - 2, dtype=U8)], dim=-1)[None].view(F8)
    s = w.shares_of(_dims(), _mapping(0))

    def rows_of(t):
        return (t[0, :, 0].long() + 256 * t[0, :, 1].long()).tolist()

    # The checkpoint interleaves gate (even) and up (odd) rows; the kernel wants, per 64 rows of
    # each, the up rows first.
    expected = []
    for group in range(i // 64):
        expected += [2 * (64 * group + j) + 1 for j in range(64)]
        expected += [2 * (64 * group + j) for j in range(64)]
    assert rows_of(w._nvfp4_w13((w13,), s)) == expected
    assert rows_of(unswizzle_block_scales(w._nvfp4_w13_scale((w13_sf,), s))) == expected


def _small_dims() -> InklingDims:
    """Six layers (layer 5 global; NVFP4 experts in 0-3, MXFP8 in 4-5), small enough to load."""
    text = SimpleNamespace(
        num_hidden_layers=6,
        local_layer_ids=[0, 1, 2, 3, 4],
        hidden_size=256,
        vocab_size=1000,
        unpadded_vocab_size=990,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=128,
        d_rel=16,
        rel_extent=64,
        sliding_window_size=64,
        sconv_kernel_size=4,
        n_layers_per_attn_res_block=6,
        n_routed_experts=8,
        num_experts_per_tok=2,
        intermediate_size=128,
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
            "compute_dtype": fmt,
            "tensors": [f"model.llm.layers.{i}.mlp.experts.w13_weight" for i in layers],
        }
        for fmt, layers in (("nvfp4", range(4)), ("mxfp8", (4, 5)))
    ]
    return InklingDims.from_config(text, {"model": {"parameter_groups": groups}})


class _RecordingCheckpoint:
    """The engine's lazy checkpoint view (`view[name][rows]` reads only those rows), recording."""

    def __init__(self, tensors):
        self.tensors = tensors
        self.reads = []

    def __contains__(self, name):
        return name in self.tensors

    def keys(self):
        return self.tensors.keys()

    def __getitem__(self, name):
        return _RecordedTensor(self, name)


class _RecordedTensor:
    def __init__(self, checkpoint: _RecordingCheckpoint, name: str):
        self.checkpoint, self.name = checkpoint, name

    def __getitem__(self, rows):
        self.checkpoint.reads.append((self.name, rows))
        return self.checkpoint.tensors[self.name][rows]


def _random_checkpoint(layout: dict, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, (shape, dtype) in layout.items():
        if dtype.itemsize == 1:
            out[name] = torch.randint(0, 256, shape, generator=g, dtype=U8).view(dtype)
        else:
            out[name] = (torch.rand(shape, generator=g) + 0.5).to(dtype)
    return out


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(U8) if t.dtype.itemsize == 1 else t


def _model(s: "w.Shares", lm_head_rows: int):
    """The parts of the engine model `load` fills, every value poisoned first."""
    model = SimpleNamespace(
        model=SimpleNamespace(w=w.declare(s)),
        lm_head=SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.empty(lm_head_rows, s.model.hidden_size, dtype=BF16), requires_grad=False
            )
        ),
    )
    for p in [*model.model.w.values(), model.lm_head.weight]:
        if p.dtype.itemsize == 1:
            _bits(p.data).fill_(0x7F)
        else:
            p.data.fill_(float("nan"))
    return model


@pytest.mark.parametrize("rank", [1, 3])
def test_load_reads_this_ranks_rows_and_fills_every_weight(rank, monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    d = _small_dims()
    s = w.shares_of(d, _mapping(rank))
    tensors = _random_checkpoint(checkpoint_layout(d), seed=rank)
    tensors["model.mtp.fc.weight"] = torch.zeros(2, 2)  # the drafter's: not the table's to read
    ckpt = _RecordingCheckpoint(tensors)
    rows = 256  # this rank's share of the vocab-parallel lm_head (1000 rows over 4 ranks)
    model = _model(s, rows)
    w.load(model, ckpt, s)

    reads = dict(ckpt.reads)
    assert len(reads) == len(ckpt.reads), "a checkpoint tensor was read twice"
    assert set(reads) == set(checkpoint_layout(d))
    for layer, entry in w.all_entries(s):
        flags = entry.expert_rows or (False,) * len(entry.sources)
        srcs = []
        for name, by_expert in zip(entry.sources, flags):
            # Expert tensors are read at this rank's experts only.
            want = s.experts() if by_expert else slice(None)
            assert reads[name] == want, (name, reads[name], want)
            srcs.append(tensors[name][want])
        expected = entry.transform(tuple(srcs), s)
        loaded = model.model.w[w._key(layer, entry.name)].data
        assert torch.equal(_bits(loaded), _bits(expected)), (layer, entry.name)

    start, end = rank * rows, min(rank * rows + rows, d.vocab_size)
    assert reads["model.llm.unembed.weight"] == slice(start, end)
    head = model.lm_head.weight.data
    assert torch.equal(head[: end - start], tensors["model.llm.unembed.weight"][start:end])
    assert not head[end - start :].any(), "rows past the vocabulary must be zero"


def test_load_refuses_a_text_decoder_tensor_the_table_does_not_name(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    d = _small_dims()
    s = w.shares_of(d, _mapping(0))
    tensors = _random_checkpoint(checkpoint_layout(d), seed=0)
    tensors["model.llm.layers.0.attn.extra.weight"] = torch.zeros(2, 2)
    with pytest.raises(AssertionError, match="does not name"):
        w.load(_model(s, 256), _RecordingCheckpoint(tensors), s)


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
