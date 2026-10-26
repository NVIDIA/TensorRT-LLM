# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Step-5 target's weight table, checked on the CPU against the checkpoint's header.

Three claims, each a way the target could load a plausible wrong model, or not load at
all, without a kernel noticing:

* across the ranks of every member (tp4, tep4, dep4, dep2) the shares tile the model:
  every query head, KV head, gate, o_proj column, indexer head and weight, dense-MLP,
  shared-expert and expert intermediate column, and every expert is held exactly once
  per group, and the replicated weights are whole on every rank;
* every source of every parameter names a checkpoint tensor, and its cut of that
  tensor's published shape, through the source's transform, lands exactly on the
  destination region in shape and dtype;
* on every rank the table consumes every checkpoint key but the MTP layers and the
  `ssmax_s` scales, which `expected_unconsumed` names; the vision tower and its
  projector are loaded whole on every rank.

The checkpoint is read for its config and safetensors headers only (no tensor data):
``STEP5_PREVIEW_CHECKPOINT``, else ``$LLM_MODELS_ROOT/Step-5-Preview-NVFP4``; the
header-backed checks skip without one.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.models.step5.step5_preview_nvfp4__sm_103__unified import (  # noqa: E501
    weights as W,
)
from tensorrt_llm._torch.configs import Step5TextConfig, Step5VisionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

#: The members, as Mapping kwargs.
_MEMBERS = {
    "tp4": dict(world_size=4, tp_size=4, moe_tp_size=4, moe_ep_size=1),
    "tep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4),
    "dep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=True),
    "dep2": dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2, enable_attention_dp=True),
}

_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "U8": torch.uint8,
    "F8_E4M3": torch.float8_e4m3fn,
}


def _checkpoint_dir() -> Path | None:
    explicit = os.environ.get("STEP5_PREVIEW_CHECKPOINT")
    if explicit:
        return Path(explicit)
    root = llm_models_root()
    return None if root is None else root / "Step-5-Preview-NVFP4"


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    path = _checkpoint_dir()
    if path is None or not (path / "model.safetensors.index.json").is_file():
        pytest.skip(
            "no Step-5-Preview-NVFP4 checkpoint: set STEP5_PREVIEW_CHECKPOINT or stage "
            "Step-5-Preview-NVFP4 under LLM_MODELS_ROOT"
        )
    return path


@pytest.fixture(scope="module")
def header(checkpoint: Path) -> dict:
    """checkpoint key -> (shape, dtype), from the safetensors headers alone."""
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for name in sorted(set(index.values())):
        with open(checkpoint / name, "rb") as fh:
            (length,) = struct.unpack("<Q", fh.read(8))
            entries = json.loads(fh.read(length))
        for key, entry in entries.items():
            if key != "__metadata__":
                out[key] = (tuple(entry["shape"]), _DTYPES[entry["dtype"]])
    assert set(out) == set(index), "the index and the shard headers disagree"
    return out


@pytest.fixture(scope="module")
def text(checkpoint: Path) -> Step5TextConfig:
    config = json.loads((checkpoint / "config.json").read_text())
    return Step5TextConfig(**config["text_config"])


@pytest.fixture(scope="module")
def vision(checkpoint: Path) -> Step5VisionConfig:
    config = json.loads((checkpoint / "config.json").read_text())
    return Step5VisionConfig(**config["vision_config"])


@pytest.fixture(scope="module")
def quant(checkpoint: Path) -> QuantConfig:
    q = json.loads((checkpoint / "hf_quant_config.json").read_text())["quantization"]
    return QuantConfig(
        quant_algo=QuantAlgo.NVFP4,
        group_size=q["group_size"],
        exclude_modules=q["exclude_modules"],
    )


def _core(text, quant, member: str, rank: int, vision=None) -> SimpleNamespace:
    core = SimpleNamespace(
        text_config=text,
        composite_config=SimpleNamespace(vision_config=vision or Step5VisionConfig()),
        model_config=SimpleNamespace(
            mapping=Mapping(rank=rank, **_MEMBERS[member]),
            quant_config=quant,
            torch_dtype=torch.bfloat16,
        ),
    )
    core.dims = W.MODEL_WEIGHTS.dims(core)
    return core


def _ranks(member: str) -> range:
    return range(_MEMBERS[member]["world_size"])


# --- the layer families and the per-rank geometry ---------------------------------------


def test_layer_families(text, quant):
    d = _core(text, quant, "tep4", 0).dims
    assert d.sparse_layers == tuple(range(3, 92, 4))
    assert d.moe_layers == tuple(range(3, 91))
    assert d.dense_mlp_layers == (0, 1, 2, 91)
    assert d.nvfp4_layers == tuple(range(3, 88))
    assert d.bf16_expert_layers == (88, 89, 90)
    assert sum(not s for s in d.sliding) == 23


@pytest.mark.parametrize(
    "member,expected",
    [
        # heads, kv heads, indexer heads, idx rows, dense I, shared I, experts, expert I
        ("tp4", (16, 1, 4, 1544, 3456, 384, 352, 384)),
        ("tep4", (16, 1, 4, 1544, 3456, 384, 88, 1536)),
        ("dep4", (64, 4, 16, 4624, 13824, 1536, 88, 1536)),
        ("dep2", (64, 4, 16, 4624, 13824, 1536, 176, 1536)),
    ],
)
def test_per_rank_geometry(text, quant, member, expected):
    d = _core(text, quant, member, 0).dims
    got = (
        d.heads,
        d.kv_heads,
        d.idx_heads,
        d.idx_rows,
        d.dense_inter,
        d.shared_inter,
        d.local_experts,
        d.moe_inter,
    )
    assert got == expected


# --- the shares tile the model ----------------------------------------------------------


def _covered(spans: list[tuple[int, int]], total: int) -> bool:
    """Whether the [start, stop) spans cover [0, total) exactly once each."""
    hits = [0] * total
    for start, stop in spans:
        for k in range(start, stop):
            hits[k] += 1
    return all(h == 1 for h in hits)


def _span(cut, axis: int, full: int) -> tuple[int, int]:
    return cut[axis].indices(full)[:2]


@pytest.mark.parametrize("member", sorted(_MEMBERS))
def test_shares_tile_the_model(text, quant, member):
    """Each split tensor's cuts over one group's ranks hold every row/column once."""
    layer = 3  # a sparse MoE layer with NVFP4 experts
    srcs = {r: W.MODEL_WEIGHTS.manifest(_core(text, quant, member, r)) for r in _ranks(member)}
    dp = _MEMBERS[member].get("enable_attention_dp", False)
    tp_group = [0] if dp else list(_ranks(member))

    def cuts(param: str, key_suffix: str, ranks) -> list:
        out = []
        for r in ranks:
            for src in srcs[r][f"l{layer}_{param}"][1]:
                keys = src.key if isinstance(src.key, tuple) else (src.key,)
                if any(k.endswith(key_suffix) for k in keys):
                    out.append(src.cut)
        return out

    full = {
        "self_attn.q_proj.weight": 12288,
        "self_attn.k_proj.weight": 768,
        "self_attn.v_proj.weight": 768,
        "self_attn.g_proj.weight": 64,
        "self_attn.sparse_indexer_q.weight": 4096,
        "self_attn.sparse_indexer_w.weight": 16,
        "share_expert.gate_proj.weight": 1536,
        "share_expert.up_proj.weight": 1536,
    }
    params = {
        "self_attn.q_proj.weight": "qkvg",
        "self_attn.k_proj.weight": "qkvg",
        "self_attn.v_proj.weight": "qkvg",
        "self_attn.g_proj.weight": "qkvg",
        "self_attn.sparse_indexer_q.weight": "idx",
        "self_attn.sparse_indexer_w.weight": "idx",
        "share_expert.gate_proj.weight": "sgu",
        "share_expert.up_proj.weight": "sgu",
    }
    for key, total in full.items():
        spans = [_span(c, 0, total) for c in cuts(params[key], key, tp_group)]
        assert _covered(spans, total), (member, key, spans)
    o_spans = [_span(c, 1, 12288) for c in cuts("o", "o_proj.weight", tp_group)]
    assert _covered(o_spans, 12288), (member, o_spans)
    sdn_spans = [_span(c, 1, 1536) for c in cuts("sdn", "down_proj.weight", tp_group)]
    assert _covered(sdn_spans, 1536), (member, sdn_spans)
    # The replicated indexer key and logits are whole on every attention rank.
    for r in tp_group:
        for src in srcs[r][f"l{layer}_idx"][1]:
            if src.key.endswith(("sparse_indexer_k.weight", "sparse_indexer_z.weight")):
                assert src.cut is None
    # Experts: every (expert, intermediate column) of FC1 and FC2 held once over all ranks.
    fc1_cells, fc2_cells = set(), set()
    for r in _ranks(member):
        (src,) = srcs[r][f"l{layer}_fc1_w"][1]
        e0, e1 = _span(src.cut, 0, 352)
        i0, i1 = _span(src.cut, 1, 1536)
        for e in range(e0, e1):
            for c in range(i0, i1, 128):
                assert (e, c) not in fc1_cells
                fc1_cells.add((e, c))
        (src,) = srcs[r][f"l{layer}_fc2_w"][1]
        e0, e1 = _span(src.cut, 0, 352)
        k0, k1 = (s * 2 for s in _span(src.cut, 2, 768))
        for e in range(e0, e1):
            for c in range(k0, k1, 128):
                assert (e, c) not in fc2_cells
                fc2_cells.add((e, c))
    expected = {(e, c) for e in range(352) for c in range(0, 1536, 128)}
    assert fc1_cells == expected and fc2_cells == expected


# --- every source lands on its destination, and the checkpoint is covered ----------------


def _cut_shape(shape: tuple, cut) -> tuple:
    if cut is None:
        return shape
    return tuple(torch.empty(shape, device="meta")[cut].shape)


def _transformed(src, shapes: list[tuple], dtypes: list[torch.dtype]):
    """(shape, dtype) the source's transform produces, from its inputs' (shape, dtype)."""
    t = src.transform
    if t is None or t is W._contig:
        return shapes[0], dtypes[0]
    if t is W._to_bf16:
        return shapes[0], torch.bfloat16
    if t is W._swizzle:
        return shapes[0], torch.uint8
    if getattr(t, "func", None) is W._resample_grid:
        assert shapes[0] == (t.keywords["grid"] ** 2, shapes[0][1]), shapes
        return (t.keywords["side"] ** 2, shapes[0][1]), dtypes[0]
    if t in (W._cat_experts, W._cat_swizzle):
        assert len({s[:1] + s[2:] for s in shapes}) == 1, shapes
        shape = (shapes[0][0], sum(s[1] for s in shapes)) + shapes[0][2:]
        return shape, torch.uint8 if t is W._cat_swizzle else dtypes[0]
    raise AssertionError(f"unmodelled transform {t}")


@pytest.mark.parametrize("member", sorted(_MEMBERS))
def test_sources_land_and_cover_the_checkpoint(text, vision, quant, header, member):
    for rank in sorted({0, _MEMBERS[member]["world_size"] - 1}):
        core = _core(text, quant, member, rank, vision)
        d = core.dims
        params = {}
        for entry in W.MODEL_WEIGHTS.WEIGHTS:
            for key, _ in W.MODEL_WEIGHTS._targets(entry, d):
                params[key] = torch.empty(
                    entry.shape(d), dtype=entry.dtype or d.dtype, device="meta"
                )
        consumed = set()
        for key, (entry, sources) in W.MODEL_WEIGHTS.manifest(core).items():
            dst_full = params[key]
            for src in sources:
                keys = src.key if isinstance(src.key, tuple) else (src.key,)
                missing = [k for k in keys if k not in header]
                assert not missing, (member, rank, key, missing)
                shapes = [_cut_shape(header[k][0], src.cut) for k in keys]
                dtypes = [header[k][1] for k in keys]
                shape, dtype = _transformed(src, shapes, dtypes)
                dst = dst_full if src.index is None else dst_full[src.index]
                assert tuple(dst.shape) == shape, (member, rank, key, keys, tuple(dst.shape), shape)
                assert dst.dtype == dtype, (member, rank, key, keys, dst.dtype, dtype)
                consumed.update(keys)
        consumed.add("lm_head.weight")
        names = SimpleNamespace(keys=lambda: header.keys())
        expected = W.MODEL_WEIGHTS.expected_unconsumed(core, names)
        assert set(header) - consumed == expected, (
            member,
            rank,
            sorted(set(header) - consumed - expected)[:8],
            sorted(expected - (set(header) - consumed))[:8],
        )
        # What is left is exactly the MTP layers and the ssmax_s scales.
        assert all(
            k.startswith(("model.layers.92.", "model.layers.93.", "model.layers.94."))
            or k.endswith("self_attn.ssmax_s")
            for k in expected
        )
        # The vision tower and its projector are all consumed, whole on every rank.
        vision_keys = {k for k in header if k.startswith(("vision_model.", "vit_large_projector."))}
        assert len(vision_keys) == 667 and vision_keys <= consumed


def test_vision_geometry(text, vision, quant):
    d = _core(text, quant, "dep4", 0, vision).dims
    got = (d.v_layers, d.v_width, d.v_heads, d.v_head_dim, d.v_inter, d.v_grid, d.v_crop_grid)
    assert got == (47, 1536, 16, 96, 8960, 52, 36)
    assert d.v_out == 6144


def test_lm_head_share():
    """Vocab-parallel under TP (128896 splits 4 ways evenly), whole under attention DP."""
    cut, pad = W.lm_head_share(32224, 128896, 3)
    assert cut == (slice(96672, 128896),)
    assert pad(torch.ones(32224, 2)).shape == (32224, 2)
    cut, _ = W.lm_head_share(128896, 128896, 2)
    assert cut == (slice(0, 128896),)
    # A vocab that does not split evenly: the last shard's missing rows are zero.
    cut, pad = W.lm_head_share(4, 10, 2)
    assert cut == (slice(8, 10),)
    padded = pad(torch.ones(2, 3))
    assert padded.shape == (4, 3) and padded[2:].abs().sum() == 0
