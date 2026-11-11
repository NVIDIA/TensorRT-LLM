# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The unified target's per-rank weight table against the FP8 checkpoint, on the CPU from its
index and safetensors headers: each rank's shard, each member's coverage and `load`'s operands."""

from __future__ import annotations

import collections
import json
import re
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified import (  # noqa: E501
    weights as WT,
)
from tensorrt_llm._torch.configs import Qwen4ExpTextConfig, Qwen4ExpVisionConfig
from tensorrt_llm._torch.models.modeling_utils import duplicate_kv_weight
from tensorrt_llm.mapping import Mapping

CHECKPOINT = "Qwen3.8-Flash-Next-FP8"

#: Each member's mapping as the LLM API's parallel arguments build it.
MEMBERS = {
    "tp1": dict(world_size=1, tp_size=1),
    "tp2": dict(world_size=2, tp_size=2),
    "tep2": dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2),
    "dep2": dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2, enable_attention_dp=True),
    "tp4": dict(world_size=4, tp_size=4),
    "tep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4),
    "dep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=True),
}
NUM_EXPERTS = 512
MTP_LAYER = 48

_STORED_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn,
                  "I64": torch.int64}  # fmt: skip

#: A routed (trunk) or draft (MTP) expert's tensors.
_EXPERT_TENSOR = re.compile(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.(weight|weight_scale_inv)$")


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    root = llm_models_root()
    path = None if root is None else root / CHECKPOINT
    if path is None or not (path / "model.safetensors.index.json").is_file():
        pytest.skip(f"{CHECKPOINT} is not under LLM_MODELS_ROOT")
    return path


@pytest.fixture(scope="module")
def index(checkpoint) -> dict[str, str]:
    return json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]


@pytest.fixture(scope="module")
def stored(checkpoint, index) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Every checkpoint tensor's stored shape and dtype, from the index and the headers."""
    specs = {}
    for file in sorted(set(index.values())):
        with open(checkpoint / file, "rb") as fh:
            (length,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(length))
        header.pop("__metadata__", None)
        for key, entry in header.items():
            assert index.get(key) == file, f"{key} is in {file}, the index says {index.get(key)}"
            specs[key] = (tuple(entry["shape"]), _STORED_DTYPES[entry["dtype"]])
    assert specs.keys() == index.keys(), "the index and the headers name different tensors"
    return specs


@pytest.fixture(scope="module")
def configs(checkpoint):
    raw = json.loads((checkpoint / "config.json").read_text())
    return Qwen4ExpTextConfig(**raw["text_config"]), Qwen4ExpVisionConfig(**raw["vision_config"])


def _mapping(member: str, rank: int) -> Mapping:
    return Mapping(rank=rank, gpus_per_node=4, pp_size=1, **MEMBERS[member])


def _dims(configs, member: str, rank: int, mtp: bool = True):
    text, vision = configs
    return WT.rank_dims(text, vision, _mapping(member, rank), mtp_enabled=mtp, dtype=torch.bfloat16)


def _world(member: str) -> int:
    return MEMBERS[member]["world_size"]


def _attention_tp(member: str) -> int:
    layout = MEMBERS[member]
    return 1 if layout.get("enable_attention_dp", False) else layout["tp_size"]


def _lm_head_rows(member: str, vocab: int) -> int:
    """The shell's lm_head rows: vocab-parallel under attention TP, whole otherwise."""
    return vocab // _attention_tp(member)


def _rows_per_shard(stored, d) -> int:
    shapes = {stored[key][0] for key in WT.ngram_shard_keys(d)}
    assert len(shapes) == 1, shapes
    return next(iter(shapes))[0]


def _rank_manifest(stored, d, member: str) -> dict[str, list[tuple]]:
    """The rank's manifest as `load` builds it, with the n-gram table and the shell's lm_head."""
    manifest = WT.MODEL_WEIGHTS.manifest(d)
    manifest["ple_table"] = WT.ngram_table_sources(d, _rows_per_shard(stored, d))
    cut, pad = WT.lm_head_share(_lm_head_rows(member, d.vocab), d.vocab, d.tp_rank)
    manifest["lm_head.weight"] = [("lm_head.weight", None, cut, pad)]
    return manifest


def _keys_and_cuts(ckpt_key, cut):
    if isinstance(ckpt_key, tuple):
        return list(zip(ckpt_key, cut if cut is not None else (None,) * len(ckpt_key)))
    return [(ckpt_key, cut)]


def _cut_numel(shape: tuple[int, ...], cut) -> int:
    """Elements of a `shape` tensor `cut` reads, after checking each slice lies inside it."""
    if cut is None:
        return int(torch.Size(shape).numel())
    extents = list(shape)
    for axis, s in enumerate(cut):
        assert isinstance(s, slice) and s.step in (None, 1), (shape, cut)
        start = 0 if s.start is None else s.start
        stop = shape[axis] if s.stop is None else s.stop
        assert 0 <= start < stop <= shape[axis], f"cut {cut} leaves the stored {shape}"
        extents[axis] = stop - start
    return int(torch.Size(extents).numel())


class _HeaderEntry:
    """A checkpoint tensor known by its header: indexing returns a meta tensor, nothing is read."""

    def __init__(self, shape, dtype):
        self.shape, self.dtype = torch.Size(shape), dtype
        self.ndim = len(shape)

    def __getitem__(self, index):
        return torch.empty(self.shape, dtype=self.dtype, device="meta")[index]


def _share(member: str, key: str) -> float:
    """The fraction of `key` a rank of `member` reads: attention and Gated-DeltaNet tensors by
    head over the attention TP ranks (two KV heads), the lm_head by vocabulary, the rest whole."""
    tp = _attention_tp(member)
    if re.search(r"\.self_attn\.(q_proj|o_proj)\.weight$", key) or key == "lm_head.weight":
        return 1 / tp
    if re.search(r"\.self_attn\.(k_proj|v_proj)\.weight$", key):
        return 1 / min(tp, 2)
    if re.search(
        r"\.linear_attn\.((in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|conv1d|out_proj)\.weight"
        r"|A_log|dt_bias)$",
        key,
    ):
        return 1 / tp
    return 1.0


def _window(member: str, rank: int) -> range:
    """The `512 / world` consecutive whole experts a rank holds, whatever the MoE split."""
    count = NUM_EXPERTS // _world(member)
    return range(count * rank, count * (rank + 1))


def _check_operand_shapes(stored, d, member: str) -> dict:
    """Each source fills its parameter exactly; each tensor is read at the rank's share, once."""
    with torch.device("meta"):
        params = dict(WT.MODEL_WEIGHTS.declare(d))
        # What `load` adds beside the table: the n-gram table and the shell's lm_head.
        params["ple_table"] = torch.empty(
            d.ple_shards * _rows_per_shard(stored, d), d.ple_head_dim, dtype=d.ple_table_dtype
        )
        params["lm_head.weight"] = torch.empty(
            _lm_head_rows(member, d.vocab), d.hidden, dtype=torch.bfloat16
        )
    entries = {key: _HeaderEntry(*spec) for key, spec in stored.items()}
    manifest = _rank_manifest(stored, d, member)
    assert manifest.keys() == params.keys()
    read = collections.Counter()
    results: dict = {}
    for param_key, sources in manifest.items():
        param = params[param_key]
        for ckpt_key, index, cut, transform in sources:
            key_cuts = _keys_and_cuts(ckpt_key, cut)
            for key, key_cut in key_cuts:
                read[key] += _cut_numel(stored[key][0], key_cut)
            dst = param if index is None else param[index]
            # Equal stored tensors, cuts and transform give equal results: read each once.
            signature = (transform, tuple((stored[k], c) for k, c in key_cuts))
            if signature not in results:
                src = WT.MODEL_WEIGHTS.read(entries, ckpt_key, cut, transform, d)
                results[signature] = (tuple(src.shape), src.dtype)
            got = (tuple(dst.shape), dst.dtype)
            assert results[signature] == got, (param_key, ckpt_key, results[signature], got)
    wrong = {
        key: (n, _share(member, key))
        for key, n in read.items()
        if n != round(torch.Size(stored[key][0]).numel() * _share(member, key))
    }
    assert not wrong, f"tensors not read at the rank's share, once: {sorted(wrong.items())[:8]}"
    return params


_HIDDEN = 2  # the projections' input axis, shrunk: no cut touches it
_FAKE_OFFSETS = dict(q=1, k=2, v=3, idx=4, o=5, qkv=10, z=11, a=12, b=13, conv=14, a_log=15,
                     dtb=16, out=17)  # fmt: skip


def _ids(tag: str, rows: int, *trailing: int) -> torch.Tensor:
    """`rows` rows whose every value is `tag`'s offset plus the row's index."""
    ids = torch.arange(rows, dtype=torch.int64) + _FAKE_OFFSETS[tag] * 10**7
    return ids.view(rows, *([1] * len(trailing))).expand(rows, *trailing).contiguous()


def _col_ids(tag: str, cols: int) -> torch.Tensor:
    return _ids(tag, cols, _HIDDEN).t().contiguous()


def _fake_checkpoint(stored) -> dict[str, torch.Tensor]:
    """Attention layer 3, Gated-DeltaNet layer 0 and MTP attention at stored rows, 2-wide input."""
    fake = {}

    def rows_of(key):
        return stored[key][0][0]

    for p in ("model.language_model.layers.3.self_attn", "mtp.layers.0.self_attn"):
        for tag, name in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")):
            fake[f"{p}.{name}.weight"] = _ids(tag, rows_of(f"{p}.{name}.weight"), _HIDDEN)
        key = f"{p}.indexer.index_qk_proj.weight"
        fake[key] = _ids("idx", rows_of(key), _HIDDEN)
        fake[f"{p}.o_proj.weight"] = _col_ids("o", stored[f"{p}.o_proj.weight"][0][1])
    p = "model.language_model.layers.0.linear_attn"
    for tag, name in (("qkv", "in_proj_qkv"), ("z", "in_proj_z"), ("a", "in_proj_a"),
                      ("b", "in_proj_b")):  # fmt: skip
        fake[f"{p}.{name}.weight"] = _ids(tag, rows_of(f"{p}.{name}.weight"), _HIDDEN)
    fake[f"{p}.conv1d.weight"] = _ids("conv", rows_of(f"{p}.conv1d.weight"), 1, 4)
    fake[f"{p}.A_log"] = _ids("a_log", rows_of(f"{p}.A_log"))
    fake[f"{p}.dt_bias"] = _ids("dtb", rows_of(f"{p}.dt_bias"))
    fake[f"{p}.out_proj.weight"] = _col_ids("out", stored[f"{p}.out_proj.weight"][0][1])
    return fake


def _assemble(d, fake, sources) -> torch.Tensor:
    """A parameter filled from its sources the way `load` fills it."""
    if len(sources) == 1 and sources[0][1] is None:
        ckpt_key, _, cut, transform = sources[0]
        return WT.MODEL_WEIGHTS.read(fake, ckpt_key, cut, transform, d)
    parts = [
        (index, WT.MODEL_WEIGHTS.read(fake, ckpt_key, cut, transform, d))
        for ckpt_key, index, cut, transform in sources
    ]
    rows = max(index[0].stop for index, _ in parts)
    out = torch.full((rows, *parts[0][1].shape[1:]), -1, dtype=parts[0][1].dtype)
    for index, part in parts:
        out[index] = part
    return out


def _expected_rows(tag: str, starts) -> torch.Tensor:
    return torch.tensor([_FAKE_OFFSETS[tag] * 10**7 + r for r in starts], dtype=torch.int64)


def _check_heads(stored, d, member: str, rank: int, params: dict) -> None:
    """The cuts select the rank's heads in fused-operand order, the heads the stock pools hold."""
    fake = _fake_checkpoint(stored)
    manifest = WT.MODEL_WEIGHTS.manifest(d)
    tp = _attention_tp(member)
    # Under attention DP and at tp1 each rank holds every head: the split is over one rank.
    r = rank if tp > 1 else 0
    assert (d.heads_q, d.q_head_start) == (24 // tp, r * 24 // tp)
    assert (d.heads_kv, d.kv_head_start) == (max(2 // tp, 1), r * 2 // tp)
    assert (d.num_k_heads, d.k_head_start) == (16 // tp, r * 16 // tp)
    assert (d.num_v_heads, d.v_head_start) == (48 // tp, r * 48 // tp)
    assert params["l0_conv_w"].shape[0] == 2 * d.key_dim + d.value_dim
    q_heads, k_heads, v_heads = (range(r * n // tp, (r + 1) * n // tp) for n in (24, 16, 48))
    kv_heads = [r * 2 // tp] if tp > 1 else [0, 1]

    def rows(heads, width, base=0):
        return [base + h * width + j for h in heads for j in range(width)]

    q_rows = [h * 512 + j for h in q_heads for j in range(256)]
    gate_rows = [h * 512 + 256 + j for h in q_heads for j in range(256)]
    # fmt: off
    expected = torch.cat((
        _expected_rows("q", q_rows), _expected_rows("k", rows(kv_heads, 256)),
        _expected_rows("v", rows(kv_heads, 256)), _expected_rows("q", gate_rows),
        _expected_rows("idx", range(640)), torch.zeros(128, dtype=torch.int64),
    ))
    # fmt: on
    attn_in = {
        key: _assemble(d, fake, manifest[key]) for key in ("l3_attn_in_w", "l48_mtp_attn_in_w")
    }
    for key, got in attn_in.items():
        assert torch.equal(got[:, 0], expected) and torch.equal(got[:, 1], expected), key
    # The KV head a rank holds is the one the built-in path's `duplicate_kv_weight` gives it.
    k_key = "mtp.layers.0.self_attn.k_proj.weight"
    builtin_k = duplicate_kv_weight(fake[k_key], 2, tp).chunk(tp)[r] if tp > 1 else fake[k_key]
    q_width = len(q_rows)
    mtp_k = attn_in["l48_mtp_attn_in_w"][q_width : q_width + builtin_k.shape[0]]
    assert torch.equal(mtp_k, builtin_k)
    o_cols = _expected_rows("o", rows(q_heads, 256))
    for key in ("l3_o_w", "l48_mtp_o_w"):
        assert torch.equal(_assemble(d, fake, manifest[key])[0], o_cols), key

    def qkv(tag):
        k, v = rows(k_heads, 128), rows(v_heads, 128, 4096)
        return torch.cat([_expected_rows(tag, r) for r in (k, rows(k_heads, 128, 2048), v)])

    z, a, b = rows(v_heads, 128), v_heads, v_heads
    in_proj = torch.cat([qkv("qkv")] + [_expected_rows(t, r) for t, r in zip("zab", (z, a, b))])
    assert torch.equal(_assemble(d, fake, manifest["l0_in_proj_w"])[:, 0], in_proj)
    assert torch.equal(_assemble(d, fake, manifest["l0_conv_w"])[:, 0, 0], qkv("conv"))
    assert torch.equal(_assemble(d, fake, manifest["l0_A_log"]), _expected_rows("a_log", v_heads))
    assert torch.equal(_assemble(d, fake, manifest["l0_dt_bias"]), _expected_rows("dtb", v_heads))
    assert torch.equal(_assemble(d, fake, manifest["l0_out_w"])[0],
                       _expected_rows("out", rows(v_heads, 128)))  # fmt: skip


#: Per-expert operand (shape, dtype); block scales hold one fp32 value per 128x128 block.
_EXPERT_SHAPES = {
    "fc1_w": ((1280, 2560), torch.float8_e4m3fn),
    "fc1_s": ((10, 20), torch.float32),
    "fc2_w": ((2560, 640), torch.float8_e4m3fn),
    "fc2_s": ((20, 5), torch.float32),
}


def _check_expert_window(d, member: str, rank: int, params: dict) -> None:
    """The rank holds `512 / world` uncut experts, trunk and draft layer alike (tp2 and tp4 too),
    and `expert_window` hands the runner exactly those parameters."""
    window = _window(member, rank)
    assert (d.expert_offset, d.local_experts, d.moe_inter) == (window.start, len(window), 640)
    manifest = WT.MODEL_WEIGHTS.manifest(d)
    for prefix, module in (("l0_", "model.language_model.layers.0.mlp"),
                           ("l48_mtp_", "mtp.layers.0.mlp")):  # fmt: skip
        for name, (shape, dtype) in _EXPERT_SHAPES.items():
            param, rows = params[f"{prefix}{name}"], manifest[f"{prefix}{name}"]
            assert (tuple(param.shape), param.dtype) == ((len(window), *shape), dtype), name
            assert [index for _, index, _, _ in rows] == [(i,) for i in range(len(rows))]
            assert all(cut is None for _, _, cut, _ in rows), (prefix, name)
        experts = [s[0] for s in manifest[f"{prefix}fc2_w"]]
        assert experts == [f"{module}.experts.{e}.down_proj.weight" for e in window]
        win = WT.expert_window(params, d, MTP_LAYER if prefix == "l48_mtp_" else 0)
        assert (win.offset, win.count) == (window.start, len(window))
        assert win.fc1_w is params[f"{prefix}fc1_w"] and win.fc2_s is params[f"{prefix}fc2_s"]


@pytest.mark.cpu_only
@pytest.mark.parametrize("member, rank", [("tp1", 0), ("tp2", 1), ("tp4", 2), ("dep4", 3)])
def test_rank_shard(stored, configs, member, rank):
    """The heads each cut selects, every operand's shape and the expert window. tep is left out:
    it differs from tp only in the `Mapping`'s MoE split, which no cut reads."""
    d = _dims(configs, member, rank)
    params = _check_operand_shapes(stored, d, member)
    _check_heads(stored, d, member, rank, params)
    _check_expert_window(d, member, rank, params)


def _stub_model(configs, member: str, rank: int, mtp: bool):
    """The shell and core that `load` reads, on meta: configs, mapping, parameters, lm_head."""
    text, vision = configs
    core = SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=text,
            mapping=_mapping(member, rank),
            spec_config=object() if mtp else None,
            torch_dtype=torch.bfloat16,
        ),
        vision_config=vision,
    )
    d = WT.MODEL_WEIGHTS.dims(core)
    with torch.device("meta"):
        core.w = WT.MODEL_WEIGHTS.declare(d)
        lm_head = torch.nn.Parameter(
            torch.empty(_lm_head_rows(member, d.vocab), d.hidden, dtype=torch.bfloat16),
            requires_grad=False,
        )

    def named_parameters():
        yield from ((f"model.w.{k}", v) for k, v in core.w.items())
        yield "lm_head.weight", lm_head

    return SimpleNamespace(
        model=core, lm_head=SimpleNamespace(weight=lm_head), named_parameters=named_parameters
    )


#: The rank `load` runs at, end to end on meta tensors, for these members.
_LOAD_RANK = {("tp1", True): 0, ("tp2", True): 1, ("tep4", False): 3, ("dep4", True): 0}


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "member, mtp",
    [(member, True) for member in MEMBERS] + [("tep4", False)],
    ids=[f"{member}-mtp" for member in MEMBERS] + ["tep4-no-mtp"],
)
def test_member_covers_the_checkpoint(stored, configs, member, mtp, monkeypatch):
    """Each tensor is read or predicted unread, never both; each expert's tensors by one rank and
    the rest by every rank; the windows tile the 512 experts; and `load` fills every parameter."""
    everything = set(stored)
    read_by = collections.Counter()
    seen = []
    for rank in range(_world(member)):
        d = _dims(configs, member, rank, mtp)
        manifest = _rank_manifest(stored, d, member)
        claimed = {
            key
            for sources in manifest.values()
            for ckpt_key, _, cut, _ in sources
            for key, _ in _keys_and_cuts(ckpt_key, cut)
        }
        unread = WT.MODEL_WEIGHTS.expected_unconsumed(d, stored.keys())
        assert claimed <= everything, (
            f"rank {rank} reads unknown {sorted(claimed - everything)[:8]}"
        )
        assert not claimed & unread, f"rank {rank}: read and unread {sorted(claimed & unread)[:8]}"
        assert claimed | unread == everything, (
            f"rank {rank}: neither read nor predicted {sorted(everything - claimed - unread)[:8]}"
        )
        read_by.update(claimed)
        seen.extend(range(d.expert_offset, d.expert_offset + d.local_experts))
    mtp_keys = set() if mtp else {k for k in everything if k.startswith("mtp.")}
    assert set(read_by) == everything - mtp_keys, "tensors no rank reads: " + str(
        sorted(everything - mtp_keys - set(read_by))[:8]
    )
    wrong = sorted(
        k for k, n in read_by.items() if n != (1 if _EXPERT_TENSOR.search(k) else _world(member))
    )
    assert not wrong, f"tensors not read by their expected rank count: {wrong[:8]}"
    assert seen == list(range(NUM_EXPERTS))
    if member in ("tp2", "tp4"):
        mapping = _mapping(member, 0)
        assert (mapping.moe_tp_size, mapping.moe_ep_size) == (mapping.tp_size, 1)

    if (member, mtp) not in _LOAD_RANK:
        return
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    model = _stub_model(configs, member, _LOAD_RANK[member, mtp], mtp)
    WT.MODEL_WEIGHTS.load(model, {key: _HeaderEntry(*spec) for key, spec in stored.items()})
    d = WT.MODEL_WEIGHTS.dims(model.model)
    assert tuple(model.model.w["ple_table"].shape) == (128 * _rows_per_shard(stored, d), 160)
    assert model.lm_head.weight.shape[0] == _lm_head_rows(member, 248320)


def _random_bytes(gen, *shape) -> torch.Tensor:
    return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)


def _random_scales(gen, *shape) -> torch.Tensor:
    return (0.001 + torch.rand(shape, generator=gen)).to(torch.bfloat16)


def _expert_operands(configs, which: str) -> None:
    """An expert's operands are the checkpoint's bytes in the runner's layout: FC1 `[up ; gate]`
    with its block scales in that order widened exactly to fp32, FC2 as stored, trunk and MTP."""
    d = _dims(configs, "tp4", 1)
    window = _window("tp4", 1)
    expert = window[0] if which == "first" else window[-1]
    local = expert - window.start
    gen = torch.Generator().manual_seed(expert * 7 + 1)
    manifest = WT.MODEL_WEIGHTS.manifest(d)
    for prefix, module in (("l5_", "model.language_model.layers.5.mlp"),
                           ("l48_mtp_", "mtp.layers.0.mlp")):  # fmt: skip
        p = f"{module}.experts.{expert}"
        fake = {
            f"{p}.up_proj.weight": _random_bytes(gen, 640, 2560).view(torch.float8_e4m3fn),
            f"{p}.gate_proj.weight": _random_bytes(gen, 640, 2560).view(torch.float8_e4m3fn),
            f"{p}.down_proj.weight": _random_bytes(gen, 2560, 640).view(torch.float8_e4m3fn),
            f"{p}.up_proj.weight_scale_inv": _random_scales(gen, 5, 20),
            f"{p}.gate_proj.weight_scale_inv": _random_scales(gen, 5, 20),
            f"{p}.down_proj.weight_scale_inv": _random_scales(gen, 20, 5),
        }

        def operand(name):
            (row,) = [s for s in manifest[f"{prefix}{name}"] if s[1] == (local,)]
            ckpt_key, _, cut, transform = row
            return WT.MODEL_WEIGHTS.read(fake, ckpt_key, cut, transform, d)

        def u8(name):
            return fake[f"{p}.{name}"].view(torch.uint8)

        fc1_w, fc1_s, fc2_w, fc2_s = (operand(n) for n in ("fc1_w", "fc1_s", "fc2_w", "fc2_s"))
        assert fc1_w.dtype == fc2_w.dtype == torch.float8_e4m3fn
        assert fc1_s.dtype == fc2_s.dtype == torch.float32
        assert torch.equal(fc1_w.view(torch.uint8),
                           torch.cat((u8("up_proj.weight"), u8("gate_proj.weight"))))  # fmt: skip
        assert torch.equal(fc2_w.view(torch.uint8), u8("down_proj.weight"))
        up_s, gate_s = (fake[f"{p}.{n}_proj.weight_scale_inv"] for n in ("up", "gate"))
        assert torch.equal(fc1_s, torch.cat((up_s, gate_s)).float())
        assert torch.equal(fc2_s, fake[f"{p}.down_proj.weight_scale_inv"].float())
        assert torch.equal(fc1_s.to(torch.bfloat16), torch.cat((up_s, gate_s)))


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("expert-first", marks=pytest.mark.cpu_only),
        pytest.param("expert-last", marks=pytest.mark.cpu_only),
    ],
)
def test_load_operands(case, request):
    """What `load` fills the routed-expert operands from: codes and scales."""
    _expert_operands(request.getfixturevalue("configs"), case.split("-")[1])
