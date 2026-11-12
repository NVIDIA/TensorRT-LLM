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
def id_space(checkpoint, index) -> dict[str, torch.Tensor]:
    """The n-gram id space's tensors (per-head offsets and vocabulary sizes), read whole."""
    from safetensors import safe_open

    tensors = {}
    for key, file in index.items():
        if key.endswith((".ngram_heads_offsets", ".ngram_heads_vocab_sizes")):
            with safe_open(str(checkpoint / file), framework="pt") as fh:
                tensors[key] = fh.get_tensor(key)
    return tensors


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
def test_member_covers_the_checkpoint(stored, configs, id_space, member, mtp, monkeypatch):
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
    WT.MODEL_WEIGHTS.load(model, _header_entries(stored, id_space))
    d = WT.MODEL_WEIGHTS.dims(model.model)
    assert tuple(model.model.w["ple_table"].shape) == (128 * _rows_per_shard(stored, d), 160)
    assert model.lm_head.weight.shape[0] == _lm_head_rows(member, 248320)


def _header_entries(stored, id_space) -> dict:
    """Every checkpoint tensor by its header, but the n-gram id space's, which `load` reads."""
    return {
        key: id_space[key] if key in id_space else _HeaderEntry(*spec)
        for key, spec in stored.items()
    }


@pytest.mark.cpu_only
def test_load_rejects_a_per_shard_padded_ngram_table(stored, configs, id_space, monkeypatch):
    """Shards padded one by one hold more padding than one divisor past the id space the
    per-head offsets and sizes address; `load` raises before it declares the table."""
    d = _dims(configs, "tp1", 0)
    rows_per_shard = _rows_per_shard(stored, d)
    prefix = f"{WT._LM}.layers.{d.ple_layer}.ple.ple_embedding"
    offsets = id_space[f"{prefix}.ngram_heads_offsets"]
    vocab = id_space[f"{prefix}.ngram_heads_vocab_sizes"].clone()
    # The ids of each shard end `pad` rows before it does.
    pad = -(-d.ngram_vocab_divisor // d.ple_shards)
    vocab[-1] = d.ple_shards * (rows_per_shard - pad) - int(offsets[-1])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    model = _stub_model(configs, "tp1", 0, True)
    entries = _header_entries(stored, {**id_space, f"{prefix}.ngram_heads_vocab_sizes": vocab})
    with pytest.raises(ValueError, match="Shards padded one by one"):
        WT.MODEL_WEIGHTS.load(model, entries)
    assert "ple_table" not in model.model.w


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


# ------------------------------------------------------------ the host n-gram table --


def _host_table_module():
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp import ple_host_table

    return ple_host_table


@pytest.mark.cpu_only
@pytest.mark.parametrize("rows, cols", [(320001536, 160), (1000, 160), (37, 4)])
def test_host_table_chunks_cover_the_rows(rows, cols):
    """Chunks are whole multiples of the alignment, tile the rows in order, and every server size
    that divides `CHUNKS` splits them into disjoint contiguous shares."""
    PH = _host_table_module()
    align = PH.CHUNK_ALIGN if cols == 160 else cols
    per = PH.chunk_rows(rows, cols, align)
    assert (per * cols) % align == 0 and per * PH.CHUNKS >= rows
    ranges = [PH.chunk_range(k, rows, cols, align) for k in range(PH.CHUNKS)]
    assert ranges[0][0] == 0 and ranges[-1][1] == rows
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    for world in (1, 2, 4, 8):
        shares = [PH.owned_rows(r, world, rows, cols, align) for r in range(world)]
        assert shares[0][0] == 0 and shares[-1][1] == rows
        assert all(a[1] == b[0] for a, b in zip(shares, shares[1:])), shares


def _broker_role(path, key, queue, go, done, die_on_go):
    """A child process: the broker loaded from its file, without importing tensorrt_llm. An owner
    publishes (or exits, `die_on_go`) when the test sets `go` and stays until it sets `done`."""
    import importlib.util
    import mmap
    import os

    spec = importlib.util.spec_from_file_location("chunk_broker", path)
    broker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(broker)
    queue.put("ready")
    got = broker.acquire(key, 300)
    if got[0] == broker.OWNER:
        queue.put("owner")
        go.wait(300)
        if die_on_go:
            os._exit(0)
        fd = os.memfd_create("t")
        os.ftruncate(fd, 4096)
        mmap.mmap(fd, 4096)[:5] = b"bytes"
        got[1].publish(fd, {"n": 4096})
        done.wait(300)
        got[1].close()
    else:
        queue.put(("attached", bytes(mmap.mmap(got[1], 4096)[:5]), got[2]))


@pytest.mark.cpu_only
@pytest.mark.parametrize("owner_dies", [False, True], ids=["publish", "owner-exits"])
def test_chunk_broker_hands_over(owner_dies):
    """The first process owns a key; a waiter that is already waiting attaches to what it
    publishes (and so does a late one), or owns the key itself when the owner exits before
    publishing. No files are left."""
    import multiprocessing
    import os
    import time
    import uuid

    PH = _host_table_module()
    path = PH._broker.__file__
    key = uuid.uuid4().hex
    ctx = multiprocessing.get_context("spawn")
    go, done = ctx.Event(), ctx.Event()
    procs = []

    def start(die_on_go=False):
        queue = ctx.Queue()
        proc = ctx.Process(target=_broker_role, args=(path, key, queue, go, done, die_on_go))
        proc.start()
        procs.append(proc)
        assert queue.get(timeout=300) == "ready"
        return queue

    try:
        assert start(owner_dies).get(timeout=60) == "owner"
        waiter = start()
        time.sleep(1.0)  # the waiter is connected and blocked on the owner
        go.set()
        if owner_dies:
            assert waiter.get(timeout=60) == "owner"
        else:
            assert waiter.get(timeout=60) == ("attached", b"bytes", {"n": 4096})
            assert start().get(timeout=60) == ("attached", b"bytes", {"n": 4096})
    finally:
        go.set()
        done.set()
        for proc in procs:
            proc.join(60)
    assert all(proc.exitcode == 0 for proc in procs), [proc.exitcode for proc in procs]
    assert not [f for f in os.listdir("/dev/shm") if key in f]


@pytest.mark.cpu_only
def test_chunk_broker_refuses_a_foreign_owner(monkeypatch):
    """A waiter whose key's name is held by a socket of another uid raises `PermissionError` (an
    `OSError`, so `node` falls back) at once instead of waiting for it to publish."""
    import os
    import time
    import uuid

    broker = _host_table_module()._broker
    key = uuid.uuid4().hex
    owner = broker.try_own(key)
    assert owner is not None
    monkeypatch.setattr(broker, "_peer_uid", lambda conn: os.getuid() + 1)
    try:
        start = time.monotonic()
        with pytest.raises(PermissionError):
            broker.acquire(key, 60)
        assert time.monotonic() - start < 5
    finally:
        owner.close()


@pytest.mark.cpu_only
def test_chunk_broker_times_out_on_a_name_nobody_listens_on():
    """A key's name bound by a socket that never listens refuses every connection; the waiter
    raises `TimeoutError` at its deadline."""
    import socket
    import time
    import uuid

    broker = _host_table_module()._broker
    key = uuid.uuid4().hex
    squatter = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    squatter.bind(broker.socket_name(key))
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            broker.acquire(key, 2.0)
        assert 2.0 <= time.monotonic() - start < 10
    finally:
        squatter.close()


@pytest.mark.cpu_only
def test_chunk_broker_frees_the_name_when_accept_fails(monkeypatch):
    """An owner whose `accept` fails (EMFILE here) closes its listener: a waiter in its backlog
    sees the connection reset and owns the key itself at once, instead of waiting out its
    timeout on a name nobody accepts on."""
    import errno
    import socket
    import threading
    import time
    import uuid

    broker = _host_table_module()._broker
    key = uuid.uuid4().hex
    gate = threading.Event()

    def failing_accept(self):
        gate.wait(30)
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(socket.socket, "accept", failing_accept)
    owner = broker.try_own(key)
    got = []
    waiter = threading.Thread(target=lambda: got.append(broker.acquire(key, 60)))
    waiter.start()
    time.sleep(0.5)  # the waiter is connected, in the owner's backlog
    start = time.monotonic()
    gate.set()
    waiter.join(30)
    try:
        assert got and got[0][0] == broker.OWNER, got
        assert time.monotonic() - start < 10
        assert not owner._thread.is_alive()
    finally:
        owner.close()
        for result in got:
            result[1].close()


@pytest.mark.cpu_only
def test_chunk_broker_reads_a_uid_above_2_31():
    """`SO_PEERCRED` is `struct ucred` (pid_t, uid_t, gid_t): an LDAP-mapped uid of 2**31 or
    more reads back unsigned."""
    import struct

    broker = _host_table_module()._broker
    uid = 2**31 + 5

    class _Conn:
        def getsockopt(self, level, option, size):
            assert size == broker._CRED.size == 12
            return struct.pack("iII", 4242, uid, 2**31 + 7)

    assert broker._peer_uid(_Conn()) == uid


def _fake_shards(rows_per_shard: int, shards: int, cols: int, seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, 256, (shards * rows_per_shard, cols), generator=g, dtype=torch.uint8)
    keys = [f"s{i}" for i in range(shards)]
    weights = {
        k: data[i * rows_per_shard : (i + 1) * rows_per_shard].view(WT.PLE_TABLE_DTYPE)
        for i, k in enumerate(keys)
    }
    return data, keys, weights


def _host_tables(monkeypatch, mode, world, data, keys, weights, rows_per_shard, identity=None):
    """`world` ranks' tables in one process: threads, so `node` ranks wait on each other, and a
    barrier in place of the server's collective agreement. `weights` is every rank's checkpoint,
    or a list of one per rank."""
    import threading

    PH = _host_table_module()
    barrier = threading.Barrier(world)
    votes = []

    def agree(ok):
        votes.append(ok)
        barrier.wait()
        return all(votes)

    monkeypatch.setattr(PH, "_all_agree", agree)
    rows, cols = data.shape
    checkpoints = weights if isinstance(weights, list) else [weights] * world
    sources = [PH.shard_rows(w, keys, rows_per_shard, WT._read) for w in checkpoints]
    tables = [
        PH.PleHostTable(
            mode,
            rows,
            cols,
            WT.PLE_TABLE_DTYPE,
            torch.device("cuda"),
            rank=r,
            world=world,
            dp=1,
            private=WT._host_resident_empty,
            identity=identity,
        )  # fmt: skip
        for r in range(world)
    ]
    threads = [threading.Thread(target=t.fill, args=sources[r]) for r, t in enumerate(tables)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(300)
    return tables


def _vmm_host_numa() -> bool:
    PH = _host_table_module()
    try:
        PH._VmmChunks(torch.cuda.current_device(), PH.CHUNK_ALIGN, 1, 0).close()
    except (RuntimeError, AttributeError):
        return False
    return True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host memory the GPU maps")
@pytest.mark.parametrize("mode", ["rank", "server", "node", "node-fallback"])
def test_host_table_modes(mode, monkeypatch):
    """`rank` holds every row, `server` this rank's rows with its window, `node` every row in
    chunks one rank creates and the other attaches to; `node` without a table identity runs
    `server`. Each holds exactly the checkpoint's bytes, in host memory."""
    import uuid

    PH = _host_table_module()
    if mode == "node" and not _vmm_host_numa():
        pytest.skip("no CUDA VMM host NUMA allocation on this platform")
    rows_per_shard, shards, cols = 40960, 8, 160
    data, keys, weights = _fake_shards(rows_per_shard, shards, cols, seed=5)
    want = {"rank": PH.RANK, "server": PH.SERVER, "node": PH.NODE, "node-fallback": PH.SERVER}
    requested = PH.NODE if mode.startswith("node") else mode
    identity = uuid.uuid4().hex if mode == "node" else None
    tables = _host_tables(monkeypatch, requested, 2, data, keys, weights, rows_per_shard, identity)
    _check_tables(tables, data, want[mode])
    if mode == "node":
        created = sorted(len(t._segments) for t in tables)
        assert created == [PH.CHUNKS // 2, PH.CHUNKS // 2], [t.describe() for t in tables]
    for t in tables:
        t.close()


def _check_tables(tables, data, mode) -> None:
    """Every rank runs `mode` and holds exactly `data`'s bytes (its rows under `server`), in
    host memory."""
    PH = _host_table_module()
    rows, cols = data.shape
    expected = data.cuda()
    for rank, t in enumerate(tables):
        assert t.mode == mode, (rank, t.mode, t.describe())
        assert t.resident_on_host(), t.describe()
        if t.mode == PH.SERVER:
            lo, hi, base = t.window
            assert (lo, hi) == PH.owned_rows(rank, len(tables), rows, cols)
            got = t.table[lo - base : hi - base]
            assert torch.equal(got.view(torch.uint8), expected[lo:hi])
            assert t.needs_sum
        else:
            assert t.window is None and not t.needs_sum
            assert torch.equal(t.table.view(torch.uint8), expected)


def _fallback_warnings(monkeypatch) -> list:
    """The warnings `ple_host_table` emits from here on, with none emitted yet this process."""
    PH = _host_table_module()
    warnings = []
    monkeypatch.setattr(PH, "_WARNED", set())
    monkeypatch.setattr(PH.logger, "warning", warnings.append)
    return warnings


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host memory the GPU maps")
def test_host_table_owner_that_cannot_create_a_chunk_hands_it_over(monkeypatch):
    """A rank whose chunk allocation fails closes the chunk it just owned: the other rank owns
    it at once instead of waiting out the attach timeout, and both run `server`, with one
    warning naming the requested mode, the mode that runs and why."""
    import time
    import uuid

    PH = _host_table_module()
    if not _vmm_host_numa():
        pytest.skip("no CUDA VMM host NUMA allocation on this platform")
    monkeypatch.setattr(PH, "_ATTACH_TIMEOUT_S", 60.0)
    warnings = _fallback_warnings(monkeypatch)
    create, failed = PH._VmmChunks.create, []

    def create_once(self, k):
        if k == 0 and not failed:
            failed.append(k)
            raise RuntimeError("CUDA driver call failed: out of host memory")
        create(self, k)

    monkeypatch.setattr(PH._VmmChunks, "create", create_once)
    rows_per_shard, shards, cols = 40960, 8, 160
    data, keys, weights = _fake_shards(rows_per_shard, shards, cols, seed=7)
    start = time.monotonic()
    tables = _host_tables(
        monkeypatch, PH.NODE, 2, data, keys, weights, rows_per_shard, uuid.uuid4().hex
    )
    elapsed = time.monotonic() - start
    try:
        assert failed == [0] and "out of host memory" in tables[0]._node_error
        assert elapsed < 20, f"the other rank waited {elapsed:.0f} s for the failed chunk"
        _check_tables(tables, data, PH.SERVER)
        assert len(warnings) == 1, warnings
        assert f"{PH.PLE_HOST_TABLE_ENV}=node requested, running server" in warnings[0]
    finally:
        for t in tables:
            t.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host memory the GPU maps")
def test_host_table_refuses_a_chunk_that_differs_from_its_checkpoint(monkeypatch):
    """Two ranks whose keys agree but whose checkpoints differ at one row of an attached chunk's
    sample: the attaching rank raises on it, and both run `server` from their own bytes."""
    import uuid

    PH = _host_table_module()
    if not _vmm_host_numa():
        pytest.skip("no CUDA VMM host NUMA allocation on this platform")
    warnings = _fallback_warnings(monkeypatch)
    rows_per_shard, shards, cols = 40960, 8, 160
    data, keys, weights = _fake_shards(rows_per_shard, shards, cols, seed=9)
    other = data.clone()
    row = PH.sample_blocks(0, *data.shape, phase=1)[3][0]
    other[row] ^= 1
    other_weights = {
        k: other[i * rows_per_shard : (i + 1) * rows_per_shard].view(WT.PLE_TABLE_DTYPE)
        for i, k in enumerate(keys)
    }
    tables = _host_tables(
        monkeypatch,
        PH.NODE,
        2,
        data,
        keys,
        [weights, other_weights],
        rows_per_shard,
        uuid.uuid4().hex,
    )
    try:
        # Row `row` is rank 0's (chunk 0): each rank's own rows are its checkpoint's.
        _check_tables(tables, data, PH.SERVER)
        assert tables[1]._node_error == f"shared chunk 0 differs from the checkpoint at row {row}"
        assert len(warnings) == 1, warnings
        assert f"{PH.PLE_HOST_TABLE_ENV}=node requested, running server" in warnings[0]
    finally:
        for t in tables:
            t.close()


@pytest.mark.cpu_only
def test_table_identity_is_the_sampled_rows():
    """The `node` key is the table's bytes at every chunk's phase-0 sample and nothing about
    where they were read from; phase 1, the rows an attaching rank compares, is disjoint."""
    PH = _host_table_module()
    rows_per_shard, shards, cols = 40960, 8, 160
    data, keys, _ = _fake_shards(rows_per_shard, shards, cols, seed=11)
    rows = data.shape[0]

    def identity(table) -> str:
        parts = {
            k: table[i * rows_per_shard : (i + 1) * rows_per_shard] for i, k in enumerate(keys)
        }
        _, read = PH.shard_rows(parts, keys, rows_per_shard, WT._read)
        return PH.table_identity(read, rows, cols, "layer 3")

    def sample(phase: int) -> set:
        return {
            r
            for k in range(PH.CHUNKS)
            for lo, hi in PH.sample_blocks(k, rows, cols, phase=phase)
            for r in range(lo, hi)
        }

    key = identity(data)
    assert identity(data.clone()) == key
    sampled, checked = sample(0), sample(1)
    assert sampled and checked and not sampled & checked
    assert all(0 <= r < rows for r in sampled | checked)
    changed = data.clone()
    changed[max(sampled)] ^= 1
    assert identity(changed) != key
    # The key hashes its sample only; the attaching rank's check covers this row.
    changed = data.clone()
    changed[min(checked)] ^= 1
    assert identity(changed) == key


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "ids, rows_per_shard, error",
    [
        (800, 208, None),
        (800, 256, "Shards padded one by one"),
        (800, 200, "not padded to a multiple"),
        (900, 208, "address 900"),
    ],
    ids=["padded-once", "padded-per-shard", "unpadded", "short"],
)
def test_ngram_table_height_matches_the_id_space(ids, rows_per_shard, error):
    """The table's height is the id space its offsets and sizes address, padded to the divisor by
    less than one divisor; otherwise the declaration raises."""
    d = SimpleNamespace(
        ple_layer=3, ple_shards=4, ple_head_dim=4, ngram_heads=2, ple_embed_dim=8,
        ple_table_dtype=WT.PLE_TABLE_DTYPE, ngram_vocab_divisor=64,
    )  # fmt: skip
    prefix = f"{WT._LM}.layers.3.ple.ple_embedding"
    weights = {
        key: torch.zeros(rows_per_shard, 4, dtype=torch.uint8).view(WT.PLE_TABLE_DTYPE)
        for key in WT.ngram_shard_keys(d)
    }
    weights[f"{prefix}.ngram_heads_offsets"] = torch.tensor([0, ids // 2])
    weights[f"{prefix}.ngram_heads_vocab_sizes"] = torch.tensor([ids // 2, ids - ids // 2])
    core = SimpleNamespace(w={"embed": torch.empty(0)})
    if error is None:
        assert WT._declare_ngram_table(core, d, weights) == rows_per_shard
        assert tuple(core.w["ple_table"].shape) == (4 * rows_per_shard, 4)
    else:
        with pytest.raises(ValueError, match=error):
            WT._declare_ngram_table(core, d, weights)


@pytest.mark.cpu_only
def test_host_table_warns_once_when_server_runs_as_rank(monkeypatch):
    """`server` under attention DP keeps a copy per rank: one warning per process says so."""
    PH = _host_table_module()
    warnings = _fallback_warnings(monkeypatch)

    def private(rows, cols, dtype, device):
        return torch.empty(rows, cols, dtype=dtype), None

    for _ in range(2):
        t = PH.PleHostTable(
            PH.SERVER, 64, 4, WT.PLE_TABLE_DTYPE, torch.device("cuda"),
            rank=0, world=2, dp=2, private=private,
        )  # fmt: skip
        assert t.mode == PH.RANK
    assert len(warnings) == 1, warnings
    assert f"{PH.PLE_HOST_TABLE_ENV}=server requested, running rank" in warnings[0]
    assert "attention DP 2" in warnings[0]
