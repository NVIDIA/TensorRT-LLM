# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The deepseek default target's per-rank weight shares, checked on the CPU.

Three claims, each a way the default target could load a plausible wrong model,
or not load at all, without any kernel noticing:

* across the ranks of every admitted layout, the shares tile the model --
  every expert and every MLP and expert intermediate column is held exactly
  once per group (the heads reassemble in the attention check below);
* a rank's operands, read through the table's own cut and transform, are the
  operands the same table builds at world size 1 -- where it cuts nothing --
  for a model the size of the rank's share, from a checkpoint holding only
  that share. The test computes the share's rows and columns itself, so the
  table's cuts are checked against arithmetic rather than against themselves,
  and the oracle is the unsharded relayout, which is what every layout
  computes at world size 1. It is also what pins the order of cut and
  relayout: the scale swizzle and the expert row shuffle permute within
  blocks, so cutting after them would not match;
* `load` reads only the rank's share of every split tensor from the lazy
  checkpoint entry -- never a whole tensor it then cuts -- and still covers the
  checkpoint in both directions, with the MTP layer off and on. Reading whole
  tensors is what put a 4-rank tray over host memory under moe_tp.

The MTP layer is held to the same three: its attention rows are a trunk
layer's one layer further along, and its bf16 shared expert and experts are
cut by the same groups as the trunk's NVFP4 ones.

No GPU: the transforms are plain torch plus `block_scale_interleave`, run on
small synthetic tensors with deepseek's real head and intermediate geometry
(and a tiny model for the end-to-end `load`). World size 1 is not a layout
routing admits; it is built here only as the oracle.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v3 import routing
from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v3.r1_0528_nvfp4__sm_10x__default import (  # noqa: E501
    weights as D,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module", autouse=True)
def _limit_cpu_threads() -> Iterator[None]:
    """Keep small tensor transforms within CPU-constrained workers' thread budget."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 2))
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)


_LAYOUTS = sorted(routing._DEFAULT_LAYOUTS)
_MTP_LAYOUTS = sorted(routing._MTP_LAYOUTS)

# The oracle: one rank, attention TP, nothing split.
_WS1 = (1, False, 1, 1)


def _first_per(key) -> list:
    """One layout per value of `key(layout)`: the layouts that share a value hand
    the test the same inputs, so the rest would rerun the same case."""
    seen: dict = {}
    for layout in _LAYOUTS:
        seen.setdefault(key(layout), layout)
    return list(seen.values())


# The attention and MLP shares depend on the attention TP size alone, the
# expert share on (moe_tp, moe_ep) alone.
_ATTN_LAYOUTS = _first_per(lambda lay: 1 if lay[1] else lay[0])
_EXPERT_LAYOUTS = _first_per(lambda lay: (lay[2], lay[3]))


def _layout_id(layout) -> str:
    ws, dp, mt, me = layout
    return f"ws{ws}-{'dp' if dp else 'tp'}-moetp{mt}-moeep{me}"


def _cfg(**overrides):
    """deepseek-r1-0528's geometry; tests shrink `hidden_size` where it only
    sets a tensor's other axis."""
    fields = dict(
        num_hidden_layers=61,
        hidden_size=7168,
        num_attention_heads=128,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_lora_rank=512,
        q_lora_rank=1536,
        vocab_size=129280,
        first_k_dense_replace=3,
        n_routed_experts=256,
        moe_intermediate_size=2048,
        n_shared_experts=1,
        intermediate_size=18432,
        num_nextn_predict_layers=1,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _mapping(layout, rank: int) -> Mapping:
    ws, dp, mt, me = layout
    return Mapping(
        world_size=ws,
        rank=rank,
        tp_size=ws,
        moe_tp_size=mt,
        moe_ep_size=me,
        enable_attention_dp=dp,
    )


def _default_core(layout, rank: int, cfg, mtp: bool = False) -> SimpleNamespace:
    core = SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=cfg,
            mapping=_mapping(layout, rank),
            torch_dtype=torch.bfloat16,
            spec_config=object() if mtp else None,
        )
    )
    core.dims = D.MODEL_WEIGHTS.dims(core)
    return core


def _blocks(start: int, width: int) -> list[int]:
    return list(range(start, start + width))


# --- a checkpoint whose entries record what was read from them ---


class _LazySlice:
    """Stands in for a safetensors `PySafeSlice`: indexing it is the read, and
    every read is recorded as (key, index, elements read)."""

    def __init__(self, key: str, full: torch.Tensor, reads: list):
        self.key, self._full, self._reads = key, full, reads

    def __getitem__(self, index):
        out = self._full[index]
        self._reads.append((self.key, index, out.numel()))
        return out


def _ckpt_spec(key: str, cfg) -> tuple[tuple[int, ...], torch.dtype]:
    """The shape and dtype the R1 checkpoint stores under `key`, at `cfg`'s sizes."""
    h, hid = cfg.num_attention_heads, cfg.hidden_size
    nope, rope, v = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
    bf16, u8, e4m3, f32 = torch.bfloat16, torch.uint8, torch.float8_e4m3fn, torch.float32
    if key.endswith(("input_scale", "weight_scale_2", "k_scale", "v_scale")):
        return (), f32
    if ".mlp.experts." in key:
        inter = cfg.moe_intermediate_size
    elif ".mlp.shared_experts." in key:
        inter = cfg.moe_intermediate_size * cfg.n_shared_experts
    else:
        inter = cfg.intermediate_size
    tail = key.rsplit(".mlp.", 1)[-1]
    # The MTP layer's MLP is bf16: modelopt excludes model.layers.61* from NVFP4.
    if key.startswith(f"model.layers.{cfg.num_hidden_layers}.") and ".mlp." in key:
        bf16_mlp = {
            "gate_proj.weight": ((inter, hid), bf16),
            "up_proj.weight": ((inter, hid), bf16),
            "down_proj.weight": ((hid, inter), bf16),
        }
        for suffix, spec in bf16_mlp.items():
            if tail.endswith(suffix):
                return spec
    nvfp4 = {
        "gate_proj.weight": ((inter, hid // 2), u8),
        "up_proj.weight": ((inter, hid // 2), u8),
        "down_proj.weight": ((hid, inter // 2), u8),
        "gate_proj.weight_scale": ((inter, hid // 16), e4m3),
        "up_proj.weight_scale": ((inter, hid // 16), e4m3),
        "down_proj.weight_scale": ((hid, inter // 16), e4m3),
    }
    for suffix, spec in nvfp4.items():
        if ".mlp." in key and tail.endswith(suffix):
            return spec
    other = {
        "input_layernorm.weight": ((hid,), bf16),
        "post_attention_layernorm.weight": ((hid,), bf16),
        "q_a_proj.weight": ((cfg.q_lora_rank, hid), bf16),
        "q_a_layernorm.weight": ((cfg.q_lora_rank,), bf16),
        "q_b_proj.weight": ((h * (nope + rope), cfg.q_lora_rank), bf16),
        "kv_a_proj_with_mqa.weight": ((cfg.kv_lora_rank + rope, hid), bf16),
        "kv_a_layernorm.weight": ((cfg.kv_lora_rank,), bf16),
        "kv_b_proj.weight": ((h * (nope + v), cfg.kv_lora_rank), bf16),
        "o_proj.weight": ((hid, h * v), bf16),
        "mlp.gate.weight": ((cfg.n_routed_experts, hid), bf16),
        "e_score_correction_bias": ((cfg.n_routed_experts,), f32),
        "model.norm.weight": ((hid,), bf16),
        "embed_tokens.weight": ((cfg.vocab_size, hid), bf16),
        "lm_head.weight": ((cfg.vocab_size, hid), bf16),
        "eh_proj.weight": ((hid, 2 * hid), bf16),
        ".enorm.weight": ((hid,), bf16),
        ".hnorm.weight": ((hid,), bf16),
        "shared_head.norm.weight": ((hid,), bf16),
        "shared_head.head.weight": ((cfg.vocab_size, hid), bf16),
    }
    for suffix, spec in other.items():
        if key.endswith(suffix):
            return spec
    raise KeyError(key)


_FULL: dict = {}


def _full(key: str, cfg) -> torch.Tensor:
    """The whole tensor under `key`: deterministic in the key, so two checkpoints
    built from one config agree. Built once per (key, config); nothing writes it."""
    memo = (key, tuple(sorted(vars(cfg).items())))
    if memo not in _FULL:
        _FULL[memo] = _make_full(key, cfg)
    return _FULL[memo]


def _make_full(key: str, cfg) -> torch.Tensor:
    shape, dtype = _ckpt_spec(key, cfg)
    g = torch.Generator().manual_seed(zlib.crc32(key.encode()))
    if dtype == torch.uint8 or dtype == torch.float8_e4m3fn:
        t = torch.randint(0, 256, shape, generator=g, dtype=torch.int32).to(torch.uint8)
        return t.view(dtype)
    return torch.randn(shape, generator=g, dtype=torch.float32).to(dtype)


class _Checkpoint(dict):
    """`key -> entry` as the engine hands it to `load`: 0-dim scalars as tensors,
    everything else lazy. Entries are built on first access, so a test that reads
    a handful of keys of a 61-layer model never builds the rest."""

    def __init__(self, cfg, keys=None):
        super().__init__((k, None) for k in keys or ())
        # Without a key list every key exists, so a test can read a few of a
        # full-size model's tensors without enumerating the rest.
        self.cfg, self.reads, self.open = cfg, [], keys is None

    def __contains__(self, key):
        return self.open or dict.__contains__(self, key)

    def __getitem__(self, key):
        entry = dict.get(self, key)
        if entry is None:
            full = _full(key, self.cfg)
            entry = full if full.ndim == 0 else _LazySlice(key, full, self.reads)
            dict.__setitem__(self, key, entry)
        return entry

    def full(self, key) -> torch.Tensor:
        return _full(key, self.cfg)


def _read_first(core, ckpt, param_key):
    """The tensor `load` copies into `param_key`'s first source, through the
    table's own cut and transform, and the cut it read with."""
    return _read_source(core, ckpt, param_key, 0)


def _read_source(core, ckpt, param_key, n: int):
    """The tensor `load` copies from `param_key`'s `n`-th source (the `n`-th local
    expert of an expert stack), through the table's own cut and transform."""
    if not hasattr(core, "manifest"):
        core.manifest = D.MODEL_WEIGHTS.manifest(core)
    ckpt_key, _, cut, transform = core.manifest[param_key][n]
    return D.MODEL_WEIGHTS.read(ckpt, ckpt_key, cut, transform, core), ckpt_key, cut


def _assert_read(ckpt, keys, index):
    """Every key was read exactly once, with exactly `index`."""
    for key in keys if isinstance(keys, tuple) else (keys,):
        got = [i for k, i, _ in ckpt.reads if k == key]
        assert got == [index[0] if len(index) == 1 else index], (key, got, index)


# --- the oracle: the world-size-1 table over a checkpoint of the rank's share ---


class _ShareCheckpoint(dict):
    """The checkpoint a one-rank model of the share's size would ship: under every
    key, `index` of the full tensor, materialized. The world-size-1 table reads it
    whole -- its cuts span the share -- so what it loads is its own relayout of
    exactly that slice."""

    def __init__(self, cfg, index):
        super().__init__()
        self.cfg, self.index = cfg, index

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        return _full(key, self.cfg)[self.index].contiguous()


_WS1_CORES: dict = {}


def _ws1(share_cfg, full_cfg, param_key, ckpt_key, index, mtp=False) -> torch.Tensor:
    """What the world-size-1 table loads into `param_key` from `ckpt_key`, for a model
    of `share_cfg`'s size whose checkpoint holds `index` of `full_cfg`'s tensors."""
    memo = (tuple(sorted(vars(share_cfg).items())), mtp)
    if memo not in _WS1_CORES:
        core = _default_core(_WS1, 0, share_cfg, mtp)
        core.manifest = D.MODEL_WEIGHTS.manifest(core)
        _WS1_CORES[memo] = core
    core = _WS1_CORES[memo]
    (source,) = [s for s in core.manifest[param_key] if s[0] == ckpt_key]
    _, _, cut, transform = source
    return D.MODEL_WEIGHTS.read(_ShareCheckpoint(full_cfg, index), ckpt_key, cut, transform, core)


def _check(core, ckpt, param_key, index, oracle, n=0) -> None:
    """`param_key`'s `n`-th source on this rank reads exactly `index` of each of its
    keys and loads exactly what `oracle(ckpt_key)` says the world-size-1 table loads
    from that share."""
    got, key, _ = _read_source(core, ckpt, param_key, n)
    _assert_read(ckpt, key, index)
    want = oracle(key)
    assert got.shape == want.shape, (param_key, got.shape, want.shape)
    assert torch.equal(got, want), param_key


@pytest.mark.parametrize("layout", _LAYOUTS, ids=_layout_id)
def test_shares_tile_intermediates_and_experts(layout):
    cfg = _cfg()
    ws = layout[0]
    dims = [_default_core(layout, r, cfg).dims for r in range(ws)]
    group = dims[: dims[0].attn_tp]
    for name, full in (
        ("dense_inter", cfg.intermediate_size),
        ("shared_inter", cfg.moe_intermediate_size * cfg.n_shared_experts),
    ):
        cols = sorted(
            c for d in group for c in _blocks(d.mlp_tp_rank * getattr(d, name), getattr(d, name))
        )
        assert cols == list(range(full)), name

    for tp_slot in range(layout[2]):
        slot = [d for d in dims if d.moe_tp_rank == tp_slot]
        experts = sorted(e for d in slot for e in _blocks(d.expert_offset, d.local_experts))
        assert experts == list(range(cfg.n_routed_experts)), (tp_slot, experts[:8])


@pytest.mark.parametrize("layout", _ATTN_LAYOUTS, ids=_layout_id)
@pytest.mark.parametrize("mtp", [False, True], ids=["trunk", "mtp"])
def test_attention_operands_are_the_ws1_operands_of_the_rank_share(layout, mtp):
    """q_b rows, the regrouped kv_b and o_proj columns of each rank are the
    world-size-1 operands of a model with the rank's heads alone, read as
    exactly that share; over the group they reassemble the projections. The
    MTP layer's are cut as a trunk layer's, one layer further along."""
    cfg = _cfg(hidden_size=256)
    nope, rope, v = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim
    layer = cfg.num_hidden_layers if mtp else 0
    prefix = f"model.layers.{layer}.self_attn"
    qs, os_ = [], []
    for rank in range(layout[0]):
        core = _default_core(layout, rank, cfg, mtp)
        d = core.dims
        if rank >= d.attn_tp:
            continue
        hr, r = d.heads, d.attn_tp_rank
        share = _cfg(hidden_size=256, num_attention_heads=hr)
        ckpt = _Checkpoint(cfg)

        q_b, key, _ = _read_first(core, ckpt, f"l{layer}_qb")
        _assert_read(ckpt, key, (slice(r * hr * (nope + rope), (r + 1) * hr * (nope + rope)),))
        qs.append(q_b)

        rows = (slice(r * hr * (nope + v), (r + 1) * hr * (nope + v)),)
        _check(
            core,
            ckpt,
            f"l{layer}_kvb",
            rows,
            lambda k: _ws1(share, cfg, f"l{layer}_kvb", k, rows, mtp),
        )

        o, key, _ = _read_first(core, ckpt, f"l{layer}_o")
        _assert_read(ckpt, key, (slice(None), slice(r * hr * v, (r + 1) * hr * v)))
        os_.append(o)
    assert torch.equal(torch.cat(qs), _full(f"{prefix}.q_b_proj.weight", cfg))
    assert torch.equal(torch.cat(os_, dim=1), _full(f"{prefix}.o_proj.weight", cfg))


def _intermediate_index(start: int, width: int, packed: bool):
    """(rows, columns, scale columns) of one intermediate slice: the gate/up rows,
    the down weight's columns (two values a byte when `packed`) and the down scale
    columns (one per 16 values)."""
    rows = slice(start, start + width)
    per_byte = 2 if packed else 1
    cols = slice(start // per_byte, (start + width) // per_byte)
    cols16 = slice(start // 16, (start + width) // 16)
    return rows, cols, cols16


@pytest.mark.parametrize("layout", _ATTN_LAYOUTS, ids=_layout_id)
@pytest.mark.parametrize("family", ["dense", "shared"])
def test_mlp_operands_are_the_ws1_operands_of_the_rank_share(layout, family):
    """The fused gate_up and the down operands (weights and swizzled scales)
    of each rank are the world-size-1 table's for the rank's intermediate slice,
    and the cut is all that is read."""
    cfg = _cfg(hidden_size=256)
    layer = 0 if family == "dense" else cfg.first_k_dense_replace
    for rank in range(layout[0]):
        core = _default_core(layout, rank, cfg)
        d = core.dims
        ckpt = _Checkpoint(cfg)
        per = getattr(d, f"{family}_inter")
        rows, cols2, cols16 = _intermediate_index(d.mlp_tp_rank * per, per, packed=True)
        share = (
            _cfg(hidden_size=256, intermediate_size=per)
            if family == "dense"
            else _cfg(hidden_size=256, moe_intermediate_size=per, n_shared_experts=1)
        )
        for name, index in (
            ("mlp_gu_w", (rows,)),
            ("mlp_gu_s", (rows,)),
            ("mlp_dn_w", (slice(None), cols2)),
            ("mlp_dn_s", (slice(None), cols16)),
        ):
            param = f"l{layer}_{name}"
            _check(
                core,
                ckpt,
                param,
                index,
                lambda k, p=param, i=index: _ws1(share, cfg, p, k, i),
            )


@pytest.mark.parametrize("layout", _EXPERT_LAYOUTS, ids=_layout_id)
def test_expert_operands_are_the_ws1_operands_of_the_rank_share(layout):
    """The first and last local expert's FC1/FC2 weights and scales on each rank
    are the world-size-1 table's for that expert at the rank's moe_tp slice, and
    the cut is all that is read."""
    cfg = _cfg(hidden_size=256)
    layer = cfg.first_k_dense_replace
    for rank in range(layout[0]):
        core = _default_core(layout, rank, cfg)
        d = core.dims
        share = _cfg(hidden_size=256, moe_intermediate_size=d.moe_inter)
        rows, cols2, cols16 = _intermediate_index(d.moe_tp_rank * d.moe_inter, d.moe_inter, True)
        for local in (0, d.local_experts - 1):
            ckpt = _Checkpoint(cfg)
            for name, index in (
                ("fc1_w", (rows,)),
                ("fc1_s", (rows,)),
                ("fc2_w", (slice(None), cols2)),
                ("fc2_s", (slice(None), cols16)),
            ):
                param = f"l{layer}_{name}"
                _check(
                    core,
                    ckpt,
                    param,
                    index,
                    lambda k, p=param, i=index: _ws1(share, cfg, p, k, i),
                    n=local,
                )


@pytest.mark.parametrize("layout", _ATTN_LAYOUTS, ids=_layout_id)
def test_mtp_shared_expert_is_the_ws1_one_of_the_rank_share(layout):
    """The MTP layer's bf16 shared expert is split over the attention TP group like
    the trunk's NVFP4 one: gate and up rows, down columns, one value an element."""
    cfg = _cfg(hidden_size=256)
    layer = cfg.num_hidden_layers
    for rank in range(layout[0]):
        core = _default_core(layout, rank, cfg, mtp=True)
        d = core.dims
        ckpt = _Checkpoint(cfg)
        per = d.shared_inter
        rows, cols, _ = _intermediate_index(d.mlp_tp_rank * per, per, packed=False)
        share = _cfg(hidden_size=256, moe_intermediate_size=per, n_shared_experts=1)
        for name, index in (("sh_gu", (rows,)), ("sh_dn", (slice(None), cols))):
            param = f"l{layer}_{name}"
            _check(
                core,
                ckpt,
                param,
                index,
                lambda k, p=param, i=index: _ws1(share, cfg, p, k, i, mtp=True),
            )


@pytest.mark.parametrize("layout", _EXPERT_LAYOUTS, ids=_layout_id)
def test_mtp_expert_operands_are_the_ws1_operands_of_the_rank_share(layout):
    """The MTP layer's bf16 experts are cut by the same EP window and moe_tp slice
    as the trunk's: `[up ; gate]` rows of the slice for FC1, its columns for FC2."""
    cfg = _cfg(hidden_size=256)
    layer = cfg.num_hidden_layers
    for rank in range(layout[0]):
        core = _default_core(layout, rank, cfg, mtp=True)
        d = core.dims
        share = _cfg(hidden_size=256, moe_intermediate_size=d.moe_inter)
        rows, cols, _ = _intermediate_index(d.moe_tp_rank * d.moe_inter, d.moe_inter, False)
        for local in (0, d.local_experts - 1):
            ckpt = _Checkpoint(cfg)
            for name, index in (("fc1", (rows,)), ("fc2", (slice(None), cols))):
                param = f"l{layer}_{name}"
                _check(
                    core,
                    ckpt,
                    param,
                    index,
                    lambda k, p=param, i=index: _ws1(share, cfg, p, k, i, mtp=True),
                    n=local,
                )
            # FC1 is [up ; gate], the order fused_moe reads, opposite to the dense
            # gate_up linear's.
            q = f"model.layers.{layer}.mlp.experts.{d.expert_offset + local}"
            fc1, _, _ = _read_source(core, _Checkpoint(cfg), f"l{layer}_fc1", local)
            up, gate = _full(f"{q}.up_proj.weight", cfg), _full(f"{q}.gate_proj.weight", cfg)
            assert torch.equal(fc1, torch.cat([up[rows], gate[rows]])), (layout, rank, local)


@pytest.mark.parametrize("layout", _MTP_LAYOUTS, ids=_layout_id)
def test_mtp_own_weights_are_replicated(layout):
    """`eh_proj`, the three MTP norms and the router are whole on every rank, read
    whole, at every layout MTP is admitted at."""
    cfg = _cfg(hidden_size=256)
    layer = cfg.num_hidden_layers
    q = f"model.layers.{layer}"
    for rank in range(layout[0]):
        core = _default_core(layout, rank, cfg, mtp=True)
        ckpt = _Checkpoint(cfg)
        for name, key in (
            ("eh", f"{q}.eh_proj.weight"),
            ("enorm", f"{q}.enorm.weight"),
            ("hnorm", f"{q}.hnorm.weight"),
            ("head_norm", f"{q}.shared_head.norm.weight"),
            ("router", f"{q}.mlp.gate.weight"),
        ):
            got, read_key, _ = _read_first(core, ckpt, f"l{layer}_{name}")
            assert read_key == key, (name, read_key)
            _assert_read(ckpt, key, (slice(None),))
            assert torch.equal(got, _full(key, cfg)), (layout, rank, name)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_lm_head_reads_the_vocab_shard_and_pads_the_last(tp):
    """A vocab that does not split evenly: each rank reads its rows and no
    others, the last shard's missing rows come back zero, and the shards
    reassemble the head. One rank holding all rows reads it whole."""
    cfg = _cfg(hidden_size=4, vocab_size=1001)
    rows = math.ceil(cfg.vocab_size / tp)
    full = _full("lm_head.weight", cfg)
    shards = []
    for rank in range(tp):
        ckpt = _Checkpoint(cfg)
        cut, pad = D.lm_head_share(rows, cfg.vocab_size, rank)
        got = D.MODEL_WEIGHTS.read(ckpt, "lm_head.weight", cut, pad, None)
        stop = min((rank + 1) * rows, cfg.vocab_size)
        _assert_read(ckpt, "lm_head.weight", (slice(rank * rows, stop),))
        assert got.shape == (rows, cfg.hidden_size)
        assert not got[stop - rank * rows :].any()
        shards.append(got[: stop - rank * rows])
    assert torch.equal(torch.cat(shards), full)


# --- `load`, end to end on a tiny model ---

# Small enough to allocate for real, and still every split the layouts take:
# 8 heads and 1024-wide intermediates split 8 ways onto 128-row tiles.
_TINY = dict(
    num_hidden_layers=3,
    hidden_size=256,
    num_attention_heads=8,
    qk_nope_head_dim=16,
    qk_rope_head_dim=8,
    v_head_dim=16,
    kv_lora_rank=64,
    q_lora_rank=64,
    vocab_size=1001,
    first_k_dense_replace=1,
    n_routed_experts=16,  # divisible by every admitted EP size, up to dep16's
    moe_intermediate_size=1024,
    n_shared_experts=1,
    intermediate_size=1024,
    num_nextn_predict_layers=1,
)


def _tiny_keys(cfg) -> list[str]:
    """Every key the R1 checkpoint carries, at the tiny model's sizes, the MTP
    layer's included."""
    nvfp4 = [
        f"{p}_proj.{s}"
        for p in ("gate", "up", "down")
        for s in ("weight", "weight_scale", "input_scale", "weight_scale_2")
    ]
    keys = ["model.norm.weight", "model.embed_tokens.weight", "lm_head.weight"]
    for i in range(cfg.num_hidden_layers):
        q = f"model.layers.{i}"
        keys += [f"{q}.input_layernorm.weight", f"{q}.post_attention_layernorm.weight"]
        keys += [
            f"{q}.self_attn.{n}"
            for n in (
                "q_a_proj.weight",
                "q_a_layernorm.weight",
                "q_b_proj.weight",
                "kv_a_proj_with_mqa.weight",
                "kv_a_layernorm.weight",
                "kv_b_proj.weight",
                "o_proj.weight",
                "k_proj.k_scale",
                "v_proj.v_scale",
            )
        ]
        if i < cfg.first_k_dense_replace:
            keys += [f"{q}.mlp.{n}" for n in nvfp4]
            continue
        keys += [f"{q}.mlp.gate.weight", f"{q}.mlp.gate.e_score_correction_bias"]
        keys += [f"{q}.mlp.shared_experts.{n}" for n in nvfp4]
        keys += [f"{q}.mlp.experts.{e}.{n}" for e in range(cfg.n_routed_experts) for n in nvfp4]
    # The MTP layer: a trunk layer's attention block and router, a bf16 shared
    # expert and bf16 experts, and its own front end and head.
    q = f"model.layers.{cfg.num_hidden_layers}"
    bf16 = [f"{p}_proj.weight" for p in ("gate", "up", "down")]
    keys += [f"{q}.input_layernorm.weight", f"{q}.post_attention_layernorm.weight"]
    keys += [
        f"{q}.self_attn.{n}"
        for n in (
            "q_a_proj.weight",
            "q_a_layernorm.weight",
            "q_b_proj.weight",
            "kv_a_proj_with_mqa.weight",
            "kv_a_layernorm.weight",
            "kv_b_proj.weight",
            "o_proj.weight",
            "k_proj.k_scale",
            "v_proj.v_scale",
        )
    ]
    keys += [f"{q}.mlp.gate.weight", f"{q}.mlp.gate.e_score_correction_bias"]
    keys += [f"{q}.mlp.shared_experts.{n}" for n in bf16]
    keys += [f"{q}.mlp.experts.{e}.{n}" for e in range(cfg.n_routed_experts) for n in bf16]
    keys += [
        f"{q}.{n}"
        for n in (
            "enorm.weight",
            "hnorm.weight",
            "eh_proj.weight",
            "shared_head.norm.weight",
            "shared_head.head.weight",
            "embed_tokens.weight",
        )
    ]
    return keys


def _split_parts(key: str, d) -> int:
    """How many ranks share `key` under these dims: the group its rows or columns
    are split over, 1 for a replicated key."""
    if ".mlp.experts." in key and key.endswith(("weight", "weight_scale")):
        return d.moe_tp
    if key.endswith(("q_b_proj.weight", "kv_b_proj.weight", "o_proj.weight")):
        return d.attn_tp
    if ".mlp." in key and key.endswith(("_proj.weight", "_proj.weight_scale")):
        return d.mlp_tp
    return 1


def _load_cases():
    """Every admitted layout without MTP, and every layout MTP is admitted at with it."""
    return [pytest.param(lay, False, id=_layout_id(lay)) for lay in _LAYOUTS] + [
        pytest.param(lay, True, id=f"{_layout_id(lay)}-mtp") for lay in _MTP_LAYOUTS
    ]


@pytest.mark.parametrize("layout, mtp", _load_cases())
def test_load_reads_only_the_rank_share(layout, mtp, monkeypatch):
    """`load` on every rank: coverage holds both ways, every key is read at most
    once, and a split key is read as exactly its share's elements -- the whole
    tensor never leaves the checkpoint on a rank that holds part of it. With MTP
    the layer is consumed but for its two aliased keys and its off-window
    experts; without it, it is left whole."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    cfg = SimpleNamespace(**_TINY)
    keys = _tiny_keys(cfg)
    mtp_prefix = f"model.layers.{cfg.num_hidden_layers}."
    for rank in range(layout[0]):
        core = _default_core(layout, rank, cfg, mtp)
        d = core.dims
        core.w = D.MODEL_WEIGHTS.declare(d)
        rows = cfg.vocab_size if d.attention_dp else math.ceil(cfg.vocab_size / layout[0])
        lm_head = nn.Parameter(torch.empty(rows, cfg.hidden_size, dtype=torch.bfloat16))
        params = [("lm_head.weight", lm_head)] + [(f"model.w.{k}", p) for k, p in core.w.items()]
        model = SimpleNamespace(
            model=core, lm_head=SimpleNamespace(weight=lm_head), named_parameters=lambda: params
        )
        ckpt = _Checkpoint(cfg, keys)
        D.MODEL_WEIGHTS.load(model, ckpt)

        seen = [k for k, _, _ in ckpt.reads]
        assert len(seen) == len(set(seen)), "a checkpoint key was read twice"
        for key, index, numel in ckpt.reads:
            full = _full(key, cfg).numel()
            if key == "lm_head.weight":
                assert numel <= rows * cfg.hidden_size, (layout, rank, index)
                continue
            parts = _split_parts(key, d)
            assert numel * parts == full, (layout, rank, key, index, numel, full)
        read_mtp = {k for k in seen if k.startswith(mtp_prefix)}
        if not mtp:
            assert not read_mtp, sorted(read_mtp)[:4]
            continue
        # Every lazy key of the layer (the 0-dim KV scales are not reads) but the two
        # aliased ones and the off-window experts.
        lo, hi = d.expert_offset, d.expert_offset + d.local_experts
        want = {
            k
            for k in keys
            if k.startswith(mtp_prefix)
            and _full(k, cfg).ndim
            and not k.endswith(("embed_tokens.weight", "shared_head.head.weight"))
            and not (".mlp.experts." in k and not lo <= int(k.split(".")[5]) < hi)
        }
        assert read_mtp == want, (sorted(want - read_mtp)[:4], sorted(read_mtp - want)[:4])
