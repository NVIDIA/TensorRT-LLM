# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The unified target's per-rank weight table, checked against the checkpoint on the CPU.

From `model.safetensors.index.json` and the safetensors headers alone -- no tensor data is
read -- at each member (tp4, tep4, dep4):

* **coverage**: at every rank, with MTP off and on, every checkpoint tensor is either read by
  the rank's table or predicted unread by `expected_unconsumed`, never both; across a member's
  ranks every tensor is read by some rank, and each routed or draft expert's tensors by
  exactly one;
* **shapes**: at ranks 0 and 3, every source read through its cut lies inside the stored
  tensor, and the cut run through the table's own transform (on meta tensors) has exactly
  the shape and dtype of the parameter it fills. A rank reads a split tensor at its share
  only, and every other tensor whole, once;
* **the modules' shapes**: the per-rank widths the forward is written against, and the head
  ranges the same rank's K/V and recurrent pools hold (`pool.pool_geometry`).

And, on small synthetic tensors whose values name their rows, that each cut selects the
rank's own heads in the order the fused operands lay them out.

tp4 splits each routed expert's intermediate four ways, a layout this table does not
declare: at tp4 the test holds what no rank reads to exactly those experts' tensors.
"""

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

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified import (  # noqa: E501
    pool as P,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified import (  # noqa: E501
    weights as WT,
)
from tensorrt_llm._torch.configs import Qwen4ExpTextConfig, Qwen4ExpVisionConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

CHECKPOINT = "Qwen3.8-Flash-Next-NVFP4"

#: Each member's mapping: attention TP4 with the experts split by intermediate (tp4) or by
#: expert (tep4), and attention DP with expert parallelism (dep4).
MEMBERS = {
    "tp4": dict(tp_size=4, moe_tp_size=4, moe_ep_size=1, enable_attention_dp=False),
    "tep4": dict(tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=False),
    "dep4": dict(tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=True),
}
RANKS = (0, 1, 2, 3)
SHAPE_RANKS = (0, 3)

_STORED_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "U8": torch.uint8,
    "F8_E4M3": torch.float8_e4m3fn,
    "I64": torch.int64,
}

#: A routed or draft expert's tensors, as opposed to its per-tensor scalars (which every
#: rank reads, for the whole routing space).
_EXPERT_TENSOR = re.compile(
    r"\.mlp\.experts\.\d+\.\w+_proj\.(weight|weight_scale|weight_scale_inv)$"
)
_TRUNK_EXPERT_TENSOR = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.\w+_proj\.(weight|weight_scale)$"
)


# ------------------------------------------------------------------ fixtures --


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    root = llm_models_root()
    path = None if root is None else root / CHECKPOINT
    if path is None or not (path / "model.safetensors.index.json").is_file():
        pytest.skip(f"{CHECKPOINT} is not under LLM_MODELS_ROOT")
    return path


@pytest.fixture(scope="module")
def stored(checkpoint) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Every checkpoint tensor's stored shape and dtype, from the index and the headers."""
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
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


# ------------------------------------------------------------------- helpers --


def _mapping(member: str, rank: int) -> Mapping:
    return Mapping(world_size=4, rank=rank, gpus_per_node=4, pp_size=1, **MEMBERS[member])


def _dims(configs, member: str, rank: int, mtp: bool = True, prefill_copy: bool = False):
    text, vision = configs
    return WT.rank_dims(
        text,
        vision,
        _mapping(member, rank),
        mtp_enabled=mtp,
        trunk_moe_backend="TRTLLM",
        moe_prefill_copy=prefill_copy,
        dtype=torch.bfloat16,
    )


def _attention_tp(member: str) -> int:
    return 1 if MEMBERS[member]["enable_attention_dp"] else 4


def _lm_head_rows(member: str, vocab: int) -> int:
    """The rows of the lm_head the shell builds: vocab-parallel under attention TP, whole
    under attention DP (`DecoderModelForCausalLM`)."""
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


def _claimed(manifest) -> set:
    return {
        key
        for sources in manifest.values()
        for ckpt_key, _, cut, _ in sources
        for key, _ in _keys_and_cuts(ckpt_key, cut)
    }


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
    """A checkpoint tensor known by its header: indexing it returns a meta tensor of the
    indexed shape, and nothing is ever read."""

    def __init__(self, shape, dtype):
        self.shape, self.dtype = torch.Size(shape), dtype
        self.ndim = len(shape)

    def __getitem__(self, index):
        return torch.empty(self.shape, dtype=self.dtype, device="meta")[index]


def _share(member: str, key: str) -> float:
    """The fraction of `key` a rank of `member` holds, worked out here from the member alone:
    attention q/o by q head and k/v by KV head (two heads over the attention TP ranks), the
    Gated-DeltaNet tensors by head over the same ranks, the shell's lm_head by vocabulary.
    Everything else is read whole."""
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


# ----------------------------------------------------------------- coverage --


@pytest.mark.parametrize("mtp", [False, True], ids=["no-mtp", "mtp"])
@pytest.mark.parametrize("member", list(MEMBERS))
def test_every_checkpoint_tensor_is_claimed(stored, configs, member, mtp):
    everything = set(stored)
    # tp4's routed experts are split by intermediate, which this table does not declare.
    gap = {k for k in everything if _TRUNK_EXPERT_TENSOR.match(k)} if member == "tp4" else set()
    read_by = collections.Counter()
    for rank in RANKS:
        d = _dims(configs, member, rank, mtp)
        claimed = _claimed(_rank_manifest(stored, d, member))
        unread = WT.MODEL_WEIGHTS.expected_unconsumed(d, stored.keys())
        assert claimed <= everything, f"rank {rank} reads keys the checkpoint lacks: " + str(
            sorted(claimed - everything)[:8]
        )
        assert not claimed & unread, f"rank {rank}: read and predicted unread: " + str(
            sorted(claimed & unread)[:8]
        )
        assert claimed | unread == everything - gap, (
            f"rank {rank}: neither read nor predicted {sorted(everything - gap - claimed - unread)[:8]}"
            f"; outside the checkpoint or in the tp4 gap {sorted((claimed | unread) - (everything - gap))[:8]}"
        )
        read_by.update(claimed)

    mtp_keys = set() if mtp else {k for k in everything if k.startswith("mtp.")}
    assert set(read_by) == everything - gap - mtp_keys, "tensors no rank reads: " + str(
        sorted(everything - gap - mtp_keys - set(read_by))[:8]
    )
    twice = sorted(k for k in read_by if _EXPERT_TENSOR.search(k) and read_by[k] != 1)
    assert not twice, f"expert tensors not read by exactly one rank: {twice[:8]}"


# -------------------------------------------------------------------- shapes --


@pytest.mark.parametrize("prefill_copy", [False, True], ids=["trtllm-gen", "with-cutedsl-copy"])
@pytest.mark.parametrize("rank", SHAPE_RANKS)
@pytest.mark.parametrize("member", list(MEMBERS))
def test_rank_operands_have_their_parameter_shapes(stored, configs, member, rank, prefill_copy):
    """Every source, read through its cut and run through its transform on meta tensors,
    fills its parameter exactly; the rank reads each tensor at its share, and once per
    layout it is kept in (the routed experts twice with the CuteDSL prefill copy)."""
    d = _dims(configs, member, rank, mtp=True, prefill_copy=prefill_copy)
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
            # Sources that agree on the stored tensors, their cuts and the transform agree
            # on the result: run each distinct one once.
            signature = (transform, tuple((stored[k], c) for k, c in key_cuts))
            if signature not in results:
                src = WT.MODEL_WEIGHTS.read(entries, ckpt_key, cut, transform, d)
                results[signature] = (tuple(src.shape), src.dtype)
            assert results[signature] == (tuple(dst.shape), dst.dtype), (
                param_key,
                ckpt_key,
                results[signature],
                (tuple(dst.shape), dst.dtype),
            )

    def copies(key):
        return 2 if prefill_copy and _TRUNK_EXPERT_TENSOR.match(key) else 1

    wrong = {
        key: (n, _share(member, key), copies(key))
        for key, n in read.items()
        if n != round(torch.Size(stored[key][0]).numel() * _share(member, key)) * copies(key)
    }
    assert not wrong, f"tensors not read at the rank's share, once: {sorted(wrong.items())[:8]}"


#: Per-rank shapes of the attention and Gated-DeltaNet weights the forward is written
#: against: (under attention TP, under attention DP).
_SPLIT_SHAPES = {
    # [q 6x256 | k 256 | v 256 | gate 6x256 | index 4x128 + 128 | pad 128]
    "l3_attn_in_w": ((4352, 2560), (14080, 2560)),
    "l3_o_w": ((2560, 1536), (2560, 6144)),
    # [q 4x128 | k 4x128 | v 12x128 | z 12x128 | a 12 | b 12]
    "l0_in_proj_w": ((4120, 2560), (16480, 2560)),
    "l0_conv_w": ((2560, 1, 4), (10240, 1, 4)),
    "l0_A_log": ((12,), (48,)),
    "l0_dt_bias": ((12,), (48,)),
    "l0_out_w": ((2560, 1536), (2560, 6144)),
    "l48_mtp_q_w": ((3072, 2560), (12288, 2560)),
    "l48_mtp_k_w": ((256, 2560), (512, 2560)),
    "l48_mtp_v_w": ((256, 2560), (512, 2560)),
    "l48_mtp_o_w": ((2560, 1536), (2560, 6144)),
}

#: The same at every member and rank: replicated weights and the draft layer's 128-expert
#: FP8 window.
_COMMON_SHAPES = {
    "l3_idx_qn": (128,),
    "l0_gnorm": (128,),
    "l0_router": (512, 2560),
    "l0_sh_gu": (2 * 640 + 8, 2560),
    "l0_sh_dn": (2560, 640),
    "l0_e_isc1": (512,),
    "l0_ahc_down": (320, 10240),
    "fhc_up": (10240, 320),
    "embed": (248320, 2560),
    "ple_key_w": (10240, 2560),
    "l48_mtp_idx_w": (640, 2560),
    "l48_mtp_fc_e": (2560, 2560),
    "l48_mtp_fc1_w": (128, 1280, 2560),
    "l48_mtp_fc1_s": (128, 10, 20),
    "l48_mtp_fc2_w": (128, 2560, 640),
    "l48_mtp_fc2_s": (128, 20, 5),
}

#: The routed experts' 128-expert NVFP4 window (tep4, dep4).
_EXPERT_WINDOW_SHAPES = {
    "l0_fc1_w": (128, 1280, 1280),
    "l0_fc1_s": (128, 1280, 160),
    "l0_fc2_w": (128, 2560, 320),
    "l0_fc2_s": (128, 2560, 40),
}


@pytest.mark.parametrize("rank", SHAPE_RANKS)
@pytest.mark.parametrize("member", list(MEMBERS))
def test_rank_shapes_are_what_the_modules_expect(stored, configs, member, rank):
    d = _dims(configs, member, rank)
    with torch.device("meta"):
        params = WT.MODEL_WEIGHTS.declare(d)
    dp = MEMBERS[member]["enable_attention_dp"]
    for key, shapes in _SPLIT_SHAPES.items():
        assert tuple(params[key].shape) == shapes[dp], (key, tuple(params[key].shape))
    for key, shape in _COMMON_SHAPES.items():
        assert tuple(params[key].shape) == shape, (key, tuple(params[key].shape))
    if member == "tp4":
        assert not any(re.match(r"l\d+_fc[12]_[ws]", k) for k in params), "tp4 experts declared"
    else:
        for key, shape in _EXPERT_WINDOW_SHAPES.items():
            assert tuple(params[key].shape) == shape, (key, tuple(params[key].shape))
        experts = [s[0] for s in WT.MODEL_WEIGHTS.manifest(d)["l0_fc2_w"]]
        assert experts == [
            f"model.language_model.layers.0.mlp.experts.{e}.down_proj.weight"
            for e in range(128 * rank, 128 * (rank + 1))
        ]
    drafts = [s[0] for s in WT.MODEL_WEIGHTS.manifest(d)["l48_mtp_fc2_w"]]
    assert drafts == [
        f"mtp.layers.0.mlp.experts.{e}.down_proj.weight"
        for e in range(128 * rank, 128 * (rank + 1))
    ]

    # The heads this rank's weights hold are the heads its pools hold.
    g = P.pool_geometry(
        configs[0],
        _mapping(member, rank),
        tokens_per_block=32,
        kv_cache_quant_algo=None,
        dtype=torch.bfloat16,
    )
    assert (d.heads_kv, d.kv_head_start) == (g.heads_kv, g.kv_head_start)
    assert (d.num_k_heads, d.k_head_start) == (g.num_k_heads, g.k_head_start)
    assert (d.num_v_heads, d.v_head_start) == (g.num_v_heads, g.v_head_start)
    assert params["l0_conv_w"].shape[0] == g.conv_dim
    assert params["l0_A_log"].shape[0] == g.num_v_heads


def test_lm_head_reads_the_rows_the_shell_built():
    vocab = 248320
    for rank in RANKS:
        cut, _ = WT.lm_head_share(vocab // 4, vocab, rank)
        assert cut == (slice(rank * 62080, (rank + 1) * 62080),)
        cut, _ = WT.lm_head_share(vocab, vocab, rank)
        assert cut == (slice(0, vocab),)


# ---------------------------------------------------------------------- load --


def _stub_model(configs, member: str, rank: int, mtp: bool):
    """The shell and core `load` reads, on the meta device: the core's configs, mapping and
    engine choices, its declared parameters, and the shell's lm_head."""
    text, vision = configs
    mapping = _mapping(member, rank)
    core = SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=text,
            mapping=mapping,
            spec_config=object() if mtp else None,
            torch_dtype=torch.bfloat16,
        ),
        vision_config=vision,
        trunk_moe_backend="TRTLLM",
        moe_prefill_copy=False,
        ple_host_offload=False,
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


def _meta_checkpoint(stored) -> dict:
    """Every checkpoint tensor as its header: 0-dim scalars as meta tensors, the rest as
    entries that index into meta tensors."""
    return {
        key: torch.empty((), dtype=dtype, device="meta")
        if not shape
        else _HeaderEntry(shape, dtype)
        for key, (shape, dtype) in stored.items()
    }


@pytest.mark.parametrize(
    "member, rank, mtp", [("tep4", 3, True), ("dep4", 0, True), ("tep4", 1, False)]
)
def test_load_fills_every_parameter_and_covers_the_checkpoint(
    stored, configs, member, rank, mtp, monkeypatch
):
    """`load` end to end at the real shapes, on meta tensors: the n-gram table declared from
    its shards, every manifest row filled through its cut, the shell's lm_head rows, and
    both coverage asserts."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    model = _stub_model(configs, member, rank, mtp)
    WT.MODEL_WEIGHTS.load(model, _meta_checkpoint(stored))
    assert tuple(model.model.w["ple_table"].shape) == (320_001_536, 160)


def test_load_refuses_the_intermediate_split(stored, configs):
    model = _stub_model(configs, "tp4", 0, mtp=False)
    with pytest.raises(AssertionError, match="intermediate split"):
        WT.MODEL_WEIGHTS.load(model, _meta_checkpoint(stored))


# --------------------------------------------------- the cuts pick the heads --

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
    """Attention layer 3, Gated-DeltaNet layer 0 and the MTP attention, at their stored row
    counts and a two-wide input axis."""
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


def _block(start: int, width: int) -> list[int]:
    return list(range(start, start + width))


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("member", ["tep4", "dep4"])
def test_cuts_select_the_rank_heads(stored, configs, member, rank):
    """tp4 cuts attention and Gated-DeltaNet exactly as tep4 does: the members differ only
    in the routed experts."""
    d = _dims(configs, member, rank)
    fake = _fake_checkpoint(stored)
    manifest = WT.MODEL_WEIGHTS.manifest(d)
    tp = _attention_tp(member)
    # Under attention DP each rank holds every head: the split is over one rank.
    r = rank if tp > 1 else 0
    q_heads = _block(r * 24 // tp, 24 // tp)
    kv_heads = [r * 2 // tp] if tp > 1 else [0, 1]
    k_heads = _block(r * 16 // tp, 16 // tp)
    v_heads = _block(r * 48 // tp, 48 // tp)

    def rows(heads, width, base=0):
        return [base + h * width + j for h in heads for j in range(width)]

    q_rows = [h * 512 + j for h in q_heads for j in range(256)]
    gate_rows = [h * 512 + 256 + j for h in q_heads for j in range(256)]
    attn_in = _assemble(d, fake, manifest["l3_attn_in_w"])
    expected = torch.cat(
        (
            _expected_rows("q", q_rows),
            _expected_rows("k", rows(kv_heads, 256)),
            _expected_rows("v", rows(kv_heads, 256)),
            _expected_rows("q", gate_rows),
            _expected_rows("idx", range(640)),
            torch.zeros(128, dtype=torch.int64),
        )
    )
    assert torch.equal(attn_in[:, 0], expected) and torch.equal(attn_in[:, 1], expected)

    o_cols = _expected_rows("o", rows(q_heads, 256))
    for key in ("l3_o_w", "l48_mtp_o_w"):
        assert torch.equal(_assemble(d, fake, manifest[key])[0], o_cols), key
    assert torch.equal(_assemble(d, fake, manifest["l48_mtp_q_w"])[:, 0],
                       _expected_rows("q", rows(q_heads, 512)))  # fmt: skip
    assert torch.equal(_assemble(d, fake, manifest["l48_mtp_k_w"])[:, 0],
                       _expected_rows("k", rows(kv_heads, 256)))  # fmt: skip

    def qkv(tag):
        return torch.cat(
            (
                _expected_rows(tag, rows(k_heads, 128)),
                _expected_rows(tag, rows(k_heads, 128, 2048)),
                _expected_rows(tag, rows(v_heads, 128, 4096)),
            )
        )

    in_proj = torch.cat(
        (
            qkv("qkv"),
            _expected_rows("z", rows(v_heads, 128)),
            _expected_rows("a", v_heads),
            _expected_rows("b", v_heads),
        )
    )
    assert torch.equal(_assemble(d, fake, manifest["l0_in_proj_w"])[:, 0], in_proj)
    assert torch.equal(_assemble(d, fake, manifest["l0_conv_w"])[:, 0, 0], qkv("conv"))
    assert torch.equal(_assemble(d, fake, manifest["l0_A_log"]), _expected_rows("a_log", v_heads))
    assert torch.equal(_assemble(d, fake, manifest["l0_dt_bias"]), _expected_rows("dtb", v_heads))
    assert torch.equal(_assemble(d, fake, manifest["l0_out_w"])[0],
                       _expected_rows("out", rows(v_heads, 128)))  # fmt: skip
