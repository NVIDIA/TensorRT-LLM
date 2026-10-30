# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The unified target's per-rank weight table, checked against the checkpoint on the CPU.

From `model.safetensors.index.json` and the safetensors headers alone -- no tensor data is
read -- at each member (tp4, tep4, dep4; dep2, the disaggregated engines' layout, at the end):

* **coverage**: at every rank, with MTP off and on, every checkpoint tensor is either read by
  the rank's table or predicted unread by `expected_unconsumed`, never both; across a member's
  ranks every tensor is read by some rank, and each routed or draft expert's tensors by
  exactly one -- except tp4's routed experts, which every rank reads at its intermediate
  slice;
* **shapes**: at ranks 0 and 3, every source read through its cut lies inside the stored
  tensor, and the cut run through the table's own transform (on meta tensors) has exactly
  the shape and dtype of the parameter it fills. A rank reads a split tensor at its share
  only, and every other tensor whole, once;
* **the modules' shapes**: the per-rank widths the forward is written against, and the head
  ranges the same rank's K/V and recurrent pools hold (`pool.pool_geometry`).

And, on small synthetic tensors whose values name their rows, that each cut selects the
rank's own heads in the order the fused operands lay them out, and that tp4's routed
experts hold the rank's intermediate slice with zero padding at the end of each half.
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
    "tp4": dict(world_size=4, tp_size=4, moe_tp_size=4, moe_ep_size=1, enable_attention_dp=False),
    "tep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=False),
    "dep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=True),
}
#: Both engines of the disaggregated-serving test: attention DP over two ranks, the routed
#: experts in two 256-expert windows. It serves without speculation.
DEP2 = dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2, enable_attention_dp=True)
_LAYOUTS = {**MEMBERS, "dep2": DEP2}
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
    return Mapping(rank=rank, gpus_per_node=4, pp_size=1, **_LAYOUTS[member])


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
    layout = _LAYOUTS[member]
    return 1 if layout["enable_attention_dp"] else layout["tp_size"]


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
    Gated-DeltaNet tensors by head over the same ranks, the shell's lm_head by vocabulary,
    and at tp4 each routed expert's tensors by intermediate. Everything else is read whole."""
    tp = _attention_tp(member)
    if member == "tp4" and _TRUNK_EXPERT_TENSOR.match(key):
        return 1 / MEMBERS[member]["moe_tp_size"]
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
        assert claimed | unread == everything, (
            f"rank {rank}: neither read nor predicted {sorted(everything - claimed - unread)[:8]}"
        )
        read_by.update(claimed)

    mtp_keys = set() if mtp else {k for k in everything if k.startswith("mtp.")}
    assert set(read_by) == everything - mtp_keys, "tensors no rank reads: " + str(
        sorted(everything - mtp_keys - set(read_by))[:8]
    )

    def readers(key):
        # tp4: every rank reads every routed expert, at its slice (checked by the shape test).
        return len(RANKS) if member == "tp4" and _TRUNK_EXPERT_TENSOR.match(key) else 1

    wrong = sorted(k for k in read_by if _EXPERT_TENSOR.search(k) and read_by[k] != readers(k))
    assert not wrong, f"expert tensors not read by their expected rank count: {wrong[:8]}"


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

#: The routed experts a rank holds: a 128-expert window of whole experts (tep4, dep4), or
#: all 512 at a 160-row intermediate slice stored 256 wide (tp4).
_EXPERT_SHAPES = {
    "tep4": {
        "l0_fc1_w": (128, 1280, 1280),
        "l0_fc1_s": (128, 1280, 160),
        "l0_fc2_w": (128, 2560, 320),
        "l0_fc2_s": (128, 2560, 40),
    },
    "tp4": {
        "l0_fc1_w": (512, 512, 1280),
        "l0_fc1_s": (512, 512, 160),
        "l0_fc2_w": (512, 2560, 128),
        "l0_fc2_s": (512, 2560, 16),
    },
}
_EXPERT_SHAPES["dep4"] = _EXPERT_SHAPES["tep4"]


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
    for key, shape in _EXPERT_SHAPES[member].items():
        assert tuple(params[key].shape) == shape, (key, tuple(params[key].shape))
    window = range(512) if member == "tp4" else range(128 * rank, 128 * (rank + 1))
    experts = [s[0] for s in WT.MODEL_WEIGHTS.manifest(d)["l0_fc2_w"]]
    assert experts == [
        f"model.language_model.layers.0.mlp.experts.{e}.down_proj.weight" for e in window
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
    "member, rank, mtp",
    [("tep4", 3, True), ("dep4", 0, True), ("tep4", 1, False), ("tp4", 2, True)],
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


# ------------------------------------------ tp4: the padded intermediate slice --


def _blk32_gather(rows: int) -> torch.Tensor:
    """The 32-row block shuffle's gather index, written out here: inside each block of 32
    rows, source row `4u + v` lands at destination row `8v + u`."""
    j = torch.arange(32)
    idx = torch.empty(32, dtype=torch.long)
    idx[(j % 4) * 8 + j // 4] = j
    return idx.repeat(rows // 32) + torch.arange(rows // 32).repeat_interleave(32) * 32


def _interleave_gather(rows: int) -> torch.Tensor:
    """`[up | gate]` -> up0, gate0, up1, gate1, ..."""
    p = torch.empty(rows, dtype=torch.long)
    p[0::2] = torch.arange(rows // 2)
    p[1::2] = torch.arange(rows // 2, rows)
    return p


def _swizzle_flat(s: torch.Tensor) -> torch.Tensor:
    """The 128x4 block-scale swizzle of one `[M, C]` matrix, flat: scale `(m, c)` lands at
    `(m // 128) * 512 * (C // 4) + (c // 4) * 512 + (m % 32) * 16 + ((m % 128) // 32) * 4
    + c % 4`."""
    m, c = s.shape
    v = s.reshape(m // 128, 4, 32, c // 4, 4).permute(0, 3, 2, 1, 4)
    return v.reshape(-1).contiguous()


def _zero_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    return torch.cat((x, x.new_zeros(rows - x.shape[0], *x.shape[1:])), 0)


def _zero_cols(x: torch.Tensor, cols: int) -> torch.Tensor:
    return torch.cat((x, x.new_zeros(x.shape[0], cols - x.shape[1])), 1)


@pytest.mark.parametrize("expert", [0, 511])
@pytest.mark.parametrize("rank", RANKS)
def test_tp4_experts_hold_the_rank_slice_zero_padded_per_half(configs, rank, expert, monkeypatch):
    """At tp4 rank `r` reads rows `[160r, 160r + 160)` of an expert's `up_proj` and
    `gate_proj` (codes and block scales) and the same columns of its `down_proj`; each half is
    zero-extended to 256 rows (FC2 to 256 columns) *before* the `[up ; gate]` concatenation,
    then interleaved, block-shuffled and swizzled as at the whole width. The bytes are
    nonzero, so every zero in the operand is padding. The scale swizzle is the formula
    `fp4_block_scale_moe_runner__r4`'s test writes out, standing in for the native op on
    the CPU; the shrunk axes (FC1's input bytes, FC2's output rows) take no cut."""
    monkeypatch.setattr(WT, "_swizzle", _swizzle_flat)
    d = _dims(configs, "tp4", rank, mtp=False)
    assert (d.expert_inter_share, d.expert_inter_start, d.expert_inter) == (160, 160 * rank, 256)
    gen = torch.Generator().manual_seed(rank * 1000 + expert)

    def nonzero(*shape):
        return torch.randint(1, 256, shape, generator=gen, dtype=torch.uint8)

    p = f"model.language_model.layers.0.mlp.experts.{expert}"
    fake = {
        f"{p}.up_proj.weight": nonzero(640, 4),
        f"{p}.gate_proj.weight": nonzero(640, 4),
        f"{p}.up_proj.weight_scale": nonzero(640, 4).view(torch.float8_e4m3fn),
        f"{p}.gate_proj.weight_scale": nonzero(640, 4).view(torch.float8_e4m3fn),
        f"{p}.down_proj.weight": nonzero(128, 320),
        f"{p}.down_proj.weight_scale": nonzero(128, 40).view(torch.float8_e4m3fn),
    }
    manifest = WT.MODEL_WEIGHTS.manifest(d)

    def operand(param_key):
        (row,) = [s for s in manifest[param_key] if s[1] == (expert,)]
        ckpt_key, _, cut, transform = row
        return WT.MODEL_WEIGHTS.read(fake, ckpt_key, cut, transform, d).view(torch.uint8)

    rows = slice(160 * rank, 160 * rank + 160)

    def fc1(kind):
        up = fake[f"{p}.up_proj.{kind}"].view(torch.uint8)[rows]
        gate = fake[f"{p}.gate_proj.{kind}"].view(torch.uint8)[rows]
        both = torch.cat((_zero_rows(up, 256), _zero_rows(gate, 256)), 0)
        return both[_interleave_gather(512)[_blk32_gather(512)]]

    assert torch.equal(operand("l0_fc1_w"), fc1("weight"))
    assert torch.equal(operand("l0_fc1_s"), _swizzle_flat(fc1("weight_scale")).view(512, 4))

    down = fake[f"{p}.down_proj.weight"][:, 80 * rank : 80 * rank + 80]
    assert torch.equal(operand("l0_fc2_w"), _zero_cols(down, 128)[_blk32_gather(128)])
    down_s = fake[f"{p}.down_proj.weight_scale"].view(torch.uint8)[:, 10 * rank : 10 * rank + 10]
    expected = _swizzle_flat(_zero_cols(down_s, 16)[_blk32_gather(128)]).view(128, 16)
    assert torch.equal(operand("l0_fc2_s"), expected)

    # The padding is where the runner's `intermediate_size` = 256 puts it: the last 96 rows
    # of each half and the last 96 of FC2's 256 input columns, nothing else.
    w1 = operand("l0_fc1_w")[torch.argsort(_interleave_gather(512)[_blk32_gather(512)])]
    for half in (w1[:256], w1[256:]):
        assert bool((half[:160] != 0).all()) and not bool(half[160:].any())


@pytest.mark.parametrize("member", list(MEMBERS))
def test_the_stored_intermediate_follows_the_engines_storage_rule(configs, member):
    """The width a rank stores each expert's intermediate at is the one the engine's own
    NVFP4 trtllm-gen MoE stores that shard at, and the width the runner is certified at."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4TRTLLMGenFusedMoEMethod

    d = _dims(configs, member, 1)
    align, _ = NVFP4TRTLLMGenFusedMoEMethod.resolve_alignments(d.hidden, d.expert_inter_share)
    assert d.expert_inter == -(-d.expert_inter_share // align) * align
    assert (d.expert_inter_share, d.expert_inter) == ((160, 256) if member == "tp4" else (640, 640))


# ---------------------------------------------------------------------- dep2 --


def test_dep2_ranks_read_the_checkpoint_in_two_256_expert_windows(stored, configs):
    """dep2 reads what dep4 reads, except the routed experts: rank `r` reads the window
    `[256r, 256r + 256)`, so across the two ranks every tensor is read, each expert's by one
    rank, and the `mtp.*` subtree by none (no speculation at dep2)."""
    everything = set(stored)
    mtp_keys = {k for k in everything if k.startswith("mtp.")}
    read_by = collections.Counter()
    for rank in range(DEP2["world_size"]):
        d = _dims(configs, "dep2", rank, mtp=False)
        claimed = _claimed(_rank_manifest(stored, d, "dep2"))
        unread = WT.MODEL_WEIGHTS.expected_unconsumed(d, stored.keys())
        assert claimed <= everything and not claimed & unread, rank
        assert claimed | unread == everything, (
            f"rank {rank}: neither read nor predicted {sorted(everything - claimed - unread)[:8]}"
        )
        read_by.update(claimed)

        dep4 = _dims(configs, "dep4", rank, mtp=False)
        non_expert = {k for k in claimed if not _TRUNK_EXPERT_TENSOR.match(k)}
        assert non_expert == {
            k
            for k in _claimed(_rank_manifest(stored, dep4, "dep4"))
            if not _TRUNK_EXPERT_TENSOR.match(k)
        }, rank

        with torch.device("meta"):
            params = WT.MODEL_WEIGHTS.declare(d)
            dep4_params = WT.MODEL_WEIGHTS.declare(dep4)
        # Every parameter dep4 declares, at dep4's shape, but the expert window's: 256
        # experts where dep4 holds 128.
        assert set(params.keys()) == set(dep4_params.keys())
        for key in dep4_params:
            ours, theirs = tuple(params[key].shape), tuple(dep4_params[key].shape)
            if ours != theirs:
                assert (ours[0], theirs[0], ours[1:]) == (256, 128, theirs[1:]), (key, ours)
        for key, shape in _EXPERT_SHAPES["dep4"].items():
            assert tuple(params[key].shape) == (256, *shape[1:]), (key, tuple(params[key].shape))
        experts = [s[0] for s in WT.MODEL_WEIGHTS.manifest(d)["l0_fc2_w"]]
        assert experts == [
            f"model.language_model.layers.0.mlp.experts.{e}.down_proj.weight"
            for e in range(256 * rank, 256 * (rank + 1))
        ]
        g = P.pool_geometry(
            configs[0],
            _mapping("dep2", rank),
            tokens_per_block=32,
            kv_cache_quant_algo=None,
            dtype=torch.bfloat16,
        )
        assert (d.heads_kv, d.num_k_heads, d.num_v_heads) == (
            g.heads_kv,
            g.num_k_heads,
            g.num_v_heads,
        )

    assert set(read_by) == everything - mtp_keys
    wrong = sorted(k for k in read_by if _EXPERT_TENSOR.search(k) and read_by[k] != 1)
    assert not wrong, f"expert tensors not read by exactly one rank: {wrong[:8]}"


def test_dep2_load_fills_every_parameter_and_covers_the_checkpoint(stored, configs, monkeypatch):
    """`load` end to end at dep2's last rank, without speculation, on meta tensors."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    model = _stub_model(configs, "dep2", 1, False)
    WT.MODEL_WEIGHTS.load(model, _meta_checkpoint(stored))
    assert model.lm_head.weight.shape[0] == 248320
