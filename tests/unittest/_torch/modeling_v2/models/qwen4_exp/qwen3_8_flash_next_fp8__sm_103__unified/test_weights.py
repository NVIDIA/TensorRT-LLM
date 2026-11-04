# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The unified target's per-rank weight table, checked against the FP8 checkpoint.

On the CPU, from `model.safetensors.index.json` and the safetensors headers alone -- no
tensor data is read -- at the members' ranks, with MTP (the checkpoint's every tensor in play)
and, where the draft layer's tensors must go unread, without:

* **coverage**: every checkpoint tensor is either read by the rank's table or predicted
  unread by `expected_unconsumed`, never both; across a member's ranks each routed or draft
  expert's tensors are read by exactly one rank and every other tensor by every rank;
* **shapes**: every source read through its cut and run through the table's own transform
  (on meta tensors) has exactly the shape and dtype of the parameter it fills, and the rank
  reads each tensor once, at its share; the per-rank widths the forward is written against;
* **load** end to end on meta tensors, both coverage asserts included.

tp2 / tep2 and tp4 / tep4 differ only in the `Mapping`'s MoE split, which no cut reads (every
member holds whole experts, `test_every_member_holds_whole_expert_windows`), and an attention-DP
rank cuts nothing, as tp1's: the per-rank checks run every rank of tp1, tp2 and tp4 and one rank
of dep2 and dep4 (`HEAD_RANKS`); coverage, which the MoE split could break, runs every member.

On small synthetic tensors whose values name their rows: each cut selects the rank's own
heads in the order the fused operands lay them out (the KV heads as `duplicate_kv_weight`
repeats them), and every member -- tp2 and tp4, whose `Mapping` puts the MoE in tensor
parallelism, included -- holds whole experts, their operands the checkpoint's bytes in the
runner's `[up ; gate]` order with the block scales widened to fp32.

On one GPU: one trunk layer's tep2 rank-0 window (experts 0..255) loaded from the checkpoint
through the table runs on the family's `fp8_block_scale_moe_runner` and matches the runner's
reference over the dequantized checkpoint experts; and the n-gram table's host offload.
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

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified import (  # noqa: E501
    weights as WT,
)
from tensorrt_llm._torch.configs import Qwen4ExpTextConfig, Qwen4ExpVisionConfig
from tensorrt_llm._torch.models.modeling_utils import duplicate_kv_weight
from tensorrt_llm.mapping import Mapping

CHECKPOINT = "Qwen3.8-Flash-Next-FP8"

#: Each member's mapping as the LLM API's parallel arguments build it. tp2 and tp4 leave the
#: MoE split to `Mapping`'s default, tensor parallelism over the TP size; tep and dep put
#: the experts in expert parallelism; dep replicates attention.
MEMBERS = {
    "tp1": dict(world_size=1, tp_size=1),
    "tp2": dict(world_size=2, tp_size=2),
    "tep2": dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2),
    "dep2": dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2, enable_attention_dp=True),
    "tp4": dict(world_size=4, tp_size=4),
    "tep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4),
    "dep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=True),
}
#: The ranks that cut different heads: every rank of tp1, tp2 and tp4, and one rank of each
#: attention-DP member.
HEAD_RANKS = [(m, r) for m in ("tp1", "tp2", "tp4") for r in range(MEMBERS[m]["world_size"])] + [
    ("dep2", 1),
    ("dep4", 3),
]

NUM_EXPERTS = 512
NUM_LAYERS = 48
MTP_LAYER = NUM_LAYERS

_STORED_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "I64": torch.int64,
}

#: A routed (trunk) or draft (MTP) expert's tensors.
_EXPERT_TENSOR = re.compile(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.(weight|weight_scale_inv)$")

_needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


# ------------------------------------------------------------------ fixtures --


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


# ------------------------------------------------------------------- helpers --


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
    """The rows of the lm_head the shell builds: vocab-parallel under attention TP, whole
    under attention DP and at one rank (`DecoderModelForCausalLM`)."""
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
    """The fraction of `key` a rank of `member` reads, worked out here from the member alone:
    attention q/o by q head and k/v by KV head (two heads over the attention TP ranks, each
    repeated at four), the Gated-DeltaNet tensors by head over the same ranks, the shell's
    lm_head by vocabulary. Everything else -- a routed expert's tensors included, at every
    member -- is read whole."""
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
    """The experts rank `rank` of `member` holds, from the member alone: `512 / world`
    consecutive whole experts, whatever the Mapping's MoE split."""
    count = NUM_EXPERTS // _world(member)
    return range(count * rank, count * (rank + 1))


# --------------------------------------------------------------- checkpoint --


@pytest.mark.cpu_only
def test_only_the_routed_experts_are_block_fp8(stored):
    """What the table is written against, read off the headers: every trunk and draft expert
    projection is e4m3 with bf16 128x128 block scales, the n-gram table e4m3 with its
    per-tensor scale, the id-space tensors int64, everything else bf16; no per-tensor
    expert scalars and no final norm weight."""
    projection = re.compile(
        r"^(model\.language_model\.layers\.\d+|mtp\.layers\.0)\.mlp\.experts\.(\d+)\."
        r"(gate|up|down)_proj\.(weight|weight_scale_inv)$"
    )
    per_module = collections.Counter()
    for key, (shape, dtype) in stored.items():
        m = projection.match(key)
        if m:
            per_module[m.group(1)] += 1
            down = m.group(3) == "down"
            if m.group(4) == "weight":
                assert (shape, dtype) == (
                    (2560, 640) if down else (640, 2560),
                    torch.float8_e4m3fn,
                ), key
            else:
                assert (shape, dtype) == ((20, 5) if down else (5, 20), torch.bfloat16), key
        elif ".ngram_embedding.shard_" in key:
            assert dtype == torch.float8_e4m3fn and shape[1] == 160, (key, shape, dtype)
        elif key.endswith(("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")):
            assert dtype == torch.int64, (key, dtype)
        else:
            assert dtype == torch.bfloat16, (key, dtype)
    assert set(per_module.values()) == {NUM_EXPERTS * 6}
    assert len(per_module) == NUM_LAYERS + 1
    assert "model.language_model.norm.weight" not in stored
    assert not [k for k in stored if k.endswith(("input_scale", "weight_scale_2"))]


# ----------------------------------------------------------------- coverage --


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "member, mtp",
    [(member, True) for member in MEMBERS] + [("tep4", False)],
    ids=[f"{member}-mtp" for member in MEMBERS] + ["tep4-no-mtp"],
)
def test_every_checkpoint_tensor_is_claimed(stored, configs, member, mtp):
    """Every member with the draft layer's tensors read, and without MTP once: the `mtp.*`
    tensors are then predicted unread, whatever the member."""
    everything = set(stored)
    read_by = collections.Counter()
    for rank in range(_world(member)):
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
    # Each expert's tensors by exactly one rank; everything else by every rank (at its share,
    # which the shape test checks).
    wrong = sorted(
        k for k, n in read_by.items() if n != (1 if _EXPERT_TENSOR.search(k) else _world(member))
    )
    assert not wrong, f"tensors not read by their expected rank count: {wrong[:8]}"


# -------------------------------------------------------------------- shapes --


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "member, rank, mtp", [(m, r, True) for m, r in HEAD_RANKS] + [("tep4", 3, False)]
)
def test_rank_operands_have_their_parameter_shapes(stored, configs, member, rank, mtp):
    """Every source, read through its cut and run through its transform on meta tensors,
    fills its parameter exactly; the rank reads each tensor at its share, once."""
    d = _dims(configs, member, rank, mtp)
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

    wrong = {
        key: (n, _share(member, key))
        for key, n in read.items()
        if n != round(torch.Size(stored[key][0]).numel() * _share(member, key))
    }
    assert not wrong, f"tensors not read at the rank's share, once: {sorted(wrong.items())[:8]}"


#: Per-rank shapes of the attention and Gated-DeltaNet weights the forward is written
#: against, by the attention TP width (1: tp1 and attention DP; 2: tp2, tep2; 4: tp4, tep4).
_SPLIT_SHAPES = {
    # [q 24/t x 256 | k | v | gate 24/t x 256 | index 4x128 + 128 | pad 128]
    "l3_attn_in_w": {1: (14080, 2560), 2: (7424, 2560), 4: (4352, 2560)},
    "l3_o_w": {1: (2560, 6144), 2: (2560, 3072), 4: (2560, 1536)},
    # [q 16/t x 128 | k 16/t x 128 | v 48/t x 128 | z 48/t x 128 | a 48/t | b 48/t]
    "l0_in_proj_w": {1: (16480, 2560), 2: (8240, 2560), 4: (4120, 2560)},
    "l0_conv_w": {1: (10240, 1, 4), 2: (5120, 1, 4), 4: (2560, 1, 4)},
    "l0_A_log": {1: (48,), 2: (24,), 4: (12,)},
    "l0_dt_bias": {1: (48,), 2: (24,), 4: (12,)},
    "l0_out_w": {1: (2560, 6144), 2: (2560, 3072), 4: (2560, 1536)},
    "l48_mtp_q_w": {1: (12288, 2560), 2: (6144, 2560), 4: (3072, 2560)},
    "l48_mtp_k_w": {1: (512, 2560), 2: (256, 2560), 4: (256, 2560)},
    "l48_mtp_v_w": {1: (512, 2560), 2: (256, 2560), 4: (256, 2560)},
    "l48_mtp_o_w": {1: (2560, 6144), 2: (2560, 3072), 4: (2560, 1536)},
}

#: The same at every member and rank: the replicated weights.
_COMMON_SHAPES = {
    "l3_idx_qn": (128,),
    "l0_gnorm": (128,),
    "l0_router": (512, 2560),
    "l0_sh_gu": (2 * 640 + 8, 2560),
    "l0_sh_dn": (2560, 640),
    "l0_ahc_down": (320, 10240),
    "fhc_up": (10240, 320),
    "embed": (248320, 2560),
    "ple_key_w": (10240, 2560),
    "ple_tscale": (1,),
    "l48_mtp_idx_w": (640, 2560),
    "l48_mtp_fc_e": (2560, 2560),
    "l48_mtp_router": (512, 2560),
}

#: One window's expert operands, `E = 512 / world` whole experts: (shape after E, dtype).
_EXPERT_SHAPES = {
    "fc1_w": ((1280, 2560), torch.float8_e4m3fn),
    "fc1_s": ((10, 20), torch.float32),
    "fc2_w": ((2560, 640), torch.float8_e4m3fn),
    "fc2_s": ((20, 5), torch.float32),
}


@pytest.mark.cpu_only
@pytest.mark.parametrize("member, rank", HEAD_RANKS)
def test_rank_shapes_are_what_the_modules_expect(configs, member, rank):
    d = _dims(configs, member, rank)
    with torch.device("meta"):
        params = WT.MODEL_WEIGHTS.declare(d)
    t = _attention_tp(member)
    for key, shapes in _SPLIT_SHAPES.items():
        assert tuple(params[key].shape) == shapes[t], (key, tuple(params[key].shape))
    for key, shape in _COMMON_SHAPES.items():
        assert tuple(params[key].shape) == shape, (key, tuple(params[key].shape))

    window = _window(member, rank)
    assert (d.expert_offset, d.local_experts) == (window.start, len(window))
    manifest = WT.MODEL_WEIGHTS.manifest(d)
    for prefix, module in (
        ("l0_", "model.language_model.layers.0.mlp"),
        ("l47_", "model.language_model.layers.47.mlp"),
        ("l48_mtp_", "mtp.layers.0.mlp"),
    ):
        for name, (shape, dtype) in _EXPERT_SHAPES.items():
            param = params[f"{prefix}{name}"]
            assert (tuple(param.shape), param.dtype) == ((len(window), *shape), dtype), (
                prefix + name,
                tuple(param.shape),
                param.dtype,
            )
        experts = [s[0] for s in manifest[f"{prefix}fc2_w"]]
        assert experts == [f"{module}.experts.{e}.down_proj.weight" for e in window]
        win = WT.expert_window(params, d, MTP_LAYER if prefix == "l48_mtp_" else int(prefix[1:-1]))
        assert (win.offset, win.count) == (window.start, len(window))
        assert win.fc1_w is params[f"{prefix}fc1_w"] and win.fc2_s is params[f"{prefix}fc2_s"]

    # The heads this rank's weights hold are the heads the stock manager's pools hold:
    # attention heads split over the attention TP ranks (each KV head repeated when the ranks
    # outnumber them), the recurrent heads over the same ranks.
    r = rank if t > 1 else 0
    assert (d.heads_q, d.q_head_start) == (24 // t, r * 24 // t)
    assert (d.heads_kv, d.kv_head_start) == (max(2 // t, 1), r * 2 // t)
    assert (d.num_k_heads, d.k_head_start) == (16 // t, r * 16 // t)
    assert (d.num_v_heads, d.v_head_start) == (48 // t, r * 48 // t)
    assert params["l0_conv_w"].shape[0] == 2 * d.key_dim + d.value_dim


@pytest.mark.cpu_only
def test_lm_head_reads_the_rows_the_shell_built():
    vocab = 248320
    for tp in (2, 4):
        for rank in range(tp):
            cut, _ = WT.lm_head_share(vocab // tp, vocab, rank)
            assert cut == (slice(rank * vocab // tp, (rank + 1) * vocab // tp),)
    cut, _ = WT.lm_head_share(vocab, vocab, 0)
    assert cut == (slice(0, vocab),)


# ---------------------------------------------------------------------- load --


def _stub_model(configs, member: str, rank: int, mtp: bool):
    """The shell and core `load` reads, on the meta device: the core's configs and mapping,
    its declared parameters, and the shell's lm_head."""
    text, vision = configs
    core = SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=text,
            mapping=_mapping(member, rank),
            spec_config=object() if mtp else None,
            torch_dtype=torch.bfloat16,
        ),
        vision_config=vision,
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
    """Every checkpoint tensor as its header, indexing into meta tensors."""
    return {key: _HeaderEntry(shape, dtype) for key, (shape, dtype) in stored.items()}


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "member, rank, mtp",
    [("tp1", 0, True), ("tp2", 1, True), ("tep4", 3, False), ("dep4", 0, True)],
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
    d = WT.MODEL_WEIGHTS.dims(model.model)
    assert tuple(model.model.w["ple_table"].shape) == (128 * _rows_per_shard(stored, d), 160)
    assert model.lm_head.weight.shape[0] == _lm_head_rows(member, 248320)


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


@pytest.mark.cpu_only
@pytest.mark.parametrize("member, rank", HEAD_RANKS)
def test_cuts_select_the_rank_heads(stored, configs, member, rank):
    """The members with one attention TP width cut attention and Gated-DeltaNet alike; they
    differ only in the routed experts, which no cut here touches."""
    d = _dims(configs, member, rank)
    fake = _fake_checkpoint(stored)
    manifest = WT.MODEL_WEIGHTS.manifest(d)
    tp = _attention_tp(member)
    # Under attention DP and at tp1 each rank holds every head: the split is over one rank.
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

    # The KV head a rank holds is the one the built-in path's `duplicate_kv_weight` gives
    # it: the stacked heads repeated, then split evenly over the ranks.
    k_key = "mtp.layers.0.self_attn.k_proj.weight"
    builtin_k = duplicate_kv_weight(fake[k_key], 2, tp).chunk(tp)[r] if tp > 1 else fake[k_key]
    assert torch.equal(_assemble(d, fake, manifest["l48_mtp_k_w"]), builtin_k)

    o_cols = _expected_rows("o", rows(q_heads, 256))
    for key in ("l3_o_w", "l48_mtp_o_w"):
        assert torch.equal(_assemble(d, fake, manifest[key])[0], o_cols), key
    assert torch.equal(_assemble(d, fake, manifest["l48_mtp_q_w"])[:, 0],
                       _expected_rows("q", rows(q_heads, 512)))  # fmt: skip
    assert torch.equal(_assemble(d, fake, manifest["l48_mtp_v_w"])[:, 0],
                       _expected_rows("v", rows(kv_heads, 256)))  # fmt: skip

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


# ------------------------------------------------ the experts: whole windows --


@pytest.mark.cpu_only
@pytest.mark.parametrize("member", list(MEMBERS))
def test_every_member_holds_whole_expert_windows(configs, member):
    """The ranks' windows tile the 512 experts, `512 / world` each, at every member -- tp2
    and tp4 included, whose Mapping splits the MoE by intermediate -- and no expert source
    carries a cut: the intermediate is never cut."""
    mapping = _mapping(member, 0)
    if member in ("tp2", "tp4"):
        assert (mapping.moe_tp_size, mapping.moe_ep_size) == (mapping.tp_size, 1)
    seen = []
    for rank in range(_world(member)):
        d = _dims(configs, member, rank)
        assert d.moe_inter == 640
        manifest = WT.MODEL_WEIGHTS.manifest(d)
        for prefix in ("l0_", "l48_mtp_"):
            for name in ("fc1_w", "fc1_s", "fc2_w", "fc2_s"):
                rows = manifest[f"{prefix}{name}"]
                assert [index for _, index, _, _ in rows] == [(i,) for i in range(len(rows))]
                assert all(cut is None for _, _, cut, _ in rows), (prefix, name)
        seen.extend(_window(member, rank))
        assert (d.expert_offset, d.local_experts) == (
            _window(member, rank).start,
            len(_window(member, rank)),
        )
    assert seen == list(range(NUM_EXPERTS))


def _random_bytes(gen, *shape) -> torch.Tensor:
    return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)


def _random_scales(gen, *shape) -> torch.Tensor:
    return (0.001 + torch.rand(shape, generator=gen)).to(torch.bfloat16)


@pytest.mark.cpu_only
@pytest.mark.parametrize("which", ["first", "last"])
def test_expert_operands_are_the_checkpoint_experts_whole(configs, which):
    """At a tp4 rank 1's window, an expert's operands are the checkpoint's bytes in the runner's
    layout: FC1 codes `[up ; gate]`, their block scales in the same order widened to fp32,
    FC2's codes as stored and its scales widened -- for a trunk layer and the MTP layer. Every
    member reads an expert the same way (its rows carry no cut); the member only says which
    experts the window holds."""
    member = "tp4"
    rank = 1
    d = _dims(configs, member, rank)
    window = _window(member, rank)
    expert = window[0] if which == "first" else window[-1]
    local = expert - window.start
    gen = torch.Generator().manual_seed(expert * 7 + rank)
    manifest = WT.MODEL_WEIGHTS.manifest(d)
    for prefix, module in (
        ("l5_", "model.language_model.layers.5.mlp"),
        ("l48_mtp_", "mtp.layers.0.mlp"),
    ):
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
        up_s, gate_s = (
            fake[f"{p}.up_proj.weight_scale_inv"],
            fake[f"{p}.gate_proj.weight_scale_inv"],
        )
        assert torch.equal(fc1_s, torch.cat((up_s, gate_s)).float())
        assert torch.equal(fc2_s, fake[f"{p}.down_proj.weight_scale_inv"].float())
        # The widening is exact: narrowing back gives the stored bf16.
        assert torch.equal(fc1_s.to(torch.bfloat16), torch.cat((up_s, gate_s)))


# ---------------------------------------------------------- PLE host offload --


@pytest.mark.cpu_only
def test_the_ple_host_offload_switch(monkeypatch):
    monkeypatch.delenv(WT.PLE_HOST_OFFLOAD_ENV, raising=False)
    assert WT.ple_host_offload_enabled()
    for value in ("1", "true", "on", "yes"):
        monkeypatch.setenv(WT.PLE_HOST_OFFLOAD_ENV, value)
        assert WT.ple_host_offload_enabled(), value
    for value in ("0", "false", "OFF", " no "):
        monkeypatch.setenv(WT.PLE_HOST_OFFLOAD_ENV, value)
        assert not WT.ple_host_offload_enabled(), value


@_needs_gpu
@pytest.mark.parametrize("offload", [True, False], ids=["host", "device"])
def test_the_ngram_table_lands_where_the_switch_says(configs, offload):
    """`_declare_ngram_table` and the table's manifest rows on small shards: host memory the
    device addresses under the offload, an ordinary device allocation without it, the
    shards concatenated in order either way."""
    d = _dims(configs, "tep2", 0, mtp=False)
    keys = WT.ngram_shard_keys(d)
    gen = torch.Generator().manual_seed(11)
    shards = {key: _random_bytes(gen, 3, d.ple_head_dim).view(WT.PLE_TABLE_DTYPE) for key in keys}
    device = torch.device("cuda")
    core = SimpleNamespace(w={"embed": torch.empty(1, device=device)}, ple_host_offload=offload)
    rows_per_shard = WT._declare_ngram_table(core, d, shards)
    table = core.w["ple_table"]
    assert rows_per_shard == 3 and tuple(table.shape) == (3 * len(keys), d.ple_head_dim)
    kind = WT.memory_kind(table.data_ptr())
    assert kind == (WT.CUDA_MEMORY_TYPE_HOST if offload else WT.CUDA_MEMORY_TYPE_DEVICE)
    WT.MODEL_WEIGHTS.fill(table, WT.ngram_table_sources(d, rows_per_shard), shards, d)
    torch.cuda.synchronize()
    expected = torch.cat([shards[k].view(torch.uint8) for k in keys]).to(device)
    assert torch.equal(table.view(torch.uint8), expected)


# ------------------------------------------- one window on the family runner --

#: A trunk MoE layer: a full-attention layer's (`i % 4 == 3`).
_LAYER = 3
_TOKENS = 128
_TOP_K = 10


def _lazy_checkpoint(checkpoint, index, keys) -> tuple[dict, list]:
    """`keys` as lazy safetensors slices, the form the engine's loader may hand `load`; and
    the open handles, which must outlive the slices."""
    import safetensors

    by_file = collections.defaultdict(list)
    for key in keys:
        by_file[index[key]].append(key)
    handles, weights = [], {}
    for file, file_keys in by_file.items():
        handle = safetensors.safe_open(str(checkpoint / file), framework="pt", device="cpu")
        handles.append(handle)
        weights.update({key: handle.get_slice(key) for key in file_keys})
    return weights, handles


def _ulp(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """(max, RMS) deviation in bf16 ulp of each reference row's max -- the statistics the
    runner's own test gates on (16 element-wise, 0.75 RMS)."""
    o, r = out.float(), ref.float()
    d = (o - r) / (r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) * 2.0**-8)
    return d.abs().max().item(), d.pow(2).mean().sqrt().item()


@_needs_gpu
def test_a_tep2_window_runs_on_the_family_runner(checkpoint, index, configs):
    """Layer 3's experts 0..255 -- tep2 rank 0's window -- filled from the checkpoint through
    the table (`fill`, the path `load` takes), handed to `fp8_block_scale_moe_runner` as the
    target hands them (`expert_window`), match the runner's reference computed over
    operands assembled here from the checkpoint tensors directly."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.fp8_block_scale_moe_runner import (  # noqa: E501
        fp8_block_scale_moe_runner as runner,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.quantization.fp8_quantize_1x128 import (  # noqa: E501
        fp8_quantize_1x128,
    )

    device = torch.device("cuda")
    d = _dims(configs, "tep2", 0, mtp=False)
    window = _window("tep2", 0)
    assert (d.expert_offset, d.local_experts) == (0, 256)
    manifest = WT.MODEL_WEIGHTS.manifest(d)
    names = [f"l{_LAYER}_{n}" for n in ("fc1_w", "fc1_s", "fc2_w", "fc2_s")]
    keys = sorted(
        {
            k
            for n in names
            for ckpt_key, *_ in manifest[n]
            for k, _ in _keys_and_cuts(ckpt_key, None)
        }
    )
    weights, handles = _lazy_checkpoint(checkpoint, index, keys)

    with torch.device("meta"):
        declared = WT.MODEL_WEIGHTS.declare(d)
    params, consumed = {}, set()
    for name in names:
        params[name] = torch.nn.Parameter(
            torch.empty_like(declared[name], device=device), requires_grad=False
        )
        consumed |= WT.MODEL_WEIGHTS.fill(params[name], manifest[name], weights, d)
    torch.cuda.synchronize()
    module = f"model.language_model.layers.{_LAYER}.mlp"
    assert consumed == {
        f"{module}.experts.{e}.{p}_proj.{s}"
        for e in window
        for p in ("gate", "up", "down")
        for s in ("weight", "weight_scale_inv")
    }
    win = WT.expert_window(params, d, _LAYER)

    # The runner's operands, assembled from the checkpoint tensors in the order its contract
    # states (FC1 rows `[up ; gate]`, fp32 block scales), independently of the table.
    ref_ops = [torch.empty_like(t) for t in (win.fc1_w, win.fc1_s, win.fc2_w, win.fc2_s)]
    for e in window:
        p, local = f"{module}.experts.{e}", e - window.start

        def raw(name):
            return weights[f"{p}.{name}"][:].to(device)

        ref_ops[0][local] = torch.cat((raw("up_proj.weight"), raw("gate_proj.weight")))
        ref_ops[1][local] = torch.cat(
            (raw("up_proj.weight_scale_inv"), raw("gate_proj.weight_scale_inv"))
        ).float()
        ref_ops[2][local] = raw("down_proj.weight")
        ref_ops[3][local] = raw("down_proj.weight_scale_inv").float()
    del handles
    for ours, theirs in zip((win.fc1_w, win.fc1_s, win.fc2_w, win.fc2_s), ref_ops):
        assert torch.equal(ours.view(torch.uint8), theirs.view(torch.uint8))

    # The activation as the target quantizes it, and a renormalized top-10 over all 512.
    gen = torch.Generator(device=device).manual_seed(3)
    x = torch.randn(_TOKENS, d.hidden, device=device, generator=gen).to(torch.bfloat16)
    xq, xsf = fp8_quantize_1x128(x, use_ue8m0=False)
    logits = torch.randn(_TOKENS, d.num_experts, device=device, generator=gen)
    top_w, top_ids = torch.topk(torch.sigmoid(logits), _TOP_K, dim=-1)
    top_w = top_w / top_w.sum(-1, keepdim=True)
    args = dict(
        routing_logits=None,
        routing_bias=None,
        hidden_states=xq,
        hidden_states_scale=xsf,
        num_experts=d.num_experts,
        top_k=_TOP_K,
        num_fused_shared_experts=None,
        n_group=None,
        topk_group=None,
        intermediate_size=d.moe_inter,
        local_expert_offset=win.offset,
        local_num_experts=win.count,
        routed_scaling_factor=None,
        routing_method_type=1,
        topk_weights=top_w.to(torch.bfloat16).contiguous(),
        topk_ids=top_ids.to(torch.int32).contiguous(),
        act_type=0,
        gemm1_clamp_limit=None,
    )
    runner.is_valid(
        gemm1_weights=win.fc1_w,
        gemm1_weights_scale=win.fc1_s,
        gemm2_weights=win.fc2_w,
        gemm2_weights_scale=win.fc2_s,
        **args,
    )
    out = runner(
        gemm1_weights=win.fc1_w,
        gemm1_weights_scale=win.fc1_s,
        gemm2_weights=win.fc2_w,
        gemm2_weights_scale=win.fc2_s,
        **args,
    )
    torch.cuda.synchronize()
    # The reference scopes its own fp32 matmul precision (`catalog._op.true_fp32_matmul`).
    ref = runner.reference(
        gemm1_weights=ref_ops[0],
        gemm1_weights_scale=ref_ops[1],
        gemm2_weights=ref_ops[2],
        gemm2_weights_scale=ref_ops[3],
        **args,
    )
    hit = (top_ids >= window.start) & (top_ids < window.stop)
    assert hit.any(-1).sum() > _TOKENS // 2, "too few rows route into the window to judge it"
    # A row with no slot in the window comes back zero.
    assert not out[~hit.any(-1)].any()
    elt, rms = _ulp(out, ref)
    print(f"  layer {_LAYER} tep2 window: {elt:.2f} ulp element-wise, {rms:.3f} ulp RMS")
    assert elt <= 16.0 and rms <= 0.75, (elt, rms)

    # Negative control: the same reference with FC1's halves swapped is far outside the gate,
    # so the gate tells a wrong `[up ; gate]` order apart on these weights.
    inter = d.moe_inter
    swapped = torch.cat((ref_ops[0][:, inter:], ref_ops[0][:, :inter]), 1)
    swapped_s = torch.cat((ref_ops[1][:, inter // 128 :], ref_ops[1][:, : inter // 128]), 1)
    wrong = runner.reference(
        gemm1_weights=swapped,
        gemm1_weights_scale=swapped_s,
        gemm2_weights=ref_ops[2],
        gemm2_weights_scale=ref_ops[3],
        **args,
    )
    _, rms_wrong = _ulp(out, wrong)
    print(f"  gate/up swapped: {rms_wrong:.3f} ulp RMS")
    assert rms_wrong > 0.75, rms_wrong
