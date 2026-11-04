# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Mistral Large 4 NVFP4 tep4 weight table against the checkpoint it reads, at tep4 (four ranks:
tp 4, moe_ep 4, moe_tp 1, one node) and at tep8 (eight ranks: tp 8, moe_ep 8, moe_tp 1, two
nodes of four).

No GPU. Three kinds of evidence, each at both world sizes:

* the table at the checkpoint's own language-model configuration (written out below, and
  compared with the configuration adapted from the checkpoint's `params.json` when it is
  present), on the ranks of the engine's mapping: which layers carry which weight at what
  per-rank shape, and that each rank's manifest reads only its own routed experts, 32 at tep4
  and 16 at tep8;
* the checkpoint itself, through its index and safetensors headers, as meta tensors, plus the
  values of its small scale tensors: the union of the ranks' sources is exactly the index's
  keys but the DSpark draft's, which every rank declares unconsumed; each rank consumes every
  one of the 1,721 language-model keys, the 276 vision-tower keys and its window's expert keys,
  27,264 and 146.42 GiB at tep4, 13,632 and 76.78 GiB at tep8, 3.17 GiB of it the vision tower
  every rank reads whole; the index the production `load` takes of every key on every rank --
  the ranks' head, row, column and vocabulary ranges and expert windows tile each sliced tensor
  exactly once, everything else is read whole; the production `load` -- every slice, transform,
  shape and dtype assert, the expert-scalar asserts on the checkpoint's own scalars and the
  two-way coverage assert -- passes on each rank; every FP8 block scale and every expert
  scalar satisfies the invariants `load` asserts; and the packed UE8M0 scale operand at each K
  a rank's slice gives (at tep8 `wo` at 1024, the dense `w2` at 3072, the shared expert's `w2`
  at 768: 6 blocks, a half-padding second word; at tep4 2048, 6144 and 1536, whole words)
  equals an independent byte-level packing and the engine's
  `transform_sf_into_required_layout`. Skipped where the checkpoint is absent (looked up as
  `mistral-large-4-nvfp4` or `Mistral-Large-4-1T-A52B-NVFP4` under `LLM_MODELS_ROOT`);
* a scaled-down configuration per world size with real CPU tensors, loaded on every rank and
  compared element for element with an independent construction of what each parameter must
  hold: the slices, the `wkv_b` regroup, both half orders, the packed scale operand padding
  included (at tep8 the dense `w2` at a 6-block K too), the expert window in the MoE runner's
  layout and its scalars, and the DSpark draft tensors left alone; plus what the streaming path
  reads off the shard files, and each assert's refusal.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from test_common.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tep4 import (  # noqa: E501
    weights as W,
)
from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.utils.fp8_utils import (
    check_sf_layout,
    inverse_transform_sf,
    transform_sf_into_required_layout,
)

pytestmark = pytest.mark.cpu_only

#: The world sizes the table serves: tep4 on one node, tep8 over two.
WORLDS = (4, 8)
#: GB300 nodes carry four devices: tep4 is one node, tep8 spans two.
GPUS_PER_NODE = 4
CHECKPOINTS = ("mistral-large-4-nvfp4", "Mistral-Large-4-1T-A52B-NVFP4")

_LAYERS = 72
#: The BF16 vision tower -- encoder, pre-projector norm, patch merger and language adapter --
#: which every rank reads whole: its keys and bytes in the checkpoint (3.17 GiB).
_VISION_KEYS, _VISION_BYTES = 276, 3_408_437_248
#: The DSpark draft's keys, all in the checkpoint's last shard, which no rank reads.
_DRAFT_KEYS = 38

#: The language-model configuration surface the table reads, at the checkpoint's values. The
#: checkpoint's configuration adds its vision encoder, whose rows are checked against the
#: checkpoint below.
_CONFIG = dict(
    num_hidden_layers=_LAYERS,
    hidden_size=6144,
    num_attention_heads=64,
    qk_nope_head_dim=64,
    qk_rope_head_dim=64,
    v_head_dim=128,
    kv_lora_rank=512,
    nope_layer_flags=[i % 4 == 3 for i in range(_LAYERS)],
    kv_source_layers=[35 if i >= 39 and i % 4 == 3 else i for i in range(_LAYERS)],
    index_sharing=["shared" if i >= 17 and (i - 17) % 4 < 2 else "full" for i in range(_LAYERS)],
    index_n_heads=32,
    index_head_dim=128,
    first_k_dense_replace=1,
    n_routed_experts=128,
    moe_intermediate_size=6144,
    n_shared_experts=1,
    intermediate_size=24576,
    vocab_size=204800,
)

_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "U8": torch.uint8,
}

_EXPERT_PARAMS = ("fc1_w", "fc1_s", "fc2_w", "fc2_s", "isc1", "isc2", "ws2_13", "ws2_2")
_EXPERT_SUFFIXES = {
    f"{proj}.{s}"
    for proj in ("w1", "w2", "w3")
    for s in ("weight", "weight_scale", "weight_scale_2", "input_scale")
}


@pytest.fixture(autouse=True)
def _no_device_memory():
    """Nothing here allocates on a GPU. The context alone proves nothing: `import tensorrt_llm`
    creates one wherever a device is visible."""
    yield
    if torch.cuda.is_initialized():
        assert torch.cuda.memory_reserved() == 0, "the weight table test allocated on a GPU"


def _mapping(world: int, rank: int) -> Mapping:
    m = Mapping(
        world_size=world,
        rank=rank,
        gpus_per_node=GPUS_PER_NODE,
        tp_size=world,
        moe_tp_size=1,
        moe_ep_size=world,
    )
    # A topology that ignores `rank=` would hand every rank rank 0's slices.
    assert (m.tp_rank, m.moe_ep_rank, m.moe_tp_rank) == (rank, rank, 0), m
    return m


def _core(config: dict, world: int, rank: int) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=SimpleNamespace(**config),
            mapping=_mapping(world, rank),
            torch_dtype=torch.bfloat16,
        )
    )


class _Core(nn.Module):
    """The surface `load` reads off the core: its config and its declared weights."""

    def __init__(self, config: dict, world: int, rank: int, device: str):
        super().__init__()
        self.model_config = _core(config, world, rank).model_config
        with torch.device(device):
            self.w = W.MODEL_WEIGHTS.declare(W.MODEL_WEIGHTS.dims(self))


class _Shell(nn.Module):
    """The causal-LM shell: the core as `model`, a vocab-parallel `lm_head`."""

    def __init__(self, config: dict, world: int, rank: int, device: str):
        super().__init__()
        self.model = _Core(config, world, rank, device)
        self.lm_head = nn.Module()
        rows = config["vocab_size"] // world
        self.lm_head.weight = nn.Parameter(
            torch.empty(rows, config["hidden_size"], dtype=torch.bfloat16, device=device),
            requires_grad=False,
        )


def _source_keys(manifest: dict) -> set:
    """Every checkpoint key the manifest reads, key tuples flattened."""
    return {k for sources in manifest.values() for ckpt, *_ in sources for k in W._keys(ckpt)}


def _is_vision(key: str) -> bool:
    return key.startswith(W.VISION_PREFIXES)


def _is_draft(key: str) -> bool:
    """A DSpark draft key: the speculative-decoding head, which these targets do not run."""
    return key.startswith("mtp_dspark.")


def _is_expert(key: str) -> bool:
    return ".experts." in key


def _is_expert_scalar(key: str) -> bool:
    return _is_expert(key) and key.endswith((".weight_scale_2", ".input_scale"))


def _role(key: str) -> str:
    """A parameter key's weight role: `l{i}_<role>` per layer, the key itself otherwise."""
    head, _, rest = key.partition("_")
    return rest if head[:1] == "l" and head[1:].isdigit() else key


def _expert_of(key: str) -> tuple[int, int, str]:
    """`layers.N.experts.E.<suffix>` -> (N, E, suffix)."""
    parts = key.split(".")
    assert parts[0] == "layers" and parts[2] == "experts", key
    return int(parts[1]), int(parts[3]), ".".join(parts[4:])


def _packed(grid: torch.Tensor) -> torch.Tensor:
    """Independent construction of the packed scale storage from fp32 `[N/128, K/128]` block
    scales: `[ceil(K/512), N]` int32, word `[w, n]` holding the UE8M0 bytes (the fp32 exponent
    field) of K blocks 4w..4w+3 of row n's block, low byte first, and zero bytes past the last K
    block. Integer arithmetic only, so it shares no code with the engine's packer."""
    n_blk, k_blk = grid.shape
    words = -(-k_blk // 4)
    exps = torch.zeros(n_blk, 4 * words, dtype=torch.int64)
    exps[:, :k_blk] = (grid.contiguous().view(torch.int32).to(torch.int64) >> 23) & 0xFF
    packed = exps[:, 0::4] | exps[:, 1::4] << 8 | exps[:, 2::4] << 16 | exps[:, 3::4] << 24
    packed = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)
    return packed.repeat_interleave(128, dim=0).t().contiguous()


# --------------------------------------------------------------------- the configuration


def _layers_of(declared, name: str) -> dict[int, tuple]:
    """Layer index -> declared shape, for every `l{i}_<name>`."""
    out = {}
    for key, param in declared.items():
        head, _, rest = key.partition("_")
        if head.startswith("l") and head[1:].isdigit() and rest == name:
            out[int(head[1:])] = tuple(param.shape)
    return out


#: The per-rank shapes, counts and bytes of the checkpoint's configuration that depend on the
#: world size. tep4: 16 of the 64 heads, the dense MLP at 6144 and the shared expert at 1536 of
#: their widths, 32 routed experts, an lm_head of 51200 rows. tep8: 8 heads, 3072, 768, 16
#: routed experts, 25600 rows. `gib` is a rank's language model, `read_gib` what it reads off
#: the checkpoint: that and the vision tower's 3.17 GiB.
_AT = {
    4: SimpleNamespace(
        heads=16,
        dense=6144,
        shared=1536,
        experts=32,
        lm_rows=51200,
        wq=(2048, 6144),
        wq_s=(12, 2048),
        ssmax=(16,),
        # 16 heads x [k_nope 128 | v 128] on NoPE layers, [k_nope 64 | v 128] on RoPE layers.
        wkv_b_nope=(4096, 512),
        wkv_b_rope=(3072, 512),
        wo=(6144, 2048),
        wo_s=(4, 6144),
        gu_dense=(12288, 6144),
        gu_shared=(3072, 6144),
        gu_s_dense=(12, 12288),
        gu_s_shared=(12, 3072),
        dn_dense=(6144, 6144),
        dn_shared=(6144, 1536),
        dn_s_dense=(12, 6144),
        dn_s_shared=(3, 6144),
        expert_keys=27264,
        gib=143.24,
        gib_hi=143.30,
        expert_gib=134.79,
        read_gib=146.42,
    ),
    8: SimpleNamespace(
        heads=8,
        dense=3072,
        shared=768,
        experts=16,
        lm_rows=25600,
        wq=(1024, 6144),
        wq_s=(12, 1024),
        ssmax=(8,),
        # 8 heads x [k_nope 128 | v 128] on NoPE layers, [k_nope 64 | v 128] on RoPE layers.
        wkv_b_nope=(2048, 512),
        wkv_b_rope=(1536, 512),
        wo=(6144, 1024),
        # K 1024: 8 blocks, two whole words.
        wo_s=(2, 6144),
        gu_dense=(6144, 6144),
        gu_shared=(1536, 6144),
        gu_s_dense=(12, 6144),
        gu_s_shared=(12, 1536),
        dn_dense=(6144, 3072),
        dn_shared=(6144, 768),
        # K 3072: 24 blocks, six words; K 768: 6 blocks, two words, the second half padding.
        dn_s_dense=(6, 6144),
        dn_s_shared=(2, 6144),
        expert_keys=13632,
        gib=73.61,
        gib_hi=73.66,
        expert_gib=67.39,
        read_gib=76.78,
    ),
}


@pytest.mark.parametrize("world, rank", [(w, r) for w in WORLDS for r in (0, w - 1)])
def test_layer_sets_and_per_rank_shapes(world, rank):
    """Which layers carry which weight, at what per-rank shape: at tep4 16 of the 64 heads
    (`wq` 2048 rows, `wo` 2048 columns, `wkv_b` 16 heads, SSMax 16), the dense MLP at 6144 and
    the shared expert at 1536 of their widths, 32 routed experts; at tep8 8 heads (1024, 1024,
    8, 8), 3072, 768 and 16 routed experts. The NoPE layers are 3, 7, ..., 71; the KV sharers
    39, 43, ..., 71; the 'shared' indexer layers 17, 18, 21, 22, ..., 69, 70."""
    at = _AT[world]
    core = _core(_CONFIG, world, rank)
    with torch.device("meta"):
        declared = W.MODEL_WEIGHTS.declare(W.MODEL_WEIGHTS.dims(core))
    nope = {i for i in range(_LAYERS) if i % 4 == 3}
    sharers = {i for i in range(39, _LAYERS, 4)}
    full = {i for i in range(_LAYERS) if i < 17 or (i - 17) % 4 >= 2}
    moe = range(1, _LAYERS)
    assert len(nope) == 18 and len(sharers) == 9 and len(full) == 44

    every = {i: None for i in range(_LAYERS)}
    expect = {
        "attn_norm": {i: (6144,) for i in every},
        "wq": {i: at.wq for i in every},
        "wq_s": {i: at.wq_s for i in every},
        "ssmax_s": {i: at.ssmax for i in every},
        "ssmax_b": {i: at.ssmax for i in every},
        "wkv_a": {i: (512 if i in nope else 576, 6144) for i in range(_LAYERS) if i not in sharers},
        "kv_a_norm": {i: (512,) for i in range(_LAYERS) if i not in sharers},
        "wkv_b": {i: at.wkv_b_nope if i in nope else at.wkv_b_rope for i in every},
        "wo": {i: at.wo for i in every},
        "wo_s": {i: at.wo_s for i in every},
        "ffn_norm": {i: (6144,) for i in every},
        "ix_q_norm_w": {i: (6144,) for i in full},
        "ix_q_norm_b": {i: (6144,) for i in full},
        "ix_wq_b": {i: (4096, 6144) for i in full},
        "ix_wq_b_s": {i: (12, 4096) for i in full},
        "ix_wk_wp": {i: (160, 6144) for i in full},
        "ix_k_norm_w": {i: (128,) for i in full},
        "ix_k_norm_b": {i: (128,) for i in full},
        "ix_ssmax_s": {i: (32,) for i in full},
        "ix_ssmax_b": {i: (32,) for i in full},
        "mlp_gu": {i: at.gu_dense if i == 0 else at.gu_shared for i in every},
        "mlp_gu_s": {i: at.gu_s_dense if i == 0 else at.gu_s_shared for i in every},
        "mlp_dn": {i: at.dn_dense if i == 0 else at.dn_shared for i in every},
        "mlp_dn_s": {i: at.dn_s_dense if i == 0 else at.dn_s_shared for i in every},
        "router": {i: (128, 6144) for i in moe},
        "fc1_w": {i: (at.experts, 12288, 3072) for i in moe},
        "fc1_s": {i: (at.experts, 12288, 384) for i in moe},
        "fc2_w": {i: (at.experts, 6144, 3072) for i in moe},
        "fc2_s": {i: (at.experts, 6144, 384) for i in moe},
        "isc1": {i: (1,) for i in moe},
        "isc2": {i: (1,) for i in moe},
        "ws2_13": {i: (at.experts,) for i in moe},
        "ws2_2": {i: (at.experts,) for i in moe},
    }
    for name, want in expect.items():
        assert _layers_of(declared, name) == want, name
    per_layer = {k for k in declared if k.startswith("l") and k.split("_")[0][1:].isdigit()}
    assert len(per_layer) == sum(len(v) for v in expect.values()), "an undeclared-for role"
    assert {k: tuple(v.shape) for k, v in declared.items() if k not in per_layer} == {
        "final_norm": (6144,),
        "embed": (204800, 6144),
        "lerope_alpha": (32,),
    }
    dtypes = {_role(k): v.dtype for k, v in declared.items()}
    assert dtypes["wq"] == dtypes["mlp_dn"] == dtypes["ix_wq_b"] == torch.float8_e4m3fn
    assert dtypes["wq_s"] == dtypes["mlp_gu_s"] == dtypes["ix_wq_b_s"] == torch.int32
    assert dtypes["fc1_w"] == dtypes["fc2_w"] == torch.uint8
    assert dtypes["fc1_s"] == dtypes["fc2_s"] == torch.float8_e4m3fn
    assert dtypes["isc1"] == dtypes["isc2"] == dtypes["ws2_13"] == dtypes["ws2_2"] == torch.float32
    assert dtypes["ix_k_norm_w"] == dtypes["lerope_alpha"] == torch.float32
    assert dtypes["wkv_b"] == dtypes["router"] == dtypes["embed"] == torch.bfloat16

    # Per rank, with the shell's lm_head (51200 rows at tep4, 25600 at tep8). The routed
    # experts are exactly their checkpoint bytes; the FP8 scale operands, one int32 word per
    # four 128-blocks per output row, are a few MiB more than the fp32 grids they are built from.
    nbytes = {k: p.numel() * p.element_size() for k, p in declared.items()}
    experts = sum(n for k, n in nbytes.items() if _role(k) in _EXPERT_PARAMS)
    total = sum(nbytes.values()) + at.lm_rows * 6144 * 2
    assert round(experts / 2**30, 2) == at.expert_gib, experts / 2**30
    assert at.gib * 2**30 < total < at.gib_hi * 2**30, total / 2**30


@pytest.mark.parametrize("world", WORLDS)
def test_each_rank_reads_only_its_expert_window(world):
    """Rank r's manifest names routed experts [n r, n r + n) and nothing else -- n is 32 at
    tep4, 16 at tep8 -- on every MoE layer, local slot s <- global expert n r + s, all twelve
    tensors of each; the one-per-layer scalars read the window's copies. The windows tile the
    128 experts. The known-not-loaded set is the other 96 experts at tep4, 112 at tep8: their
    scalars predicted, their codes and block scales only where found."""
    n = _AT[world].experts
    seen = []
    for rank in range(world):
        core = _core(_CONFIG, world, rank)
        manifest = W.MODEL_WEIGHTS.manifest(core)
        d = W.MODEL_WEIGHTS.dims(core)
        assert (d.local_experts, d.expert_offset) == (n, n * rank)
        lo, hi = n * rank, n * rank + n
        experts: dict = {}
        for key, sources in manifest.items():
            for ckpt, index, *_ in sources:
                keys = W._keys(ckpt)
                if not any(_is_expert(k) for k in keys):
                    continue
                assert all(_is_expert(k) for k in keys), (key, ckpt)
                param = _role(key)
                assert param in _EXPERT_PARAMS, key
                for k in keys:
                    layer, e, suffix = _expert_of(k)
                    assert lo <= e < hi, (rank, key, k)
                    assert key == f"l{layer}_{param}", (key, k)
                    if param in ("isc1", "isc2"):
                        assert index == (0,), (key, index)
                    else:
                        assert index[0] == e - lo, (key, k, index)
                    experts.setdefault((layer, e), set()).add(suffix)
        assert set(experts) == {(i, e) for i in range(1, _LAYERS) for e in range(lo, hi)}
        assert all(v == _EXPERT_SUFFIXES for v in experts.values())
        seen.extend(experts)

        predicted = W.MODEL_WEIGHTS.expected_unconsumed(core, {})
        assert len(predicted) == 71 * (128 - n) * 6
        assert all(_is_expert_scalar(k) for k in predicted)
        assert all(not lo <= _expert_of(k)[1] < hi for k in predicted)
    # Exactly once: no (layer, expert) in two windows, none in no window.
    assert sorted(seen) == [(i, e) for i in range(1, _LAYERS) for e in range(128)]


@pytest.mark.parametrize("world", WORLDS)
def test_manifest_and_declaration_agree(world):
    """A declared weight with no source allocates memory nothing fills."""
    for rank in range(world):
        core = _core(_CONFIG, world, rank)
        manifest = W.MODEL_WEIGHTS.manifest(core)
        with torch.device("meta"):
            declared = W.MODEL_WEIGHTS.declare(W.MODEL_WEIGHTS.dims(core))
        assert set(manifest) == set(declared)
        assert all(manifest.values())


# --------------------------------------------------------------------- the checkpoint


def _checkpoint_dir() -> Path | None:
    root = llm_models_root()
    if root is None:
        return None
    for name in CHECKPOINTS:
        path = root / name
        if (path / "params.json").is_file() and (path / W.SHARD_INDEX).is_file():
            return path
    return None


@pytest.fixture(scope="module")
def checkpoint():
    path = _checkpoint_dir()
    if path is None:
        pytest.skip(f"checkpoint {CHECKPOINTS} is not under LLM_MODELS_ROOT ({llm_models_root()})")
    index = json.loads((path / W.SHARD_INDEX).read_text())["weight_map"]
    headers = {}
    for name in sorted(set(index.values())):
        with open(path / name, "rb") as f:
            (size,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(size))
        header.pop("__metadata__", None)
        for key, entry in header.items():
            headers[key] = (_SAFETENSORS_DTYPES[entry["dtype"]], tuple(entry["shape"]))
    assert set(headers) == set(index), "index and safetensors headers name different tensors"
    assert len(set(index.values())) == 32
    # The configuration the engine adapts from the checkpoint's params.json, field by field.
    # The loads below read the dict they are handed, never the shard files that the directory
    # the loader records would point `load` at.
    config = dict(vars(MistralConfigLoader().load_pretrained_config(str(path))))
    config.pop("_name_or_path")
    return SimpleNamespace(path=path, index=index, headers=headers, config=config)


@pytest.fixture(scope="module")
def expert_scalars(checkpoint):
    """Every routed expert's per-tensor scalars, read off the shard files by the production
    reader."""
    d = W.MODEL_WEIGHTS.dims(_core(checkpoint.config, 4, 0))
    keys = W._expert_scalar_keys(d)
    return W._read_scalars(keys, checkpoint.index, (checkpoint.path, checkpoint.index))


def _meta(checkpoint) -> dict:
    return {
        key: torch.empty(shape, dtype=dtype, device="meta")
        for key, (dtype, shape) in checkpoint.headers.items()
    }


def test_checkpoint_config_is_the_one_tested(checkpoint):
    """The configuration above is the checkpoint's, on every key the table reads."""
    assert {k: checkpoint.config[k] for k in _CONFIG} == _CONFIG


#: The FP8 128x128-block linears (`FP8_PB_WO`): e4m3 codes, fp32 `[N/128, 1, K/128, 1]` scales.
_FP8_LINEAR = re.compile(
    r"layers\.\d+\.(attention\.(wq|wo|indexer\.wq_b)|feed_forward\.w[123]|shared_experts\.w[123])"
)


def test_checkpoint_dtypes_by_role(checkpoint):
    """What each key stores, which is what the table's rows read: the FP8 linears e4m3 with
    fp32 block scales, the routed experts NVFP4 (packed codes, e4m3 block scales, fp32
    per-tensor scalars), every other language-model tensor and the vision tower bf16."""
    for key, (dtype, shape) in checkpoint.headers.items():
        if _is_draft(key):
            continue
        module, _, leaf = key.rpartition(".")
        if _is_expert(key):
            want = {
                "weight": torch.uint8,
                "weight_scale": torch.float8_e4m3fn,
                "weight_scale_2": torch.float32,
                "input_scale": torch.float32,
            }[leaf]
            assert dtype == want, key
            assert (shape == ()) == _is_expert_scalar(key), (key, shape)
        elif _FP8_LINEAR.fullmatch(module):
            want = {"weight": torch.float8_e4m3fn, "weight_scale": torch.float32}[leaf]
            assert dtype == want, key
        else:
            assert dtype == torch.bfloat16, key


@pytest.mark.parametrize("world", WORLDS)
def test_union_over_ranks_is_every_key_but_the_draft(checkpoint, world):
    """Every key of the index but the DSpark draft's 38 is read by some rank (lm_head's
    `output.weight` by the shell); per rank, read and known-not-loaded split the index, the
    known-not-loaded the draft and the other ranks' experts, and a rank reads every one of the
    1,721 language-model keys, the 276 vision-tower keys and its window's expert keys: 27,264
    at tep4 (71 layers x 32 experts x 3 projections x 4 tensors), 13,632 at tep8
    (71 x 16 x 3 x 4)."""
    at = _AT[world]
    keys = set(checkpoint.index)
    draft = {k for k in keys if _is_draft(k)}
    vision = {k for k in keys if _is_vision(k)}
    lm = {k for k in keys - draft - vision if not _is_expert(k)}
    assert len(keys) == 111091 and len(draft) == _DRAFT_KEYS and len(vision) == _VISION_KEYS
    assert len(lm) == 1721
    union = set()
    for rank in range(world):
        core = _core(checkpoint.config, world, rank)
        read = _source_keys(W.MODEL_WEIGHTS.manifest(core)) | {"output.weight"}
        skipped = W.MODEL_WEIGHTS.expected_unconsumed(core, checkpoint.index)
        assert not read & set(skipped)
        assert read | set(skipped) == keys
        assert {k for k in read if not _is_expert(k)} == lm | vision
        assert len({k for k in read if _is_expert(k)}) == at.expert_keys == 71 * at.experts * 12
        assert {k for k in skipped if not _is_expert(k)} == draft
        assert {skipped[k] for k in draft} == {"DSpark draft disabled"}
        assert len(skipped) == _DRAFT_KEYS + 71 * (128 - at.experts) * 12
        union |= read
    assert union == keys - draft


@pytest.mark.parametrize("world", WORLDS)
def test_per_rank_bytes_from_the_headers(checkpoint, world):
    """What each rank moves, from the safetensors headers through the production slices: the
    same 146.42 GiB on every rank at tep4, 134.79 GiB of it its routed-expert window; 76.78 GiB
    at tep8, 67.39 GiB of it its window; at both, 3.17 GiB of it the vision tower, whole, and
    143.24 and 73.61 GiB the language model."""
    at = _AT[world]
    for rank in range(world):
        core = _core(checkpoint.config, world, rank)
        d = W.MODEL_WEIGHTS.dims(core)
        meta = {}

        def entry(key):
            if key not in meta:
                dtype, shape = checkpoint.headers[key]
                meta[key] = torch.empty(shape, dtype=dtype, device="meta")
            return meta[key]

        total = experts = vision = 0
        for sources in W.MODEL_WEIGHTS.manifest(core).values():
            for ckpt, _, _, shard in sources:
                for key in W._keys(ckpt):
                    t = W._materialize(entry(key), shard)
                    n = t.numel() * t.element_size()
                    total += n
                    experts += n if _is_expert(key) else 0
                    vision += n if _is_vision(key) else 0
        dtype, (vocab, hidden) = checkpoint.headers["output.weight"]
        total += vocab // d.tp_size * hidden * torch.empty((), dtype=dtype).element_size()
        assert vision == _VISION_BYTES, (rank, vision)
        assert round(total / 2**30, 2) == at.read_gib, (rank, total / 2**30)
        assert round((total - vision) / 2**30, 2) == at.gib, (rank, (total - vision) / 2**30)
        assert round(experts / 2**30, 2) == at.expert_gib, (rank, experts / 2**30)


@pytest.mark.parametrize("world, rank", [(w, r) for w in WORLDS for r in range(w)])
def test_load_on_meta_matches_the_headers(checkpoint, expert_scalars, world, rank):
    """The production `load` over meta tensors shaped like the safetensors headers, with the
    checkpoint's own expert scalars: the scalar asserts pass on every layer, every source is
    sliced and transformed to its declared per-rank shape and dtype, every declared parameter
    is filled, and every key is consumed or known-not-loaded."""
    weights = _meta(checkpoint)
    weights.update(expert_scalars)
    model = _Shell(checkpoint.config, world, rank, "meta")
    W.MODEL_WEIGHTS.load(model, weights)


def test_scale_invariants_hold_on_the_checkpoint(checkpoint, expert_scalars):
    """Every FP8 block scale goes through the production transform (squeezed from
    `[N/128, 1, K/128, 1]`, asserted UE8M0, packed), and every expert scalar passes the
    production check: per layer, w1 == w3 on both scalars of every expert, and one input_scale
    per projection over all 128 experts."""
    fp8 = sorted(
        k
        for k, (dtype, _) in checkpoint.headers.items()
        if k.endswith(".weight_scale") and dtype == torch.float32
    )
    assert len(fp8) == 404 == 72 + 72 + 44 + 71 * 3 + 3
    by_file: dict = {}
    for key in fp8:
        by_file.setdefault(checkpoint.index[key], []).append(key)
    for name, keys in sorted(by_file.items()):
        with safe_open(str(checkpoint.path / name), framework="pt", device="cpu") as handle:
            for key in keys:
                s = handle.get_tensor(key)
                n_blk, _, k_blk, _ = s.shape
                packed = W._block_sf(s, None)
                assert packed.shape == (W._sf_words(k_blk * 128), n_blk * 128), key

    assert len(expert_scalars) == 71 * 128 * 6
    for world in WORLDS:
        W._check_expert_scalars(
            expert_scalars, W.MODEL_WEIGHTS.dims(_core(checkpoint.config, world, 0))
        )
    for i in range(1, _LAYERS):
        isc1 = expert_scalars[f"layers.{i}.experts.0.w1.input_scale"].item()
        isc2 = expert_scalars[f"layers.{i}.experts.0.w2.input_scale"].item()
        assert 0 < isc1 < 1 and 0 < isc2 < 1, (i, isc1, isc2)


class _Cut:
    """A checkpoint entry that records each index taken of it -- what `load` reads of that key
    -- and hands back a meta tensor of the header's shape so indexed."""

    def __init__(self, key: str, dtype: torch.dtype, shape: tuple, log: list):
        self.key, self.dtype, self.shape, self.log = key, dtype, shape, log

    def __getitem__(self, index):
        self.log.append((self.key, index))
        return torch.empty(self.shape, dtype=self.dtype, device="meta")[index]


def _region(index, shape: tuple) -> tuple[tuple[int, int], ...]:
    """The `[start, stop)` an index of basic slices takes on each dimension of `shape`."""
    if index is Ellipsis:
        index = ()
    index = index if isinstance(index, tuple) else (index,)
    assert all(isinstance(s, slice) for s in index), index
    out = []
    for dim, size in enumerate(shape):
        start, stop, step = (index[dim] if dim < len(index) else slice(None)).indices(size)
        assert step == 1, index
        out.append((start, stop))
    return tuple(out)


def _sliced(key: str, world: int) -> tuple[int, int] | None:
    """`(dimension, per-rank width)` of a key the ranks split, None for one each rank reads
    whole: h heads of 128 (`wq` rows, `wo` columns), of `wkv_b`'s per-head rows (192 on RoPE
    layers, 256 on NoPE layers) and of SSMax; the dense and shared-expert rows (`w1`, `w3`) or
    columns (`w2`), their block scales on the same 128-blocks (scale axis 0 for rows, 2 for
    columns); the vocabulary rows of `output`. At tep4 h is 16, the widths 6144 and 1536 and the
    vocabulary rows 51200; at tep8 8, 3072, 768 and 25600."""
    at = _AT[world]
    h = at.heads
    if key == "output.weight":
        return 0, at.lm_rows
    m = re.fullmatch(r"layers\.(\d+)\.(.+)", key)
    if m is None:
        return None
    layer, rest = int(m.group(1)), m.group(2)
    table = {
        "attention.wq.weight": (0, h * 128),
        "attention.wq.weight_scale": (0, h),
        "attention.wo.weight": (1, h * 128),
        "attention.wo.weight_scale": (2, h),
        "attention.ssmax_scaling.s": (0, h),
        "attention.ssmax_scaling.b": (0, h),
        "attention.wkv_b.weight": (0, h * (256 if _CONFIG["nope_layer_flags"][layer] else 192)),
    }
    for module, width in (("feed_forward", at.dense), ("shared_experts", at.shared)):
        for w in ("w1", "w3"):
            table[f"{module}.{w}.weight"] = (0, width)
            table[f"{module}.{w}.weight_scale"] = (0, width // 128)
        table[f"{module}.w2.weight"] = (1, width)
        table[f"{module}.w2.weight_scale"] = (2, width // 128)
    return table.get(rest)


@pytest.mark.parametrize("world", WORLDS)
def test_the_ranks_tile_every_sliced_tensor(checkpoint, world):
    """The index the production `load` takes of every checkpoint key, on each rank, at the
    headers' shapes. Each key is read at most once per rank. Every language-model key is read
    by every rank: a sliced one -- heads, rows, columns, vocabulary rows, block scales on the
    same boundary -- along exactly one dimension, rank r at [r w, (r + 1) w), so the ranks tile
    it exactly once; any other whole. A routed expert's codes and block scales are read whole
    by the one rank whose window holds it, and its per-tensor scalars whole by every rank, for
    the asserts. Every vision key is read whole by every rank, no DSpark draft key by any. The
    bytes so read are the loader's own per-rank total, as the headers give: 146.42 GiB at tep4,
    76.78 GiB at tep8."""
    at = _AT[world]
    regions: dict = {}
    for rank in range(world):
        log: list = []
        weights = {
            key: _Cut(key, dtype, shape, log) for key, (dtype, shape) in checkpoint.headers.items()
        }
        W.MODEL_WEIGHTS.load(_Shell(checkpoint.config, world, rank, "meta"), weights)
        keys = [key for key, _ in log]
        assert len(keys) == len(set(keys)), "a key was read twice"
        total = window = 0
        for key, index in log:
            dtype, shape = checkpoint.headers[key]
            region = _region(index, shape)
            regions.setdefault(key, {})[rank] = region
            n = torch.empty((), dtype=dtype).element_size()
            for start, stop in region:
                n *= stop - start
            total += n
            window += n if _is_expert(key) and not _is_expert_scalar(key) else 0
        assert round(total / 2**30, 2) == at.read_gib, (rank, total / 2**30)
        assert round(window / 2**30, 2) == at.expert_gib, (rank, window / 2**30)

    sliced = 0
    for key, (_, shape) in checkpoint.headers.items():
        whole = tuple((0, size) for size in shape)
        if _is_draft(key):
            assert key not in regions, key
        elif _is_expert_scalar(key):
            assert regions[key] == {r: whole for r in range(world)}, key
        elif _is_expert(key):
            assert regions[key] == {_expert_of(key)[1] // at.experts: whole}, key
        elif (cut := _sliced(key, world)) is None:
            assert regions[key] == {r: whole for r in range(world)}, key
        else:
            dim, width = cut
            assert width * world == shape[dim], (key, shape)
            for r in range(world):
                want = list(whole)
                want[dim] = (r * width, (r + 1) * width)
                assert regions[key][r] == tuple(want), (key, r, regions[key][r])
            sliced += 1
    # 72 x (wq, wo, their scales, SSMax s and b, wkv_b) + 6 dense + 71 x 6 shared + output.
    assert sliced == 72 * 7 + 6 + 71 * 6 + 1


@pytest.mark.parametrize(
    "world, param, module, k, words",
    [
        (4, "mlp_dn_s", "shared_experts.w2", 1536, 3),
        (4, "mlp_dn_s", "feed_forward.w2", 6144, 12),
        (4, "wo_s", "attention.wo", 2048, 4),
        (8, "mlp_dn_s", "shared_experts.w2", 768, 2),
        (8, "mlp_dn_s", "feed_forward.w2", 3072, 6),
        (8, "wo_s", "attention.wo", 1024, 2),
    ],
)
def test_packed_k_operand_of_each_rank_on_the_checkpoint(
    checkpoint, world, param, module, k, words
):
    """The DeepGEMM scale operand of each down projection at the K a rank's column slice
    gives, built by the production path (`fill`'s steps: the row's shard on the lazy entry,
    then its transform) from the checkpoint's own scales, on every layer and rank. It equals
    an independent byte-level packing of the test's own slice of the stored grid and
    `transform_sf_into_required_layout(recipe=(1, 128, 128))` on that slice, layout and strides
    included. At tep8's K 768 that is 6 blocks in 2 words: bytes 0..3 of word 0 and 0..1 of
    word 1 hold blocks 0..5, and bytes 2..3 of word 1 are zero; tep4's slices fill whole words."""
    k_blk = k // 128
    cores = [_core(checkpoint.config, world, rank) for rank in range(world)]
    manifests = [W.MODEL_WEIGHTS.manifest(core) for core in cores]
    checked = 0
    for i in range(_LAYERS):
        key = f"layers.{i}.{module}.weight_scale"
        if key not in checkpoint.index:
            continue
        with safe_open(
            str(checkpoint.path / checkpoint.index[key]), framework="pt", device="cpu"
        ) as handle:
            stored = handle.get_tensor(key)
            built = []
            for core, manifest in zip(cores, manifests):
                [(ckpt, index, transform, shard)] = manifest[f"l{i}_{param}"]
                assert (ckpt, index) == (key, None)
                entry = W._materialize(handle.get_slice(key), shard)
                built.append(transform(entry.contiguous(), core))
        for rank, got in enumerate(built):
            n = stored.shape[0] * 128
            grid = stored[:, 0, rank * k_blk : (rank + 1) * k_blk, 0]
            assert got.shape == (words, n) and got.dtype == torch.int32, (key, got.shape)
            assert torch.equal(got, _packed(grid)), (key, rank)
            operand = got.t()
            engine = transform_sf_into_required_layout(
                grid.contiguous(), mn=n, k=k, recipe=(1, 128, 128), is_sfa=False
            )
            assert engine.shape == operand.shape and engine.stride() == operand.stride()
            assert torch.equal(engine, operand), (key, rank)
            check_sf_layout(
                operand,
                mn=n,
                k=k,
                gran=(1, 128),
                num_groups=None,
                tma_stride_check=True,
                type_check=torch.int,
            )
            assert torch.equal(inverse_transform_sf(operand, mn=n, k=k), grid), (key, rank)
            # Byte slot 4w + j of a row is byte j (low first) of its word w: K block 4w + j.
            slots = got.view(torch.uint8).view(words, n, 4).permute(1, 0, 2).reshape(n, 4 * words)
            assert bool((slots[:, :k_blk] > 0).all()), (key, rank)
            assert bool((slots[:, k_blk:] == 0).all()), (key, rank)
            checked += 1
    layers = {"shared_experts.w2": 71, "feed_forward.w2": 1, "attention.wo": 72}[module]
    assert checked == world * layers


# --------------------------------------------------------------------- values, scaled down

_TINY_LAYERS = 8
#: Every structural feature at a size a CPU fills in a second, at tep4: a dense layer, RoPE and
#: NoPE layers (3, 7), a KV sharer (7, reading 3), 'shared' indexer layers (2, 5, 6), two heads
#: and two experts per rank, one 128-block per rank on the shared expert.
_TINY_TEP4 = dict(
    num_hidden_layers=_TINY_LAYERS,
    hidden_size=256,
    num_attention_heads=8,
    qk_nope_head_dim=64,
    qk_rope_head_dim=64,
    v_head_dim=128,
    kv_lora_rank=128,
    nope_layer_flags=[i % 4 == 3 for i in range(_TINY_LAYERS)],
    kv_source_layers=[0, 1, 2, 3, 4, 5, 6, 3],
    index_sharing=["full", "full", "shared", "full", "full", "shared", "shared", "full"],
    index_n_heads=4,
    index_head_dim=128,
    first_k_dense_replace=1,
    n_routed_experts=8,
    moe_intermediate_size=512,
    n_shared_experts=1,
    intermediate_size=1024,
    vocab_size=512,
)
#: World size -> the scaled-down configuration. At tep8 the same per rank -- two heads, two
#: experts, one 128-block of the shared expert -- and the dense layer at 6 128-blocks per rank,
#: so its `w2` scale operand has tep8's half-padding second word.
_TINY = {
    4: _TINY_TEP4,
    8: dict(
        _TINY_TEP4,
        num_attention_heads=16,
        n_routed_experts=16,
        moe_intermediate_size=1024,
        intermediate_size=6144,
    ),
}


def _tiny_checkpoint(world: int, seed: int = 0) -> dict:
    """A Mistral-named checkpoint at `_TINY[world]`: e4m3 weights with power-of-two fp32 block
    scales stored `[N/128, 1, K/128, 1]` where the real one has them, NVFP4 routed experts --
    packed codes, e4m3 block scales, a weight_scale_2 per expert that w1 and w3 share, one
    input_scale per projection and layer -- bf16 elsewhere, and two DSpark draft tensors."""
    c = SimpleNamespace(**_TINY[world])
    g = torch.Generator().manual_seed(seed)
    bf16 = lambda *s: torch.randn(*s, generator=g).to(torch.bfloat16)  # noqa: E731
    scalar = lambda: torch.rand((), generator=g) * 1e-3 + 1e-5  # noqa: E731

    def fp8(name, n, k):
        codes = (torch.rand(n, k, generator=g) * 800 - 400).to(torch.float8_e4m3fn)
        exps = torch.randint(-15, -7, (n // 128, 1, k // 128, 1), generator=g).float()
        return {f"{name}.weight": codes, f"{name}.weight_scale": torch.exp2(exps)}

    def nvfp4(name, n, k, ws2, isc):
        codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=g)
        sf = (torch.rand(n, k // 16, generator=g) * 400 + 1).to(torch.float8_e4m3fn)
        return {
            f"{name}.weight": codes,
            f"{name}.weight_scale": sf,
            f"{name}.weight_scale_2": ws2.clone(),
            f"{name}.input_scale": isc.clone(),
        }

    h, qk = c.hidden_size, c.qk_nope_head_dim + c.qk_rope_head_dim
    ck = {"tok_embeddings.weight": bf16(c.vocab_size, h), "norm.weight": bf16(h)}
    ck["output.weight"] = bf16(c.vocab_size, h)
    ck["lerope.alpha"] = bf16(c.qk_rope_head_dim // 2)
    ck["mtp_dspark.backbone.final_norm.weight"] = bf16(32)
    ck["mtp_dspark.backbone.layers.0.attention.wq.weight"] = bf16(32, 64)
    for i in range(c.num_hidden_layers):
        p = f"layers.{i}"
        nope = c.nope_layer_flags[i]
        ck[f"{p}.attention_norm.weight"] = bf16(h)
        ck[f"{p}.ffn_norm.weight"] = bf16(h)
        ck.update(fp8(f"{p}.attention.wq", c.num_attention_heads * qk, h))
        ck.update(fp8(f"{p}.attention.wo", h, c.num_attention_heads * c.v_head_dim))
        ck[f"{p}.attention.ssmax_scaling.s"] = bf16(c.num_attention_heads)
        ck[f"{p}.attention.ssmax_scaling.b"] = bf16(c.num_attention_heads)
        per_head = (qk if nope else c.qk_nope_head_dim) + c.v_head_dim
        ck[f"{p}.attention.wkv_b.weight"] = bf16(c.num_attention_heads * per_head, c.kv_lora_rank)
        if c.kv_source_layers[i] == i:
            lat = c.kv_lora_rank + (0 if nope else c.qk_rope_head_dim)
            ck[f"{p}.attention.wkv_a_with_mqa.weight"] = bf16(lat, h)
            ck[f"{p}.attention.kv_a_norm.weight"] = bf16(c.kv_lora_rank)
        if c.index_sharing[i] == "full":
            x = f"{p}.attention.indexer"
            ck.update(fp8(f"{x}.wq_b", c.index_n_heads * c.index_head_dim, h))
            ck[f"{x}.wk.weight"] = bf16(c.index_head_dim, h)
            ck[f"{x}.weights_proj.weight"] = bf16(c.index_n_heads, h)
            for norm, n in (("q_input_norm", h), ("k_norm", c.index_head_dim)):
                ck[f"{x}.{norm}.weight"] = bf16(n)
                ck[f"{x}.{norm}.bias"] = bf16(n)
            ck[f"{x}.ssmax_scaling.s"] = bf16(c.index_n_heads)
            ck[f"{x}.ssmax_scaling.b"] = bf16(c.index_n_heads)
        if i < c.first_k_dense_replace:
            for w, (n, k) in (("w1", (c.intermediate_size, h)), ("w3", (c.intermediate_size, h))):
                ck.update(fp8(f"{p}.feed_forward.{w}", n, k))
            ck.update(fp8(f"{p}.feed_forward.w2", h, c.intermediate_size))
            continue
        ck[f"{p}.gate.weight"] = bf16(c.n_routed_experts, h)
        inter = c.moe_intermediate_size
        m = f"{p}.shared_experts"
        ck.update(fp8(f"{m}.w1", inter, h))
        ck.update(fp8(f"{m}.w3", inter, h))
        ck.update(fp8(f"{m}.w2", h, inter))
        isc1, isc2 = scalar(), scalar()
        for e in range(c.n_routed_experts):
            m = f"{p}.experts.{e}"
            ws2_13 = scalar()
            ck.update(nvfp4(f"{m}.w1", inter, h, ws2_13, isc1))
            ck.update(nvfp4(f"{m}.w3", inter, h, ws2_13, isc1))
            ck.update(nvfp4(f"{m}.w2", h, inter, scalar(), isc2))
    return ck


def _bytes(t: torch.Tensor) -> torch.Tensor:
    """Compare by bit pattern: e4m3 has no `==`, and it is exactness that is asserted."""
    return t.contiguous().view(torch.uint8) if t.element_size() == 1 else t


def _grid(s: torch.Tensor) -> torch.Tensor:
    """An FP8 block scale as stored, `[N/128, 1, K/128, 1]`, as its `[N/128, K/128]` grid."""
    return s.reshape(s.shape[0], s.shape[2])


def _decode_sf(storage: torch.Tensor, n: int, k_blocks: int) -> torch.Tensor:
    """Independent reading of the packed scale storage `[ceil(k_blocks/4), n]` int32: word
    `[w, row]` holds the UE8M0 bytes of K blocks 4w..4w+3 of `row`, low byte first."""
    words = storage.t().contiguous()  # [n, words]
    out = torch.empty(n, k_blocks, dtype=torch.float32)
    for kb in range(k_blocks):
        byte = (words[:, kb // 4] >> (8 * (kb % 4))) & 0xFF
        out[:, kb] = torch.exp2(byte.float() - 127)
    return out


def _runner_rows(n: int, interleave: bool) -> torch.Tensor:
    """The stored row each row of an expert's runner operand holds, one row at a time: row
    `32b + 8v + u` takes row `32b + 4u + v` of the interleaved stack, whose row `2i` is up row
    `i` (stored row `i`) and row `2i + 1` gate row `i` (stored row `n/2 + i`)."""
    rows = []
    for r in range(n):
        b, within = divmod(r, 32)
        v, u = divmod(within, 8)
        j = 32 * b + 4 * u + v
        rows.append((j // 2 if j % 2 == 0 else n // 2 + j // 2) if interleave else j)
    return torch.tensor(rows)


def _swizzle_offsets(m: int, c: int) -> torch.Tensor:
    """`[m, c]`: where scale byte `(row, col)` lands in the 128x4-swizzled buffer."""
    r = torch.arange(m).view(-1, 1)
    k = torch.arange(c).view(1, -1)
    return (r // 128) * 128 * c + (k // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + k % 4


def _swizzled(s: torch.Tensor) -> torch.Tensor:
    """`[m, c]` scale bytes scattered to their swizzled offsets, viewed `[m, c]` again."""
    m, c = s.shape
    out = torch.empty(m * c, dtype=torch.uint8)
    out[_swizzle_offsets(m, c).reshape(-1)] = s.reshape(-1)
    return out.view(m, c)


def _unswizzled(s: torch.Tensor) -> torch.Tensor:
    m, c = s.shape
    return s.reshape(-1)[_swizzle_offsets(m, c)]


def _expected(ck: dict, world: int, rank: int) -> dict:
    """What each parameter of `rank` must hold, built from the checkpoint by plain slicing,
    with fp32 block-scale grids standing for the packed scale storage (`*_s` int32)."""
    c = SimpleNamespace(**_TINY[world])
    heads = c.num_attention_heads // world
    qk, v = c.qk_nope_head_dim + c.qk_rope_head_dim, c.v_head_dim
    local = c.n_routed_experts // world
    out = {}

    def rows(t, n):
        return t[rank * n : (rank + 1) * n]

    def cols(t, n):
        return t[:, rank * n : (rank + 1) * n]

    for i in range(c.num_hidden_layers):
        p, q = f"layers.{i}", f"l{i}_"
        nope = qk if c.nope_layer_flags[i] else c.qk_nope_head_dim
        out[q + "attn_norm"] = ck[f"{p}.attention_norm.weight"]
        out[q + "ffn_norm"] = ck[f"{p}.ffn_norm.weight"]
        out[q + "wq"] = rows(ck[f"{p}.attention.wq.weight"], heads * qk)
        out[q + "wq_s"] = rows(_grid(ck[f"{p}.attention.wq.weight_scale"]), heads * qk // 128)
        out[q + "wo"] = cols(ck[f"{p}.attention.wo.weight"], heads * v)
        out[q + "wo_s"] = cols(_grid(ck[f"{p}.attention.wo.weight_scale"]), heads * v // 128)
        for s in ("s", "b"):
            out[q + f"ssmax_{s}"] = rows(ck[f"{p}.attention.ssmax_scaling.{s}"], heads)
        kvb = rows(ck[f"{p}.attention.wkv_b.weight"], heads * (nope + v))
        per_head = [kvb[h * (nope + v) : (h + 1) * (nope + v)] for h in range(heads)]
        out[q + "wkv_b"] = torch.cat([t[:nope] for t in per_head] + [t[nope:] for t in per_head])
        if c.kv_source_layers[i] == i:
            out[q + "wkv_a"] = ck[f"{p}.attention.wkv_a_with_mqa.weight"]
            out[q + "kv_a_norm"] = ck[f"{p}.attention.kv_a_norm.weight"]
        if c.index_sharing[i] == "full":
            x = f"{p}.attention.indexer"
            out[q + "ix_q_norm_w"] = ck[f"{x}.q_input_norm.weight"]
            out[q + "ix_q_norm_b"] = ck[f"{x}.q_input_norm.bias"]
            out[q + "ix_wq_b"] = ck[f"{x}.wq_b.weight"]
            out[q + "ix_wq_b_s"] = _grid(ck[f"{x}.wq_b.weight_scale"])
            out[q + "ix_wk_wp"] = torch.cat([ck[f"{x}.wk.weight"], ck[f"{x}.weights_proj.weight"]])
            out[q + "ix_k_norm_w"] = ck[f"{x}.k_norm.weight"].float()
            out[q + "ix_k_norm_b"] = ck[f"{x}.k_norm.bias"].float()
            out[q + "ix_ssmax_s"] = ck[f"{x}.ssmax_scaling.s"]
            out[q + "ix_ssmax_b"] = ck[f"{x}.ssmax_scaling.b"]
        if i < c.first_k_dense_replace:
            m, inter = f"{p}.feed_forward", c.intermediate_size // world
        else:
            m, inter = f"{p}.shared_experts", c.moe_intermediate_size // world
        # Gate rows first.
        out[q + "mlp_gu"] = torch.cat(
            [_bytes(rows(ck[f"{m}.{w}.weight"], inter)) for w in ("w1", "w3")]
        )
        out[q + "mlp_gu_s"] = torch.cat(
            [rows(_grid(ck[f"{m}.{w}.weight_scale"]), inter // 128) for w in ("w1", "w3")]
        )
        out[q + "mlp_dn"] = cols(ck[f"{m}.w2.weight"], inter)
        out[q + "mlp_dn_s"] = cols(_grid(ck[f"{m}.w2.weight_scale"]), inter // 128)
        if i < c.first_k_dense_replace:
            continue
        out[q + "router"] = ck[f"{p}.gate.weight"]
        window = [f"{p}.experts.{rank * local + s}" for s in range(local)]
        # Up rows first, interleaved and block-shuffled; the block scales also swizzled.
        fc1 = _runner_rows(2 * c.moe_intermediate_size, True)
        fc2 = _runner_rows(c.hidden_size, False)

        def up_gate(e, suffix):
            return torch.cat([_bytes(ck[f"{e}.{w}.{suffix}"]) for w in ("w3", "w1")])[fc1]

        out[q + "fc1_w"] = torch.stack([up_gate(e, "weight") for e in window])
        out[q + "fc1_s"] = torch.stack([_swizzled(up_gate(e, "weight_scale")) for e in window])
        out[q + "fc2_w"] = torch.stack([ck[f"{e}.w2.weight"][fc2] for e in window])
        out[q + "fc2_s"] = torch.stack(
            [_swizzled(_bytes(ck[f"{e}.w2.weight_scale"])[fc2]) for e in window]
        )
        out[q + "isc1"] = ck[f"{window[0]}.w1.input_scale"].reshape(1)
        out[q + "isc2"] = ck[f"{window[0]}.w2.input_scale"].reshape(1)
        out[q + "ws2_13"] = torch.stack([ck[f"{e}.w1.weight_scale_2"] for e in window])
        out[q + "ws2_2"] = torch.stack([ck[f"{e}.w2.weight_scale_2"] for e in window])
    out["final_norm"] = ck["norm.weight"]
    out["embed"] = ck["tok_embeddings.weight"]
    out["lerope_alpha"] = ck["lerope.alpha"].float()
    return out


@pytest.mark.parametrize("world", WORLDS)
def test_tiny_scale_operands_cover_whole_and_partial_words(world):
    """The scaled-down table's scale operands at a rank, by K blocks: 2 for every K of the
    hidden size 256 and for `wo` (2 heads x 128), 1 for the shared `w2` (128 of 512 at tep4, of
    1024 at tep8), and for the dense `w2` 2 at tep4 (256 of 1024) and 6 at tep8 (768 of 6144)
    -- the shared expert's case at tep8's full size, 6 blocks in 2 words. Every one has a
    partly padded last word, which the values test below compares whole."""
    d = W.MODEL_WEIGHTS.dims(_core(_TINY[world], world, 0))
    dense, dense_words = {4: (256, 1), 8: (768, 2)}[world]
    assert (d.heads, d.dense_inter, d.shared_inter, d.local_experts) == (2, dense, 128, 2)
    with torch.device("meta"):
        declared = W.MODEL_WEIGHTS.declare(d)
    assert tuple(declared["l0_mlp_dn_s"].shape) == (dense_words, 256)
    assert tuple(declared["l1_mlp_dn_s"].shape) == (1, 256)
    assert tuple(declared["l1_wo_s"].shape) == (1, 256)


@pytest.mark.parametrize("world, rank", [(w, r) for w in WORLDS for r in range(w)])
def test_tiny_load_values(world, rank):
    """Every parameter of every rank, bit for bit against `_expected`; each scale operand's
    whole int32 storage, padding bytes included, against `_packed`."""
    ck = _tiny_checkpoint(world)
    model = _Shell(_TINY[world], world, rank, "cpu")
    W.MODEL_WEIGHTS.load(model, dict(ck))
    got = dict(model.model.w)
    want = _expected(ck, world, rank)
    assert set(got) == set(want)
    for key, ref in want.items():
        param = got[key].data
        if key.endswith("_s") and param.dtype == torch.int32:
            # DeepGEMM's operand is the transpose of the storage: MN-major, row stride N.
            n = param.shape[1]
            k = ref.shape[1] * 128
            operand = param.t()
            check_sf_layout(
                operand,
                mn=n,
                k=k,
                gran=(1, 128),
                num_groups=None,
                tma_stride_check=True,
                type_check=torch.int,
            )
            assert torch.equal(param, _packed(ref)), key
            per_row = ref.repeat_interleave(128, dim=0)
            assert torch.equal(_decode_sf(param, n, ref.shape[1]), per_row), key
            assert torch.equal(inverse_transform_sf(operand, mn=n, k=k), ref), key
            if key.endswith(("wq_s", "wo_s", "ix_wq_b_s", "mlp_dn_s")):
                engine = transform_sf_into_required_layout(
                    ref.contiguous(), mn=n, k=k, recipe=(1, 128, 128), is_sfa=False
                )
                assert engine.shape == operand.shape and engine.stride() == operand.stride()
                assert torch.equal(engine, operand), key
            continue
        if ref.dtype == torch.uint8 and param.dtype == torch.float8_e4m3fn:
            assert key.endswith(("mlp_gu", "fc1_s", "fc2_s")), key
        else:
            assert param.dtype == ref.dtype, (key, param.dtype, ref.dtype)
        assert torch.equal(_bytes(param), _bytes(ref)), key
    lm_rows = _TINY[world]["vocab_size"] // world
    assert torch.equal(
        model.lm_head.weight.data, ck["output.weight"][rank * lm_rows : (rank + 1) * lm_rows]
    )


@pytest.mark.parametrize("world", WORLDS)
def test_tiny_expert_halves_are_up_then_gate(world):
    """Undo the block shuffle (and the scales' swizzle) of FC1: its even rows are w3 (up), its
    odd rows w1 (gate), codes and block scales alike -- the order the NVFP4 runner's FC1 reads,
    up first, the opposite of `mlp_gu`'s gate first."""
    ck = _tiny_checkpoint(world)
    model = _Shell(_TINY[world], world, 1, "cpu")
    W.MODEL_WEIGHTS.load(model, dict(ck))
    inter, e = _TINY[world]["moe_intermediate_size"], 2  # rank 1's local slot 0
    w = model.model.w
    shuffled = _runner_rows(2 * inter, False)
    unshuffle = torch.empty_like(shuffled)
    unshuffle[shuffled] = torch.arange(2 * inter)
    codes = w["l4_fc1_w"][0][unshuffle]
    assert torch.equal(codes[0::2], ck[f"layers.4.experts.{e}.w3.weight"])
    assert torch.equal(codes[1::2], ck[f"layers.4.experts.{e}.w1.weight"])
    scales = _unswizzled(_bytes(w["l4_fc1_s"][0]))[unshuffle]
    assert torch.equal(scales[0::2], _bytes(ck[f"layers.4.experts.{e}.w3.weight_scale"]))
    assert torch.equal(scales[1::2], _bytes(ck[f"layers.4.experts.{e}.w1.weight_scale"]))


@pytest.mark.parametrize("world", WORLDS)
def test_tiny_layout_is_applied_once_per_load(world):
    """A second `load` into the same parameters starts again from the stored bytes, so it lands
    on the same layout rather than shuffling the first one's result again."""
    ck = _tiny_checkpoint(world)
    model = _Shell(_TINY[world], world, world - 1, "cpu")
    W.MODEL_WEIGHTS.load(model, dict(ck))
    first = {k: _bytes(v.data).clone() for k, v in model.model.w.items() if k.startswith("l5_fc")}
    W.MODEL_WEIGHTS.load(model, dict(ck))
    for key, value in first.items():
        assert torch.equal(_bytes(model.model.w[key].data), value), key


def _write_shards(ck: dict, directory: Path, files: int = 3) -> None:
    """`ck` as Mistral-named shard files with their index, keys dealt round-robin, so one
    expert's tensors -- and w1's and w3's scalars -- land in different files."""
    from safetensors.torch import save_file

    names = [f"consolidated-{i + 1:05d}-of-{files:05d}.safetensors" for i in range(files)]
    weight_map = {key: names[i % files] for i, key in enumerate(sorted(ck))}
    for name in names:
        part = {k: v.contiguous() for k, v in ck.items() if weight_map[k] == name}
        save_file(part, str(directory / name))
    (directory / W.SHARD_INDEX).write_text(json.dumps({"weight_map": weight_map}))


class _ReadLog:
    """`safe_open` that records every key read through it."""

    def __init__(self, real):
        self.real, self.keys = real, []

    def __call__(self, *args, **kwargs):
        log = self

        class Handle:
            def __enter__(self):
                self.h = log.real(*args, **kwargs).__enter__()
                return self

            def __exit__(self, *exc):
                return self.h.__exit__(*exc)

            def get_slice(self, key):
                log.keys.append(key)
                return self.h.get_slice(key)

        return Handle()


@pytest.mark.parametrize("world, rank", [(w, r) for w in WORLDS for r in range(w)])
def test_tiny_load_streams_from_the_shard_files(world, rank, tmp_path, monkeypatch):
    """With the checkpoint's shard files found through the directory the engine records on
    the config, every source is read from them -- the engine's dict is never read: here it
    holds meta tensors, which cannot be copied out -- and the result is bit for bit the
    dict path's. Each key is read once; of the other ranks' experts only the scalars are read,
    and nothing of the DSpark draft. Each file's cached pages are dropped once read."""
    ck = _tiny_checkpoint(world)
    _write_shards(ck, tmp_path)
    log = _ReadLog(W.safe_open)
    dropped = []
    drop = W._drop_page_cache
    monkeypatch.setattr(W, "safe_open", log)
    monkeypatch.setattr(W, "_drop_page_cache", lambda path: (dropped.append(path), drop(path)))
    streamed = _Shell(_TINY[world], world, rank, "cpu")
    streamed.model.model_config.pretrained_config._name_or_path = str(tmp_path)
    W.MODEL_WEIGHTS.load(streamed, {k: torch.empty_like(v, device="meta") for k, v in ck.items()})
    monkeypatch.undo()
    direct = _Shell(_TINY[world], world, rank, "cpu")
    W.MODEL_WEIGHTS.load(direct, dict(ck))
    got, want = dict(streamed.named_parameters()), dict(direct.named_parameters())
    assert set(got) == set(want)
    for key in want:
        assert torch.equal(_bytes(got[key].data), _bytes(want[key].data)), key

    read = set(log.keys)
    assert len(read) == len(log.keys), "a key was read twice"
    lo, hi = 2 * rank, 2 * rank + 2
    off = {k for k in ck if _is_expert(k) and not lo <= _expert_of(k)[1] < hi}
    own = {k for k in ck if not _is_draft(k)} - off
    assert read == own | {k for k in off if _is_expert_scalar(k)}
    assert sorted(p.name for p in dropped) == sorted(p.name for p in tmp_path.glob("*.safetensors"))


@pytest.mark.parametrize("world", WORLDS)
def test_tiny_load_falls_back_to_the_dict_without_an_index(world, tmp_path):
    """A recorded directory without the shard index is not the checkpoint: the dict is read."""
    ck = _tiny_checkpoint(world)
    model = _Shell(_TINY[world], world, 2, "cpu")
    model.model.model_config.pretrained_config._name_or_path = str(tmp_path)
    W.MODEL_WEIGHTS.load(model, dict(ck))


@pytest.mark.parametrize("world", WORLDS)
def test_tiny_load_without_draft_tensors(world):
    """The DSpark draft is never asserted on: a checkpoint without it loads too."""
    ck = {k: v for k, v in _tiny_checkpoint(world).items() if not _is_draft(k)}
    W.MODEL_WEIGHTS.load(_Shell(_TINY[world], world, 1, "cpu"), ck)


@pytest.mark.parametrize("world", WORLDS)
def test_tiny_load_without_off_window_codes(world):
    """The other ranks' expert codes and block scales are never read nor asserted present."""
    ck = _tiny_checkpoint(world)
    for e in range(2, _TINY[world]["n_routed_experts"]):
        for proj in ("w1", "w2", "w3"):
            del ck[f"layers.3.experts.{e}.{proj}.weight"]
            del ck[f"layers.3.experts.{e}.{proj}.weight_scale"]
    W.MODEL_WEIGHTS.load(_Shell(_TINY[world], world, 0, "cpu"), ck)


def _load_raises(ck: dict, world: int, match: str, rank: int = 0) -> None:
    with pytest.raises(AssertionError, match=match):
        W.MODEL_WEIGHTS.load(_Shell(_TINY[world], world, rank, "cpu"), ck)


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize(
    "key",
    [
        "layers.3.experts.1.w3.weight",
        "layers.3.experts.1.w1.weight_scale",
        "layers.3.experts.0.w3.weight_scale_2",
        "layers.3.experts.1.w2.input_scale",
    ],
)
def test_tiny_load_refuses_a_missing_in_window_tensor(key, world):
    ck = _tiny_checkpoint(world)
    del ck[key]
    _load_raises(ck, world, f"checkpoint key missing: {key}")


@pytest.mark.parametrize("world, expert", [(4, 5), (8, 13)])
def test_tiny_load_refuses_a_missing_off_window_scalar(world, expert):
    """The other ranks' scalars are read for the per-layer asserts, so they must be there."""
    ck = _tiny_checkpoint(world)
    del ck[f"layers.3.experts.{expert}.w2.input_scale"]
    _load_raises(ck, world, f"checkpoint key missing: layers.3.experts.{expert}.w2.input_scale")


@pytest.mark.parametrize("world", WORLDS)
def test_tiny_load_refuses_an_unknown_tensor(world):
    ck = _tiny_checkpoint(world)
    ck["layers.2.attention.wq_extra.weight"] = torch.zeros(4)
    _load_raises(ck, world, "checkpoint coverage")


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize(
    "key",
    [
        "layers.0.attention.wq.weight_scale",
        "layers.0.feed_forward.w3.weight_scale",
        "layers.4.shared_experts.w2.weight_scale",
        "layers.4.attention.indexer.wq_b.weight_scale",
        "layers.5.attention.wo.weight_scale",
    ],
)
def test_tiny_load_refuses_a_scale_that_is_not_ue8m0(key, world):
    ck = _tiny_checkpoint(world)
    ck[key] = ck[key].clone()
    ck[key].view(-1)[0] *= 1.5
    _load_raises(ck, world, "not a positive normal power of two")


@pytest.mark.parametrize("world", WORLDS)
def test_tiny_load_refuses_a_two_dimensional_fp8_scale(world):
    """A `[N/128, K/128]` block-scale layout is not this one's."""
    ck = _tiny_checkpoint(world)
    key = "layers.4.attention.indexer.wq_b.weight_scale"
    ck[key] = _grid(ck[key]).contiguous()
    _load_raises(ck, world, r"not \[N/128, 1, K/128, 1\]")


@pytest.mark.parametrize("world, expert", [(4, 0), (4, 6), (8, 0), (8, 13)])
@pytest.mark.parametrize("scalar", ["weight_scale_2", "input_scale"])
def test_tiny_load_refuses_w1_and_w3_scalars_that_differ(world, expert, scalar):
    """On this rank's experts and on the others' alike."""
    ck = _tiny_checkpoint(world)
    key = f"layers.5.experts.{expert}.w3.{scalar}"
    ck[key] = ck[key] * 2
    experts = _TINY[world]["n_routed_experts"]
    want = f"layers.5: w1 and w3 {scalar} differ on 1 of {experts} experts, first expert {expert}:"
    _load_raises(ck, world, want)


@pytest.mark.parametrize("world, expert", [(4, 7), (8, 15)])
@pytest.mark.parametrize("proj", ["w1", "w2"])
def test_tiny_load_refuses_an_input_scale_that_varies_over_the_experts(world, expert, proj):
    """Rank 0 sees the other ranks' experts' input_scale: one off-window expert disagreeing is
    a quantization rank 0's window would not share."""
    ck = _tiny_checkpoint(world)
    for p in {proj, "w3"} if proj == "w1" else {proj}:
        key = f"layers.6.experts.{expert}.{p}.input_scale"
        ck[key] = ck[key] * 2
    want = f"layers.6: {proj}.input_scale is not one value over the experts: expert {expert}"
    _load_raises(ck, world, want)


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("value", [0.0, -1e-4, float("inf"), float("nan")])
def test_tiny_load_refuses_a_scalar_that_is_not_a_positive_finite_scale(value, world):
    ck = _tiny_checkpoint(world)
    ck["layers.2.experts.3.w2.weight_scale_2"] = torch.tensor(value)
    _load_raises(
        ck, world, "layers.2: w2.weight_scale_2 of expert 3 is .*not a positive finite scale"
    )


def test_shared_scalar_refuses_copies_that_differ():
    """The row transform asserts on its own copies too, whatever ran before it."""
    one = torch.tensor(0.5)
    assert W._shared_scalar((one, one.clone(), one.clone()), None) is one
    with pytest.raises(AssertionError, match="1 of 3 copies"):
        W._shared_scalar((one, one.clone(), torch.tensor(0.25)), None)


@pytest.mark.parametrize(
    "value, exact",
    [
        (1.0, True),
        (2.0**-126, True),
        (2.0**127, True),
        (2.0**-15, True),
        (0.0, False),
        (-(2.0**-12), False),
        (3.0 * 2.0**-12, False),
        (2.0**-130, False),  # subnormal
        (float("inf"), False),
        (float("nan"), False),
    ],
)
def test_ue8m0_assert(value, exact):
    s = torch.full((2, 3), 2.0**-10)
    s[1, 2] = value
    if exact:
        W._assert_ue8m0(s)
    else:
        with pytest.raises(AssertionError, match="1 of 6"):
            W._assert_ue8m0(s)


def test_packed_reference_itself():
    """`_packed` on a hand-written grid: 6 K blocks with exponent bytes 120..125 in one
    128-row block -> word 0 = 120 | 121 << 8 | 122 << 16 | 123 << 24, word 1 = 124 | 125 << 8,
    the same for all 128 rows; a byte of 128 or more lands in the sign bit."""
    grid = torch.exp2(torch.arange(120, 126, dtype=torch.float32) - 127).view(1, 6)
    got = _packed(grid)
    assert got.shape == (2, 128)
    assert bool((got[0] == 120 | 121 << 8 | 122 << 16 | 123 << 24).all())
    assert bool((got[1] == 124 | 125 << 8).all())
    high = _packed(torch.tensor([[1.0, 1.0, 1.0, 2.0]]))
    assert bool((high[0] == (127 | 127 << 8 | 127 << 16 | 128 << 24) - 2**32).all())


@pytest.mark.parametrize("world", [4, 8])
def test_native_dspark_manifest_shards_and_off_coverage(world):
    native = {
        "n_layers": 3,
        "depth": 3,
        "target_activation_layers": "1,18,36,54,71",
        "dspark": {"markov_rank": 512},
    }
    # Tensor dimensions from the checkpoint's safetensors header, independently
    # of the table's shape lambdas. Values encode the sliced axis so a row /
    # column or head-boundary error is visible without allocating the weights.
    projections = {
        "wq": (8192, 6144, 0),
        "wkv_b": (12288, 512, 0),
        "wo": (6144, 8192, 1),
        "w1": (24576, 6144, 0),
        "w3": (24576, 6144, 0),
        "w2": (6144, 24576, 1),
    }
    shards = {key: [] for key in projections}
    for rank in range(world):
        core = _core(_CONFIG, world=world, rank=rank)
        off = W.MODEL_WEIGHTS.manifest(core)
        core.model_config.spec_config = SimpleNamespace(
            _draft_model_config=native, target_layer_ids=[1, 18, 36, 54, 71]
        )
        manifest = W.MODEL_WEIGHTS.manifest(core)
        added = set(manifest) - set(off)
        assert len(added) == 38
        keys = {manifest[name][0][0] for name in added}
        assert len(keys) == 38 and all(key.startswith("mtp_dspark.") for key in keys)
        assert not keys.intersection(W.MODEL_WEIGHTS.expected_unconsumed(core, dict.fromkeys(keys)))
        with torch.device("meta"):
            declared = W.MODEL_WEIGHTS.declare(W.MODEL_WEIGHTS.dims(core))
        assert tuple(declared["draft_target_projection"].shape) == (6144, 30720)
        assert tuple(declared["draft_joint"].shape) == (1536, 7168)
        assert tuple(declared["draft_joint_bias"].shape) == (1536,)
        assert tuple(declared["draft_mask"].shape) == (2, 6144)
        assert all(declared[name].dtype == torch.bfloat16 for name in added)
        for role, (rows, cols, axis) in projections.items():
            ids = torch.arange((rows, cols)[axis])
            source = (
                ids[:, None].expand(rows, cols) if axis == 0 else ids[None, :].expand(rows, cols)
            )
            key, index, transform, shard = manifest[f"l0_draft_{role}"][0]
            assert index is None and transform is None
            sliced = source if shard is None else shard(source)
            assert tuple(sliced.shape) == tuple(declared[f"l0_draft_{role}"].shape)
            shards[role].append(sliced[:, 0] if axis == 0 else sliced[0])
        core.model_config.spec_config = None
        assert keys.issubset(W.MODEL_WEIGHTS.expected_unconsumed(core, dict.fromkeys(keys)))
    for role, parts in shards.items():
        rows, cols, axis = projections[role]
        expected = torch.arange((rows, cols)[axis])
        torch.testing.assert_close(torch.cat(parts), expected)
