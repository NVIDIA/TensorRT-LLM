# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Mistral Large 4 NVFP4 dep4 weight table against the checkpoint it reads, at both of the worlds
it serves: four ranks on one node, and eight on two nodes of four (dep8; rank r is local rank
r % 4 on node r // 4); attention data parallel, the routed experts expert-parallel over every
rank.

No GPU. Three kinds of evidence:

* the table at the checkpoint's own language-model configuration (written out below, and
  compared with the configuration adapted from the checkpoint's `params.json` when it is
  present): which layers carry which weight at what per-rank shape -- every head and the full
  MLP widths on every rank -- and that each rank's manifest reads only its own routed experts
  (32 at four ranks, 16 at eight) and the same replicated rest, the windows tiling the 128
  experts exactly once, at eight ranks the first four on node 0 and the last four on node 1;
* the checkpoint itself, through its index and safetensors headers, as meta tensors, plus the
  values of its small scale tensors: what each key stores; the union of the ranks' sources is
  exactly the index's keys but the DSpark draft's, which every rank declares unconsumed; each
  rank consumes all 1,721 language-model keys and all 276 vision-tower keys whole and its
  window's expert keys, 27,264 and 159.86 GiB in all at four ranks, 13,632 and 92.46 GiB at
  eight, 3.17 GiB of it the vision tower; the production `load` -- every transform, shape and
  dtype assert, the expert-scalar asserts on the checkpoint's own scalars and the two-way
  coverage assert -- passes on each rank, reading each key once and whole, beyond the manifest
  only the other ranks' expert scalars, for the asserts (163,584 B at four ranks, 190,848 B at
  eight); at eight ranks each node's four ranks read 294.65 GiB of distinct bytes; and every
  FP8 block scale and every expert scalar satisfies the invariants `load` asserts. Skipped
  where the checkpoint is absent (looked up as `mistral-large-4-nvfp4` or
  `Mistral-Large-4-1T-A52B-NVFP4` under `LLM_MODELS_ROOT`);
* a scaled-down configuration with real CPU tensors, loaded on every rank and compared
  element for element with an independent construction of what each parameter must hold:
  the replicated weights whole and identical on every rank, the `wkv_b` regroup, both half
  orders, the packed scale operand, the expert window in the MoE runner's layout and its
  scalars, the whole `lm_head`, and the DSpark draft tensors left alone; plus what the
  streaming path reads off the shard files, and each assert's refusal.
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

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__dep4 import (  # noqa: E501
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

#: The worlds the table serves: four ranks on one node, eight over two nodes of four (dep8).
WORLDS = (4, 8)
#: Devices per GB300 node on this cluster: one node holds four ranks, two hold eight.
GPUS_PER_NODE = 4
CHECKPOINTS = ("mistral-large-4-nvfp4", "Mistral-Large-4-1T-A52B-NVFP4")

_LAYERS = 72
_EXPERTS = 128

#: Each world, and each rank of each world.
_each_world = pytest.mark.parametrize("world", WORLDS, ids=lambda w: f"world{w}")
_each_rank = pytest.mark.parametrize(
    "world, rank",
    [(w, r) for w in WORLDS for r in range(w)],
    ids=[f"world{w}-rank{r}" for w in WORLDS for r in range(w)],
)

#: Per world, a rank's declared routed experts and all its language-model parameters with the
#: whole lm_head (GiB), as the table declares them at `_CONFIG`.
_DECLARED_GIB = {4: (134.79, 156.80), 8: (67.39, 89.41)}
#: The BF16 vision tower -- encoder, pre-projector norm, patch merger and language adapter --
#: which every rank reads whole: its keys and bytes in the checkpoint (3.17 GiB).
_VISION_KEYS, _VISION_BYTES = 276, 3_408_437_248
#: The DSpark draft's keys, all in the checkpoint's last shard, which no rank reads.
_DRAFT_KEYS = 38
#: Per world, what one rank reads off the checkpoint: its window of routed experts (GiB), the
#: window with the replicated rest, the vision tower among it (bytes and GiB), and what the
#: production `load` reads in all -- that and the other ranks' expert scalars, for its asserts
#: -- in keys and bytes.
_READS = {
    4: dict(window_gib=134.79, total=171_644_394_304, total_gib=159.86, keys=70157),
    8: dict(window_gib=67.39, total=99_280_040_128, total_gib=92.46, keys=63341),
}
_LOAD_READ = {4: 171_644_557_888, 8: 99_280_230_976}

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
    n_routed_experts=_EXPERTS,
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


def _mapping(rank: int, world: int) -> Mapping:
    """Rank `rank` of the `world`-rank deployment, as the engine builds it. A topology that
    ignored `rank=` (with MPI disabled, `Mapping` takes the process's own rank) would hand every
    rank rank 0's window, so the rank is checked."""
    mapping = Mapping(
        world_size=world,
        rank=rank,
        gpus_per_node=GPUS_PER_NODE,
        tp_size=world,
        moe_tp_size=1,
        moe_ep_size=world,
        enable_attention_dp=True,
    )
    assert (mapping.tp_rank, mapping.moe_ep_rank, mapping.moe_tp_rank) == (rank, rank, 0)
    return mapping


def _core(config: dict, rank: int, world: int) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=SimpleNamespace(**config),
            mapping=_mapping(rank, world),
            torch_dtype=torch.bfloat16,
        )
    )


class _Core(nn.Module):
    """The surface `load` reads off the core: its config and its declared weights."""

    def __init__(self, config: dict, rank: int, world: int, device: str):
        super().__init__()
        self.model_config = _core(config, rank, world).model_config
        with torch.device(device):
            self.w = W.MODEL_WEIGHTS.declare(W.MODEL_WEIGHTS.dims(self))


class _Shell(nn.Module):
    """The causal-LM shell: the core as `model`, and the `lm_head` it builds under attention
    DP, the full vocabulary (`lm_rows` overrides that, to build the tensor-parallel shape)."""

    def __init__(
        self, config: dict, rank: int, world: int, device: str, lm_rows: int | None = None
    ):
        super().__init__()
        self.model = _Core(config, rank, world, device)
        self.lm_head = nn.Module()
        rows = config["vocab_size"] if lm_rows is None else lm_rows
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


# --------------------------------------------------------------------- the configuration


def _layers_of(declared, name: str) -> dict[int, tuple]:
    """Layer index -> declared shape, for every `l{i}_<name>`."""
    out = {}
    for key, param in declared.items():
        head, _, rest = key.partition("_")
        if head.startswith("l") and head[1:].isdigit() and rest == name:
            out[int(head[1:])] = tuple(param.shape)
    return out


@pytest.mark.parametrize(
    "world, rank", [(4, 0), (8, 0), (8, 7)], ids=["world4-rank0", "world8-rank0", "world8-rank7"]
)
def test_layer_sets_and_per_rank_shapes(world, rank):
    """Which layers carry which weight, at what per-rank shape: all 64 heads of `wq`, `wkv_b`,
    `wo` and SSMax, and the dense MLP (I 24576) and shared expert (I 6144) at full width, on
    every rank; 32 routed experts per rank at four ranks, 16 at eight. The NoPE layers are 3, 7,
    ..., 71; the KV sharers 39, 43, ..., 71; the 'shared' indexer layers 17, 18, 21, 22, ..., 69,
    70."""
    core = _core(_CONFIG, rank, world)
    local = _EXPERTS // world
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
        "wq": {i: (8192, 6144) for i in every},
        "wq_s": {i: (12, 8192) for i in every},
        "ssmax_s": {i: (64,) for i in every},
        "ssmax_b": {i: (64,) for i in every},
        "wkv_a": {i: (512 if i in nope else 576, 6144) for i in range(_LAYERS) if i not in sharers},
        "kv_a_norm": {i: (512,) for i in range(_LAYERS) if i not in sharers},
        "wkv_b": {i: (16384 if i in nope else 12288, 512) for i in every},
        "wo": {i: (6144, 8192) for i in every},
        "wo_s": {i: (16, 6144) for i in every},
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
        "mlp_gu": {i: (49152 if i == 0 else 12288, 6144) for i in every},
        "mlp_gu_s": {i: (12, 49152 if i == 0 else 12288) for i in every},
        "mlp_dn": {i: (6144, 24576 if i == 0 else 6144) for i in every},
        "mlp_dn_s": {i: (48 if i == 0 else 12, 6144) for i in every},
        "router": {i: (128, 6144) for i in moe},
        "fc1_w": {i: (local, 12288, 3072) for i in moe},
        "fc1_s": {i: (local, 12288, 384) for i in moe},
        "fc2_w": {i: (local, 6144, 3072) for i in moe},
        "fc2_s": {i: (local, 6144, 384) for i in moe},
        "isc1": {i: (1,) for i in moe},
        "isc2": {i: (1,) for i in moe},
        "ws2_13": {i: (local,) for i in moe},
        "ws2_2": {i: (local,) for i in moe},
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

    # Per rank, with the shell's whole lm_head. The routed experts are exactly their checkpoint
    # bytes; the FP8 scale operands, one int32 word per four 128-blocks per output row, are
    # 0.12 GiB more than the fp32 grids they are built from.
    nbytes = {k: p.numel() * p.element_size() for k, p in declared.items()}
    experts = sum(n for k, n in nbytes.items() if _role(k) in _EXPERT_PARAMS)
    total = sum(nbytes.values()) + 204800 * 6144 * 2
    assert (round(experts / 2**30, 2), round(total / 2**30, 2)) == _DECLARED_GIB[world], (
        experts / 2**30,
        total / 2**30,
    )


@_each_world
def test_each_rank_reads_only_its_expert_window(world):
    """Rank r's manifest names routed experts [Er, Er + E), E = 128 / world (32 at four ranks,
    16 at eight), and nothing else, on every MoE layer, local slot s <- global expert Er + s, all
    twelve tensors of each; the one-per-layer scalars read the window's copies. The windows are
    disjoint and cover experts 0..127, node by node: at four ranks the one node holds them all;
    at eight node 0 (ranks 0-3, local ranks 0-3) holds experts 0..63, node 1 (ranks 4-7)
    64..127. The known-not-loaded set is the other 128 - E experts: their scalars predicted,
    their codes and block scales only where found."""
    local = _EXPERTS // world
    nodes = world // GPUS_PER_NODE
    seen = set()
    windows = {}
    by_node: dict[int, set] = {node: set() for node in range(nodes)}
    for rank in range(world):
        core = _core(_CONFIG, rank, world)
        mapping = core.model_config.mapping
        assert (mapping.local_rank, mapping.node_rank) == (
            rank % GPUS_PER_NODE,
            rank // GPUS_PER_NODE,
        )
        manifest = W.MODEL_WEIGHTS.manifest(core)
        lo, hi = local * rank, local * rank + local
        d = W.MODEL_WEIGHTS.dims(core)
        assert (d.local_experts, d.expert_offset) == (local, lo)
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
        seen |= set(experts)
        windows[rank] = {e for _, e in experts}
        by_node[mapping.node_rank] |= windows[rank]

        predicted = W.MODEL_WEIGHTS.expected_unconsumed(core, {})
        assert len(predicted) == 71 * (_EXPERTS - local) * 6
        assert all(_is_expert_scalar(k) for k in predicted)
        assert all(not lo <= _expert_of(k)[1] < hi for k in predicted)
    assert seen == {(i, e) for i in range(1, _LAYERS) for e in range(_EXPERTS)}
    assert sum(len(w) for w in windows.values()) == _EXPERTS
    assert set().union(*windows.values()) == set(range(_EXPERTS))
    per_node = _EXPERTS // nodes
    assert by_node == {n: set(range(per_node * n, per_node * n + per_node)) for n in range(nodes)}
    if world == 8:
        assert by_node == {0: set(range(64)), 1: set(range(64, 128))}


@_each_world
def test_every_rank_reads_the_same_replicated_sources(world):
    """Outside the routed experts every rank's manifest is the same: the same parameters from
    the same checkpoint keys at the same indices, with no per-rank slice anywhere."""

    def replicated(rank):
        manifest = W.MODEL_WEIGHTS.manifest(_core(_CONFIG, rank, world))
        return {
            key: [(ckpt, index) for ckpt, index, _ in sources]
            for key, sources in manifest.items()
            if _role(key) not in _EXPERT_PARAMS
        }

    first = replicated(0)
    assert len(first) == len(W.MODEL_WEIGHTS.manifest(_core(_CONFIG, 0, world))) - 71 * 8
    for rank in range(1, world):
        assert replicated(rank) == first, rank
    for key, sources in first.items():
        for ckpt, index in sources:
            assert not any(_is_expert(k) for k in W._keys(ckpt)), (key, ckpt)


@_each_world
def test_manifest_and_declaration_agree(world):
    """A declared weight with no source allocates memory nothing fills."""
    for rank in range(world):
        core = _core(_CONFIG, rank, world)
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
    reader: the same keys at either world."""
    d = W.MODEL_WEIGHTS.dims(_core(checkpoint.config, 0, WORLDS[0]))
    keys = W._expert_scalar_keys(d)
    return W._read_scalars(keys, checkpoint.index, (checkpoint.path, checkpoint.index))


def _meta(checkpoint) -> dict:
    return {
        key: torch.empty(shape, dtype=dtype, device="meta")
        for key, (dtype, shape) in checkpoint.headers.items()
    }


def _nbytes(checkpoint, key: str) -> int:
    dtype, shape = checkpoint.headers[key]
    n = torch.empty((), dtype=dtype).element_size()
    for s in shape:
        n *= s
    return n


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


@_each_world
def test_union_over_ranks_is_every_key_but_the_draft(checkpoint, world):
    """Every key of the index but the DSpark draft's 38 is read by some rank (lm_head's
    `output.weight` by the shell); per rank, read and known-not-loaded split the index, the
    known-not-loaded the draft and the other ranks' experts, and a rank reads every one of the
    1,721 language-model keys, the 276 vision-tower keys and its window's expert keys: 71 layers
    x 32 experts x 3 projections x 4 tensors = 27,264 at four ranks, 71 x 16 x 3 x 4 = 13,632 at
    eight."""
    local = _EXPERTS // world
    keys = set(checkpoint.index)
    draft = {k for k in keys if _is_draft(k)}
    vision = {k for k in keys if _is_vision(k)}
    lm = {k for k in keys - draft - vision if not _is_expert(k)}
    assert len(keys) == 111091 and len(draft) == _DRAFT_KEYS and len(vision) == _VISION_KEYS
    assert len(lm) == 1721
    union = set()
    for rank in range(world):
        core = _core(checkpoint.config, rank, world)
        read = _source_keys(W.MODEL_WEIGHTS.manifest(core)) | {"output.weight"}
        skipped = W.MODEL_WEIGHTS.expected_unconsumed(core, checkpoint.index)
        assert not read & set(skipped)
        assert read | set(skipped) == keys
        assert {k for k in read if not _is_expert(k)} == lm | vision
        own = len({k for k in read if _is_expert(k)})
        assert own == {4: 27264, 8: 13632}[world] == 71 * local * 3 * 4
        assert {k for k in skipped if not _is_expert(k)} == draft
        assert {skipped[k] for k in draft} == {"DSpark draft disabled"}
        assert len(skipped) == _DRAFT_KEYS + 71 * (_EXPERTS - local) * 12
        union |= read
    assert union == keys - draft


def _category(key: str) -> str:
    if _is_vision(key):
        return "vision"
    if _is_expert(key):
        return "routed"
    if ".shared_experts." in key:
        return "shared"
    if ".feed_forward." in key:
        return "dense"
    if key in ("tok_embeddings.weight", "output.weight"):
        return "embed+output"
    if ".attention." in key:
        return "attention+indexer"
    if key.endswith(".gate.weight"):
        return "router"
    return "norms+lerope"


@_each_world
def test_per_rank_bytes_from_the_headers(checkpoint, world):
    """What each rank moves, from the safetensors headers through the production manifest:
    every language-model tensor and the vision tower whole and this rank's routed-expert window,
    the same on every rank -- 171,644,394,304 B (159.86 GiB) at four ranks, 134.79 GiB of it the
    window, and 99,280,040,128 B (92.46 GiB) at eight, 67.39 GiB of it the window -- 21.89 GiB
    the replicated language-model rest: attention and indexer 9.19, shared experts 7.49,
    embedding and output 4.6875, the dense layer-0 MLP 0.42, the router 0.10 -- and the
    vision tower's 3,408,437,248 B (3.17 GiB) (each source is read whole, so it is the headers'
    own byte count)."""
    local, reads = _EXPERTS // world, _READS[world]
    lm = sum(
        _nbytes(checkpoint, k)
        for k in checkpoint.index
        if not _is_vision(k) and not _is_draft(k) and not _is_expert(k)
    )
    assert round(lm / 2**30, 2) == 21.89, lm / 2**30
    vision = sum(_nbytes(checkpoint, k) for k in checkpoint.index if _is_vision(k))
    assert vision == _VISION_BYTES, vision
    for rank in range(world):
        core = _core(checkpoint.config, rank, world)
        lo, hi = local * rank, local * rank + local
        window = sum(
            _nbytes(checkpoint, k)
            for k in checkpoint.index
            if _is_expert(k) and lo <= _expert_of(k)[1] < hi
        )
        meta = {}

        def entry(key):
            if key not in meta:
                dtype, shape = checkpoint.headers[key]
                meta[key] = torch.empty(shape, dtype=dtype, device="meta")
            return meta[key]

        by_category: dict[str, int] = {}
        for sources in W.MODEL_WEIGHTS.manifest(core).values():
            for ckpt, _, _ in sources:
                for key in W._keys(ckpt):
                    t = W._materialize(entry(key))
                    n = t.numel() * t.element_size()
                    by_category[_category(key)] = by_category.get(_category(key), 0) + n
        by_category["embed+output"] += _nbytes(checkpoint, "output.weight")
        total = sum(by_category.values())
        assert by_category["routed"] == window, (rank, by_category["routed"], window)
        assert by_category["vision"] == vision, (rank, by_category["vision"], vision)
        assert total == lm + vision + window == reads["total"], (rank, total, lm + vision + window)
        assert round(total / 2**30, 2) == reads["total_gib"], (rank, total / 2**30)
        # The embedding and the whole lm_head: 2 x 2.34375 GiB, exactly.
        assert by_category.pop("embed+output") == 2 * 204800 * 6144 * 2
        gib = {c: round(n / 2**30, 2) for c, n in by_category.items()}
        assert gib == {
            "routed": reads["window_gib"],
            "attention+indexer": 9.19,
            "shared": 7.49,
            "dense": 0.42,
            "router": 0.10,
            "norms+lerope": 0.0,
            "vision": 3.17,
        }, (rank, gib)


def test_per_node_bytes_from_the_headers(checkpoint):
    """At eight ranks over two nodes of four (dep8), what each node's page cache serves: the
    distinct keys its four ranks read. Both nodes read the whole replicated set -- the language
    model's 21.89 GiB and the vision tower's 3.17 GiB -- and each one half of the expert stack,
    node 0 experts 0..63 and node 1 experts 64..127 (269.58 GiB): 316,373,102,656 B
    (294.65 GiB) per node, against the four ranks' 4 x 92.46 GiB of reads."""
    world = 8
    nodes: dict[int, set] = {0: set(), 1: set()}
    for rank in range(world):
        core = _core(checkpoint.config, rank, world)
        read = _source_keys(W.MODEL_WEIGHTS.manifest(core)) | {"output.weight"}
        nodes[core.model_config.mapping.node_rank] |= read
    expert_halves = []
    for node, keys in nodes.items():
        experts = {_expert_of(k)[1] for k in keys if _is_expert(k)}
        assert experts == set(range(64 * node, 64 * node + 64)), node
        routed = sum(_nbytes(checkpoint, k) for k in keys if _is_expert(k))
        vision = sum(_nbytes(checkpoint, k) for k in keys if _is_vision(k))
        rest = sum(_nbytes(checkpoint, k) for k in keys if not _is_expert(k) and not _is_vision(k))
        assert round(routed / 2**30, 2) == 269.58 and round(rest / 2**30, 2) == 21.89, node
        assert vision == _VISION_BYTES, (node, vision)
        assert routed + rest + vision == 316_373_102_656, (node, routed + rest + vision)
        assert round((routed + rest + vision) / 2**30, 2) == 294.65, node
        expert_halves.append({k for k in keys if _is_expert(k)})
    assert not expert_halves[0] & expert_halves[1]
    assert nodes[0] - expert_halves[0] == nodes[1] - expert_halves[1]


class _Recorded:
    """A checkpoint entry that records each read `load` makes of it -- its key and the bytes it
    takes -- and hands back the entry so indexed."""

    def __init__(self, key: str, entry: torch.Tensor, log: list):
        self.key, self.entry, self.log = key, entry, log

    def __getitem__(self, index):
        t = self.entry[index]
        self.log.append((self.key, t.numel() * t.element_size()))
        return t


@_each_rank
def test_load_on_meta_matches_the_headers(checkpoint, expert_scalars, world, rank):
    """The production `load` over meta tensors shaped like the safetensors headers, with the
    checkpoint's own expert scalars: the scalar asserts pass on every layer, every source is
    transformed to its declared per-rank shape and dtype, every declared parameter is filled,
    and every key is consumed or known-not-loaded. What it reads, recorded key by key, is the
    loader's own per-rank total: each key once and whole, every vision key and no DSpark draft
    key, the expert codes and block scales of this rank's window only -- the manifest's bytes
    and the other experts' scalars, read for the asserts: 70,157 keys and 171,644,557,888 B at
    four ranks (the other 96 experts' 40,896 scalars, 163,584 B), 63,341 keys and
    99,280,230,976 B at eight (the other 112 experts' 47,712 scalars, 190,848 B)."""
    local, reads = _EXPERTS // world, _READS[world]
    log: list = []
    weights = {key: _Recorded(key, t, log) for key, t in _meta(checkpoint).items()}
    weights.update({key: _Recorded(key, t, log) for key, t in expert_scalars.items()})
    model = _Shell(checkpoint.config, rank, world, "meta")
    W.MODEL_WEIGHTS.load(model, weights)

    keys = [key for key, _ in log]
    assert len(keys) == len(set(keys)) == reads["keys"], "a key was read twice"
    assert not any(_is_draft(k) for k in keys)
    assert {k for k in keys if _is_vision(k)} == {k for k in checkpoint.index if _is_vision(k)}
    for key in keys:
        if _is_expert(key) and not _is_expert_scalar(key):
            assert _expert_of(key)[1] // local == rank, key
    read = sum(n for _, n in log)
    off_window = {k for k in keys if _is_expert(k) and _expert_of(k)[1] // local != rank}
    off_scalars = 71 * (_EXPERTS - local) * 6
    assert len(off_window) == off_scalars and all(_is_expert_scalar(k) for k in off_window)
    assert read == reads["total"] + off_scalars * 4 == _LOAD_READ[world], read
    assert all(n == _nbytes(checkpoint, k) for k, n in log), "a key was read in part"


@_each_world
def test_scale_invariants_hold_on_the_checkpoint(checkpoint, expert_scalars, world):
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

    d = W.MODEL_WEIGHTS.dims(_core(checkpoint.config, 0, world))
    assert len(expert_scalars) == 71 * 128 * 6
    W._check_expert_scalars(expert_scalars, d)
    for i in range(1, _LAYERS):
        isc1 = expert_scalars[f"layers.{i}.experts.0.w1.input_scale"].item()
        isc2 = expert_scalars[f"layers.{i}.experts.0.w2.input_scale"].item()
        assert 0 < isc1 < 1 and 0 < isc2 < 1, (i, isc1, isc2)


# --------------------------------------------------------------------- values, scaled down

_TINY_LAYERS = 8
#: Every structural feature at a size a CPU fills in a second: a dense layer, RoPE and NoPE
#: layers (3, 7), a KV sharer (7, reading 3), 'shared' indexer layers (2, 5, 6), two experts
#: per rank (`_tiny`), several 128-blocks on every FP8 axis.
_TINY = dict(
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
    moe_intermediate_size=512,
    n_shared_experts=1,
    intermediate_size=1024,
    vocab_size=512,
)
_TINY_LOCAL = 2


def _tiny(world: int) -> dict:
    """`_TINY` at `world` ranks: two routed experts per rank, 8 at four ranks and 16 at eight."""
    return dict(_TINY, n_routed_experts=_TINY_LOCAL * world)


def _tiny_checkpoint(world: int, seed: int = 0) -> dict:
    """A Mistral-named checkpoint at `_tiny(world)`: e4m3 weights with power-of-two fp32 block scales
    stored `[N/128, 1, K/128, 1]` where the real one has them, NVFP4 routed experts -- packed
    codes, e4m3 block scales, a weight_scale_2 per expert that w1 and w3 share, one input_scale
    per projection and layer -- bf16 elsewhere, and two DSpark draft tensors."""
    c = SimpleNamespace(**_tiny(world))
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


def _expected(ck: dict, rank: int, world: int) -> dict:
    """What each parameter of `rank` must hold, built from the checkpoint whole -- only the
    routed experts depend on the rank -- with fp32 block-scale grids standing for the packed
    scale storage (`*_s` int32)."""
    c = SimpleNamespace(**_tiny(world))
    heads = c.num_attention_heads
    qk, v = c.qk_nope_head_dim + c.qk_rope_head_dim, c.v_head_dim
    local = c.n_routed_experts // world
    out = {}

    for i in range(c.num_hidden_layers):
        p, q = f"layers.{i}", f"l{i}_"
        nope = qk if c.nope_layer_flags[i] else c.qk_nope_head_dim
        out[q + "attn_norm"] = ck[f"{p}.attention_norm.weight"]
        out[q + "ffn_norm"] = ck[f"{p}.ffn_norm.weight"]
        out[q + "wq"] = ck[f"{p}.attention.wq.weight"]
        out[q + "wq_s"] = _grid(ck[f"{p}.attention.wq.weight_scale"])
        out[q + "wo"] = ck[f"{p}.attention.wo.weight"]
        out[q + "wo_s"] = _grid(ck[f"{p}.attention.wo.weight_scale"])
        for s in ("s", "b"):
            out[q + f"ssmax_{s}"] = ck[f"{p}.attention.ssmax_scaling.{s}"]
        kvb = ck[f"{p}.attention.wkv_b.weight"]
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
        m = f"{p}.feed_forward" if i < c.first_k_dense_replace else f"{p}.shared_experts"
        # Gate rows first.
        out[q + "mlp_gu"] = torch.cat([_bytes(ck[f"{m}.{w}.weight"]) for w in ("w1", "w3")])
        out[q + "mlp_gu_s"] = torch.cat([_grid(ck[f"{m}.{w}.weight_scale"]) for w in ("w1", "w3")])
        out[q + "mlp_dn"] = ck[f"{m}.w2.weight"]
        out[q + "mlp_dn_s"] = _grid(ck[f"{m}.w2.weight_scale"])
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


@_each_rank
def test_tiny_load_values(world, rank):
    """Every parameter of every rank, bit for bit against `_expected`."""
    ck = _tiny_checkpoint(world)
    model = _Shell(_tiny(world), rank, world, "cpu")
    W.MODEL_WEIGHTS.load(model, dict(ck))
    got = dict(model.model.w)
    want = _expected(ck, rank, world)
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
    assert torch.equal(model.lm_head.weight.data, ck["output.weight"])


@_each_world
def test_tiny_replicated_weights_are_identical_on_every_rank(world):
    """Under attention DP every rank holds the same bytes outside its expert window, the whole
    `lm_head` included, and disjoint windows of the routed experts, one per rank."""
    ck = _tiny_checkpoint(world)
    models = []
    for rank in range(world):
        model = _Shell(_tiny(world), rank, world, "cpu")
        W.MODEL_WEIGHTS.load(model, dict(ck))
        models.append(dict(model.named_parameters()))
    first = models[0]
    for rank, params in enumerate(models[1:], start=1):
        assert set(params) == set(first)
        for key, param in params.items():
            same = torch.equal(_bytes(param.data), _bytes(first[key].data))
            routed = _role(key.removeprefix("model.w.")) in _EXPERT_PARAMS
            # isc1/isc2 are the layer's one value, the same in every window.
            assert same == (not routed or key.endswith(("_isc1", "_isc2"))), (rank, key)


@_each_world
def test_tiny_expert_halves_are_up_then_gate(world):
    """Undo the block shuffle (and the scales' swizzle) of FC1: its even rows are w3 (up), its
    odd rows w1 (gate), codes and block scales alike -- the order the NVFP4 runner's FC1 reads,
    up first, the opposite of `mlp_gu`'s gate first."""
    ck = _tiny_checkpoint(world)
    model = _Shell(_tiny(world), 1, world, "cpu")
    W.MODEL_WEIGHTS.load(model, dict(ck))
    inter, e = _TINY["moe_intermediate_size"], _TINY_LOCAL  # rank 1's local slot 0
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


@_each_world
def test_tiny_layout_is_applied_once_per_load(world):
    """A second `load` into the same parameters starts again from the stored bytes, so it lands
    on the same layout rather than shuffling the first one's result again."""
    ck = _tiny_checkpoint(world)
    model = _Shell(_tiny(world), 3, world, "cpu")
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


@_each_rank
def test_tiny_load_streams_from_the_shard_files(world, rank, tmp_path, monkeypatch):
    """With the checkpoint's shard files found through the directory the engine records on
    the config, every source is read from them -- the engine's dict is never read: here it
    holds meta tensors, which cannot be copied out -- and the result is bit for bit the
    dict path's. Each key is read once; of the other ranks' experts only the scalars are read,
    and nothing of the DSpark draft. Each shard file's cached pages are dropped once, after
    its sources are read."""
    ck = _tiny_checkpoint(world)
    _write_shards(ck, tmp_path)
    log = _ReadLog(W.safe_open)
    dropped = []
    monkeypatch.setattr(W, "safe_open", log)
    monkeypatch.setattr(W, "_drop_page_cache", lambda path: dropped.append(path.name))
    streamed = _Shell(_tiny(world), rank, world, "cpu")
    streamed.model.model_config.pretrained_config._name_or_path = str(tmp_path)
    W.MODEL_WEIGHTS.load(streamed, {k: torch.empty_like(v, device="meta") for k, v in ck.items()})
    monkeypatch.undo()
    direct = _Shell(_tiny(world), rank, world, "cpu")
    W.MODEL_WEIGHTS.load(direct, dict(ck))
    got, want = dict(streamed.named_parameters()), dict(direct.named_parameters())
    assert set(got) == set(want)
    for key in want:
        assert torch.equal(_bytes(got[key].data), _bytes(want[key].data)), key

    read = set(log.keys)
    assert len(read) == len(log.keys), "a key was read twice"
    lo, hi = _TINY_LOCAL * rank, _TINY_LOCAL * rank + _TINY_LOCAL
    off = {k for k in ck if _is_expert(k) and not lo <= _expert_of(k)[1] < hi}
    own = {k for k in ck if not _is_draft(k)} - off
    assert read == own | {k for k in off if _is_expert_scalar(k)}
    assert sorted(dropped) == sorted(p.name for p in tmp_path.glob("*.safetensors"))


@_each_world
def test_tiny_load_falls_back_to_the_dict_without_an_index(world, tmp_path):
    """A recorded directory without the shard index is not the checkpoint: the dict is read."""
    ck = _tiny_checkpoint(world)
    model = _Shell(_tiny(world), 2, world, "cpu")
    model.model.model_config.pretrained_config._name_or_path = str(tmp_path)
    W.MODEL_WEIGHTS.load(model, dict(ck))


@_each_world
def test_tiny_load_without_draft_tensors(world):
    """The DSpark draft is never asserted on: a checkpoint without it loads too."""
    ck = {k: v for k, v in _tiny_checkpoint(world).items() if not _is_draft(k)}
    W.MODEL_WEIGHTS.load(_Shell(_tiny(world), 1, world, "cpu"), ck)


@_each_world
def test_tiny_load_without_off_window_codes(world):
    """The other ranks' expert codes and block scales are never read nor asserted present."""
    ck = _tiny_checkpoint(world)
    for e in range(_TINY_LOCAL, _TINY_LOCAL * world):
        for proj in ("w1", "w2", "w3"):
            del ck[f"layers.3.experts.{e}.{proj}.weight"]
            del ck[f"layers.3.experts.{e}.{proj}.weight_scale"]
    W.MODEL_WEIGHTS.load(_Shell(_tiny(world), 0, world, "cpu"), ck)


def _load_raises(ck: dict, match: str, world: int, rank: int = 0) -> None:
    with pytest.raises(AssertionError, match=match):
        W.MODEL_WEIGHTS.load(_Shell(_tiny(world), rank, world, "cpu"), ck)


@_each_world
def test_tiny_load_refuses_a_vocab_parallel_lm_head(world):
    """The whole `output.weight` does not fit a tensor-parallel `[vocab/world, hidden]` lm_head,
    and nothing slices it to fit."""
    model = _Shell(_tiny(world), 1, world, "cpu", lm_rows=_TINY["vocab_size"] // world)
    with pytest.raises(AssertionError, match="output.weight"):
        W.MODEL_WEIGHTS.load(model, _tiny_checkpoint(world))


@pytest.mark.parametrize(
    "key",
    [
        "layers.3.experts.1.w3.weight",
        "layers.3.experts.1.w1.weight_scale",
        "layers.3.experts.0.w3.weight_scale_2",
        "layers.3.experts.1.w2.input_scale",
    ],
)
@_each_world
def test_tiny_load_refuses_a_missing_in_window_tensor(key, world):
    ck = _tiny_checkpoint(world)
    del ck[key]
    _load_raises(ck, f"checkpoint key missing: {key}", world)


@pytest.mark.parametrize("world, expert", [(4, 5), (8, 13)], ids=["world4", "world8"])
def test_tiny_load_refuses_a_missing_off_window_scalar(world, expert):
    """The other ranks' scalars are read for the per-layer asserts, so they must be there: here
    those of an expert of rank 2 at four ranks, of rank 6 at eight."""
    ck = _tiny_checkpoint(world)
    del ck[f"layers.3.experts.{expert}.w2.input_scale"]
    _load_raises(ck, f"checkpoint key missing: layers.3.experts.{expert}.w2.input_scale", world)


@_each_world
def test_tiny_load_refuses_an_unknown_tensor(world):
    ck = _tiny_checkpoint(world)
    ck["layers.2.attention.wq_extra.weight"] = torch.zeros(4)
    _load_raises(ck, "checkpoint coverage", world)


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
@_each_world
def test_tiny_load_refuses_a_scale_that_is_not_ue8m0(key, world):
    """Rank 0 checks the whole grid: the bad block is the last one, which a tensor-parallel
    rank 0 would never read."""
    ck = _tiny_checkpoint(world)
    ck[key] = ck[key].clone()
    ck[key].view(-1)[-1] *= 1.5
    _load_raises(ck, "not a positive normal power of two", world)


@_each_world
def test_tiny_load_refuses_a_two_dimensional_fp8_scale(world):
    """A `[N/128, K/128]` block-scale layout is not this one's."""
    ck = _tiny_checkpoint(world)
    key = "layers.4.attention.indexer.wq_b.weight_scale"
    ck[key] = _grid(ck[key]).contiguous()
    _load_raises(ck, r"not \[N/128, 1, K/128, 1\]", world)


@pytest.mark.parametrize(
    "world, expert",
    [(4, 0), (4, 6), (8, 0), (8, 14)],
    ids=["world4-0", "world4-6", "world8-0", "world8-14"],
)
@pytest.mark.parametrize("scalar", ["weight_scale_2", "input_scale"])
def test_tiny_load_refuses_w1_and_w3_scalars_that_differ(world, expert, scalar):
    """On this rank's experts and on the others' alike."""
    ck = _tiny_checkpoint(world)
    key = f"layers.5.experts.{expert}.w3.{scalar}"
    ck[key] = ck[key] * 2
    count = _TINY_LOCAL * world
    want = f"layers.5: w1 and w3 {scalar} differ on 1 of {count} experts, first expert {expert}:"
    _load_raises(ck, want, world)


@_each_world
@pytest.mark.parametrize("proj", ["w1", "w2"])
def test_tiny_load_refuses_an_input_scale_that_varies_over_the_experts(proj, world):
    """Rank 0 sees the other ranks' experts' input_scale: one off-window expert disagreeing --
    here the last rank's, at eight ranks on the other node -- is a quantization rank 0's window
    would not share."""
    ck = _tiny_checkpoint(world)
    last = _TINY_LOCAL * world - 1
    for p in {proj, "w3"} if proj == "w1" else {proj}:
        key = f"layers.6.experts.{last}.{p}.input_scale"
        ck[key] = ck[key] * 2
    want = f"layers.6: {proj}.input_scale is not one value over the experts: expert {last}"
    _load_raises(ck, want, world)


@pytest.mark.parametrize("value", [0.0, -1e-4, float("inf"), float("nan")])
@_each_world
def test_tiny_load_refuses_a_scalar_that_is_not_a_positive_finite_scale(value, world):
    ck = _tiny_checkpoint(world)
    ck["layers.2.experts.3.w2.weight_scale_2"] = torch.tensor(value)
    want = "layers.2: w2.weight_scale_2 of expert 3 is .*not a positive finite scale"
    _load_raises(ck, want, world)


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
            key, index, transform = manifest[f"l0_draft_{role}"][0]
            assert index is None and transform is None
            sliced = source
            assert tuple(sliced.shape) == tuple(declared[f"l0_draft_{role}"].shape)
            shards[role].append(sliced[:, 0] if axis == 0 else sliced[0])
        core.model_config.spec_config = None
        assert keys.issubset(W.MODEL_WEIGHTS.expected_unconsumed(core, dict.fromkeys(keys)))
    for role, parts in shards.items():
        rows, cols, axis = projections[role]
        expected = torch.arange((rows, cols)[axis])
        for part in parts:
            torch.testing.assert_close(part, expected)
