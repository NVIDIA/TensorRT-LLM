# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Mistral Large 4 NVFP4 dep4/dep8 weight table, manifest, and synthetic CPU load tests.

No GPU. Attention data parallel, with routed experts expert-parallel over every rank, at both
worlds the table serves (dep4 on one node of four; dep8 on two nodes of four). Two kinds of
evidence:

* the table at the checkpoint's language-model configuration: which layers carry which weight at
  what per-rank shape -- every head and the full MLP widths on every rank -- that each rank's
  manifest reads only its own routed experts (32 at four ranks, 16 at eight) and the same
  replicated rest, the windows tiling the 128 experts exactly once, and that manifest and
  declaration agree;
* a scaled-down configuration with real CPU tensors, loaded on every rank and compared element for
  element with an independent construction of what each parameter must hold: the replicated weights
  whole and identical on every rank, the `wkv_b` regroup, both half orders, the packed scale
  operand, the expert window in the MoE runner's layout and its scalars, the whole `lm_head`, DSpark
  draft manifests, shard streaming, and each assert's refusal.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.modeling_v2.models.mistral_large4 import weight_table as WT
from tensorrt_llm._torch.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__dep import (  # noqa: E501
    weights as W,
)
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
    return {k for sources in manifest.values() for ckpt, *_ in sources for k in WT._keys(ckpt)}


def _is_vision(key: str) -> bool:
    return key.startswith(WT.VISION_PREFIXES)


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
                keys = WT._keys(ckpt)
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
            key: [(ckpt, index) for ckpt, index, _, _ in sources]
            for key, sources in manifest.items()
            if _role(key) not in _EXPERT_PARAMS
        }

    first = replicated(0)
    assert len(first) == len(W.MODEL_WEIGHTS.manifest(_core(_CONFIG, 0, world))) - 71 * 8
    for rank in range(1, world):
        assert replicated(rank) == first, rank
    for key, sources in first.items():
        for ckpt, index in sources:
            assert not any(_is_expert(k) for k in WT._keys(ckpt)), (key, ckpt)


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
