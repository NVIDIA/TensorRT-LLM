# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Rubin targets' weight tables against the ModelOpt NVFP4 export, on the CPU.

Each target's weights.py owns its table and its copy of the table machinery, so every test runs on
each layout (dep16, helix8, tep8; the table tests also on dep12 and dep8, context servers of the
dep16 target over 12 and 8 ranks):

* ``dims`` gives every rank its share, and the shares tile the model's heads, intermediates and
  routed experts.
* The table machinery (``declare``, ``manifest``, ``read``, ``fill_table``, ``load``,
  ``open_checkpoint_weights``) on a small synthetic table: every parameter written once, only the
  cut read, coverage asserted both ways.
* The target's tables (``w1_rows`` and ``w2_rows``) against a snapshot of the export's
  safetensors index and shard headers, no tensor bytes: every source key exists, every key is
  read or an expected leftover, every cut is in bounds, every relayout gives its parameter's
  shape and dtype (run on meta tensors), the routed experts are read once at this rank's share,
  and the parameter bytes equal the bytes read.
* The routed rows of ``w2_rows`` are ``core.w`` rows by global layer, named as the ``_MoeW`` view
  reads them, at the shapes and dtypes the expert entries take, the same on every server role.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import math
import re
from collections import Counter
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.config import KimiK31TextConfig
from tensorrt_llm.mapping import CpType, Mapping

__extra_import_path__ = ["ops/moe"]
from _rubin_a2a_kernel_split import kernel_target_rank  # noqa: E402

pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"


class _Layout(NamedTuple):
    roles: tuple[str, ...]  # the server roles the target serves
    mapping: dict  # the Mapping arguments besides the rank
    megamoe: bool = False  # the table holds MegaMoE's routed rows (dep16's, on either role)

    @property
    def world(self) -> int:
        return self.mapping["world_size"]


_LAYOUTS = {
    # Attention DP over 16 ranks, the routed experts EP16.
    "dep16": _Layout(
        ("context", "generation"),
        dict(world_size=16, tp_size=16, moe_tp_size=1, moe_ep_size=16, enable_attention_dp=True),
        megamoe=True,
    ),
    # dep16's target on a context server over 12 ranks: attention DP 12, the routed experts EP12.
    # Its table is dep16's, so it declares MegaMoE's rows on either role.
    "dep12": _Layout(
        ("context",),
        dict(world_size=12, tp_size=12, moe_tp_size=1, moe_ep_size=12, enable_attention_dp=True),
        megamoe=True,
    ),
    # dep16's target on a context server over 8 ranks: attention DP 8, the routed experts EP8.
    "dep8": _Layout(
        ("context",),
        dict(world_size=8, tp_size=8, moe_tp_size=1, moe_ep_size=8, enable_attention_dp=True),
        megamoe=True,
    ),
    # Helix CP over 8 ranks for MLA, the same ranks as TP8 elsewhere, MoE TP2 x EP4.
    "helix8": _Layout(
        ("generation",),
        dict(
            world_size=8,
            tp_size=1,
            cp_size=8,
            cp_config={"cp_type": CpType.HELIX},
            moe_tp_size=2,
            moe_ep_size=4,
        ),
    ),
    # Attention TP over 8 ranks, the routed experts EP8.
    "tep8": _Layout(
        ("context", "generation"), dict(world_size=8, tp_size=8, moe_tp_size=1, moe_ep_size=8)
    ),
}

#: A layout another layout's target serves, and that target's layout (its weights.py).
_TARGET_OF = {"dep12": "dep16", "dep8": "dep16"}
#: The layouts whose target has its own weights.py: the table machinery runs once per module.
_OWN_TABLES = [layout for layout in _LAYOUTS if layout not in _TARGET_OF]

# --- the export, as its index and shard headers state it ---

#: The MLA (DSA) layers; every other layer is KDA. Layer 0 has the dense MLP, 1-92 the latent MoE.
_MLA = (*range(3, 88, 4), 91, 92)
_LAYERS = {
    "all": tuple(range(93)),
    "dense": (0,),
    "moe": tuple(range(1, 93)),
    "mla": _MLA,
    "kda": tuple(i for i in range(93) if i not in _MLA),
}
_EXPERTS = 896

# nvidia/kimi_k31_pro_nvfp4 rev 1d82032: its safetensors index and the headers of its 96 shards
# (992,691 tensors). The text model's: (key under "language_model.", dtype, shape, layers), {i}
# the layer and {e} the routed expert.
_L = "model.layers.{i}."
_ATTN = _L + "self_attn."
_MOE = _L + "block_sparse_moe."
_EXP = _MOE + "experts.{e}."
_CKPT = (
    ("lm_head.weight", "BF16", (163840, 7168), None),
    ("model.embed_tokens.weight", "BF16", (163840, 7168), None),
    (_EXP + "w1.input_scale", "F32", (), "moe"),
    (_EXP + "w1.weight", "U8", (3072, 1792), "moe"),
    (_EXP + "w1.weight_scale", "F8_E4M3", (3072, 224), "moe"),
    (_EXP + "w1.weight_scale_2", "F32", (), "moe"),
    (_EXP + "w2.input_scale", "F32", (), "moe"),
    (_EXP + "w2.weight", "U8", (3584, 1536), "moe"),
    (_EXP + "w2.weight_scale", "F8_E4M3", (3584, 192), "moe"),
    (_EXP + "w2.weight_scale_2", "F32", (), "moe"),
    (_EXP + "w3.input_scale", "F32", (), "moe"),
    (_EXP + "w3.weight", "U8", (3072, 1792), "moe"),
    (_EXP + "w3.weight_scale", "F8_E4M3", (3072, 224), "moe"),
    (_EXP + "w3.weight_scale_2", "F32", (), "moe"),
    (_MOE + "gate.e_score_correction_bias", "F32", (896,), "moe"),
    (_MOE + "gate.weight", "BF16", (896, 7168), "moe"),
    (_MOE + "routed_expert_down_proj.weight", "BF16", (3584, 7168), "moe"),
    (_MOE + "routed_expert_norm.weight", "BF16", (3584,), "moe"),
    (_MOE + "routed_expert_up_proj.weight", "BF16", (7168, 3584), "moe"),
    (_MOE + "shared_experts.down_proj.weight", "F8_E4M3", (7168, 6144), "moe"),
    (_MOE + "shared_experts.down_proj.weight_scale", "U8", (7168, 192), "moe"),
    (_MOE + "shared_experts.gate_proj.weight", "F8_E4M3", (6144, 7168), "moe"),
    (_MOE + "shared_experts.gate_proj.weight_scale", "U8", (6144, 224), "moe"),
    (_MOE + "shared_experts.up_proj.weight", "F8_E4M3", (6144, 7168), "moe"),
    (_MOE + "shared_experts.up_proj.weight_scale", "U8", (6144, 224), "moe"),
    (_L + "input_layernorm.weight", "BF16", (7168,), "all"),
    (_L + "mlp.down_proj.weight", "BF16", (7168, 33792), "dense"),
    (_L + "mlp.gate_proj.weight", "BF16", (33792, 7168), "dense"),
    (_L + "mlp.up_proj.weight", "BF16", (33792, 7168), "dense"),
    (_L + "mlp_res_norm.weight", "BF16", (7168,), "all"),
    (_L + "mlp_res_proj.weight", "BF16", (1, 7168), "all"),
    (_L + "post_attention_layernorm.weight", "BF16", (7168,), "all"),
    (_L + "self_attention_res_norm.weight", "BF16", (7168,), "all"),
    (_L + "self_attention_res_proj.weight", "BF16", (1, 7168), "all"),
    (_ATTN + "A_log", "F32", (128,), "kda"),
    (_ATTN + "b_proj.weight", "F8_E4M3", (96, 7168), "kda"),
    (_ATTN + "b_proj.weight_scale", "U8", (96, 224), "kda"),
    (_ATTN + "dt_bias", "F32", (12288,), "kda"),
    (_ATTN + "f_a_proj.weight", "F8_E4M3", (128, 7168), "kda"),
    (_ATTN + "f_a_proj.weight_scale", "U8", (128, 224), "kda"),
    (_ATTN + "f_b_proj.weight", "BF16", (12288, 128), "kda"),
    (_ATTN + "g_proj.weight", "BF16", (12288, 7168), "mla"),
    (_ATTN + "g_proj.weight", "F8_E4M3", (12288, 7168), "kda"),
    (_ATTN + "g_proj.weight_scale", "U8", (12288, 224), "kda"),
    (_ATTN + "indexer.k_norm.bias", "BF16", (128,), "mla"),
    (_ATTN + "indexer.k_norm.weight", "BF16", (128,), "mla"),
    (_ATTN + "indexer.weights_proj.weight", "BF16", (32, 7168), "mla"),
    (_ATTN + "indexer.wk.weight", "BF16", (128, 7168), "mla"),
    (_ATTN + "indexer.wq_b.weight", "BF16", (4096, 1536), "mla"),
    (_ATTN + "k_conv1d.weight", "F32", (12288, 1, 4), "kda"),
    (_ATTN + "k_proj.weight", "F8_E4M3", (12288, 7168), "kda"),
    (_ATTN + "k_proj.weight_scale", "U8", (12288, 224), "kda"),
    (_ATTN + "kv_a_layernorm.weight", "BF16", (512,), "mla"),
    (_ATTN + "kv_a_proj_with_mqa.weight", "BF16", (576, 7168), "mla"),
    (_ATTN + "kv_b_proj.weight", "BF16", (24576, 512), "mla"),
    (_ATTN + "o_norm.weight", "F32", (128,), "kda"),
    (_ATTN + "o_proj.weight", "BF16", (7168, 12288), "mla"),
    (_ATTN + "o_proj.weight", "F8_E4M3", (7168, 12288), "kda"),
    (_ATTN + "o_proj.weight_scale", "U8", (7168, 384), "kda"),
    (_ATTN + "q_a_layernorm.weight", "BF16", (1536,), "mla"),
    (_ATTN + "q_a_proj.weight", "BF16", (1536, 7168), "mla"),
    (_ATTN + "q_b_proj.weight", "BF16", (18432, 1536), "mla"),
    (_ATTN + "q_conv1d.weight", "F32", (12288, 1, 4), "kda"),
    (_ATTN + "q_proj.weight", "F8_E4M3", (12288, 7168), "kda"),
    (_ATTN + "q_proj.weight_scale", "U8", (12288, 224), "kda"),
    (_ATTN + "v_conv1d.weight", "F32", (12288, 1, 4), "kda"),
    (_ATTN + "v_proj.weight", "F8_E4M3", (12288, 7168), "kda"),
    (_ATTN + "v_proj.weight_scale", "U8", (12288, 224), "kda"),
    ("model.norm.weight", "BF16", (7168,), None),
    ("model.output_attn_res_norm.weight", "BF16", (7168,), None),
    ("model.output_attn_res_proj.weight", "BF16", (1, 7168), None),
)
# The vision tower's and projector's tensors: only their number matters here, never their names.
_VISION = {"vision_tower.": 165, "mm_projector.": 3}
_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "U8": torch.uint8,
    "F8_E4M3": torch.float8_e4m3fn,
}
_EXPERT_KEY = re.compile(r"\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.")

# K3.1's text config, as the export's config.json has it.
_TEXT = dict(
    hidden_size=7168,
    num_hidden_layers=93,
    vocab_size=163840,
    num_attention_heads=96,
    num_key_value_heads=96,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    intermediate_size=33792,
    moe_intermediate_size=3072,
    num_experts=896,
    num_experts_per_token=16,
    num_shared_experts=2,
    first_k_dense_replace=1,
    moe_layer_freq=1,
    routed_expert_hidden_size=3584,
    attn_res_block_size=12,
    rms_norm_eps=1e-5,
    linear_attn_config={
        "full_attn_layers": [i + 1 for i in _MLA],
        "kda_layers": [i + 1 for i in _LAYERS["kda"]],
        "head_dim": 128,
        "num_heads": 96,
        "short_conv_kernel_size": 4,
    },
    dsa_config={
        "dsa_layers": [i + 1 for i in _MLA],
        "index_head_dim": 128,
        "index_k_norm_type": "layernorm",
        "index_num_heads": 32,
        "index_rope_dim": 0,
        "index_skip_topk_offset": 1,
        "index_topk": 2048,
        "index_topk_freq": 1,
    },
)


@pytest.fixture(scope="module")
def snapshot() -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Every checkpoint key -> (dtype, shape)."""
    keys = {}
    for pattern, dtype, shape, layers in _CKPT:
        spec = (_DTYPES[dtype], shape)
        template = "language_model." + pattern
        for i in _LAYERS[layers] if layers else (None,):
            key = template if i is None else template.replace("{i}", str(i))
            if "{e}" in key:
                keys.update({key.replace("{e}", str(e)): spec for e in range(_EXPERTS)})
            else:
                keys[key] = spec
    for prefix, count in _VISION.items():
        keys.update({f"{prefix}{j}": (torch.bfloat16, (1,)) for j in range(count)})
    assert len(keys) == 992_691
    return keys


def _weights(layout: str):
    """The weights module of the target that serves ``layout``."""
    target = _TARGET_OF.get(layout, layout)
    return importlib.import_module(f"{_FAMILY}.kimi_k31_pro_nvfp4__sm_107__{target}.weights")


def _src_scalar(template: str):
    """A 0-dim checkpoint tensor (``{i}`` the layer), written to element 0 of a [1] parameter."""
    return lambda d, i, t=template: [(t.format(i=i), (0,))]


def _core(layout: str, rank: int, role: str | None = None) -> SimpleNamespace:
    """``rank``'s core on a ``role`` server (default: the layout's last role, generation where the
    layout serves it)."""
    core = SimpleNamespace(
        role=role or _LAYOUTS[layout].roles[-1],
        model_config=SimpleNamespace(
            pretrained_config=KimiK31TextConfig(**_TEXT),
            mapping=Mapping(rank=rank, **copy.deepcopy(_LAYOUTS[layout].mapping)),
            torch_dtype=torch.bfloat16,
        ),
    )
    core.dims = _weights(layout).dims(core)
    return core


# --- dims ---


def _expected_share(layout: str, rank: int) -> dict:
    """The share ``dims`` gives ``rank``; one column per layout: dep16, helix8, tep8, dep12, dep8
    (None: that layout's dims has no such field). dep12's expert windows are the all-to-all kernel's
    896 = 8 x 75 + 4 x 74: ranks 0-7 hold 75 experts, ranks 8-11 hold 74."""
    dep12_local = 75 if rank < 8 else 74
    dep12_offset = 75 * rank if rank < 8 else 600 + 74 * (rank - 8)
    shares = dict(
        dp_size=(16, None, None, 12, 8),
        kda_tp=(1, 8, 8, 1, 1),
        kda_tp_rank=(0, rank, rank, 0, 0),
        kda_heads=(96, 12, 12, 96, 96),
        mla_heads=(96, 96, 12, 96, 96),
        mla_o_heads=(96, 12, 12, 96, 96),
        helix_cp=(1, 8, 1, 1, 1),
        helix_cp_rank=(0, rank, 0, 0, 0),
        moe_tp=(1, 2, 1, 1, 1),
        moe_tp_rank=(0, rank // 4, 0, 0, 0),
        moe_ep=(16, 4, 8, 12, 8),
        moe_ep_rank=(rank, rank % 4, rank, rank, rank),
        local_experts=(56, 224, 112, dep12_local, 112),
        expert_offset=(56 * rank, 224 * (rank % 4), 112 * rank, dep12_offset, 112 * rank),
        moe_inter=(3072, 1536, 3072, 3072, 3072),
        shared_inter=(6144, 768, 768, 6144, 6144),
        dense_inter=(33792, 4224, 4224, 33792, 33792),
    )
    column = ("dep16", "helix8", "tep8", "dep12", "dep8").index(layout)
    return {field: values[column] for field, values in shares.items() if values[column] is not None}


@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_dims_of_every_rank(layout):
    """Each rank's share, and the model facts every rank reads."""
    for rank in range(_LAYOUTS[layout].world):
        d = _core(layout, rank).dims
        want = _expected_share(layout, rank)
        assert {field: getattr(d, field) for field in want} == want, rank
        assert d.role == _LAYOUTS[layout].roles[-1]
        assert (d.num_layers, d.hidden, d.vocab, d.dtype) == (93, 7168, 163840, torch.bfloat16)
        assert d.mla_layers == _MLA and d.kda_layers == _LAYERS["kda"]
        assert d.moe_layers == _LAYERS["moe"] and d.dense_layers == (0,)
        assert (d.index_heads, d.index_dim) == (32, 128)
        assert (d.num_experts, d.latent, d.attn_res_block) == (896, 3584, 12)
        assert (d.q_lora, d.kv_lora, d.nope, d.rope, d.v_dim) == (1536, 512, 128, 64, 128)
        assert (d.kda_head_dim, d.kda_conv) == (128, 4)


@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_shares_tile_the_model(layout):
    """Over the ranks, every routed expert is held moe_tp times, once per MoE TP slice; the KDA
    heads and the MLP intermediates split into whole, disjoint parts."""
    world = _LAYOUTS[layout].world
    dims = [_core(layout, rank).dims for rank in range(world)]
    holders = Counter()
    for d in dims:
        for e in range(d.expert_offset, d.expert_offset + d.local_experts):
            holders[(e, d.moe_tp_rank)] += 1
    d0 = dims[0]
    assert set(holders) == {(e, t) for e in range(_EXPERTS) for t in range(d0.moe_tp)}
    assert set(holders.values()) == {world // (d0.moe_tp * d0.moe_ep)}
    assert d0.moe_inter * d0.moe_tp == 3072
    tp_ranks = Counter(d.kda_tp_rank for d in dims)
    assert sorted(tp_ranks) == list(range(d0.kda_tp))
    assert d0.kda_heads * d0.kda_tp == 96 and d0.mla_o_heads * d0.kda_tp == 96
    assert d0.shared_inter * d0.kda_tp == 6144 and d0.dense_inter * d0.kda_tp == 33792


@pytest.mark.parametrize(
    "layout, rows",
    [
        (layout, rows)
        for layout in _LAYOUTS
        for rows in ("whole", "tp")
        # Under KDA TP 1 (dep16) the vocab-parallel cut is the whole vocabulary.
        if rows == "whole" or _expected_share(layout, 0)["kda_tp"] > 1
    ],
)
def test_lm_head_rows_tile_the_vocab(layout, rows):
    """The LM head's cuts, whole or vocab-parallel over the TP group, cover the vocabulary."""
    W = _weights(layout)
    d = _core(layout, 0).dims
    n = d.vocab if rows == "whole" else d.vocab // d.kda_tp
    covered = Counter()
    for rank in range(_LAYOUTS[layout].world):
        (cut,), _ = W.lm_head_share(n, d.vocab, _core(layout, rank).dims.kda_tp_rank)
        assert cut.stop - cut.start == n
        covered.update(range(cut.start, cut.stop, 4096))
    assert sorted(covered) == list(range(0, d.vocab, 4096))


# --- the machinery, on a small synthetic table ---


class _LazySlice:
    """Stands in for a safetensors slice: indexing it is the read, and every read is recorded."""

    def __init__(self, key: str, full: torch.Tensor, reads: list):
        self.key, self._full, self._reads = key, full, reads

    def get_shape(self) -> list[int]:
        return list(self._full.shape)

    def __getitem__(self, index):
        out = self._full[index]
        self._reads.append((self.key, out.numel()))
        return out


def _tiny_dims(W):
    d = W.Dims(*([0] * len(W.Dims._fields)))
    return d._replace(
        num_layers=2, hidden=4, vocab=6, dtype=torch.bfloat16, kda_tp=2, kda_tp_rank=1
    )


def _cat(t: tuple[torch.Tensor, ...], core) -> torch.Tensor:
    return torch.cat(t, dim=0)


def _tiny_rows(W) -> tuple:
    return (
        W.W("norm", shape=lambda d: (4,), src="t.layers.{i}.norm"),
        W.W(
            "proj",
            shape=lambda d: (2, 4),
            src="t.layers.{i}.proj",
            cut=W.cut_rows(lambda d: (d.kda_tp_rank, 2)),
        ),
        W.W(
            "pair",
            shape=lambda d: (2, 4),
            src=W.src_keys("t.layers.{i}.a", "t.layers.{i}.b"),
            transform=_cat,
            layers=lambda d: [0],
        ),
        W.W(
            "scale",
            shape=lambda d: (1,),
            dtype=torch.float32,
            src=_src_scalar("t.layers.{i}.s"),
            layers=lambda d: [1],
        ),
        W.W("derived", shape=lambda d: (3,), src=None, layers=None),
        W.W("expert", shape=lambda d: (2, 4), src="t.layers.{i}.h", layers=lambda d: [1]),
    )


def _tiny_checkpoint(reads: list) -> dict:
    g = torch.Generator().manual_seed(0)

    def bf16(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g).bfloat16()

    full = {
        "t.layers.0.norm": bf16(4),
        "t.layers.1.norm": bf16(4),
        "t.layers.0.proj": bf16(4, 4),
        "t.layers.1.proj": bf16(4, 4),
        "t.layers.0.a": bf16(1, 4),
        "t.layers.0.b": bf16(1, 4),
        "t.layers.1.s": torch.tensor(0.5),
        "t.layers.1.h": bf16(2, 4),
    }
    return {key: _LazySlice(key, t, reads) for key, t in full.items()}


class _TinyCore(nn.Module):
    def __init__(self, W, rows):
        super().__init__()
        self.dims = _tiny_dims(W)
        self.model_config = SimpleNamespace()
        self.w = W.declare(self.dims, rows)
        for p in self.w.values():
            p.data.zero_()


@pytest.mark.parametrize("layout", _OWN_TABLES)
def test_declare_and_manifest_agree(layout):
    """``declare`` allocates the core.w rows; ``manifest`` names them as ``named_parameters`` does,
    a derived parameter without sources."""
    W = _weights(layout)
    rows = _tiny_rows(W)
    core = _TinyCore(W, rows)
    assert set(core.w) == {
        "l0_norm",
        "l1_norm",
        "l0_proj",
        "l1_proj",
        "l0_pair",
        "l1_scale",
        "derived",
        "l1_expert",
    }
    mf = W.manifest(core, rows)
    assert set(mf) == {n for n, _ in core.named_parameters()}
    assert mf["w.derived"] == []
    assert mf["w.l1_expert"] == [("t.layers.1.h", None, None, None)]
    assert mf["w.l1_proj"] == [("t.layers.1.proj", None, (slice(2, 4),), None)]
    assert mf["w.l1_scale"][0][:2] == ("t.layers.1.s", (0,))
    with pytest.raises(AssertionError, match="two rows fill w.l0_norm"):
        W.manifest(core, rows + rows[:1])


@pytest.mark.parametrize("layout", _OWN_TABLES)
def test_read_takes_only_the_cut(layout):
    W = _weights(layout)
    reads = []
    ckpt = _tiny_checkpoint(reads)
    core = _TinyCore(W, _tiny_rows(W))
    got = W.read(ckpt, "t.layers.1.proj", (slice(2, 4),), None, core)
    assert torch.equal(got, ckpt["t.layers.1.proj"]._full[2:4])
    assert reads == [("t.layers.1.proj", 8)]
    assert W.read(ckpt, "t.layers.1.s", None, None, core).shape == ()
    with pytest.raises(KeyError, match="checkpoint key missing: t.nope"):
        W.read(ckpt, "t.nope", None, None, core)


@pytest.mark.parametrize("layout", _OWN_TABLES)
def test_fill_table_fills_every_parameter_once(layout):
    W = _weights(layout)
    rows = _tiny_rows(W)
    reads = []
    ckpt = _tiny_checkpoint(reads)
    core = _TinyCore(W, rows)
    consumed = W.fill_table(core, ckpt, rows, dict(core.named_parameters()))
    full = {k: s._full for k, s in ckpt.items()}
    assert consumed == set(full)
    assert torch.equal(core.w["l1_proj"], full["t.layers.1.proj"][2:4])
    assert torch.equal(core.w["l0_pair"], torch.cat([full["t.layers.0.a"], full["t.layers.0.b"]]))
    assert core.w["l1_scale"].item() == 0.5
    assert torch.equal(core.w["l1_expert"], full["t.layers.1.h"])
    assert not core.w["derived"].any()
    # Only the cut leaves the checkpoint, and each key once.
    assert dict(Counter(k for k, _ in reads)) == {k: 1 for k in full}
    assert dict(reads)["t.layers.0.proj"] == 8


@pytest.mark.parametrize("layout", _OWN_TABLES)
def test_fill_table_names_what_it_misses(layout):
    W = _weights(layout)
    rows = _tiny_rows(W)
    core = _TinyCore(W, rows)
    params = dict(core.named_parameters())
    with pytest.raises(AssertionError, match=r"parameters no row fills: \['w.extra'\]"):
        W.fill_table(core, _tiny_checkpoint([]), rows, {**params, "w.extra": params["w.derived"]})
    # The scalar declared in the model dtype: the FP32 source does not fit it.
    scale = W.W(
        "scale", shape=lambda d: (1,), src=_src_scalar("t.layers.{i}.s"), layers=lambda d: [1]
    )
    bad = (*rows[:3], scale, *rows[4:])
    bad_core = _TinyCore(W, bad)
    with pytest.raises(AssertionError, match="float32"):
        W.fill_table(bad_core, _tiny_checkpoint([]), bad, dict(bad_core.named_parameters()))
    gap = W.W(
        "slots",
        shape=lambda d: (3, 4),
        src=lambda d, i: [("t.layers.0.norm", (0,)), ("t.layers.1.norm", (2,))],
        layers=None,
    )
    core.w["slots"] = nn.Parameter(torch.zeros(3, 4, dtype=torch.bfloat16), requires_grad=False)
    with pytest.raises(AssertionError, match="slots"):
        W.fill_table(core, _tiny_checkpoint([]), (*rows, gap), dict(core.named_parameters()))


class _Shell(nn.Module):
    def __init__(self, core, vocab_rows: int):
        super().__init__()
        self.model = core
        self.lm_head = nn.Linear(4, vocab_rows, bias=False, dtype=torch.bfloat16)
        self.draft_model = nn.Module()  # the drafter: no shell parameter (w3_rows fills core.w)


def _load_setup(W, monkeypatch):
    rows = _tiny_rows(W)
    core = _TinyCore(W, rows)
    core.w["d_proj"] = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16), requires_grad=False)
    monkeypatch.setattr(W, "w1_rows", lambda d: rows[:5])
    monkeypatch.setattr(W, "w2_rows", lambda d: rows[5:])
    monkeypatch.setattr(W, "w2_offwindow_keys", lambda core, weights: {"t.layers.1.experts.9"})
    monkeypatch.setattr(W, "modelopt_nvfp4_mismatch", lambda model_config: None)
    reads = []
    ckpt = _tiny_checkpoint(reads)
    head = torch.randn(6, 4).bfloat16()
    ckpt[W.CKPT_LM_HEAD] = _LazySlice(W.CKPT_LM_HEAD, head, reads)
    for key in ("vision_tower.blocks.0.weight", "mm_projector.proj.weight", "t.layers.1.experts.9"):
        ckpt[key] = _LazySlice(key, torch.zeros(1), reads)
    return core, ckpt, head


@pytest.mark.parametrize("layout", _OWN_TABLES)
def test_load_covers_both_ways(layout, monkeypatch):
    """The shell's LM head takes its vocab-parallel rows; the drafter's parameters are left to
    ``w3_rows``; the vision tower and the other ranks' experts are the only leftovers."""
    W = _weights(layout)
    core, ckpt, head = _load_setup(W, monkeypatch)
    model = _Shell(core, vocab_rows=3)
    W.load(model, ckpt)
    assert torch.equal(model.lm_head.weight, head[3:6])  # kda_tp_rank 1 of 2
    assert not core.w["d_proj"].any()


@pytest.mark.parametrize("layout", _OWN_TABLES)
def test_load_names_unexpected_leftovers(layout, monkeypatch):
    W = _weights(layout)
    core, ckpt, _ = _load_setup(W, monkeypatch)
    ckpt["t.surprise"] = _LazySlice("t.surprise", torch.zeros(1), [])
    with pytest.raises(AssertionError, match=r"unexpected leftovers \['t.surprise'\]"):
        W.load(_Shell(core, vocab_rows=3), ckpt)


@pytest.mark.parametrize("layout", _OWN_TABLES)
def test_load_refuses_another_checkpoint(layout, monkeypatch):
    W = _weights(layout)
    core, ckpt, _ = _load_setup(W, monkeypatch)
    monkeypatch.setattr(W, "modelopt_nvfp4_mismatch", lambda model_config: "not the export")
    with pytest.raises(
        ValueError, match="reads the ModelOpt NVFP4 export as stored: not the export"
    ):
        W.load(_Shell(core, vocab_rows=3), ckpt)


@pytest.mark.parametrize("layout", _OWN_TABLES)
def test_open_checkpoint_weights_reads_lazily(layout, tmp_path):
    W = _weights(layout)
    a = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    save_file({"x.a": a}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({"x.s": torch.tensor(2.0)}, str(tmp_path / "model-00002-of-00002.safetensors"))
    with W.open_checkpoint_weights(str(tmp_path)) as weights:
        assert sorted(weights) == ["x.a", "x.s"]
        assert weights.checkpoint_dir == str(tmp_path)
        assert torch.equal(W.read(weights, "x.a", (slice(2, 4),), None, None), a[2:4])
        assert W.read(weights, "x.s", None, None, None).item() == 2.0
    assert len(weights) == 0
    with pytest.raises(FileNotFoundError):
        with W.open_checkpoint_weights(str(tmp_path / "empty")):
            pass


# --- the target's table against the export ---


def _keys(source) -> tuple[str, ...]:
    return source if isinstance(source, tuple) else (source,)


def _key_cuts(W, source, cut) -> list:
    """``[(checkpoint key, its cut), ...]``: a ``Cuts`` gives each key of a key tuple its own."""
    keys = _keys(source)
    cuts = cut if isinstance(cut, W.Cuts) else (cut,) * len(keys)
    assert len(cuts) == len(keys), (source, cut)
    return list(zip(keys, cuts))


def _cut_shape(shape: tuple[int, ...], cut) -> tuple[int, ...]:
    """The shape a cut reads; asserts it is in bounds and not empty."""
    out = list(shape)
    for axis, s in enumerate(cut or ()):
        start, stop, step = s.indices(shape[axis])
        assert step == 1 and 0 <= start < stop and (s.stop is None or s.stop <= shape[axis]), (
            shape,
            cut,
        )
        out[axis] = stop - start
    return tuple(out)


def _target_rows(W, core) -> tuple:
    return (*W.w1_rows(core.dims), *W.w2_rows(core.dims))


def _declared(W, core, rows) -> dict:
    """Destination name -> (shape, dtype) the rows declare."""
    d = core.dims
    return {
        name: (tuple(row.shape(d)), row.dtype or d.dtype)
        for row in rows
        for name, _ in W._destinations(row, d)
    }


# The first, a middle and the last rank of each layout, and dep12's last 75-expert and first
# 74-expert ranks.
_RANKS = [
    (layout, rank)
    for layout, spec in _LAYOUTS.items()
    for rank in sorted(
        {0, spec.world // 2 - 1, spec.world - 1} | ({7, 8} if layout == "dep12" else set())
    )
]


@pytest.mark.parametrize("layout, rank", _RANKS)
def test_table_covers_the_checkpoint(snapshot, layout, rank):
    """Every source key exists, and every key is read or one of the leftovers this rank predicts."""
    W = _weights(layout)
    core = _core(layout, rank)
    consumed = {W.CKPT_LM_HEAD}
    for name, sources in W.manifest(core, _target_rows(W, core)).items():
        for key, _, _, _ in sources:
            missing = [k for k in _keys(key) if k not in snapshot]
            assert not missing, (name, missing)
            consumed.update(_keys(key))
    leftover = set(snapshot) - consumed
    expected = W.expected_unconsumed(core, snapshot)
    assert leftover == expected, (sorted(leftover - expected)[:5], sorted(expected - leftover)[:5])


@pytest.mark.parametrize("layout, rank", _RANKS)
def test_every_source_fills_its_slot(snapshot, layout, rank):
    """Each cut is in bounds, and the relayout of the cut, run on meta tensors, has the shape and
    dtype its destination declares (the first and last source of each destination)."""
    W = _weights(layout)
    core = _core(layout, rank)
    rows = _target_rows(W, core)
    declared = _declared(W, core, rows)
    for name, sources in W.manifest(core, rows).items():
        shape, dtype = declared[name]
        for key, index, cut, transform in sources[:1] + sources[-1:]:
            src = tuple(
                torch.empty(_cut_shape(snapshot[k][1], c), dtype=snapshot[k][0], device="meta")
                for k, c in _key_cuts(W, key, cut)
            )
            if transform is None:
                out = src[0]
            else:
                out = transform(src if isinstance(key, tuple) else src[0], core)
            slot = shape if index is None else shape[len(index) :]
            assert (tuple(out.shape), out.dtype) == (slot, dtype), (name, key)


@pytest.mark.parametrize("layout, rank", _RANKS)
def test_rank_reads_its_share_once(snapshot, layout, rank):
    """A routed expert's weights and block scales are read at this rank's MoE TP slice when the
    expert is in its window and never otherwise; no other tensor is read past its size; and the
    parameters hold what was read (within 1%: casts, scale padding, derived parameters)."""
    W = _weights(layout)
    core = _core(layout, rank)
    d = core.dims
    rows = _target_rows(W, core)
    read = Counter()
    for sources in W.manifest(core, rows).values():
        for key, _, cut, _ in sources:
            for k, c in _key_cuts(W, key, cut):
                read[k] += math.prod(_cut_shape(snapshot[k][1], c))
    lo, hi = d.expert_offset, d.expert_offset + d.local_experts
    for key, (_, shape) in snapshot.items():
        full = math.prod(shape)
        expert = _EXPERT_KEY.search(key)
        if expert and shape:
            want = full // d.moe_tp if lo <= int(expert.group(2)) < hi else 0
            assert read[key] == want, (key, read[key], want)
        elif shape:
            assert read[key] <= full, (key, read[key], full)
    read_bytes = sum(n * snapshot[k][0].itemsize for k, n in read.items())
    param_bytes = sum(
        math.prod(shape) * dtype.itemsize for shape, dtype in _declared(W, core, rows).values()
    )
    assert abs(param_bytes - read_bytes) <= 0.01 * read_bytes, (param_bytes, read_bytes)


# The export pads A_log from the 96 KDA heads to 128 entries (zeros); the tail may stay unread.
_PADDED = {".self_attn.A_log": 96}


@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_cuts_tile_across_ranks(snapshot, layout):
    """Over all ranks, every element along a cut axis of a tensor is read equally often: the
    replicated tensors by every rank, the split ones once per share. A cut shifted onto the
    neighbouring share stays in bounds but breaks this."""
    W = _weights(layout)
    coverage: dict[tuple[str, int], torch.Tensor] = {}
    for rank in range(_LAYOUTS[layout].world):
        core = _core(layout, rank)
        for sources in W.manifest(core, _target_rows(W, core)).values():
            for key, _, cut, _ in sources:
                for k, c in _key_cuts(W, key, cut):
                    shape = snapshot[k][1]
                    for axis, s in enumerate(c or (slice(None),) * min(len(shape), 1)):
                        counts = coverage.setdefault(
                            (k, axis), torch.zeros(shape[axis], dtype=torch.int32)
                        )
                        counts[s] += 1
    uneven = []
    for (key, axis), counts in coverage.items():
        real = next((n for suffix, n in _PADDED.items() if key.endswith(suffix)), None)
        counts = counts[:real] if real is not None and axis == 0 else counts
        if int(counts.min()) != int(counts.max()):
            uneven.append((key, axis))
    assert not uneven, uneven[:5]


def _expert_rows(d, megamoe: bool) -> dict:
    """The routed-expert parameters the MoE block reads, by core.w name: (shape, dtype). The
    expert entries take them as stored: FC1 [E, 2I, latent / 2] (16 gate rows, then 16 up rows),
    its swizzled block scales [E, 2I, latent / 16], FC2 [E, latent, I / 2] and its
    [E, latent, I / 16]; the FP32 input scales and per-expert alphas."""
    e, i, h = d.local_experts, d.moe_inter, d.latent
    rows = {
        "moe_fc1_w": ((e, 2 * i, h // 2), torch.uint8),
        "moe_fc1_s": ((e, 2 * i, h // 16), torch.uint8),
        "moe_fc2_w": ((e, h, i // 2), torch.uint8),
        "moe_fc2_s": ((e, h, i // 16), torch.uint8),
        "moe_fc31_in_scale": ((1,), torch.float32),
        "moe_fc2_in_scale": ((1,), torch.float32),
        "moe_fc31_alpha": ((e,), torch.float32),
        "moe_fc2_alpha": ((e,), torch.float32),
    }
    if megamoe:
        # MegaMoE's FC2 input scale per expert (the layer's: one input_scale per layer).
        rows["moe_fc1_norm_const"] = ((e,), torch.float32)
    return {f"w.l{layer}_{name}": spec for layer in d.moe_layers for name, spec in rows.items()}


@pytest.mark.parametrize(
    "layout, role", [(layout, role) for layout, spec in _LAYOUTS.items() for role in spec.roles]
)
def test_routed_expert_rows_are_core_w_rows(layout, role):
    """The routed rows of ``w2_rows`` are exactly the core.w parameters the MoE block reads (0
    missing, 0 extra), by global layer, at the shapes and dtypes the expert entries take; every
    role the same."""
    W, spec = _weights(layout), _LAYOUTS[layout]
    core = _core(layout, spec.world - 1, role)
    d = core.dims
    got = {n: s for n, s in _declared(W, core, W.w2_rows(d)).items() if "_moe_shared_" not in n}
    want = _expert_rows(d, spec.megamoe)
    missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
    assert not missing and not extra, (missing[:4], extra[:4])
    assert got == want
    shared = {n for n in _declared(W, core, W.w2_rows(d)) if "_moe_shared_" in n}
    assert len(shared) == 4 * len(d.moe_layers)
    other = _core(layout, spec.world - 1, spec.roles[0]).dims
    assert {r.name for r in W.w2_rows(other)} == {r.name for r in W.w2_rows(d)}
    if spec.megamoe:
        from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4MegaMoECuteDslMethod

        # MegaMoE's per-slot block-scale buffers are these rows' bytes, flattened.
        i, h = d.moe_inter, d.latent
        assert NVFP4MegaMoECuteDslMethod.fc1_sf_flat_size(i, h) == 2 * i * (h // 16)
        assert NVFP4MegaMoECuteDslMethod.fc2_sf_flat_size(h, i) == h * (i // 16)


# --- the routed-expert windows against the all-to-all kernel's split ---


@pytest.mark.parametrize("layout", ["dep16", "dep12", "dep8"])
def test_expert_windows_are_the_all_to_alls(layout):
    """Every rank holds exactly the experts the context server's all-to-all dispatch sends it, and
    the windows tile the 896 experts in rank order."""
    world = _LAYOUTS[layout].world
    owner = [kernel_target_rank(e, _EXPERTS, world) for e in range(_EXPERTS)]
    start = 0
    for rank in range(world):
        d = _core(layout, rank).dims
        window = range(d.expert_offset, d.expert_offset + d.local_experts)
        assert d.expert_offset == start, (rank, d.expert_offset, start)
        assert [e for e in range(_EXPERTS) if owner[e] == rank] == list(window), rank
        start = window.stop
    assert start == _EXPERTS


def test_dep8_windows_are_8_x_112():
    windows = [
        (d.expert_offset, d.local_experts) for d in (_core("dep8", rank).dims for rank in range(8))
    ]
    assert windows == [(112 * r, 112) for r in range(8)]


def test_dep8_is_a_context_split_only():
    W = _weights("dep8")
    d = _core("dep8", 7).dims
    W.w2_rows(d)
    with pytest.raises(ValueError, match=re.escape("(1, 8) on a generation server")):
        W.w2_rows(d._replace(role="generation"))


def test_dep12_windows_are_8_x_75_and_4_x_74():
    windows = [
        (d.expert_offset, d.local_experts)
        for d in (_core("dep12", rank).dims for rank in range(12))
    ]
    assert windows == [(75 * r, 75) for r in range(8)] + [(600 + 74 * r, 74) for r in range(4)]


_SENT_TO_RANK_3 = "the all-to-all sends EP rank 3 the experts (first, count) (225, 75)"


@pytest.mark.parametrize(
    "changes, named",
    [
        (dict(moe_ep=4), "routed split (moe_tp, moe_ep) (1, 4) on a context server, not one of"),
        # EP8 is a context server's split, but rank 3's EP8 window is not its EP12 one.
        (
            dict(moe_ep=8),
            "(225, 75), but the all-to-all sends EP rank 3 the experts (first, count) (336, 112)",
        ),
        (dict(moe_tp=2), "routed split (moe_tp, moe_ep) (2, 12) on a context server, not one of"),
        # EP12 is a context server's split only.
        (
            dict(role="generation"),
            "routed split (moe_tp, moe_ep) (1, 12) on a generation server, not one of ((1, 16),)",
        ),
        (dict(role="draft"), "role 'draft', not one of ('context', 'generation')"),
        # The even split's window on a dep12 rank: not where the dispatch sends rank 3's rows.
        (dict(expert_offset=3 * 74, local_experts=74), f"(222, 74), but {_SENT_TO_RANK_3}"),
        (dict(local_experts=74), f"(225, 74), but {_SENT_TO_RANK_3}"),
    ],
    ids=[
        "ep4",
        "ep8-window",
        "moe-tp2",
        "ep12-generation",
        "other-role",
        "even-window",
        "short-window",
    ],
)
def test_expert_check_refuses_other_splits_and_windows(changes, named):
    W = _weights("dep12")
    d = _core("dep12", 3).dims
    W.w2_rows(d)  # the window routing builds passes
    with pytest.raises(ValueError, match=re.escape(named)):
        W.w2_rows(d._replace(**changes))


@pytest.mark.parametrize("rank", [0, 15])
def test_expert_check_takes_ep16_on_both_roles(rank):
    W = _weights("dep16")
    for role in ("context", "generation"):
        W.w2_rows(_core("dep16", rank, role).dims)


def test_expert_check_does_not_trust_the_window_dims_computes(monkeypatch):
    """The check holds the window to the dispatch's owner rule, written apart from the window
    rule ``dims`` uses: a wrong window rule (the even split's floor, 74 experts from 74 r) fails
    the table build instead of passing its own comparison."""
    W = _weights("dep12")
    monkeypatch.setattr(W, "a2a_expert_window", lambda e, ep, r: (r * (e // ep), e // ep))
    d = _core("dep12", 3, "context").dims
    assert (d.expert_offset, d.local_experts) == (222, 74)
    with pytest.raises(ValueError, match=re.escape(f"(222, 74), but {_SENT_TO_RANK_3}")):
        W.w2_rows(d)


@pytest.mark.parametrize("rank", [0, 7, 8, 11])
def test_dep12_expert_rows_read_their_window(rank):
    """A dep12 rank's routed rows: 75 or 74 experts at the export's shapes, filled from its window's
    checkpoint experts in order; every other expert's weights are leftovers, its input_scale read."""
    W = _weights("dep12")
    core = _core("dep12", rank)
    d = core.dims
    lo, n = d.expert_offset, d.local_experts
    assert n == (75 if rank < 8 else 74)
    declared = _declared(W, core, W.w2_rows(d))
    assert declared["w.l1_moe_fc1_w"] == ((n, 6144, 1792), torch.uint8)
    assert declared["w.l1_moe_fc1_s"] == ((n, 6144, 224), torch.uint8)
    assert declared["w.l92_moe_fc2_w"] == ((n, 3584, 1536), torch.uint8)
    assert declared["w.l92_moe_fc2_s"] == ((n, 3584, 192), torch.uint8)
    assert declared["w.l1_moe_fc31_alpha"] == ((n,), torch.float32)
    sources = W.manifest(core, W.w2_rows(d))["w.l1_moe_fc1_w"]

    def key(e: int, part: str) -> str:
        return f"language_model.model.layers.1.block_sparse_moe.experts.{e}.{part}"

    want = [(key(e, "w1.weight"), key(e, "w3.weight")) for e in range(lo, lo + n)]
    assert [k for k, _, _, _ in sources] == want
    assert [index for _, index, _, _ in sources] == [(j,) for j in range(n)]
    off = W.w2_offwindow_keys(core, None)
    assert len(off) == len(d.moe_layers) * (_EXPERTS - n) * 3 * 3
    assert key(lo, "w1.weight") not in off and key(lo + n - 1, "w2.weight_scale") not in off
    other = lo + n if lo + n < _EXPERTS else lo - 1
    assert key(other, "w1.weight") in off and key(other, "w1.input_scale") not in off


# --- dense DWDP on a dep8 context server ---

#: A generation server's MXFP8 lever rows (``MXFP8_MLA`` / ``MXFP8_MOE``), which the dep16 target
#: names for dense DWDP too; a context server's table does not hold them.
_MX_LEVER_NAMES = frozenset(
    f"{n}_{part}" for n in ("mla_g", "mla_o", "moe_latent_down") for part in "ws"
)
#: The engine's size threshold (``DEFAULT_MIN_BYTES``): smaller storages stay in place.
_DENSE_MIN_BYTES = 1 << 20


@pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="the dep16 modeling imports the CuTe DSL"
)
def test_dep8_dense_dwdp_names_are_rows_of_the_context_table():
    """Every weight the dep16 target names for dense DWDP is a row of its table on a dep8 context
    server, but a generation server's MXFP8 lever rows; the bytes it moves are the replicated
    dense weights of the 92 MoE layers (about 61 GiB per rank, of which dense DWDP keeps one
    row in eight), and no routed expert."""
    mod = importlib.import_module(f"{_FAMILY}.kimi_k31_pro_nvfp4__sm_107__dep16.modeling")
    W = _weights("dep8")
    core = _core("dep8", 0, "context")
    d = core.dims
    declared = _declared(W, core, (*W.w1_rows(d), *W.w2_rows(d)))
    moved, absent = 0, set()
    for i in d.moe_layers:
        attn = mod._DENSE_ATTN_KDA if i in d.kda_layers else mod._DENSE_ATTN_MLA
        for name in (*attn, *mod._DENSE_MOE):
            assert "_fc" not in name and not name.endswith("router_bias"), name
            if f"w.l{i}_{name}" not in declared:
                absent.add(name)
                continue
            shape, dtype = declared[f"w.l{i}_{name}"]
            nbytes = math.prod(shape) * dtype.itemsize
            if nbytes >= _DENSE_MIN_BYTES:
                moved += nbytes
    assert absent == _MX_LEVER_NAMES
    assert 60 << 30 < moved < 63 << 30, moved
