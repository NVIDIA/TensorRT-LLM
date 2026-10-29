# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Mistral Large 4 NVFP4 tp4 weight table against the checkpoint it reads, at tp4 (four ranks on
one node) and at tp8 (eight ranks over two nodes of four).

No GPU. Against the tep4 table only the routed experts move: every rank holds all 128 experts of
a layer, each cut along its intermediate width to rank r's I -- `w1`/`w3` codes and block scales
by output rows `[Ir, Ir + I)`, `w2`'s codes by input bytes `[Ir/2, (I/2)(r + 1))` and its block
scales by columns `[Ir/16, (I/16)(r + 1))` -- and the MoE runner's layout is applied to that
slice. I is 1536 at tp4 (768 code bytes, 96 scale columns) and 768 at tp8 (384 and 48). Four
kinds of evidence, at each world size:

* the table at the checkpoint's own language-model configuration (written out below, and
  compared with the configuration adapted from the checkpoint's `params.json` when it is
  present) on each rank of the engine's mapping (tp and moe_tp at the world size, moe_ep 1, four
  ranks per node): which layers carry which weight at what per-rank shape, that every rank's
  manifest reads every expert through its width slice, and that the ranks' slices tile each
  expert exactly once;
* the checkpoint itself, through its index and safetensors headers, as meta tensors, plus the
  values of its small scale tensors: every rank reads all 1,721 language-model keys, all
  109,056 expert keys and all 276 vision-tower keys, each at the per-rank shape and dtype
  `_PER_RANK` states, 157,212,207,424 bytes (146.42 GiB) at tp4 and 82,442,488,768 bytes
  (76.78 GiB) at tp8 -- the language model's 153,803,770,176 and 79,034,051,520 and the vision
  tower's 3,408,437,248, whole -- which is also what the production `load` itself reads; the
  index `load` takes of every key on every rank -- the ranks' head, row, column and vocabulary
  ranges and expert width slices tile each sliced tensor exactly once, everything else is read
  whole; the production `load` -- every slice, transform, shape and dtype assert, the
  expert-scalar asserts on the checkpoint's own scalars and the two-way coverage assert, the
  DSpark draft its only known-not-loaded set -- passes on each rank; every FP8 block scale and
  every expert scalar satisfies the invariants `load` asserts;
  and the packed UE8M0 scale operand at each K a rank's slice gives (at tp8 `wo` at 1024, the
  dense `w2` at 3072, the shared expert's `w2` at 768: 6 blocks, a half-padding second word)
  equals an independent byte-level packing and the engine's `transform_sf_into_required_layout`;
* the checkpoint's bytes for sampled experts of layers 1, 35 and 71: the production `load` puts
  each rank's slice of them, in the runner layout, exactly where the stock TRTLLM-Gen NVFP4 MoE
  preparation at moe_tp_size 4 or 8 puts it (its own slicing, permutations and scale
  interleave, run on CPU), and where the whole stored expert, cut here and laid out by this
  file's own construction, puts it; the ranks' slices, read through the production shards,
  reassemble the stored tensors; and the whole expert laid out before slicing differs wherever
  the layout depends on the slice (the negative control). Skipped where the checkpoint is
  absent (looked up as `mistral-large-4-nvfp4` or `Mistral-Large-4-1T-A52B-NVFP4` under
  `LLM_MODELS_ROOT`);
* a scaled-down configuration with real CPU tensors, loaded on every rank and compared element
  for element with an independent construction of what each parameter must hold: the slices,
  the `wkv_b` regroup, both half orders, the packed scale operand padding included (at tp8 the
  dense `w2` at a 6-block K here too), every expert's width slice in the MoE runner's layout and
  its scalars, and the DSpark draft tensors left alone; plus what the streaming path reads off
  the shard files -- each expert code and block scale as this rank's slice of the entry, never
  the whole, and each file's cached pages dropped once read -- and each assert's refusal.
"""

from __future__ import annotations

import contextlib
import json
import re
import struct
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from test_common.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tp4 import (  # noqa: E501
    weights as W,
)
from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader
from tensorrt_llm._torch.modules.linear import TensorParallelMode, load_weight_shard
from tensorrt_llm._torch.moe.fused_moe.quantization import (
    NVFP4TRTLLMGenFusedMoEMethod,
    trtllmgen_maybe_get_cached_w2_permute_indices,
    trtllmgen_maybe_get_cached_w3_w1_permute_indices,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.utils.fp8_utils import (
    check_sf_layout,
    inverse_transform_sf,
    transform_sf_into_required_layout,
)

pytestmark = pytest.mark.cpu_only

#: The world sizes: tp4, four ranks on one node, and tp8, eight ranks over two.
WORLDS = (4, 8)
#: GB300 nodes carry four devices.
GPUS_PER_NODE = 4
CHECKPOINTS = ("mistral-large-4-nvfp4", "Mistral-Large-4-1T-A52B-NVFP4")

_LAYERS = 72
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
_DTYPE_NAMES = {v: k for k, v in _SAFETENSORS_DTYPES.items()}

_PROJECTIONS = ("w1", "w2", "w3")
_EXPERT_PARAMS = ("fc1_w", "fc1_s", "fc2_w", "fc2_s", "isc1", "isc2", "ws2_13", "ws2_2")
#: The expert parameters the width slice cuts and the runner layout rearranges.
_LAID_OUT = ("fc1_w", "fc1_s", "fc2_w", "fc2_s")
_EXPERT_SUFFIXES = {
    f"{proj}.{s}"
    for proj in _PROJECTIONS
    for s in ("weight", "weight_scale", "weight_scale_2", "input_scale")
}

#: Per world size, what one rank holds of each tensor-parallel width: query heads (of 64), the
#: dense MLP's intermediate (of 24576), the shared expert's (of 6144), each routed expert's slice
#: (of 6144) and lm_head's vocabulary rows (of 204800).
_GEOMETRY = {
    4: SimpleNamespace(heads=16, dense=6144, shared=1536, inter=1536, vocab=51200),
    8: SimpleNamespace(heads=8, dense=3072, shared=768, inter=768, vocab=25600),
}

#: Each sliced expert parameter's source as stored (`[out, in/2]` codes, `[out, in/16]` block
#: scales), the axis rank r's slice cuts and its width there. tp4: w1/w3 rows [1536r, + 1536);
#: w2's code bytes [768r, + 768) and block-scale columns [96r, + 96). tp8: w1/w3 rows
#: [768r, + 768); w2's code bytes [384r, + 384) and block-scale columns [48r, + 48).
_SLICED = {
    4: {
        "fc1_w": ((6144, 3072), 0, 1536),
        "fc1_s": ((6144, 384), 0, 1536),
        "fc2_w": ((6144, 3072), 1, 768),
        "fc2_s": ((6144, 384), 1, 96),
    },
    8: {
        "fc1_w": ((6144, 3072), 0, 768),
        "fc1_s": ((6144, 384), 0, 768),
        "fc2_w": ((6144, 3072), 1, 384),
        "fc2_s": ((6144, 384), 1, 48),
    },
}

#: Every text key a rank reads, by `(key with layer N and expert E, stored dtype, stored shape,
#: the shape this rank moves)`, with its count; identical on the ranks of a world size. Written
#: from the safetensors headers and the sharding the module docstring states, not from the
#: table's code. 1,721 language-model keys sharded exactly as at tep4 or tep8 (`output.weight` by
#: the shell's vocab-parallel lm_head) and 71 x 128 x 3 x 4 = 109,056 expert keys: the codes and
#: block scales width-sliced, the per-tensor scalars whole. The keys every rank moves whole at
#: either size:
_PER_RANK_WHOLE = {
    ("layers.N.attention.indexer.k_norm.bias", "BF16", (128,), (128,)): 44,
    ("layers.N.attention.indexer.k_norm.weight", "BF16", (128,), (128,)): 44,
    ("layers.N.attention.indexer.q_input_norm.bias", "BF16", (6144,), (6144,)): 44,
    ("layers.N.attention.indexer.q_input_norm.weight", "BF16", (6144,), (6144,)): 44,
    ("layers.N.attention.indexer.ssmax_scaling.b", "BF16", (32,), (32,)): 44,
    ("layers.N.attention.indexer.ssmax_scaling.s", "BF16", (32,), (32,)): 44,
    ("layers.N.attention.indexer.weights_proj.weight", "BF16", (32, 6144), (32, 6144)): 44,
    ("layers.N.attention.indexer.wk.weight", "BF16", (128, 6144), (128, 6144)): 44,
    ("layers.N.attention.indexer.wq_b.weight", "F8_E4M3", (4096, 6144), (4096, 6144)): 44,
    ("layers.N.attention.indexer.wq_b.weight_scale", "F32", (32, 1, 48, 1), (32, 1, 48, 1)): 44,
    ("layers.N.attention.kv_a_norm.weight", "BF16", (512,), (512,)): 63,
    ("layers.N.attention.wkv_a_with_mqa.weight", "BF16", (512, 6144), (512, 6144)): 9,
    ("layers.N.attention.wkv_a_with_mqa.weight", "BF16", (576, 6144), (576, 6144)): 54,
    ("layers.N.attention_norm.weight", "BF16", (6144,), (6144,)): 72,
    ("layers.N.experts.E.w1.input_scale", "F32", (), ()): 9088,
    ("layers.N.experts.E.w1.weight_scale_2", "F32", (), ()): 9088,
    ("layers.N.experts.E.w2.input_scale", "F32", (), ()): 9088,
    ("layers.N.experts.E.w2.weight_scale_2", "F32", (), ()): 9088,
    ("layers.N.experts.E.w3.input_scale", "F32", (), ()): 9088,
    ("layers.N.experts.E.w3.weight_scale_2", "F32", (), ()): 9088,
    ("layers.N.ffn_norm.weight", "BF16", (6144,), (6144,)): 72,
    ("layers.N.gate.weight", "BF16", (128, 6144), (128, 6144)): 71,
    ("lerope.alpha", "BF16", (32,), (32,)): 1,
    ("norm.weight", "BF16", (6144,), (6144,)): 1,
    ("tok_embeddings.weight", "BF16", (204800, 6144), (204800, 6144)): 1,
}
#: and the sharded keys, by world size: a quarter or an eighth of the heads, the dense and shared
#: MLP widths, the vocabulary and every expert's width.
_PER_RANK_SHARDED = {
    4: {
        ("layers.N.attention.ssmax_scaling.b", "BF16", (64,), (16,)): 72,
        ("layers.N.attention.ssmax_scaling.s", "BF16", (64,), (16,)): 72,
        ("layers.N.attention.wkv_b.weight", "BF16", (12288, 512), (3072, 512)): 54,
        ("layers.N.attention.wkv_b.weight", "BF16", (16384, 512), (4096, 512)): 18,
        ("layers.N.attention.wo.weight", "F8_E4M3", (6144, 8192), (6144, 2048)): 72,
        ("layers.N.attention.wo.weight_scale", "F32", (48, 1, 64, 1), (48, 1, 16, 1)): 72,
        ("layers.N.attention.wq.weight", "F8_E4M3", (8192, 6144), (2048, 6144)): 72,
        ("layers.N.attention.wq.weight_scale", "F32", (64, 1, 48, 1), (16, 1, 48, 1)): 72,
        ("layers.N.experts.E.w1.weight", "U8", (6144, 3072), (1536, 3072)): 9088,
        ("layers.N.experts.E.w1.weight_scale", "F8_E4M3", (6144, 384), (1536, 384)): 9088,
        ("layers.N.experts.E.w2.weight", "U8", (6144, 3072), (6144, 768)): 9088,
        ("layers.N.experts.E.w2.weight_scale", "F8_E4M3", (6144, 384), (6144, 96)): 9088,
        ("layers.N.experts.E.w3.weight", "U8", (6144, 3072), (1536, 3072)): 9088,
        ("layers.N.experts.E.w3.weight_scale", "F8_E4M3", (6144, 384), (1536, 384)): 9088,
        ("layers.N.feed_forward.w1.weight", "F8_E4M3", (24576, 6144), (6144, 6144)): 1,
        ("layers.N.feed_forward.w1.weight_scale", "F32", (192, 1, 48, 1), (48, 1, 48, 1)): 1,
        ("layers.N.feed_forward.w2.weight", "F8_E4M3", (6144, 24576), (6144, 6144)): 1,
        ("layers.N.feed_forward.w2.weight_scale", "F32", (48, 1, 192, 1), (48, 1, 48, 1)): 1,
        ("layers.N.feed_forward.w3.weight", "F8_E4M3", (24576, 6144), (6144, 6144)): 1,
        ("layers.N.feed_forward.w3.weight_scale", "F32", (192, 1, 48, 1), (48, 1, 48, 1)): 1,
        ("layers.N.shared_experts.w1.weight", "F8_E4M3", (6144, 6144), (1536, 6144)): 71,
        ("layers.N.shared_experts.w1.weight_scale", "F32", (48, 1, 48, 1), (12, 1, 48, 1)): 71,
        ("layers.N.shared_experts.w2.weight", "F8_E4M3", (6144, 6144), (6144, 1536)): 71,
        ("layers.N.shared_experts.w2.weight_scale", "F32", (48, 1, 48, 1), (48, 1, 12, 1)): 71,
        ("layers.N.shared_experts.w3.weight", "F8_E4M3", (6144, 6144), (1536, 6144)): 71,
        ("layers.N.shared_experts.w3.weight_scale", "F32", (48, 1, 48, 1), (12, 1, 48, 1)): 71,
        ("output.weight", "BF16", (204800, 6144), (51200, 6144)): 1,
    },
    8: {
        ("layers.N.attention.ssmax_scaling.b", "BF16", (64,), (8,)): 72,
        ("layers.N.attention.ssmax_scaling.s", "BF16", (64,), (8,)): 72,
        ("layers.N.attention.wkv_b.weight", "BF16", (12288, 512), (1536, 512)): 54,
        ("layers.N.attention.wkv_b.weight", "BF16", (16384, 512), (2048, 512)): 18,
        ("layers.N.attention.wo.weight", "F8_E4M3", (6144, 8192), (6144, 1024)): 72,
        ("layers.N.attention.wo.weight_scale", "F32", (48, 1, 64, 1), (48, 1, 8, 1)): 72,
        ("layers.N.attention.wq.weight", "F8_E4M3", (8192, 6144), (1024, 6144)): 72,
        ("layers.N.attention.wq.weight_scale", "F32", (64, 1, 48, 1), (8, 1, 48, 1)): 72,
        ("layers.N.experts.E.w1.weight", "U8", (6144, 3072), (768, 3072)): 9088,
        ("layers.N.experts.E.w1.weight_scale", "F8_E4M3", (6144, 384), (768, 384)): 9088,
        ("layers.N.experts.E.w2.weight", "U8", (6144, 3072), (6144, 384)): 9088,
        ("layers.N.experts.E.w2.weight_scale", "F8_E4M3", (6144, 384), (6144, 48)): 9088,
        ("layers.N.experts.E.w3.weight", "U8", (6144, 3072), (768, 3072)): 9088,
        ("layers.N.experts.E.w3.weight_scale", "F8_E4M3", (6144, 384), (768, 384)): 9088,
        ("layers.N.feed_forward.w1.weight", "F8_E4M3", (24576, 6144), (3072, 6144)): 1,
        ("layers.N.feed_forward.w1.weight_scale", "F32", (192, 1, 48, 1), (24, 1, 48, 1)): 1,
        ("layers.N.feed_forward.w2.weight", "F8_E4M3", (6144, 24576), (6144, 3072)): 1,
        ("layers.N.feed_forward.w2.weight_scale", "F32", (48, 1, 192, 1), (48, 1, 24, 1)): 1,
        ("layers.N.feed_forward.w3.weight", "F8_E4M3", (24576, 6144), (3072, 6144)): 1,
        ("layers.N.feed_forward.w3.weight_scale", "F32", (192, 1, 48, 1), (24, 1, 48, 1)): 1,
        ("layers.N.shared_experts.w1.weight", "F8_E4M3", (6144, 6144), (768, 6144)): 71,
        ("layers.N.shared_experts.w1.weight_scale", "F32", (48, 1, 48, 1), (6, 1, 48, 1)): 71,
        ("layers.N.shared_experts.w2.weight", "F8_E4M3", (6144, 6144), (6144, 768)): 71,
        ("layers.N.shared_experts.w2.weight_scale", "F32", (48, 1, 48, 1), (48, 1, 6, 1)): 71,
        ("layers.N.shared_experts.w3.weight", "F8_E4M3", (6144, 6144), (768, 6144)): 71,
        ("layers.N.shared_experts.w3.weight_scale", "F32", (48, 1, 48, 1), (6, 1, 48, 1)): 71,
        ("output.weight", "BF16", (204800, 6144), (25600, 6144)): 1,
    },
}
#: Besides, every rank reads the BF16 vision tower whole -- encoder, pre-projector norm, patch
#: merger and language adapter -- at either size: 276 keys, the transformer's 9 on each of its 30
#: layers.
_PER_RANK_VISION = {
    ("patch_merger.merging_layer.weight", "BF16", (2048, 8192), (2048, 8192)): 1,
    ("pre_mm_projector_norm.weight", "BF16", (2048,), (2048,)): 1,
    ("vision_encoder.ln_pre.weight", "BF16", (2048,), (2048,)): 1,
    ("vision_encoder.patch_conv.weight", "BF16", (2048, 3, 14, 14), (2048, 3, 14, 14)): 1,
    ("vision_language_adapter.w_in.weight", "BF16", (6144, 2048), (6144, 2048)): 1,
    ("vision_language_adapter.w_out.weight", "BF16", (6144, 6144), (6144, 6144)): 1,
    **{
        (f"vision_encoder.transformer.layers.N.{name}.weight", "BF16", shape, shape): 30
        for name, shape in (
            ("attention.wq", (2048, 2048)),
            ("attention.wk", (2048, 2048)),
            ("attention.wv", (2048, 2048)),
            ("attention.wo", (2048, 2048)),
            ("attention_norm", (2048,)),
            ("feed_forward.w1", (6144, 2048)),
            ("feed_forward.w2", (2048, 6144)),
            ("feed_forward.w3", (6144, 2048)),
            ("ffn_norm", (2048,)),
        )
    },
}
_PER_RANK = {
    world: {**_PER_RANK_WHOLE, **_PER_RANK_SHARDED[world], **_PER_RANK_VISION} for world in WORLDS
}

#: What each rank moves of the language model, from the headers (also plain arithmetic on them,
#: apart from the table: a quarter or an eighth of every expert code and block scale, every
#: expert scalar, the rest as at tep4 or tep8). tep8's rank moves 79,033,860,672 bytes of it,
#: its 16 whole experts 72,364,354,176 of them: the same bytes up to the per-tensor scalars of
#: the 112 experts a tep8 rank does not hold, 71 x 112 x 6 x 4 = 190,848 bytes.
_RANK_LM_BYTES = {4: 153_803_770_176, 8: 79_034_051_520}
_RANK_EXPERT_BYTES = {4: 144_728_871_936, 8: 72_364_545_024}
#: The same in GiB, to the second decimal: (rank, routed experts).
_RANK_GIB = {4: (143.24, 134.79), 8: (73.61, 67.39)}
#: The vision tower's keys and bytes (3.17 GiB), and all a rank moves: the language model and
#: the vision tower, 146.42 GiB at tp4 and 76.78 GiB at tp8.
_VISION_KEYS, _VISION_BYTES = 276, 3_408_437_248
_RANK_BYTES = {world: n + _VISION_BYTES for world, n in _RANK_LM_BYTES.items()}
_RANK_READ_GIB = {4: 146.42, 8: 76.78}


@pytest.fixture(autouse=True)
def _no_device_memory():
    """Nothing here allocates on a GPU. The context alone proves nothing: `import tensorrt_llm`
    creates one wherever a device is visible."""
    yield
    if torch.cuda.is_initialized():
        assert torch.cuda.memory_reserved() == 0, "the weight table test allocated on a GPU"


def _mapping(world: int, rank: int) -> Mapping:
    """The engine's mapping at tp4 or tp8: tensor parallel throughout, the routed experts tensor
    parallel (moe_tp_size the world size, moe_ep_size 1), no attention DP, four ranks per
    node."""
    m = Mapping(
        world_size=world,
        rank=rank,
        gpus_per_node=GPUS_PER_NODE,
        tp_size=world,
        moe_tp_size=world,
        moe_ep_size=1,
    )
    # A topology that ignores `rank=` would hand every rank rank 0's slices.
    assert (m.tp_rank, m.moe_tp_rank, m.moe_ep_rank) == (rank, rank, 0), m
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


def _bytes(t: torch.Tensor) -> torch.Tensor:
    """Compare by bit pattern: e4m3 has no `==`, and it is exactness that is asserted."""
    return t.contiguous().view(torch.uint8) if t.element_size() == 1 else t


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


#: Parametrizations: each world size, and each rank of each.
_WORLDS = pytest.mark.parametrize("world", WORLDS, ids=[f"tp{w}" for w in WORLDS])
_WORLD_RANKS = pytest.mark.parametrize(
    "world, rank", [pytest.param(w, r, id=f"tp{w}-{r}") for w in WORLDS for r in range(w)]
)


@pytest.fixture(scope="module")
def manifests() -> dict:
    """Each rank's manifest at the checkpoint's configuration, by world size and rank."""
    return {
        world: {
            rank: W.MODEL_WEIGHTS.manifest(_core(_CONFIG, world, rank)) for rank in range(world)
        }
        for world in WORLDS
    }


# --------------------------------------------------------------------- the configuration


def _layers_of(declared, name: str) -> dict[int, tuple]:
    """Layer index -> declared shape, for every `l{i}_<name>`."""
    out = {}
    for key, param in declared.items():
        head, _, rest = key.partition("_")
        if head.startswith("l") and head[1:].isdigit() and rest == name:
            out[int(head[1:])] = tuple(param.shape)
    return out


@_WORLD_RANKS
def test_the_width_slice_is_read_off_the_mapping(world, rank):
    """A gate at tp4 passes tensor_parallel_size 4 alone, one at tp8 tensor_parallel_size 8 alone,
    and the engine's Mapping then derives moe_tp_size at the world size and moe_ep_size 1 itself:
    the same mapping as the explicit one. Rank r is moe_tp_rank r of an expert window that is
    every expert, so the table reads expert_offset 0, local_experts 128, and an I-wide slice at Ir
    (I = 1536 at tp4, 768 at tp8); the rest of its geometry is tep4's at tp4 and tep8's at tp8."""
    g = _GEOMETRY[world]
    bare = Mapping(world_size=world, rank=rank, gpus_per_node=GPUS_PER_NODE, tp_size=world)
    for m in (_mapping(world, rank), bare):
        assert (m.tp_size, m.tp_rank, m.moe_tp_size, m.moe_tp_rank) == (world, rank, world, rank)
        assert (m.moe_ep_size, m.moe_ep_rank, m.enable_attention_dp) == (1, 0, False)
    d = W.MODEL_WEIGHTS.dims(_core(_CONFIG, world, rank))
    assert (d.local_experts, d.expert_offset, d.moe_inter, d.moe_tp_rank) == (128, 0, g.inter, rank)
    assert (d.tp_rank, d.tp_size, d.heads, d.shared_inter, d.dense_inter) == (
        rank,
        world,
        g.heads,
        g.shared,
        g.dense,
    )


@pytest.mark.parametrize(
    "world, rank", [pytest.param(w, r, id=f"tp{w}-{r}") for w in WORLDS for r in (0, w - 1)]
)
def test_layer_sets_and_per_rank_shapes(world, rank):
    """Which layers carry which weight, at what per-rank shape: 16 of the 64 heads at tp4 and 8
    at tp8 (`wq` 128 rows and `wo` 128 columns a head, `wkv_b` and SSMax per head), the dense MLP
    at 6144 or 3072 and the shared expert at 1536 or 768 of their widths, all 128 routed experts
    each at this rank's 1536- or 768-wide slice. The NoPE layers are 3, 7, ..., 71; the KV sharers
    39, 43, ..., 71; the 'shared' indexer layers 17, 18, 21, 22, ..., 69, 70. Every row but the
    routed experts' is tep4's at tp4 and tep8's at tp8."""
    g = _GEOMETRY[world]
    core = _core(_CONFIG, world, rank)
    with torch.device("meta"):
        declared = W.MODEL_WEIGHTS.declare(W.MODEL_WEIGHTS.dims(core))
    nope = {i for i in range(_LAYERS) if i % 4 == 3}
    sharers = {i for i in range(39, _LAYERS, 4)}
    full = {i for i in range(_LAYERS) if i < 17 or (i - 17) % 4 >= 2}
    moe = range(1, _LAYERS)
    assert len(nope) == 18 and len(sharers) == 9 and len(full) == 44

    every = {i: None for i in range(_LAYERS)}
    # Packed scale operands: one int32 word per four 128-blocks of K per output row, the last
    # word partly padding where the block count is not a multiple of four.
    words = {k: -(-k // 512) for k in (g.heads * 128, g.dense, g.shared)}
    expect = {
        "attn_norm": {i: (6144,) for i in every},
        "wq": {i: (g.heads * 128, 6144) for i in every},
        "wq_s": {i: (12, g.heads * 128) for i in every},
        "ssmax_s": {i: (g.heads,) for i in every},
        "ssmax_b": {i: (g.heads,) for i in every},
        "wkv_a": {i: (512 if i in nope else 576, 6144) for i in range(_LAYERS) if i not in sharers},
        "kv_a_norm": {i: (512,) for i in range(_LAYERS) if i not in sharers},
        # Per head [k_nope 128 | v 128] on NoPE layers, [k_nope 64 | v 128] on RoPE layers.
        "wkv_b": {i: (g.heads * (256 if i in nope else 192), 512) for i in every},
        "wo": {i: (6144, g.heads * 128) for i in every},
        # K 2048 at tp4: 16 blocks, four whole words; K 1024 at tp8: 8 blocks, two whole words.
        "wo_s": {i: (words[g.heads * 128], 6144) for i in every},
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
        "mlp_gu": {i: (2 * (g.dense if i == 0 else g.shared), 6144) for i in every},
        "mlp_gu_s": {i: (12, 2 * (g.dense if i == 0 else g.shared)) for i in every},
        "mlp_dn": {i: (6144, g.dense if i == 0 else g.shared) for i in every},
        # tp4: K 6144, 48 blocks in 12 words; K 1536, 12 blocks in 3. tp8: K 3072, 24 blocks in
        # six words; K 768, 6 blocks in two words, the second half padding.
        "mlp_dn_s": {i: (words[g.dense if i == 0 else g.shared], 6144) for i in every},
        "router": {i: (128, 6144) for i in moe},
        "fc1_w": {i: (128, 2 * g.inter, 3072) for i in moe},
        "fc1_s": {i: (128, 2 * g.inter, 384) for i in moe},
        "fc2_w": {i: (128, 6144, g.inter // 2) for i in moe},
        "fc2_s": {i: (128, 6144, g.inter // 16) for i in moe},
        "isc1": {i: (1,) for i in moe},
        "isc2": {i: (1,) for i in moe},
        "ws2_13": {i: (128,) for i in moe},
        "ws2_2": {i: (128,) for i in moe},
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

    # Per rank, with the shell's lm_head (51200 rows at tp4, 25600 at tp8). The routed experts
    # are exactly a quarter (tp4) or an eighth (tp8) of their checkpoint bytes per layer,
    # 128 x (3072 x 3072 + 3072 x 384 + 6144 x 768 + 6144 x 96) at tp4 -- what tep4's 32 whole
    # experts come to -- and 128 x (1536 x 3072 + 1536 x 384 + 6144 x 384 + 6144 x 48) at tp8 --
    # what tep8's 16 do -- and 1,032 bytes of scalars; the FP8 scale operands, one int32 word per
    # four 128-blocks per output row, are a few MiB more than the fp32 grids they are built from.
    per_expert, expert_bytes, low, high = {
        4: (15925248, 144_728_727_096, 143.24, 143.30),
        8: (7962624, 72_364_400_184, 73.61, 73.66),
    }[world]
    nbytes = {k: p.numel() * p.element_size() for k, p in declared.items()}
    experts = sum(n for k, n in nbytes.items() if _role(k) in _EXPERT_PARAMS)
    total = sum(nbytes.values()) + g.vocab * 6144 * 2
    assert experts == 71 * (128 * per_expert + 1032) == expert_bytes
    assert round(experts / 2**30, 2) == _RANK_GIB[world][1], experts / 2**30
    assert low * 2**30 < total < high * 2**30, total / 2**30


@_WORLDS
def test_each_rank_reads_every_expert_through_its_width_slice(manifests, world):
    """Rank r's manifest names all 128 routed experts on every MoE layer, local slot e <- global
    expert e, all twelve tensors of each: the codes and block scales through the row's shard,
    the per-tensor scalars whole; the one-per-layer scalars read every expert's copies. The
    known-not-loaded set predicts nothing: no expert is outside the window."""
    for rank, manifest in manifests[world].items():
        experts: dict = {}
        for key, sources in manifest.items():
            for ckpt, index, _, shard in sources:
                keys = W._keys(ckpt)
                if not any(_is_expert(k) for k in keys):
                    continue
                assert all(_is_expert(k) for k in keys), (key, ckpt)
                param = _role(key)
                assert param in _EXPERT_PARAMS, key
                assert (shard is not None) == (param in _LAID_OUT), key
                for k in keys:
                    layer, e, suffix = _expert_of(k)
                    assert key == f"l{layer}_{param}", (key, k)
                    if param in ("isc1", "isc2"):
                        assert index == (0,), (key, index)
                    else:
                        assert index[0] == e, (key, k, index)
                    experts.setdefault((layer, e), set()).add(suffix)
        assert set(experts) == {(i, e) for i in range(1, _LAYERS) for e in range(128)}
        assert all(v == _EXPERT_SUFFIXES for v in experts.values())
        assert W.MODEL_WEIGHTS.expected_unconsumed(_core(_CONFIG, world, rank), {}) == {}


@_WORLDS
def test_width_slices_tile_each_expert_exactly_once(manifests, world):
    """Every source of an expert row, on every MoE layer, shares the row's one shard (FC1's w3
    and w1 alike). Applied to a counter shaped like the stored tensor, rank r's shard is a view
    at exactly rank r's range -- at tp4 w1/w3 rows [1536r, + 1536), w2's code bytes [768r, + 768)
    and block-scale columns [96r, + 96); at tp8 w1/w3 rows [768r, + 768), w2's code bytes
    [384r, + 384) and block-scale columns [48r, + 48); the other axis whole -- and adding one
    through each rank's view leaves every element of the stored tensor at exactly one."""
    for role, (shape, axis, width) in _SLICED[world].items():
        counts = torch.zeros(shape, dtype=torch.int8)
        for rank, manifest in manifests[world].items():
            sources = [src for i in range(1, _LAYERS) for src in manifest[f"l{i}_{role}"]]
            assert len(sources) == 71 * 128 * (2 if role.startswith("fc1") else 1)
            shard = sources[0][3]
            assert all(src[3] is shard for src in sources), role
            view = shard(counts)
            want = list(shape)
            want[axis] = width
            assert tuple(view.shape) == tuple(want), (role, rank)
            assert view.data_ptr() - counts.data_ptr() == rank * width * counts.stride(axis)
            view.add_(1)
        assert bool((counts == 1).all()), role


@_WORLDS
def test_the_per_rank_slices_are_laid_out_unpadded(world):
    """The cut at 1536r (tp4) or 768r (tp8) splits no 16-input scale block (768 = 48 x 16) and no
    code byte. The production layouts take each per-rank slot as it is: FC1's 2 x 1536 = 3072 or
    2 x 768 = 1536 rows and FC2's 6144 are whole 32-row shuffle blocks and 128-row swizzle tiles,
    FC1's 384 and FC2's 96 or 48 scale columns whole 4-column tiles (each layout asserts its own
    rule). The stock TRTLLM-Gen NVFP4 method resolves this geometry, hidden 6144 and per-rank
    intermediate 1536 or 768, to weight and hidden alignment 32 -- 768 is a multiple of 32, and
    of the 128 its scale shuffle needs -- which both meet, so it pads nothing either: the
    unpadded chain `_stock` runs below is what it builds here."""
    d = W.MODEL_WEIGHTS.dims(_core(_CONFIG, world, 0))
    assert d.moe_inter % W.NVFP4_BLOCK == 0 and d.moe_inter % W.E2M1_PER_BYTE == 0
    per_rank = {
        4: {
            "fc1_w": (3072, 3072),
            "fc1_s": (3072, 384),
            "fc2_w": (6144, 768),
            "fc2_s": (6144, 96),
        },
        8: {
            "fc1_w": (1536, 3072),
            "fc1_s": (1536, 384),
            "fc2_w": (6144, 384),
            "fc2_s": (6144, 48),
        },
    }[world]
    laid_out = {}
    for entry in W.MODEL_WEIGHTS.WEIGHTS:
        if entry.layout is not None:
            slot = torch.empty(per_rank[entry.name], dtype=entry.dtype, device="meta")
            laid_out[entry.name] = tuple(entry.layout(slot).shape)
    assert laid_out == per_rank
    weight, hidden = NVFP4TRTLLMGenFusedMoEMethod.resolve_alignments(d.hidden, d.moe_inter)
    assert (weight, hidden) == (32, 32)
    assert d.moe_inter % weight == 0 and d.hidden % hidden == 0


@_WORLDS
def test_manifest_and_declaration_agree(manifests, world):
    """A declared weight with no source allocates memory nothing fills."""
    for rank, manifest in manifests[world].items():
        with torch.device("meta"):
            declared = W.MODEL_WEIGHTS.declare(W.MODEL_WEIGHTS.dims(_core(_CONFIG, world, rank)))
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


def _read_whole(checkpoint, keys) -> dict[str, torch.Tensor]:
    """The given tensors off the shard files, whole, one file at a time."""
    by_file: dict = {}
    for key in keys:
        by_file.setdefault(checkpoint.index[key], []).append(key)
    out = {}
    for name, names in sorted(by_file.items()):
        with safe_open(str(checkpoint.path / name), framework="pt", device="cpu") as handle:
            for key in names:
                out[key] = handle.get_tensor(key)
    return out


@pytest.fixture(scope="module")
def expert_scalars(checkpoint):
    """Every routed expert's per-tensor scalars, read off the shard files by the production
    reader. Every rank at either world size holds every expert, so tp4's rank 0 names them all."""
    d = W.MODEL_WEIGHTS.dims(_core(checkpoint.config, 4, 0))
    keys = W._expert_scalar_keys(d)
    return W._read_scalars(keys, checkpoint.index, (checkpoint.path, checkpoint.index))


def _meta(checkpoint) -> dict:
    return {
        key: torch.empty(shape, dtype=dtype, device="meta")
        for key, (dtype, shape) in checkpoint.headers.items()
    }


def _normalized(key: str) -> str:
    """A checkpoint key with its layer index (the language model's or the vision transformer's)
    as N and its expert index as E."""
    key = re.sub(r"^((vision_encoder\.transformer\.)?layers)\.\d+\.", r"\1.N.", key)
    return re.sub(r"\.experts\.\d+\.", ".experts.E.", key)


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


@_WORLDS
def test_every_rank_reads_every_key_but_the_draft(checkpoint, world):
    """The index holds 111,091 keys, 38 of them the DSpark draft's and 276 the vision tower's; of
    the rest, 1,721 are the language model's and 109,056 = 71 layers x 128 experts x 3
    projections x 4 tensors the routed experts'. Every rank reads every one of the 111,053
    (lm_head's `output.weight` by the shell) and no draft key, and its known-not-loaded set is
    exactly the draft, declared as the disabled DSpark draft."""
    keys = set(checkpoint.index)
    draft = {k for k in keys if _is_draft(k)}
    vision = {k for k in keys if _is_vision(k)}
    text = keys - draft - vision
    lm = {k for k in text if not _is_expert(k)}
    assert len(keys) == 111091 and len(draft) == _DRAFT_KEYS and len(vision) == _VISION_KEYS
    assert len(lm) == 1721 and len(text - lm) == 109056 == 71 * 128 * 3 * 4
    for rank in range(world):
        core = _core(checkpoint.config, world, rank)
        read = _source_keys(W.MODEL_WEIGHTS.manifest(core)) | {"output.weight"}
        skipped = W.MODEL_WEIGHTS.expected_unconsumed(core, checkpoint.index)
        assert read == text | vision
        assert set(skipped) == draft
        assert set(skipped.values()) == {"DSpark draft disabled"}


@_WORLD_RANKS
def test_per_rank_shape_and_dtype_of_every_key(checkpoint, world, rank):
    """Every key the rank reads, through the production manifest and shards on header-shaped
    meta tensors, by stored dtype and shape and the shape this rank moves, is `_PER_RANK`: the
    slices keep the dtype, the shell's lm_head takes `output.weight`'s vocabulary quarter at
    tp4 and eighth at tp8, and the vision tower is moved whole."""
    core = _core(checkpoint.config, world, rank)
    got: Counter = Counter()
    for sources in W.MODEL_WEIGHTS.manifest(core).values():
        for ckpt, _, _, shard in sources:
            for key in W._keys(ckpt):
                dtype, shape = checkpoint.headers[key]
                moved = W._materialize(torch.empty(shape, dtype=dtype, device="meta"), shard)
                assert moved.dtype == dtype, key
                got[(_normalized(key), _DTYPE_NAMES[dtype], shape, tuple(moved.shape))] += 1
    dtype, shape = checkpoint.headers["output.weight"]
    lm_head = _Shell(checkpoint.config, world, rank, "meta").lm_head.weight
    got[("output.weight", _DTYPE_NAMES[dtype], shape, tuple(lm_head.shape))] += 1
    assert dict(got) == _PER_RANK[world]
    assert sum(n for (k, *_), n in got.items() if not _is_expert(k) and not _is_vision(k)) == 1721
    assert sum(n for (k, *_), n in got.items() if _is_expert(k)) == 109056
    assert sum(n for (k, *_), n in got.items() if _is_vision(k)) == _VISION_KEYS


@_WORLDS
def test_per_rank_bytes_from_the_headers(checkpoint, world):
    """What each rank moves, from the safetensors headers through the production slices. At tp4
    the same 157,212,207,424 bytes (146.42 GiB) on every rank: the language model's
    153,803,770,176 (143.24 GiB), 144,728,871,936 (134.79 GiB) of it its width slices of the
    routed experts and their scalars -- what tep4's 32 whole experts per rank came to, to the
    second decimal of a GiB -- and the vision tower's 3,408,437,248 (3.17 GiB), whole. At tp8
    the same 82,442,488,768 bytes (76.78 GiB) on every rank: the language model's 79,034,051,520
    (73.61 GiB), 72,364,545,024 (67.39 GiB) of it its width slices of the routed experts and
    their scalars -- a tep8 rank's bytes and 190,848 more, the scalars of the 112 experts outside
    a tep8 window -- and the same vision tower."""
    expert_bytes = vision_bytes = 0
    for key, (dtype, shape) in checkpoint.headers.items():
        n = torch.empty(shape, dtype=dtype, device="meta").nbytes
        if _is_expert(key):
            expert_bytes += n if _is_expert_scalar(key) else n // world
        vision_bytes += n if _is_vision(key) else 0
    assert expert_bytes == _RANK_EXPERT_BYTES[world]
    assert vision_bytes == _VISION_BYTES
    for rank in range(world):
        core = _core(checkpoint.config, world, rank)
        d = W.MODEL_WEIGHTS.dims(core)
        total = experts = vision = 0
        for sources in W.MODEL_WEIGHTS.manifest(core).values():
            for ckpt, _, _, shard in sources:
                for key in W._keys(ckpt):
                    dtype, shape = checkpoint.headers[key]
                    t = W._materialize(torch.empty(shape, dtype=dtype, device="meta"), shard)
                    total += t.nbytes
                    experts += t.nbytes if _is_expert(key) else 0
                    vision += t.nbytes if _is_vision(key) else 0
        dtype, (vocab, hidden) = checkpoint.headers["output.weight"]
        total += vocab // d.tp_size * hidden * torch.empty((), dtype=dtype).element_size()
        assert (total, experts, vision) == (
            _RANK_BYTES[world],
            _RANK_EXPERT_BYTES[world],
            _VISION_BYTES,
        ), rank
    if world == 8:
        assert _RANK_LM_BYTES[8] - 79_033_860_672 == 71 * 112 * 6 * 4
    assert round(_RANK_LM_BYTES[world] / 2**30, 2) == _RANK_GIB[world][0]
    assert round(_RANK_EXPERT_BYTES[world] / 2**30, 2) == _RANK_GIB[world][1]
    assert round(_RANK_BYTES[world] / 2**30, 2) == _RANK_READ_GIB[world]


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
    """`(dimension, per-rank width)` of a key the ranks split, None for one each rank reads whole:
    16 heads of 128 at tp4, 8 at tp8 (`wq` rows, `wo` columns), of `wkv_b`'s per-head rows (192
    on RoPE layers, 256 on NoPE layers) and of SSMax; 6144 or 3072 dense and 1536 or 768
    shared-expert rows (`w1`, `w3`) or columns (`w2`), their block scales on the same 128-blocks
    (scale axis 0 for rows, 2 for columns); 51200 or 25600 vocabulary rows of `output`; and every
    routed expert's 1536- or 768-wide slice: `w1`/`w3` rows, codes and block scales alike, `w2`'s
    768 or 384 code bytes and 96 or 48 scale columns."""
    g = _GEOMETRY[world]
    if key == "output.weight":
        return 0, g.vocab
    m = re.fullmatch(r"layers\.(\d+)\.(.+)", key)
    if m is None:
        return None
    layer, rest = int(m.group(1)), m.group(2)
    expert = re.fullmatch(r"experts\.\d+\.(w[123])\.(weight|weight_scale)", rest)
    if expert is not None:
        proj, leaf = expert.groups()
        if proj != "w2":
            return 0, g.inter
        return 1, g.inter // 2 if leaf == "weight" else g.inter // 16
    table = {
        "attention.wq.weight": (0, g.heads * 128),
        "attention.wq.weight_scale": (0, g.heads),
        "attention.wo.weight": (1, g.heads * 128),
        "attention.wo.weight_scale": (2, g.heads),
        "attention.ssmax_scaling.s": (0, g.heads),
        "attention.ssmax_scaling.b": (0, g.heads),
        "attention.wkv_b.weight": (
            0,
            g.heads * (256 if _CONFIG["nope_layer_flags"][layer] else 192),
        ),
    }
    for module, width in (("feed_forward", g.dense), ("shared_experts", g.shared)):
        for w in ("w1", "w3"):
            table[f"{module}.{w}.weight"] = (0, width)
            table[f"{module}.{w}.weight_scale"] = (0, width // 128)
        table[f"{module}.w2.weight"] = (1, width)
        table[f"{module}.w2.weight_scale"] = (2, width // 128)
    return table.get(rest)


@_WORLDS
def test_ranks_tile_every_sliced_tensor(checkpoint, world):
    """The index the production `load` takes of every checkpoint key, on each of the four ranks
    at tp4 and the eight at tp8, at the headers' shapes. Each key is read at most once per rank.
    Every text key is read by every rank: a sliced one -- heads, rows, columns, vocabulary rows,
    block scales on the same boundary, each expert's width slice -- along exactly one dimension,
    rank r at [r w, (r + 1) w), so the ranks tile it exactly once; any other whole, the routed
    experts' per-tensor scalars and every vision key included. No DSpark draft key is read. The
    bytes so read are the loader's own per-rank total, and it is the headers' 157,212,207,424 at
    tp4 and 82,442,488,768 at tp8."""
    regions: dict = {}
    for rank in range(world):
        log: list = []
        weights = {
            key: _Cut(key, dtype, shape, log) for key, (dtype, shape) in checkpoint.headers.items()
        }
        W.MODEL_WEIGHTS.load(_Shell(checkpoint.config, world, rank, "meta"), weights)
        keys = [key for key, _ in log]
        assert len(keys) == len(set(keys)), "a key was read twice"
        total = experts = 0
        for key, index in log:
            dtype, shape = checkpoint.headers[key]
            region = _region(index, shape)
            regions.setdefault(key, {})[rank] = region
            n = torch.empty((), dtype=dtype).element_size()
            for start, stop in region:
                n *= stop - start
            total += n
            experts += n if _is_expert(key) else 0
        assert (total, experts) == (_RANK_BYTES[world], _RANK_EXPERT_BYTES[world]), rank

    sliced = 0
    for key, (_, shape) in checkpoint.headers.items():
        whole = tuple((0, size) for size in shape)
        if _is_draft(key):
            assert key not in regions, key
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
    # 72 x (wq, wo, their scales, SSMax s and b, wkv_b) + 6 dense + 71 x 6 shared + output, and
    # 71 x 128 x 3 x (codes, block scales) of the routed experts.
    assert sliced == 72 * 7 + 6 + 71 * 6 + 1 + 71 * 128 * 6


@_WORLD_RANKS
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

    d = W.MODEL_WEIGHTS.dims(_core(checkpoint.config, 4, 0))
    assert len(expert_scalars) == 71 * 128 * 6
    W._check_expert_scalars(expert_scalars, d)
    for i in range(1, _LAYERS):
        isc1 = expert_scalars[f"layers.{i}.experts.0.w1.input_scale"].item()
        isc2 = expert_scalars[f"layers.{i}.experts.0.w2.input_scale"].item()
        assert 0 < isc1 < 1 and 0 < isc2 < 1, (i, isc1, isc2)


@pytest.mark.parametrize(
    "world, param, module, k, words",
    [
        pytest.param(4, "mlp_dn_s", "shared_experts.w2", 1536, 3, id="tp4-shared"),
        pytest.param(4, "mlp_dn_s", "feed_forward.w2", 6144, 12, id="tp4-dense"),
        pytest.param(4, "wo_s", "attention.wo", 2048, 4, id="tp4-wo"),
        pytest.param(8, "mlp_dn_s", "shared_experts.w2", 768, 2, id="tp8-shared"),
        pytest.param(8, "mlp_dn_s", "feed_forward.w2", 3072, 6, id="tp8-dense"),
        pytest.param(8, "wo_s", "attention.wo", 1024, 2, id="tp8-wo"),
    ],
)
def test_packed_k_operand_of_each_rank_on_the_checkpoint(
    checkpoint, world, param, module, k, words
):
    """The DeepGEMM scale operand of each down projection at the K a rank's column slice gives,
    built by the production path (`fill`'s steps: the row's shard on the lazy entry, then its
    transform) from the checkpoint's own scales, on every layer and rank. It equals an
    independent byte-level packing of the test's own slice of the stored grid and
    `transform_sf_into_required_layout(recipe=(1, 128, 128))` on that slice, layout and strides
    included. At tp4 every K is whole words (12, 48 and 16 blocks); at tp8 the shared expert's
    K 768 is 6 blocks in 2 words: bytes 0..3 of word 0 and 0..1 of word 1 hold blocks 0..5, and
    bytes 2..3 of word 1 are zero."""
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


# --------------------------------------------------------------------- real experts, sliced

#: Sampled routed experts: an early, a middle and a late layer, each at both ends of the expert
#: axis and once between.
_SAMPLED = {1: (0, 77, 127), 35: (3, 64, 126), 71: (0, 101, 127)}

#: The stock TRTLLM-Gen NVFP4 method's `epilogue_tile_m`, which picks its 32-row shuffle.
_EPILOGUE_TILE_M = 128


@pytest.fixture(scope="module")
def stored_experts(checkpoint):
    """The sampled experts' codes and block scales, read whole: the independent side of the
    comparisons below (the production loader never reads an expert whole)."""
    keys = [
        W._expert_key(i, e, f"{proj}.{s}")
        for i, experts in _SAMPLED.items()
        for e in experts
        for proj in _PROJECTIONS
        for s in ("weight", "weight_scale")
    ]
    return _read_whole(checkpoint, keys)


def _stock(stored: dict, i: int, e: int, rank: int, tp: int) -> dict:
    """One expert as the stock TRTLLM-Gen NVFP4 MoE method prepares it for moe_tp_rank `rank`
    of moe_tp_size `tp`: `load_weight_shard` slices w1/w3 as COLUMN (output rows) and w2 as ROW
    (input columns), codes and block scales alike; FC1 stacks `[w3 ; w1]`; then the method's
    own permutations, `shuffle_matrix` and, on the block scales, `block_scale_interleave`,
    composed as its `_shuffle_*` methods compose them. Those methods first move the buffer to a
    GPU; the ops they call run on CPU, so this calls them directly."""
    cache: dict = {}

    def shard(proj, s, mode):
        t = stored[W._expert_key(i, e, f"{proj}.{s}")]
        return load_weight_shard(t, tp, rank, mode).contiguous()

    col, row = TensorParallelMode.COLUMN, TensorParallelMode.ROW
    w31 = torch.cat([shard("w3", "weight", col), shard("w1", "weight", col)])
    perm = trtllmgen_maybe_get_cached_w3_w1_permute_indices(w31, cache, _EPILOGUE_TILE_M)
    fc1_w = torch.ops.trtllm.shuffle_matrix(w31, perm)

    s31 = torch.cat([shard("w3", "weight_scale", col), shard("w1", "weight_scale", col)])
    s31 = s31.view(torch.uint8)
    perm = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
        s31, cache, _EPILOGUE_TILE_M, num_elts_per_sf=W.NVFP4_BLOCK
    )
    fc1_s = torch.ops.trtllm.block_scale_interleave(torch.ops.trtllm.shuffle_matrix(s31, perm))

    w2 = shard("w2", "weight", row)
    perm = trtllmgen_maybe_get_cached_w2_permute_indices(w2, cache, _EPILOGUE_TILE_M)
    fc2_w = torch.ops.trtllm.shuffle_matrix(w2, perm)

    s2 = shard("w2", "weight_scale", row).view(torch.uint8)
    perm = trtllmgen_maybe_get_cached_w2_permute_indices(
        s2, cache, _EPILOGUE_TILE_M, num_elts_per_sf=W.NVFP4_BLOCK
    )
    fc2_s = torch.ops.trtllm.block_scale_interleave(torch.ops.trtllm.shuffle_matrix(s2, perm))
    return dict(
        fc1_w=fc1_w,
        fc1_s=fc1_s.reshape(s31.shape),
        fc2_w=fc2_w,
        fc2_s=fc2_s.reshape(s2.shape),
    )


def _formula(stored: dict, i: int, e: int, world: int, rank: int) -> dict:
    """The same expert slice from the whole stored expert: cut here in torch, then laid out by
    this file's own row-at-a-time construction (`_runner_rows`, `_swizzled`), which the
    scaled-down tests below also use."""

    def part(proj, s, axis):
        t = _bytes(stored[W._expert_key(i, e, f"{proj}.{s}")])
        n = t.shape[axis] // world
        return t.narrow(axis, rank * n, n)

    fc1 = _runner_rows(2 * _GEOMETRY[world].inter, True)
    fc2 = _runner_rows(6144, False)
    return dict(
        fc1_w=torch.cat([part("w3", "weight", 0), part("w1", "weight", 0)])[fc1],
        fc1_s=_swizzled(
            torch.cat([part("w3", "weight_scale", 0), part("w1", "weight_scale", 0)])[fc1]
        ),
        fc2_w=part("w2", "weight", 1)[fc2],
        fc2_s=_swizzled(part("w2", "weight_scale", 1)[fc2]),
    )


def _load_with_real_experts(checkpoint, expert_scalars, world: int, rank: int):
    """The production `load` of `rank` on header-shaped meta tensors, except for the sampled
    layers' eight expert parameters: real CPU tensors, first filled with 0xab bytes and NaN,
    whose sources are the sampled experts' lazy entries of the shard files and, for every other
    expert, a zero stand-in of the stored shape. Returns the core."""
    model = _Shell(checkpoint.config, world, rank, "meta")
    w = model.model.w
    for i in _SAMPLED:
        for role in _EXPERT_PARAMS:
            p = w[f"l{i}_{role}"]
            if p.element_size() == 1:
                real = torch.full(p.shape, 0xAB, dtype=torch.uint8).view(p.dtype)
            else:
                real = torch.full(p.shape, float("nan"), dtype=p.dtype)
            w[f"l{i}_{role}"] = nn.Parameter(real, requires_grad=False)
    weights = _meta(checkpoint)
    weights.update(expert_scalars)
    with contextlib.ExitStack() as stack:
        handles: dict = {}
        for i, sampled in _SAMPLED.items():
            for e in range(128):
                for proj in _PROJECTIONS:
                    for s in ("weight", "weight_scale"):
                        key = W._expert_key(i, e, f"{proj}.{s}")
                        dtype, shape = checkpoint.headers[key]
                        if e not in sampled:
                            zeros = torch.zeros(shape[-1], dtype=torch.uint8).view(dtype)
                            weights[key] = zeros.expand(shape)
                            continue
                        name = checkpoint.index[key]
                        if name not in handles:
                            handles[name] = stack.enter_context(
                                safe_open(str(checkpoint.path / name), framework="pt", device="cpu")
                            )
                        weights[key] = handles[name].get_slice(key)
        W.MODEL_WEIGHTS.load(model, weights)
    return model.model


@_WORLD_RANKS
def test_real_experts_slice_then_layout(checkpoint, expert_scalars, stored_experts, world, rank):
    """The production `load`, reading the sampled experts' real entries through its shards,
    holds in slot e of each sampled layer's FC1/FC2 codes and block scales exactly the stock
    preparation's bytes of expert e at moe_tp_size 4 or 8 -- which this file's own construction
    from the whole stored expert reproduces too -- and in every other slot its zero stand-in's, so
    every slot was written by its own expert. The per-tensor scalars are whole: every expert's
    `weight_scale_2`, the layer's one FC1 and FC2 `input_scale`."""
    core = _load_with_real_experts(checkpoint, expert_scalars, world, rank)
    for i, sampled in _SAMPLED.items():
        for e in sampled:
            stock = _stock(stored_experts, i, e, rank, world)
            formula = _formula(stored_experts, i, e, world, rank)
            for role in _LAID_OUT:
                assert torch.equal(stock[role], formula[role]), (i, e, role)
                got = _bytes(core.w[f"l{i}_{role}"].data[e])
                assert torch.equal(got, stock[role]), (i, e, role, rank)
        for role in _LAID_OUT:
            param = _bytes(core.w[f"l{i}_{role}"].data)
            nonzero = torch.count_nonzero(param.reshape(128, -1), dim=1)
            assert {e for e in range(128) if nonzero[e]} == set(sampled), (i, role)

        def scalars(name):
            return torch.stack([expert_scalars[W._expert_key(i, e, name)] for e in range(128)])

        assert torch.equal(core.w[f"l{i}_ws2_13"].data, scalars("w1.weight_scale_2"))
        assert torch.equal(core.w[f"l{i}_ws2_2"].data, scalars("w2.weight_scale_2"))
        assert torch.equal(core.w[f"l{i}_isc1"].data, scalars("w1.input_scale")[:1])
        assert torch.equal(core.w[f"l{i}_isc2"].data, scalars("w2.input_scale")[:1])


@_WORLDS
def test_real_slices_reassemble_the_stored_experts(checkpoint, stored_experts, manifests, world):
    """Each rank's production shard read off the shard file -- a row read of w1/w3, a strided
    column read of w2 -- gives that rank's slice of a sampled expert's codes and block scales,
    and the four (tp4) or eight (tp8) slices joined on the cut axis are the stored tensor, byte
    for byte."""
    for i, sampled in _SAMPLED.items():
        for e in sampled:
            for proj in _PROJECTIONS:
                for s, kind in (("weight", "_w"), ("weight_scale", "_s")):
                    key = W._expert_key(i, e, f"{proj}.{s}")
                    role = ("fc2" if proj == "w2" else "fc1") + kind
                    shape, axis, width = _SLICED[world][role]
                    parts = []
                    path = str(checkpoint.path / checkpoint.index[key])
                    with safe_open(path, framework="pt", device="cpu") as handle:
                        for rank in range(world):
                            shard = manifests[world][rank][f"l{i}_{role}"][0][3]
                            part = W._materialize(handle.get_slice(key), shard)
                            assert (
                                part.shape[axis] == width
                                and part.shape[1 - axis] == shape[1 - axis]
                            )
                            parts.append(_bytes(part))
                    assert torch.equal(torch.cat(parts, axis), _bytes(stored_experts[key])), key


@_WORLDS
def test_real_experts_layout_then_slice(stored_experts, world):
    """The negative control: each sampled expert laid out whole -- the stock preparation at
    moe_tp_size 1, the layout a rank holding whole experts uses -- then cut to rank r, against
    the per-rank layout the tests above pin. The cut differs where the layout depends on the
    slice: on FC2's block scales, whose 128x4 swizzle orders bytes by the column count (384
    whole, 96 per rank at tp4, 48 at tp8), and on FC1 cut at the checkpoint's ranges -- rows
    [Ir, + I) of the w3 half and of the w1 half, I = 1536 at tp4 and 768 at tp8 -- which the
    interleave has spread over the stack. It coincides where the layout leaves the cut alone: FC1
    cut into contiguous 2I-row chunks (the interleave pairs up row i with gate row i, so a rank's
    pairs are one run of the stack, which starts on a 32-row shuffle block and a 128-row swizzle
    tile), and FC2's codes, whose shuffle moves rows while the cut takes columns."""
    inter = _GEOMETRY[world].inter
    for i, sampled in _SAMPLED.items():
        for e in sampled:
            whole = _stock(stored_experts, i, e, rank=0, tp=1)
            for r in range(world):
                mine = _stock(stored_experts, i, e, r, world)
                for role in ("fc1_w", "fc1_s"):
                    t = whole[role]
                    halves = [t[h * 6144 + r * inter : h * 6144 + (r + 1) * inter] for h in (0, 1)]
                    assert not torch.equal(torch.cat(halves), mine[role]), (i, e, r, role)
                    rows = slice(r * 2 * inter, (r + 1) * 2 * inter)
                    assert torch.equal(t[rows], mine[role]), (i, e, r, role)
                fc2_s = whole["fc2_s"][:, r * inter // 16 : (r + 1) * inter // 16]
                assert not torch.equal(fc2_s, mine["fc2_s"]), (i, e, r)
                fc2_w = whole["fc2_w"][:, r * inter // 2 : (r + 1) * inter // 2]
                assert torch.equal(fc2_w, mine["fc2_w"]), (i, e, r)


# --------------------------------------------------------------------- values, scaled down

_TINY_LAYERS = 8
#: Every structural feature at a size a CPU fills in a second, by world size: a dense layer,
#: RoPE and NoPE layers (3, 7), a KV sharer (7, reading 3), 'shared' indexer layers (2, 5, 6),
#: two heads per rank, every expert cut to 128 per rank (FC1 256 rows, FC2 64 code bytes and 8
#: scale columns per rank: the shuffle and swizzle tile them), one 128-block per rank on the
#: shared expert. Eight experts and the dense layer at 2 128-blocks per rank at tp4; sixteen
#: experts and the dense layer at 6 128-blocks per rank at tp8, so its `w2` scale operand has the
#: half-padding second word the shared expert's has at full size.
_TINY_COMMON = dict(
    num_hidden_layers=_TINY_LAYERS,
    hidden_size=256,
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
    n_shared_experts=1,
    vocab_size=512,
)
_TINY = {
    4: dict(
        _TINY_COMMON,
        num_attention_heads=8,
        n_routed_experts=8,
        moe_intermediate_size=512,
        intermediate_size=1024,
    ),
    8: dict(
        _TINY_COMMON,
        num_attention_heads=16,
        n_routed_experts=16,
        moe_intermediate_size=1024,
        intermediate_size=6144,
    ),
}
#: Each routed expert's per-rank slice in the scaled-down table, at either world size.
_TINY_INTER = 128


def _tiny_checkpoint(world: int, seed: int = 0) -> dict:
    """A Mistral-named checkpoint at `_TINY`: e4m3 weights with power-of-two fp32 block scales
    stored `[N/128, 1, K/128, 1]` where the real one has them, NVFP4 routed experts -- packed
    codes, e4m3 block scales, a weight_scale_2 per expert that w1 and w3 share, one input_scale
    per projection and layer -- bf16 elsewhere, and two DSpark draft tensors."""
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
        experts = [f"{p}.experts.{e}" for e in range(c.n_routed_experts)]
        # Every expert, this rank's 128 rows of w3 and w1 (up rows first), interleaved and
        # block-shuffled, and its 64 code bytes / 8 scale columns of w2, block-shuffled; the
        # block scales also swizzled.
        fc1 = _runner_rows(2 * _TINY_INTER, True)
        fc2 = _runner_rows(c.hidden_size, False)

        def up_gate(e, suffix):
            return torch.cat(
                [_bytes(rows(ck[f"{e}.{w}.{suffix}"], _TINY_INTER)) for w in ("w3", "w1")]
            )[fc1]

        def down(e, suffix, per_column):
            return _bytes(cols(ck[f"{e}.w2.{suffix}"], _TINY_INTER // per_column))[fc2]

        out[q + "fc1_w"] = torch.stack([up_gate(e, "weight") for e in experts])
        out[q + "fc1_s"] = torch.stack([_swizzled(up_gate(e, "weight_scale")) for e in experts])
        out[q + "fc2_w"] = torch.stack([down(e, "weight", 2) for e in experts])
        out[q + "fc2_s"] = torch.stack([_swizzled(down(e, "weight_scale", 16)) for e in experts])
        out[q + "isc1"] = ck[f"{experts[0]}.w1.input_scale"].reshape(1)
        out[q + "isc2"] = ck[f"{experts[0]}.w2.input_scale"].reshape(1)
        out[q + "ws2_13"] = torch.stack([ck[f"{e}.w1.weight_scale_2"] for e in experts])
        out[q + "ws2_2"] = torch.stack([ck[f"{e}.w2.weight_scale_2"] for e in experts])
    out["final_norm"] = ck["norm.weight"]
    out["embed"] = ck["tok_embeddings.weight"]
    out["lerope_alpha"] = ck["lerope.alpha"].float()
    return out


@_WORLDS
def test_tiny_scale_operands_cover_whole_and_partial_words(world):
    """The scaled-down table's scale operands at a rank, by K blocks: 2 for every K of the
    hidden size 256 and for `wo` (2 heads x 128), 1 for the shared `w2` (128 of 512 at tp4, of
    1024 at tp8), and for the dense `w2` 2 at tp4 (256 of 1024) and 6 at tp8 (768 of 6144) -- at
    tp8 the shared expert's case at full size, 6 blocks in 2 words. Every one has a partly padded
    last word, which the values test below compares whole. Every rank holds all 8 (tp4) or 16
    (tp8) experts, each at its 128-wide slice."""
    d = W.MODEL_WEIGHTS.dims(_core(_TINY[world], world, 0))
    experts, dense, dense_words = {4: (8, 256, 1), 8: (16, 768, 2)}[world]
    assert (d.heads, d.dense_inter, d.shared_inter) == (2, dense, 128)
    assert (d.local_experts, d.expert_offset, d.moe_inter) == (experts, 0, _TINY_INTER)
    with torch.device("meta"):
        declared = W.MODEL_WEIGHTS.declare(d)
    assert tuple(declared["l0_mlp_dn_s"].shape) == (dense_words, 256)
    assert tuple(declared["l1_mlp_dn_s"].shape) == (1, 256)
    assert tuple(declared["l1_wo_s"].shape) == (1, 256)


@_WORLD_RANKS
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
            per_row = ref.repeat_interleave(128, dim=0)
            assert torch.equal(_decode_sf(param, n, ref.shape[1]), per_row), key
            assert torch.equal(param, _packed(ref)), key
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


@_WORLDS
def test_tiny_expert_halves_are_up_then_gate(world):
    """Undo the block shuffle (and the scales' swizzle) of FC1: its even rows are this rank's
    rows of w3 (up), its odd rows the same rows of w1 (gate), codes and block scales alike --
    the order the NVFP4 runner's FC1 reads, up first, the opposite of `mlp_gu`'s gate first."""
    ck = _tiny_checkpoint(world)
    rank, e = 1, 2
    model = _Shell(_TINY[world], world, rank, "cpu")
    W.MODEL_WEIGHTS.load(model, dict(ck))
    w = model.model.w
    mine = slice(rank * _TINY_INTER, (rank + 1) * _TINY_INTER)
    shuffled = _runner_rows(2 * _TINY_INTER, False)
    unshuffle = torch.empty_like(shuffled)
    unshuffle[shuffled] = torch.arange(2 * _TINY_INTER)
    codes = w["l4_fc1_w"][e][unshuffle]
    assert torch.equal(codes[0::2], ck[f"layers.4.experts.{e}.w3.weight"][mine])
    assert torch.equal(codes[1::2], ck[f"layers.4.experts.{e}.w1.weight"][mine])
    scales = _unswizzled(_bytes(w["l4_fc1_s"][e]))[unshuffle]
    assert torch.equal(scales[0::2], _bytes(ck[f"layers.4.experts.{e}.w3.weight_scale"][mine]))
    assert torch.equal(scales[1::2], _bytes(ck[f"layers.4.experts.{e}.w1.weight_scale"][mine]))


@_WORLDS
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
    """`safe_open` that records every key read through it, and for each the index its lazy
    entry was read at and the shape that read returned."""

    def __init__(self, real):
        self.real, self.keys, self.reads = real, [], []

    def __call__(self, *args, **kwargs):
        log = self

        class Entry:
            def __init__(self, key, entry):
                self.key, self.entry = key, entry

            def __getitem__(self, index):
                out = self.entry[index]
                log.reads.append((self.key, index, tuple(out.shape)))
                return out

        class Handle:
            def __enter__(self):
                self.h = log.real(*args, **kwargs).__enter__()
                return self

            def __exit__(self, *exc):
                return self.h.__exit__(*exc)

            def get_slice(self, key):
                log.keys.append(key)
                return Entry(key, self.h.get_slice(key))

        return Handle()


@_WORLD_RANKS
def test_tiny_load_streams_from_the_shard_files(world, rank, tmp_path, monkeypatch):
    """With the checkpoint's shard files found through the directory the engine records on
    the config, every source is read from them -- the engine's dict is never read: here it
    holds meta tensors, which cannot be copied out -- and the result is bit for bit the dict
    path's. Each key is read once, every key but the DSpark draft's, nothing of the draft. Every
    expert's codes and block scales are read as this rank's slice of the entry -- w1/w3 rows
    [128r, + 128), w2's code bytes [64r, + 64) and scale columns [8r, + 8) -- never whole; the
    per-tensor scalars whole. Each file's cached pages are dropped once read."""
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
    assert read == {k for k in ck if not _is_draft(k)}
    reads = {}
    for key, index, shape in log.reads:
        assert key not in reads, f"{key}: its entry was read twice"
        reads[key] = (index, shape)
    for key in ck:
        if not _is_expert(key):
            continue
        index, shape = reads[key]
        stored = tuple(ck[key].shape)
        if _is_expert_scalar(key):
            assert (index, shape) == (Ellipsis, ()), key
        elif ".w2." in key:
            width = _TINY_INTER // (2 if key.endswith(".weight") else 16)
            assert index == (slice(None), slice(rank * width, (rank + 1) * width)), (key, index)
            assert shape == (stored[0], width), key
        else:
            assert index == slice(rank * _TINY_INTER, (rank + 1) * _TINY_INTER), (key, index)
            assert shape == (_TINY_INTER, stored[1]), key
    assert sorted(p.name for p in dropped) == sorted(p.name for p in tmp_path.glob("*.safetensors"))


@_WORLDS
def test_tiny_load_falls_back_to_the_dict_without_an_index(world, tmp_path):
    """A recorded directory without the shard index is not the checkpoint: the dict is read."""
    ck = _tiny_checkpoint(world)
    model = _Shell(_TINY[world], world, 2, "cpu")
    model.model.model_config.pretrained_config._name_or_path = str(tmp_path)
    W.MODEL_WEIGHTS.load(model, dict(ck))


@_WORLDS
def test_tiny_load_without_draft_tensors(world):
    """The DSpark draft is never asserted on: a checkpoint without it loads too."""
    ck = {k: v for k, v in _tiny_checkpoint(world).items() if not _is_draft(k)}
    W.MODEL_WEIGHTS.load(_Shell(_TINY[world], world, 1, "cpu"), ck)


def _load_raises(ck: dict, match: str, world: int, rank: int = 0) -> None:
    with pytest.raises(AssertionError, match=match):
        W.MODEL_WEIGHTS.load(_Shell(_TINY[world], world, rank, "cpu"), ck)


@pytest.mark.parametrize(
    "world, key",
    [
        (4, "layers.3.experts.6.w3.weight"),
        (4, "layers.3.experts.1.w1.weight_scale"),
        (4, "layers.3.experts.7.w2.weight"),
        (4, "layers.3.experts.0.w3.weight_scale_2"),
        (4, "layers.3.experts.5.w2.input_scale"),
        (8, "layers.3.experts.6.w3.weight"),
        (8, "layers.3.experts.1.w1.weight_scale"),
        (8, "layers.3.experts.13.w2.weight"),
        (8, "layers.3.experts.0.w3.weight_scale_2"),
        (8, "layers.3.experts.15.w2.input_scale"),
    ],
)
def test_tiny_load_refuses_a_missing_expert_tensor(world, key):
    """Every rank reads every expert, so any expert's tensor missing is refused on rank 0."""
    ck = _tiny_checkpoint(world)
    del ck[key]
    _load_raises(ck, f"checkpoint key missing: {key}", world)


@_WORLDS
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
@_WORLDS
def test_tiny_load_refuses_a_scale_that_is_not_ue8m0(world, key):
    ck = _tiny_checkpoint(world)
    ck[key] = ck[key].clone()
    ck[key].view(-1)[0] *= 1.5
    _load_raises(ck, "not a positive normal power of two", world)


@_WORLDS
def test_tiny_load_refuses_a_two_dimensional_fp8_scale(world):
    """A `[N/128, K/128]` block-scale layout is not this one's."""
    ck = _tiny_checkpoint(world)
    key = "layers.4.attention.indexer.wq_b.weight_scale"
    ck[key] = _grid(ck[key]).contiguous()
    _load_raises(ck, r"not \[N/128, 1, K/128, 1\]", world)


@pytest.mark.parametrize(
    "world, expert", [(4, 0), (4, 6), (8, 0), (8, 13)], ids=["tp4-0", "tp4-6", "tp8-0", "tp8-13"]
)
@pytest.mark.parametrize("scalar", ["weight_scale_2", "input_scale"])
def test_tiny_load_refuses_w1_and_w3_scalars_that_differ(world, expert, scalar):
    ck = _tiny_checkpoint(world)
    key = f"layers.5.experts.{expert}.w3.{scalar}"
    ck[key] = ck[key] * 2
    experts = _TINY[world]["n_routed_experts"]
    want = f"layers.5: w1 and w3 {scalar} differ on 1 of {experts} experts, first expert {expert}:"
    _load_raises(ck, want, world)


@_WORLDS
@pytest.mark.parametrize("proj", ["w1", "w2"])
def test_tiny_load_refuses_an_input_scale_that_varies_over_the_experts(world, proj):
    """One quantization of the replicated activations serves every expert's slice on every
    rank: one expert disagreeing -- the last, 7 at tp4 and 15 at tp8 -- is refused."""
    ck = _tiny_checkpoint(world)
    last = _TINY[world]["n_routed_experts"] - 1
    for p in {proj, "w3"} if proj == "w1" else {proj}:
        key = f"layers.6.experts.{last}.{p}.input_scale"
        ck[key] = ck[key] * 2
    want = f"layers.6: {proj}.input_scale is not one value over the experts: expert {last}"
    _load_raises(ck, want, world)


@_WORLDS
@pytest.mark.parametrize("value", [0.0, -1e-4, float("inf"), float("nan")])
def test_tiny_load_refuses_a_scalar_that_is_not_a_positive_finite_scale(world, value):
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
