# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The 64-head `wkv_b` regroup of the dep4 weight table, on the checkpoint's own tensors.

Under attention DP every rank holds `wkv_b` whole, all 64 heads, and the table regroups it
from the checkpoint's per-head `[k_nope | v]` rows into `[all heads' k_nope ; all heads' v]`,
the K-B and V-B operands of the absorbed attention. A regroup at the tep4 count of 16 heads,
applied to the whole tensor, has exactly the same shape -- 12288 rows on a RoPE layer, 16384
on a NoPE layer -- and puts most rows in the wrong place, so no shape check sees it.

On one RoPE layer (0) and one NoPE layer (3) of Mistral-Large-4-1T-A52B-NVFP4, the table's
own manifest row (its source key, destination and transform) is applied on every rank -- of
four ranks on one node, and of eight over two nodes of four (dep8) -- to the stored tensor and
compared, bit for bit, with an independent per-head split of the same tensor; and the 16-head
regroup is shown to differ from it. No GPU. Skipped where the checkpoint is absent (looked up
as `mistral-large-4-nvfp4` or `Mistral-Large-4-1T-A52B-NVFP4` under `LLM_MODELS_ROOT`).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from test_common.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__dep4 import (  # noqa: E501
    weights as W,
)
from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

#: The attention-DP worlds the target serves: four ranks on one node, eight over two.
WORLDS = (4, 8)
#: GB300 nodes here hold four devices.
GPUS_PER_NODE = 4
#: The tep4 target's ranks: its 16 heads per rank are the regroup shown wrong here.
TEP4_RANKS = 4
CHECKPOINTS = ("mistral-large-4-nvfp4", "Mistral-Large-4-1T-A52B-NVFP4")
#: (layer, k_nope width): a RoPE layer and a NoPE layer, whose nope spends the rope dims too.
LAYERS = ((0, 64), (3, 128))


@pytest.fixture(scope="module")
def checkpoint():
    root = llm_models_root()
    for name in CHECKPOINTS:
        path = None if root is None else Path(root) / name
        if path is not None and all((path / f).is_file() for f in ("params.json", W.SHARD_INDEX)):
            # The configuration the engine adapts from the checkpoint's params.json.
            config = MistralConfigLoader().load_pretrained_config(str(path))
            weight_map = json.loads((path / W.SHARD_INDEX).read_text())["weight_map"]
            return SimpleNamespace(path=path, config=config, weight_map=weight_map)
    pytest.skip(f"checkpoint {CHECKPOINTS} is not under LLM_MODELS_ROOT ({root})")


def _core(config, rank: int, world: int) -> SimpleNamespace:
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
    return SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=config,
            mapping=mapping,
            torch_dtype=torch.bfloat16,
        )
    )


def _stored(checkpoint, key: str) -> torch.Tensor:
    with safe_open(
        str(checkpoint.path / checkpoint.weight_map[key]), framework="pt", device="cpu"
    ) as handle:
        return handle.get_tensor(key)


def _per_head_split(t: torch.Tensor, heads: int, nope: int, v: int) -> torch.Tensor:
    """Head by head, off the stored rows: head h owns rows `h * (nope + v)` to
    `(h + 1) * (nope + v)`, k_nope first. All heads' k_nope, then all heads' v."""
    k_rows, v_rows = [], []
    for h in range(heads):
        first = h * (nope + v)
        k_rows.append(t[first : first + nope])
        v_rows.append(t[first + nope : first + nope + v])
    return torch.cat(k_rows + v_rows, dim=0)


@pytest.mark.parametrize("world", WORLDS, ids=lambda w: f"world{w}")
@pytest.mark.parametrize("layer, nope", LAYERS, ids=["rope-layer0", "nope-layer3"])
def test_wkv_b_regroup_is_the_per_head_split_of_all_64_heads(checkpoint, layer, nope, world):
    cfg = checkpoint.config
    heads, v = cfg.num_attention_heads, cfg.v_head_dim
    assert heads == 64 and (cfg.qk_nope_head_dim if layer % 4 != 3 else 128) == nope
    key = f"layers.{layer}.attention.wkv_b.weight"
    stored = _stored(checkpoint, key)
    assert tuple(stored.shape) == (heads * (nope + v), cfg.kv_lora_rank), stored.shape
    expected = _per_head_split(stored, heads, nope, v)

    for rank in range(world):
        core = _core(cfg, rank, world)
        d = W.MODEL_WEIGHTS.dims(core)
        rows = W.MODEL_WEIGHTS.manifest(core)[f"l{layer}_wkv_b"]
        assert len(rows) == 1, rows
        ckpt, index, transform = rows[0]
        assert (ckpt, index) == (key, None), (ckpt, index)
        got = transform(stored.contiguous(), core)
        # Two rows share the name over disjoint layer sets: the RoPE and the NoPE layers'.
        declared = next(
            r for r in W.MODEL_WEIGHTS.WEIGHTS if r.name == "wkv_b" and layer in r.layer_indices(d)
        ).shape(d)
        assert tuple(got.shape) == tuple(declared) == tuple(expected.shape)
        assert torch.equal(got, expected), f"rank {rank}: the regroup is not the per-head split"

    # The tep4 head count on the whole tensor: each "head" is then four real heads' rows, split
    # into a 4 * (nope + v) - v block and a v block. The same shape, other rows.
    t3 = stored.reshape(heads // TEP4_RANKS, -1, stored.shape[1])
    nope16 = t3.shape[1] - v
    tep4 = torch.cat(
        [t3[:, :nope16].reshape(-1, t3.shape[2]), t3[:, nope16:].reshape(-1, t3.shape[2])]
    )
    assert tep4.shape == expected.shape
    moved = (tep4 != expected).any(dim=1).float().mean().item()
    print(f"layer {layer}: a 16-head regroup moves {moved:.1%} of the {len(expected)} rows")
    assert moved > 0.5, moved
