# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The default target's ``weights.py`` on CPU: the TP shard view against the engine's, the refusal of an unknown key,
the language-model / vision key routing, the routed-expert layout and slot loading (the safe_open and no-index
fallbacks)."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default import (
    weights as default_weights,
)
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
from tensorrt_llm._torch.modules.linear import TensorParallelMode, load_weight_shard

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def target_weights():
    """The default target's ``weights`` module."""
    return default_weights


# ---------------------------------------------------------------------------
# The checkpoint helpers.
# ---------------------------------------------------------------------------

_KDA_WIDTH = 96 * 128  # KDA heads x head dim
_OTHER = 64  # width of the axis that is not split; it does not change the slicing

# (shape, split dim): the split axis has the width of the checkpoint tensor it stands for.
_SHARD_CASES = [
    ((_KDA_WIDTH, _OTHER), 0),  # KDA q / k / v / f_b / g_b projections: output rows
    ((_OTHER, _KDA_WIDTH), 1),  # KDA o_proj: input columns
]
_TP_SIZES = (1, 2, 3, 4, 5, 8, 12, 16, 32)


@pytest.mark.parametrize("shape, dim", _SHARD_CASES)
def test_tp_shard_is_load_weight_shard(target_weights, shape, dim):
    tensor = torch.randn(shape, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    mode = TensorParallelMode.ROW if dim == 1 else TensorParallelMode.COLUMN
    cpu = torch.device("cpu")
    for tp_size in _TP_SIZES:
        for tp_rank in range(tp_size):
            expected = load_weight_shard(tensor, tp_size, tp_rank, mode, device=cpu)
            got = target_weights.tp_shard(tensor, tp_size, tp_rank, dim, cpu)
            assert got.dtype == expected.dtype
            assert got.shape == expected.shape, (tp_size, tp_rank)
            # The same view of the same storage, so the same bytes reach the parameter.
            assert got.stride() == expected.stride()
            assert got.storage_offset() == expected.storage_offset()
            assert got.untyped_storage().data_ptr() == expected.untyped_storage().data_ptr()
            assert torch.equal(got, expected)


@contextmanager
def _lazy_checkpoint(path):
    """The engine's lazy read of a checkpoint directory (the path ``kimi_k31`` checkpoints take)."""
    weights = HfWeightLoader()._load_lazy_safetensors(str(path))
    try:
        yield weights
    finally:
        weights.clear()


# ---------------------------------------------------------------------------
# The checkpoint split and the routed-expert layout.
# ---------------------------------------------------------------------------

_LANG = ["language_model.model.embed_tokens.weight", "language_model.lm_head.weight"]
_VISION = ["vision_tower.patch_embed.proj.weight", "mm_projector.proj.0.weight"]


class _VisionTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loaded = None

    def load_weights(self, weights) -> bool:
        self.loaded = sorted(weights.keys())
        return True


def _model(skip_mm_encoder: bool, mm_encoder=None) -> SimpleNamespace:
    return SimpleNamespace(
        _skip_mm_encoder=skip_mm_encoder,
        mm_encoder=mm_encoder,
        llm=SimpleNamespace(name="text model"),
    )


@pytest.fixture
def text_loads(monkeypatch, target_weights):
    """Record what the text model's loader receives instead of loading it."""
    calls = []

    def load_text_model(llm, weights):
        calls.append((llm, sorted(weights.keys()), getattr(weights, "checkpoint_prefix", None)))

    monkeypatch.setattr(target_weights, "load_text_model", load_text_model)
    return calls


def test_unknown_key_fails_naming_it(target_weights, text_loads):
    weights = {k: None for k in _LANG + _VISION + ["model.layers.0.extra.weight"]}
    with pytest.raises(ValueError, match="model.layers.0.extra.weight"):
        target_weights.load(_model(skip_mm_encoder=True), weights)
    assert not text_loads


def test_vision_keys_go_to_the_local_tower(target_weights, text_loads):
    tower = _VisionTower()
    model = _model(skip_mm_encoder=False, mm_encoder=tower)
    target_weights.load(model, {k: None for k in _LANG + _VISION})
    assert set(_VISION) <= set(tower.loaded)
    assert model.mm_encoder is tower
    ((_, keys, _),) = text_loads
    assert keys == ["lm_head.weight", "model.embed_tokens.weight"]


def test_packed_mxfp4_expert_reads_the_checkpoint_expert_into_its_slot(target_weights, monkeypatch):
    calls = []
    monkeypatch.setattr(
        target_weights,
        "load_packed_mxfp4_expert",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    experts = SimpleNamespace(
        initial_local_expert_ids=[1, 2],
        w3_w1_weight="w31",
        w3_w1_weight_scale="w31s",
        w2_weight="w2",
        w2_weight_scale="w2s",
        _packed_mxfp4_loaded_slots=set(),
        intermediate_size=3072,
        tp_size=4,
        tp_rank=3,
    )
    target_weights._load_packed_mxfp4_expert(experts, "experts", 5, 2, 1, lambda key: key)
    ((args, kwargs),) = calls
    assert args == ("w31", "w31s", "w2", "w2s", 1, experts._packed_mxfp4_loaded_slots)
    assert kwargs == {
        "intermediate_size": 3072,
        "moe_tp_size": 4,
        "moe_tp_rank": 3,
        **{
            f"{w}_{name}": f"experts.5.{w}.{kind}"
            for w in ("w1", "w2", "w3")
            for name, kind in (("weight", "weight_packed"), ("weight_scale", "weight_scale"))
        },
    }
    # The slot serves the expert the routed experts' partition assigns it.
    with pytest.raises(ValueError, match="expects global expert 2"):
        target_weights._load_packed_mxfp4_expert(experts, "experts", 5, 7, 1, lambda key: key)


# ---------------------------------------------------------------------------
# The routed-expert streaming of the text-model loader, on small CPU tensors.
# ---------------------------------------------------------------------------


def _llm(layers) -> SimpleNamespace:
    anchor = nn.Parameter(torch.zeros(1))
    return SimpleNamespace(
        model=SimpleNamespace(layers=layers),
        parameters=lambda: iter([anchor]),
        model_config=SimpleNamespace(
            mapping=SimpleNamespace(tp_rank=0),
            pretrained_config=SimpleNamespace(_name_or_path=""),
        ),
    )


_KINDS = ("weight_packed", "weight_scale")
_EXPERT_BASE = "model.layers.0.block_sparse_moe.experts"


def _expert_tensor(expert: int, w: str, kind: str) -> torch.Tensor:
    gen = torch.Generator().manual_seed(100 * expert + 10 * int(w[1]) + _KINDS.index(kind))
    if kind == "weight_packed":
        return torch.randint(0, 256, (4, 8), generator=gen, dtype=torch.uint8)
    return torch.randint(0, 256, (4, 1), generator=gen, dtype=torch.uint8)


def _record_expert(experts, base, ckpt_expert_idx, slot_expert_idx, local_slot_id, get_tensor):
    prefix = f"{base}.{ckpt_expert_idx}"
    # A copy: the streamed views live in a buffer the next shard overwrites.
    experts.got[local_slot_id] = {
        f"{w}.{kind}": get_tensor(f"{prefix}.{w}.{kind}").clone()
        for w in ("w1", "w2", "w3")
        for kind in _KINDS
    }
    experts.loaded.add(local_slot_id)


def _write_expert_checkpoint(directory, with_index: bool = True):
    """Experts 0-7 of layer 0 in two shards, with expert 5's w2 in the other shard, under ``language_model.``."""
    shards = {"a.safetensors": {}, "b.safetensors": {}}
    for expert in range(8):
        for w in ("w1", "w2", "w3"):
            for kind in _KINDS:
                shard = (
                    "a.safetensors"
                    if (expert < 4) != (expert == 5 and w == "w2")
                    else "b.safetensors"
                )
                key = f"language_model.{_EXPERT_BASE}.{expert}.{w}.{kind}"
                shards[shard][key] = _expert_tensor(expert, w, kind)
    weight_map = {}
    for name, tensors in shards.items():
        save_file(tensors, str(directory / name))
        weight_map.update(dict.fromkeys(tensors, name))
    if with_index:
        import json

        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )


def _moe(target_weights, expert_ids):
    spec = target_weights.ExpertCkptSpec(
        kinds=_KINDS, loader=_record_expert, loaded_slots_attr="loaded"
    )
    experts = SimpleNamespace(got={}, loaded=set(), expert_size_per_partition=len(expert_ids))
    return SimpleNamespace(
        expert_ckpt_spec=spec,
        ckpt_expert_ids=list(expert_ids),
        local_expert_ids=list(expert_ids),
        routed_experts=experts,
    )


def _assert_loaded(moe):
    for slot, expert in enumerate(moe.ckpt_expert_ids):
        got = moe.routed_experts.got[slot]
        for w in ("w1", "w2", "w3"):
            for kind in _KINDS:
                assert torch.equal(got[f"{w}.{kind}"], _expert_tensor(expert, w, kind)), (
                    expert,
                    w,
                    kind,
                )


@pytest.mark.parametrize("path", ["mapped file", "lazy dict"], ids=["safe_open", "no-index"])
def test_routed_experts_load_this_ranks_slots(target_weights, path, tmp_path, monkeypatch):
    """Experts 2, 3 and 6 stream per shard file, expert 5 reads its two shards; every slot gets its checkpoint
    tensors whichever path reads them."""
    _write_expert_checkpoint(tmp_path, with_index=path != "lazy dict")
    if path == "mapped file":
        monkeypatch.setattr(target_weights, "read_safetensors_slices", lambda *args: None)
    moe = _moe(target_weights, [2, 3, 5, 6])
    llm = _llm([SimpleNamespace(is_moe=True, block_sparse_moe=moe)])
    with _lazy_checkpoint(tmp_path) as weights:
        lm_weights = target_weights.strip_prefix(weights, target_weights.LANG_PREFIX)
        target_weights._load_expert_slices(llm, lm_weights, [(moe, _EXPERT_BASE)])
    assert moe.routed_experts.loaded == {0, 1, 2, 3}
    _assert_loaded(moe)
