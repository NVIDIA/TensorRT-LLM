# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Step-5 target's vision tower and projector against StepFun's reference.

The tower (``vision.py``) runs over the checkpoint's own vision weights, read through
the target's weight table (``weights.py``, the crop grid's resampled position embedding
included), on the tiles of ``merlion.png`` -- one overview and six crops -- and its 655
rows are judged against the two outputs of StepFun's ``get_multimodal_embeddings`` in
``oracle/vision/merlion.safetensors``: ``image_embeds`` (the reference run in bf16) and
``image_embeds_fp32`` (the same modules in fp32).

**The band.** The bf16 reference is itself far from the fp32 one after 47 blocks: its
per-row cosine against fp32 is ``_REF_MIN_COS`` at worst and ``_REF_MEAN_COS`` on
average. A bf16 tower cannot be held closer to either reference than that, and this one
differs from the bf16 reference on purpose in two places: it turns q and k by exact rope
angles where the reference rounds its angle table to bf16 first (0.39 off the exact
rotation at worst, ``vision_rope_2d_interleaved``'s note), and its ``quick_gelu`` rounds
once. So the tower is held, against each reference, to the bf16 reference's own
agreement with fp32, less a margin: per-row cosine at least ``_REF_MIN_COS - _MIN_MARGIN``
on every row and ``_REF_MEAN_COS - _MEAN_MARGIN`` on average. The rows in the wrong
order -- the overview's first, as a tower that ignored the crop order would emit them
-- land far outside it (checked).

GPU only. Needs ``STEP5_PREVIEW_CHECKPOINT`` (else ``$LLM_MODELS_ROOT/Step-5-Preview-NVFP4``)
and ``STEP5_PREVIEW_VISION_ORACLE``; skipped without them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.models.step5.step5_preview_nvfp4__sm_103__unified import (  # noqa: E501
    input_processor as ip,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.step5.step5_preview_nvfp4__sm_103__unified import (  # noqa: E501
    vision as V,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.step5.step5_preview_nvfp4__sm_103__unified import (  # noqa: E501
    weights as W,
)
from tensorrt_llm._torch.configs import Step5TextConfig, Step5VisionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

#: The bf16 reference against the fp32 one, per-row cosine over the 655 rows (measured
#: from the oracle file by `test_reference_agreement_is_what_the_band_assumes`).
_REF_MIN_COS = 0.9298
_REF_MEAN_COS = 0.9967
#: How far below the references' own agreement the tower may land.
_MIN_MARGIN = 0.02
_MEAN_MARGIN = 0.002

#: merlion.png: six crops, then the overview.
_MERLION_ROWS = 6 * 81 + 169


def _checkpoint_dir() -> Path | None:
    explicit = os.environ.get("STEP5_PREVIEW_CHECKPOINT")
    if explicit:
        return Path(explicit)
    root = llm_models_root()
    return None if root is None else Path(root) / "Step-5-Preview-NVFP4"


_CHECKPOINT = _checkpoint_dir()
_ORACLE = os.environ.get("STEP5_PREVIEW_VISION_ORACLE")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        not (_CHECKPOINT and (_CHECKPOINT / "config.json").is_file() and _ORACLE),
        reason="set STEP5_PREVIEW_CHECKPOINT and STEP5_PREVIEW_VISION_ORACLE",
    ),
]


def _vision_entries():
    return [e for e in W.MODEL_WEIGHTS.WEIGHTS if e.prefix == "v" or e.name.startswith("v_")]


def _core(max_num_tokens: int) -> SimpleNamespace:
    """What the tower and the weight table read off the core, at one rank."""
    config = json.loads((_CHECKPOINT / "config.json").read_text())
    quant = json.loads((_CHECKPOINT / "hf_quant_config.json").read_text())["quantization"]
    core = SimpleNamespace(
        text_config=Step5TextConfig(**config["text_config"]),
        composite_config=SimpleNamespace(
            vision_config=Step5VisionConfig(**config["vision_config"])
        ),
        model_config=SimpleNamespace(
            mapping=Mapping(world_size=1, tp_size=1),
            quant_config=QuantConfig(
                quant_algo=QuantAlgo.NVFP4,
                group_size=quant["group_size"],
                exclude_modules=quant["exclude_modules"],
            ),
            torch_dtype=torch.bfloat16,
            max_num_tokens=max_num_tokens,
        ),
    )
    core.dims = W.MODEL_WEIGHTS.dims(core)
    return core


@pytest.fixture(scope="module")
def vision_weights() -> dict:
    """The vision tower's parameters, each filled the way `load` fills it."""
    from safetensors import safe_open

    index = json.loads((_CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]
    core = _core(8192)
    entries = _vision_entries()
    keys = set()
    for entry in entries:
        for _, i in W.MODEL_WEIGHTS._targets(entry, core.dims):
            for src in entry.sources(core.dims, i):
                keys.update(src.key if isinstance(src.key, tuple) else (src.key,))
    tensors = {}
    for shard in sorted({index[k] for k in keys}):
        with safe_open(str(_CHECKPOINT / shard), framework="pt") as handle:
            for key in keys:
                if index[key] == shard:
                    tensors[key] = handle.get_tensor(key)
    vision_keys = {k for k in index if k.startswith(("vision_model.", "vit_large_projector."))}
    assert keys == vision_keys, sorted(keys ^ vision_keys)[:8]

    w = {}
    for entry in entries:
        for name, i in W.MODEL_WEIGHTS._targets(entry, core.dims):
            (src,) = entry.sources(core.dims, i)
            value = W.MODEL_WEIGHTS.read(tensors, src, "cuda")
            dtype = entry.dtype or core.dims.dtype
            assert tuple(value.shape) == entry.shape(core.dims) and value.dtype == dtype, name
            w[name] = value.contiguous()
    return w


def _tower(w: dict, max_num_tokens: int = 8192) -> V.VisionTower:
    core = _core(max_num_tokens)
    core.w = w
    return V.VisionTower(core)


@pytest.fixture(scope="module")
def oracle() -> dict:
    from safetensors import safe_open

    with safe_open(_ORACLE, framework="pt") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def _run(tower: V.VisionTower, oracle: dict) -> torch.Tensor:
    with torch.inference_mode():
        return tower(
            oracle["pixel_values"].to("cuda", torch.bfloat16),
            oracle["patch_pixel_values"].to("cuda", torch.bfloat16),
            oracle["num_patches"].tolist(),
        )


def _row_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a.double(), b.double().to(a.device), dim=-1)


def test_reference_agreement_is_what_the_band_assumes(oracle):
    """The band is anchored on the two references' own agreement; re-measure it."""
    cos = _row_cos(oracle["image_embeds"], oracle["image_embeds_fp32"])
    assert abs(cos.min().item() - _REF_MIN_COS) < 1e-4, cos.min().item()
    assert abs(cos.mean().item() - _REF_MEAN_COS) < 1e-4, cos.mean().item()


@pytest.mark.parametrize("reference", ["image_embeds", "image_embeds_fp32"])
def test_merlion_rows_match_the_reference(vision_weights, oracle, reference):
    out = _run(_tower(vision_weights), oracle)
    assert out.shape == (_MERLION_ROWS, 4096) and out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    cos = _row_cos(out, oracle[reference])
    print(f"{reference}: per-row cosine min {cos.min().item():.5f} mean {cos.mean().item():.6f}")
    assert cos.min().item() >= _REF_MIN_COS - _MIN_MARGIN, cos.min().item()
    assert cos.mean().item() >= _REF_MEAN_COS - _MEAN_MARGIN, cos.mean().item()
    # The overview's rows first, as a tower blind to the crop order would put them: the
    # band refuses it.
    swapped = torch.cat([out[6 * 81 :], out[: 6 * 81]])
    assert _row_cos(swapped, oracle[reference]).mean().item() < _REF_MEAN_COS - 10 * _MEAN_MARGIN


def test_cutting_the_tiles_into_calls_changes_no_row(vision_weights, oracle):
    """All seven tiles in one call against the smallest calls (one overview's rows, so
    four calls): attention is per tile and the rest row-wise, so the rows agree to the
    kernels' own shape-to-shape spread."""
    whole = _run(_tower(vision_weights, 16384), oracle)
    tilewise = _run(_tower(vision_weights, 1), oracle)
    cos = _row_cos(whole, tilewise)
    print(f"call cut: per-row cosine min {cos.min().item():.7f}")
    assert cos.min().item() > 0.9999, cos.min().item()


def test_the_profiling_tiles_run(vision_weights):
    """The processor's profiling payload at the default encoder budget runs through the
    tower in one call and yields one row per placeholder its images would carry."""
    from tensorrt_llm.inputs.registry import create_input_processor

    processor = create_input_processor(str(_CHECKPOINT), None)
    assert isinstance(processor, ip.Step5PreviewInputProcessor)
    budget = 8192
    data = processor.get_dummy_mm_data(max_num_encoder_tokens=budget, mm_counts={"image": 1})
    image = data["image"]
    assert image["num_patches"] == [4]
    tower = _tower(vision_weights, budget)
    assert len(tower._calls([V._Tile(False, j) for j in range(4)] + [V._Tile(True, 0)])) == 1
    with torch.inference_mode():
        out = tower(
            image["pixel_values"].cuda(), image["patch_pixel_values"].cuda(), image["num_patches"]
        )
    assert out.shape == (169 + 4 * 81, 4096)
