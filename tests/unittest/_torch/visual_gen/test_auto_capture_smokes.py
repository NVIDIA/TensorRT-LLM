# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto-path capture smoke tests — guards against silent fusion regressions.

For each family adapter:
1. Import the adapter (catches Diffusers minor-bump renames at import time).
2. Build `example_inputs` against the real Diffusers transformer (catches
   signature drift).
3. `torch.export` the transformer + apply rewrites.
4. Assert the rewritten GraphModule contains the expected
   `visgen_auto::sdpa` / `visgen_auto::dit_qk_norm_rope` node counts.

The node-count assertion is the bit that catches **silent perf
regressions** from PyTorch / Diffusers bumps: if a decomposition change
breaks the QK-rope pattern matcher, the fusion falls back to eager. The
test runs end-to-end without errors but flags the missing fusion.

Heavy: each capture builds the real Diffusers transformer (BF16 weights).
Gated on `LLM_MODELS_ROOT` + `RUN_VISGEN_AUTO_SMOKES=1`. CI stage operator
opts in by setting the env var on appropriate GPU runners.
"""

import os
from pathlib import Path

import pytest
import torch

_RUN_FLAG = os.environ.get("RUN_VISGEN_AUTO_SMOKES", "").lower() in ("1", "true", "yes")


def _llm_models_root() -> Path:
    """Mirror the resolution order used elsewhere in `tests/unittest/_torch/visual_gen/`
    (see ``test_model_loader.py``): respect ``LLM_MODELS_ROOT`` env if set,
    else fall back to the two TRT-LLM CI mount points. Returned path may
    not exist — callers must check before use (we skip the test on miss
    rather than fail, so this is usable on workstations without the CI
    data mount).
    """
    if "LLM_MODELS_ROOT" in os.environ:
        return Path(os.environ["LLM_MODELS_ROOT"])
    for candidate in (
        Path("/home/scratch.trt_llm_data_ci/llm-models/"),
        Path("/scratch.trt_llm_data/llm-models/"),
    ):
        if candidate.exists():
            return candidate
    # Neither set nor available; return a non-existent sentinel so the
    # `ckpt.exists()` check in the test body produces a clean skip with
    # a helpful path in the message.
    return Path("/llm-models-not-configured")


_FAMILIES = [
    pytest.param(
        "FluxPipeline",
        "FLUX.1-dev",
        # FLUX.1 has 19 dual-stream + 38 single-stream blocks → 57 total
        # attention sites. The QK-rope fusion replaces the (RMSNorm + RoPE)
        # cluster feeding each SDPA's Q/K with a single `dit_qk_norm_rope`
        # call; the SDPA itself stays (it's what runs the actual attention,
        # just with pre-fused Q/K inputs). So expect ≈ 57 of each, and a
        # missing-fusion regression shows up as `dit_qk_norm_rope < ~50`.
        {"dit_qk_norm_rope_ge": 50, "sdpa_ge": 50},
        id="flux1",
    ),
    pytest.param(
        "Flux2Pipeline",
        "FLUX.2-dev",
        # FLUX.2 has 8 dual + 48 single blocks → 56 total SDPA sites. After
        # the 2026-05-18 matcher extension (`_resolve_split_chunk_site`),
        # ALL 56 sites fuse: 48 single-stream go through the new
        # `linear → split_with_sizes → getitem → chunk(3, -1)` path, and
        # 8 dual-stream go through `_resolve_double_stream_site`. Floor of
        # 50 catches anything that regresses below "almost full coverage".
        {"dit_qk_norm_rope_ge": 50, "sdpa_ge": 50},
        id="flux2",
    ),
    pytest.param(
        "StableDiffusion3Pipeline",
        "stable-diffusion-3.5-medium",
        # SD3.5-medium has 24 MM-DiT blocks → 24 SDPA sites. No QK-RMSNorm
        # fusion path; SDPA stays as `visgen_auto::sdpa` (not fused into
        # `dit_qk_norm_rope`). Floor of 20 catches a 4-block regression.
        {"dit_qk_norm_rope_ge": 0, "sdpa_ge": 20},
        id="sd35",
    ),
]


@pytest.mark.gpu
@pytest.mark.skipif(not _RUN_FLAG, reason="Set RUN_VISGEN_AUTO_SMOKES=1 to opt in")
@pytest.mark.parametrize("pipeline_class,subdir,expected", _FAMILIES)
def test_family_capture_and_rewrite_node_count(pipeline_class, subdir, expected):
    """Capture + rewrite a real Diffusers transformer and assert the
    expected number of `visgen_auto::*` nodes appear in the GraphModule.

    This test fails LOUDLY if a Diffusers/PyTorch bump silently regresses
    the FX pattern matcher to no-op (fusion falls back to eager —
    everything still runs, just slower).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    root = _llm_models_root()
    ckpt = root / subdir
    if not ckpt.exists():
        pytest.skip(f"checkpoint {ckpt} not available")

    from diffusers import DiffusionPipeline

    from tensorrt_llm._torch.visual_gen.auto.auto_pipeline import _resolve_adapter
    from tensorrt_llm._torch.visual_gen.auto.capture import capture_transformer
    from tensorrt_llm._torch.visual_gen.auto.rewrite import apply_rewrites
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig

    # Build the Diffusers pipeline (real class signatures so we catch
    # adapter signature drift), then immediately drop the heavy ancillary
    # components — we only need the transformer for capture, and the
    # combined text-encoder + VAE + transformer set blows past 140 GiB
    # for FLUX.1 / FLUX.2 on a single H200. Move only the transformer
    # to CUDA.
    pipe = DiffusionPipeline.from_pretrained(str(ckpt), torch_dtype=torch.bfloat16)
    transformer = pipe.transformer
    # Free text encoders, VAE, etc. — they keep ~80 GiB of bf16 weights
    # alive on FLUX.2 between Mistral + T5 + VAE.
    for attr in (
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "tokenizer",
        "tokenizer_2",
        "tokenizer_3",
        "vae",
    ):
        if hasattr(pipe, attr):
            setattr(pipe, attr, None)
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    transformer = transformer.to("cuda")
    adapter = _resolve_adapter(type(transformer).__name__)
    model_config = DiffusionModelConfig.from_pretrained(str(ckpt))

    ep = capture_transformer(transformer, adapter, model_config)
    gm = ep.module()
    apply_rewrites(gm, adapter.rewrite_policy(model_config))

    # Count target node calls in the rewritten GM.
    sdpa_target = torch.ops.visgen_auto.sdpa.default
    qkr_target = torch.ops.visgen_auto.dit_qk_norm_rope.default
    sdpa_count = sum(
        1 for n in gm.graph.nodes if n.op == "call_function" and n.target is sdpa_target
    )
    qkr_count = sum(1 for n in gm.graph.nodes if n.op == "call_function" and n.target is qkr_target)

    if "dit_qk_norm_rope_ge" in expected:
        assert qkr_count >= expected["dit_qk_norm_rope_ge"], (
            f"{pipeline_class}: expected ≥{expected['dit_qk_norm_rope_ge']} "
            f"dit_qk_norm_rope nodes after rewrite, got {qkr_count}. "
            "Likely cause: a Diffusers or PyTorch bump changed the FX pattern "
            "the qk_rope_fusion matcher walks; fusion fell back to eager and "
            "perf regressed silently. Investigate `auto/qk_rope_fusion.py`."
        )
    if "sdpa_ge" in expected:
        assert sdpa_count >= expected["sdpa_ge"], (
            f"{pipeline_class}: expected ≥{expected['sdpa_ge']} "
            f"visgen_auto::sdpa nodes after rewrite, got {sdpa_count}. "
            "The SDPA rewriter's pattern probably stopped matching — "
            "`aten._scaled_dot_product_*` decomposition changed in PyTorch?"
        )


@pytest.mark.skipif(not _RUN_FLAG, reason="Set RUN_VISGEN_AUTO_SMOKES=1 to opt in")
def test_family_adapter_imports():
    """Pre-flight: every family adapter must import cleanly. Catches
    `from diffusers... import X` failures at import time rather than
    later under a confusing torch.export traceback.
    """
    from tensorrt_llm._torch.visual_gen.auto.families import (  # noqa: F401
        Flux2Adapter,
        FluxAdapter,
        LTX2Adapter,
        LTXAdapter,
        MMDiTAdapter,
        PixArtAdapter,
        SanaAdapter,
        SD3Adapter,
        WanAdapter,
    )
