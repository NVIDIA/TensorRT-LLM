# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration check for the Wan 2.2 NVFP4 layer-fusion path.

Loads the ModelOpt-quantized Wan2.2 T2V 14B checkpoint
(huggingface.co/nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4) into
``WanTransformer3DModel`` and verifies, in this order:

1.  ``DiffusionModelConfig.from_pretrained`` resolves
    ``quant_algo == NVFP4`` and ``force_dynamic_quantization == False``
    from the checkpoint's embedded ``quantization_config``.

2.  After ``model.load_weights(...)``, each *unignored* down-stream Linear
    (``attn1.qkv_proj``, ``attn2.to_q``, ``ffn.up_proj``) has its
    ``input_scale`` populated with a non-default calibrated value (we treat
    an unmodified ``Parameter(torch.empty([1]))`` placeholder as "not
    calibrated"; the ModelOpt checkpoint always overwrites it).

3.  After ``model.post_load_weights()``, ``_try_attach_nvfp4_scale``
    attaches ``nvfp4_scale`` to ``norm1`` / ``norm2`` / ``norm3`` of every
    unignored block (3..36) and **none** of the ignored blocks
    (0..2 + 37..39).

4.  At least one block's ``ffn._use_fused_gelu_tanh_quant`` is True and at
    least one ``norm{1,2,3}`` reports ``is_nvfp4`` + non-None
    ``nvfp4_scale`` -- i.e. the fused paths actually activate end-to-end.

The test deliberately stops *before* a full T2V pipeline run; the goal is
to catch checkpoint-format mismatches and gate-wiring bugs in seconds
rather than minutes. A separate forward-pass smoke test exercises a
single block end-to-end to confirm no runtime errors on the fused path.

Run:
    pytest -v -s tests/unittest/_torch/visual_gen/test_wan22_nvfp4_fusion_integration.py

Override checkpoint path:
    WAN22_T2V_NVFP4_PATH=/path/to/Wan2.2-T2V-A14B-Diffusers-NVFP4 \\
        pytest -v -s tests/unittest/_torch/visual_gen/test_wan22_nvfp4_fusion_integration.py

A/B kill switch (off-by-default fusion):
    TRTLLM_DISABLE_NVFP4_LAYERNORM_FUSION=0 pytest -v -s ...
"""

import fnmatch
import gc
import os
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.mlp import MLP
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, VisualGenArgs
from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.quantization import QuantAlgo


# Note: previously these two env vars were set at module import time via
# os.environ.setdefault(...). That leaked into any pytest session that
# happened to collect this file (even if all tests in it skipped) and could
# silently flip global toggles for unrelated tests. The autouse fixture
# below scopes the modifications to this module and restores the original
# values on teardown. Both env vars are read at function-call time (not at
# import) by tensorrt_llm._utils.mpi_disabled and by the _ln_quant_type
# helper in transformer_wan.py, so installing them via the fixture is
# sufficient even though the trtllm imports above already ran.
# See PR #14773 review feedback.
@pytest.fixture(scope="module", autouse=True)
def _wan22_fusion_test_env():
    """Module-scoped env-var toggle for the fusion integration suite.

    - TLLM_DISABLE_MPI=1 prevents the runtime from attempting MPI bring-up
      in single-process test environments.
    - TRTLLM_DISABLE_NVFP4_LAYERNORM_FUSION=0 overrides the production
      default (off) so the LayerNorm fusion path is actually exercised.
    """
    keys = ("TLLM_DISABLE_MPI", "TRTLLM_DISABLE_NVFP4_LAYERNORM_FUSION")
    previous = {k: os.environ.get(k) for k in keys}
    os.environ["TLLM_DISABLE_MPI"] = "1"
    os.environ["TRTLLM_DISABLE_NVFP4_LAYERNORM_FUSION"] = "0"
    try:
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# Wan 2.2 ModelOpt ``ignore`` patterns (mirrors transformer/config.json).
# We assert that all blocks 0..2 + 37..39 stay BF16 and the corresponding
# norms do NOT receive an ``nvfp4_scale``.
_IGNORED_BLOCK_INDICES = {0, 1, 2, 37, 38, 39}
_UNIGNORED_BLOCK_INDICES = set(range(40)) - _IGNORED_BLOCK_INDICES


# ----------------------------------------------------------------------
# Skip gates
# ----------------------------------------------------------------------


def _checkpoint_path() -> str:
    """Resolve the calibrated NVFP4 checkpoint path. Prefer the env var; fall
    back to the conventional ``/models`` bind-mount used by the build
    container, and finally the host scratch path."""
    candidates = [
        os.environ.get("WAN22_T2V_NVFP4_PATH"),
        "/models/Wan2.2-T2V-A14B-Diffusers-NVFP4",
        "/home/scratch.anikaj_libs/trunk_08042025/models/Wan2.2-T2V-A14B-Diffusers-NVFP4",
    ]
    for c in candidates:
        if c and Path(c, "transformer", "config.json").exists():
            return c
    return ""


_CKPT_PATH = _checkpoint_path()


skip_if_no_checkpoint = pytest.mark.skipif(
    not _CKPT_PATH,
    reason="Wan 2.2 NVFP4 checkpoint not found. Set WAN22_T2V_NVFP4_PATH or place "
    "the model at /models/Wan2.2-T2V-A14B-Diffusers-NVFP4.",
)


def _has_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        sm = get_sm_version()
    except RuntimeError:
        # get_sm_version() raises RuntimeError when CUDA device query fails
        # (e.g., driver not loaded). Per the TRT-LLM coding guideline that
        # except clauses should be narrowed to expected errors, only swallow
        # that specific failure; anything else (e.g., ImportError) should
        # propagate so it isn't masked.
        return False
    return 100 <= sm < 120


skip_if_no_sm100 = pytest.mark.skipif(
    not _has_sm100(),
    reason="Wan 2.2 NVFP4 fused kernels require SM100 (Blackwell). Skipping on non-Blackwell.",
)


skip_if_no_fused_op = pytest.mark.skipif(
    not hasattr(torch.ops.trtllm, "fused_layernorm_quantize"),
    reason="trtllm::fused_layernorm_quantize op is not registered. Rebuild the wheel.",
)


# ----------------------------------------------------------------------
# Module-scoped model loader (load once, share across tests)
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_wan22_nvfp4():
    """Load the calibrated NVFP4 transformer once for the module."""
    if not _CKPT_PATH:
        pytest.skip("checkpoint missing")

    args = VisualGenArgs(model=_CKPT_PATH)
    model_config = DiffusionModelConfig.from_pretrained(_CKPT_PATH, args=args)

    # Sanity: the loader must have picked up the embedded quantization_config.
    assert model_config.quant_config.quant_algo == QuantAlgo.NVFP4, (
        f"Expected NVFP4 quant_algo from embedded checkpoint config, got "
        f"{model_config.quant_config.quant_algo!r}. Did the diffusers layout "
        f"loader drop the quantization_config field?"
    )
    assert model_config.force_dynamic_quantization is False, (
        f"Expected static activation quant (input_activations.dynamic=false in "
        f"checkpoint), got force_dynamic_quantization="
        f"{model_config.force_dynamic_quantization}. Fused paths would silently "
        f"stay off."
    )

    model = WanTransformer3DModel(model_config=model_config).to("cuda").eval()

    # Stream the sharded safetensors into a single flat dict matching the
    # HF state-dict layout, then hand off to the model's loader (which
    # remaps ffn.net.0.proj -> ffn.up_proj, ffn.net.2 -> ffn.down_proj, etc).
    import safetensors.torch as st

    weights: dict = {}
    transformer_dir = Path(_CKPT_PATH, "transformer")
    shard_files = sorted(transformer_dir.glob("diffusion_pytorch_model-*.safetensors"))
    assert shard_files, f"No sharded safetensors found under {transformer_dir}"
    for shard in shard_files:
        weights.update(st.load_file(str(shard), device="cpu"))

    model.load_weights(weights)
    model.post_load_weights()
    del weights
    gc.collect()
    torch.cuda.empty_cache()

    yield model, model_config

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _block_index_from_name(name: str) -> int:
    """Extract the block index from a fully-qualified module name like
    ``blocks.17.ffn.down_proj`` -- returns -1 if the module is not under a
    transformer block."""
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "blocks":
        try:
            return int(parts[1])
        except ValueError:
            return -1
    return -1


def _is_ignored_by_modelopt(qual_name: str, ignore_patterns: list) -> bool:
    """Mirror the ModelOpt ``ignore`` semantics used by NVFP4LinearMethod:
    a Linear is excluded when its name matches any glob in the list."""
    for pat in ignore_patterns or []:
        if fnmatch.fnmatch(qual_name, pat):
            return True
    return False


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


@skip_if_no_checkpoint
@skip_if_no_sm100
@skip_if_no_fused_op
def test_quant_config_resolves_static_nvfp4(loaded_wan22_nvfp4):
    """Assertion #1: checkpoint resolves to static NVFP4."""
    _model, model_config = loaded_wan22_nvfp4
    qc = model_config.quant_config
    assert qc.quant_algo == QuantAlgo.NVFP4
    assert qc.group_size == 16
    assert qc.exclude_modules is not None and len(qc.exclude_modules) > 0
    assert model_config.force_dynamic_quantization is False


@skip_if_no_checkpoint
@skip_if_no_sm100
@skip_if_no_fused_op
def test_calibrated_input_scales_populated(loaded_wan22_nvfp4):
    """Assertion #2: every unignored downstream Linear has a calibrated
    input_scale (non-default value, populated by the ModelOpt loader)."""
    model, model_config = loaded_wan22_nvfp4
    ignore = list(model_config.quant_config.exclude_modules or [])

    inspected, populated = 0, 0
    for name, module in model.named_modules():
        if not isinstance(module, Linear):
            continue
        if not getattr(module, "has_nvfp4", False):
            continue
        if _is_ignored_by_modelopt(name, ignore):
            continue
        inspected += 1
        ipsc = getattr(module, "input_scale", None)
        # The placeholder Parameter from create_weights is torch.empty([1])
        # with uninitialized values. The ModelOpt loader overwrites it with
        # the per-tensor activation scale (FP32, always > 0 for non-degenerate
        # layers). Use isfinite + nonzero as the calibration sentinel.
        if (
            ipsc is not None
            and torch.is_tensor(ipsc)
            and torch.isfinite(ipsc).all()
            and ipsc.abs().sum().item() > 0.0
        ):
            populated += 1

    assert inspected > 0, "No NVFP4 Linears found in the model -- loader misconfigured?"
    assert populated == inspected, (
        f"Only {populated}/{inspected} unignored NVFP4 Linears have a calibrated "
        f"input_scale. Expected all of them. The fused fast paths can only "
        f"activate where input_scale is real."
    )


@skip_if_no_checkpoint
@skip_if_no_sm100
@skip_if_no_fused_op
def test_layernorm_nvfp4_scale_attached_on_unignored_blocks(loaded_wan22_nvfp4):
    """Assertion #3: norm{1,2,3} have nvfp4_scale attached for every
    unignored block and zero attached on ignored blocks."""
    model, _ = loaded_wan22_nvfp4

    attached = {1: set(), 2: set(), 3: set()}
    not_attached = {1: set(), 2: set(), 3: set()}
    for idx, block in enumerate(model.blocks):
        for norm_idx, norm in [(1, block.norm1), (2, block.norm2), (3, block.norm3)]:
            if not isinstance(norm, LayerNorm):
                # AdaLN may live as a different module type on some configs;
                # skip but do not fail.
                continue
            has_scale = getattr(norm, "nvfp4_scale", None) is not None
            (attached if has_scale else not_attached)[norm_idx].add(idx)

    # Every unignored block should have nvfp4_scale on norm1, norm2, norm3.
    for n in (1, 2, 3):
        missing = _UNIGNORED_BLOCK_INDICES - attached[n]
        assert not missing, (
            f"norm{n} is missing nvfp4_scale on unignored blocks {sorted(missing)}. "
            f"Check _try_attach_nvfp4_scale in WanTransformer3DModel.post_load_weights."
        )
        unexpected = _IGNORED_BLOCK_INDICES & attached[n]
        assert not unexpected, (
            f"norm{n} has nvfp4_scale on ignored blocks {sorted(unexpected)}; the "
            f"downstream Linear should be BF16 (no input_scale) for those blocks."
        )


@skip_if_no_checkpoint
@skip_if_no_sm100
@skip_if_no_fused_op
def test_mlp_fused_gate_activates(loaded_wan22_nvfp4):
    """Assertion #4a: at least one block's FFN has _use_fused_gelu_tanh_quant
    set, and the count matches the unignored-block count exactly."""
    model, _ = loaded_wan22_nvfp4

    fused, total_mlp = [], 0
    for idx, block in enumerate(model.blocks):
        mlp = block.ffn
        if not isinstance(mlp, MLP):
            continue
        total_mlp += 1
        if getattr(mlp, "_use_fused_gelu_tanh_quant", False):
            fused.append(idx)

    assert total_mlp == 40, f"Expected 40 MLP blocks, got {total_mlp}"
    fused_set = set(fused)
    assert fused_set == _UNIGNORED_BLOCK_INDICES, (
        f"Fused gelu_tanh+NVFP4 MLP path is on for blocks {sorted(fused_set)}, "
        f"expected exactly {sorted(_UNIGNORED_BLOCK_INDICES)} (the unignored "
        f"set). Mismatch typically means the MLP gate in mlp.py is using a "
        f"different signal than the ModelOpt ignore list."
    )


@skip_if_no_checkpoint
@skip_if_no_sm100
@skip_if_no_fused_op
def test_layernorm_fused_path_advertised(loaded_wan22_nvfp4):
    """Assertion #4b: at least one norm reports is_nvfp4 + nvfp4_scale, i.e.
    the LayerNorm fused path will fire at forward time."""
    model, _ = loaded_wan22_nvfp4

    advertised = 0
    for _, module in model.named_modules():
        if not isinstance(module, LayerNorm):
            continue
        if getattr(module, "is_nvfp4", False) and getattr(module, "nvfp4_scale", None) is not None:
            advertised += 1

    # 34 unignored blocks * 3 LayerNorms = 102 expected attachments minimum.
    # Use a lower bound to stay robust to incidental architecture tweaks.
    assert advertised >= 3 * len(_UNIGNORED_BLOCK_INDICES), (
        f"Only {advertised} LayerNorms have both is_nvfp4=True and nvfp4_scale "
        f"populated. Expected >= {3 * len(_UNIGNORED_BLOCK_INDICES)} "
        f"(3 norms * {len(_UNIGNORED_BLOCK_INDICES)} unignored blocks)."
    )


def _set_fusion_active(model, active: bool):
    """Toggle every per-module signal that controls whether the fused
    NVFP4 paths fire at forward time. Restoring requires the original
    snapshot, which the caller is expected to capture via
    ``_snapshot_fusion_state`` BEFORE calling with active=False."""
    for _, norm in model.named_modules():
        if isinstance(norm, LayerNorm):
            if active:
                norm.is_nvfp4 = norm._is_nvfp4_orig
                norm.nvfp4_scale = norm._nvfp4_scale_orig
            else:
                norm.is_nvfp4 = False
                # Setting nvfp4_scale to None is belt-and-braces; the
                # is_nvfp4=False gate above already routes to _forward_unfused.
                norm.nvfp4_scale = None
    for _, mlp in model.named_modules():
        if isinstance(mlp, MLP):
            if active:
                mlp._use_fused_gelu_tanh_quant = mlp._use_fused_gelu_tanh_quant_orig
            else:
                mlp._use_fused_gelu_tanh_quant = False


def _snapshot_fusion_state(model):
    """Record the original fused-path flags so a later _set_fusion_active(True)
    call can restore them exactly. Idempotent if called twice."""
    for _, norm in model.named_modules():
        if isinstance(norm, LayerNorm) and not hasattr(norm, "_is_nvfp4_orig"):
            norm._is_nvfp4_orig = getattr(norm, "is_nvfp4", False)
            norm._nvfp4_scale_orig = getattr(norm, "nvfp4_scale", None)
    for _, mlp in model.named_modules():
        if isinstance(mlp, MLP) and not hasattr(mlp, "_use_fused_gelu_tanh_quant_orig"):
            mlp._use_fused_gelu_tanh_quant_orig = getattr(mlp, "_use_fused_gelu_tanh_quant", False)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


@skip_if_no_checkpoint
@skip_if_no_sm100
@skip_if_no_fused_op
@torch.inference_mode()
def test_fused_vs_unfused_output_matches(loaded_wan22_nvfp4):
    """A/B regression guard: fused and unfused paths must produce the same
    transformer output (cosine sim >= 0.995) on the calibrated checkpoint.

    Same model, two forward passes:
      A. Force all fused paths OFF (``is_nvfp4=False`` on every LayerNorm
         and ``_use_fused_gelu_tanh_quant=False`` on every MLP). LayerNorm
         emits BF16, Linear does its own static NVFP4 quant in
         ``NVFP4LinearMethod._input_prepare``. MLP runs separate
         ``gelu(approximate=tanh)`` + dynamic-aware quant in the Linear.
      B. Restore the original flags so both the fused LN+NVFP4 quant kernel
         AND the fused gelu_tanh+NVFP4 quant kernel fire.

    Drift between these two paths means either:
      - the fused kernel writes a different FP4 byte than
        fp4_quantize(unfused activation), OR
      - the swizzled SF layout differs between the two paths, OR
      - the modulation cast/precision differs.
    All three are real risks and would silently corrupt model output.
    """
    model, _ = loaded_wan22_nvfp4
    _snapshot_fusion_state(model)

    torch.manual_seed(2026)
    B, C, T, H, W = 1, 16, 1, 60, 104
    text_seq_len = 77
    hidden_states = torch.randn(B, C, T, H, W, device="cuda", dtype=torch.bfloat16)
    timestep = torch.tensor([500.0], device="cuda", dtype=torch.float32)
    encoder_hidden_states = torch.randn(B, text_seq_len, 4096, device="cuda", dtype=torch.bfloat16)

    # Path A: fusion OFF (baseline).
    _set_fusion_active(model, active=False)
    out_a = (
        model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )
        .float()
        .clone()
    )

    # Path B: fusion ON.
    _set_fusion_active(model, active=True)
    out_b = (
        model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )
        .float()
        .clone()
    )

    cos = _cosine_similarity(out_a, out_b)
    # 0.995 is loose enough to absorb FP4 rounding noise between the two
    # quantization paths but tight enough to catch real bugs (a missing
    # bias add, wrong modulation order, broken SF layout, etc).
    assert cos >= 0.995, (
        f"Fused vs unfused output cosine similarity {cos:.5f} < 0.995. "
        f"Fused paths likely diverge numerically -- inspect modulation cast "
        f"order in layer_norm._forward_nvfp4_fused or the SF swizzle in "
        f"fusedLayerNormQuant.cu."
    )

    # Bonus diagnostics: also assert max abs diff is bounded, normalised by
    # the unfused output's typical magnitude. Helps a future failure
    # surface whether the drift is global (alpha bug) vs local (one block).
    abs_diff = (out_a - out_b).abs()
    rel_diff = abs_diff.max().item() / (out_a.abs().mean().item() + 1e-6)
    print(f"\n  fused vs unfused: cos={cos:.5f}, max|delta|/mean|out_a|={rel_diff:.3f}")


@skip_if_no_checkpoint
@skip_if_no_sm100
@skip_if_no_fused_op
@torch.inference_mode()
def test_forward_smoke_runs_without_error(loaded_wan22_nvfp4):
    """Run a single forward pass with a small latent + text condition to
    confirm the fused kernels execute end-to-end without shape/dtype/CUDA
    errors. We do NOT validate output values here -- that belongs in the
    fused-vs-unfused A/B numerical test."""
    model, _ = loaded_wan22_nvfp4

    # 60x104 spatial latent -> sequence length 1 * 30 * 52 = 1560 (matches
    # the Wan2.2 480p T2V pipeline), 1 latent frame.
    B, C, T, H, W = 1, 16, 1, 60, 104
    text_seq_len = 77
    hidden_states = torch.randn(B, C, T, H, W, device="cuda", dtype=torch.bfloat16)
    timestep = torch.tensor([500.0], device="cuda", dtype=torch.float32)
    encoder_hidden_states = torch.randn(B, text_seq_len, 4096, device="cuda", dtype=torch.bfloat16)

    out = model(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
    )
    assert out.shape == hidden_states.shape, (
        f"Output shape {out.shape} differs from input shape {hidden_states.shape}; "
        f"unpatchify probably misconfigured."
    )
    assert torch.isfinite(out).all(), (
        "Output contains NaN/Inf -- a fused kernel likely produced garbage "
        "(check input_scale wiring or alpha population)."
    )
