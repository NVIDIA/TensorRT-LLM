# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A/B benchmark for the Wan 2.2 NVFP4 layer-fusion path.

Loads the calibrated ModelOpt-quantized Wan 2.2 T2V 14B transformer once,
then runs the same forward pass in two configurations within a single
process. Each configuration is wrapped in a distinct NVTX range so nsys
can separate them cleanly:

* ``Wan22 NVFP4 baseline (fusion=off)``
* ``Wan22 NVFP4 fused (fusion=on)``

Inside each region we burn warmup iterations (excluded from the
cuda-event timing) and then collect ``--iters`` timed steps. We log
wall time per iteration and a quick latency delta.

Why a single process: avoids re-loading the 14B safetensors twice and
keeps the autotuner cache + page cache identical between A and B,
isolating the kernel-level effect we want to measure.

Usage (in the build container):

    # quick A/B without nsys (just wall-time deltas)
    python3 tests/unittest/_torch/visual_gen/bench_wan22_nvfp4_fusion.py \\
        --warmup 5 --iters 20

    # full nsys trace, one capture covers both regions
    nsys profile -o wan22_ab.nsys-rep \\
        --trace=cuda,nvtx,osrt --cuda-event-trace=true \\
        python3 tests/unittest/_torch/visual_gen/bench_wan22_nvfp4_fusion.py \\
            --warmup 5 --iters 20

    # NVTX summary diff
    nsys stats --report nvtxsum wan22_ab.nsys-rep \\
        | tee wan22_ab.nvtxsum.txt

Override checkpoint path:
    WAN22_T2V_NVFP4_PATH=/path/to/Wan2.2-T2V-A14B-Diffusers-NVFP4 python3 ...
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any

# Prepend the source tree to sys.path so this bench script picks up
# in-flight Python edits (e.g. the Fp4QuantizedTensor reshape fix in
# tensorrt_llm/_torch/modules/linear.py) instead of the older installed
# wheel under ~/.local/lib/...
# Layout: <repo_root>/tests/unittest/_torch/visual_gen/bench_*.py
#   -> repo_root is parents[4]
_REPO_ROOT = Path(__file__).resolve().parents[4]
if (_REPO_ROOT / "tensorrt_llm" / "__init__.py").exists():
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("TLLM_DISABLE_MPI", "1")
# Construct LayerNorms with is_nvfp4=True; we toggle the actual fused path
# per-region via the module attributes below.
os.environ.setdefault("TRTLLM_DISABLE_NVFP4_LAYERNORM_FUSION", "0")


def _checkpoint_path() -> str:
    """Resolve the calibrated NVFP4 checkpoint path."""
    candidates = [
        os.environ.get("WAN22_T2V_NVFP4_PATH"),
        "/models/Wan2.2-T2V-A14B-Diffusers-NVFP4",
        "/home/scratch.trt_llm_data_ci/llm-models/Wan2.2-T2V-A14B-Diffusers-NVFP4",
    ]
    for c in candidates:
        if c and Path(c, "transformer", "config.json").exists():
            return c
    raise FileNotFoundError(
        "Wan 2.2 NVFP4 checkpoint not found. Set WAN22_T2V_NVFP4_PATH or "
        "place the model at /models/Wan2.2-T2V-A14B-Diffusers-NVFP4."
    )


def _snapshot_and_force_fusion(model: Any, *, active: bool) -> None:
    """Snapshot the post-load fused-state, then toggle every per-module switch.

    On first call records the original ``is_nvfp4`` / ``nvfp4_scale`` /
    ``_use_fused_gelu_tanh_quant`` flags; subsequent calls flip them
    according to ``active``. Mirrors
    ``test_wan22_nvfp4_fusion_integration._set_fusion_active`` so the
    benchmark and correctness test share the same A/B semantics.
    """
    from tensorrt_llm._torch.modules.layer_norm import LayerNorm
    from tensorrt_llm._torch.modules.mlp import MLP

    for _, norm in model.named_modules():
        if not isinstance(norm, LayerNorm):
            continue
        if not hasattr(norm, "_is_nvfp4_orig"):
            norm._is_nvfp4_orig = getattr(norm, "is_nvfp4", False)
            norm._nvfp4_scale_orig = getattr(norm, "nvfp4_scale", None)
        if active:
            norm.is_nvfp4 = norm._is_nvfp4_orig
            norm.nvfp4_scale = norm._nvfp4_scale_orig
        else:
            norm.is_nvfp4 = False
            norm.nvfp4_scale = None
    for _, mlp in model.named_modules():
        if not isinstance(mlp, MLP):
            continue
        if not hasattr(mlp, "_use_fused_gelu_tanh_quant_orig"):
            mlp._use_fused_gelu_tanh_quant_orig = getattr(
                mlp, "_use_fused_gelu_tanh_quant", False
            )
        mlp._use_fused_gelu_tanh_quant = (
            mlp._use_fused_gelu_tanh_quant_orig if active else False
        )


# Latent shape presets. The transformer takes a 5-D latent
# [B, C=16, T, H, W] and patchifies with kernel=(1,2,2), yielding
# M = B * T * (H/2) * (W/2) tokens fed to the LayerNorm/Linear fused kernels.
#
# Pixel-space → latent math:
#   - VAE: spatial 8x downsample, temporal 4x (+1) downsample.
#   - 720p == 720x1280 pixels → 90x160 latent  (H=90, W=160).
#   - 81 pixel frames → 21 latent frames       (T=21).
#   - 480p == 480x832 pixels → 60x104 latent   (H=60, W=104).
#   - 1 pixel frame → 1 latent frame           (T=1).
_RESOLUTION_PRESETS = {
    "480p_1frame": dict(B=1, C=16, T=1, H=60, W=104),
    # Reviewer-requested production default for Wan 2.2 NVFP4. M jumps from
    # 1560 (480p/1f) to 75600 (720p/81f) -- a 48x increase. Tests system-
    # scale behavior of the fused kernels at the actual production token
    # count.
    "720p_81frames": dict(B=1, C=16, T=21, H=90, W=160),
}


def _build_inputs(resolution: str, device: str = "cuda") -> tuple[Any, Any, Any]:
    """Construct (latent, timestep, text_emb) for the chosen resolution preset.

    Text condition is fixed at 77 tokens of hidden_size 4096 (UMT5 output
    surface for Wan 2.2). Single batch.
    """
    import torch

    if resolution not in _RESOLUTION_PRESETS:
        raise ValueError(
            f"unknown resolution {resolution!r}; choose from "
            f"{sorted(_RESOLUTION_PRESETS)}")
    cfg = _RESOLUTION_PRESETS[resolution]
    B, C, T, H, W = cfg["B"], cfg["C"], cfg["T"], cfg["H"], cfg["W"]
    num_tokens = B * T * (H // 2) * (W // 2)
    print(f"  latent shape: B={B} C={C} T={T} H={H} W={W}")
    print(f"  -> post-patchify tokens (M): {num_tokens}")

    torch.manual_seed(2026)
    text_seq_len = 77
    hidden_states = torch.randn(B, C, T, H, W, device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([500.0], device=device, dtype=torch.float32)
    encoder_hidden_states = torch.randn(
        B, text_seq_len, 4096, device=device, dtype=torch.bfloat16
    )
    return hidden_states, timestep, encoder_hidden_states


def _cosine_similarity(a: Any, b: Any) -> float:
    """Tensor-wide cosine similarity between two equally-shaped tensors.

    Operates in fp32 to avoid bf16 dot-product saturation; returns a Python
    float for clean logging.
    """
    a32 = a.float().flatten()
    b32 = b.float().flatten()
    denom = a32.norm() * b32.norm()
    if denom.item() == 0.0:
        return float("nan")
    return (a32.dot(b32) / denom).item()


def _capture_output(model: Any, inputs: tuple[Any, Any, Any]) -> Any:
    """One forward pass, full output captured for numerical A/B comparison."""
    import torch

    hs, ts, enc = inputs
    with torch.inference_mode():
        return model(hidden_states=hs, timestep=ts, encoder_hidden_states=enc)


def _load_model(ckpt_path: str) -> Any:
    """Load the calibrated Wan 2.2 NVFP4 transformer checkpoint."""
    import safetensors.torch as st
    import torch

    from tensorrt_llm._torch.visual_gen.config import (
        DiffusionModelConfig,
        VisualGenArgs,
    )
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import (
        WanTransformer3DModel,
    )

    args = VisualGenArgs(model=ckpt_path)
    model_config = DiffusionModelConfig.from_pretrained(ckpt_path, args=args)
    assert model_config.force_dynamic_quantization is False, (
        "Expected static NVFP4 from this checkpoint. The benchmark only "
        "makes sense when fused paths actually fire."
    )
    model = WanTransformer3DModel(model_config=model_config).to("cuda").eval()

    weights: dict = {}
    transformer_dir = Path(ckpt_path, "transformer")
    shard_files = sorted(transformer_dir.glob("diffusion_pytorch_model-*.safetensors"))
    for shard in shard_files:
        weights.update(st.load_file(str(shard), device="cpu"))
    model.load_weights(weights)
    model.post_load_weights()
    del weights
    gc.collect()
    torch.cuda.empty_cache()
    return model


def _time_region(
    model: Any,
    inputs: tuple[Any, Any, Any],
    *,
    warmup: int,
    iters: int,
    label: str,
    color: str,
) -> float:
    """Run timed iterations inside a single NVTX range.

    Burns ``warmup`` untimed iterations to warm caches / autotuners, then
    measures ``iters`` iterations wrapped in one NVTX range. Returns the
    average per-iteration latency in milliseconds.
    """
    import torch

    from tensorrt_llm._utils import nvtx_range

    with torch.inference_mode():
        hs, ts, enc = inputs

        # Warmup: bake the autotuner caches, kernel-selection state machine,
        # and graph capture (if any future capture lands) so the timed window
        # measures steady-state. Tagged so nsys can ignore it.
        with nvtx_range(f"{label} warmup", color="grey"):
            for _ in range(warmup):
                _ = model(hidden_states=hs, timestep=ts, encoder_hidden_states=enc)
            torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        with nvtx_range(label, color=color):
            for i in range(iters):
                starts[i].record()
                _ = model(hidden_states=hs, timestep=ts, encoder_hidden_states=enc)
                ends[i].record()
            torch.cuda.synchronize()

        per_iter_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        avg = sum(per_iter_ms) / len(per_iter_ms)
        p50 = sorted(per_iter_ms)[len(per_iter_ms) // 2]
        print(
            f"  [{label}] avg={avg:.3f} ms  p50={p50:.3f} ms  "
            f"min={min(per_iter_ms):.3f}  max={max(per_iter_ms):.3f}  n={iters}"
        )
        return avg


def main() -> None:
    """Run the benchmark CLI."""
    import torch

    from tensorrt_llm._utils import get_sm_version

    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--resolution",
        choices=sorted(_RESOLUTION_PRESETS),
        default="480p_1frame",
        help="Latent shape preset. 720p_81frames matches the Wan 2.2 NVFP4 "
        "production default (M=75600 tokens vs 1560 for 480p_1frame).",
    )
    parser.add_argument(
        "--order",
        choices=["baseline_first", "fused_first"],
        default="baseline_first",
        help="Run order. baseline_first is safer if the autotuner ever picks "
        "a different tactic in cold vs warm cache.",
    )
    parser.add_argument(
        "--numerical-ab",
        action="store_true",
        help="In addition to wall-time A/B, capture one full forward pass per "
        "configuration and report cosine similarity. Doubles peak GPU memory "
        "(stores two complete output tensors).",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only one fused forward pass and report shape/finiteness "
        "(no warmup, no iters). Useful as the first sanity check at high "
        "resolution where the full A/B may OOM.",
    )
    args = parser.parse_args()

    sm = get_sm_version()
    if not (100 <= sm < 120):
        print(f"This benchmark requires SM100 (Blackwell). Detected SM{sm}. Exiting.")
        sys.exit(0)

    ckpt_path = _checkpoint_path()
    print(f"checkpoint: {ckpt_path}")
    print(f"resolution preset: {args.resolution}")
    print("loading model ...")
    model = _load_model(ckpt_path)
    inputs = _build_inputs(args.resolution)

    if args.smoke_only:
        print()
        print("-" * 72)
        print("Smoke run (fused path on, single forward, no timing):")
        _snapshot_and_force_fusion(model, active=True)
        with torch.inference_mode():
            out = model(
                hidden_states=inputs[0],
                timestep=inputs[1],
                encoder_hidden_states=inputs[2],
            )
        torch.cuda.synchronize()
        finite = torch.isfinite(out).all().item()
        print(f"  output shape: {tuple(out.shape)}")
        print(f"  matches input shape: {tuple(out.shape) == tuple(inputs[0].shape)}")
        print(f"  all finite: {finite}")
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  peak CUDA memory: {peak_mb:.1f} MB")
        print("-" * 72)
        return

    runs = [
        ("Wan22 NVFP4 baseline (fusion=off)", False, "blue"),
        ("Wan22 NVFP4 fused (fusion=on)", True, "green"),
    ]
    if args.order == "fused_first":
        runs.reverse()

    latencies = {}
    outputs = {}
    for label, active, color in runs:
        _snapshot_and_force_fusion(model, active=active)
        if args.numerical_ab:
            # Capture one untimed forward for the cos-sim comparison.
            outputs[label] = _capture_output(model, inputs).detach().to(torch.float32).cpu()
            torch.cuda.synchronize()
        latencies[label] = _time_region(
            model,
            inputs,
            warmup=args.warmup,
            iters=args.iters,
            label=label,
            color=color,
        )

    print()
    print("=" * 72)
    print("Summary:")
    base_key = next(k for k in latencies if "baseline" in k)
    fused_key = next(k for k in latencies if "fused" in k)
    base_ms = latencies[base_key]
    fused_ms = latencies[fused_key]
    speedup = base_ms / fused_ms if fused_ms > 0 else float("inf")
    delta_pct = 100.0 * (base_ms - fused_ms) / base_ms
    print(f"  resolution: {args.resolution}")
    print(f"  baseline (fusion=off): {base_ms:.3f} ms / iter")
    print(f"  fused    (fusion=on ): {fused_ms:.3f} ms / iter")
    print(f"  speedup: {speedup:.3f}x   (latency drop {delta_pct:+.2f}%)")
    if args.numerical_ab:
        cs = _cosine_similarity(outputs[base_key], outputs[fused_key])
        max_abs = (outputs[base_key] - outputs[fused_key]).abs().max().item()
        print(f"  cosine sim (fused vs baseline output): {cs:.6f}")
        print(f"  max abs diff:                          {max_abs:.4e}")
    print("=" * 72)


if __name__ == "__main__":
    main()
