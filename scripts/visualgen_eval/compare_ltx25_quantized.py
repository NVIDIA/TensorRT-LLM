#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Separate unchanged inputs from lossy denoising drift and raw-media quality."""

import argparse
import json
import math
from pathlib import Path

import torch
from compare_ltx2_5_traces import compare


def validate_policy(policy):
    """Reject absent, negative or nonfinite quality limits before scoring."""
    keys = {
        "video_lpips",
        "audio_spectral_convergence",
        "audio_energy_weighted_log_error",
    }
    limits = policy.get("limits")
    if not isinstance(limits, dict) or set(limits) != keys:
        raise ValueError(f"Quality policy must define exactly these limits: {sorted(keys)}")
    for key, value in limits.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Quality limit {key} must be numeric")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Quality limit {key} must be finite and nonnegative")


def audio_spectra(actual, expected):
    """Score original stereo waveforms without resampling, alignment or gain changes."""
    rows = {}
    for size in (256, 1024, 2048):
        window = torch.hann_window(size)
        a, b = [
            torch.stft(
                w.float(), n_fft=size, hop_length=size // 4, window=window, return_complex=True
            ).abs()
            for w in (actual, expected)
        ]
        error = (a.clamp_min(1e-5).log() - b.clamp_min(1e-5).log()).abs()
        norm, energy = b.norm().item(), b.square().sum().item()
        rows[str(size)] = {
            "spectral_convergence": (a - b).norm().item() / norm if norm else None,
            "energy_weighted_log_error": (error * b.square()).sum().item() / energy
            if energy
            else None,
            "unweighted_log_error": error.mean().item(),
        }
    return rows


def evaluate(candidate, reference, policy=None, require_low_precision=True):
    """T1 input invariants remain gated; T2 values use an explicit media policy."""
    import lpips

    if policy is not None:
        validate_policy(policy)
    paired = compare(candidate, reference, quality=False)
    rows, invariant_failures, structural_failures = {}, [], []
    for name, value in paired["tensors"].items():
        invariant = name.startswith("contexts.") or name in ("steps.0.latent", "steps.1.latent")
        if name.startswith("initializations."):
            _, index, key = name.split(".")
            invariant = int(index) < 2 or key in ("noise", "denoise_mask")
        value = dict(value)
        value["within_transparent_band"] = value.pop("pass")
        value["classification"] = "T1 unchanged input" if invariant else "T2 propagated drift"
        if not value.get("finite", False):
            structural_failures.append(name)
        if invariant and not value["within_transparent_band"]:
            invariant_failures.append(name)
        rows[name] = value
    a, b = candidate["video"].float(), reference["video"].float()
    if a.min() < 0 or a.max() > 1 or b.min() < 0 or b.max() > 1:
        raise ValueError("Raw video must be FHWC in [0,1]")
    metric = lpips.LPIPS(net="alex").cuda().eval()
    with torch.inference_mode():
        video_lpips = (
            metric(a.permute(0, 3, 1, 2).cuda() * 2 - 1, b.permute(0, 3, 1, 2).cuda() * 2 - 1)
            .mean()
            .item()
        )
    spectra = audio_spectra(candidate["audio"], reference["audio"])
    failures = []
    if policy:
        limits = policy["limits"]
        if video_lpips >= limits["video_lpips"]:
            failures.append("video_lpips")
        for size, row in spectra.items():
            for key in ("spectral_convergence", "energy_weighted_log_error"):
                if row[key] is None or row[key] > limits["audio_" + key]:
                    failures.append(f"audio.{size}.{key}")
    active = (
        candidate["metrics"].get("fp8_linear_count", 0) > 0
        or candidate["metrics"].get("nvfp4_linear_count", 0) > 0
        or candidate["metrics"].get("quantized_attention_layers", 0) > 0
    )
    if require_low_precision and not active:
        structural_failures.append("no_active_fp8_linear")
    execution_order = [key for key in rows if key.startswith("contexts.")]
    for prefix in (
        "initializations.0.",
        "initializations.1.",
        *(f"steps.{index}." for index in range(16)),
        "stage_outputs.stage_1.",
        "stage_outputs.upsampled_video",
        "initializations.2.",
        "initializations.3.",
        *(f"steps.{index}." for index in range(16, 22)),
        "stage_outputs.stage_2.",
        "raw.",
    ):
        execution_order.extend(key for key in rows if key.startswith(prefix))
    return {
        "status": (
            "PASS" if not (invariant_failures or structural_failures or failures) else "FAIL"
        )
        if policy
        else "DIAGNOSTIC_ONLY",
        "policy": policy,
        "precision": candidate["metrics"].get("precision"),
        "fp8_linear_count": candidate["metrics"].get("fp8_linear_count"),
        "fp8_lora_targets": candidate["metrics"].get("fp8_lora_targets"),
        "nvfp4_linear_count": candidate["metrics"].get("nvfp4_linear_count", 0),
        "nvfp4_lora_targets": candidate["metrics"].get("nvfp4_lora_targets", 0),
        "attention": candidate["metrics"].get("attention", "vanilla"),
        "quantized_attention_layers": candidate["metrics"].get("quantized_attention_layers", 0),
        "t1_invariant_failures": invariant_failures,
        "structural_failures": structural_failures,
        "quality_failures": failures,
        "first_difference": next(
            (key for key in execution_order if not rows[key].get("bitwise_equal", False)), None
        ),
        "first_outside_transparent_band": next(
            (key for key in execution_order if not rows[key]["within_transparent_band"]), None
        ),
        "video_lpips": video_lpips,
        "audio_spectra": spectra,
        "raw_video": rows["raw.video"],
        "raw_audio": rows["raw.audio"],
        "tensors": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    torch.set_num_threads(8)
    policy = json.loads(args.policy.read_text()) if args.policy else None
    result = evaluate(
        torch.load(args.candidate, map_location="cpu", weights_only=True),
        torch.load(args.reference, map_location="cpu", weights_only=True),
        policy,
    )
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "tensors"}, indent=2))
    if (
        result["t1_invariant_failures"]
        or result["structural_failures"]
        or result["quality_failures"]
    ):
        raise AssertionError("Quantized pipeline comparison failed; inspect report")


if __name__ == "__main__":
    main()
