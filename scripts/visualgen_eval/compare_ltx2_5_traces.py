#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report native component errors against independently generated official traces."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def tensor_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Measure transparent BF16 parity without requiring bitwise identity."""
    result = {
        "shape": list(actual.shape),
        "reference_shape": list(expected.shape),
        "dtype": str(actual.dtype),
        "reference_dtype": str(expected.dtype),
    }
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {**result, "pass": False, "reason": "shape_or_dtype"}
    a, b = actual.double().flatten(), expected.double().flatten()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return {**result, "finite": False, "pass": False}
    delta = a - b
    norm, error = b.norm().item(), delta.norm().item()
    relative = error / norm if norm else (0.0 if error == 0 else None)
    cosine = F.cosine_similarity(a, b, dim=0).item() if a.norm() and norm else float(error == 0)
    absolute = delta.abs()
    return {
        **result,
        "finite": True,
        "bitwise_equal": torch.equal(actual, expected),
        "max_abs": absolute.max().item(),
        "mae": absolute.mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "normalized_rmse": relative,
        "cosine": cosine,
        "p99_abs_sampled": torch.quantile(
            absolute[:: max(1, absolute.numel() // 1000000)], 0.99
        ).item(),
        "allclose_1e3": torch.allclose(a, b, rtol=1e-3, atol=1e-3),
        "pass": relative is not None and relative <= 1e-2 and cosine >= 0.9999,
    }


def compare(candidate: dict, reference: dict, *, quality: bool = True) -> dict:
    """Keep initialization, steps and raw outputs separate in the diagnostic report."""
    if candidate["recipe"] != reference["recipe"]:
        raise ValueError("Candidate/reference generation recipes differ")
    if (
        len(candidate.get("initializations", [])) != 4
        or len(reference.get("initializations", [])) != 4
    ):
        raise ValueError("Expected all four video/audio noise initializations")
    for trace in (candidate, reference):
        stages = trace.get("stage_outputs", {})
        if set(stages) != {"stage_1", "upsampled_video", "stage_2"}:
            raise ValueError("Expected both denoising stages and latent upsampler outputs")
        if not isinstance(stages["upsampled_video"], torch.Tensor):
            raise ValueError("Expected upsampled_video to be a tensor")
        if any(set(stages[name]) != {"video", "audio"} for name in ("stage_1", "stage_2")):
            raise ValueError("Expected both video and audio stage outputs")
    rows = {}
    for key in ("video", "audio"):
        rows["contexts." + key] = tensor_metrics(
            candidate["contexts"][key], reference["contexts"][key]
        )
    for index, (a, b) in enumerate(
        zip(candidate.get("initializations", []), reference.get("initializations", []), strict=True)
    ):
        for key in ("noise", "initial_latent", "clean_latent", "denoise_mask"):
            rows[f"initializations.{index}.{key}"] = tensor_metrics(a[key], b[key])
    if len(candidate["steps"]) != 22 or len(reference["steps"]) != 22:
        raise ValueError("Expected all 22 video/audio Euler updates")
    for index, (a, b) in enumerate(zip(candidate["steps"], reference["steps"], strict=True)):
        if a["sigma"] != b["sigma"]:
            raise ValueError(f"Sigma mismatch at update {index}")
        for key in ("latent", "next"):
            rows[f"steps.{index}.{key}"] = tensor_metrics(a[key], b[key])
    for stage, value in candidate.get("stage_outputs", {}).items():
        if isinstance(value, dict):
            for key in value:
                rows[f"stage_outputs.{stage}.{key}"] = tensor_metrics(
                    value[key], reference["stage_outputs"][stage][key]
                )
        else:
            rows["stage_outputs." + stage] = tensor_metrics(
                value, reference["stage_outputs"][stage]
            )
    for key in ("video", "audio"):
        rows["raw." + key] = tensor_metrics(candidate[key], reference[key])
    lpips_score = None
    if quality:
        import lpips

        device = "cuda" if torch.cuda.is_available() else "cpu"
        metric = lpips.LPIPS(net="alex").to(device).eval()
        video, golden = candidate["video"].float(), reference["video"].float()
        with torch.inference_mode():
            lpips_score = (
                metric(
                    video.permute(0, 3, 1, 2).to(device) * 2 - 1,
                    golden.permute(0, 3, 1, 2).to(device) * 2 - 1,
                )
                .mean()
                .item()
            )
    failures = [name for name, value in rows.items() if not value["pass"]]
    return {
        "thresholds": {
            "component_normalized_rmse": 0.01,
            "component_cosine": 0.9999,
            "video_lpips": 0.05,
        },
        "tensors": rows,
        "first_failure": failures[0] if failures else None,
        "failures": failures,
        "lpips": lpips_score,
        "pass": not failures and (lpips_score is None or lpips_score < 0.05),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--no-quality", action="store_true")
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    torch.set_num_threads(8)
    result = compare(
        torch.load(args.candidate, map_location="cpu", weights_only=True),
        torch.load(args.reference, map_location="cpu", weights_only=True),
        quality=not args.no_quality,
    )
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "tensors"}, indent=2))
    if not result["pass"]:
        raise AssertionError(
            "Native parity failed; inspect per-tensor metrics and earliest difference"
        )


if __name__ == "__main__":
    main()
