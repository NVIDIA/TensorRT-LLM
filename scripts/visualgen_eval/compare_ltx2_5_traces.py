#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Compare trusted local BF16 trace artifacts against an independent official run."""

import argparse
import json
from pathlib import Path

import lpips
import torch


def compare(candidate: dict, reference: dict) -> dict:
    """Require exact contexts/Euler states and the unchanged raw-video LPIPS gate."""
    if candidate["recipe"] != reference["recipe"]:
        raise ValueError("Candidate/reference generation recipes differ")
    if len(candidate["steps"]) != 22 or len(reference["steps"]) != 22:
        raise ValueError("Expected all 22 video/audio Euler traces")
    contexts_exact = all(
        torch.equal(candidate["contexts"][key], reference["contexts"][key])
        for key in ("video", "audio")
    )
    steps_exact = all(
        a["sigma"] == b["sigma"]
        and torch.equal(a["latent"], b["latent"])
        and torch.equal(a["next"], b["next"])
        for a, b in zip(candidate["steps"], reference["steps"])
    )
    video = candidate["video"].squeeze(0).float() / 255
    golden = reference["video"].float()
    if video.shape != golden.shape:
        raise ValueError("Candidate/reference video shapes differ")
    metric = lpips.LPIPS(net="alex").eval()
    with torch.inference_mode():
        score = (
            metric(video.permute(0, 3, 1, 2) * 2 - 1, golden.permute(0, 3, 1, 2) * 2 - 1)
            .mean()
            .item()
        )
    audio = candidate["audio"].squeeze(0)[:, : reference["audio"].shape[-1]]
    return {
        "contexts_exact": contexts_exact,
        "steps_exact": steps_exact,
        "lpips": score,
        "audio_rmse": (audio.float() - reference["audio"].float()).square().mean().sqrt().item(),
        "pass": contexts_exact and steps_exact and score < 0.05,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    torch.set_num_threads(8)
    result = compare(
        torch.load(args.candidate, map_location="cpu", weights_only=True),
        torch.load(args.reference, map_location="cpu", weights_only=True),
    )
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise AssertionError("Official BF16 parity/quality gate failed; inspect the saved report")


if __name__ == "__main__":
    main()
