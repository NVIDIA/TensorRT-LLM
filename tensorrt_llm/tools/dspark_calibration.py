# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline chain-confidence STS fitting from correctly paired observation shards.

This CPU tool imports PyTorch, not TensorRT-LLM or CUDA. Collection is a separate
runtime adapter: each row must join raw logits to full-block prefix-accept labels
from the same stamped draft sequence. Calibration is not task qualification or
an economic admission receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

__all__ = [
    "default_temperature_grid",
    "expected_calibration_error",
    "fit_sts_temperatures",
    "load_shards",
]

_EPS_PROB = 1e-8


def default_temperature_grid() -> torch.Tensor:
    """Return the established 41-point log-spaced STS grid over [0.1, 10]."""
    return torch.logspace(math.log10(0.1), math.log10(10.0), steps=41)


def expected_calibration_error(
    *, probs: torch.Tensor, targets: torch.Tensor, num_bins: int
) -> float:
    """Compute population-weighted cumulative-prefix calibration error."""
    probs = probs.reshape(-1).to(torch.float64).clamp(_EPS_PROB, 1.0 - _EPS_PROB)
    targets = targets.reshape(-1).to(torch.float64)
    total = probs.numel()
    if total == 0:
        return float("nan")
    bin_index = (probs * num_bins).long().clamp_(0, num_bins - 1)
    count = torch.zeros(num_bins, dtype=torch.float64)
    pred_sum = torch.zeros(num_bins, dtype=torch.float64)
    target_sum = torch.zeros(num_bins, dtype=torch.float64)
    count.scatter_add_(0, bin_index, torch.ones_like(probs))
    pred_sum.scatter_add_(0, bin_index, probs)
    target_sum.scatter_add_(0, bin_index, targets)
    denom = count.clamp_min(1.0)
    bin_error = (pred_sum / denom - target_sum / denom).abs()
    return float((bin_error * count).sum().item() / total)


def _validate_rows(logits: torch.Tensor, prefix_mask: torch.Tensor) -> None:
    if (
        not isinstance(logits, torch.Tensor)
        or not isinstance(prefix_mask, torch.Tensor)
        or logits.device.type != "cpu"
        or prefix_mask.device.type != "cpu"
        or not logits.is_floating_point()
        or logits.ndim != 2
        or logits.shape != prefix_mask.shape
        or logits.shape[0] == 0
        or logits.shape[1] == 0
    ):
        raise ValueError("CPU logits and prefix_mask must have identical nonempty [N, K] shapes")
    if not torch.isfinite(logits).all() or not torch.isfinite(prefix_mask).all():
        raise ValueError("Calibration observations must be finite")
    if not ((prefix_mask == 0) | (prefix_mask == 1)).all():
        raise ValueError("Prefix labels must be binary full-block acceptance observations")
    if (prefix_mask[:, 1:] > prefix_mask[:, :-1]).any():
        raise ValueError("A later accepted prefix cannot follow an earlier rejected prefix")


def fit_sts_temperatures(
    *, logits: torch.Tensor, prefix_mask: torch.Tensor, grid: torch.Tensor, num_bins: int = 15
) -> dict[str, list[float]]:
    """Fit each temperature left-to-right against cumulative survival ECE.

    Args:
        logits: Raw confidence logits, CPU tensor [N, K].
        prefix_mask: Binary cumulative-prefix labels, CPU tensor [N, K].
        grid: Positive finite temperature candidates, CPU tensor [M].
        num_bins: Positive number of calibration-error bins.
    """
    _validate_rows(logits, prefix_mask)
    if type(num_bins) is not int or num_bins < 1:
        raise ValueError("num_bins must be a positive integer")
    if (
        not isinstance(grid, torch.Tensor)
        or grid.device.type != "cpu"
        or grid.ndim != 1
        or grid.numel() == 0
        or not torch.isfinite(grid).all()
        or not (grid > 0).all()
    ):
        raise ValueError("The temperature grid must be a nonempty positive finite CPU vector")
    logits = logits.to(torch.float64)
    prefix_mask = prefix_mask.to(torch.float64)
    num_samples, block_size = logits.shape
    grid_values = grid.to(torch.float64).tolist()
    temperatures, ece_before, ece_after = [], [], []
    survival_at_one = torch.ones(num_samples, dtype=torch.float64)
    survival_fitted = torch.ones(num_samples, dtype=torch.float64)
    for position in range(block_size):
        position_logits = logits[:, position]
        position_target = prefix_mask[:, position]
        survival_at_one = survival_at_one * torch.sigmoid(position_logits)
        ece_before.append(
            expected_calibration_error(
                probs=survival_at_one, targets=position_target, num_bins=num_bins
            )
        )
        best_t = grid_values[0]
        best_survival = survival_fitted * torch.sigmoid(position_logits / best_t)
        best_ece = expected_calibration_error(
            probs=best_survival, targets=position_target, num_bins=num_bins
        )
        for temperature in grid_values[1:]:
            candidate = survival_fitted * torch.sigmoid(position_logits / temperature)
            ece = expected_calibration_error(
                probs=candidate, targets=position_target, num_bins=num_bins
            )
            if ece < best_ece:
                best_ece, best_t, best_survival = ece, temperature, candidate
        temperatures.append(float(best_t))
        ece_after.append(float(best_ece))
        survival_fitted = best_survival
    return {"temperatures": temperatures, "ece_before": ece_before, "ece_after": ece_after}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def load_shards(
    paths: list[Path], physical_k: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
    """Load safe CPU tensor shards without conflating widths or label pairings.

    Pairing metadata is a producer contract, not independent proof of the
    running checkpoint or label provenance. The original shard hashes remain
    recorded; source or GPU freshness is not rewritten or certified.
    """
    if type(physical_k) is not int or physical_k < 1 or not paths:
        raise ValueError("An explicit positive physical K and input shards are required")
    canonical = [path.resolve(strict=True) for path in paths]
    if len(set(canonical)) != len(canonical):
        raise ValueError("Duplicate calibration shards would double-count samples")
    logits, masks, hashes = [], [], {}
    for path, resolved in zip(paths, canonical):
        if path.is_symlink() or not resolved.is_file():
            raise ValueError("Calibration inputs must be regular, nonsymlink files")
        before = resolved.stat()
        pin = _sha256_file(resolved)
        blob = torch.load(resolved, map_location="cpu", weights_only=True)
        if not isinstance(blob, dict) or not isinstance(blob.get("meta"), dict):
            raise ValueError("Calibration shards require explicit pairing metadata")
        if blob["meta"].get("pairing") != "draft_seq_ring":
            raise ValueError("Recollect labels paired with the same stamped draft sequence")
        raw_logits, prefix_mask = blob.get("logits"), blob.get("prefix_mask")
        _validate_rows(raw_logits, prefix_mask)
        if raw_logits.shape[1] != physical_k:
            raise ValueError("The calibration shard width does not match the explicit physical K")
        after = resolved.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or _sha256_file(resolved) != pin:
            raise ValueError("Calibration shard changed during reading")
        logits.append(raw_logits)
        masks.append(prefix_mask)
        hashes[str(path)] = pin
    return torch.cat(logits, 0), torch.cat(masks, 0), hashes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--physical-k", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument("--minimum-samples", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.minimum_samples < 1:
        raise ValueError("minimum-samples must be positive")
    logits, masks, hashes = load_shards(args.shards, args.physical_k)
    if logits.shape[0] < args.minimum_samples:
        raise ValueError("Not enough paired observations to fit the requested calibration")
    result = fit_sts_temperatures(
        logits=logits, prefix_mask=masks, grid=default_temperature_grid(), num_bins=args.num_bins
    )
    rates = masks.to(torch.float64).mean(dim=0).tolist()
    payload = {
        "sts_temperatures": result["temperatures"],
        **result,
        "physical_k": args.physical_k,
        "num_samples": int(logits.shape[0]),
        "num_bins": args.num_bins,
        "raw_shard_sha256": hashes,
        "pairing": "draft_seq_ring",
        "prefix_acceptance_rate": rates,
        "degenerate_positions": [index for index, rate in enumerate(rates) if rate in (0.0, 1.0)],
        "objective": "left-to-right cumulative-prefix ECE;41-point temperature grid",
        "producer_pairing_independently_verified": False,
        "task_accuracy_or_economic_admission_qualified": False,
        "artifact_freshness_validated": False,
    }
    if args.output.is_symlink():
        raise ValueError("Calibration output cannot be a symlink")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(args.output, flags, 0o600), "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "num_samples": payload["num_samples"],
                "physical_k": args.physical_k,
                "degenerate_positions": payload["degenerate_positions"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
