#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Apply calibrated Skip Softmax metadata to a Wan 2.2 T2V-A14B checkpoint."""

import argparse
import json
import math
import os
import stat
import tempfile
from pathlib import Path

COMPONENTS = (
    (
        "transformer",
        "transformer_sparse_attention.json",
        {"a": 2142.7334009837796, "b": 4.282667871834358},
    ),
    (
        "transformer_2",
        "transformer_2_sparse_attention.json",
        {"a": 314.68242763240454, "b": 6.166472427782809},
    ),
)
MODEL_INDEX_SIGNATURE = {
    "_class_name": "WanPipeline",
    "boundary_ratio": 0.875,
    "transformer": ["diffusers", "WanTransformer3DModel"],
    "transformer_2": ["diffusers", "WanTransformer3DModel"],
}
TRANSFORMER_SIGNATURE = {
    "_class_name": "WanTransformer3DModel",
    "patch_size": [1, 2, 2],
    "num_attention_heads": 40,
    "attention_head_dim": 128,
    "num_layers": 40,
    "in_channels": 16,
    "out_channels": 16,
    "text_dim": 4096,
    "freq_dim": 256,
}


class CalibrationError(Exception):
    """Raised when the checkpoint or calibration package is incompatible."""


def _as_object(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CalibrationError(f"{location} must be a JSON object")
    return value


def _load_json(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError as error:
        raise CalibrationError(f"missing required file: {path}") from error
    except json.JSONDecodeError as error:
        raise CalibrationError(f"invalid JSON in {path}: {error}") from error
    except OSError as error:
        raise CalibrationError(f"cannot read {path}: {error}") from error
    return _as_object(value, str(path))


def _validate_signature(
    document: dict[str, object], expected: dict[str, object], path: Path
) -> None:
    mismatches = [
        f"{key}={document.get(key)!r} (expected {expected_value!r})"
        for key, expected_value in expected.items()
        if document.get(key) != expected_value
    ]
    if mismatches:
        details = ", ".join(mismatches)
        raise CalibrationError(f"{path} is not the supported Wan 2.2 T2V-A14B config: {details}")


def _validate_overlay(
    overlay: dict[str, object], path: Path, expected_coefficients: dict[str, float]
) -> dict[str, object]:
    if set(overlay) != {"sparse_attention_config"}:
        raise CalibrationError(f"{path} must contain only sparse_attention_config")

    sparse_config = _as_object(
        overlay["sparse_attention_config"], f"{path}: sparse_attention_config"
    )
    config_groups = _as_object(
        sparse_config.get("config_groups"), f"{path}: sparse_attention_config.config_groups"
    )
    if set(config_groups) != {"group_0"}:
        raise CalibrationError(f"{path} must contain exactly one config group named group_0")

    group = _as_object(config_groups["group_0"], f"{path}: config_groups.group_0")
    if group.get("algorithm") != "skip_softmax":
        raise CalibrationError(f"{path}: group_0.algorithm must be skip_softmax")
    if group.get("targets") != ["WanAttention"]:
        raise CalibrationError(f"{path}: group_0.targets must be ['WanAttention']")

    ignore = group.get("ignore")
    if (
        not isinstance(ignore, list)
        or len(ignore) != 44
        or not all(isinstance(item, str) for item in ignore)
        or len(set(ignore)) != len(ignore)
    ):
        raise CalibrationError(f"{path}: group_0.ignore must contain 44 unique module names")

    threshold = _as_object(
        group.get("threshold_scale_factor"), f"{path}: group_0.threshold_scale_factor"
    )
    if threshold.get("formula") != "a * exp(b * target_sparsity)":
        raise CalibrationError(f"{path}: unsupported threshold_scale_factor formula")
    coefficients = _as_object(
        threshold.get("coefficients"), f"{path}: threshold_scale_factor.coefficients"
    )
    if coefficients != expected_coefficients:
        raise CalibrationError(
            f"{path}: coefficients {coefficients!r} do not match {expected_coefficients!r}"
        )
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
        for value in coefficients.values()
    ):
        raise CalibrationError(f"{path}: coefficients must be positive finite numbers")

    target_sparsity = group.get("target_sparsity")
    if (
        not isinstance(target_sparsity, (int, float))
        or isinstance(target_sparsity, bool)
        or not 0 <= target_sparsity <= 1
    ):
        raise CalibrationError(f"{path}: target_sparsity must be between 0 and 1")

    return sparse_config


def _stage_json(path: Path, document: dict[str, object]) -> Path:
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(document, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.chmod(mode)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise CalibrationError(f"cannot stage update for {path}: {error}") from error
    if temporary_path is None:
        raise CalibrationError(f"cannot stage update for {path}")
    return temporary_path


def apply_calibration(model_dir: Path, overlay_dir: Path, force: bool) -> None:
    """Validate and merge both component-specific calibration overlays.

    Args:
        model_dir: Local Wan 2.2 T2V-A14B Diffusers checkpoint directory.
        overlay_dir: Directory containing the packaged calibration overlays.
        force: Replace an existing, different sparse-attention configuration.
    """
    model_dir = model_dir.resolve()
    overlay_dir = overlay_dir.resolve()
    model_index_path = model_dir / "model_index.json"
    _validate_signature(_load_json(model_index_path), MODEL_INDEX_SIGNATURE, model_index_path)

    updates: list[tuple[Path, dict[str, object]]] = []
    for component, overlay_name, expected_coefficients in COMPONENTS:
        config_path = model_dir / component / "config.json"
        config = _load_json(config_path)
        _validate_signature(config, TRANSFORMER_SIGNATURE, config_path)

        overlay_path = overlay_dir / overlay_name
        sparse_config = _validate_overlay(
            _load_json(overlay_path), overlay_path, expected_coefficients
        )
        existing = config.get("sparse_attention_config")
        if existing == sparse_config:
            continue
        if existing is not None and not force:
            raise CalibrationError(
                f"{config_path} already contains different sparse_attention_config metadata; "
                "use a fresh checkpoint copy or pass --force"
            )

        updated_config = dict(config)
        updated_config["sparse_attention_config"] = sparse_config
        updates.append((config_path, updated_config))

    staged: list[tuple[Path, Path]] = []
    try:
        staged = [(path, _stage_json(path, document)) for path, document in updates]
        for path, temporary_path in staged:
            try:
                os.replace(temporary_path, path)
            except OSError as error:
                raise CalibrationError(f"cannot replace {path}: {error}") from error
            print(f"Updated {path}")
    finally:
        for _, temporary_path in staged:
            temporary_path.unlink(missing_ok=True)

    if not updates:
        print("Calibration metadata is already up to date.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Local Wan2.2-T2V-A14B-Diffusers checkpoint directory.",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Calibration overlay directory (defaults to this example directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing sparse_attention_config metadata when it differs.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        apply_calibration(args.model_dir, args.overlay_dir, args.force)
    except CalibrationError as error:
        print(f"error: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
