# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install a matched pair of baseline captures into an architecture-specific QA directory."""

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def file_sha256(handle) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()


def stage_references(manifest: Path, artifacts: dict[str, Path], output: Path) -> Path:
    source = json.loads(manifest.read_text())
    origin = {key: source[key] for key in ("baseline_commit", "checkpoint_revision")}
    if not all(re.fullmatch(r"[0-9a-f]{40}", value) for value in origin.values()):
        raise ValueError("Supply immutable Git and checkpoint revisions in the baseline manifest")

    output.mkdir(parents=True, exist_ok=True)
    backends = {}
    common_case = None
    with TemporaryDirectory(dir=output) as temporary:
        staged = Path(temporary) / "reference"
        staged.mkdir()
        for backend, path in artifacts.items():
            destination = staged / f"{backend}.pt"
            shutil.copyfile(path, destination)
            with destination.open("rb") as handle:
                digest = file_sha256(handle)
                handle.seek(0)
                reference = torch.load(handle, map_location="cpu", weights_only=True)
            case = dict(reference["case"])
            if case.pop("backend") != backend or not reference["target"].endswith(
                ".minimax_h3.pipeline_minimax_h3.MiniMaxH3Pipeline"
            ):
                raise ValueError(f"{path} does not contain the {backend} V1 baseline")
            if common_case is None:
                common_case = case
            elif case != common_case:
                raise ValueError(
                    "Both backend captures must use the same checkpoint, SM and request"
                )
            limits = {key: source[backend][key] for key in ("max_lpips", "max_audio_log_stft")}
            if not all(math.isfinite(v) and v >= 0 for v in limits.values()):
                raise ValueError(f"Invalid quality limits for {backend}")
            backends[backend] = {"sha256": digest, **limits}

        major, minor = common_case["sm"]
        result = {**origin, "sm": [major, minor], "backends": backends}
        (staged / "8-step-v2.json").write_text(json.dumps(result, indent=2) + "\n")
        target = output / f"sm{major}{minor}"
        staged.rename(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trtllm", type=Path, required=True)
    parser.add_argument("--cute", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        stage_references(args.manifest, {"TRTLLM": args.trtllm, "CUTEDSL": args.cute}, args.output)
    )
