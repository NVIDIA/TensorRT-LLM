#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download pinned BF16 source assets for all native LTX-2.5 precision modes."""

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

MODEL_ID = "Lightricks/LTX-2.5"
MODEL_REVISION = "49b093a39c19169a8a81721dcc541e07e945232f"
ADAPTER_ID = "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control"
ADAPTER_REVISION = "20a580c8606c06a0988d40375a61309e0fe5893d"
CONTROL_ID = "Lightricks/Canny-Control-Dataset"
CONTROL_REVISION = "2b7f1b86150dac8b519ad8cf6d5d4a0351c6dc03"


def download_plan(model_root: Path) -> list[dict[str, str]]:
    """Describe exact Hub revisions and local destinations for one shared recipe."""
    filenames = [
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    ]
    plan = [
        {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "repo_type": "model",
            "filename": filename,
            "local_dir": str(model_root),
        }
        for filename in filenames
    ]
    plan.append(
        {
            "repo_id": ADAPTER_ID,
            "revision": ADAPTER_REVISION,
            "repo_type": "model",
            "filename": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
            "local_dir": str(model_root / "loras"),
        }
    )
    plan.append(
        {
            "repo_id": CONTROL_ID,
            "revision": CONTROL_REVISION,
            "repo_type": "dataset",
            "filename": "bear_reference.mp4",
            "local_dir": str(model_root / "validation" / "raw"),
        }
    )
    return plan


def sha256(path: Path) -> str:
    """Hash the downloaded bytes for the baseline provenance manifest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("artifacts/ltx25/models/LTX-2.5"))
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/ltx25/runs"))
    parser.add_argument(
        "--list-only", action="store_true", help="Print the plan without network or writes."
    )
    args = parser.parse_args()
    plan = download_plan(args.model_root.resolve())
    if args.list_only:
        print(json.dumps(plan, indent=2))
        return

    from huggingface_hub import hf_hub_download

    run = args.run_root.resolve() / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ_download"
    )
    run.mkdir(parents=True, exist_ok=False)
    (run / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(run / "download.log"), logging.StreamHandler()],
    )
    logging.info("Download record: %s", run)
    for name in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.info(
        "All native precision modes use the BF16 transformer; prequant weights are not supported."
    )
    records = []
    for item in plan:
        logging.info("Downloading %s@%s/%s", item["repo_id"], item["revision"], item["filename"])
        path = Path(hf_hub_download(**item))
        records.append(
            {**item, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
        (run / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    logging.info("Complete. The raw control clip still needs frame-rate/geometry validation.")


if __name__ == "__main__":
    main()
