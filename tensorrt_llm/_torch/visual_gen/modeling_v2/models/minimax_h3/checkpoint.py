# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path


def load_fasth3_manifest(checkpoint_dir: str) -> dict:
    manifest_path = Path(checkpoint_dir) / "fastvideo_inference.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
        if str(manifest.get("schema_version", "")).startswith("fasth3-"):
            return manifest
    return {}
