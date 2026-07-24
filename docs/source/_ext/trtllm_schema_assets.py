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
"""Write generated schema assets into the Sphinx HTML output."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _ensure_repo_root_on_syspath() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _write_schema_assets(app) -> None:
    builder = getattr(app, "builder", None)
    if not builder:
        return
    if builder.name not in {"html", "dirhtml"}:
        return

    repo_root = _ensure_repo_root_on_syspath()
    generator_path = repo_root / "scripts" / "generate_trtllm_serve_schemas.py"
    spec = importlib.util.spec_from_file_location("trtllm_serve_schema_generator", generator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load trtllm-serve schema generator from {generator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out_dir = Path(builder.outdir) / "_static" / "schemas"
    written_paths = module.write_schemas(out_dir)
    for path in written_paths:
        LOGGER.info("Wrote trtllm-serve schema: %s", path)


def _on_build_finished(app, exception) -> None:
    if exception is not None:
        return
    _write_schema_assets(app)


def setup(app):
    app.connect("build-finished", _on_build_finished)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
