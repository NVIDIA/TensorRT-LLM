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
"""Publish configuration schemas alongside each successful HTML docs build."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sphinx.application import Sphinx


def _write_schema_assets(app: Sphinx, exception: Exception | None) -> None:
    if exception is not None or app.builder.format != "html":
        return
    generator = Path(__file__).resolve().parents[3] / "scripts/generate_trtllm_serve_schemas.py"
    write_schemas = runpy.run_path(str(generator))["write_schemas"]
    write_schemas(Path(app.outdir) / "_static" / "schemas", version=app.config.version)


def setup(app: Sphinx) -> dict:
    app.connect("build-finished", _write_schema_assets)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
