# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vendored PrimTS tree must stay self-contained.

TRT-LLM pins a flashinfer-python release whose own ``flashinfer.attention``
tree is older than the vendored one and whose ``flashinfer_api`` has no trace
templates. A vendored module that imports either would silently bind to the
installed copy, so the compatibility patch rewrites those imports; this test
keeps a future re-vendor from reintroducing them.
"""

import pathlib
import re

import tensorrt_llm._torch.attention.backends.prims_ts as vendored

_ESCAPING_IMPORT = re.compile(
    r"^\s*(?:from\s+flashinfer\.(?:attention|trace)\b|import\s+flashinfer\.(?:attention|trace)\b)",
    re.MULTILINE,
)


def test_vendored_prims_ts_does_not_import_flashinfer_attention_or_trace() -> None:
    root = pathlib.Path(vendored.__file__).resolve().parent
    offenders = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if _ESCAPING_IMPORT.search(path.read_text(encoding="utf-8"))
    )
    assert not offenders, f"vendored modules import the installed flashinfer tree: {offenders}"
