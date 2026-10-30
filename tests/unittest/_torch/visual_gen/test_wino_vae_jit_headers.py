# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The runtime-compiled copies of the Wan VAE norm headers (shipped in the wheel so the JIT
needs no source checkout) must match the compiled originals up to their wrapper lines."""

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[4]
JIT = REPO / "tensorrt_llm/_torch/visual_gen/models/wan/cuda"
CPP = REPO / "cpp/tensorrt_llm"

PAIRS = [
    (CPP / "common/packedVector.cuh", JIT / "packed_vector.cuh"),
    (CPP / "kernels/vaeNormCommon.cuh", JIT / "vae_norm_common.cuh"),
]

# Lines that may differ: includes, namespace open / close, the header comment, blank lines.
# The JIT copy marks constexpr helpers ``__host__ __device__`` (no relaxed-constexpr flag there);
# the qualifier pair is dropped on both sides, a bare ``__device__`` must match.
_WRAPPER = re.compile(
    r"^\s*(#include\b|namespace\b|}\s*//\s*namespace\b|TRTLLM_NAMESPACE_(BEGIN|END)\b|//|$)"
)


def _body(path: pathlib.Path) -> list[str]:
    lines = []
    after_namespace = False
    for line in path.read_text().splitlines():
        if after_namespace and line.strip() == "{":  # the namespace's opening brace
            after_namespace = False
            continue
        after_namespace = line.lstrip().startswith("namespace ")
        if _WRAPPER.match(line):
            continue
        line = line.replace("__host__ __device__ ", "")
        line = line.replace("vaewino::common::", "common::")
        lines.append(line.rstrip())
    return lines


@pytest.mark.parametrize("original,copy", PAIRS, ids=lambda p: p.name)
def test_jit_header_matches_compiled_original(original: pathlib.Path, copy: pathlib.Path) -> None:
    assert original.exists() and copy.exists()
    a, b = _body(original), _body(copy)
    assert a == b, "\n".join(
        f"{i}: {x!r} != {y!r}" for i, (x, y) in enumerate(zip(a, b)) if x != y
    ) + ("" if len(a) == len(b) else f"\nlength {len(a)} vs {len(b)}")
