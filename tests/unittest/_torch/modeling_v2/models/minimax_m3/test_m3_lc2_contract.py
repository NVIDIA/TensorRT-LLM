# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source contracts of the family library's fixed MiniMax-M3 kernels (``csrc/m3``), read without a GPU.

* The sm_107 tp2 long-context sparse decode (LC2, cluster and split forms) reads the query rows and the top-k table
  only after ``griddepcontrol.wait``: on identity-selection steps the producer may release the grid before it writes
  them, so an earlier read would see the previous step's values.
* Every UE8M0 block-scale conversion that a reciprocal follows guards the zero code: the byte 0 is 2^-127, an fp32
  denormal whose flush-to-zero reciprocal is +inf, so such a block must quantize to zero instead.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

# tests/unittest/_torch/modeling_v2/models/minimax_m3/ -> the repository root.
_CSRC = (
    Path(__file__).resolve().parents[6]
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/csrc/m3"
)
_LC2 = ("sm107Tp2SparseDecodeLc2.cu", "sm107Tp2SparseDecodeLc2Split.cu")
_E8M0 = ("mcAllreduceNorm.cu", "mxfp8LowMGemm.cu")


def _code(text: str) -> str:
    """The source without comments, so a comment naming a read does not count as one."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


@pytest.mark.parametrize("source", _LC2)
def test_lc2_reads_q_and_the_top_k_table_after_the_grid_dependency(source):
    code = _code((_CSRC / source).read_text())
    wait = code.index("griddepcontrol.wait")
    for read in ("p.q + ", "p.topk["):
        first = code.find(read)
        assert first != -1, f"{source}: no {read!r} read"
        assert first > wait, f"{source}: {read!r} is read before griddepcontrol.wait"


@pytest.mark.parametrize("source", _E8M0)
def test_every_ue8m0_conversion_guards_the_zero_code(source):
    code = _code((_CSRC / source).read_text())
    conversions = code.count("__nv_cvt_float_to_e8m0")
    assert conversions > 0, source
    assert code.count("code.__x != 0") == conversions, (
        f"{source}: {conversions} UE8M0 conversions, not each followed by a zero-code guard"
    )
