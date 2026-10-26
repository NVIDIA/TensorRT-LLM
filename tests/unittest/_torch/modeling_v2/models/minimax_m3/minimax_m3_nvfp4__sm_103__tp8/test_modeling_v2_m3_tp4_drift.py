# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp8 target's files that are byte copies of the tp4 target's stay so.

The two MiniMax-M3 targets share no code by design (``test_modeling_v2_claims.test_targets_do_not_share_files``): the
tp8 target was made as a copy of the tp4 one and keeps its own copy of every file, tp8-specific or not. The files
listed here are still identical, so a fix to one copy applies to the other unchanged. This test fails on the first
edit to either side: a change to one copy must be considered for the other -- applied to both, or the file dropped
from this list with the reason the copies now differ.

CPU only: it reads the files by path relative to the packages.
"""

from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2 as _modeling_v2

# Nothing of either target is imported (no GPU, no built extensions needed): the paths are the packages', resolved
# from the imported parents rather than from this file, which lives under tests/.
pytestmark = pytest.mark.cpu_only

_M3 = Path(_modeling_v2.__file__).resolve().parent / "models" / "minimax_m3"
_TP4 = _M3 / "minimax_m3_nvfp4__sm_103__tp4"
_TP8 = _M3 / "minimax_m3_nvfp4__sm_103__tp8"

#: Paths relative to the target directory that are byte copies of the tp4 target's.
#: (``ops/attention/m3_msa_attention.py`` left the list: its reference dequantizes NVFP4 pages with each target's own
#: long-context sparse decode entry, ``m3_sparse_decode_lc`` at TP4 and ``m3_sparse_decode_lc8`` at TP8.)
IDENTICAL_TO_TP4 = (
    "ops/embedding_firstnorm/__init__.py",
    "ops/embedding_firstnorm/m3_embedding_firstnorm_kernel.py",
    "ops/embedding_firstnorm/op.py",
    "ops/long_select.py",
    "ops/m3_force_vec.py",
    "ops/m3_index_score.py",
    "ops/m3_prequant.py",
    "ops/m3_router_gemv.py",
    "ops/m3_sampled_accept.py",
    "ops/m3_shard_accept.py",
    "ops/m3_spec_tail.py",
    "ops/msa_kv_update.py",
    "ops/mx_gemv/m3_mx_gemv_kernel.py",
    "ops/triton_sparse_decode_mq.py",
    "ops/__init__.py",
    "ops/activation/__init__.py",
    "ops/attention/__init__.py",
    "ops/attention/m3_draft_attention.py",
    "ops/attention/m3_fp8_indexer_qk_norm_rope.py",
    "ops/attention/m3_select_blocks.py",
    "ops/attention/msa_kv_update.py",
    "ops/comm/__init__.py",
    "ops/embedding/__init__.py",
    "ops/embedding/m3_embedding.py",
    "ops/gemm/__init__.py",
    "ops/gemm/m3_fp32_linear.py",
    "ops/moe/__init__.py",
    "ops/moe/m3_router_gemv_fp4.py",
    "ops/moe/m3_router_gemv.py",
    "ops/moe/m3_router_tc.py",
    "ops/norm/__init__.py",
    "ops/norm/m3_embedding_firstnorm.py",
    "ops/spec/__init__.py",
    "ops/spec/m3_capture_norm.py",
    "ops/spec/m3_chunk_norm.py",
    "ops/spec/m3_draft_input.py",
    "ops/spec/m3_force_vec.py",
    "ops/spec/m3_slot_scatter.py",
    "ops/spec/m3_step_gather.py",
)


def test_the_targets_are_where_this_test_reads_them() -> None:
    for path in (_TP4 / "modeling.py", _TP8 / "modeling.py"):
        assert path.is_file(), path


def _differs(a: Path, b: Path) -> str:
    """Empty when the two files are byte-identical, else where they first differ."""
    x, y = a.read_bytes(), b.read_bytes()
    if x == y:
        return ""
    first = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
    line = x[:first].count(b"\n") + 1
    return f"{a} and {b} differ from line {line} ({len(x)} vs {len(y)} bytes)"


@pytest.mark.parametrize("relative", IDENTICAL_TO_TP4)
def test_the_copy_is_the_tp4_file(relative) -> None:
    ours, theirs = _TP8 / relative, _TP4 / relative
    assert ours.is_file(), f"{relative}: gone from the tp8 target; drop it from IDENTICAL_TO_TP4"
    assert theirs.is_file(), f"{relative}: gone from the tp4 target; drop it from IDENTICAL_TO_TP4"
    diff = _differs(ours, theirs)
    assert not diff, (
        f"{diff}. The targets share no code, so a change to one copy must be considered for the other: apply it to "
        "both, or drop the file from IDENTICAL_TO_TP4 stating why the copies now differ."
    )


def test_the_list_names_every_identical_file() -> None:
    """A file that became identical again (or was copied unchanged later) is listed, so it is guarded too."""
    unlisted = sorted(
        str(path.relative_to(_TP8))
        for path in _TP8.rglob("*.py")
        if "__pycache__" not in path.parts
        and str(path.relative_to(_TP8)) not in IDENTICAL_TO_TP4
        and (_TP4 / path.relative_to(_TP8)).is_file()
        and not _differs(path, _TP4 / path.relative_to(_TP8))
    )
    assert not unlisted, f"byte copies of the tp4 target not in IDENTICAL_TO_TP4: {unlisted}"
