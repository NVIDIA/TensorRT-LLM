# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The VR S10 scope (tp4): the lean drafter glue at every draft length (S10_K7, default ON); K <= 3 when killed."""

import subprocess
import sys

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
        dspark,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

_CODE = (
    "from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
    "minimax_m3_1_flash_nvfp4__sm_107__tp4 import dspark\n"
    "print(dspark.S10_MAX_DRAFT_LEN, dspark.s10_applies(7), dspark.s10_applies(4))"
)


def _run(**env_over):
    env = {
        k: v
        for k, v in __import__("os").environ.items()
        if k not in ("TRTLLM_M31_ENABLE_S10_K7", "TRTLLM_M31_DISABLE_S10_K7")
    }
    env.update(env_over)
    out = subprocess.run(
        [sys.executable, "-c", _CODE], env=env, capture_output=True, text=True, check=True
    )
    return out.stdout.split()[-3:]


@pytest.mark.parametrize("k", [1, 3, 4, 7, 15])
def test_s10_applies_every_draft_length_by_default(k: int) -> None:
    if dspark.S10_K7:
        assert dspark.S10_MAX_DRAFT_LEN == 63
        assert dspark.s10_applies(k) is True
    else:  # TRTLLM_M31_DISABLE_S10_K7=1 in this process
        assert dspark.S10_MAX_DRAFT_LEN == 3
        assert dspark.s10_applies(k) is (k <= 3)


def test_s10_k7_default_on_and_kill_switch() -> None:
    """S10_K7 is default ON (read at import); TRTLLM_M31_DISABLE_S10_K7=1 bounds S10 to K <= 3 again; the old opt-in
    TRTLLM_M31_ENABLE_S10_K7=1 is inert."""
    assert _run() == ["63", "True", "True"]
    assert _run(TRTLLM_M31_ENABLE_S10_K7="1") == ["63", "True", "True"]
    assert _run(TRTLLM_M31_DISABLE_S10_K7="1") == ["3", "False", "False"]
