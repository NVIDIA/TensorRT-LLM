# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP32 o_proj push chain above 48 rows changes nothing else: the TP8 / TP16 / drafter push GEMM tables, the TP32
rows up to 48, the finishing-mix caps of the other TP worlds and the KDA o_proj deferral at every other TP world stay
as they were; ``K31_TP32_OPROJ_PUSH_WIDE=0`` restores the 48-row TP32 table."""

import os
import subprocess
import sys

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep import (
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.comm import (
    bf16_ar_push_gemm as pg,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.norm import (
    oproj_push_attn_res_add_rmsnorm as opn,
)

T = pg.SplitKTactic
H = 7168
_DRAFTER = (
    (8, T(128, 8, 1, 7)),
    (16, T(64, 16, 1, 7)),
    (24, T(64, 24, 1, 7)),
    (32, T(64, 32, 1, 7)),
    (48, T(64, 48, 1, 6)),
    (64, T(64, 64, 1, 6)),
    (72, T(64, 72, 1, 5)),
    (80, T(64, 80, 1, 5)),
)
#: Every table but TP32's, as before the TP32 extension.
OTHER_TABLES = {
    (H, 1536): ((32, T(64, 32, 1, 9)), (64, T(64, 64, 1, 6)), (72, T(64, 128, 1, 4))),
    (H, 768): (
        (8, T(128, 8, 1, 6)),
        (16, T(64, 16, 1, 7)),
        (24, T(64, 24, 1, 7)),
        (32, T(64, 32, 1, 7)),
        (48, T(64, 48, 1, 6)),
        (64, T(64, 64, 1, 6)),
        (72, T(64, 72, 1, 5)),
        (80, T(64, 80, 1, 5)),
        (192, T(128, 64, 1, 3)),
    ),
    **{(H, k): _DRAFTER for k in (256, 512, 896, 1792)},
    (H, 448): (
        (8, T(128, 8, 1, 4)),
        (16, T(64, 16, 1, 4)),
        (24, T(64, 24, 1, 4)),
        (32, T(64, 32, 1, 4)),
    ),
}
#: The TP32 (K = 384) rows up to 48, as before the extension.
TP32_TO_48 = (
    (2, T(64, 8, 1, 3)),
    (8, T(128, 8, 1, 3)),
    (16, T(64, 16, 1, 3)),
    (32, T(64, 32, 1, 3)),
    (48, T(64, 48, 1, 3)),
)


def test_other_tables_unchanged():
    assert set(pg._TACTICS) == set(OTHER_TABLES) | {(H, 384)}
    for key, rows in OTHER_TABLES.items():
        assert pg._TACTICS[key] == rows, key


def test_tp32_rows_up_to_48_unchanged():
    assert pg._TACTICS[(H, 384)][: len(TP32_TO_48)] == TP32_TO_48
    for m in range(1, 49):
        assert pg.tactic_for(m, H, 384) == next(t for bound, t in TP32_TO_48 if m <= bound), m


def test_tp32_wide_rows_single_k_pass():
    # cuBLAS runs these shapes as one whole-K kernel: only a single K pass reproduces its bits.
    for m in range(49, pg.MAX_M + 1):
        tactic = pg.tactic_for(m, H, 384)
        assert tactic is not None and tactic.split_k == 1 and tactic.ab_stages == 3, (m, tactic)
    assert pg.tactic_for(pg.MAX_M + 1, H, 384) is None


def test_switch_restores_the_48_row_table():
    code = (
        f"import {pg.__name__} as pg\n"
        "print(pg.tactic_for(48, 7168, 384) is not None, pg.tactic_for(49, 7168, 384) is None, "
        "pg.tactic_for(192, 7168, 768) is not None)"
    )
    env = {**os.environ, pg.TP32_WIDE_ENV: "0"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split()[-3:] == ["True", "True", "True"], out.stdout


@pytest.mark.parametrize("world", [4, 8, 16])
def test_finish_caps_of_other_worlds_unchanged(world):
    assert opn.FUSED_FINISH_MAX_ROWS[world] == 95
    assert opn.IN_PLACE_FINISH_MAX_ROWS[world] == 128


def test_tp32_finish_caps():
    # Up to 48 rows the finishing mix as before; above, the in-place finish up to the PDL mix's last row count.
    assert opn.FUSED_FINISH_MAX_ROWS[32] == 48
    assert opn.IN_PLACE_FINISH_MAX_ROWS[32] == modeling._K31_MIX_PERSISTENT_ROWS - 1


def _rows(m: int) -> torch.Tensor:
    return torch.empty(m, H, dtype=torch.bfloat16)


@pytest.mark.parametrize("tp", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("m", [1, 8, 48, 49, 96, 191, 192, 256])
def test_kda_defers_on_every_residual_add_layer_off_tp32(tp, m):
    assert modeling._k31_defer_kda_oproj(_rows(m), tp)
    assert not modeling._k31_defer_kda_oproj(None, tp)


@pytest.mark.parametrize("m", [1, 8, 48, 49, 96, 128, 191])
def test_tp32_kda_defers_on_the_pdl_mix_rows(m):
    assert modeling._k31_defer_kda_oproj(_rows(m), 32)


@pytest.mark.parametrize("m", [192, 256])
def test_tp32_kda_keeps_the_persistent_mix_rows(m):
    assert modeling._K31_MIX_PERSISTENT_ROWS == 192
    assert not modeling._k31_defer_kda_oproj(_rows(m), 32)
    assert not modeling._k31_defer_kda_oproj(None, 32)
