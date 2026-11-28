# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tep4 rank's projections past `low_m_bf16_mm`'s table (`TEP_PROJ_ENV`), CPU only.

* **Switch** (`_tep_proj`): off by default; on only with the tep4 fast forms, else it warns.
* **Entries**: `_tep_wide_projection` asks the table for its `wide` 1 rows; the GEMV rows stay
  on `bf16_gemv_pdl`. The trunk shared expert of a decode step takes `wide` 2 up to
  `WIDE_MAX_ROWS` rows under the switch, cuBLAS otherwise.

Which kernel each `wide` row picks, and its numerics, are held by
`ops/gemm/test_modeling_v2_low_m_bf16_mm.py` (GPU: cells and plan pins).
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "value, tep_fast, expected, warns",
    [(None, True, False, False), ("1", True, True, False), ("1", False, False, True)],
)
def test_switch(monkeypatch, value, tep_fast, expected, warns):
    warned = []
    monkeypatch.setattr(M.logger, "warning", warned.append)
    if value is None:
        monkeypatch.delenv(M.TEP_PROJ_ENV, raising=False)
    else:
        monkeypatch.setenv(M.TEP_PROJ_ENV, value)
    assert M._tep_proj(SimpleNamespace(tep_fast=tep_fast)) is expected
    assert bool(warned) is warns


@pytest.mark.cpu_only
@pytest.mark.parametrize("rows, k, expected", [(4, 1536, "gemv"), (12, 1536, 1), (12, 2560, 1)])
def test_wide_projection(monkeypatch, rows, k, expected):
    calls = []
    monkeypatch.setattr(M, "bf16_gemv_pdl", lambda a, b, pdl: calls.append("gemv"))
    monkeypatch.setattr(M, "low_m_bf16_mm", lambda a, b, pdl, wide=0: calls.append(wide))
    M._tep_wide_projection(torch.zeros(rows, k), torch.zeros(k, 2560))
    assert calls == [expected]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "tep_proj, decode, rows, expected",
    [
        (True, True, 16, ["wide", "wide"]),
        (True, True, 17, ["cublas", "cublas"]),
        (True, False, 4, ["cublas", "cublas"]),  # a prefill step
        (False, True, 4, ["cublas", "cublas"]),
    ],
)
def test_shared_expert(monkeypatch, tep_proj, decode, rows, expected):
    calls = []
    monkeypatch.setattr(
        M, "_tep_wide_mm", lambda a, b: calls.append("wide") or torch.zeros(a.shape[0], 12)
    )
    monkeypatch.setattr(
        M, "cublas_mm", lambda a, b: calls.append("cublas") or torch.zeros(a.shape[0], 12)
    )
    monkeypatch.setattr(M, "silu_and_mul_rows", lambda x, n: x)
    core = SimpleNamespace(tep_proj=tep_proj, shared_inter=4)
    L = SimpleNamespace(sh_gu_g=None, sh_gug=None, sh_dn=None)
    shared, gate = M._shared_expert(core, torch.zeros(rows, 8), L, object() if decode else None)
    assert calls == expected and gate.shape == (rows, 1)
