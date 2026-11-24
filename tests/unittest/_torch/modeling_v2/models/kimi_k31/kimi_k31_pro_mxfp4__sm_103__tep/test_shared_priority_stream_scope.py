# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE shared expert takes the high-priority stream only at attention TP8 up to the MoE front's rows; every other
layout and row count keeps the shared-expert aux stream."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep import (
    modeling,
)


def test_scope_is_tp8():
    assert modeling._K31_SHARED_PRIORITY_TP == (8,)
    assert modeling._K31_FRONT_FOLD_MAX_ROWS == 32


@pytest.mark.parametrize("m", [1, 5, 10, 15, 31, 32])
def test_tp8_front_rows_take_priority(m):
    assert modeling._k31_shared_on_priority_stream(8, m)


@pytest.mark.parametrize("m", [33, 35, 64, 192])
def test_tp8_above_front_rows_keep_aux_stream(m):
    assert not modeling._k31_shared_on_priority_stream(8, m)


@pytest.mark.parametrize("tp", [1, 2, 4, 16, 32])
@pytest.mark.parametrize("m", [1, 5, 8, 32, 33, 64])
def test_other_tp_keep_aux_stream(tp, m):
    assert not modeling._k31_shared_on_priority_stream(tp, m)
