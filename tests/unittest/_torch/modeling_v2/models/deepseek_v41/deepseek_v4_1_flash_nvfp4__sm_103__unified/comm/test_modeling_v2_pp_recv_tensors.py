# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the pp_recv_tensors certification: the receiving half of
the hand-over matrix that ``test_modeling_v2_pp_send_tensors.py`` starts, in the same
one-node form (PP2 x TP2). The two-tray cells are run by hand (see the entry's
``note``)."""

import pytest
from test_modeling_v2_pp_send_tensors import FORMS, run_pp_send_recv_matrix


@pytest.mark.no_xdist
@pytest.mark.parametrize("form", list(FORMS))
def test_pp_recv_op_matrix(form: str) -> None:
    run_pp_send_recv_matrix(form)
