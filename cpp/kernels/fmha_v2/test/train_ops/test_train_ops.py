# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pytest
from fmha.filter_rules import *
from fmha.utils import *
from train_ops.fmha_unit_test import run_test


def run_train_ops_harness(seq_len, head_dim, rules=[], **kwargs):
    for rule in rules:
        rule(FmhaArgs(), **kwargs)
    run_test(seq_len, head_dim)


@apply_rule(sm80_only)
@pytest.mark.parametrize('rules', [[]])
@pytest.mark.parametrize('seq_len', [1024, 2048], ids=lambda x: 's' + str(x))
@pytest.mark.parametrize('head_dim', [40, 64, 80, 96, 128],
                         ids=lambda x: 'd' + str(x))
def test_train_ops_fp16(seq_len, head_dim, rules, gpu_compute_cap):
    run_train_ops_harness(seq_len,
                          head_dim,
                          rules,
                          gpu_compute_cap=gpu_compute_cap)
