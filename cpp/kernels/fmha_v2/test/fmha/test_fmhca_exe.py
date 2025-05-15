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
from filter_rules import *
from utils import *

fmha_exe_path = 'bin/fmhca.exe'


@apply_rule(sm80_only)
@pytest.mark.fmhca
@pytest.mark.parametrize('rules', [[]])
@pytest.mark.parametrize('fmha_arg', combinations_fmhca())
def test_fmhca(fmha_arg, rules, dryrun, disable_rules, gpu_compute_cap):
    fmha_harness(fmha_exe_path,
                 fmha_arg,
                 rules,
                 dryrun,
                 gpu_compute_cap=gpu_compute_cap)
