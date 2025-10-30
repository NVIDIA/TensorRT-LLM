# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# test of tests

import pytest
from filter_rules import *
from utils import *

fmha_exe_path = 'bin/fmha.exe'


# debug
@pytest.mark.debug
@pytest.mark.parametrize('rules', [[]])
@pytest.mark.parametrize('fmha_arg', [FmhaArgs()])
def test_fmha_single_case(fmha_arg, rules, dryrun, disable_rules,
                          gpu_compute_cap):
    fmha_harness(fmha_exe_path,
                 fmha_arg,
                 rules,
                 dryrun,
                 gpu_compute_cap=gpu_compute_cap)


# debug
@pytest.mark.debug
@apply_rule(corner_case_rule)
@pytest.mark.parametrize('rules', [[]])
@pytest.mark.parametrize('fmha_arg',
                         [FmhaArgs()._replace(seq_len=16, head_dim=64)])
def test_corner_case_rule(fmha_arg, rules, dryrun, disable_rules,
                          gpu_compute_cap):
    fmha_harness(fmha_exe_path,
                 fmha_arg,
                 rules,
                 dryrun,
                 gpu_compute_cap=gpu_compute_cap)


# debug
@pytest.mark.debug
@apply_rule(compute_cap_specific_rule)
@pytest.mark.parametrize('rules', [[]])
@pytest.mark.parametrize('fmha_arg', [FmhaArgs()])
def test_compute_cap_specific_rule(fmha_arg, rules, dryrun, disable_rules,
                                   mock_gpu_compute_cap):
    fmha_harness(fmha_exe_path,
                 fmha_arg,
                 rules,
                 dryrun,
                 gpu_compute_cap=mock_gpu_compute_cap)


# debug; try running large seq and large head size without error tolerance
@pytest.mark.debug
@apply_rule(sm80_only)
@pytest.mark.parametrize('rules', [[]])
@pytest.mark.parametrize('fmha_arg',
                         [FmhaArgs()._replace(seq_len=4096, head_dim=256)])
@pytest.mark.xfail
def test_controlled_validation_fail(fmha_arg, rules, dryrun, disable_rules,
                                    gpu_compute_cap):
    fmha_harness(fmha_exe_path,
                 fmha_arg,
                 rules,
                 dryrun,
                 gpu_compute_cap=gpu_compute_cap)
