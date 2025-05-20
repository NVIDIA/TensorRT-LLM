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

fmha_exe_path = 'bin/fmha.exe'

####################################################################################################
# FP16
####################################################################################################


@apply_rule(error_relaxation_rule)
@apply_rule(avoid_oom_rule)
@apply_rule(corner_case_rule)
@apply_rule(compute_cap_specific_rule)
@apply_rule(flash_attn_limitation_rule)
@pytest.mark.fmha
@pytest.mark.parametrize('rules', [[], [force_non_granular_tiling]], ids=idfn)
@pytest.mark.parametrize('fmha_arg', combinations_fp16())
def test_fmha_fp16(fmha_arg, rules, dryrun, disable_rules, gpu_compute_cap):
    fmha_harness(fmha_exe_path,
                 fmha_arg,
                 rules,
                 dryrun,
                 gpu_compute_cap=gpu_compute_cap)


@apply_rule(error_relaxation_rule)
@apply_rule(corner_case_rule)
@apply_rule(compute_cap_specific_rule)
@apply_rule(flash_attn_limitation_rule)
@pytest.mark.fmha
@pytest.mark.parametrize(
    'rules',
    [[], [force_non_granular_tiling], [
        causal_mask, multi_query_attention, pad_s
    ], [causal_mask, grouped_query_attention],
     [force_non_granular_tiling, causal_mask, multi_query_attention, pad_s],
     [force_non_granular_tiling, causal_mask, grouped_query_attention]],
    ids=idfn)
@pytest.mark.parametrize('fmha_arg', reduced2x_combinations_fp16())
def test_fmha_flash_extended_fp16(fmha_arg, rules, dryrun, disable_rules,
                                  gpu_compute_cap):
    fmha_harness(fmha_exe_path,
                 fmha_arg,
                 rules,
                 dryrun,
                 gpu_compute_cap=gpu_compute_cap)


@apply_rule(non_sm90_only)
@apply_rule(error_relaxation_rule)
@apply_rule(corner_case_rule)
@apply_rule(compute_cap_specific_rule)
@apply_rule(non_flash_attn_limitation_rule)
@pytest.mark.fmha
@pytest.mark.parametrize(
    'rules',
    [[force_non_flash_attention],
     [force_non_flash_attention, causal_mask, multi_query_attention, pad_s],
     [v1]],
    ids=idfn)
@pytest.mark.parametrize('fmha_arg', reduced2x_combinations_fp16())
def test_fmha_classic_extended_fp16(fmha_arg, rules, dryrun, disable_rules,
                                    gpu_compute_cap):
    fmha_harness(fmha_exe_path,
                 fmha_arg,
                 rules,
                 dryrun,
                 gpu_compute_cap=gpu_compute_cap)


####################################################################################################
# BF16
####################################################################################################


@apply_rule(error_relaxation_rule)
@apply_rule(avoid_oom_rule)
@apply_rule(corner_case_rule)
@apply_rule(compute_cap_specific_rule)
@apply_rule(flash_attn_limitation_rule)
@pytest.mark.fmha
@pytest.mark.parametrize('rules', [[], [force_non_granular_tiling]], ids=idfn)
@pytest.mark.parametrize('fmha_arg', combinations_bf16())
def test_fmha_bf16(fmha_arg, rules, dryrun, disable_rules, gpu_compute_cap):
    fmha_harness(fmha_exe_path,
                 fmha_arg,
                 rules,
                 dryrun,
                 gpu_compute_cap=gpu_compute_cap)


@apply_rule(error_relaxation_rule)
@apply_rule(corner_case_rule)
@apply_rule(compute_cap_specific_rule)
@apply_rule(flash_attn_limitation_rule)
@pytest.mark.fmha
@pytest.mark.parametrize(
    'rules',
    [[], [force_non_granular_tiling], [
        causal_mask, multi_query_attention, pad_s
    ], [force_non_granular_tiling, causal_mask, multi_query_attention, pad_s]],
    ids=idfn)
@pytest.mark.parametrize('fmha_arg', reduced2x_combinations_bf16())
def test_fmha_flash_extended_bf16(fmha_arg, rules, dryrun, disable_rules,
                                  gpu_compute_cap):
    fmha_harness(fmha_exe_path,
                 fmha_arg,
                 rules,
                 dryrun,
                 gpu_compute_cap=gpu_compute_cap)


@apply_rule(non_sm90_only)
@apply_rule(error_relaxation_rule)
@apply_rule(corner_case_rule)
@apply_rule(compute_cap_specific_rule)
@apply_rule(non_flash_attn_limitation_rule)
@pytest.mark.fmha
@pytest.mark.parametrize(
    'rules',
    [[force_non_flash_attention],
     [force_non_flash_attention, causal_mask, multi_query_attention, pad_s]],
    ids=idfn)
@pytest.mark.parametrize('fmha_arg', reduced2x_combinations_bf16())
def test_fmha_classic_extended_bf16(fmha_arg, rules, dryrun, disable_rules,
                                    gpu_compute_cap):
    fmha_harness(fmha_exe_path,
                 fmha_arg,
                 rules,
                 dryrun,
                 gpu_compute_cap=gpu_compute_cap)


####################################################################################################
# INT8
####################################################################################################


@apply_rule(avoid_oom_rule)
@apply_rule(corner_case_rule)
@apply_rule(compute_cap_specific_rule)
@pytest.mark.fmha
@pytest.mark.parametrize('rules', [[]], ids=idfn)
@pytest.mark.parametrize('fmha_arg', combinations_int8())
def test_fmha_int8(fmha_arg, rules, dryrun, disable_rules, gpu_compute_cap):
    fmha_harness(fmha_exe_path,
                 fmha_arg,
                 rules,
                 dryrun,
                 gpu_compute_cap=gpu_compute_cap)


@apply_rule(sm80_only)
@pytest.mark.fmha
@pytest.mark.parametrize('rules', [[]], ids=idfn)
@pytest.mark.parametrize('fmha_arg', combinations_int8_interleaved())
def test_fmha_int8_interleaved(fmha_arg, rules, dryrun, disable_rules,
                               gpu_compute_cap):
    fmha_harness(fmha_exe_path,
                 fmha_arg,
                 rules,
                 dryrun,
                 gpu_compute_cap=gpu_compute_cap)


####################################################################################################
# FP8
####################################################################################################


@apply_rule(war_cublas_sm89_fp8_not_supported)
@apply_rule(compute_cap_specific_rule)
@pytest.mark.fmha
@pytest.mark.parametrize('rules', [[]], ids=idfn)
@pytest.mark.parametrize('fmha_arg', combinations_e4m3())
def test_fmha_e4m3(fmha_arg, rules, dryrun, disable_rules, gpu_compute_cap):
    fmha_harness(fmha_exe_path,
                 fmha_arg,
                 rules,
                 dryrun,
                 gpu_compute_cap=gpu_compute_cap)


####################################################################################################
# ad-hoc benchmarking: will not run by default; requires `-m bench` to run
####################################################################################################


@apply_rule(error_relaxation_rule)
@apply_rule(warmup_runs)
@pytest.mark.bench
@pytest.mark.parametrize('rules', [[], [force_non_granular_tiling]], ids=idfn)
@pytest.mark.parametrize('fmha_arg', combinations_fp16_bench())
def test_fmha_fp16_bench(fmha_arg, rules, dryrun, disable_rules,
                         gpu_compute_cap):
    fmha_harness(fmha_exe_path,
                 fmha_arg,
                 rules,
                 dryrun,
                 gpu_compute_cap=gpu_compute_cap)


@apply_rule(error_relaxation_rule)
@apply_rule(warmup_runs)
@pytest.mark.bench
@pytest.mark.parametrize('rules', [[], [force_non_granular_tiling]], ids=idfn)
@pytest.mark.parametrize('fmha_arg', combinations_fp16_sd_bench())
def test_fmha_fp16_sd_bench(fmha_arg, rules, dryrun, disable_rules,
                            gpu_compute_cap):
    fmha_harness(fmha_exe_path,
                 fmha_arg,
                 rules,
                 dryrun,
                 gpu_compute_cap=gpu_compute_cap)
