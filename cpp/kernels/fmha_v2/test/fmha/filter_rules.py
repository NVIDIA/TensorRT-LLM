# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os

import pytest


# The host softmax reference kernel has limited set of supported sizes
def host_softmax_limitation_rule(fmha_arg, **kwargs):
    if fmha_arg.seq_len not in [
            32, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192, 16384,
            32768
    ]:
        print(fmha_arg)
        pytest.skip(
            reason=
            f'host_softmax_limitation_rule: Host reference softmax only works for seq_len'
            '= [32, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192, 16384, 32768]. {fmha_arg}'
        )


# Relax error tolerance as longer sequence length or head size may accumulate larger errors
def error_relaxation_rule(fmha_arg, **kwargs):
    if fmha_arg.precision == 'bf16':
        return ['-epsilon', '0.025']
    if fmha_arg.seq_len >= 8192:
        return ['-epsilon', '0.022']
    elif fmha_arg.seq_len >= 512 or fmha_arg.head_dim >= 512:
        return ['-epsilon', '0.02']


# Flash attention currently supports bf16/fp16 only
def flash_attn_limitation_rule(fmha_arg, **kwargs):
    cc_tag = kwargs['gpu_compute_cap']
    if fmha_arg.seq_len >= 1024 and fmha_arg.precision == 'int8':
        pytest.skip(
            reason=
            f'flash_attn_limitation_rule: int8 does not yet support seq_len>=1024. '
            '{fmha_arg}')
    if cc_tag == 70 and (fmha_arg.head_dim > 128 or fmha_arg.head_dim <= 16):
        pytest.skip(
            reason=
            f'flash_attn_limitation_rule {cc_tag}: head_dim not supported {fmha_arg}'
        )
    if cc_tag == 75 and (fmha_arg.head_dim > 256):
        pytest.skip(
            reason=
            f'flash_attn_limitation_rule {cc_tag}: head_dim not supported {fmha_arg}'
        )
    if cc_tag == 89 and (fmha_arg.head_dim > 256):
        pytest.skip(
            reason=
            f'flash_attn_limitation_rule {cc_tag}: head_dim not supported due to SMEM'
            ' size. {fmha_arg}')


# Non-flash (ie classic) attention has limited supported size
def non_flash_attn_limitation_rule(fmha_arg, **kwargs):
    cc_tag = kwargs['gpu_compute_cap']
    if cc_tag < 80 and fmha_arg.seq_len == 512:
        pytest.skip(
            reason=
            f'non_flash_attn_limitation_rule: seq_len not supported due to SMEM size. '
            '{fmha_arg}')
    if cc_tag == 75 and fmha_arg.seq_len == 96:
        pytest.skip(
            reason=
            f'non_flash_attn_limitation_rule: seq_len not implemented in arch. {fmha_arg}'
        )
    if fmha_arg.seq_len > 512:
        pytest.skip(
            reason=
            f'non_flash_attn_limitation_rule: does not support seq_len > 512. '
            '{fmha_arg}')
    if fmha_arg.head_dim not in [16, 32, 64]:
        pytest.skip(
            reason=
            f'non_flash_attn_limitation_rule: does not support head_dim other than 16, 32, 64. '
            '{fmha_arg}')


# Some sizes that should work but haven't been made to work
def corner_case_rule(fmha_arg, **kwargs):
    corner_cases = []
    corner_cases.append([16, 64])
    corner_cases.append([32, 16])
    corner_cases.append([32, 32])
    corner_cases.append([32, 64])
    corner_cases.append([64, 16])
    corner_cases.append([64, 32])
    corner_cases.append([96, 16])
    corner_cases.append([96, 32])
    corner_cases.append([192, 16])
    corner_cases.append([192, 32])
    corner_cases.append([192, 64])
    corner_cases.append([384, 16])
    if not fmha_arg.precision == 'int8':
        corner_cases.append([384, 32])

    if [fmha_arg.seq_len, fmha_arg.head_dim] in corner_cases:
        pytest.skip(
            reason=
            f'corner_case_rule: seq_len/head_dim combo not supported by kernels. {fmha_arg}'
        )
    if fmha_arg.seq_len == 32 and fmha_arg.precision == 'int8':
        pytest.skip(
            reason=
            f'corner_case_rule: seq_len/precision combo not supported by kernels. {fmha_arg}'
        )
    if fmha_arg.head_dim > 64 and fmha_arg.precision == 'int8':
        pytest.skip(
            reason=
            f'corner_case_rule: head_dim/precision combo not supported by kernels. {fmha_arg}'
        )


# Ballpark estimation to avoid possible OOM issue inside fmha.exe
def avoid_oom_rule(fmha_arg, **kwargs):
    bytes_per_elt = 1 if fmha_arg.precision == 'int8' else 2
    b = fmha_arg.batch
    s = fmha_arg.seq_len
    h = fmha_arg.num_head
    d = fmha_arg.head_dim
    if max(bytes_per_elt * s * s * b * h,
           4 * bytes_per_elt * b * s * h * d) > (4 * 1024 * 1024 * 1024):
        pytest.skip(
            reason=
            f'avoid_oom_rule: QKVO matrices or P matrix exceed 4 GiB. {fmha_arg}'
        )


# WAR
def war_cublas_sm89_fp8_not_supported(fmha_arg, **kwargs):
    if kwargs['gpu_compute_cap'] == 89 and fmha_arg.precision == 'e4m3':
        return ['-skip-checks']


def compute_cap_specific_rule(fmha_arg, **kwargs):
    cc_tag = kwargs['gpu_compute_cap']
    if cc_tag == 70:
        return _sm70_rule(fmha_arg)
    if cc_tag == 72:
        return _sm70_rule(fmha_arg)
    if cc_tag == 75:
        return _sm75_rule(fmha_arg)
    if cc_tag == 80:
        return _sm80_rule(fmha_arg)
    if cc_tag == 86:
        return _sm80_rule(fmha_arg)
    if cc_tag == 89:
        return _sm89_rule(fmha_arg)
    if cc_tag == 90:
        return _sm90_rule(fmha_arg)


def sm80_only(fmha_arg, **kwargs):
    cc_tag = kwargs['gpu_compute_cap']
    if cc_tag != 80:
        pytest.skip(reason=f'sm_80 only')


def sm90_only(fmha_arg, **kwargs):
    cc_tag = kwargs['gpu_compute_cap']
    if cc_tag != 90:
        pytest.skip(reason=f'sm_90 only')


def non_sm90_only(fmha_arg, **kwargs):
    cc_tag = kwargs['gpu_compute_cap']
    if cc_tag == 90:
        pytest.skip(reason=f'non sm_90 only')


def force_non_granular_tiling(fmha_arg, **kwargs):
    return ['-force-non-granular-tiling']


def force_non_flash_attention(fmha_arg, **kwargs):
    return ['-force-non-flash-attention']


def warmup_runs(fmha_arg, **kwargs):
    return ['-warm-up-runs', '1']


def causal_mask(fmha_arg, **kwargs):
    return ['-causal-mask']


def multi_query_attention(fmha_arg, **kwargs):
    return ['-multi-query-attention']


def grouped_query_attention(fmha_arg, **kwargs):
    assert fmha_arg.num_head % 2 == 0  # assert equal number of heads across query groups
    num_kv_head = fmha_arg.num_head / 2  # 2 query groups
    return ['-grouped-query-attention', str(num_kv_head)]


def pad_s(fmha_arg, **kwargs):
    return ['-pad-s']


def v1(fmha_arg, **kwargs):
    return ['-v1']


#### INTERNAL RULES


# internal rules; use @apply_rule(compute_cap_specific_rule) instead
def _non_sm90_rule(fmha_arg):
    if fmha_arg.use_tma or not fmha_arg.force_non_warp_specialization:
        pytest.skip(
            reason=f'rule: tma/use_warp_specialization not supported. {fmha_arg}'
        )
    if fmha_arg.force_non_flash_attention and \
        (fmha_arg.head_dim not in [16, 32, 64] or fmha_arg.seq_len > 512):
        pytest.skip(
            reason=
            f'rule: head_dim/seq_len is not supported for non-flash-attention kernels. {fmha_arg}'
        )


def _sm70_rule(fmha_arg):
    if fmha_arg.precision in ['int8', 'e4m3', 'bf16', 'fp16-fp32']:
        pytest.skip(
            reason=
            f'sm70_rule: int8/fp8/bf16/fp16-fp32 not supported. {fmha_arg}')
    _non_sm90_rule(fmha_arg)


def _sm75_rule(fmha_arg):
    if fmha_arg.precision in ['int8', 'e4m3', 'bf16', 'fp16-fp32']:
        pytest.skip(
            reason=
            f'sm75_rule: int8/fp8/bf16/fp16-fp32 not supported. {fmha_arg}')
    _non_sm90_rule(fmha_arg)


def _sm80_rule(fmha_arg):
    if fmha_arg.precision in ['e4m3']:
        pytest.skip(reason=f'sm80_rule: fp8 not supported. {fmha_arg}')
    _non_sm90_rule(fmha_arg)


def _sm89_rule(fmha_arg):
    if 'ENABLE_SM89_QMMA' not in os.environ:
        pytest.skip(reason=f'sm89_rule: kernels not generated. {fmha_arg}')
    _non_sm90_rule(fmha_arg)


def _sm90_rule(fmha_arg):
    if 'ENABLE_SM90' not in os.environ:
        pytest.skip(reason=f'sm90_rule: kernels not generated. {fmha_arg}')
    if 'SM90_USE_HMMA' not in os.environ and fmha_arg.precision in ['bf16']:
        pytest.skip(reason=f'sm90_rule: HMMA kernels not generated. {fmha_arg}')
    if 'SM90_USE_IMMA' not in os.environ and fmha_arg.precision == 'int8' and \
        (fmha_arg.head_dim != 64 or fmha_arg.seq_len not in [64, 128, 256, 384, 512]):
        pytest.skip(
            reason=
            f'sm90_rule: non_64_head_size IMMA kernels not generated. {fmha_arg}'
        )

    precision_check = (fmha_arg.precision in ['fp16', 'bf16']) or \
        ('ENABLE_HMMA_FP32' in os.environ and fmha_arg.precision == 'fp16-fp32')

    if precision_check and (fmha_arg.head_dim not in [32, 64, 128, 256]):
        pytest.skip(reason=f'sm90_rule: head_dims not supported;. \
            {fmha_arg}')

    if precision_check \
        and fmha_arg.force_non_warp_specialization \
        and (fmha_arg.head_dim not in [32, 64] or fmha_arg.seq_len not in [64, 128, 256, 384, 512]):
        pytest.skip(
            reason=f'sm90_rule: cases not supported for this head_size/seq_len;. \
            {fmha_arg}')

    if (fmha_arg.force_non_warp_specialization or fmha_arg.force_non_flash_attention) \
        and fmha_arg.use_tma \
        and (fmha_arg.head_dim not in [64] or fmha_arg.seq_len not in [64, 128, 256] or \
            fmha_arg.precision in ['fp16-fp32', 'bf16']):
        pytest.skip(
            reason=
            f'sm90_rule: cases not supported for non-flash-attention tma kernels;. \
            {fmha_arg}')

    if precision_check \
        and fmha_arg.force_non_warp_specialization \
        and not fmha_arg.force_non_flash_attention:
        pytest.skip(
            reason=
            f'sm90_rule: cases not supported for non-warp-specialized flash-attention tma kernels;. \
            {fmha_arg}')

    if fmha_arg.force_non_flash_attention and \
        (fmha_arg.head_dim not in [16, 32, 64] or fmha_arg.seq_len not in [64, 128, 256, 384, 512]):
        pytest.skip(
            reason=
            f'sm90_rule: cases not supported for non-flash-attention kernels;. \
            {fmha_arg}')
