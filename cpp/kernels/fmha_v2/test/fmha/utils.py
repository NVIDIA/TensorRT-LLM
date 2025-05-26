# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import copy
import functools
import subprocess
from collections import namedtuple
from itertools import product
from typing import Callable, List

import pytest

field2arg = {
    'seq_len': '-s',
    'seq_len_q': '-s-q',  # fmhca specific
    'seq_len_kv': '-s-kv',  # fmhca specific
    'min_seq_len': '-min-s',
    'head_dim': '-d',
    'batch': '-b',
    'num_head': '-h',
    'fp16': '-fp16',
    'bf16': '-bf16',
    'fp16_fp32': '-fp16-fp32',
    'int8': '-int8',
    'e4m3': '-e4m3',
    'use_interleaved': '-il',
    'use_tma': '-use-tma',
    'force_non_warp_specialization': '-force-non-warp-specialization',
    'force_non_flash_attention': '-force-non-flash-attention'
}

fields_fmha = [
    'seq_len', 'min_seq_len', 'head_dim', 'batch', 'num_head', 'precision',
    'use_interleaved', 'use_tma', 'force_non_warp_specialization',
    'force_non_flash_attention'
]

fields_fmhca = [
    'seq_len_q',
    'seq_len_kv',
    'min_seq_len',
    'head_dim',
    'batch',
    'num_head',
    'precision',
    'use_interleaved',
]

FmhaArgs = namedtuple('FmhaArgs',
                      fields_fmha,
                      defaults=(256, 1, 64, 1, 1, 'fp16', False, False, True,
                                False))
FmhcaArgs = namedtuple('FmhcaArgs',
                       fields_fmhca,
                       defaults=(256, 256, 1, 64, 1, 1, 'fp16', False))


# custom test name
def idfn(x: List[Callable] or Callable) -> str:
    if isinstance(x, List):
        if len(x) > 0:
            return ".".join([i.__name__ for i in x])
        else:
            return "default"
    else:
        return x.__name__


def combinations_base():
    seq_lens = [
        32, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192, 16384,
        32768
    ]
    head_dims = [16, 32, 40, 64, 80, 128, 160, 256, 512]
    min_seq_lens = [1]
    num_heads = [4]
    batches = [3]
    precision = ['fp16', 'bf16', 'fp16_fp32']
    use_interleaved = [False]
    use_tma = [False, True]
    force_non_warp_specialization = [False, True]
    force_non_flash_attention = [False, True]

    # base combination
    fmha_args = [
        FmhaArgs(*combo) for combo in
        product(seq_lens, min_seq_lens, head_dims, batches, num_heads,
                precision, use_interleaved, use_tma,
                force_non_warp_specialization, force_non_flash_attention)
    ]

    # + min_seq_len = seq_len
    fmha_args_increment1 = fmha_args + [
        fmha_arg._replace(min_seq_len=fmha_arg.seq_len)
        for fmha_arg in fmha_args
    ]

    return fmha_args_increment1


def combinations_fp16():
    fmha_args = combinations_base()
    return [fmha_arg._replace(precision='fp16') for fmha_arg in fmha_args]


def reduced_combinations_fp16():
    fmha_args = combinations_fp16()
    return [
        fmha_arg for fmha_arg in fmha_args
        if fmha_arg.seq_len in [128, 256, 512]
    ]


def reduced2x_combinations_fp16():
    fmha_args = combinations_fp16()
    return [
        fmha_arg for fmha_arg in fmha_args if
        fmha_arg.seq_len in [128, 256, 512] and fmha_arg.head_dim in [32, 64]
    ]


def combinations_bf16():
    fmha_args = combinations_base()
    return [fmha_arg._replace(precision='bf16') for fmha_arg in fmha_args]


def reduced_combinations_bf16():
    fmha_args = combinations_bf16()
    return [
        fmha_arg for fmha_arg in fmha_args
        if fmha_arg.seq_len in [128, 256, 512]
    ]


def reduced2x_combinations_bf16():
    fmha_args = combinations_bf16()
    return [
        fmha_arg for fmha_arg in fmha_args if
        fmha_arg.seq_len in [128, 256, 512] and fmha_arg.head_dim in [32, 64]
    ]


def combinations_int8():
    seq_lens = [32, 64, 96, 128, 192, 256, 384, 512]
    head_dims = [16, 32, 64]
    min_seq_lens = [1]
    num_heads = [2]
    batches = [3]
    precision = ['int8']

    # base combination
    fmha_args = [
        FmhaArgs(*combo) for combo in product(seq_lens, min_seq_lens, head_dims,
                                              batches, num_heads, precision)
    ]

    # + min_seq_len = seq_len
    fmha_args_increment1 = fmha_args + [
        fmha_arg._replace(min_seq_len=fmha_arg.seq_len)
        for fmha_arg in fmha_args
    ]

    return fmha_args_increment1


def combinations_fp16_bench():
    seq_lens = [512]
    head_dims = [64, 128, 256, 512]
    min_seq_lens = [512]
    num_heads = [16]
    batches = [16]
    precision = ['fp16']

    # base combination
    fmha_args = [
        FmhaArgs(*combo) for combo in product(seq_lens, min_seq_lens, head_dims,
                                              batches, num_heads, precision)
    ]

    # + min_seq_len = seq_len
    fmha_args_increment1 = fmha_args + [
        fmha_arg._replace(seq_len=1024, min_seq_len=1024, batch=8)
        for fmha_arg in fmha_args
    ]
    fmha_args_increment2 = fmha_args_increment1 + [
        fmha_arg._replace(seq_len=2048, min_seq_len=2048, batch=4)
        for fmha_arg in fmha_args
    ]
    fmha_args_increment3 = fmha_args_increment2 + [
        fmha_arg._replace(seq_len=4096, min_seq_len=4096, batch=2)
        for fmha_arg in fmha_args
    ]
    fmha_args_increment4 = fmha_args_increment3 + [
        fmha_arg._replace(seq_len=32768, min_seq_len=32768, batch=1)
        for fmha_arg in fmha_args
    ]

    return fmha_args_increment4


def combinations_fp16_sd_bench():
    seq_lens = [4096]
    min_seq_lens = [4096]
    head_dims = [40]
    num_heads = [8]
    batches = [2, 4, 8, 16, 32]
    precision = ['fp16']

    # base combination
    fmha_args = [
        FmhaArgs(*combo) for combo in product(seq_lens, min_seq_lens, head_dims,
                                              batches, num_heads, precision)
    ]

    # + min_seq_len = seq_len
    fmha_args_increment1 = fmha_args + [
        fmha_arg._replace(seq_len=1024, min_seq_len=1024, head_dim=80)
        for fmha_arg in fmha_args
    ]
    fmha_args_increment2 = fmha_args_increment1 + [
        fmha_arg._replace(seq_len=256, min_seq_len=256, head_dim=160)
        for fmha_arg in fmha_args
    ]
    fmha_args_increment3 = fmha_args_increment2 + [
        fmha_arg._replace(seq_len=64, min_seq_len=64, head_dim=160)
        for fmha_arg in fmha_args
    ]

    return fmha_args_increment3


def combinations_fmhca():
    """
    bin/fmhca.exe -b 1 -s-q 4096 -min-s 4096 -d  40
    bin/fmhca.exe -b 1 -s-q 4096 -min-s 4096 -d  80
    bin/fmhca.exe -b 1 -s-q 4096 -min-s 4096 -d 160

    bin/fmhca.exe -b 4 -s-q 4096 -min-s 4096 -d  40
    bin/fmhca.exe -b 4 -s-q 4096 -min-s 4096 -d  80
    bin/fmhca.exe -b 4 -s-q 4096 -min-s 4096 -d 160

    bin/fmhca.exe -b 1 -s-q 2304 -min-s 2304 -d  40
    bin/fmhca.exe -b 1 -s-q 2304 -min-s 2304 -d  80
    bin/fmhca.exe -b 1 -s-q 2304 -min-s 2304 -d 160

    bin/fmhca.exe -b 4 -s-q 2304 -min-s 2304 -d  40
    bin/fmhca.exe -b 4 -s-q 2304 -min-s 2304 -d  80
    bin/fmhca.exe -b 4 -s-q 2304 -min-s 2304 -d 160

    bin/fmhca.exe -b 1 -s-q 1024 -min-s 1024 -d  40
    bin/fmhca.exe -b 1 -s-q 1024 -min-s 1024 -d  80
    bin/fmhca.exe -b 1 -s-q 1024 -min-s 1024 -d 160

    bin/fmhca.exe -b 4 -s-q 1024 -min-s 1024 -d  40
    bin/fmhca.exe -b 4 -s-q 1024 -min-s 1024 -d  80
    bin/fmhca.exe -b 4 -s-q 1024 -min-s 1024 -d 160
    """
    seq_len_qs = [1024, 2304, 4096]
    seq_len_kvs = [77]  # ?
    min_seq_len = [1]
    head_dims = [40, 80, 160]
    num_heads = [16]
    batches = [1, 4]
    precision = ['fp16']

    # base combination
    fmha_args = [
        FmhcaArgs(*combo) for combo in product(
            seq_len_qs,
            seq_len_kvs,
            min_seq_len,
            head_dims,
            batches,
            num_heads,
            precision,
        )
    ]

    # min_seq_len = seq_len
    fmha_args_increment1 = [
        fmha_arg._replace(min_seq_len=fmha_arg.seq_len_q)
        for fmha_arg in fmha_args
    ]

    return fmha_args_increment1


def combinations_e4m3():
    """
    bin/fmha.exe -v 0 -runs 1 -s 512 -d 64 -min-s 1 -b 1 -h 1 -e4m3
    bin/fmha.exe -v 0 -runs 1 -s 384 -d 64 -min-s 1 -b 1 -h 1 -e4m3
    bin/fmha.exe -v 0 -runs 1 -s 256 -d 64 -min-s 1 -b 1 -h 1 -e4m3
    bin/fmha.exe -v 0 -runs 1 -s 128 -d 64 -min-s 1 -b 1 -h 1 -e4m3

    bin/fmha.exe -v 0 -runs 1 -s 512 -d 64 -min-s 1 -b 1 -h 1 -e4m3 -ignore-b1opt
    bin/fmha.exe -v 0 -runs 1 -s 384 -d 64 -min-s 1 -b 1 -h 1 -e4m3 -ignore-b1opt
    bin/fmha.exe -v 0 -runs 1 -s 256 -d 64 -min-s 1 -b 1 -h 1 -e4m3 -ignore-b1opt
    bin/fmha.exe -v 0 -runs 1 -s 128 -d 64 -min-s 1 -b 1 -h 1 -e4m3 -ignore-b1opt
    """
    seq_lens = [128, 256, 384, 512]
    head_dims = [64]
    min_seq_lens = [1]
    num_heads = [2]
    batches = [3]
    precision = ['e4m3']

    # base combination
    fmha_args = [
        FmhaArgs(*combo) for combo in product(seq_lens, min_seq_lens, head_dims,
                                              batches, num_heads, precision)
    ]

    # + min_seq_len = seq_len
    fmha_args_increment1 = fmha_args + [
        fmha_arg._replace(min_seq_len=fmha_arg.seq_len)
        for fmha_arg in fmha_args
    ]

    return fmha_args_increment1


def combinations_int8_interleaved():
    """
    bin/fmha.exe -v 0 -runs 1 -il -int8 -s 128 -d 64 -min-s 128 -b 1
    bin/fmha.exe -v 0 -runs 1 -il -int8 -s 128 -d 64 -min-s 128 -b 128
    bin/fmha.exe -v 0 -runs 1 -il -int8 -s 128 -d 64 -min-s   1 -b 1
    bin/fmha.exe -v 0 -runs 1 -il -int8 -s 128 -d 64 -min-s   1 -b 128

    bin/fmha.exe -v 0 -runs 1 -il -int8 -s 192 -d 64 -min-s 192 -b 1
    bin/fmha.exe -v 0 -runs 1 -il -int8 -s 192 -d 64 -min-s 192 -b 128
    bin/fmha.exe -v 0 -runs 1 -il -int8 -s 192 -d 64 -min-s   1 -b 1
    bin/fmha.exe -v 0 -runs 1 -il -int8 -s 192 -d 64 -min-s   1 -b 128

    bin/fmha.exe -v 0 -runs 1 -il -int8 -s 256 -d 64 -min-s 256 -b 1
    bin/fmha.exe -v 0 -runs 1 -il -int8 -s 256 -d 64 -min-s 256 -b 128
    bin/fmha.exe -v 0 -runs 1 -il -int8 -s 256 -d 64 -min-s   1 -b 1
    bin/fmha.exe -v 0 -runs 1 -il -int8 -s 256 -d 64 -min-s   1 -b 128

    bin/fmha.exe -v 0 -runs 1 -il -int8 -s 384 -d 64 -min-s 384 -b 1
    bin/fmha.exe -v 0 -runs 1 -il -int8 -s 384 -d 64 -min-s 384 -b 128
    bin/fmha.exe -v 0 -runs 1 -il -int8 -s 384 -d 64 -min-s   1 -b 1
    bin/fmha.exe -v 0 -runs 1 -il -int8 -s 384 -d 64 -min-s   1 -b 128
    """
    seq_lens = [128, 192, 256, 384]
    head_dims = [64]
    min_seq_lens = [1]
    num_heads = [1, 16]
    batches = [1, 128]
    precision = ['int8']
    use_interleaved = [True]

    # base combination
    fmha_args = [
        FmhaArgs(*combo)
        for combo in product(seq_lens, min_seq_lens, head_dims, batches,
                             num_heads, precision, use_interleaved)
    ]

    # + min_seq_len = seq_len
    fmha_args_increment1 = fmha_args + [
        fmha_arg._replace(min_seq_len=fmha_arg.seq_len)
        for fmha_arg in fmha_args
    ]

    return fmha_args_increment1


def combinations_small():
    seq_lens = [4096]
    head_dims = [64]
    min_seq_lens = [1]
    num_heads = [1]
    batches = [1, 512]
    precision = ['fp16']

    # base combination
    fmha_args = [
        FmhaArgs(*combo) for combo in product(seq_lens, min_seq_lens, head_dims,
                                              batches, num_heads, precision)
    ]

    return fmha_args


def base_command(exe_path):
    return [exe_path, '-v', '0', '-runs', '1']


def apply_rule(rule):
    """
    decorator for tests which accepts rule f(fmha_arg, **kwargs) as argument to filter out specific
    combinations of arguments and kwargs
    """

    def apply_rule_(fmha_harness):
        # make wrapper looks like original fmha_harness to avoid interference with pytest inner workings
        @functools.wraps(fmha_harness)
        def fmha_harness_wrapper(**kwargs):
            # rules (dtype = pytest.fixture) is mutable; deepcopy to avoid changing test states
            rules_copy = copy.deepcopy(kwargs.get('rules', []))
            # if disable_rules exists and is False, apply rules
            # if it somehow does not exist, assume we want to apply rules
            try:
                if not kwargs['disable_rules']:
                    rules_copy.append(rule)
            except:
                rules_copy.append(rule)
            kwargs_copy = copy.deepcopy(kwargs)
            kwargs_copy['rules'] = rules_copy
            return fmha_harness(**kwargs_copy)

        return fmha_harness_wrapper

    return apply_rule_


def sanitize_prompt(prompt):
    return [l for l in prompt if l != '']


def fmha_harness(exe_path, fmha_arg, rules=[], dryrun=False, **kwargs):
    """
    exe_path: path to executable
    fmha_arg: arguments to pass the executable
    rules: a list of functionals f(fmha_arg, **kwargs) that accepts fmha_arg and additional argument
           for filtering out specific inputs
    dryrun: print command line without actually invoking it
    **kwargs: optional kwargs to pass to rules
    """
    # print(str(fmha_arg))
    prompt = base_command(exe_path)
    for rule in rules:
        rule_added_prompt = rule(fmha_arg, **kwargs)
        prompt += rule_added_prompt if rule_added_prompt is not None else ""
    for k, v in fmha_arg._asdict().items():
        if k == 'precision':
            prompt += [
                field2arg[v]
            ]  # kv pair (precision, dtype) maps to -dtype in command line
        elif k == 'use_interleaved' or k == 'use_tma' or k == 'force_non_warp_specialization' \
            or k == 'force_non_flash_attention':
            if v is True:
                prompt += [
                    field2arg[k]
                ]  # kv pair (key, true) maps to field2arg in command line
        else:
            prompt += [field2arg[k]]
            prompt += [str(fmha_arg._asdict()[k])]
    prompt = sanitize_prompt(prompt)
    print(f'Full prompt: "{" ".join(prompt)}"')
    if not dryrun:
        try:
            subprocess.run(prompt, check=True)
        except subprocess.CalledProcessError as e:
            pytest.fail(
                f'Exception caught during subprocess call: "{" ".join(prompt)}" returns {e.returncode}'
            )
