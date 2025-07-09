#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import os
from collections import defaultdict, namedtuple
from pathlib import Path

# Example: input_fname='mhaUtils.cuh', output_fname='mha_utils_cuh.h', content_var_name='mha_utils_cuh_content', fname_var_name='mha_utils_cuh_header'
Entry = namedtuple(
    'Entry',
    ['input_fname', 'output_fname', 'content_var_name', 'fname_var_name'])

parser = argparse.ArgumentParser(
    description='Generate cpp headers from kernel source file for NVRTC.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-o',
                    '--output',
                    help='output header file name',
                    default='generated/xqa_sources.h')
parser.add_argument('--embed-cuda-headers',
                    action='store_true',
                    help='embed cuda headers',
                    default=True)
parser.add_argument('--no-embed-cuda-headers',
                    dest='embed_cuda_headers',
                    action='store_false')
parser.add_argument(
    '--cuda_root',
    help='CUDA Toolkit path (applicable when --embed-cuda-headers is ON)',
    default='/usr/local/cuda')

args = parser.parse_args()


def convert_to_raw_cpp_string(s: str):

    def stringify(x: bytes):
        return "\\" + format(x, "03o")

    b = bytes(s, 'utf-8')
    return ''.join(map(stringify, b))


def get_canonized_str(s: str):
    tokens = []
    n = len(s)
    i = 0
    while i < n and not s[i].isalpha() and not s[i].isdigit():
        i += 1
    while i < n:
        j = i + 1
        while j < n and (s[j].islower() or s[j].isdigit()):
            j += 1
        tokens.append(s[i:j].lower())
        while j < n and not s[j].isalpha() and not s[j].isdigit():
            j += 1
        i = j
    return '_'.join(tokens)


def get_entry_from_input_fname(input_fname: str):
    canonized_str = get_canonized_str(os.path.basename(input_fname))
    output_fname = None
    output_fname = args.output
    return Entry(input_fname=input_fname,
                 output_fname=output_fname,
                 content_var_name=canonized_str + '_content',
                 fname_var_name=canonized_str + "_fname")


def is_header(entry: Entry):
    return entry.input_fname[-3:] != ".cu"


SOURCE_FILES = [
    'cuda_hint.cuh', 'defines.h', 'ldgsts.cuh', 'mha.cu', 'mha_sm90.cu',
    'mla_sm120.cu', 'mha.h', 'mhaUtils.cuh', 'mma.cuh', 'platform.h',
    'utils.cuh', 'utils.h', 'mha_stdheaders.cuh', 'mha_components.cuh',
    'mla_sm120.cuh', 'gmma.cuh', 'gmma_impl.cuh', 'barriers.cuh', 'tma.h',
    'specDec.h'
]

CUDA_HEADERS = [
    'cuda_bf16.h', 'cuda_bf16.hpp', 'cuda_fp16.h', 'cuda_fp16.hpp',
    'cuda_fp8.h', 'cuda_fp8.hpp', 'vector_types.h', 'vector_functions.h',
    'device_types.h'
]

all_files = SOURCE_FILES
if args.embed_cuda_headers:
    all_files += [
        os.path.join(args.cuda_root, 'include', i) for i in CUDA_HEADERS
    ]

all_entries = map(get_entry_from_input_fname, all_files)

TEMPLATE_PROLOGUE = '''/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once
namespace tensorrt_llm {
namespace kernels {
'''

TEMPLATE_CONTENT = '''inline constexpr const char* {content_var_name} = "{content}";
inline constexpr const char* {fname_var_name} = "{fname}";
'''

TEMPLATE_EPILOGUE = '''}
}
'''

D = defaultdict(list)
for entry in all_entries:
    output_fname = entry.output_fname
    D[output_fname].append(entry)

for output_fname, entries in D.items():
    output_content = ''
    output_content += TEMPLATE_PROLOGUE
    for entry in entries:
        with open(entry.input_fname, 'r') as f:
            input_content = f.read()
        output_content += TEMPLATE_CONTENT.format(
            content_var_name=entry.content_var_name,
            content=convert_to_raw_cpp_string(input_content),
            fname_var_name=entry.fname_var_name,
            fname=os.path.basename(entry.input_fname))

    output_content += "inline constexpr char const* xqa_headers_content[] = {\n"
    for entry in entries:
        if is_header(entry):
            output_content += "    " + entry.content_var_name + ",\n"
    output_content += "};\n"

    output_content += "inline constexpr char const* xqa_headers_name[] = {\n"
    for entry in entries:
        if is_header(entry):
            output_content += "    " + entry.fname_var_name + ",\n"
    output_content += "};\n"

    output_content += TEMPLATE_EPILOGUE

    output_dir = os.path.dirname(entry.output_fname)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(entry.output_fname, 'w') as f:
        f.write(output_content)
