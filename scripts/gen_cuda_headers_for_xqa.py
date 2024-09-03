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
#
# Embed system CUDA headers in c++ arries.

import argparse
import os
from collections import namedtuple
from pathlib import Path

parser = argparse.ArgumentParser(
    description='Embed system CUDA headers in cpp arries',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--output_file', help='Output c++ file name', required=True)
parser.add_argument(
    '--input_files',
    help='Input CUDA header file name list, separated by ","',
    default=
    'cuda_bf16.h,cuda_bf16.hpp,cuda_fp16.h,cuda_fp16.hpp,cuda_fp8.h,cuda_fp8.hpp,vector_types.h,vector_functions.h'
)
parser.add_argument('--cuda_root',
                    help='CUDA Toolkit path',
                    default='/usr/local/cuda')
parser.add_argument(
    '--chunk-size',
    type=int,
    help=
    'Max length for each literal string in the output. Strings would be split into multiple smaller substrings if the length exceeds chunk-size.',
    default=80)

args = parser.parse_args()

TEMPLATE_PROLOGUE = '''/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Prepend the magic string to disable NVRTC encryption.
TEMPLATE_CONTENT = '''constexpr const char* {content_var_name} = "j3iAA#$)7"{content};
constexpr const char* {fname_var_name} = "{fname}";
'''

TEMPLATE_EPILOGUE = '''}
}
'''


# Input: "ThisIsAString.h" / "this_is_a_string.h"
# Output: "this_is_a_string_h"
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


# Returned string includes the surrounding double quotation marks.
def convert_to_cpp_raw_str(s: str):
    chunk_size = args.chunk_size
    if len(s) <= chunk_size:

        def stringify(x: bytes):
            return "\\" + format(x, "03o")

        b = bytes(s, 'utf-8')
        return '"' + ''.join(map(stringify, b)) + '"'
    else:
        string_array = []
        i = 0
        while i < len(s):
            string_array.append(s[i:i + chunk_size])
            i += chunk_size
        return '\n'.join(map(convert_to_cpp_raw_str, string_array))


Entry = namedtuple('Entry', ['content_var_name', 'fname_var_name'])
entries = []

output_content = ''
output_content += TEMPLATE_PROLOGUE
for input_file in args.input_files.split(','):
    fname_var_name = get_canonized_str(input_file) + '_fname'
    content_var_name = get_canonized_str(input_file) + '_content'
    input_full_path = os.path.join(args.cuda_root, 'include', input_file)
    with open(input_full_path, 'r') as f:
        input_content = f.read()
    output_content += TEMPLATE_CONTENT.format(
        content_var_name=content_var_name,
        content=convert_to_cpp_raw_str(input_content),
        fname_var_name=fname_var_name,
        fname=input_file)
    entries.append(
        Entry(content_var_name=content_var_name, fname_var_name=fname_var_name))

output_content += "constexpr char const* cuda_headers_content[] = {\n"
for entry in entries:
    output_content += "    " + entry.content_var_name + ",\n"
output_content += "};\n"

output_content += "constexpr char const* cuda_headers_name[] = {\n"
for entry in entries:
    output_content += "    " + entry.fname_var_name + ",\n"
output_content += "};\n"

output_content += TEMPLATE_EPILOGUE

output_dir = os.path.dirname(args.output_file)
Path(output_dir).mkdir(parents=True, exist_ok=True)

with open(args.output_file, 'w') as f:
    f.write(output_content)
