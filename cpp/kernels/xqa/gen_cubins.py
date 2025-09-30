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

# NOTE: this file is for cubin generation, should not in final code release.

import itertools
import multiprocessing
import os
import shutil
import subprocess
import sys
from collections import namedtuple
from typing import List, Tuple

CompileMacro = namedtuple('CompileMacro', 'macro_name short_name value')

CompileMacroOption = namedtuple('CompileMacroOption',
                                'macro_name short_name options')

CompileArchMacrosAndFile = namedtuple('CompileArchMacrosAndFile',
                                      'arch macro_list input_file_name')

build_func_name_prefix = 'xqa_kernel'
arch_options = [80, 86, 90]
config_list = [
    # for llama v2 70b
    [
        CompileMacroOption('DTYPE', 'dt', ['__half', '__nv_bfloat16']),
        CompileMacroOption('HEAD_ELEMS', 'd', [128, 256]),
        CompileMacroOption('BEAM_WIDTH', 'beam', [1]),
        CompileMacroOption('CACHE_ELEM_ENUM', 'kvt', [0, 1, 2]),
        CompileMacroOption(
            'TOKENS_PER_PAGE', 'pagedKV',
            [0, 16, 32, 64, 128]),  # 0 denotes contiguous kv cache.
        CompileMacroOption('HEAD_GRP_SIZE', 'nqpkv', [8]),
        CompileMacroOption('M_TILESIZE', 'm', [8]),
    ],
    # for gptj beamWidth=4
    [
        CompileMacroOption('DTYPE', 'dt', ['__half', '__nv_bfloat16']),
        CompileMacroOption('HEAD_ELEMS', 'd', [256]),
        CompileMacroOption('BEAM_WIDTH', 'beam', [4]),
        CompileMacroOption('CACHE_ELEM_ENUM', 'kvt', [0, 1, 2]),
        CompileMacroOption(
            'TOKENS_PER_PAGE', 'pagedKV',
            [0, 16, 32, 64, 128]),  # 0 denotes contiguous kv cache.
        CompileMacroOption('HEAD_GRP_SIZE', 'nqpkv', [1]),
        CompileMacroOption('M_TILESIZE', 'm', [4]),
    ]
]

clean_cubin = True

cubin_dir = "cubin/"

nvcc_bin = 'nvcc'
nvcc_flags = '-std=c++17 -O3 -cubin -DGENERATE_CUBIN=1 -DNDEBUG --use_fast_math -Xptxas=-v --allow-unsupported-compiler --expt-relaxed-constexpr -t 0'
# nvcc_flags = '-std=c++17 -G -cubin -DGENERATE_CUBIN=1 -Xptxas=-v --allow-unsupported-compiler --expt-relaxed-constexpr -t 0'

cpp_file_prefix_text = R"""/*
* SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION &
* AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
namespace tensorrt_llm
{
namespace kernels
{
// clang-format off
"""

cpp_file_suffex_text = R"""
// clang-format on
} // namespace kernels
} // namespace tensorrt_llm
"""

cubin_meta_info_struct_prefix_text = R"""
static const struct XQAKernelMetaInfo
{
    Data_type mDataType;
    Data_type mKVDataType;
    unsigned int mHeadDim;
    unsigned int mBeamWidth;
    unsigned int mNumQHeadsOverKV;
    unsigned int mMTileSize;
    unsigned int mTokensPerPage;
    bool mPagedKVCache;
    bool mMultiQueryTokens;
    unsigned int mSM;
    const unsigned long long* mCubin;
    unsigned int mCubinSize;
    const char* mFuncName;
} sXqaKernelMetaInfo[] = {
"""

cubin_meta_info_struct_suffix_text = R"""
};
"""

is_spec_dec = False


def generate_cubin_meta_info_line(arch: int, compile_macros: List[CompileMacro],
                                  function_name: str, cubin_size: int,
                                  is_last: bool, is_spec_dec: bool):
    data_type_str = None
    kv_data_type_str = None
    head_dim = None
    beam_width = None
    num_q_heads_per_kv = None
    m_tilesize = None
    paged_kv_cache = None
    tokens_per_page = None
    for compile_macro in compile_macros:
        if compile_macro.macro_name == 'DTYPE':
            data_type_upper_case = map_disp_value(compile_macro.value).upper()
            data_type_str = 'DATA_TYPE_' + data_type_upper_case
        if compile_macro.macro_name == 'CACHE_ELEM_ENUM':
            if compile_macro.value == 0:
                assert data_type_str is not None
                kv_data_type = '__half' if data_type_str == 'DATA_TYPE_FP16' else '__nv_bfloat16'
            elif compile_macro.value == 1:
                kv_data_type = 'int8_t'
            else:
                assert compile_macro.value == 2
                kv_data_type = '__nv_fp8_e4m3'
            kv_type_upper_case = map_disp_value(kv_data_type).upper()
            kv_data_type_str = 'DATA_TYPE_' + kv_type_upper_case
        if compile_macro.macro_name == 'BEAM_WIDTH':
            beam_width = compile_macro.value
        if compile_macro.macro_name == 'HEAD_ELEMS':
            head_dim = compile_macro.value
        if compile_macro.macro_name == 'HEAD_GRP_SIZE':
            num_q_heads_per_kv = compile_macro.value
        if compile_macro.macro_name == 'M_TILESIZE':
            m_tilesize = compile_macro.value
        if compile_macro.macro_name == 'TOKENS_PER_PAGE':
            tokens_per_page = compile_macro.value
            # Power of 2 tokens per page.
            assert (tokens_per_page % 2 == 0)
            paged_kv_cache = 'true' if tokens_per_page > 0 else 'false'

    use_medusa = 'true' if is_spec_dec else 'false'
    assert data_type_str is not None
    assert kv_data_type_str is not None
    assert head_dim is not None
    assert beam_width is not None
    assert num_q_heads_per_kv is not None
    unique_func_name = "kernel_mha"
    fields = [
        data_type_str, kv_data_type_str,
        str(head_dim),
        str(beam_width),
        str(num_q_heads_per_kv),
        str(m_tilesize),
        str(tokens_per_page), paged_kv_cache, use_medusa, f'kSM_{arch}',
        f'{function_name}_cubin', f'{function_name}_cubin_len',
        f'"{unique_func_name}"'
    ]
    field_str = ', '.join(fields)
    line_segs = ["{ ", field_str, "}"]
    if not is_last:
        line_segs.append(',')
    return ''.join(line_segs)


def construct_name(
    func_name_prefix: str,
    arch: int,
    other_name_info: List[str],
    suffix: str = "",
) -> str:
    str_segments = [func_name_prefix, *other_name_info, f"sm_{arch}"]
    name_wo_suffix = '_'.join(str_segments)
    full_name = name_wo_suffix + suffix
    return full_name


name_mapping_dict = {
    '__half': 'fp16',
    '__nv_bfloat16': 'bf16',
    '__nv_fp8_e4m3': 'e4m3',
    'int8_t': 'int8',
    'float': 'fp32',
}


def map_disp_value(value):
    if isinstance(value, str):
        if value in name_mapping_dict.keys():
            return name_mapping_dict[value]
    return value


def build_name_info(compile_macros: List[CompileMacro]):
    short_names = [compile_macro.short_name for compile_macro in compile_macros]
    values = []
    for compile_macro in compile_macros:
        if compile_macro.short_name == 'kvt':
            if compile_macro.value == 0:
                assert compile_macros[0].short_name == 'dt'
                value = compile_macros[0].value
            elif compile_macro.value == 1:
                value = "int8_t"
            elif compile_macro.value == 2:
                value = "__nv_fp8_e4m3"
        else:
            value = compile_macro.value
        values.append(value)
    disp_values = [map_disp_value(value) for value in values]
    name_info = [
        f"{short_name}_{disp_value}"
        for short_name, disp_value in list(zip(short_names, disp_values))
    ]
    if "pagedKV_0" in name_info:
        name_info.remove("pagedKV_0")
    return name_info


def build_commands(
    func_name_prefix: str,
    arch: int,
    input_filename: str,
    compile_macros: List[CompileMacro],
) -> Tuple[str, str, str]:
    arch_str = str(arch) + 'a' if arch in (90, ) else str(arch)
    arch_option = f"-arch=compute_{arch_str} -code=sm_{arch_str}"
    name_info = build_name_info(compile_macros)
    macro_options = [
        f"-D{compile_macro.macro_name}={compile_macro.value}"
        for compile_macro in compile_macros
    ]

    macro_options = []
    for compile_macro in compile_macros:
        if compile_macro.macro_name == "DTYPE":
            if compile_macro.value == "__half":
                macro_options.append(f"-DINPUT_FP16=1")
            elif compile_macro.value == "__nv_bfloat16":
                macro_options.append(f"-DINPUT_FP16=0")
        else:
            macro_options.append(
                f"-D{compile_macro.macro_name}={compile_macro.value}")

    function_name = construct_name(func_name_prefix, arch, name_info)
    macro_options.append(f"-DKERNEL_FUNC_NAME={function_name}")
    all_macro_option = ' '.join(macro_options)
    cubin_file_name = construct_name(func_name_prefix, arch, name_info,
                                     ".cubin")
    output_option = " ".join(["-o", cubin_file_name])
    nvcc_command = " ".join([
        nvcc_bin, nvcc_flags, arch_option, output_option, all_macro_option,
        input_filename
    ])
    xxd_command = " ".join(["xxd -i", cubin_file_name])
    return nvcc_command, xxd_command, cubin_file_name


def save_cubin_cpp_file(xxd_output, func_name_prefix, arch, compile_macros):
    name_info = build_name_info(compile_macros)
    cubin_cpp_file_name = construct_name(func_name_prefix, arch, name_info,
                                         ".cubin.cpp")
    with open(cubin_cpp_file_name, "w") as f:
        f.write(''.join(
            [cpp_file_prefix_text, xxd_output, cpp_file_suffex_text]))


def convert_cubin_cpp_xxd(xxd_command: str, cubin_file_name: str):
    result = subprocess.run(xxd_command.split(' '),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=True,
                            shell=False)
    cubin_cpp_str = result.stdout
    cubin_size = os.path.getsize(cubin_file_name)
    return cubin_cpp_str, cubin_size


def convert_cubin_cpp_np(cubin_file_name: str):
    import numpy as np
    cubin_size = os.path.getsize(cubin_file_name)
    with open(cubin_file_name, 'rb') as f:
        cubin_bin_data = f.read()
    remainder = len(cubin_bin_data) % 8
    if remainder != 0:
        padding = b'\x00' * (8 - remainder)
        cubin_bin_data += padding
    array = np.frombuffer(cubin_bin_data, dtype=np.uint64)
    array_name = cubin_file_name.replace('.', '_')
    elements_per_line = 4
    cpp_array_content = ',\n'.join(
        ', '.join(
            '0x{:016x}ULL'.format(array[i])
            for i in range(start, min(start + elements_per_line, len(array))))
        for start in range(0, len(array), elements_per_line))
    cpp_array = 'unsigned long long ' + array_name + '[] = {\n' + cpp_array_content + '\n};\n' + 'unsigned int ' \
                + array_name + '_len = ' + str(cubin_size) + ';\n'
    return cpp_array, cubin_size


def run_cubin_gen(arch_micro_file_list: CompileArchMacrosAndFile):
    nvcc_command, xxd_command, cubin_file_name = build_commands(
        build_func_name_prefix, arch_micro_file_list.arch,
        arch_micro_file_list.input_file_name, arch_micro_file_list.macro_list)
    function_name = construct_name(
        build_func_name_prefix, arch_micro_file_list.arch,
        build_name_info(arch_micro_file_list.macro_list))
    print(f'generating for {function_name}... command: {nvcc_command}')
    cubin_size = None
    try:
        result = subprocess.run(nvcc_command.split(' '),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=True,
                                shell=False)
        # cubin_cpp_str, cubin_size = convert_cubin_cpp_xxd(xxd_command, cubin_file_name)
        cubin_cpp_str, cubin_size = convert_cubin_cpp_np(cubin_file_name)
        save_cubin_cpp_file(cubin_cpp_str, cubin_dir + build_func_name_prefix,
                            arch_micro_file_list.arch,
                            arch_micro_file_list.macro_list)
        if clean_cubin:
            os.remove(cubin_file_name)
    except subprocess.CalledProcessError as e:
        print(e.stderr)
    print(f'generating for {function_name} done')
    return function_name, cubin_size


def generate_compile_arch_macro_list(compile_macro_options: list):
    option_values = [
        compile_macro_option.options
        for compile_macro_option in compile_macro_options
    ]
    option_macro_names = [
        compile_macro_option.macro_name
        for compile_macro_option in compile_macro_options
    ]
    option_short_names = [
        compile_macro_option.short_name
        for compile_macro_option in compile_macro_options
    ]
    arch_and_macro_list = []
    for arch in arch_options:
        assert isinstance(arch, int)
        for option_combination in itertools.product(*option_values):
            if "__half" in option_combination and "__nv_bfloat16" in option_combination:
                continue
            assert option_macro_names[3] == "CACHE_ELEM_ENUM"
            # fp8 kv cache is only supported on sm89 and next.
            if option_combination[3] == 2 and arch < 89:
                continue
            compile_macros = [
                CompileMacro(*x) for x in zip(
                    option_macro_names, option_short_names, option_combination)
            ]
            if arch in (90, ) and option_combination[
                    3] == 2 and option_combination[2] == 1 and not is_spec_dec:
                input_file_name = "mha_sm90.cu"
            else:
                input_file_name = "mha.cu"
            arch_and_macro_list.append(
                CompileArchMacrosAndFile(arch, compile_macros, input_file_name))
    return arch_and_macro_list


def generate_header_file_contents(
        all_arch_macros: List[CompileArchMacrosAndFile],
        name_size_list: List[Tuple[str, int]], is_spec_dec: bool):
    cubin_data_array = []
    cubin_length_array = []
    meta_line_array = []
    for i, (arch_macro,
            name_size) in enumerate(list(zip(all_arch_macros, name_size_list))):
        arch = arch_macro.arch
        macros = arch_macro.macro_list
        #function_name = construct_name(build_func_name_prefix, arch, build_name_info(macros))
        function_name, cubin_size = name_size
        cubin_variable_name = f"{function_name}_cubin"
        cubin_data_array.append(
            f"extern unsigned long long {cubin_variable_name}[];\n")
        cubin_length_array.append(
            f"extern uint32_t {cubin_variable_name}_len;\n")
        meta_line_array.append(
            generate_cubin_meta_info_line(arch, macros, function_name,
                                          cubin_size,
                                          i == len(all_arch_macros) - 1,
                                          is_spec_dec))
    cubin_data = ''.join(cubin_data_array)
    cubin_length = ''.join(cubin_length_array)
    meta_struct = ''.join([
        cubin_meta_info_struct_prefix_text, '\n'.join(meta_line_array),
        cubin_meta_info_struct_suffix_text
    ])
    return '\n'.join([cubin_data, cubin_length, meta_struct])


if __name__ == "__main__":

    if os.path.exists(cubin_dir):
        shutil.rmtree(cubin_dir)
    os.mkdir(cubin_dir)

    if len(sys.argv) > 1 and sys.argv[1] == 'spec_dec':
        is_spec_dec = True
        nvcc_flags = '-std=c++17 -O3 -cubin -DGENERATE_CUBIN=1 -DNDEBUG -DSPEC_DEC --use_fast_math -Xptxas=-v --allow-unsupported-compiler --expt-relaxed-constexpr -t 0'
        arch_options = [80, 86, 89, 90]
        config_list = [[
            CompileMacroOption('DTYPE', 'dt', ['__half', '__nv_bfloat16']),
            CompileMacroOption('HEAD_ELEMS', 'd', [128]),
            CompileMacroOption('BEAM_WIDTH', 'beam', [1]),
            CompileMacroOption('CACHE_ELEM_ENUM', 'kvt', [0, 1, 2]),
            CompileMacroOption('TOKENS_PER_PAGE', 'pagedKV',
                               [0, 64, 128]),  # 0 denotes contiguous kv cache.
            CompileMacroOption('HEAD_GRP_SIZE', 'nqpkv', [0]),
            CompileMacroOption('M_TILESIZE', 'm', [16, 32]),
        ]]
    arch_macro_lists = []
    for cfg in config_list:
        arch_macro_lists.extend(generate_compile_arch_macro_list(cfg))
    cpu_count = os.cpu_count()
    thread_count = cpu_count // 2 if cpu_count >= 2 else cpu_count
    with multiprocessing.Pool(processes=thread_count) as pool:
        name_size_list = pool.map(run_cubin_gen, arch_macro_lists)
    header_file_contents = generate_header_file_contents(
        arch_macro_lists, name_size_list, is_spec_dec)

    with open(cubin_dir + build_func_name_prefix + '_cubin.h', "w") as f:
        f.write("".join(
            [cpp_file_prefix_text, header_file_contents, cpp_file_suffex_text]))
