# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

function(trtllm_copy_with_fmt source_dir destination_dir)
  file(
    GLOB_RECURSE source_files CONFIGURE_DEPENDS
    LIST_DIRECTORIES FALSE
    "${source_dir}/*")
  file(REMOVE_RECURSE "${destination_dir}")
  foreach(source_file IN LISTS source_files)
    file(RELATIVE_PATH relative_path "${source_dir}" "${source_file}")
    file(READ "${source_file}" content)
    if(relative_path STREQUAL "jit_kernels/impls/runtime_utils.hpp")
      # fmt's hexadecimal floats include a 0x prefix; std::format's do not. Keep
      # the JIT float literals identical to the standard formatter.
      set(float_format
          "std::format(\"{}0x{:a}f\", std::signbit(v) ? \"-\" : \"\", std::abs(v))"
      )
      string(FIND "${content}" "${float_format}" float_format_offset)
      if(float_format_offset EQUAL -1)
        message(
          FATAL_ERROR
            "DeepGEMM float literal formatter has changed; update the fmt compatibility conversion"
        )
      endif()
      string(REPLACE "${float_format}"
                     "::tensorrt_llm::deepGemm::formatFloatLiteral(v)" content
                     "${content}")
      string(REPLACE "#include <format>"
                     "#include \"formatCompatibility.h\"\n#include <format>"
                     content "${content}")
    endif()
    string(REPLACE "#include <format>" "#include <fmt/format.h>" content
                   "${content}")
    string(REPLACE "std::format(" "fmt::format(" content "${content}")
    file(WRITE "${destination_dir}/${relative_path}" "${content}")
    set_property(
      DIRECTORY
      APPEND
      PROPERTY CMAKE_CONFIGURE_DEPENDS "${source_file}")
  endforeach()
endfunction()
