#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#

function(resolve_dirs dirs_out dirs_in)
  set(dirs_resolved "")

  foreach(dir "${dirs_in}")
    if(IS_SYMLINK "${dir}")
      file(READ_SYMLINK "${dir}" dir_resolved)
      if(NOT IS_ABSOLUTE "${dir_resolved}")
        get_filename_component(dir_prefix "${dir}" DIRECTORY)
        set(dir_resolved "${dir_prefix}/${dir_resolved}")
      endif()
      list(APPEND dirs_resolved "${dir_resolved}")
    else()
      list(APPEND dirs_resolved "${dir}")
    endif()
  endforeach()

  set(${dirs_out}
      "${dirs_resolved}"
      PARENT_SCOPE)
endfunction()
