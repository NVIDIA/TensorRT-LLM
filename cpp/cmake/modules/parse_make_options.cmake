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

function(parse_make_options options result)
  foreach(option ${options})
    string(REGEX REPLACE "(-D|-)" "" option ${option})
    string(REPLACE "=" ";" option ${option})
    list(GET option 0 option_name)
    list(GET option 1 option_value)
    set(${result}_${option_name}
        ${option_value}
        PARENT_SCOPE)
  endforeach()
endfunction()
