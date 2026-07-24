# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION &
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
set(SRCS executorWorker.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(EXECUTOR_WORKER_TARGET executorWorker)

add_executable(${EXECUTOR_WORKER_TARGET} ${SRCS})

target_link_libraries(${EXECUTOR_WORKER_TARGET}
                      PUBLIC ${SHARED_TARGET} nvinfer_plugin_tensorrt_llm)

target_compile_features(${EXECUTOR_WORKER_TARGET} PRIVATE cxx_std_17)
