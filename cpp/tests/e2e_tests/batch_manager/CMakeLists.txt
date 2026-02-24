# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION &
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

# guidedDecoderTest requires model tokenizer info, so it's easier to run it with
# e2e tests instead of unit tests.
add_gtest(guidedDecoderTest guidedDecoderTest.cpp)
add_gtest(trtEncoderModelTest trtEncoderModelTest.cpp)
add_gtest(trtGptModelTest trtGptModelTest.cpp)
add_gtest(trtGptModelRealDecoderTest trtGptModelRealDecoderTest.cpp)
target_link_libraries(trtGptModelRealDecoderTest PRIVATE testingUtils)
