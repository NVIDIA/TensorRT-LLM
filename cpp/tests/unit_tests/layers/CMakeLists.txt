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

set(SAMPLING_LAYER_TEST_SRC
    baseSamplingLayerTest.cpp samplingLayerTest.cpp topKSamplingLayerTest.cpp
    topPSamplingLayerTest.cpp externalDraftTokensLayerTest.cpp)
add_gtest(samplingLayerTest "${SAMPLING_LAYER_TEST_SRC}")

set(BEAM_SEARCH_LAYER_TEST_SRC baseSamplingLayerTest.cpp
                               beamSearchLayerTest.cpp)
add_gtest(beamSearchLayerTest "${BEAM_SEARCH_LAYER_TEST_SRC}")

set(LOOKAHEAD_POOLMANAGER_TEST_SRC randomLlm.cpp lookaheadPoolManagerTest.cpp)
add_gtest(lookaheadPoolManagerTest "${LOOKAHEAD_POOLMANAGER_TEST_SRC}")

set(LOOKAHEAD_ALGORITHM_TEST_SRC randomLlm.cpp lookaheadAlgorithmTest.cpp)
add_gtest(lookaheadAlgorithmTest "${LOOKAHEAD_ALGORITHM_TEST_SRC}")

set(LOOKAHEAD_RANDOMLLM_TEST_SRC randomLlm.cpp lookaheadRandomLlmTest.cpp)
add_gtest(lookaheadRandomLlmTest "${LOOKAHEAD_RANDOMLLM_TEST_SRC}")

set(LOOKAHEAD_DECODING_TEST_SRC randomLlm.cpp lookaheadDecodingLayerTest.cpp)
add_gtest(lookaheadDecodingLayerTest "${LOOKAHEAD_DECODING_TEST_SRC}")

add_gtest(dynamicDecodeLayerTest dynamicDecodeLayerTest.cpp)
add_gtest(eagleLayerTest eagleLayerTest.cpp)
add_gtest(explicitDraftTokensLayerTest explicitDraftTokensLayerTest.cpp)
add_gtest(layerUtilsTest layerUtilsTest.cpp)
add_gtest(medusaDecodeLayerTest medusaDecodeLayerTest.cpp)
