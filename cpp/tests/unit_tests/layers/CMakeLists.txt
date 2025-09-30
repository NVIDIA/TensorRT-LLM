# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: NVIDIA TensorRT
# Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

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
