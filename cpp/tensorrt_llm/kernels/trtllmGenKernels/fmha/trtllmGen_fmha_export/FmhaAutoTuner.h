/*
 * Copyright (c) 2011-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "FmhaOptions.h"

namespace fmha {

///////////////////////////////////////////////////////////////////////////////////////////////////

// Calculate the numCtasX, numCtasY and numCtasZ.
std::tuple<int32_t, int32_t, int32_t> computeNumCtas(FmhaOptions& options,
                                                     int32_t multiProcessorCount,
                                                     bool enablesLogging = true);

///////////////////////////////////////////////////////////////////////////////////////////////////

// AutoTuner to select the kernel based on the heuristics.
// We might also use performance tests to select the best one from the candidates in the future.
// Ensure that multiProcessorCount is obtained from the current device's properties;
// if it does not match the current device, FmhaAutoTuner may fail to choose the optimal Fmha
// configuration.
class FmhaAutoTuner {

public:
  // The constructor.
  FmhaAutoTuner(FmhaOptions const& options,
                FmhaOptionsFromArgs const& optionsFromArgs,
                int32_t multiProcessorCount)
    : mMultiProcessorCount(multiProcessorCount)
    , mOptions(options)
    , mOptionsFromArgs(optionsFromArgs) {}

public:
  // Get the mmaOpsPerClk.
  static int32_t getMmaOpsPerClk(FmhaOptions const& options);

  // Select the GQA generation kernel.
  void selectGqaGenerationKernel();

  // Select the kernel and update the options.
  std::tuple<FmhaOptions, FmhaOptionsFromArgs, int32_t> selectKernel();

  // Select the MLA generation kernel.
  void selectMlaGenerationKernel();

private:
  // Enables the cgaReduction if all clusters can be launched in one wave.
  void enableCgaReduction(int32_t numCtasX, int32_t numCtasY, int32_t numCtasZ);

  // Get the cluster size.
  int32_t getClusterSize();

  // Get the swapsMmaAbTileSizeQ.
  int32_t getSwapsMmaAbTileSizeQ() const;

  // Get the maximum number of active clusters for a given cluster size which considers the
  // floorsweeping configurations.
  int32_t getMaxNumActiveClusters(int32_t clusterSize);

  // Selects the tileSizeQ for GQA generation kernels.
  void selectTileSizeQForGqaGeneration();

  // Set ctaDim.
  void setCtaDim();

  // Set mHeadDimPerStageKv.
  void setHeadDimPerStageKv();

  // Sets the kernel type and tileSizeQ for GQA generation kernels.
  void setGqaKernelTypeAndTileSizeQ();

  // Set the numInstsQ and numInstsKv.
  void setNumInstsQAndKv(FmhaOptions& options, bool forceSet = false, bool updateSetFlags = true);

  // Set mNumKPartitionsMmaPv and mNumKPartitionsTileP.
  void setNumKPartitionsMmaPvAndTileP();

  // Set MMA order, interleavesMufuAndSums, and usesOrderedSequence.
  void setMmaOrder();

  // Set softmax configs.
  void setSoftmaxConfigs();

private:
  // The ctaDim.
  int mCtaDim{512};
  // The multiProcessorCount.
  int mMultiProcessorCount;
  // The FmhaOptions.
  FmhaOptions mOptions;
  // The FmhaOptionsFromArgs.
  FmhaOptionsFromArgs mOptionsFromArgs;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha