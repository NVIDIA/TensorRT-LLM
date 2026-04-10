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
#ifndef TLLM_GEN_EXPORT_INTERFACE
#include <trtllm/gen/Kernel.h>
#else
#include <string>
#include <vector>
class Kernel;
#endif // TLLM_GEN_EXPORT_INTERFACE

namespace trtllm {
namespace gen {

////////////////////////////////////////////////////////////////////////////////////////////////////

class CudaRunnerImpl;

////////////////////////////////////////////////////////////////////////////////////////////////////

class CudaRunner {
public:
  // Ctor.
  CudaRunner(Kernel* kernel);
  // Dtor.
  ~CudaRunner();

  // The options to compile the kernel.
  using Options = std::vector<std::string>;
  // Compile the kernel.
  void compile(Options options = Options{},
               bool loadFromCubin = false,
               bool exportCubin = false,
               int32_t numInstances = 1);

  // Get the kernel.
  Kernel const* getKernel() const;

  // The dimensions of the grid. It will be determined automatically if Grid{} is used.
  using Grid = std::vector<int32_t>;
  // The dimension of the cluster during the runtime.
  using Cluster = std::vector<int32_t>;

  // Run the kernel.
  void run(void* kernelParams,
           void* cudaStream,
           Grid grid = Grid{},
           Cluster cluster = Cluster{},
           int32_t instanceId = 0);

  // Set the kernel.
  void setKernel(Kernel const* kernel);

private:
  // The only instance of the implementation.
  CudaRunnerImpl* mImpl;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gen
} // namespace trtllm
