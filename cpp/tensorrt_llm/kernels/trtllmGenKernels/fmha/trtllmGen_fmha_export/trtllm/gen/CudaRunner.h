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
#include <optional>
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

  // Opaque linked program and launch metadata; no filesystem policy lives in the runner.
  struct CompiledArtifact {
    // The serialized jitify2::LinkedProgram, which holds the CUBIN and lowered kernel names.
    std::string mProgram;
    // The dynamic shared memory size in bytes. Loading uses it instead of launching GetSmemSize.
    int32_t mSmemSize{0};
  };

  // Compile the kernel and return a linked NVRTC artifact when available.
  // Returning nullopt means no serializable artifact was produced, not that compilation failed.
  std::optional<CompiledArtifact> compile(Options options = Options{},
                                          bool loadFromCubin = false,
                                          bool exportCubin = false,
                                          int32_t numInstances = 1);

  // Return the SHA-256 digest of bytes as uppercase hex; empty without TLLM_ENABLE_CUDA.
  static std::string computeSha256(std::string const& bytes);

  // Get the kernel.
  Kernel const* getKernel() const;

  // Version of the runtime compilation NVRTC instance encoded like CUDA_VERSION;
  // zero when unavailable.
  static int getNvRtcVersion();

  // Load an artifact returned by compile() or loaded from disk, looking up the current
  // kernel by name. Returns false and leaves the runner unchanged if the artifact cannot be used.
  // Requires a bound kernel. A successful load restores one instance, used with instanceId == 0.
  bool loadArtifact(CompiledArtifact const& artifact);

  // The dimensions of the grid. It will be determined automatically if Grid{} is used.
  using Grid = std::vector<int32_t>;
  // The dimension of the cluster during the runtime.
  using Cluster = std::vector<int32_t>;

  // Run the kernel.
  void run(void* kernelParams,
           void* cudaStream,
           Grid grid = Grid{},
           Cluster cluster = Cluster{},
           int32_t instanceId = 0
#ifdef TLLM_TEST
           ,
           void* cudaEvent = nullptr
#endif // TLLM_TEST
  );

  // Set the kernel.
  void setKernel(Kernel const* kernel);

private:
  // The only instance of the implementation.
  CudaRunnerImpl* mImpl;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gen
} // namespace trtllm
