// clang-format off
#include "checkpointing_ssu_config.inc"
#include <flashinfer/mamba/checkpointing_ssu.cuh>
#include <flashinfer/mamba/launch_checkpointing_ssu.cuh>
// clang-format on

namespace flashinfer::mamba::checkpointing {

template void launchCheckpointingSsu<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t,
                                     state_scale_t>(CheckpointingSsuParams&, cudaStream_t);

}  // namespace flashinfer::mamba::checkpointing
