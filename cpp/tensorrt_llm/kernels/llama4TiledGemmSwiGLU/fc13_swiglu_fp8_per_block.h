#pragma once

#include "fc13_swiglu_fp8.h"

namespace tensorrt_llm::kernels::llama4_fc_swiglu_tiled {

void launch_fc13_swiglu_fp8(LaunchParams params, int tile_token, int tile_out);

dim3 get_grid_size(int num_tokens, int hidden_in, int hidden_out, int tile_token, int tile_out);

bool is_supported(int num_tokens, int hidden_in, int hidden_out, int tile_token, int tile_out);

} // namespace tensorrt_llm::kernels::llama4_fc_swiglu_tiled
