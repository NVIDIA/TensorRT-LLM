#pragma once

#include "cuda_fp8.h"

#include <optional>
#include <string>
#include <vector>

namespace tensorrt_llm::kernels::llama4_fc_swiglu_tiled {

struct Tactic {
    std::string algo;
    int tile_token;
    int tile_out;

    std::string to_string() const {
        return algo + "-tile_token:" + std::to_string(tile_token) + "-tile_out:" + std::to_string(tile_out);
    }

    bool operator==(Tactic const& other) const
    {
        return algo == other.algo && tile_token == other.tile_token && tile_out == other.tile_out;
    }
};

struct LaunchParams {
    __nv_fp8_e4m3 const* A;
    __nv_fp8_e4m3 const* B;
    __nv_fp8_e4m3* C;
    float const* in_scale;
    float const* out_scale_inv;
    int num_tokens;
    int hidden_in;
    int hidden_out;
    cudaStream_t stream;
};

void launch_fc13_swiglu_fp8(LaunchParams const& params, Tactic const& tactic);

void llama4_fc_swiglu_tiled_fp8_op(
    int num_tokens, int hidden_in, int hidden_out, void const* A, void const* B, void* C, void const *in_scale, void const *out_scale_inv, cudaStream_t stream);

Tactic get_heuristic_tactic(int num_tokens, int hidden_in, int hidden_out);

std::optional<Tactic> get_hard_coded_heuristic_tactic(int num_tokens, int hidden_in, int hidden_out);

std::vector<Tactic> get_available_tactics(int num_tokens, int hidden_in, int hidden_out);

bool is_supported(int num_tokens, int hidden_in, int hidden_out, Tactic tactic);

int get_grid_size(int num_tokens, int hidden_in, int hidden_out, Tactic tactic);

// CPU reference implementation
void ref_fc13_swiglu_fp8(std::vector<__nv_fp8_e4m3> const& A, // Input tensor [num_tokens][hidden_in]
    std::vector<__nv_fp8_e4m3> const& B,                      // Input tensor [2 * hidden_out][hidden_in]
    std::vector<__nv_fp8_e4m3>& C,                            // Output tensor [num_tokens][hidden_out]
    int num_tokens, int hidden_in, int hidden_out, float in_scale, float out_scale_inv);

} // namespace tensorrt_llm::kernels::llama4_fc_swiglu_tiled
