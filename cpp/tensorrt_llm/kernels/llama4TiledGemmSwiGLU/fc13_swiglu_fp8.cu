#include "fc13_swiglu_fp8.h"
#include "fc13_swiglu_fp8_per_block.h"

#include <map>
#include <stdexcept>

namespace tensorrt_llm::kernels::llama4_fc_swiglu_tiled
{

std::vector<int> get_tile_tokens() {
    return {1, 2, 3, 4, 8};
}

std::vector<int> get_tile_outs() {
    return {1, 2, 4};
}

void launch_fc13_swiglu_fp8(
    LaunchParams const& params,
    Tactic const& tactic
) {
    if (tactic.algo == "per_block") {
        launch_fc13_swiglu_fp8(params, tactic.tile_token, tactic.tile_out);
    } else {
        throw std::runtime_error("Unsupported algorithm");
    }
}

int get_grid_size(int num_tokens, int hidden_in, int hidden_out, Tactic tactic)
{
    dim3 grid_size;
    if (tactic.algo == "per_block") {
        grid_size = get_grid_size(num_tokens, hidden_in, hidden_out, tactic.tile_token, tactic.tile_out);
    } else {
        throw std::runtime_error("get_grid_size: unsupported algorithm");
    }
    return grid_size.x * grid_size.y * grid_size.z;
}

bool is_supported(int num_tokens, int hidden_in, int hidden_out, Tactic tactic) {
    if (tactic.algo == "per_block") {
        return is_supported(num_tokens, hidden_in, hidden_out, tactic.tile_token, tactic.tile_out);
    } else {
        throw std::runtime_error("is_supported: unsupported algorithm");
    }
}

void llama4_fc_swiglu_tiled_fp8_op(
    int num_tokens, int hidden_in, int hidden_out, void const* A, void const* B, void* C, void const *in_scale, void const *out_scale_inv, cudaStream_t stream)
{
    LaunchParams params = {static_cast<__nv_fp8_e4m3 const*>(A), static_cast<__nv_fp8_e4m3 const*>(B),
        static_cast<__nv_fp8_e4m3*>(C), static_cast<float const*>(in_scale), static_cast<float const*>(out_scale_inv),
        num_tokens, hidden_in, hidden_out, stream};
    Tactic tactic = get_heuristic_tactic(num_tokens, hidden_in, hidden_out);

    launch_fc13_swiglu_fp8(params, tactic);
}

std::optional<Tactic> get_hard_coded_heuristic_tactic(int num_tokens, int hidden_in, int hidden_out) {

    std::map<std::tuple<int, int, int>, Tactic> hard_coded_tactics = {
        {{1, 5120, 2048}, {"per_block", 1, 2}},
        {{1, 5120, 1024}, {"per_block", 1, 1}},
        {{2, 5120, 2048}, {"per_block", 2, 2}},
        {{2, 5120, 1024}, {"per_block", 1, 2}},
        {{4, 5120, 2048}, {"per_block", 2, 2}},
        {{4, 5120, 1024}, {"per_block", 2, 2}},
        {{8, 5120, 2048}, {"per_block", 4, 1}},
        {{8, 5120, 1024}, {"per_block", 4, 1}},
    };

    auto it = hard_coded_tactics.find({num_tokens, hidden_in, hidden_out});
    if (it != hard_coded_tactics.end()) {
        return it->second;
    }

    return std::nullopt;
}

Tactic get_heuristic_tactic(int num_tokens, int hidden_in, int hidden_out)
{
    // Try hard-coded heuristics first.
    auto hard_coded_tactic = get_hard_coded_heuristic_tactic(num_tokens, hidden_in, hidden_out);
    if (hard_coded_tactic) {
        return hard_coded_tactic.value();
    }

    // Fall back to analysis-based heuristics.
    std::string algo = "per_block";
    int tile_token = 1;
    int tile_out = 1;

    const std::vector<int> tile_tokens = get_tile_tokens();
    for (auto t = tile_tokens.rbegin(); t != tile_tokens.rend(); ++t) {
        int const cta_threshold = 1024;
        if (num_tokens % *t == 0
            && get_grid_size(num_tokens, hidden_in, hidden_out, Tactic{algo, *t, tile_out}) >= cta_threshold) {
            tile_token = *t;
            break;
        }
    }

    const std::vector<int> tile_outs = get_tile_outs();
    for (auto t = tile_outs.rbegin(); t != tile_outs.rend(); ++t) {
        int const cta_threshold = 1024;
        if (hidden_in % *t == 0
            && get_grid_size(num_tokens, hidden_in, hidden_out, Tactic{algo, tile_token, *t}) >= cta_threshold) {
            tile_out = *t;
            break;
        }
    }

    Tactic tactic = {algo, tile_token, tile_out};
    if (!is_supported(num_tokens, hidden_in, hidden_out, tactic)) {
        throw std::runtime_error("get_heuristic_tactic: unsupported tactic: " + tactic.to_string());
    }

    return tactic;
}

std::vector<Tactic> get_available_tactics(int num_tokens, int hidden_in, int hidden_out) {
    const std::vector<int> tile_tokens = get_tile_tokens();
    const std::vector<int> tile_outs = get_tile_outs();
    std::vector<Tactic> tactics;
    for (int tile_token : tile_tokens) {
        for (int tile_out : tile_outs) {
            if (is_supported(num_tokens, hidden_in, hidden_out, tile_token, tile_out)) {
                tactics.push_back(Tactic{"per_block", tile_token, tile_out});
            }
        }
    }
    return tactics;
}

// CPU reference implementation
void ref_fc13_swiglu_fp8(std::vector<__nv_fp8_e4m3> const& A, // Input tensor [num_tokens][hidden_in]
    std::vector<__nv_fp8_e4m3> const& B,                      // Input tensor [2 * hidden_out][hidden_in]
    std::vector<__nv_fp8_e4m3>& C,                            // Output tensor [num_tokens][hidden_out]
    int num_tokens, int hidden_in, int hidden_out, float in_scale, float out_scale_inv)
{
    for (int token_idx = 0; token_idx < num_tokens; token_idx++) {
        for (int row = 0; row < hidden_out; row++) {
            float gate_sum = 0.0f;
            float linear_sum = 0.0f;
            for (int i = 0; i < hidden_in; i++) {
                gate_sum += float(A[token_idx * hidden_in + i]) *
                       float(B[row * hidden_in + i]);
                linear_sum += float(A[token_idx * hidden_in + i]) *
                       float(B[(row + hidden_out) * hidden_in + i]);
            }
            float gate = gate_sum * in_scale;
            float gate_value = gate / (1.0f + expf(-gate));
            float linear = linear_sum * in_scale;
            C[token_idx * hidden_out + row] = __nv_fp8_e4m3(gate_value * linear * out_scale_inv);
        }
    }
}

} // namespace tensorrt_llm::kernels::llama4_fc_swiglu_tiled
