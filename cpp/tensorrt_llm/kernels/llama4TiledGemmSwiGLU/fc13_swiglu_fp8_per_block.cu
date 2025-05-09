#include "fc13_swiglu_fp8_per_block.h"
#include "fc13_swiglu_fp8_per_block_template.cuh"
#include "tensorrt_llm/kernels/llama4Utils.cuh"

#include <stdexcept>

namespace tensorrt_llm::kernels::llama4_fc_swiglu_tiled {

void launch_kernel(
    dim3 grid_dim,
    dim3 block_dim,
    cudaStream_t stream,
    void* kernel_func,
    void* args[],
    int num_args
) {
    cudaLaunchConfig_t config;
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = 0;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    config.attrs = attrs;
    config.numAttrs = 0;

    attrs[config.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[config.numAttrs++].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchKernelExC(&config, (void const*) kernel_func, args);
}

template <int TILE_TOKEN, int TILE_OUT>
void dispatch_fc13_swiglu_fp8_hidden_in(__nv_fp8_e4m3 const* __restrict__ A, __nv_fp8_e4m3 const* __restrict__ B,
    __nv_fp8_e4m3* __restrict__ C, float const* __restrict__ in_scale, float const* __restrict__ out_scale_inv,
    int num_tokens, int hidden_in, int hidden_out, cudaStream_t stream)
{
    void* func_ptr;
    constexpr int step = BLOCK_SIZE * VEC_SIZE;

    if (hidden_in % step == 0) {
        if (hidden_in <= step) {
            func_ptr = get_func_ptr_aligned_true_1024_(TILE_TOKEN, TILE_OUT);
        } else if (hidden_in <= 2 * step) {
            func_ptr = get_func_ptr_aligned_true_2048_(TILE_TOKEN, TILE_OUT);
        } else if (hidden_in <= 3 * step) {
            func_ptr = get_func_ptr_aligned_true_3072_(TILE_TOKEN, TILE_OUT);
        } else if (hidden_in <= 4 * step) {
            func_ptr = get_func_ptr_aligned_true_4096_(TILE_TOKEN, TILE_OUT);
        } else if (hidden_in <= 5 * step) {
            func_ptr = get_func_ptr_aligned_true_5120_(TILE_TOKEN, TILE_OUT);
        } else {
            func_ptr = get_func_ptr_aligned_true_0_(TILE_TOKEN, TILE_OUT);
        }
    } else {
        if (hidden_in <= step) {
            func_ptr = get_func_ptr_aligned_false_1024_(TILE_TOKEN, TILE_OUT);
        } else if (hidden_in <= 2 * step) {
            func_ptr = get_func_ptr_aligned_false_2048_(TILE_TOKEN, TILE_OUT);
        } else if (hidden_in <= 3 * step) {
            func_ptr = get_func_ptr_aligned_false_3072_(TILE_TOKEN, TILE_OUT);
        } else if (hidden_in <= 4 * step) {
            func_ptr = get_func_ptr_aligned_false_4096_(TILE_TOKEN, TILE_OUT);
        } else if (hidden_in <= 5 * step) {
            func_ptr = get_func_ptr_aligned_false_5120_(TILE_TOKEN, TILE_OUT);
        } else {
            func_ptr = get_func_ptr_aligned_false_0_(TILE_TOKEN, TILE_OUT);
        }
    }

    const dim3 grid_size = get_grid_size(num_tokens, hidden_in, hidden_out, TILE_TOKEN, TILE_OUT);

    void* args[] = {(void*)&A, (void*)&B, (void*)&C, (void*)&in_scale, (void*)&out_scale_inv, (void*)&num_tokens, (void*)&hidden_in, (void*)&hidden_out};
    launch_kernel(grid_size, dim3(BLOCK_SIZE), stream, func_ptr, args, 8);
}

template <int TILE_TOKEN>
void dispatch_fc13_swiglu_fp8_tile_out(__nv_fp8_e4m3 const* __restrict__ A, __nv_fp8_e4m3 const* __restrict__ B,
    __nv_fp8_e4m3* __restrict__ C, float const* __restrict__ in_scale, float const* __restrict__ out_scale_inv,
    int num_tokens, int hidden_in, int hidden_out, int tile_out, cudaStream_t stream)
{
    if (tile_out == 1) {
        dispatch_fc13_swiglu_fp8_hidden_in<TILE_TOKEN, 1>(A, B, C, in_scale, out_scale_inv, num_tokens, hidden_in, hidden_out, stream);
    } else if (tile_out == 2) {
        dispatch_fc13_swiglu_fp8_hidden_in<TILE_TOKEN, 2>(A, B, C, in_scale, out_scale_inv, num_tokens, hidden_in, hidden_out, stream);
    } else if (tile_out == 4) {
        dispatch_fc13_swiglu_fp8_hidden_in<TILE_TOKEN, 4>(A, B, C, in_scale, out_scale_inv, num_tokens, hidden_in, hidden_out, stream);
    } else {
        throw std::runtime_error("per_block: Unsupported tile_out size " + std::to_string(tile_out));
    }
}

void dispatch_fc13_swiglu_fp8_tile_token(__nv_fp8_e4m3 const* __restrict__ A, __nv_fp8_e4m3 const* __restrict__ B,
    __nv_fp8_e4m3* __restrict__ C, float const* __restrict__ in_scale, float const* __restrict__ out_scale_inv,
    int num_tokens, int hidden_in, int hidden_out, int tile_token, int tile_out, cudaStream_t stream)
{
    if (tile_token == 1) {
        dispatch_fc13_swiglu_fp8_tile_out<1>(A, B, C, in_scale, out_scale_inv, num_tokens, hidden_in, hidden_out, tile_out, stream);
    } else if (tile_token == 2) {
        dispatch_fc13_swiglu_fp8_tile_out<2>(A, B, C, in_scale, out_scale_inv, num_tokens, hidden_in, hidden_out, tile_out, stream);
    } else if (tile_token == 3) {
        dispatch_fc13_swiglu_fp8_tile_out<3>(A, B, C, in_scale, out_scale_inv, num_tokens, hidden_in, hidden_out, tile_out, stream);
    } else if (tile_token == 4) {
        dispatch_fc13_swiglu_fp8_tile_out<4>(A, B, C, in_scale, out_scale_inv, num_tokens, hidden_in, hidden_out, tile_out, stream);
    } else if (tile_token == 8) {
        dispatch_fc13_swiglu_fp8_tile_out<8>(A, B, C, in_scale, out_scale_inv, num_tokens, hidden_in, hidden_out, tile_out, stream);
    } else {
        throw std::runtime_error("per_block: Unsupported tile_token size " + std::to_string(tile_token));
    }
}

void launch_fc13_swiglu_fp8(
    LaunchParams params,
    int tile_token,
    int tile_out
) {
    __nv_fp8_e4m3 const* A = params.A;
    __nv_fp8_e4m3 const* B = params.B;
    __nv_fp8_e4m3* C = params.C;
    float const* in_scale = params.in_scale;
    float const* out_scale_inv = params.out_scale_inv;
    int num_tokens = params.num_tokens;
    int hidden_in = params.hidden_in;
    int hidden_out = params.hidden_out;
    cudaStream_t stream = params.stream;

    if (!is_supported(num_tokens, hidden_in, hidden_out, tile_token, tile_out)) {
        throw std::runtime_error("per_block: unsupported tactic");
    }

    dispatch_fc13_swiglu_fp8_tile_token(A, B, C, in_scale, out_scale_inv, num_tokens, hidden_in, hidden_out, tile_token, tile_out, stream);
}

#define CEIL_DIV(A, B) ((A) + (B) -1) / (B)

dim3 get_grid_size(int num_tokens, int hidden_in, int hidden_out, int tile_token, int tile_out) {
    if (num_tokens % tile_token != 0) {
        throw std::runtime_error("per_block: num_tokens must be divisible by tile_token");
    }
    if (hidden_out % tile_out != 0) {
        throw std::runtime_error("perf_block: hidden_out must be divisible by tile_out");
    }
    return dim3(CEIL_DIV(hidden_out, tile_out), CEIL_DIV(num_tokens, tile_token),1);
}

bool is_supported(int num_tokens, int hidden_in, int hidden_out, int tile_token, int tile_out) {
    if (num_tokens % tile_token != 0) {
        return false;
    }
    if (hidden_out % tile_out != 0) {
        return false;
    }
    if (hidden_in % BLOCK_SIZE * VEC_SIZE != 0) {
        return false;
    }
    return true;
}

} // namespace tensorrt_llm::kernels::llama4_fc_swiglu_tiled
