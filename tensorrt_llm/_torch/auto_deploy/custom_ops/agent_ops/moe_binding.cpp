
// moe_binding.cpp
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

// Your .cu must IMPLEMENT this.
void launch_gpu_implementation(void* x, void* selected_experts, void* routing_weights, void** w1_weights,
    void** w2_weights, void* output, int batch_size, int hidden_dim, int intermediate_dim, int num_experts,
    int num_selected, cudaStream_t stream);

torch::Tensor moe_forward(torch::Tensor x, torch::Tensor selected_experts, torch::Tensor routing_weights,
    torch::Tensor w1_weight, torch::Tensor w2_weight)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(selected_experts.is_cuda(), "selected_experts must be CUDA");
    TORCH_CHECK(routing_weights.is_cuda(), "routing_weights must be CUDA");
    TORCH_CHECK(w1_weight.is_cuda(), "w1_weight must be CUDA");
    TORCH_CHECK(w2_weight.is_cuda(), "w2_weight must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(selected_experts.is_contiguous(), "selected_experts must be contiguous");
    TORCH_CHECK(routing_weights.is_contiguous(), "routing_weights must be contiguous");
    TORCH_CHECK(w1_weight.is_contiguous(), "w1_weight must be contiguous");
    TORCH_CHECK(w2_weight.is_contiguous(), "w2_weight must be contiguous");
    TORCH_CHECK(selected_experts.dtype() == torch::kInt32, "selected_experts must be int32");

    auto const dev = x.device();
    TORCH_CHECK(selected_experts.device() == dev && routing_weights.device() == dev, "device mismatch");
    TORCH_CHECK(w1_weight.device() == dev && w2_weight.device() == dev, "weights must be on same device as x");

    TORCH_CHECK(x.dim() == 2, "x must be [B, H]");
    TORCH_CHECK(selected_experts.dim() == 2, "selected_experts must be [B, S]");
    TORCH_CHECK(routing_weights.sizes() == selected_experts.sizes(), "routing_weights must match selected_experts");
    TORCH_CHECK(w1_weight.dim() == 3, "w1_weight must be [N, I, H]");
    TORCH_CHECK(w2_weight.dim() == 3, "w2_weight must be [N, H, I]");

    int const B = (int) x.size(0);
    int const H = (int) x.size(1);
    int const S = (int) selected_experts.size(1);
    int const N = (int) w1_weight.size(0); // num_experts
    int const I = (int) w1_weight.size(1); // intermediate_dim

    TORCH_CHECK(w2_weight.size(0) == N, "w2_weight must have same num_experts as w1_weight");
    TORCH_CHECK(w1_weight.size(2) == H, "w1_weight last dim must equal H");
    TORCH_CHECK(w2_weight.size(1) == H, "w2_weight second dim must equal H");
    TORCH_CHECK(w2_weight.size(2) == I, "w2_weight last dim must equal I");

    // Extract pointers to each expert's weight matrix
    std::vector<void*> w1_ptrs;
    w1_ptrs.reserve(N);
    std::vector<void*> w2_ptrs;
    w2_ptrs.reserve(N);

    char* w1_base = (char*) w1_weight.data_ptr();
    char* w2_base = (char*) w2_weight.data_ptr();
    size_t w1_expert_stride = I * H * w1_weight.element_size();
    size_t w2_expert_stride = H * I * w2_weight.element_size();

    for (int i = 0; i < N; ++i)
    {
        w1_ptrs.push_back(w1_base + i * w1_expert_stride);
        w2_ptrs.push_back(w2_base + i * w2_expert_stride);
    }

    auto out = torch::empty_like(x);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_gpu_implementation(x.data_ptr(), selected_experts.data_ptr(), routing_weights.data_ptr(), w1_ptrs.data(),
        w2_ptrs.data(), out.data_ptr(), B, H, I, N, S, stream);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("moe_forward", &moe_forward, "MoE forward (CUDA)");
}
