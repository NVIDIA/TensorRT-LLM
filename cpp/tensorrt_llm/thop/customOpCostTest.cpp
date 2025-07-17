#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "tensorrt_llm/thop/thUtils.h"

// namespace torch_ext
// {
//     torch::Tensor custom_op_cost_test(torch::Tensor input, torch::Tensor weight, torch::Tensor input_sf, torch::Tensor weight_sf) {
//         // auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
//         // auto output = torch::empty({input.size(0), input.size(1)}, input.options());
//         // return output;

//         return torch::empty({0}, input.options());
//     }
// } // namespace torch_ext

// TORCH_LIBRARY_FRAGMENT(trtllm, m)
// {
//     m.def(
//         "custom_op_cost_test(Tensor input, Tensor weight, Tensor input_sf, Tensor weight_sf)"
//         "-> (Tensor)");
// }

// TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
// {
//     m.impl("custom_op_cost_test", &torch_ext::custom_op_cost_test);
// }

namespace torch_ext
{
    torch::Tensor custom_op_cost_test(torch::Tensor input, torch::Tensor weight, torch::Tensor input_sf, torch::Tensor weight_sf) {
        // auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
        // auto output = torch::empty({input.size(0), input.size(1)}, input.options());
        // return output;

        return torch::empty({0}, input.options());
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> custom_op_cost_test_16_input_tensors(
        torch::Tensor input_sf_1, torch::Tensor weight_sf_1, 
        torch::Tensor input_sf_2, torch::Tensor weight_sf_2, 
        torch::Tensor input_sf_3, torch::Tensor weight_sf_3, 
        torch::Tensor input_sf_4, torch::Tensor weight_sf_4, 
        torch::Tensor input_sf_5, torch::Tensor weight_sf_5, 
        torch::Tensor input_sf_6, torch::Tensor weight_sf_6, 
        torch::Tensor input_sf_7, torch::Tensor weight_sf_7, 
        torch::Tensor input_sf_8, torch::Tensor weight_sf_8)
    {
        // auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
        // auto output = torch::empty({input.size(0), input.size(1)}, input.options());
        // return output;

        return std::make_tuple(torch::empty({0}, input_sf_1.options()), torch::empty({0}, input_sf_2.options()), torch::empty({0}, input_sf_3.options()), torch::empty({0}, input_sf_4.options()), torch::empty({0}, input_sf_5.options()), torch::empty({0}, input_sf_6.options()));
    }
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "custom_op_cost_test(Tensor input, Tensor weight, Tensor input_sf, Tensor weight_sf)"
        "-> (Tensor)");
    m.def(
        "custom_op_cost_test_16_input_tensors(Tensor input_sf_1, Tensor weight_sf_1, Tensor input_sf_2, Tensor weight_sf_2, Tensor input_sf_3, Tensor weight_sf_3, Tensor input_sf_4, Tensor weight_sf_4, Tensor input_sf_5, Tensor weight_sf_5, Tensor input_sf_6, Tensor weight_sf_6, Tensor input_sf_7, Tensor weight_sf_7, Tensor input_sf_8, Tensor weight_sf_8)"
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("custom_op_cost_test", &torch_ext::custom_op_cost_test);
    m.impl("custom_op_cost_test_16_input_tensors", &torch_ext::custom_op_cost_test_16_input_tensors);
}
