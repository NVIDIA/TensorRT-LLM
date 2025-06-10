#pragma once

namespace tensorrt_llm::kernels::cutlass_kernels
{

// Note update moe.py to match
enum class ActivationType
{
    Gelu = 0,
    Relu,
    Silu,
    Swiglu,
    Geglu,
    Identity,
    InvalidType
};

} // namespace tensorrt_llm::kernels::cutlass_kernels
