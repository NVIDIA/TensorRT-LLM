#pragma once

#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::layers
{

void printTokens(char const* name, runtime::ITensor::SharedPtr const& tensor);
#define PRINT_TOKENS(x) tensorrt_llm::layers::printTokens(#x, x)

void printTokens2d(char const* name, runtime::ITensor::SharedPtr const& tensor);
#define PRINT_TOKENS2D(x) tensorrt_llm::layers::printTokens2d(#x, x)

void printTensor(char const* name, runtime::ITensor::SharedPtr const& tensor);
#define PRINT_TENSOR(x) tensorrt_llm::layers::printTensor(#x, x)

} // namespace tensorrt_llm::layers
