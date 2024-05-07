
#include <sstream>

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"

namespace tensorrt_llm::layers
{

using namespace tensorrt_llm::runtime;
using TensorPtr = ITensor::SharedPtr;

void printTokens2d(char const* name, TensorPtr const& tensor)
{
    auto M = tensor->getShape().d[0];
    auto N = tensor->getShape().d[1];
    auto tr = BufferRange<TokenIdType>(*tensor);
    std::ostringstream buf;
    buf << name << ": " << tensor->getShape() << "(\n";
    for (SizeType mi = 0; mi < M; mi++)
    {
        for (SizeType ni = 0; ni < N; ni++)
        {
            auto token = tr[mi * N + ni];
            if (token >= 0 && token <= 255)
            {
                buf << "'" << static_cast<char>(token) << "'";
            }
            else
            {
                buf << token << "'";
            }
            buf << (ni == (N - 1) ? ';' : ',');
        }
        if (mi != M - 1)
        {
            buf << std::endl;
        }
    }
    buf << ")" << std::endl;
    TLLM_LOG_DEBUG(buf.str());
}

void printTokens(char const* name, TensorPtr const& tensor)
{
    std::ostringstream buf;
    buf << name << ": " << tensor->getShape() << "(";
    for (auto const& token : BufferRange<TokenIdType>(*tensor))
    {
        if (token >= 0 && token <= 255)
        {
            buf << "'" << static_cast<char>(token) << "',";
        }
        else
        {
            buf << token << ",";
        }
    }
    buf << ")" << std::endl << std::flush;
    TLLM_LOG_DEBUG(buf.str());
}

void printTensor(char const* name, TensorPtr const& tensor)
{
    std::ostringstream buf;
    buf << name << ": " << tensor->getShape() << "(";
    for (auto const& token : BufferRange<TokenIdType>(*tensor))
    {
        buf << token << ",";
    }
    buf << ")" << std::endl << std::flush;
    TLLM_LOG_DEBUG(buf.str());
}

} // namespace tensorrt_llm::layers
