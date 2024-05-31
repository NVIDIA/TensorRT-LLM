
#include <sstream>

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"

namespace tensorrt_llm::layers
{

using namespace tensorrt_llm::runtime;
using TensorPtr = ITensor::SharedPtr;

ITensor::UniquePtr slice(
    ITensor::SharedPtr tensor, std::initializer_list<SizeType32> const& offsetDims, size_t const sizeDim)
{
    auto shape = tensor->getShape();
    TLLM_CHECK(offsetDims.size() > 0);
    TLLM_CHECK(shape.nbDims >= offsetDims.size());
    std::vector<size_t> volumes(shape.nbDims);

    int i;
    volumes[shape.nbDims - 1] = 1;
    for (i = shape.nbDims - 2; i >= 0; i--)
    {
        volumes[i] = shape.d[i + 1] * volumes[i + 1];
    }

    size_t offset = 0;
    i = 0;
    for (auto itd = offsetDims.begin(); itd != offsetDims.end(); itd++)
    {
        TLLM_CHECK(0 <= (*itd) && (*itd) < shape.d[i]);
        offset += (*itd) * volumes[i++];
    }

    ITensor::Shape dims;
    dims.nbDims = shape.nbDims - offsetDims.size() + 1;
    dims.d[0] = sizeDim;
    for (i = 1; i < dims.nbDims; i++)
    {
        dims.d[i] = shape.d[i - 1 + offsetDims.size()];
    }

    size_t size = ITensor::volume(dims);

    return std::make_unique<TensorView>(std::move(tensor), offset, size, dims);
}

} // namespace tensorrt_llm::layers
