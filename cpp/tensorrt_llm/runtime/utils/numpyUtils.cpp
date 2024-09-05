/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/runtime/utils/numpyUtils.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <NvInferRuntime.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::runtime::utils
{

std::string getNumpyTypeDesc(nvinfer1::DataType type)
{
    using dt = nvinfer1::DataType;
    static std::unordered_map<dt, std::string> const type_map{{dt::kBOOL, "?"}, {dt::kUINT8, "u1"}, {dt::kINT8, "i1"},
        {dt::kINT32, "i4"}, {dt::kINT64, "i8"}, {dt::kHALF, "f2"}, {dt::kFLOAT, "f4"}};

    if (type == dt::kBF16)
    {
        TLLM_LOG_WARNING(
            "getNumpyTypeDesc(TYPE_BF16) returns an invalid type 'x' since Numpy doesn't "
            "support bfloat16 as of now, it will be properly extended if numpy supports. "
            "Please refer for the discussions https://github.com/numpy/numpy/issues/19808.");
    }

    return type_map.count(type) > 0 ? type_map.at(type) : "x";
}

nvinfer1::DataType typeFromNumpyDesc(std::string const& type)
{
    TLLM_LOG_DEBUG("numpy type: %s", type.c_str());

    using dt = nvinfer1::DataType;
    static std::unordered_map<std::string, dt> const type_map{{"?", dt::kBOOL}, {"u1", dt::kUINT8}, {"i1", dt::kINT8},
        {"i4", dt::kINT32}, {"i8", dt::kINT64}, {"f2", dt::kHALF}, {"f4", dt::kFLOAT}};
    TLLM_CHECK_WITH_INFO(type_map.count(type) > 0, "numpy data type '" + type + "' not supported");
    return type_map.at(type);
}

void parseNpyIntro(FILE*& f_ptr, uint32_t& header_len, uint32_t& start_data)
{
    char const magic[]
        = "\x93"
          "NUMPY";
    char magic_test[sizeof(magic)] = "\0";

    size_t n_elems = fread((void*) magic_test, sizeof(char), sizeof(magic) - 1, f_ptr);
    if (n_elems != sizeof(magic) - 1 || std::string(magic) != std::string(magic_test))
    {
        throw std::runtime_error("Could read magic token in NPY file");
    }

    uint8_t npy_major = 0;
    uint8_t npy_minor = 0;
    n_elems = fread((void*) &npy_major, sizeof(uint8_t), 1, f_ptr);
    n_elems += fread((void*) &npy_minor, sizeof(uint8_t), 1, f_ptr);

    TLLM_LOG_DEBUG("npy format version: %d.%d", npy_major, npy_minor);

    if (npy_major == 1)
    {
        uint16_t header_len_u16 = 0;
        n_elems = fread((void*) &header_len_u16, sizeof(uint16_t), 1, f_ptr);
        header_len = header_len_u16;
    }
    else if (npy_major == 2)
    {
        uint32_t header_len_u32 = 0;
        n_elems = fread((void*) &header_len_u32, sizeof(uint32_t), 1, f_ptr);
        header_len = header_len_u32;
    }
    else
    {
        throw std::runtime_error("Unsupported npy version: " + std::to_string(npy_major));
    }

    start_data = 8 + 2 * npy_major + header_len;
}

int parseNpyHeader(FILE*& f_ptr, uint32_t header_len, nvinfer1::DataType& type, std::vector<size_t>& shapeVec)
{
    char* header_c = (char*) malloc(header_len * sizeof(char));
    size_t n_elems = fread((void*) header_c, sizeof(char), header_len, f_ptr);
    if (n_elems != header_len)
    {
        free(header_c);
        return -1;
    }
    std::string header(header_c, header_len);
    free(header_c);

    TLLM_LOG_DEBUG("npy header: %s", header.c_str());

    size_t start, end;
    start = header.find("'descr'") + 7;
    start = header.find("'", start);
    // ignore byte order specifier
    if (header[start + 1] == '<' || header[start + 1] == '>' || header[start + 1] == '=')
    {
        ++start;
    }
    end = header.find("'", start + 1);
    type = typeFromNumpyDesc(header.substr(start + 1, end - start - 1));

    start = header.find("'fortran_order'") + 15;
    start = header.find(":", start);
    end = header.find(",", start + 1);
    if (header.substr(start + 1, end - start - 1).find("False") == std::string::npos)
    {
        throw std::runtime_error("Unsupported value for fortran_order while reading npy file");
    }

    start = header.find("'shape'") + 7;
    start = header.find("(", start);
    end = header.find(")", start + 1);

    std::istringstream shape_stream(header.substr(start + 1, end - start - 1));
    std::string token;

    shapeVec.clear();
    while (std::getline(shape_stream, token, ','))
    {
        if (token.find_first_not_of(' ') == std::string::npos)
        {
            break;
        }
        shapeVec.push_back(std::stoul(token));
    }

    return 0;
}

//! \brief Create new tensor from numpy file.
[[nodiscard]] ITensor::UniquePtr loadNpy(
    BufferManager const& manager, std::string const& npyFile, MemoryType const where)
{
    FILE* f_ptr = fopen(npyFile.c_str(), "rb");
    if (f_ptr == nullptr)
    {
        throw std::runtime_error("Could not open file " + npyFile);
    }
    uint32_t header_len, start_data;
    utils::parseNpyIntro(f_ptr, header_len, start_data);

    nvinfer1::DataType type;
    std::vector<size_t> shape;
    utils::parseNpyHeader(f_ptr, header_len, type, shape);

    nvinfer1::Dims dims;
    dims.nbDims = shape.size();
    std::copy(shape.begin(), shape.end(), dims.d);

    auto readWhere = where == MemoryType::kGPU ? MemoryType::kPINNEDPOOL : where;
    auto tensor = manager.allocate(readWhere, dims, type);
    auto data = tensor->data();
    auto eltSize = BufferDataType(tensor->getDataType()).getSize();
    auto size = tensor->getSize();

    size_t n_elems = fread(data, eltSize, size, f_ptr);
    auto const statusCode = fclose(f_ptr);
    TLLM_CHECK_WITH_INFO(statusCode == 0 && n_elems == size, "reading tensor failed");

    if (where == MemoryType::kGPU)
    {
        auto gpuTensor = manager.copyFrom(*tensor, MemoryType::kGPU);
        return gpuTensor;
    }
    return tensor;
}

void saveNpy(BufferManager const& manager, ITensor const& tensor, std::string const& filename)
{
    // Save tensor to NPY 1.0 format (see https://numpy.org/neps/nep-0001-npy-format.html)
    auto const tensorSize = tensor.getSize();
    auto const& shape = tensor.getShape();
    auto const where = tensor.getMemoryType();
    auto const dtype = tensor.getDataType();

#ifdef ENABLE_BF16
    if (dtype == nvinfer1::DataType::kBF16)
    {
        TLLM_CHECK(where == MemoryType::kGPU);
        auto tensorFp32 = manager.gpu(shape, nvinfer1::DataType::kFLOAT);
        auto dataFp32 = bufferCast<float>(*tensorFp32);
        auto dataBf16 = bufferCast<__nv_bfloat16 const>(tensor);
        tc::invokeCudaD2DcpyConvert(dataFp32, dataBf16, tensorSize);
        saveNpy(manager, *tensorFp32, filename);
        return;
    }
#endif

    if (where == MemoryType::kGPU)
    {
        auto tensorHost = manager.copyFrom(tensor, MemoryType::kPINNEDPOOL);
        manager.getStream().synchronize();
        saveNpy(manager, *tensorHost, filename);
        return;
    }

    char const magic[]
        = "\x93"
          "NUMPY";
    uint8_t const npy_major = 1;
    uint8_t const npy_minor = 0;

    std::stringstream header_stream;
    header_stream << "{'descr': '" << getNumpyTypeDesc(dtype) << "', 'fortran_order': False, 'shape': (";
    for (auto i = 0; i < shape.nbDims; ++i)
    {
        header_stream << shape.d[i];
        if (i + 1 < shape.nbDims || shape.nbDims == 1)
        {
            header_stream << ", ";
        }
    }
    header_stream << ")}";
    int base_length = 6 + 4 + static_cast<int>(header_stream.str().size());
    int pad_length = 16 * ((base_length + 1 + 15) / 16); // Take ceiling of base_length + 1 (for '\n' ending)
    for (int i = 0; i < pad_length - base_length; ++i)
    {
        header_stream << ((i == pad_length - base_length - 1) ? "\n" : "\x20");
    }
    std::string header = header_stream.str();
    auto const header_len = static_cast<uint16_t>(header.size());

    FILE* f_ptr = fopen(filename.c_str(), "wb");
    TLLM_CHECK_WITH_INFO(f_ptr != nullptr, tc::fmtstr("Unable to open %s for writing.\n", filename.c_str()));

    fwrite(magic, sizeof(char), sizeof(magic) - 1, f_ptr);
    fwrite(&npy_major, sizeof(uint8_t), 1, f_ptr);
    fwrite(&npy_minor, sizeof(uint8_t), 1, f_ptr);
    fwrite(&header_len, sizeof(uint16_t), 1, f_ptr);
    fwrite(header.c_str(), sizeof(char), header_len, f_ptr);
    auto const eltSize = BufferDataType(dtype).getSize();
    fwrite(tensor.data(), eltSize, tensorSize, f_ptr);

    fclose(f_ptr);
}

} // namespace tensorrt_llm::runtime::utils
