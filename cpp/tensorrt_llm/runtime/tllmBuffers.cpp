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

#include "tensorrt_llm/runtime/tllmBuffers.h"

namespace tensorrt_llm::runtime
{
template <typename TAllocator>
typename PoolAllocator<TAllocator>::PoolType& PoolAllocator<TAllocator>::getPool()
{
    static PoolType pool;
    return pool;
}

MulticastTensorView::MulticastTensorView(std::weak_ptr<MulticastTensor> const& tensor, ViewType viewType)
    : mTensor(tensor)
    , mViewType(viewType)
    , mDims(mTensor.lock()->getShape())
{
}

MulticastTensorView::MulticastTensorView(MulticastTensorView&& other) noexcept
    : mTensor(std::move(other.mTensor))
    , mViewType(other.mViewType)
    , mDims(mTensor.lock()->getShape())
{
}

MulticastTensorView& MulticastTensorView::operator=(MulticastTensorView&& other) noexcept
{
    if (this != &other)
    {
        // Reset tensor.
        mTensor.reset();
        mTensor.swap(other.mTensor);
        mViewType = other.mViewType;
        mDims = mTensor.lock()->getShape();
    }
    return *this;
}

std::shared_ptr<MulticastBuffer> MulticastTensorView::lock() const
{
    auto sp = mTensor.lock();
    TLLM_CHECK(sp != nullptr);
    return sp;
}

///////////////////////////////////////
// MulticastTensorView ITensor methods
///////////////////////////////////////
nvinfer1::Dims const& MulticastTensorView::getShape() const
{
    return mDims;
}

void MulticastTensorView::reshape(nvinfer1::Dims const& dims)
{
    auto new_size = nonNegative(volume(dims));
    if (new_size > getCapacity())
    {
        TLLM_THROW("MulticastTensorView::reshape() cannot be larger than origin tensor.");
    }
    mDims = dims;
}

///////////////////////////////////////
// MulticastTensorView IBuffer methods
///////////////////////////////////////
void* MulticastTensorView::_data() const
{
    switch (mViewType)
    {
    case ViewType::kUNICAST: return lock()->data();
    case ViewType::kMULTICAST: return lock()->dataMC();
    case ViewType::kIPC_LIST: return lock()->dataIpcList();
    }
    TLLM_THROW("Invalid mViewType");
    return nullptr;
}

std::size_t MulticastTensorView::getSize() const
{
    return lock()->getSize();
}

std::size_t MulticastTensorView::getCapacity() const
{
    return lock()->getCapacity();
}

nvinfer1::DataType MulticastTensorView::getDataType() const
{
    return lock()->getDataType();
}

MemoryType MulticastTensorView::getMemoryType() const
{
    return lock()->getMemoryType();
}

// explicit instantiations
template class PoolAllocator<PinnedAllocator>;
} // namespace tensorrt_llm::runtime
