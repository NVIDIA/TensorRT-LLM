// Adapted from https://github.com/nunchaku-tech/nunchaku
// @article{
//   li2024svdquant,
//   title={SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models},
//   author={Li*, Muyang and Lin*, Yujun and Zhang*, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze and Meng, Chenlin and Zhu, Jun-Yan and Han, Song},
//   journal={arXiv preprint arXiv:2411.05007},
//   year={2024}
// }

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <torch/extension.h>

#include "common.h"
#include "Tensor.h"

class BufferTorchTensor : public Buffer {
public:
    BufferTorchTensor(at::Tensor tensor) : tensor(std::move(tensor)) {
        this->size        = this->tensor.numel() * this->tensor.itemsize();
        this->ptr         = this->tensor.data_ptr();
        this->device.type = this->tensor.is_cuda() ? Device::CUDA : Device::CPU;
        this->device.idx  = this->tensor.get_device();
    }
    virtual bool isAsyncBuffer() override {
        // TODO: figure out how torch manages memory
        return this->device.type == Device::CUDA;
    }

private:
    at::Tensor tensor;
};

class TorchOpContext {
public:
    TorchOpContext();
    TorchOpContext(const TorchOpContext &) = delete;
    TorchOpContext(TorchOpContext &&)      = delete;
    ~TorchOpContext();
};

Tensor from_torch(at::Tensor input);
at::Tensor to_torch(Tensor input);

class TensorsProviderTorch : public TensorsProvider {
public:
    TensorsProviderTorch(std::map<std::string, at::Tensor> dict) : storage(std::move(dict)) {}

    virtual bool contains(const std::string &key) const override {
        return storage.contains(key);
    }
    virtual Tensor getTensor(const std::string &key) override {
        if (!storage.contains(key)) {
            return Tensor{};
        }
        return from_torch(storage.at(key));
    }

private:
    std::map<std::string, at::Tensor> storage;
};
