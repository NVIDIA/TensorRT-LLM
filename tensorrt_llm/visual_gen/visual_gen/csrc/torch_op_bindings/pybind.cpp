// Adapted from https://github.com/nunchaku-tech/nunchaku
// @article{
//   li2024svdquant,
//   title={SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models},
//   author={Li*, Muyang and Lin*, Yujun and Zhang*, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze
//   and Meng, Chenlin and Zhu, Jun-Yan and Han, Song}, journal={arXiv preprint arXiv:2411.05007}, year={2024}
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

#include "ops.h"
#include "torch.h"
#include <pybind11/pybind11.h>
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def_submodule("ops")
        .def("gemm_w4a4", nunchaku::ops::gemm_w4a4)
        .def("quantize_w4a4_act_fuse_lora", nunchaku::ops::quantize_w4a4_act_fuse_lora);
}
