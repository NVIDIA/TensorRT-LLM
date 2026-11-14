// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: MIT

#include <torch/extension.h>

#include "prefill_attention_api.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("run", &fmha_sm100::prefill_bf16x::prefill_run, "SM100 Q8KV4 paged sparse-prefill K1",
             py::arg("q"), py::arg("packed_k"), py::arg("packed_v"), py::arg("k_scale"),
             py::arg("v_scale"), py::arg("kv_indices"), py::arg("kv_indptr"), py::arg("cu_seqlens_q"),
             py::arg("cu_seqlens_k"), py::arg("seqused_k"), py::arg("k2q_row_ptr"),
             py::arg("qsplit_indices"), py::arg("scheduler_metadata"), py::arg("work_count"),
             py::arg("o_partial"), py::arg("lse_partial"), py::arg("k_global_scale"),
             py::arg("v_global_scale"), py::arg("softmax_scale"), py::arg("output_scale"),
             py::arg("combine_tile_counters") = py::none());
  module.def("block_scale_shift", &fmha_sm100::prefill_bf16x::block_scale_shift,
             "Block-scale shift this extension was built with");
  module.def("data_path", &fmha_sm100::prefill_bf16x::data_path,
             "Data path this extension was built with: 0 = E4M3, 1 = BF16");
  module.def("locality_domain_count", &fmha_sm100::prefill_bf16x::locality_domain_count,
             "Memory locality domains of a device", py::arg("device"));
  module.def("locality_page_bytes", &fmha_sm100::prefill_bf16x::locality_page_bytes,
             "Allocation granularity of domain-localized memory", py::arg("device"));
  module.def("sm_locality_domains", &fmha_sm100::prefill_bf16x::sm_locality_domains,
             "Locality domain of every SM id, -1 where unknown", py::arg("device"));
  module.def("alloc_locality_striped", &fmha_sm100::prefill_bf16x::alloc_locality_striped,
             "Device buffer whose pages alternate between the locality domains",
             py::arg("bytes"), py::arg("device"));
}
