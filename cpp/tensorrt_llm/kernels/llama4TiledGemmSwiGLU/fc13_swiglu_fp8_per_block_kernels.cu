#include "fc13_swiglu_fp8_per_block_template.cuh"

namespace tensorrt_llm::kernels::llama4_fc_swiglu_tiled
{

DEFINE_GET_FUNC_PTR(1024, true);
DEFINE_GET_FUNC_PTR(2048, true);
DEFINE_GET_FUNC_PTR(3072, true);
DEFINE_GET_FUNC_PTR(4096, true);
DEFINE_GET_FUNC_PTR(5120, true);
DEFINE_GET_FUNC_PTR(0, true);

DEFINE_GET_FUNC_PTR(1024, false);
DEFINE_GET_FUNC_PTR(2048, false);
DEFINE_GET_FUNC_PTR(3072, false);
DEFINE_GET_FUNC_PTR(4096, false);
DEFINE_GET_FUNC_PTR(5120, false);
DEFINE_GET_FUNC_PTR(0, false);

} // namespace tensorrt_llm::kernels::llama4_fc_swiglu_tiled
