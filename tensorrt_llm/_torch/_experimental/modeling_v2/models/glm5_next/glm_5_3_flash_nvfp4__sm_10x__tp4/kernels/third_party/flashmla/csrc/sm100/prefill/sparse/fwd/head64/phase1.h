#pragma once

#include "params.h"

namespace sm100::fwd::head64 {

template<int D_QK>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params);

}
