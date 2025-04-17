# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/softplus.py

import triton
import triton.language as tl
from packaging import version

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")

if TRITON3:

    @triton.jit
    def softplus(dt):
        return tl.math.log(tl.math.exp(dt) + 1)

else:

    @triton.jit
    def softplus(dt):
        return tl.math.log1p(tl.exp(dt))
