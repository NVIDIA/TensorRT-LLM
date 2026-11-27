# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from utils.util import skip_ray

from .llm_postprocess_test_utils import (
    run_llm_with_postprocess_parallel,
    run_llm_with_postprocess_parallel_and_result_handler,
)


@pytest.mark.threadleak(enabled=False)
def test_llm_with_postprocess_parallel():
    run_llm_with_postprocess_parallel(tp_size=1)


@skip_ray
@pytest.mark.parametrize("streaming", [True, False])
def test_llm_with_postprocess_parallel_and_result_handler(streaming):
    run_llm_with_postprocess_parallel_and_result_handler(streaming, "pytorch", tp_size=1)
