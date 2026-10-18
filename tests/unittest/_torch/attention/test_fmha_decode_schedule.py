# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.attention.backends.prims_ts.kernels.fmha_decode.fmha_decode_kernel import (
    _build_decode_gen_schedule,
)


@pytest.mark.parametrize(
    "cfg,kwargs,error",
    [
        pytest.param(
            SimpleNamespace(use_block_sparse=True, num_insts_kv=1),
            {},
            "block-sparse attention requires num_insts_kv == 2",
            id="block-sparse-needs-two-instances",
        ),
        pytest.param(
            SimpleNamespace(
                use_block_sparse=False,
                num_insts_kv=1,
                use_keeps_mma_ab=True,
                uses_tmem_p=False,
            ),
            {},
            "one-instance KeepsMmaAb is enabled only for the staged headDim=256 profile",
            id="unsupported-one-instance-keeps-mma-ab",
        ),
        pytest.param(
            SimpleNamespace(
                use_block_sparse=False,
                num_insts_kv=2,
                use_keeps_mma_ab=False,
                use_paged_kv=False,
                uses_scattered_page_route=False,
            ),
            {"use_native_paged_kv": True},
            "native paged-KV ABI requires cfg.use_paged_kv=True",
            id="native-paged-kv-needs-paged-config",
        ),
        pytest.param(
            SimpleNamespace(
                use_block_sparse=False,
                num_insts_kv=2,
                use_keeps_mma_ab=False,
                use_paged_kv=True,
                uses_scattered_page_route=False,
            ),
            {"use_native_paged_kv": True},
            "native paged-KV ABI requires block tables, capacity, and row stride",
            id="native-paged-kv-needs-table-metadata",
        ),
        pytest.param(
            SimpleNamespace(
                use_block_sparse=False,
                num_insts_kv=2,
                use_keeps_mma_ab=False,
                uses_scattered_page_route=True,
            ),
            {},
            "scattered page routes require the native paged-KV ABI",
            id="scattered-routes-require-native-paged-kv",
        ),
    ],
)
def test_build_decode_gen_schedule_rejects_invalid_configurations(
    cfg: SimpleNamespace, kwargs: dict, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        _build_decode_gen_schedule(cfg, total_kv_tiles=1, **kwargs)
