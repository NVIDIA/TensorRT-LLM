# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The build-time flag check accepts the intended FP8_DG production env on SM100/103 and rejects
FP8_DG with the persistent attn_res kernel (the default-on POST_FP8 switch is inert without it)."""

import os

os.environ["TRTLLM_K3_ATTN_RES_FP8_DG"] = "1"
os.environ["TRTLLM_K3_ATTN_RES_PDL"] = "2"
os.environ["TRTLLM_K3_ATTN_RES_FP8_OUT"] = "1"
os.environ["TRTLLM_K3_ATTN_RES_FP8_OUT_POST"] = "1"
os.environ.pop("TRTLLM_K3_ATTN_RES_PERSISTENT_FUSE", None)

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100/SM103 GPU",
)


def test_fp8_dg_env_accepted_and_persistent_rejected():
    import tensorrt_llm._torch.models.modeling_kimi_linear as m

    m._check_attn_res_fp8_flags()  # must not raise (POST_FP8 defaults on but is inert here)
    old = m._K3_ATTN_RES_PERSISTENT_FUSE
    m._K3_ATTN_RES_PERSISTENT_FUSE = True
    try:
        try:
            m._check_attn_res_fp8_flags()
        except ValueError:
            raised = True
        else:
            raised = False
        assert raised, "FP8_DG with PERSISTENT_FUSE must be rejected"
    finally:
        m._K3_ATTN_RES_PERSISTENT_FUSE = old
