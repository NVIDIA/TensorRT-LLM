# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""k3_attn_res FP8 output in DeepGEMM's packed scale layout (TRTLLM_K3_ATTN_RES_FP8_DG=1).

The codes and scales against ``fp8_quantize_1x128_packed_ue8m0`` of the kernel's own BF16 output,
bitwise, for both kernel variants (exact / vectorized), with and without an addend, and the padding
scale rows (``M <= r < ceil4(M)``) zero. The variable is read at import, so it is set first."""

import os

os.environ["TRTLLM_K3_ATTN_RES_FP8_DG"] = "1"

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100/SM103 GPU",
)

H = 7168


@pytest.mark.parametrize("m", [1, 2, 3, 6, 8, 12, 16, 24, 32, 33, 48, 64, 65, 96])
def test_attn_res_fp8_deepgemm_bitwise(m):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res as ar

    assert ar.SF_DEEPGEMM
    g = torch.Generator(device="cuda")
    g.manual_seed(100 + m)
    n = 5
    w = tuple(
        (torch.randn(H, generator=g, device="cuda") * sc + b).bfloat16()
        for sc, b in ((0.05, 0.0), (0.1, 1.0), (0.1, 1.0))
    )
    for exact in (True, False):
        for add in (False, True):
            layer = (torch.randn(m, H, generator=g, device="cuda") * 2).bfloat16()
            addend = (torch.randn(m, H, generator=g, device="cuda") * 2).bfloat16()
            block = (torch.randn(n - 1, m, H, generator=g, device="cuda") * 2).bfloat16()
            out = torch.empty(m, H, dtype=torch.bfloat16, device="cuda")
            upd = torch.empty_like(out) if add else None
            fp8 = torch.empty(m, H, dtype=torch.float8_e4m3fn, device="cuda")
            m_pad4 = (m + 3) // 4 * 4
            scale = torch.full((ar.fp8_scale_numel(m),), 0xFF, dtype=torch.uint8, device="cuda")
            assert scale.numel() == m_pad4 * (H // 128)
            ar.run_k3_attn_res(
                layer,
                (addend,) if add else (),
                block,
                *w,
                1e-6,
                1e-6,
                out,
                upd,
                exact=exact,
                fp8_output=fp8,
                fp8_scale=scale,
            )
            ref_codes, ref_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(out, False)
            assert (fp8.view(torch.uint8) == ref_codes.view(torch.uint8)).all()
            words = scale.view(torch.int32)
            got = words.as_strided((m, H // 512), (1, m_pad4))
            assert got.shape == ref_scale.shape and got.stride() == ref_scale.stride()
            assert (got.contiguous() == ref_scale.contiguous()).all()
            # the padding rows (m <= r < ceil4(m)) are zero
            full = words.view(H // 512, m_pad4)
            if m_pad4 > m:
                assert (full[:, m:] == 0).all()
