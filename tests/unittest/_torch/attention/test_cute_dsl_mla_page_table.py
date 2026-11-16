# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise partial MLA tiles with page tables that have no tile padding.

Run under compute-sanitizer memcheck with PYTORCH_NO_CUDA_MEMORY_CACHING=1
so reads past a tensor allocation are reported even if masked by attention.
"""

import math

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not IS_CUTLASS_DSL_AVAILABLE
    or get_sm_version() not in (100, 103),
    reason="CuTe DSL MLA decode requires SM100/SM103",
)


@pytest.mark.parametrize("page_size", [32, 64])
@pytest.mark.parametrize("is_persistent", [False, True])
@pytest.mark.parametrize("split_kv", [1, 2])
def test_cute_dsl_mla_partial_page_table(
    page_size: int, is_persistent: bool, split_kv: int
) -> None:
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    torch.manual_seed(17)
    batch_size, num_heads, latent_dim, rope_dim = 64, 128, 512, 64
    device = torch.device("cuda")
    # Uniform attention keeps the unnormalized probabilities exact in FP8,
    # isolating page selection and masking from softmax quantization error.
    q_storage = torch.zeros(
        batch_size, 1, num_heads, latent_dim + rope_dim, dtype=torch.float8_e4m3fn, device=device
    )
    q_latent = q_storage[..., :latent_dim].permute(2, 3, 1, 0)
    q_rope = q_storage[..., latent_dim:].permute(2, 3, 1, 0)
    output_storage = torch.empty(
        batch_size, 1, num_heads, latent_dim, dtype=torch.bfloat16, device=device
    )
    output = output_storage.permute(2, 3, 1, 0)
    runner = CuteDSLNVMlaDecodeBlackwellRunner(
        in_dtype=cutlass.Float8E4M3FN,
        num_heads=num_heads,
        seq_len_q=1,
        page_size=page_size,
        max_batch_size=batch_size,
    )
    workspace = torch.empty(
        runner.get_max_padded_workspace_size(num_heads, 1, latent_dim, batch_size, cutlass.Float32),
        dtype=torch.uint8,
        device=device,
    )
    scale = 1.0 / math.sqrt(latent_dim + rope_dim)
    # FP8 runners read both scales from device tensors at inputs[10:12].
    scale_inputs = [
        None,  # kv_bounds
        torch.tensor([scale], dtype=torch.float32, device=device),
        torch.tensor([1.0], dtype=torch.float32, device=device),
    ]

    # Reuse the compiled kernel across padded and unpadded table widths.
    # The interleaved K/V slots match KVCacheManagerV1's page-table strides.
    for num_blocks in (4, 1, 2, 3, 5):
        kv_len = num_blocks * page_size - 1
        num_pages = batch_size * num_blocks
        cache = (
            torch.randn(
                num_pages, page_size, latent_dim + rope_dim, dtype=torch.bfloat16, device=device
            )
            * 0.1
        ).to(torch.float8_e4m3fn)
        table_storage = torch.zeros(batch_size, 2, num_blocks, dtype=torch.int32, device=device)
        table_storage[:, 0, :] = torch.arange(num_pages, dtype=torch.int32, device=device).view(
            batch_size, num_blocks
        )
        page_table = table_storage[:, 0, :].transpose(0, 1)
        cache_seqs = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
        runner.forward(
            [
                q_latent,
                q_rope,
                cache[..., :latent_dim].permute(1, 2, 0),
                cache[..., latent_dim:].permute(1, 2, 0),
                page_table,
                cache_seqs,
                output,
                workspace,
                None,
            ]
            + scale_inputs,
            tactic=((128, 128), (128, 256), split_kv, is_persistent),
        )
        torch.cuda.synchronize()

        values = cache.float().view(batch_size, num_blocks * page_size, -1)[:, :kv_len, :latent_dim]
        expected = values.mean(dim=1, keepdim=True).expand(-1, num_heads, -1)
        torch.testing.assert_close(output_storage[:, 0].float(), expected, atol=1e-4, rtol=5e-3)
