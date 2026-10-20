# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.visual_gen.models.minimax_h3.vsa import (
    build_h3_vsa_metadata,
    h3_vsa_attention,
)


def test_h3_vsa_geometry():
    metadata = build_h3_vsa_metadata((7, 5, (5, 5, 7)), 0.9, "cpu")
    valid = metadata.source >= 0
    assert metadata.sizes.sum() == 187
    assert metadata.prefix_tiles == 2
    torch.testing.assert_close(metadata.source[metadata.inverse], torch.arange(187))
    assert torch.unique(metadata.source[valid]).numel() == 187
    # Partial text/audio tiles must not be pooled together.
    assert metadata.sizes[:2].tolist() == [7, 5]


@pytest.mark.parametrize("sparsity", [0.0, 0.5, 0.9])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_h3_vsa_trtllm_masked_reference(sparsity):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("PrimTS block-sparse test requires Blackwell")
    torch.manual_seed(42)
    metadata = build_h3_vsa_metadata((7, 5, (5, 5, 7)), sparsity, "cuda")
    q, k, v = [torch.randn(1, 187, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    gate = torch.randn_like(q) * 0.1
    backend = TrtllmAttention(
        num_heads=2,
        num_kv_heads=2,
        head_dim=128,
        dtype=torch.bfloat16,
        attention_metadata_state={},
    )
    reference = []

    def run(qt, kt, vt, sparse_backend_args):
        seq = qt.shape[1]
        tensors = [x.view(1, seq, 2, 128) for x in (qt, kt, vt)]
        actual = backend.forward(
            *tensors, batch_size=1, seq_len=seq, sparse_backend_args=sparse_backend_args
        )
        routes = sparse_backend_args.block_sparse_inputs
        blocks = torch.zeros(1, 2, seq // 64, seq // 64, dtype=torch.bool, device="cuda")
        for head in range(2):
            for row in range(seq // 64):
                start, end = routes.block_indptr[0, head, row : row + 2].tolist()
                blocks[0, head, row, routes.block_indices[start:end].long()] = True
        prefix = metadata.prefix_tiles
        assert blocks[:, :, :prefix].all()
        assert blocks[..., :prefix].all()
        mask = blocks.repeat_interleave(64, -2).repeat_interleave(64, -1)
        mask &= (metadata.source >= 0)[None, None, None, :]
        expected = torch.nn.functional.scaled_dot_product_attention(
            *[x.transpose(1, 2) for x in tensors],
            attn_mask=mask,
        ).transpose(1, 2)
        actual = actual.view_as(expected)
        valid = metadata.source >= 0
        torch.testing.assert_close(actual[:, valid], expected[:, valid], atol=0.02, rtol=0.02)
        pooled = [
            x.view(1, -1, 64, 2, 128).sum(2, dtype=torch.float32)
            / metadata.sizes[None, :, None, None]
            for x in tensors
        ]
        qp, kp, vp = [x.transpose(1, 2) for x in pooled]
        coarse = (qp @ kp.transpose(-2, -1) / 128**0.5).softmax(-1) @ vp
        coarse = coarse.transpose(1, 2).to(q.dtype)[:, metadata.inverse // 64]
        reference.append(expected[:, metadata.inverse] + coarse * gate)
        return actual.flatten(2)

    output = h3_vsa_attention(q, k, v, gate, metadata, run)
    torch.testing.assert_close(output, reference[0], atol=0.02, rtol=0.02)
