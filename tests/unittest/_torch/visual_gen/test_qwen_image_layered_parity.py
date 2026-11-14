# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Per-module parity tests for Qwen-Image-Layered."""

import pytest
import torch


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for BF16 parity tests.",
)


@requires_cuda
def test_qwen_layered_block_singleton_attention_parity() -> None:
    """L2/T1: Layered block image/text outputs match the ordinary unmasked Qwen block.

    The same weights and inputs isolate the Layered attention substitution. A
    singleton must preserve the attention call contract without batch metadata.
    """
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.models.qwen_image.transformer_qwen_image import (
        QwenImageTransformerBlock,
    )
    from tensorrt_llm._torch.visual_gen.models.qwen_image_layered.transformer_qwen_image_layered import (
        QwenImageLayeredJointAttention,
    )

    torch.manual_seed(0)
    config = DiffusionModelConfig(attention={"backend": "VANILLA"})
    kwargs = dict(dim=32, num_attention_heads=2, attention_head_dim=16, config=config)
    block = QwenImageTransformerBlock(**kwargs, dtype=torch.bfloat16).cuda().eval()
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.normal_(std=0.1)
    image = torch.randn(1, 8, 32, device="cuda", dtype=torch.bfloat16)
    text = torch.randn(1, 3, 32, device="cuda", dtype=torch.bfloat16)
    temb = torch.randn(1, 32, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = block(image, text, temb)
    attention = QwenImageLayeredJointAttention(**kwargs).cuda().eval()
    attention.load_state_dict(block.attn.state_dict(), strict=True)
    block.attn = attention
    assert isinstance(block.attn, QwenImageLayeredJointAttention)
    with torch.inference_mode():
        actual = block(image, text, temb)
    for stream, output, reference in zip(("text", "image"), actual, expected):
        delta = (output.float() - reference.float()).flatten()
        reference_norm = reference.float().norm().item()
        assert reference_norm > 0, f"{stream}: zero-norm reference"
        relative_l2 = delta.norm().item() / reference_norm
        cosine = _cosine(output, reference)
        max_abs = delta.abs().max().item()
        metrics = f"{stream}: relative_l2={relative_l2}, cosine={cosine}, max_abs={max_abs}"
        print(metrics)
        assert relative_l2 <= 1e-2 and cosine >= 0.9999, metrics
        torch.testing.assert_close(output, reference, rtol=1e-3, atol=1e-3, msg=metrics)


@requires_cuda
def test_qwen_layered_timestep_proj_embedding_additional_cond_parity():
    transformer_qwenimage = pytest.importorskip(
        "diffusers.models.transformers.transformer_qwenimage"
    )
    RefTimestep = getattr(transformer_qwenimage, "QwenTimestepProjEmbeddings", None)
    if RefTimestep is None:
        pytest.skip("diffusers QwenTimestepProjEmbeddings is unavailable")

    from tensorrt_llm._torch.visual_gen.models.qwen_image import (
        QwenTimestepProjEmbeddings as OurTimestep,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    try:
        ref = RefTimestep(embedding_dim=32, use_additional_t_cond=True).to(dtype).to(device).eval()
    except TypeError:
        pytest.skip("diffusers QwenTimestepProjEmbeddings lacks use_additional_t_cond")

    our = OurTimestep(embedding_dim=32, use_additional_t_cond=True).to(dtype).to(device).eval()
    our.load_state_dict(ref.state_dict(), strict=True)

    timestep = torch.tensor([0.25, 0.75], dtype=dtype, device=device)
    hidden_states = torch.zeros(2, 1, 32, dtype=dtype, device=device)
    additional_t_cond = torch.tensor([0, 1], dtype=torch.long, device=device)

    with torch.inference_mode():
        ref_out = ref(timestep, hidden_states, additional_t_cond)
        our_out = our(timestep, hidden_states, additional_t_cond)

    sim = _cosine(ref_out, our_out)
    assert sim > 0.999, f"cosine={sim}"


@requires_cuda
@pytest.mark.parametrize(
    "layer_fhws",
    [
        [(1, 4, 4), (1, 4, 4), (1, 4, 4)],
        [(1, 4, 6), (1, 2, 6), (1, 4, 6)],
    ],
)
@pytest.mark.parametrize("max_txt_seq_len", [16, 64])
def test_qwen_embed_layer3d_rope_parity(layer_fhws, max_txt_seq_len):
    transformer_qwenimage = pytest.importorskip(
        "diffusers.models.transformers.transformer_qwenimage"
    )
    RefLayerRope = getattr(transformer_qwenimage, "QwenEmbedLayer3DRope", None)
    if RefLayerRope is None:
        pytest.skip("diffusers QwenEmbedLayer3DRope is unavailable")

    from tensorrt_llm._torch.visual_gen.models.qwen_image_layered.transformer_qwen_image_layered import (
        QwenEmbedLayer3DRope,
    )

    device = torch.device("cuda")
    ref = RefLayerRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True).to(device)
    our = QwenEmbedLayer3DRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True).to(device)

    ref_vid, ref_txt = ref([layer_fhws], max_txt_seq_len=max_txt_seq_len, device=device)
    our_vid, our_txt = our([layer_fhws], max_txt_seq_len=max_txt_seq_len, device=device)

    assert torch.equal(ref_vid, our_vid), "layered video freqs differ"
    assert torch.equal(ref_txt, our_txt), "layered text freqs differ"
