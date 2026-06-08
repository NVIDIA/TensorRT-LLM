# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

    from tensorrt_llm._torch.visual_gen.models.qwen_image import QwenEmbedLayer3DRope

    device = torch.device("cuda")
    ref = RefLayerRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True).to(device)
    our = QwenEmbedLayer3DRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True).to(device)

    ref_vid, ref_txt = ref([layer_fhws], max_txt_seq_len=max_txt_seq_len, device=device)
    our_vid, our_txt = our([layer_fhws], max_txt_seq_len=max_txt_seq_len, device=device)

    assert torch.equal(ref_vid, our_vid), "layered video freqs differ"
    assert torch.equal(ref_txt, our_txt), "layered text freqs differ"
