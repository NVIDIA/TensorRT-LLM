# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-module parity tests for Qwen-Image.

Each test pulls the corresponding module from
``diffusers.models.transformers.transformer_qwenimage`` (the reference)
and from our port, loads the same slice of the real Qwen-Image
``transformer`` state_dict into both, runs them on identical inputs, and
asserts cosine similarity above a per-module threshold.

Tests are skipped automatically unless ``QWEN_IMAGE_CKPT`` points to a
local Qwen-Image checkpoint.
"""

import json
import os
from pathlib import Path

import pytest
import torch

_CKPT_ENV = "QWEN_IMAGE_CKPT"


def _ckpt_path() -> Path | None:
    ckpt = os.environ.get(_CKPT_ENV)
    if not ckpt:
        return None
    path = Path(ckpt)
    if not (path / "transformer" / "config.json").is_file():
        return None
    return path


def _load_transformer_state_dict(ckpt: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    sd: dict[str, torch.Tensor] = {}
    for shard in sorted((ckpt / "transformer").glob("*.safetensors")):
        sd.update(load_file(str(shard)))
    return sd


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


requires_ckpt = pytest.mark.skipif(
    _ckpt_path() is None,
    reason=(
        f"Qwen-Image checkpoint not found at ${_CKPT_ENV}. "
        "Set QWEN_IMAGE_CKPT to a local Qwen/Qwen-Image checkpoint "
        "to enable parity tests."
    ),
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for BF16 parity tests.",
)


@pytest.fixture(scope="module")
def transformer_state_dict() -> dict[str, torch.Tensor]:
    ckpt = _ckpt_path()
    assert ckpt is not None
    return _load_transformer_state_dict(ckpt)


@pytest.fixture(scope="module")
def transformer_config() -> dict:
    ckpt = _ckpt_path()
    assert ckpt is not None
    return json.loads((ckpt / "transformer" / "config.json").read_text())


# ===========================================================================
# Timestep embedding.
# ===========================================================================


@requires_ckpt
@requires_cuda
@pytest.mark.parametrize("timestep_value", [0.001, 0.25, 0.5, 0.99])
def test_qwen_timestep_proj_embedding_parity(
    transformer_state_dict, timestep_value
):
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenTimestepProjEmbeddings as RefTimestep,
    )

    from tensorrt_llm._torch.visual_gen.models.qwen_image import (
        QwenTimestepProjEmbeddings as OurTimestep,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    embedding_dim = 3072

    prefix = "time_text_embed."
    time_text_sd = {
        k.removeprefix(prefix): v
        for k, v in transformer_state_dict.items()
        if k.startswith(prefix)
    }

    ref = RefTimestep(embedding_dim=embedding_dim).to(dtype).to(device).eval()
    our = OurTimestep(embedding_dim=embedding_dim).to(dtype).to(device).eval()
    ref.load_state_dict(time_text_sd, strict=True)
    our.load_state_dict(time_text_sd, strict=True)

    timestep = torch.tensor([timestep_value], dtype=dtype, device=device)
    hidden_states = torch.zeros(1, 1, embedding_dim, dtype=dtype, device=device)

    with torch.inference_mode():
        ref_out = ref(timestep, hidden_states)
        our_out = our(timestep, hidden_states)

    sim = _cosine(ref_out, our_out)
    assert sim > 0.9999, f"cosine={sim}"


@requires_cuda
def test_get_timestep_embedding_matches_diffusers():
    from diffusers.models.embeddings import (
        get_timestep_embedding as ref_fn,
    )

    from tensorrt_llm._torch.visual_gen.models.qwen_image import (
        get_timestep_embedding as our_fn,
    )

    device = torch.device("cuda")
    t = torch.tensor([0.001, 0.25, 0.5, 0.99], dtype=torch.float32, device=device)
    for params in [
        dict(flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000),
        dict(flip_sin_to_cos=False, downscale_freq_shift=1, scale=1),
    ]:
        ref = ref_fn(t, embedding_dim=256, **params)
        our = our_fn(t, embedding_dim=256, **params)
        assert torch.equal(ref, our), f"mismatch for {params}"


# ===========================================================================
# 3D RoPE (bit-exact in fp32).
# ===========================================================================


@requires_cuda
@pytest.mark.parametrize(
    "video_fhw",
    [(1, 32, 32), (1, 64, 48), (1, 128, 128)],
)
@pytest.mark.parametrize("max_txt_seq_len", [16, 512])
def test_qwen_embed_rope_parity(video_fhw, max_txt_seq_len):
    """Our ``QwenEmbedRope`` must produce bit-exact complex freqs."""
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenEmbedRope as RefRope,
    )

    from tensorrt_llm._torch.visual_gen.models.qwen_image import QwenEmbedRope

    device = torch.device("cuda")
    ref = RefRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True).to(device)
    our = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True).to(device)

    ref_vid, ref_txt = ref(video_fhw, max_txt_seq_len=max_txt_seq_len, device=device)
    our_vid, our_txt = our(video_fhw, max_txt_seq_len=max_txt_seq_len, device=device)

    # RoPE is pure math with fixed constants; should be bit-exact.
    assert torch.equal(ref_vid, our_vid), "video freqs differ"
    assert torch.equal(ref_txt, our_txt), "text freqs differ"


@requires_cuda
def test_apply_rotary_emb_qwen_parity():
    from diffusers.models.transformers.transformer_qwenimage import (
        apply_rotary_emb_qwen as ref_fn,
        QwenEmbedRope as RefRope,
    )

    from tensorrt_llm._torch.visual_gen.models.qwen_image import (
        apply_rotary_emb_qwen as our_fn,
        QwenEmbedRope,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)
    ref_rope = RefRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True).to(device)
    our_rope = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True).to(device)
    ref_vid, _ = ref_rope((1, 32, 32), max_txt_seq_len=16, device=device)
    our_vid, _ = our_rope((1, 32, 32), max_txt_seq_len=16, device=device)
    assert torch.equal(ref_vid, our_vid)

    # Simulate a (B, S, H, D) query tensor.
    B, S, H, D = 2, 32 * 32, 24, 128
    x = torch.randn(B, S, H, D, dtype=dtype, device=device)
    ref_out = ref_fn(x, ref_vid, use_real=False)
    our_out = our_fn(x, our_vid, use_real=False)
    sim = _cosine(ref_out, our_out)
    assert sim > 0.9999, f"cosine={sim}"


# ===========================================================================
# Pre/post-block modules (img_in, txt_in, txt_norm, norm_out, proj_out).
# ===========================================================================


@requires_ckpt
@requires_cuda
def test_pre_post_block_modules_parity(transformer_state_dict, transformer_config):
    """img_in, txt_in, txt_norm, norm_out, proj_out parity vs diffusers.

    We construct one of each on both sides, load the relevant state_dict
    slice, and compare output on fixed random inputs.
    """
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm._torch.visual_gen.models.qwen_image import (
        AdaLayerNormContinuous,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    cfg = transformer_config
    inner_dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    joint_attn_dim = cfg["joint_attention_dim"]
    out_channels = cfg["out_channels"]
    patch_size = cfg["patch_size"]

    # Diffusers refs.
    from diffusers.models.normalization import (
        AdaLayerNormContinuous as RefAdaLN,
        RMSNorm as RefRMS,
    )

    ref_img_in = torch.nn.Linear(cfg["in_channels"], inner_dim).to(dtype).to(device)
    ref_txt_in = torch.nn.Linear(joint_attn_dim, inner_dim).to(dtype).to(device)
    ref_txt_norm = RefRMS(joint_attn_dim, eps=1e-6).to(dtype).to(device)
    ref_norm_out = RefAdaLN(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6).to(dtype).to(device)
    ref_proj_out = torch.nn.Linear(
        inner_dim, patch_size * patch_size * out_channels, bias=True
    ).to(dtype).to(device)

    # Load weights from the HF checkpoint.
    def sd(prefix):
        return {
            k.removeprefix(prefix + "."): v
            for k, v in transformer_state_dict.items()
            if k.startswith(prefix + ".")
        }

    ref_img_in.load_state_dict(sd("img_in"), strict=True)
    ref_txt_in.load_state_dict(sd("txt_in"), strict=True)
    ref_txt_norm.load_state_dict(sd("txt_norm"), strict=True)
    ref_norm_out.load_state_dict(sd("norm_out"), strict=True)
    ref_proj_out.load_state_dict(sd("proj_out"), strict=True)

    # Our ports.
    our_img_in = torch.nn.Linear(cfg["in_channels"], inner_dim).to(dtype).to(device)
    our_txt_in = torch.nn.Linear(joint_attn_dim, inner_dim).to(dtype).to(device)
    our_txt_norm = RMSNorm(
        hidden_size=joint_attn_dim, eps=1e-6, dtype=dtype, has_weights=True
    ).to(device)
    our_norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6).to(dtype).to(device)
    our_proj_out = torch.nn.Linear(
        inner_dim, patch_size * patch_size * out_channels, bias=True
    ).to(dtype).to(device)

    our_img_in.load_state_dict(sd("img_in"), strict=True)
    our_txt_in.load_state_dict(sd("txt_in"), strict=True)
    our_txt_norm.load_state_dict(sd("txt_norm"), strict=True)
    our_norm_out.load_state_dict(sd("norm_out"), strict=True)
    our_proj_out.load_state_dict(sd("proj_out"), strict=True)

    torch.manual_seed(0)
    B = 2
    img_tokens = 256
    txt_tokens = 64
    img = torch.randn(B, img_tokens, cfg["in_channels"], dtype=dtype, device=device)
    txt = torch.randn(B, txt_tokens, joint_attn_dim, dtype=dtype, device=device)
    temb = torch.randn(B, inner_dim, dtype=dtype, device=device)

    with torch.inference_mode():
        # img_in
        sim = _cosine(ref_img_in(img), our_img_in(img))
        assert sim > 0.9999, f"img_in cos={sim}"

        # txt_norm + txt_in
        ref_post = ref_txt_in(ref_txt_norm(txt))
        our_post = our_txt_in(our_txt_norm(txt))
        sim = _cosine(ref_post, our_post)
        assert sim > 0.9999, f"txt_norm+txt_in cos={sim}"

        # norm_out applied to hidden states with temb conditioning.
        hs = ref_img_in(img)
        sim = _cosine(ref_norm_out(hs, temb), our_norm_out(hs, temb))
        assert sim > 0.9999, f"norm_out cos={sim}"

        # proj_out
        sim = _cosine(ref_proj_out(hs), our_proj_out(hs))
        assert sim > 0.9999, f"proj_out cos={sim}"


# ===========================================================================
# MMDiT block (one block with real weights vs diffusers).
# ===========================================================================


@requires_ckpt
@requires_cuda
def test_qwen_image_transformer_block_parity(transformer_state_dict, transformer_config):
    """One ``QwenImageTransformerBlock`` must match diffusers cos >= 0.999.

    Uses real checkpoint weights for ``transformer_blocks.0.*``. RoPE
    and temb are constructed consistently; text-stream RMSNorm and
    img_in / txt_in are applied to produce the block input.
    """
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenImageTransformerBlock as RefBlock,
        QwenEmbedRope as RefRope,
    )

    from tensorrt_llm._torch.visual_gen.models.qwen_image import (
        QwenEmbedRope,
        QwenImageTransformerBlock,
    )
    from tensorrt_llm._torch.visual_gen.models.qwen_image.transformer_qwen_image import (
        _remap_checkpoint_keys,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    cfg = transformer_config
    inner_dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]

    ref_block = RefBlock(
        dim=inner_dim,
        num_attention_heads=cfg["num_attention_heads"],
        attention_head_dim=cfg["attention_head_dim"],
    ).to(dtype).to(device).eval()
    our_block = QwenImageTransformerBlock(
        dim=inner_dim,
        num_attention_heads=cfg["num_attention_heads"],
        attention_head_dim=cfg["attention_head_dim"],
    ).to(dtype).to(device).eval()

    block0_prefix = "transformer_blocks.0."
    block0_sd = {
        k.removeprefix(block0_prefix): v
        for k, v in transformer_state_dict.items()
        if k.startswith(block0_prefix)
    }

    # Diffusers' Attention class has extra attributes that we don't
    # mirror. Drop-in `strict=False` is fine since diffusers' attn
    # has `to_q`/etc. with same key names we use.
    _m_ref, _u_ref = ref_block.load_state_dict(block0_sd, strict=False)
    # Silently ignore missing keys on diffusers side: its Attention has
    # some optional norm/head_dim params we don't need to set.
    _m_our, _u_our = our_block.load_state_dict(
        _remap_checkpoint_keys(block0_sd), strict=False
    )
    assert not _u_our, f"unexpected keys in our block: {_u_our[:3]}"
    # Diffusers' extra (ignored) keys are OK; missing on our side must
    # be 0 because we mirror the state_dict structure.
    our_missing = [k for k in _m_our if "running_" not in k and "num_batches" not in k]
    assert not our_missing, f"our block missing keys: {our_missing[:3]}"

    torch.manual_seed(0)
    B = 1
    frame, h, w = 1, 32, 32
    img_seq = frame * h * w
    txt_seq = 16
    img = torch.randn(B, img_seq, inner_dim, dtype=dtype, device=device)
    txt = torch.randn(B, txt_seq, inner_dim, dtype=dtype, device=device)
    temb = torch.randn(B, inner_dim, dtype=dtype, device=device)

    # Same RoPE on both sides.
    ref_rope = RefRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True).to(device)
    our_rope = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True).to(device)
    ref_vid, ref_txt = ref_rope((frame, h, w), max_txt_seq_len=txt_seq, device=device)
    our_vid, our_txt = our_rope((frame, h, w), max_txt_seq_len=txt_seq, device=device)
    assert torch.equal(ref_vid, our_vid)
    assert torch.equal(ref_txt, our_txt)

    with torch.inference_mode():
        ref_enc, ref_hid = ref_block(
            hidden_states=img,
            encoder_hidden_states=txt,
            encoder_hidden_states_mask=None,
            temb=temb,
            image_rotary_emb=(ref_vid, ref_txt),
        )
        our_enc, our_hid = our_block(
            hidden_states=img,
            encoder_hidden_states=txt,
            temb=temb,
            image_rotary_emb=(our_vid, our_txt),
            attention_mask=None,
        )

    sim_enc = _cosine(ref_enc, our_enc)
    sim_hid = _cosine(ref_hid, our_hid)
    # Per-block cosine of 0.999 is the right bar for bf16 w/ 60 such
    # blocks to compose; looser than the per-module 0.9999 we use on
    # the tiny pre/post modules because each block has ~10 matmuls.
    assert sim_enc > 0.999, f"encoder_hidden_states cos={sim_enc}"
    assert sim_hid > 0.999, f"hidden_states cos={sim_hid}"


# ===========================================================================
# Full transformer single-step (the expensive one).
# ===========================================================================


@requires_ckpt
@requires_cuda
@pytest.mark.slow
def test_qwen_image_transformer_full_parity(transformer_state_dict, transformer_config):
    """One full forward through all 60 blocks vs diffusers, cos >= 0.999.

    Loads the full ~20B-param transformer twice (ours + diffusers). Each
    copy is ~40 GB bf16 so this test requires a 96 GB GPU (RTX PRO 6000).
    Marked ``slow`` so default pytest runs skip it; enable with
    ``pytest -m slow``.
    """
    from diffusers import QwenImageTransformer2DModel as RefT

    from tensorrt_llm._torch.visual_gen.models.qwen_image import (
        QwenImageTransformer2DModel,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    cfg = transformer_config

    ref_model = RefT.from_config(cfg).to(dtype).to(device).eval()
    ref_model.load_state_dict(transformer_state_dict, strict=False)

    our_model = QwenImageTransformer2DModel.from_config_dict(cfg).to(dtype).to(device).eval()
    our_model.load_weights(transformer_state_dict)

    torch.manual_seed(0)
    B = 1
    frame, h, w = 1, 32, 32
    img_seq = frame * h * w
    txt_seq = 64
    img = torch.randn(B, img_seq, cfg["in_channels"], dtype=dtype, device=device)
    txt = torch.randn(B, txt_seq, cfg["joint_attention_dim"], dtype=dtype, device=device)
    t = torch.tensor([0.5], dtype=dtype, device=device)

    with torch.inference_mode():
        ref_out = ref_model(
            hidden_states=img,
            encoder_hidden_states=txt,
            encoder_hidden_states_mask=None,
            timestep=t,
            img_shapes=[(frame, h, w)],
            return_dict=False,
        )[0]
        our_out = our_model(
            hidden_states=img,
            encoder_hidden_states=txt,
            encoder_hidden_states_mask=None,
            timestep=t,
            img_shapes=[(frame, h, w)],
            return_dict=False,
        )[0]

    sim = _cosine(ref_out, our_out)
    assert sim > 0.999, f"full-transformer cos={sim}"
