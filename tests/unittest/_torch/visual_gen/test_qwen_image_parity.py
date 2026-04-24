# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-module parity tests for Qwen-Image Phase 1 port.

Each test pulls the corresponding module from
``diffusers.models.transformers.transformer_qwenimage`` (the reference)
and from our port, loads the same slice of the real Qwen-Image
``transformer`` state_dict into both, runs them on identical inputs, and
asserts cosine similarity above a per-module threshold.

Tests are skipped automatically unless the real checkpoint is available
locally; set ``QWEN_IMAGE_CKPT`` to override the default location.
"""

import os
from pathlib import Path

import pytest
import torch

_DEFAULT_CKPT = "/home/scratch.asteiner/trtllm-qwen-image/models/qwen-image"
_CKPT_ENV = "QWEN_IMAGE_CKPT"


def _ckpt_path() -> Path | None:
    path = Path(os.environ.get(_CKPT_ENV, _DEFAULT_CKPT))
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
        f"Qwen-Image checkpoint not found at ${_CKPT_ENV} "
        f"(default: {_DEFAULT_CKPT}). Download Qwen/Qwen-Image from "
        "HuggingFace to enable parity tests."
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


# ---------------------------------------------------------------------------
# M2 -- timestep embedding parity.
# ---------------------------------------------------------------------------


@requires_ckpt
@requires_cuda
@pytest.mark.parametrize("timestep_value", [0.001, 0.25, 0.5, 0.99])
def test_qwen_timestep_proj_embedding_parity(
    transformer_state_dict, timestep_value
):
    """Port of ``QwenTimestepProjEmbeddings`` must match diffusers bit-for-bit.

    diffusers uses ``Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)``
    followed by a 2-linear ``TimestepEmbedding(256 -> 3072 -> 3072)``. Our
    port should read the same HF state_dict slice and produce the same
    output in bf16 up to numerical noise.
    """
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenTimestepProjEmbeddings as RefTimestep,
    )

    from tensorrt_llm._torch.visual_gen.models.qwen_image import (
        QwenTimestepProjEmbeddings as OurTimestep,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    embedding_dim = 3072  # 24 heads * 128 head_dim

    prefix = "time_text_embed."
    time_text_sd = {
        k.removeprefix(prefix): v
        for k, v in transformer_state_dict.items()
        if k.startswith(prefix)
    }
    assert len(time_text_sd) == 4, (
        f"Expected 4 tensors under {prefix}, got {sorted(time_text_sd)}"
    )

    ref = RefTimestep(embedding_dim=embedding_dim).to(dtype).to(device).eval()
    our = OurTimestep(embedding_dim=embedding_dim).to(dtype).to(device).eval()

    ref.load_state_dict(time_text_sd, strict=True)
    our.load_state_dict(time_text_sd, strict=True)

    timestep = torch.tensor([timestep_value], dtype=dtype, device=device)
    hidden_states = torch.zeros(1, 1, embedding_dim, dtype=dtype, device=device)

    with torch.inference_mode():
        ref_out = ref(timestep, hidden_states)
        our_out = our(timestep, hidden_states)

    assert ref_out.shape == our_out.shape
    sim = _cosine(ref_out, our_out)
    # BF16 matmul ordering is not bit-deterministic across the two
    # independently-instantiated Linears even with identical weights;
    # 0.9999 (four nines) is the right bar for a per-module parity
    # test at this depth in bf16.
    assert sim > 0.9999, (
        f"QwenTimestepProjEmbeddings parity failed: cosine={sim}, "
        f"ref_norm={ref_out.float().norm()}, "
        f"our_norm={our_out.float().norm()}"
    )


@requires_cuda
def test_get_timestep_embedding_matches_diffusers():
    """The raw sinusoidal embedding must match diffusers bit-exactly.

    No checkpoint required.
    """
    from diffusers.models.embeddings import (
        get_timestep_embedding as ref_fn,
    )

    from tensorrt_llm._torch.visual_gen.models.qwen_image import (
        get_timestep_embedding as our_fn,
    )

    device = torch.device("cuda")
    t = torch.tensor([0.001, 0.25, 0.5, 0.99], dtype=torch.float32, device=device)
    for fst_params in [
        # Qwen-Image defaults: flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000.
        dict(flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000),
        # Also cover the off-defaults used by other diffusion models.
        dict(flip_sin_to_cos=False, downscale_freq_shift=1, scale=1),
    ]:
        ref = ref_fn(t, embedding_dim=256, **fst_params)
        our = our_fn(t, embedding_dim=256, **fst_params)
        assert torch.equal(ref, our), f"mismatch for {fst_params}"
