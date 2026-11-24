# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import math
from typing import Dict

import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_utils import DecoderModel
from tensorrt_llm.functional import RopeEmbeddingUtils

from .. import weights as target_weights


def _runtime_position_cap(model_config, pretrained_config, slack: int = 0) -> int:
    """Fallback RoPE table length: the served length, not the advertised one.

    Both drafter families size a position table from the checkpoint's
    max_position_embeddings. K3 advertises 1,048,576: ~256 MiB per rank as
    complex64, ~768 MiB transient via full-size fp32 cos/sin, while the context
    cache is bounded by the runtime max_seq_len. Only the table length moves;
    YaRN's correction range comes from original_max_position_embeddings.

    Shrinking the table this way is also what made its bound matter: sized from
    max_position_embeddings it could not be indexed past, sized from
    model_config.max_seq_len it can. Hence _runtime_position_ceiling, which
    every drafter prefers over this value whenever a worker has published it.
    """
    runtime = getattr(model_config, "max_seq_len", None)
    return int(runtime or getattr(pretrained_config, "max_position_embeddings", 163840)) + slack


def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def build_dspark_mla_yarn_rope(
    *,
    dim: int,
    base: float,
    scaling_factor: float,
    original_max_position_embeddings: int,
    max_position_embeddings: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    mscale: float = 1.0,
    mscale_all_dim: float = 0.0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """YaRN cos/sin tables for an MLA drafter's ``qk_rope_head_dim`` slice.

    Repacks ``RopeEmbeddingUtils.create_sinusoidal_positions_yarn``, the same HF
    DeepSeek-V2 transcription the drafters are distilled under. NOT reused:
    ``RopeParams.create_rope_const_params()`` pins the fused MLA kernel's GPT-J
    packing and hard-codes ``device='cuda'``.

    ``duplicate_data=True`` though only the first half is read as a complex
    pair: the util's two modes disagree by 1 ulp on 6 of 8224 entries, and the
    duplicated one is what every measured drafter run used.

    Returns ``(cos, sin)``, both ``[max_position_embeddings, dim]`` fp32, with
    the frequency half duplicated so ``rotate_half`` applies.
    """
    # The util returns a flat numpy [pos][slot](cos, sin); split the pair.
    _, packed = RopeEmbeddingUtils.create_sinusoidal_positions_yarn(
        num_pos=max_position_embeddings,
        dim=dim,
        base=base,
        scaling_factor=scaling_factor,
        original_max_position_embeddings=original_max_position_embeddings,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        mscale=mscale,
        mscale_all_dim=mscale_all_dim,
        duplicate_data=True,
    )
    table = torch.from_numpy(packed).reshape(max_position_embeddings, dim, 2)
    return (
        table[..., 0].contiguous().to(device),
        table[..., 1].contiguous().to(device),
    )


def build_dspark_mla_yarn_freqs_cis(*, device: str = "cuda", **kwargs) -> torch.Tensor:
    """The same YaRN table as :func:`build_dspark_mla_yarn_rope`, adjacent-pair.

    Returns ``[max_position_embeddings, dim // 2]`` complex64: the convention the
    fused ``cute_dsl_dspark_rmsnorm_rope`` kernel consumes, where the last dim of
    the rotated slice is read as (re, im) pairs.

    HF DeepSeek de-interleaves and rotates halves instead, writing the same
    values to lanes i and i + dim/2 rather than 2i and 2i+1. That permutation
    cancels inside ``q_rope . k_rope``, so either convention works -- but every
    producer of a rope slice must use the SAME one, and mixing them only shows
    up as lower AL. ``mscale`` rides in the modulus, not the phase.
    """
    cos, sin = build_dspark_mla_yarn_rope(device=device, **kwargs)
    half = cos.shape[-1] // 2
    return torch.complex(cos[..., :half], sin[..., :half]).contiguous()


class K3DsparkModel(DecoderModel):
    def __init__(self, model_config, mapping_with_cp=None):
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        self.w = nn.ParameterDict()
        self.layers = target_weights.declare_draft(self.w, cfg, model_config, mapping_with_cp)
        self.collectives = nn.ModuleList(
            op
            for layer in self.layers
            for op in (layer.self_attn.output_all_reduce, layer.mlp_all_reduce)
            if op is not None
        )
        self.embed_tokens = None

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if hasattr(self, "norm"):
            target_weights.bind_draft(self)
        return self


class K3DsparkForCausalLM(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        helix = model_config.mapping if model_config.mapping.has_cp_helix() else None
        config = copy.copy(model_config)
        config._frozen = False
        if helix is not None:
            config.mapping = helix.repurpose_helix_cp_to_tp()
        config._frozen = model_config._frozen
        self.model_config = config
        self.model = K3DsparkModel(config, helix)
        self.lm_head = None

    def load_weights(self, weights, weight_mapper=None, **kwargs):
        target_weights.load_draft(self.model, weights)


def _resolve_dspark_mla_rope_params(pretrained_config, model_config=None, slack: int = 0) -> Dict:
    """RoPE settings for an MLA drafter, from either HF spelling.

    Transformers v5 checkpoints (TorchSpec) carry ``rope_parameters``; older
    ones carry ``rope_scaling``. Only YaRN is accepted: both published MLA
    drafters use it, and silently treating an unknown scaling as plain RoPE
    would cost acceptance without an error.
    """
    scaling = (
        getattr(pretrained_config, "rope_parameters", None)
        or getattr(pretrained_config, "rope_scaling", None)
        or {}
    )
    rope_type = scaling.get("rope_type") or scaling.get("type") or "default"
    if rope_type != "yarn":
        raise ValueError(
            f"MLA DSpark drafter declares rope_type={rope_type!r}; only 'yarn' is "
            "supported (the block decode builds its own YaRN table)."
        )
    theta = scaling.get("rope_theta") or getattr(pretrained_config, "rope_theta", 10000.0)
    return {
        "theta": float(theta),
        "scaling_factor": float(scaling["factor"]),
        "original_max_positions": int(scaling["original_max_position_embeddings"]),
        # Table length, not YaRN math: the correction range below uses
        # original_max_positions, so capping this at the served length is free.
        # K3 advertises 1,048,576, which would build a ~256 MiB complex64 table
        # per rank -- through full-size fp32 cos/sin, so ~768 MiB transient --
        # for a drafter whose point is 5760 B/token instead of 20480.
        "max_positions": _runtime_position_cap(model_config, pretrained_config, slack),
        "beta_fast": float(scaling.get("beta_fast", 32.0)),
        "beta_slow": float(scaling.get("beta_slow", 1.0)),
        "mscale": float(scaling.get("mscale", 1.0)),
        "mscale_all_dim": float(scaling.get("mscale_all_dim", 0.0)),
    }
