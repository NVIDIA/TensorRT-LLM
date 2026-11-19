# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA parameter types."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

from tensorrt_llm._torch.attention.backends.sparse.dsa.params import (  # noqa: F401
    DSABackendForwardArgs,
    DSAMetadataParams,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.params import DSAParams as _EngineDSAParams


@dataclass(frozen=True)
class K31DSAParams(_EngineDSAParams):
    """The engine's DSA backend parameters plus what the Kimi K3.1 indexer also needs."""

    # Rotated channels of each indexer head. None means the MLA's
    # qk_rope_head_dim; 0 means a position-free (NoPE) indexer.
    index_rope_dim: Optional[int] = None
    # Epsilon of the indexer key LayerNorm. None means DeepSeek-V3.2's 1e-6.
    index_k_norm_eps: Optional[float] = None


def k31_dsa_params(params, pretrained_config=None):
    """The family's ``K31DSAParams`` for the engine's DSA parameters ``params``; ``params`` itself
    when it is not DSA (None, another algorithm) or already the package's. The indexer's rotated
    channels and key-norm epsilon come from the (text) config's ``dsa_config``."""
    if params is None or isinstance(params, K31DSAParams):
        return params
    if getattr(params, "algorithm", None) != "dsa":
        return params
    values = {f.name: getattr(params, f.name) for f in dataclasses.fields(params) if f.init}
    text = getattr(pretrained_config, "text_config", None) or pretrained_config
    dsa = getattr(text, "dsa_config", None)
    values["index_rope_dim"] = None if dsa is None else dsa.get("index_rope_dim")
    # The indexer key LayerNorm uses the model's norm epsilon.
    values["index_k_norm_eps"] = None if dsa is None else getattr(text, "rms_norm_eps", None)
    return K31DSAParams(**values)
