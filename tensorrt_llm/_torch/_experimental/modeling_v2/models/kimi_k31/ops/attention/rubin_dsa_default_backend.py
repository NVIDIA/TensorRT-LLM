# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA attention backend classes of the default Kimi K3.1 Rubin target: carriers of its metadata.

The default target returns one from ``attention_backend_for``. As for the specialized targets, the
engine reads only its ``Metadata`` and checks that it is a TRTLLM backend; the target runs its MLA
layers in ``modeling.py`` and never builds a backend instance.
"""

from __future__ import annotations

from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import DSATrtllmAttention

from .rubin_dsa_backend import rubin_dsa_backend_cls
from .rubin_dsa_default_fp4 import INDEX_K_DTYPES
from .rubin_dsa_default_metadata import RubinDsaDefaultHelixMetadata, RubinDsaDefaultMetadata

#: The latent KV cache dtypes the default target serves.
KV_CACHE_DTYPES = ("fp8", "nvfp4")


class RubinDsaDefaultBackend(DSATrtllmAttention):
    """dep16 and tep8 at an NVFP4 KV cache or FP4 index keys."""

    Metadata = RubinDsaDefaultMetadata


class RubinDsaDefaultHelixBackend(DSATrtllmAttention):
    """helix8 at an NVFP4 KV cache or FP4 index keys."""

    Metadata = RubinDsaDefaultHelixMetadata


def default_dsa_backend_cls(kv: str, ik: str, helix: bool) -> type:
    """The backend class of a latent KV dtype ``kv``, an index-key dtype ``ik`` and a layout
    (``helix``: helix8, else dep16 / tep8).

    FP8 / FP8 is the specialized recipe: the specialized carrier of the layout, so the default
    target builds the specialized metadata class. Any other pair: the default carrier. Raises on
    a dtype the default target does not serve.
    """
    if kv not in KV_CACHE_DTYPES or ik not in INDEX_K_DTYPES:
        raise ValueError(
            f"the default Kimi K3.1 Rubin target runs a {KV_CACHE_DTYPES} KV cache and "
            f"{INDEX_K_DTYPES} index keys; got {kv!r} and {ik!r}"
        )
    helix = bool(helix)
    if (kv, ik) == ("fp8", "fp8"):
        return rubin_dsa_backend_cls(kv_lens_refresh_once=helix, helix_fused_meta=helix)
    return RubinDsaDefaultHelixBackend if helix else RubinDsaDefaultBackend


__all__ = [
    "KV_CACHE_DTYPES",
    "RubinDsaDefaultBackend",
    "RubinDsaDefaultHelixBackend",
    "default_dsa_backend_cls",
]
