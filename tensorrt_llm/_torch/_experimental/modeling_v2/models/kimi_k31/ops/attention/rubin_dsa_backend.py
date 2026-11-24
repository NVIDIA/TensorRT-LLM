# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA attention backend classes of the Kimi K3.1 Rubin targets: carriers of their metadata.

A target returns one from ``attention_backend_for``. The engine reads only its ``Metadata`` (the
class it builds the step's attention metadata from) and checks that it is a TRTLLM backend. The
targets run their MLA layers in ``modeling.py`` and never build a backend instance.
"""

from __future__ import annotations

from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import DSATrtllmAttention

from .rubin_dsa_metadata import RubinDSAHelixMetadata, RubinDSAMetadata


class RubinDSABackend(DSATrtllmAttention):
    """dep16 and tep8."""

    Metadata = RubinDSAMetadata


class RubinDSAHelixBackend(DSATrtllmAttention):
    """helix8."""

    Metadata = RubinDSAHelixMetadata


def rubin_dsa_backend_cls(*, kv_lens_refresh_once: bool, helix_fused_meta: bool) -> type:
    """The backend class whose metadata has these two settings."""
    for cls in (RubinDSABackend, RubinDSAHelixBackend):
        meta = cls.Metadata
        if (getattr(meta, "kv_lens_refresh_once", False), meta.HELIX) == (
            kv_lens_refresh_once,
            helix_fused_meta,
        ):
            return cls
    raise ValueError(
        "no Kimi K3.1 Rubin DSA backend has "
        f"kv_lens_refresh_once={kv_lens_refresh_once}, helix_fused_meta={helix_fused_meta}"
    )


def attention_backend_for(backend_name: str, sparse_params, backend_cls: type) -> type:
    """``backend_cls`` when the engine asks for the TRTLLM DSA backend; raises otherwise."""
    if backend_name.upper() != "TRTLLM" or getattr(sparse_params, "algorithm", None) != "dsa":
        raise ValueError(
            "the Kimi K3.1 Rubin targets run the TRTLLM DSA attention backend only; got "
            f"{backend_name} with {getattr(sparse_params, 'algorithm', None)!r} sparse parameters"
        )
    return backend_cls
