# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA attention backend classes of the Kimi K3.1 Rubin targets: carriers of their metadata.

A target declares one (``RuntimeHooks.attention_backend_cls``). The engine checks that it subclasses
the TRTLLM DSA backend it resolves and reads only its ``Metadata`` (the class it builds the step's
attention metadata from). The targets run their MLA layers in ``modeling.py`` and never build a
backend instance.
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
