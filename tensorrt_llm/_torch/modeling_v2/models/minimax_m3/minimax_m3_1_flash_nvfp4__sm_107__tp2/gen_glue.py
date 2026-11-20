# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GEN_NATIVE_GLUE kill switches: no Triton kernel on the TP2 generation step.

The item is default ON and bundles four native paths of the TP4 / TP8 targets:

* ``FSEL``: the shared two-head FSEL top-16 block select (``m31_decode_fsel``).
* ``NATIVE_GLUE``: the drafter's native SwiGLU installer (``ops/native_swiglu``), the C++ drafter SiLU on the generic
  drafter MLP too (``ops/spec/m3_draft_silu``), and the ordered native step-input scatter for warm-up and eager
  flushes (``RuntimeHooks.pack_cpp_scatter``).
* ``SWQ_CPP``: the C++ SwiGLU + MXFP8 producer at every SWQ row (17..32), Triton ``_swq_kernel`` only as fallback.

``TRTLLM_M31_DISABLE_GEN_NATIVE_GLUE=1`` turns all of them off; each part's switch (``TRTLLM_M31_DISABLE_TP2_FSEL``,
``TRTLLM_M31_DISABLE_GEN_GLUE_NATIVE``, ``TRTLLM_M31_DISABLE_SHARED_SWIGLU_MXFP8_NATIVE``) turns one part off. The
per-item switches (``TRTLLM_M31_DISABLE_DRAFT_SWIGLU`` / ``_PACK_CPP_SCATTER``) also work. Values follow
``m31_switches`` (strict, read once at import).
"""

from .m31_switches import m31_disabled

ITEM = "GEN_NATIVE_GLUE"
PARTS = ("FSEL", "NATIVE_GLUE", "SHARED_SWIGLU_MXFP8_NATIVE")
# Each part's kill switch id.
_PART_SWITCH = {
    "FSEL": "TP2_FSEL",
    "NATIVE_GLUE": "GEN_GLUE_NATIVE",
    "SHARED_SWIGLU_MXFP8_NATIVE": "SHARED_SWIGLU_MXFP8_NATIVE",
}


def glue_disabled(part: str, *legacy: str) -> bool:
    """Whether GEN-GLUE part ``part`` is off: the item switch, the part switch, or one of its ``legacy`` item ids."""
    assert part in PARTS, part
    return (
        m31_disabled(ITEM)
        or m31_disabled(_PART_SWITCH[part])
        or any(m31_disabled(x) for x in legacy)
    )


_LOGGED: list = []


def log_engagement(logger, rank: int, **parts) -> None:
    """One line per process naming each part's state, so served / MB logs prove what ran."""
    if _LOGGED:
        return
    _LOGGED.append(True)
    state = ", ".join(f"{k}={v}" for k, v in parts.items())
    logger.info(f"[M3.1 TP2] GEN_NATIVE_GLUE engaged on rank {rank}: {state}")
