"""DiffVAE NA fallbacks when natten is unavailable: Triton then eager SDPA.
Vendored from comfy-kitchen (Apache-2.0). Selection order for NATTEN-kind recipes::
    natten -> (loud warning) Triton -> eager tiled SDPA
Keyframe decode has its own pair, since NATTEN cannot express a joint window::
    (CuTe DSL, when installed) -> Triton joint -> pure-torch joint
and no warning: that is the only ladder there is, not a degraded one.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.fallback_na.eager import na3d as eager_na3d
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.fallback_na.joint_eager import (
    sdpa_materializes_scores as joint_sdpa_materializes_scores,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.keyframes import KeyframePlanes
    from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.attention import NAAttentionCallable, NeighborhoodAttention3D

logger = logging.getLogger(__name__)

_NO_NATTEN_WARNING = (
    "================================================================================\n"
    "DiffVAE: natten is NOT installed. Falling back to a slower neighborhood-attention\n"
    "backend (Triton if available, else pure-PyTorch tiled SDPA). This path is for\n"
    "compatibility only — install natten for production DiffVAE decode:\n"
    "  uv sync --package ltx-core --extra natten\n"
    "================================================================================"
)


def triton_na_available() -> bool:
    """True when CUDA is available and the ``triton`` package imports cleanly.
    Triton on Windows is supported via the community ``triton-windows`` builds
    (https://github.com/triton-lang/triton-windows); there is no platform ban here.
    Import failures beyond ``ImportError`` (e.g. ``OSError`` when libcuda is missing)
    are treated as unavailable so hosts fall back to eager SDPA.
    """
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401, PLC0415
    except (ImportError, OSError):
        return False
    return True


@torch.library.custom_op("ltx_core::na_attention_eager", mutates_args=())
def na_attention_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: list[int],
) -> torch.Tensor:
    """Opaque eager tiled-SDPA NA so Dynamo does not graph-break into the Python loop."""
    return eager_na3d(q, k, v, kernel_size=kernel_size, is_causal=None, scale=1.0)


@na_attention_eager.register_fake
def _na_attention_eager_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: list[int],
) -> torch.Tensor:
    del k, v, kernel_size
    return torch.empty_like(q)


@torch.library.custom_op("ltx_core::na_attention_triton", mutates_args=())
def na_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: list[int],
) -> torch.Tensor:
    """Opaque Triton NA launch so the surrounding block graph stays whole under compile."""
    from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.fallback_na.triton_na import (  # noqa: PLC0415
        na3d as triton_na3d,
    )

    return triton_na3d(q, k, v, kernel_size=kernel_size, is_causal=None, scale=1.0)


@na_attention_triton.register_fake
def _na_attention_triton_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: list[int],
) -> torch.Tensor:
    del k, v, kernel_size
    return torch.empty_like(q)


@torch.library.custom_op("ltx_core::na_attention_joint_eager", mutates_args=())
def na_attention_joint_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    kernel_size: list[int],
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opaque joint NA. Slot tables are built inside, so their T-dependent extent never
    reaches Dynamo and cannot fight the decoder's ``mark_dynamic`` on T/H/W."""
    from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.fallback_na.joint_eager import (  # noqa: PLC0415
        joint_na3d,
    )

    kernel = (kernel_size[0], kernel_size[1], kernel_size[2])
    return joint_na3d(
        q,
        k,
        v,
        keyframe_q,
        keyframe_k,
        keyframe_v,
        keyframe_times,
        keyframe_valid,
        kernel_size=kernel,
        num_slots=num_slots,
    )


@na_attention_joint_eager.register_fake
def _na_attention_joint_eager_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    kernel_size: list[int],
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del k, v, keyframe_k, keyframe_v, keyframe_times, keyframe_valid, kernel_size, num_slots
    return torch.empty_like(q), torch.empty_like(keyframe_q)


@torch.library.custom_op("ltx_core::na_attention_joint_triton", mutates_args=())
def na_attention_joint_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    kernel_size: list[int],
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opaque joint Triton NA. Slot tables are built inside for the same reason as the
    eager op: their T-dependent extent must not reach Dynamo."""
    from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.fallback_na.joint_triton import (  # noqa: PLC0415
        joint_na3d as triton_joint_na3d,
    )

    return triton_joint_na3d(
        q,
        k,
        v,
        keyframe_q,
        keyframe_k,
        keyframe_v,
        keyframe_times,
        keyframe_valid,
        kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
        num_slots=num_slots,
    )


@na_attention_joint_triton.register_fake
def _na_attention_joint_triton_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    kernel_size: list[int],
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del k, v, keyframe_k, keyframe_v, keyframe_times, keyframe_valid, kernel_size, num_slots
    return torch.empty_like(q), torch.empty_like(keyframe_q)


def _joint_dtypes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Cast Q/K to V's dtype per stream (RoPE may have promoted them)."""
    if q.dtype != v.dtype or k.dtype != v.dtype:
        q, k = q.to(dtype=v.dtype), k.to(dtype=v.dtype)
    if keyframe_q.dtype != keyframe_v.dtype or keyframe_k.dtype != keyframe_v.dtype:
        keyframe_q = keyframe_q.to(dtype=keyframe_v.dtype)
        keyframe_k = keyframe_k.to(dtype=keyframe_v.dtype)
    return q, k, v, keyframe_q, keyframe_k, keyframe_v


class TritonJointAttention:
    """Triton joint video+keyframe NA (hosts with CUDA + a working Triton install)."""

    def __call__(
        self,
        attn: NeighborhoodAttention3D,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        keyframe_q: torch.Tensor,
        keyframe_k: torch.Tensor,
        keyframe_v: torch.Tensor,
        keyframe_planes: KeyframePlanes,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.keyframes import KEYFRAME_CONTEXT_SLOTS  # noqa: PLC0415

        if not triton_na_available():
            raise ImportError(
                "Triton joint neighborhood attention requires CUDA and the triton package "
                "(on Windows: https://github.com/triton-lang/triton-windows)."
            )
        return na_attention_joint_triton(
            *_joint_dtypes(q, k, v, keyframe_q, keyframe_k, keyframe_v),
            keyframe_planes.times.device_tensor,
            keyframe_planes.valid.device_tensor,
            list(attn.kernel_size),
            KEYFRAME_CONTEXT_SLOTS,
        )


class EagerJointAttention:
    """Pure-torch joint video+keyframe NA over query bricks (always available).
    The backend for hosts with neither Triton nor the CuTe DSL kernel -- CPU, macOS/MPS, Windows
    without a built extra. Within a few percent of the Triton kernel end to end.
    """

    def __call__(
        self,
        attn: NeighborhoodAttention3D,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        keyframe_q: torch.Tensor,
        keyframe_k: torch.Tensor,
        keyframe_v: torch.Tensor,
        keyframe_planes: KeyframePlanes,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.keyframes import KEYFRAME_CONTEXT_SLOTS  # noqa: PLC0415

        return na_attention_joint_eager(
            *_joint_dtypes(q, k, v, keyframe_q, keyframe_k, keyframe_v),
            keyframe_planes.times.device_tensor,
            keyframe_planes.valid.device_tensor,
            list(attn.kernel_size),
            KEYFRAME_CONTEXT_SLOTS,
        )


class AutoJointAttention:
    """Joint NA that picks its kernel from the *tensors*, not from the host.
    Chooses per call rather than at install time because ``joint_attention_function`` is
    installed for every mode on every NA module, and a CPU decoder on a CUDA host is an
    ordinary case (unit tests, CPU dev). Deciding by ``triton_na_available()`` alone would
    hand CPU tensors to a Triton kernel.
    """

    def __init__(self) -> None:
        self._triton = TritonJointAttention() if triton_na_available() else None
        self._eager = EagerJointAttention()

    def __call__(
        self,
        attn: NeighborhoodAttention3D,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        keyframe_q: torch.Tensor,
        keyframe_k: torch.Tensor,
        keyframe_v: torch.Tensor,
        keyframe_planes: KeyframePlanes,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        backend = self._triton if self._triton is not None and q.is_cuda else self._eager
        return backend(attn, q, k, v, keyframe_q, keyframe_k, keyframe_v, keyframe_planes)


class EagerSdpaAttention:
    """Limited-workspace tiled SDPA NA (always available)."""

    def __call__(
        self,
        attn: NeighborhoodAttention3D,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if q.dtype != v.dtype or k.dtype != v.dtype:
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
        return na_attention_eager(q, k, v, list(attn.kernel_size))


class TritonNaAttention:
    """Triton flash-style NA (hosts with CUDA + a working Triton install)."""

    def __call__(
        self,
        attn: NeighborhoodAttention3D,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if not triton_na_available():
            raise ImportError(
                "Triton neighborhood attention requires CUDA and the triton package "
                "(on Windows: https://github.com/triton-lang/triton-windows)."
            )
        if q.dtype != v.dtype or k.dtype != v.dtype:
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
        return na_attention_triton(q, k, v, list(attn.kernel_size))


def warn_no_natten(*, backend: str) -> None:
    """Emit the loud no-natten banner and which fallback backend was chosen."""
    logger.warning(_NO_NATTEN_WARNING)
    logger.warning("DiffVAE NA fallback: using %s.", backend)


def fallback_na_attention() -> NAAttentionCallable:
    """Pick Triton if usable, else eager; emit the no-natten warning."""
    if triton_na_available():
        warn_no_natten(backend="Triton na3d")
        return TritonNaAttention()
    warn_no_natten(backend="eager tiled SDPA na3d")
    return EagerSdpaAttention()


def joint_na_attention() -> object:
    """The joint (keyframe-aware) NA backend: Triton on CUDA tensors, eager otherwise.
    Never NATTEN, which cannot express the joint window. The CuTe DSL kernel can, and under
    ``BLACKWELL_DSL`` ``apply`` wraps this in ``DSLJointAttention``, which prefers the kernel
    and falls back here for the calls it cannot serve. So this
    is independent of ``natten_available()`` and emits no warning: unlike the video-only
    fallbacks this is not a degraded path, it is the only one there is.
    """
    return AutoJointAttention()


__all__ = [
    "AutoJointAttention",
    "EagerJointAttention",
    "EagerSdpaAttention",
    "TritonJointAttention",
    "TritonNaAttention",
    "fallback_na_attention",
    "joint_na_attention",
    "joint_sdpa_materializes_scores",
    "na_attention_eager",
    "na_attention_joint_eager",
    "na_attention_joint_triton",
    "na_attention_triton",
    "triton_na_available",
    "warn_no_natten",
]
