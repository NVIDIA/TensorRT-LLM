"""The CuTe DSL kernel as a joint (video + keyframe) NA backend.
A third option at ``NeighborhoodAttention3D.joint_attention_function``, next to the Triton
and eager ones. The DSL kernel *can* express the joint window: it folds each visible
keyframe plane's ``Kh x Kw`` panel into the same fixed-offset accumulator as the local
window, so continuing the accumulation is already one softmax over both key sets -- no
rescale and no state merge, because the softmax offset is a bound rather than an online
max (see :mod:`ltx_kernels.vae.softmax_bound`).
It is not always selectable, so this is a *preference* rather than an install: the kernel
needs a datacenter Blackwell GPU, CUDA tensors, a filled softmax bound (only the
``BLACKWELL_DSL`` mode registers one) and a volume at least as large as its halo panel.
Whenever any of that fails the call goes to the backend this wraps.
Two things the seam offers that the kernel has no notion of, both handled here rather than
in the kernel:
``keyframe_valid``
    Invalid planes are dropped before the launch and their output rows restored as zeros
    afterwards. Dropping is order-preserving, so the surviving planes rank identically
    under the ``(|dt|, index)`` tie-break -- the compaction cannot change which plane a
    query selects, only what it is numbered.
``float64 / CPU tensors``
    Handed straight to the wrapped backend; the kernel is bf16 on device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.attention import (
    JointNAAttentionCallable,
    NeighborhoodAttention3D,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.dsl_kernels.attn import (
    NA_SOFTMAX_BOUND_BUFFER,
    na_dsl_available,
    require_softmax_bound,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.keyframes import KeyframePlanes


@torch.library.custom_op("ltx_core::na_attention_joint_dsl", mutates_args=())
def na_attention_joint_dsl(  # noqa: PLR0913 -- a custom_op schema takes only flat tensors/ints
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    softmax_bound: torch.Tensor,
    kernel_size: list[int],
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opaque joint DSL NA. Slot tables are built inside for the same reason as the others.
    Their extent depends on ``T``, which the decoder marks dynamic at five sites, so a table
    built outside would put a guard on those symbols -- an error there rather than a
    recompile.
    """
    from ltx_kernels.vae import run_na_attention_bound  # noqa: PLC0415
    from ltx_kernels.vae.keyframe_slots import nearest_keyframe_slots  # noqa: PLC0415

    from tensorrt_llm._torch.visual_gen.models.ltx_core.devices import upload_from_host  # noqa: PLC0415

    planes = int(keyframe_q.shape[1])
    # Plane selection and slot tables are host work; free when the times and validity arrive on
    # the host, one read of each otherwise.
    keep_host = keyframe_valid.detach().cpu().nonzero().flatten()
    live_times = keyframe_times.detach().cpu().index_select(0, keep_host)
    keep = upload_from_host(keep_host, q.device)
    video_out, keyframe_out = run_na_attention_bound(
        q,
        k,
        v,
        kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
        k_bound=softmax_bound,
        k_keyframes=keyframe_k.index_select(1, keep),
        v_keyframes=keyframe_v.index_select(1, keep),
        keyframe_slots=nearest_keyframe_slots(live_times, int(q.shape[1]), slots=num_slots),
        q_keyframes=keyframe_q.index_select(1, keep),
        keyframe_times=live_times,
    )
    video_out = video_out.reshape(q.shape)
    if int(keep.numel()) == planes:
        return video_out, keyframe_out.reshape(keyframe_q.shape)
    # Restore the dropped rows as zeros: an invalid plane's every key is masked, and the
    # contract is that such a row comes out exactly zero rather than NaN.
    restored = keyframe_q.new_zeros(keyframe_q.shape)
    if int(keep.numel()):
        restored.index_copy_(1, keep, keyframe_out.reshape(1, int(keep.numel()), *keyframe_q.shape[2:]))
    return video_out, restored


@na_attention_joint_dsl.register_fake
def _na_attention_joint_dsl_fake(  # noqa: PLR0913 -- mirrors the op's schema
    q: torch.Tensor,
    k: torch.Tensor,  # noqa: ARG001
    v: torch.Tensor,  # noqa: ARG001
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,  # noqa: ARG001
    keyframe_v: torch.Tensor,  # noqa: ARG001
    keyframe_times: torch.Tensor,  # noqa: ARG001
    keyframe_valid: torch.Tensor,  # noqa: ARG001
    softmax_bound: torch.Tensor,  # noqa: ARG001
    kernel_size: list[int],  # noqa: ARG001
    num_slots: int,  # noqa: ARG001
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(keyframe_q)


def dsl_joint_supported(attn: NeighborhoodAttention3D, q: torch.Tensor) -> bool:
    """Whether the DSL joint kernel can serve *this* call.
    Per call rather than at install time: ``joint_attention_function`` is installed on every
    NA module for every mode, and both the softmax bound and the volume are properties of the
    module and the tensors, not of the host.
    """
    if not q.is_cuda or q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if getattr(attn, NA_SOFTMAX_BOUND_BUFFER, None) is None:
        return False
    from ltx_kernels.vae import na_supported  # noqa: PLC0415

    _, time, height, width, num_heads, head_dim = q.shape
    return na_supported(
        num_heads=num_heads,
        head_dim=head_dim,
        kernel_size=tuple(attn.kernel_size),
        T=time,
        H=height,
        W=width,
    )


class DSLJointAttention(JointNAAttentionCallable):
    """Joint NA on the CuTe DSL kernel, falling back to ``fallback`` when it cannot serve.
    The fallback is not a nicety: this is installed for every mode, so a CPU decoder, a
    volume smaller than the kernel's halo panel, or a checkpoint loaded without the DSL
    SDOps all land here and must still decode.
    """

    def __init__(self, fallback: JointNAAttentionCallable) -> None:
        self._fallback = fallback
        self._available = na_dsl_available()

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

        if not self._available or not dsl_joint_supported(attn, q):
            return self._fallback(attn, q, k, v, keyframe_q, keyframe_k, keyframe_v, keyframe_planes)
        # The op builds its plane tables on the host, so it gets the planes' host tensors.
        keyframe_times, keyframe_valid = keyframe_planes.times.host_tensor, keyframe_planes.valid.host_tensor
        dtype = keyframe_v.dtype
        return na_attention_joint_dsl(
            q.to(dtype=dtype),
            k.to(dtype=dtype),
            v.to(dtype=dtype),
            keyframe_q.to(dtype=dtype),
            keyframe_k.to(dtype=dtype),
            keyframe_v,
            keyframe_times,
            keyframe_valid,
            require_softmax_bound(attn),
            list(attn.kernel_size),
            KEYFRAME_CONTEXT_SLOTS,
        )
