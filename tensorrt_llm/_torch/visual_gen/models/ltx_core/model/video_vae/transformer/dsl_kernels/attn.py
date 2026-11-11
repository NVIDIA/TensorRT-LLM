"""CuTe DSL neighborhood-attention callable for DiffVAE NA modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.attention import NAAttentionCallable

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.attention import NeighborhoodAttention3D

#: Buffer holding the kernels' softmax bound. Registered by BLACKWELL_DSL apply
#: (NaN sentinel) and filled by DiffVAE decoder SDOps (``na_dsl_kernel=True``).
NA_SOFTMAX_BOUND_BUFFER = "softmax_bound"

_SOFTMAX_BOUND_UNSET = (
    f"{NA_SOFTMAX_BOUND_BUFFER} is unset: apply DiffVAEMode.BLACKWELL_DSL "
    "with video_decoder_sd_ops_for_checkpoint(na_dsl_kernel=True)"
)


def na_dsl_available() -> bool:
    """Whether the CuTe DSL neighborhood-attention kernel can run on this GPU."""
    try:
        from ltx_kernels.vae import na_attn_available  # noqa: PLC0415
    except ImportError:
        return False
    return na_attn_available()


def require_softmax_bound(attn: NeighborhoodAttention3D) -> torch.Tensor:
    """Return ``attn``'s filled softmax bound, or raise if apply/SDOps were unpaired.
    Apply registers a NaN 0-d buffer; SDOps overwrite it at load. Rejecting meta, empty,
    and non-finite values catches both a missed key under ``assign=True`` (meta left
    behind) and an apply-without-SDOps path that left the NaN sentinel on CUDA.
    The non-finite probe is an asynchronous assert: it raises at once for a CPU buffer, and
    as a device-side assert for a CUDA one, so the per-block call never blocks the host on
    the GPU. It is skipped while ``torch.compile`` is tracing: ``isfinite().all()`` is
    data-dependent and forces specialization that conflicts with the decoder's hard
    ``mark_dynamic`` on T/H/W. Meta/empty checks still run under compile.
    """
    bound = getattr(attn, NA_SOFTMAX_BOUND_BUFFER, None)
    if bound is None or bound.is_meta or bound.numel() == 0:
        raise RuntimeError(_SOFTMAX_BOUND_UNSET)
    if not torch.compiler.is_compiling():
        torch._assert_async(torch.isfinite(bound).all(), _SOFTMAX_BOUND_UNSET)
    return bound


@torch.library.custom_op("ltx_core::na_attention_dsl", mutates_args=())
def na_attention_dsl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: list[int],
    softmax_bound: torch.Tensor,
) -> torch.Tensor:
    """One launch of the DSL NA kernel, as an opaque op so the block's graph stays whole."""
    from ltx_kernels.vae import run_na_attention_bound  # noqa: PLC0415

    return run_na_attention_bound(q, k, v, kernel_size=tuple(kernel_size), k_bound=softmax_bound)


@na_attention_dsl.register_fake
def _na_attention_dsl_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: list[int],
    softmax_bound: torch.Tensor,
) -> torch.Tensor:
    del k, v, kernel_size, softmax_bound
    batch, t, h, w, num_heads, head_dim = q.shape
    return q.new_empty((batch, t, h, w, num_heads * head_dim))


class DSLAttention(NAAttentionCallable):
    """The CuTe DSL neighborhood-attention kernel."""

    def __call__(
        self,
        attn: NeighborhoodAttention3D,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        return na_attention_dsl(q, k, v, list(attn.kernel_size), require_softmax_bound(attn))
