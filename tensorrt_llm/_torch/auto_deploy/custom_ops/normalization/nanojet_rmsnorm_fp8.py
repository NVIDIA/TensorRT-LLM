"""nanojet RMSNorm with the quantize in its own epilogue.

Every FP8 consumer downstream of a norm otherwise pays a separate pass over the activations
just to convert them. ``unified_rmsnorm`` already writes e4m3 directly when asked, which is
how nanojet drives this natively — no standalone quantize kernel in the layer at all.
"""

import torch

from ....nanojet_utils import is_nanojet_available

_REGISTERED = False


def register() -> bool:
    """Define the ops, importing nanojet only now. Idempotent; returns availability."""
    global _REGISTERED
    if _REGISTERED:
        return True
    if not is_nanojet_available():
        return False
    _REGISTERED = True

    from nanojet_kernels import ops

    @torch.library.custom_op("auto_deploy::nanojet_rmsnorm_fp8", mutates_args=())
    def nanojet_rmsnorm_fp8(
        hidden_states: torch.Tensor, weight: torch.Tensor, eps: float, quantize_scale: float
    ) -> torch.Tensor:
        """``e4m3(rmsnorm(x))`` in one launch.

        ``quantize_scale`` is the reciprocal of the consumers' dequant scale — nanojet's
        ``fp8_scale`` multiplies — folded on the host at graph-build time so nothing syncs
        to the device per call.
        """
        shape = hidden_states.shape
        hidden = shape[-1]
        output = ops.unified_rmsnorm(
            hidden_states.reshape(-1, hidden),
            weight,
            eps=eps,
            out_dtype=torch.float8_e4m3fn,
            fp8_scale=quantize_scale,
        )
        return output.view(shape)

    @nanojet_rmsnorm_fp8.register_fake
    def _nanojet_rmsnorm_fp8_fake(
        hidden_states: torch.Tensor, weight: torch.Tensor, eps: float, quantize_scale: float
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states, dtype=torch.float8_e4m3fn)

    return True
