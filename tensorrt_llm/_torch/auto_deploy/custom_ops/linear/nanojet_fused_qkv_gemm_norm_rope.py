"""nanojet fused QKV projection with Q/K norm and RoPE, in one kernel."""

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

    @torch.library.custom_op("auto_deploy::nanojet_fused_qkv_gemm_norm_rope", mutates_args=())
    def nanojet_fused_qkv_gemm_norm_rope(
        hidden_states: torch.Tensor,
        qkv_weight: torch.Tensor,
        query_norm_weight: torch.Tensor,
        key_norm_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        position_ids: torch.Tensor,
        input_scale: torch.Tensor,
        eps: float,
        query_scale: float,
        key_scale: float,
        value_scale: float,
        query_size: int,
        key_value_size: int,
    ) -> torch.Tensor:
        """Project, normalize and rotate in one launch.

        ``qkv_weight`` is the three projections stacked ``[q + 2kv, hidden]``; Q, K and V
        come back stacked on the last dim, for the graph to slice apart as views.
        """
        batch, seq, hidden_size = hidden_states.shape
        num_tokens = batch * seq
        flattened = hidden_states.reshape(num_tokens, hidden_size)
        positions = position_ids.reshape(-1)
        if positions.numel() != num_tokens:
            raise RuntimeError(
                f"nanojet_fused_qkv_gemm_norm_rope got {positions.numel()} position ids for "
                f"{num_tokens} tokens ({batch}x{seq}). Every token needs its own position."
            )
        if flattened.dtype == torch.float8_e4m3fn:
            quantized = flattened
        else:
            quantized, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                flattened, input_scale
            )

        packed = torch.empty(
            num_tokens,
            query_size + 2 * key_value_size,
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        ops.fused_qkv_gemm_norm_rope(
            packed,
            quantized,
            qkv_weight,
            query_scale,
            key_scale,
            value_scale,
            query_size,
            key_value_size,
            query_norm_weight,
            key_norm_weight,
            eps,
            cos_sin_cache,
            positions,
        )
        return packed.view(batch, seq, -1)

    @nanojet_fused_qkv_gemm_norm_rope.register_fake
    def _(
        hidden_states: torch.Tensor,
        qkv_weight: torch.Tensor,
        query_norm_weight: torch.Tensor,
        key_norm_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        position_ids: torch.Tensor,
        input_scale: torch.Tensor,
        eps: float,
        query_scale: float,
        key_scale: float,
        value_scale: float,
        query_size: int,
        key_value_size: int,
    ) -> torch.Tensor:
        return torch.empty(
            hidden_states.shape[0],
            hidden_states.shape[1],
            query_size + 2 * key_value_size,
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )

    return True
