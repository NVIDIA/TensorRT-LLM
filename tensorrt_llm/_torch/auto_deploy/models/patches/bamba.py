"""A patch for the Bamba model to make it compatible with torch.export."""

from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers.models.bamba.modeling_bamba import (
    BambaMixer,
    BambaModel,
    HybridMambaAttentionDynamicCache,
    apply_mask_to_padding_states,
    pad_tensor_by_size,
    reshape_into_chunks,
    segment_sum,
)

from ...export.interface import BaseExportPatch, ExportPatchRegistry


# Original implementation:
# https://github.com/huggingface/transformers/blob/06f8004e5cd9d06cfbffc3f47afb6c2b43bcb3d2/
# src/transformers/models/bamba/modeling_bamba.py#L719
# NOTE: we remove the cache-related code paths.
def _bamba_mixer_torch_forward(
    self,
    input_states,
    cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    # `input_states` is of shape `[B, T, hidden_dim]`, and during generate, each successive call
    # will have T = T + 1.
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype

    # 1. Gated MLP's linear projection
    input_states = apply_mask_to_padding_states(input_states, attention_mask)
    projected_states = self.in_proj(input_states)
    gate, hidden_states_B_C, dt = projected_states.split(
        [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
    )

    use_precomputed_states = (
        cache_params is not None
        and cache_params.has_previous_state
        and seq_len == 1
        and cache_params.conv_states[self.layer_idx].shape[0]
        == cache_params.ssm_states[self.layer_idx].shape[0]
        == batch_size
        and cache_position is not None
        and cache_position[0] > 0
    )

    # 2. Convolution sequence transformation
    if use_precomputed_states:
        cache_params.conv_states[self.layer_idx] = cache_params.conv_states[self.layer_idx].roll(
            shifts=-1, dims=-1
        )
        cache_params.conv_states[self.layer_idx][:, :, -1] = hidden_states_B_C[:, 0, :].to(
            cache_params.conv_states[self.layer_idx].device
        )

        # We need to guarantee that anything regarding the cache is on the same device
        conv_states = cache_params.conv_states[self.layer_idx].to(device=self.conv1d.weight.device)

        hidden_states_B_C = torch.sum(conv_states * self.conv1d.weight.squeeze(1), dim=-1)
        if self.use_conv_bias:
            hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
        hidden_states_B_C = self.act(hidden_states_B_C)
    else:
        # Init cache
        if cache_params is not None:
            hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
            conv_states = nn.functional.pad(
                hidden_states_B_C_transposed,
                (self.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0),
            )
            cache_params.conv_states[self.layer_idx].copy_(conv_states)

        hidden_states_B_C = self.act(
            self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        )

    hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
    hidden_states, B, C = torch.split(
        hidden_states_B_C,
        [
            self.intermediate_size,
            self.n_groups * self.ssm_state_size,
            self.n_groups * self.ssm_state_size,
        ],
        dim=-1,
    )

    # 3. SSM transformation
    A = -torch.exp(self.A_log.float())  # [num_heads]
    if use_precomputed_states:
        y, ssm_state = torch.ops.auto_deploy.ssm_transform_cached(
            hidden_states=hidden_states,
            A=A,
            B=B,
            C=C,
            D=self.D,
            dt=dt,
            dt_bias=self.dt_bias,
            ssm_state_size=self.ssm_state_size,
            time_step_limit=list(self.time_step_limit),
            batch_size=batch_size,
            seq_len=seq_len,
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            n_groups=self.n_groups,
            chunk_size=self.chunk_size,
            cached_ssm_state=cache_params.ssm_states[self.layer_idx],
        )
    else:
        y, ssm_state = torch.ops.auto_deploy.ssm_transform(
            hidden_states=hidden_states,
            A=A,
            B=B,
            C=C,
            D=self.D,
            dt=dt,
            dt_bias=self.dt_bias,
            ssm_state_size=self.ssm_state_size,
            time_step_limit=list(self.time_step_limit),
            batch_size=batch_size,
            seq_len=seq_len,
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            n_groups=self.n_groups,
            chunk_size=self.chunk_size,
        )

    # Init cache
    if ssm_state is not None and cache_params is not None:
        cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

    scan_output = self.norm(y, gate)

    # end ssd naive
    # !! This is the end of the uncached code path.

    # 4. Final linear projection
    contextualized_states = self.out_proj(scan_output.to(dtype))  # [batch, seq_len, hidden_size]
    return contextualized_states


# The original implementation looks at `cache_position[0]` to decide what to do which does not
# play well with export. Plus, we do not want it to be updated anyway.
def _bamba_model_update_mamba_mask(self, attention_mask, cache_position):
    return None


@torch.library.custom_op("auto_deploy::ssm_transform", mutates_args={})
def _ssm_transform(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state_size: int,
    # NOTE: `torch` custom ops do not like `Tuple` inputs. Using `List` is the suggested WAR.
    time_step_limit: List[float],
    batch_size: int,
    seq_len: int,
    head_dim: int,
    num_heads: int,
    n_groups: int,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # !! This is the uncached code path from the original implementation.
    # begin ssd naive implementation without einsums
    dt = nn.functional.softplus(dt + dt_bias)
    dt = torch.clamp(dt, time_step_limit[0], time_step_limit[1])
    hidden_states = hidden_states.reshape(batch_size, seq_len, -1, head_dim).float()
    B = B.reshape(batch_size, seq_len, -1, ssm_state_size).float()
    C = C.reshape(batch_size, seq_len, -1, ssm_state_size).float()
    B = B.repeat_interleave(num_heads // n_groups, dim=2, output_size=num_heads)
    C = C.repeat_interleave(num_heads // n_groups, dim=2, output_size=num_heads)
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    D_residual = D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

    # Discretize x and A
    hidden_states = hidden_states * dt[..., None]
    A = A.to(hidden_states.dtype) * dt

    # Rearrange into blocks/chunks
    hidden_states, A, B, C = [
        reshape_into_chunks(t, pad_size, chunk_size) for t in (hidden_states, A, B, C)
    ]

    # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    # This is the analog of a causal mask
    L = torch.exp(segment_sum(A))

    # Contraction of C and B to get G (attention-weights like)
    G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]  # shape: (b, c, l, s, h, n)
    G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)

    # Compute M, equivalent to applying attention mask to weights
    M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
    M = M_intermediate.sum(dim=-1)

    # Compute Y_diag (apply to values)
    Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
    states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    previous_states = torch.zeros_like(states[:, :1])
    states = torch.cat([previous_states, states], dim=1)
    decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
    decay_chunk = decay_chunk.transpose(1, 3)
    new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
    states, ssm_state = new_states[:, :-1], new_states[:, -1]
    # states = new_states[:, :-1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    C_times_states = C[..., None, :] * states[:, :, None, ...]
    state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
    Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    y = Y_diag + Y_off
    # [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
    y = y.reshape(batch_size, -1, num_heads, head_dim)

    y = y + D_residual
    # Cutting off padded chunks
    if pad_size > 0:
        y = y[:, :seq_len, :, :]
    y = y.reshape(batch_size, seq_len, -1)

    return y, ssm_state


@_ssm_transform.register_fake
def _ssm_transform_meta(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state_size: int,
    # NOTE: `torch` custom ops do not like `Tuple` inputs. Using `List` is the suggested WAR.
    time_step_limit: List[float],
    batch_size: int,
    seq_len: int,
    head_dim: int,
    num_heads: int,
    n_groups: int,
    chunk_size: int,
):
    return (
        torch.empty_like(hidden_states),
        torch.empty(batch_size, num_heads, head_dim, ssm_state_size, dtype=hidden_states.dtype),
    )


@torch.library.custom_op("auto_deploy::ssm_transform_cached", mutates_args={})
def _ssm_transform_cached(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state_size: int,
    # NOTE: `torch` custom ops do not like `Tuple` inputs. Using `List` is the suggested WAR.
    time_step_limit: List[float],
    batch_size: int,
    seq_len: int,
    head_dim: int,
    num_heads: int,
    n_groups: int,
    chunk_size: int,
    cached_ssm_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # We need to guarantee that anything regarding the cache is on the same device
    cache_device = cached_ssm_state.device

    # Note: there is no need to pad parameter matrices here, as there is just one new token
    # for batched generation
    dt = dt[:, 0, :][:, None, ...]
    dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], head_dim)
    # [num_heads] -> [num_heads, head_dim]
    dt_bias = dt_bias[..., None].expand(dt_bias.shape[0], head_dim)

    dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
    dt = torch.clamp(dt, time_step_limit[0], time_step_limit[1])
    A = A[..., None, None].expand(num_heads, head_dim, ssm_state_size).to(dtype=torch.float32)
    # [bsz, num_heads, head_dim, state_size]
    dA = (torch.exp(dt[..., None] * A)).to(device=cache_device)

    # Discretize B
    # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
    # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
    B = B.reshape(batch_size, n_groups, -1)[..., None, :]
    B = B.expand(batch_size, n_groups, num_heads // n_groups, B.shape[-1]).contiguous()
    B = B.reshape(batch_size, -1, B.shape[-1])
    # [bsz, num_heads, head_dim, state_size]
    dB = dt[..., None] * B[..., None, :]

    # Discretize x into dB
    # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
    hidden_states = hidden_states.reshape(batch_size, -1, head_dim)
    dBx = (dB * hidden_states[..., None]).to(device=cache_device)

    # State calculation
    # cached_ssm_state.copy_(cached_ssm_state * dA + dBx)
    cached_ssm_state = cached_ssm_state * dA + dBx

    # Subsequent output
    # [bsz, n_groups * state_size] -> [bsz, num_heads, state_size]
    C = C.reshape(batch_size, n_groups, -1)[..., None, :]
    C = C.expand(batch_size, n_groups, num_heads // n_groups, C.shape[-1]).contiguous()
    C = C.reshape(batch_size, -1, C.shape[-1])
    # [bsz, num_heads, head_dim]

    ssm_states = cached_ssm_state.to(device=C.device, dtype=C.dtype)  # Shape: [b, h, d, n]
    # Reshape ssm_states to merge the first two dimensions
    ssm_states_reshaped = ssm_states.view(
        batch_size * num_heads, head_dim, ssm_state_size
    )  # Shape: [b*h, d, n]
    C_reshaped = C.view(batch_size * num_heads, ssm_state_size, 1)  # Shape: [b*h, n, 1]
    y = torch.bmm(ssm_states_reshaped, C_reshaped)
    y = y.view(batch_size, num_heads, head_dim)

    # D skip connection
    # [num_heads] -> [num_heads, head_dim]
    D = D[..., None].expand(D.shape[0], head_dim)
    y = (y + hidden_states * D).to(y.dtype)

    # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
    y = y.reshape(batch_size, -1)[:, None, ...]
    return y, cached_ssm_state


@_ssm_transform_cached.register_fake
def _ssm_transform_meta(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state_size: int,
    # NOTE: `torch` custom ops do not like `Tuple` inputs. Using `List` is the suggested WAR.
    time_step_limit: List[float],
    batch_size: int,
    seq_len: int,
    head_dim: int,
    num_heads: int,
    n_groups: int,
    chunk_size: int,
    cached_ssm_state: torch.Tensor,
):
    return (
        torch.empty_like(hidden_states),
        torch.empty(batch_size, num_heads, head_dim, ssm_state_size, dtype=hidden_states.dtype),
    )


# NOTE: this would need to be applied earlier than other patches, since the `_init_weights` (which
# is called by `post_init`) is called before we run `forward`.
def _bamba_pretrained_model_init_weights(self, module):
    # ARGH python does not like this.
    super()._init_weights(module)
    if isinstance(module, BambaMixer):
        module.dt_bias.data.fill_(1.0)
        # ? Why is this even necessary? `BambaMixer.__init__` already sets
        # A = torch.arange(1, self.num_heads + 1)
        # self.A_log = nn.Parameter(torch.log(A))
        # module.A_log.data = torch.log(torch.arange(1, module.num_heads + 1))
        module.D.data.fill_(1.0)


def _cache_bool(self) -> bool:
    # This is a workaround for the bug on this line:
    # https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/models/bamba/modeling_bamba.py#L1211
    # The base `Cache` class from which `HybridMambaAttentionDynamicCache` inherits has a `__len__`
    # special method implement that returns `False` when the instance is initialized, but stores no
    # cache.
    # The original line should really be checking for `if past_key_values is not None` but adding
    # a patch for `BambaModel.forward` just for that reason seems overly intrusive.
    return True


@ExportPatchRegistry.register("bamba")
class BambaModelPatch(BaseExportPatch):
    """Patch for `BambaMixer`."""

    def _apply_patch(self):
        self.original_values["BambaMixer.torch_forward"] = BambaMixer.torch_forward
        self.original_values["BambaModel._update_mamba_mask"] = BambaModel._update_mamba_mask
        # NOTE: there is `HybridMambaAttentionDynamicCache.__bool__` to save.
        # self.original_values["BambaPreTrainedModel._init_weights"] = BambaPreTrainedModel._init_weights

        BambaMixer.torch_forward = _bamba_mixer_torch_forward
        BambaModel._update_mamba_mask = _bamba_model_update_mamba_mask
        HybridMambaAttentionDynamicCache.__bool__ = _cache_bool
        # BambaPreTrainedModel._init_weights = _bamba_pretrained_model_init_weights

    def _revert_patch(self):
        BambaMixer.torch_forward = self.original_values["BambaMixer.torch_forward"]
        BambaModel._update_mamba_mask = self.original_values["BambaModel._update_mamba_mask"]
        del HybridMambaAttentionDynamicCache.__bool__
        # BambaPreTrainedModel._init_weights = self.original_values[
        #     "BambaPreTrainedModel._init_weights"
        # ]
