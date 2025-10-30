"""A patch for the Bamba model to make it compatible with torch.export."""

from typing import Optional

import torch
from torch import nn
from transformers.models.bamba.modeling_bamba import (
    BambaMixer,
    BambaModel,
    BambaPreTrainedModel,
    HybridMambaAttentionDynamicCache,
    apply_mask_to_padding_states,
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

    use_caching = cache_params is not None

    # 2. Convolution sequence transformation (cached/uncached handled inside the op)
    if use_caching:
        # Prepare dense metadata for cached flattened op
        seq_len_t = torch.full((batch_size,), seq_len, device=input_states.device, dtype=torch.int)
        seq_start_t = torch.arange(
            0, batch_size * seq_len, seq_len, device=input_states.device, dtype=torch.int
        )
        slot_idx_t = torch.arange(batch_size, device=input_states.device, dtype=torch.long)
        use_initial_states_t = torch.zeros(batch_size, device=input_states.device, dtype=torch.bool)
    if use_caching:
        hidden_states_B_C = self.act(
            torch.ops.auto_deploy.torch_cached_causal_conv1d(
                # INPUTS
                hidden_states_B_C,
                self.conv1d.weight,
                self.conv1d.bias,
                # METADATA
                seq_len_t,
                seq_start_t,
                slot_idx_t,
                use_initial_states_t,
                # CACHES
                cache_params.conv_states[self.layer_idx],
                # CONSTANTS
                self.conv1d.stride[0],
                self.conv1d.padding[0],
                self.conv1d.dilation[0],
                self.conv1d.groups,
                self.conv1d.padding_mode,
            )
        )
    else:
        hidden_states_B_C = self.act(
            torch.ops.auto_deploy.torch_causal_conv1d(
                hidden_states_B_C,
                self.conv1d.weight,
                self.conv1d.bias,
                self.conv1d.stride[0],
                self.conv1d.padding[0],
                self.conv1d.dilation[0],
                self.conv1d.groups,
                self.conv1d.padding_mode,
            )
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

    if use_caching:
        # Use new flattened cached op for both cache updates and outputs
        y = torch.ops.auto_deploy.torch_cached_ssm(
            # INPUTS
            hidden_states=hidden_states.view(batch_size, seq_len, -1, self.head_dim),
            A=A,
            B=B.view(batch_size, seq_len, -1, self.ssm_state_size),
            C=C.view(batch_size, seq_len, -1, self.ssm_state_size),
            D=self.D,
            dt=dt,
            dt_bias=self.dt_bias,
            # METADATA
            seq_len=seq_len_t,
            seq_start=seq_start_t,
            slot_idx=slot_idx_t,
            use_initial_states=use_initial_states_t,
            # CACHES
            ssm_state_cache=cache_params.ssm_states[self.layer_idx],
            # CONSTANTS
            time_step_limit=list(self.time_step_limit),
            chunk_size=self.chunk_size,
        )
    else:
        y = torch.ops.auto_deploy.torch_ssm(
            hidden_states=hidden_states.view(batch_size, seq_len, -1, self.head_dim),
            A=A,
            B=B.view(batch_size, seq_len, -1, self.ssm_state_size),
            C=C.view(batch_size, seq_len, -1, self.ssm_state_size),
            D=self.D,
            dt=dt,
            dt_bias=self.dt_bias,
            time_step_limit=list(self.time_step_limit),
            chunk_size=self.chunk_size,
        )

    y = y.view(batch_size, seq_len, -1)

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


def _bamba_model_update_causal_mask(
    self,
    attention_mask,
    input_tensor,
    cache_position,
    past_key_values,
    output_attentions,
):
    # Force attention to use causal mode without explicit masks
    return None


# NOTE: this would need to be applied earlier than other patches, since the `_init_weights` (which
# is called by `post_init`) is called before we run `forward`.
def _bamba_pretrained_model_init_weights(self, module):
    # ARGH python does not like this.
    super(BambaPreTrainedModel, self)._init_weights(module)
    if isinstance(module, BambaMixer):
        module.dt_bias.data.fill_(1.0)
        # ? Why is this even necessary? `BambaMixer.__init__` already sets
        A = torch.arange(1, module.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
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
        self.original_values["BambaModel._update_causal_mask"] = BambaModel._update_causal_mask
        # NOTE: there is `HybridMambaAttentionDynamicCache.__bool__` to save.
        # self.original_values["BambaPreTrainedModel._init_weights"] = BambaPreTrainedModel._init_weights

        BambaMixer.torch_forward = _bamba_mixer_torch_forward
        BambaModel._update_mamba_mask = _bamba_model_update_mamba_mask
        BambaModel._update_causal_mask = _bamba_model_update_causal_mask
        HybridMambaAttentionDynamicCache.__bool__ = _cache_bool
        # BambaPreTrainedModel._init_weights = _bamba_pretrained_model_init_weights

    def _revert_patch(self):
        BambaMixer.torch_forward = self.original_values["BambaMixer.torch_forward"]
        BambaModel._update_mamba_mask = self.original_values["BambaModel._update_mamba_mask"]
        BambaModel._update_causal_mask = self.original_values["BambaModel._update_causal_mask"]
        del HybridMambaAttentionDynamicCache.__bool__
        # BambaPreTrainedModel._init_weights = self.original_values[
        #     "BambaPreTrainedModel._init_weights"
        # ]


# NOTE: patch that is used during build model time
BambaPreTrainedModel._init_weights = _bamba_pretrained_model_init_weights
