from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .linear import Linear, TensorParallelMode


@dataclass
class MambaCacheParams:
    conv_states: torch.Tensor = torch.Tensor()
    ssm_states: torch.Tensor = torch.Tensor()
    indices: torch.Tensor = torch.Tensor()


class MambaCacheManager:

    def __init__(
        self,
        *,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        n_groups: int,
        head_dim: int,
        num_mamba_layers: int,
        max_batch_size: int,
        dtype: Optional[torch.dtype] = None,
        config: Optional[ModelConfig] = None,
    ):

        config = config or ModelConfig()
        tp_size = config.mapping.tp_size

        d_inner = d_model * expand
        conv_dim = d_inner + 2 * n_groups * d_state
        nheads = d_inner // head_dim

        assert nheads % tp_size == 0, "nheads must be divisible by tp_size"
        assert conv_dim % tp_size == 0, "conv_dim must be divisible by tp_size"

        conv_dim = conv_dim // tp_size
        nheads = nheads // tp_size

        device = torch.device("cuda")

        self.conv_states = torch.empty(
            size=[num_mamba_layers, max_batch_size, d_conv - 1, conv_dim],
            dtype=dtype,
            device=device,
        )

        self.ssm_states = torch.empty(
            size=[
                num_mamba_layers,
                max_batch_size,
                nheads,
                d_state,
                head_dim,
            ],
            dtype=dtype,
            device=device,
        )

    def get_params(self, attn_metadata: AttentionMetadata) -> MambaCacheParams:
        # request_ids is set to None when warming up the engine
        # we set this warmup request to zero position
        request_ids = ([0] if attn_metadata.request_ids is None else
                       attn_metadata.request_ids)

        # this implements the simplest possible mapping, indices = request_ids
        indices = torch.as_tensor(request_ids, dtype=torch.int32)

        # return cache params
        return MambaCacheParams(self.conv_states, self.ssm_states, indices)


class RMSNormGroup(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_groups: int,
        eps: float = 1e-5,
        dtype: Optional[torch.dtype] = None,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__()

        config = config or ModelConfig()
        tp_size = config.mapping.tp_size
        tp_rank = config.mapping.tp_rank

        assert (hidden_size %
                num_groups == 0), "hidden_size must be divisible by num_groups"

        assert (hidden_size %
                tp_size == 0), "hidden_size must be divisible by tp_size"

        assert (num_groups %
                tp_size == 0), "num_groups must be divisible by tp_size"

        tp_hidden_size = hidden_size // tp_size
        tp_ngroups = num_groups // tp_size

        self.num_groups = tp_ngroups
        self.group_size = tp_hidden_size // tp_ngroups

        self.weight = nn.Parameter(torch.empty(tp_hidden_size, dtype=dtype))
        self.variance_epsilon = eps
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        input_dtype = x.dtype
        y = x.to(torch.float32)

        # Reshape to [*, num_groups, group_size]
        orig_shape = y.shape
        y = y.view(*y.shape[:-1], self.num_groups, self.group_size)

        # Compute variance over the group_size dimension
        variance = y.pow(2).mean(-1, keepdim=True)

        # Normalize within each group
        y = y * torch.rsqrt(variance + self.variance_epsilon)

        # Reshape back to original shape
        y = y.view(*orig_shape)

        # Apply the scaling weight
        y = self.weight * y.to(input_dtype)

        return y


class Mamba2(nn.Module):

    def __init__(
        self,
        *,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        n_groups: int,
        head_dim: int,
        chunk_size: int,
        layer_idx: int,
        bias: bool = False,
        conv_bias: bool = True,
        delta_rank: int = 0,
        delta_softplus: bool = True,
        remove_padding: bool = True,
        apply_silu: bool = True,
        rms_norm_eps: float = 1e-5,
        dtype: Optional[torch.dtype] = None,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__()

        config = config or ModelConfig()
        self.mapping = config.mapping
        tp_rank = config.mapping.tp_rank
        tp_size = config.mapping.tp_size

        d_inner = d_model * expand
        nheads = d_inner // head_dim
        d_in_proj = 2 * d_inner + 2 * n_groups * d_state + nheads
        conv_dim = d_inner + 2 * n_groups * d_state

        # TP
        self.tp_conv_dim = conv_dim // tp_size
        self.tp_d_inner = d_inner // tp_size
        self.tp_nheads = nheads // tp_size
        self.tp_ngroups = n_groups // tp_size

        self.layer_idx = layer_idx
        self.d_conv = d_conv
        self.d_state = d_state
        self.chunk_size = chunk_size
        self.delta_rank = delta_rank
        self.delta_softplus = delta_softplus
        self.remove_padding = remove_padding
        self.is_mamba2 = True
        self.apply_silu = apply_silu

        # tp
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        # paged state parameters
        self.slot_mapping = None
        self.is_paged_state = False

        # in_proj
        self.in_proj = Linear(
            d_model,
            d_in_proj,
            bias=bias,
            dtype=dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=config.get_quant_config(),
        )

        # conv1d, reuse Linear to store weights since it has support for TP > 1 already
        self.conv1d = Linear(
            d_conv,
            conv_dim,
            bias=conv_bias,
            dtype=dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=config.get_quant_config(),
            skip_create_weights=config.skip_create_weights,
        )

        # A
        self.A = nn.Parameter(
            torch.empty(self.tp_nheads,
                        dtype=torch.float32,
                        requires_grad=False))

        # D
        self.D = nn.Parameter(
            torch.empty(self.tp_nheads,
                        dtype=torch.float32,
                        requires_grad=False))

        # dt_bias
        self.dt_bias = nn.Parameter(
            torch.empty(self.tp_nheads,
                        dtype=torch.float32,
                        requires_grad=False))

        # norm
        self.norm = RMSNormGroup(d_inner,
                                 n_groups,
                                 eps=rms_norm_eps,
                                 dtype=dtype,
                                 config=config)

        # out_proj
        self.out_proj = Linear(d_inner,
                               d_model,
                               bias=bias,
                               dtype=dtype,
                               mapping=self.mapping,
                               tensor_parallel_mode=TensorParallelMode.ROW,
                               quant_config=config.get_quant_config())

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_cache_params: MambaCacheParams,
    ) -> torch.Tensor:

        # calculate split size
        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.seq_lens.shape[0] - num_contexts
        sum_seq = torch.cumsum(attn_metadata.seq_lens, dim=0)
        split_ctx = sum_seq[num_contexts - 1] if num_contexts > 0 else 0
        split_gen = sum_seq[-1] - split_ctx
        split_size = [split_ctx, split_gen]

        # make the split
        split_indices = torch.split(mamba_cache_params.indices,
                                    [num_contexts, num_generations])
        split_seq_lens = torch.split(attn_metadata.seq_lens,
                                     [num_contexts, num_generations])
        split_hidden_states = torch.split(hidden_states, split_size)

        # a batch can have either:
        # * only context requests
        # * only generation requests
        # * both context and generation requests
        # req_type = 0 -> context
        # req_type = 1 -> generation
        batch = None
        # both context and generation requests
        if num_contexts > 0 and num_generations > 0:
            batch = [0, 1]
        # only context requests
        elif num_contexts > 0:
            batch = [0]
        # only generation requests
        elif num_generations > 0:
            batch = [1]

        out = []
        for req_type in batch:

            # read conv and ssm states
            split_batch = split_indices[req_type].tolist()
            conv_states = mamba_cache_params.conv_states[
                self.layer_idx][split_batch]
            ssm_states = mamba_cache_params.ssm_states[
                self.layer_idx][split_batch]

            # host request types
            host_request_types = (torch.zeros_like(split_seq_lens[req_type])
                                  if req_type == 0 else torch.ones_like(
                                      split_seq_lens[req_type]))

            # last token ids
            last_token_ids = torch.cumsum(split_seq_lens[req_type],
                                          dim=0,
                                          dtype=torch.int32).cuda()

            # in_proj
            zxbcdt = self.in_proj(split_hidden_states[req_type])

            conv_weight = self.conv1d.weight.unsqueeze(1).permute(
                1, 2, 0).contiguous()
            xbc, conv_states_out = torch.ops.trtllm.mamba_conv1d(
                zxbcdt,
                conv_weight,
                self.conv1d.bias,
                conv_states,
                host_request_types,
                last_token_ids,
                split_seq_lens[req_type],
                self.slot_mapping,
                self.tp_conv_dim,
                self.d_conv,
                self.tp_d_inner,  # pre_stride
                self.tp_nheads,  # post_stride
                self.remove_padding,
                self.apply_silu,
                self.is_paged_state,
            )

            # selective scan
            y, ssm_states_out = torch.ops.trtllm.selective_scan(
                xbc,
                ssm_states,
                zxbcdt,
                self.dt_bias,
                self.A,
                xbc,
                self.D,
                host_request_types,
                last_token_ids,
                zxbcdt,
                split_seq_lens[req_type],
                self.slot_mapping,
                self.tp_d_inner,
                self.d_state,
                self.tp_nheads,
                self.tp_ngroups,
                self.chunk_size,
                self.delta_rank,
                self.delta_softplus,
                self.remove_padding,
                self.is_mamba2,
                self.is_paged_state,
            )

            # group norm
            y = self.norm(y)

            # out_proj
            y = self.out_proj(y)

            # append output
            out.append(y)

            # update conv and ssm states
            for i, idx in enumerate(split_batch):
                mamba_cache_params.conv_states[self.layer_idx][idx].copy_(
                    conv_states_out[i])
                mamba_cache_params.ssm_states[self.layer_idx][idx].copy_(
                    ssm_states_out[i])

        out = torch.cat(out, dim=0)
        return out
