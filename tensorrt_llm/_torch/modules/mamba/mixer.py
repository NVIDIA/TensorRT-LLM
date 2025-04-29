from typing import Optional

import torch
from einops import rearrange, repeat
from torch import nn

from ...attention_backend import AttentionMetadata
from ...model_config import ModelConfig
from ..linear import Linear, TensorParallelMode
from .causal_conv1d import causal_conv1d_update, causal_conv1d_varlen_states
from .layernorm_gated import RMSNorm as RMSNormGated
from .selective_state_update import selective_state_update
from .ssd_combined import mamba_split_conv1d_scan_combined


class MambaMixer(nn.Module):

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
        self.head_dim = head_dim
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
            skip_create_weights_in_init=config.skip_create_weights_in_init,
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
        self.norm = RMSNormGated(
            self.tp_d_inner,
            eps=rms_norm_eps,
            norm_before_gate=False,
            group_size=self.tp_d_inner // self.tp_ngroups,
            dtype=dtype,
        )

        # out_proj
        self.out_proj = Linear(
            d_inner,
            d_model,
            bias=bias,
            dtype=dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=config.get_quant_config(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:

        # warm up does not prepare resources, there are two warmup requests
        is_warmup = attn_metadata.kv_cache_manager is None or attn_metadata.request_ids == [
            0
        ]

        # calculate split size
        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.seq_lens.shape[0] - num_contexts
        sum_seq = torch.cumsum(attn_metadata.seq_lens, dim=0)
        split_ctx = sum_seq[num_contexts - 1] if num_contexts > 0 else 0
        split_gen = sum_seq[-1] - split_ctx
        split_size = [split_ctx, split_gen]

        # handle warm up request
        if not is_warmup:
            state_indices = attn_metadata.kv_cache_manager.get_state_indices()
            split_indices = torch.split(state_indices,
                                        [num_contexts, num_generations])

        split_seq_lens = torch.split(attn_metadata.seq_lens,
                                     [num_contexts, num_generations])

        # in_proj
        zxbcdt = self.in_proj(hidden_states)
        split_zxbcdt = torch.split(zxbcdt, split_size, dim=0)

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

            if not is_warmup:
                indices = split_indices[req_type].to(torch.long).to(
                    torch.device("cuda"))
                conv_states = attn_metadata.kv_cache_manager.get_conv_states(
                    self.layer_idx)
                ssm_states = attn_metadata.kv_cache_manager.get_ssm_states(
                    self.layer_idx)
            else:
                indices = None

            z, xbc, dt = torch.split(
                split_zxbcdt[req_type],
                [self.tp_d_inner, self.tp_conv_dim, self.tp_nheads],
                dim=-1,
            )

            # prefill
            if req_type == 0:

                cu_seqlens = (torch.cat(
                    [
                        torch.zeros(1),
                        torch.cumsum(split_seq_lens[req_type], dim=0)
                    ],
                    dim=0,
                ).to(torch.int32).to(torch.device("cuda")))

                conv_states_out = causal_conv1d_varlen_states(
                    xbc, cu_seqlens, state_len=self.d_conv)

                if not is_warmup:
                    conv_states.index_copy_(0, indices, conv_states_out)

                # Temporary fix to make mamba layer close to original implementation
                ctx_seq_lens = split_seq_lens[req_type].tolist()
                ctx_zxbcdt = torch.split(split_zxbcdt[req_type],
                                         ctx_seq_lens,
                                         dim=0)
                split_y = []
                split_ssm_states = []
                for i in range(len(ctx_zxbcdt)):
                    y, ssm_states_out = mamba_split_conv1d_scan_combined(
                        ctx_zxbcdt[i].unsqueeze(0),
                        self.conv1d.weight.permute(0, 1).contiguous(),
                        self.conv1d.bias,
                        self.dt_bias,
                        self.A,
                        D=self.D,
                        chunk_size=self.chunk_size,
                        activation="silu",
                        headdim=self.head_dim,
                        ngroups=self.tp_ngroups,
                        norm_before_gate=False,
                        initial_states=None,
                        return_final_states=True,
                    )

                    split_y.append(y.squeeze(0))
                    split_ssm_states.append(ssm_states_out)

                y = torch.cat(split_y, dim=0)
                ssm_states_out = torch.cat(split_ssm_states, dim=0)

                # norm
                y = self.norm(y)

            # decode
            else:

                # get conv and ssm states for decode
                if not is_warmup:
                    conv_states_in = conv_states[indices]
                    ssm_states_in = ssm_states[indices]

                # update conv states
                xbc = causal_conv1d_update(
                    xbc,
                    conv_states_in,
                    self.conv1d.weight.permute(0, 1).contiguous(),
                    self.conv1d.bias,
                    "silu",
                )

                # copy new conv states
                if not is_warmup:
                    conv_states.index_copy_(0, indices, conv_states_in)

                x, B, C = torch.split(
                    xbc,
                    [
                        self.tp_d_inner,
                        self.tp_ngroups * self.d_state,
                        self.tp_ngroups * self.d_state,
                    ],
                    dim=-1,
                )

                ssm_states_out = ssm_states_in.transpose(2, 3)

                A = repeat(self.A,
                           "h -> h p n",
                           p=self.head_dim,
                           n=self.d_state).to(dtype=torch.float32)
                dt = repeat(dt, "b h -> b h p", p=self.head_dim)
                dt_bias = repeat(self.dt_bias, "h -> h p", p=self.head_dim)
                D = repeat(self.D, "h -> h p", p=self.head_dim)
                B = rearrange(B, "b (g n) -> b g n", g=self.tp_ngroups)
                C = rearrange(C, "b (g n) -> b g n", g=self.tp_ngroups)
                x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.head_dim)

                y = selective_state_update(
                    ssm_states_out,
                    x_reshaped,
                    dt,
                    A,
                    B,
                    C,
                    D,
                    z=None,
                    dt_bias=dt_bias,
                    dt_softplus=self.delta_softplus,
                )

                y = rearrange(y, "b h p -> b (h p)")

                # gated norm
                y = self.norm(y, z)

            # copy new ssm states
            if not is_warmup:
                ssm_states_out = ssm_states_out.transpose(2, 3)
                ssm_states.index_copy_(0, indices, ssm_states_out)

            # append output
            out.append(y)

        out = torch.cat(out, dim=0)

        # out_proj
        out = self.out_proj(out)

        return out
