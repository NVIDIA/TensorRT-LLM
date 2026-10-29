# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned TP4/TP8 MTP kernels and their pre-profile state.

This is a plain Python object, not an nn.Module: its callbacks may reference the
target without registering a parent module under the draft layer. The target
owns the full-context attention arena; this helper owns only draft operands and
small buffers. All eligibility decisions precede the first draft KV write.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.distributed.ops import _get_mnnvl_workspace_comm
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.utils import EventType

from .ops.comm._mnnvl_workspace import hop_workspace
from .ops.comm.mnnvl_moe_finalize_hop import mnnvl_moe_finalize_hop
from .ops.gemm.draft_hidden_half import draft_hidden_half
from .ops.gemm.low_m_gemm import low_m_gemm
from .ops.gemm.router_gemm import router_gemm
from .ops.moe.lowm_bf16_moe import glm_lowm_bf16_moe
from .ops.spec._draft_exchange import DraftExchangeWorkspace, _all_succeeded
from .ops.spec._draft_token_table import build_draft_token_table
from .ops.spec.draft_ehproj_table import glm_draft_ehproj_table
from .ops.spec.draft_glue_rmsnorm_cat import glm_draft_glue_rmsnorm_cat
from .ops.spec.draft_sample_glue import glm_draft_sample_glue
from .ops.spec.gather_last_accepted_rows import glm_gather_last_accepted_rows

if TYPE_CHECKING:
    from .modeling import DecodeTarget
    from .weights import DraftWeights

DRAFT_SHAPES = {(4, 5): 3, (8, 3): 2, (8, 7): 1}  # (TP, draft length): maximum batch


class FirstDraftRowSelect:
    """Select hidden and residual together after the complete first attention."""

    def __init__(self, batch_indices, num_accepted, tokens_per_request: int) -> None:
        self.batch_indices = batch_indices
        self.num_accepted = num_accepted
        self.tokens_per_request = tokens_per_request
        self.gather_ids = None

    def __call__(self, hidden, residual):
        if self.gather_ids is not None:
            raise RuntimeError("the first draft rows were already selected")
        self.gather_ids, hidden, residual = glm_gather_last_accepted_rows(
            hidden,
            self.batch_indices,
            self.num_accepted,
            self.tokens_per_request,
            0,
            second=residual,
        )
        return hidden, residual


def _bf16_matrix(tensor, shape) -> bool:
    return (
        tensor.dtype == torch.bfloat16
        and tuple(tensor.shape) == shape
        and tensor.is_cuda
        and tensor.is_contiguous()
        and tensor.data_ptr() % 32 == 0
    )


def _ordinary_norm(norm, hidden: int) -> bool:
    return (
        _bf16_matrix(norm.weight, (hidden,))
        and not norm.use_gemma
        and not norm.use_cuda_tile
        and not norm.return_hp_output
        and not norm.is_nvfp4
    )


def _static_layout_supported(decode, layer, lm_head, weights) -> bool:
    mapping = decode.mapping
    config = decode.core.model_config.pretrained_config
    hidden, experts = 6144, 256
    inter = 2048 // mapping.tp_size
    columns = 2 * hidden // mapping.tp_size
    embed = decode.core.embed_tokens.weight
    gate = layer.mlp.gate
    bias = gate.e_score_correction_bias
    norms = (
        layer.enorm,
        layer.hnorm,
        layer.input_layernorm,
        layer.post_attention_layernorm,
        layer.shared_head.norm,
    )
    indexer = layer.self_attn.mqa.indexer
    return (
        mapping.tp_size in (4, 8)
        and mapping.tp_size == mapping.world_size
        and mapping.moe_ep_size == 1
        and mapping.pp_size == mapping.cp_size == 1
        and not mapping.enable_attention_dp
        and config.hidden_size == hidden
        and config.n_routed_experts == experts
        and config.n_group == 1
        and config.num_experts_per_tok == 8
        and config.moe_intermediate_size // mapping.tp_size == inter
        and indexer is not None
        and indexer.mtp_index_share
        and all(_ordinary_norm(norm, hidden) for norm in norms)
        and layer.enorm.variance_epsilon == layer.hnorm.variance_epsilon
        and layer.eh_proj.tp_mode == TensorParallelMode.ROW
        and layer.eh_proj.reduce_output
        and layer.eh_proj.bias is None
        and _bf16_matrix(layer.eh_proj.weight, (hidden, columns))
        and (mapping.tp_size == 4 or _bf16_matrix(weights.eh_proj, (hidden, 2 * hidden)))
        and _bf16_matrix(weights.w3_w1, (experts, 2 * inter, hidden))
        and _bf16_matrix(weights.w2, (experts, hidden, inter))
        and _bf16_matrix(gate.weight, (experts, hidden))
        and bias.dtype in (torch.float32, torch.bfloat16)
        and bias.is_cuda
        and bias.is_contiguous()
        and tuple(bias.shape) == (experts,)
        and lm_head.tp_mode == TensorParallelMode.COLUMN
        and getattr(lm_head, "padding_size", 1) == 0
        and lm_head.weight.shape[0] % 8 == 0
        and lm_head.weight.shape[0] * mapping.tp_size == lm_head.num_embeddings
        and _bf16_matrix(lm_head.weight, (lm_head.num_embeddings // mapping.tp_size, hidden))
        and _bf16_matrix(embed, (lm_head.num_embeddings, hidden))
    )


class SmallBatchDraftKernels:
    """TP4 B<=3/K5 or TP8 B<=2/K3 and B1/K7 state, allocated before profiling."""

    MAX_ROWS = 24

    def __init__(
        self,
        decode,
        layer,
        lm_head,
        weights,
        sample_exchange,
        table_exchange=None,
        *,
        draft_len: int,
    ) -> None:
        self.decode = decode
        self.layer = layer
        self.lm_head = lm_head
        self.sample_exchange = sample_exchange
        self.table_exchange = table_exchange
        self.use_table = table_exchange is not None
        self.DRAFT_LEN = draft_len
        self.MAX_BATCH = DRAFT_SHAPES[decode.mapping.tp_size, draft_len]
        self.embed_table = decode.core.embed_tokens.weight.detach()
        self.enorm_weight = layer.enorm.weight.detach()
        self.hnorm_weight = layer.hnorm.weight.detach()
        self.w3_w1, self.w2 = weights.w3_w1, weights.w2
        config = decode.core.model_config.pretrained_config
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        self.hidden_size = config.hidden_size
        columns = 2 * self.hidden_size // decode.mapping.tp_size
        self.window = (decode.mapping.tp_rank * columns, (decode.mapping.tp_rank + 1) * columns)
        self.pdl = get_env_enable_pdl()
        self.zero_residual = torch.zeros(
            (self.MAX_ROWS, self.hidden_size), dtype=torch.bfloat16, device=self.embed_table.device
        )
        # Never alias the target's full-row TopK output during accepted-row compaction.
        self.topk_stash = torch.empty(
            (self.MAX_BATCH, config.index_topk), dtype=torch.int32, device=self.embed_table.device
        )
        if self.use_table:
            self.token_table, self.hidden_rows = build_draft_token_table(
                weights.eh_proj,
                self.embed_table,
                self.enorm_weight,
                layer.enorm.variance_epsilon,
                decode.mapping.tp_rank,
                decode.mapping.tp_size,
                lm_head.weight.shape[0],
            )
            self.half_start = torch.cuda.Event()
            self.half_done = torch.cuda.Event()
            self.identity_routing = {}
            for rows in range(1, self.MAX_BATCH + 1):
                expert_weights = torch.zeros((rows, self.top_k), dtype=torch.bfloat16)
                expert_rows = torch.full((rows, self.top_k), -1, dtype=torch.int32)
                expert_weights[:, 0] = 1
                expert_rows[:, 0] = torch.arange(rows, dtype=torch.int32)
                self.identity_routing[rows] = (
                    expert_weights.to(self.embed_table.device),
                    expert_rows.to(self.embed_table.device),
                )
        mla = layer.self_attn
        self.l2_after_fused_a = [
            tensor
            for tensor in (decode._q_b_absorbed[layer.layer_idx], mla.v_b_proj, mla.o_proj.weight)
            if tensor.is_contiguous()
        ]

    def supports(self, metadata, first_rows: int, batch_size: int) -> bool:
        return (
            1 <= batch_size <= self.MAX_BATCH
            and first_rows == batch_size * (self.DRAFT_LEN + 1)
            and first_rows <= self.MAX_ROWS
            and self.decode.supports_small_batch_draft(metadata, first_rows, batch_size, self.layer)
        )

    def forward_step(
        self,
        input_ids,
        position_ids,
        hidden_states,
        attn_metadata,
        *,
        eh_proj_input=None,
        eh_proj_output=None,
        row_select=None,
    ):
        """Full first attention, optional row selection, then the B-row BF16 MoE."""
        if eh_proj_input is not None and eh_proj_output is not None:
            raise ValueError("eh_proj_input and eh_proj_output are mutually exclusive")
        layer = self.layer
        rows = hidden_states.shape[0]
        if eh_proj_output is None:
            if eh_proj_input is None:
                eh_proj_input = glm_draft_glue_rmsnorm_cat(
                    self.embed_table,
                    input_ids,
                    hidden_states,
                    self.enorm_weight,
                    self.hnorm_weight,
                    layer.enorm.variance_epsilon,
                    *self.window,
                    self.pdl,
                )
            partial = low_m_gemm(
                eh_proj_input,
                layer.eh_proj.weight,
                eh_proj_input.new_empty((rows, self.hidden_size)),
            )
            hidden_states, residual = self.decode._hop(
                layer.input_layernorm, partial, self.zero_residual[:rows]
            )
        else:
            hidden_states, residual = eh_proj_output
        attn_out = self.decode._attention(
            layer, position_ids, hidden_states, attn_metadata, False, draft_state=self
        )
        hidden_states, residual = self.decode._post_attention_hop(layer, attn_out, residual)
        if row_select is not None:
            hidden_states, residual = row_select(hidden_states, residual)
        return self._moe_postnorm(hidden_states, residual)

    def _moe_postnorm(self, hidden_states, residual):
        mlp = self.layer.mlp
        gate = mlp.gate

        def routed():
            logits = router_gemm(hidden_states, gate.weight)
            return glm_lowm_bf16_moe(
                hidden_states,
                logits,
                gate.e_score_correction_bias,
                self.w3_w1,
                self.w2,
                self.top_k,
                self.routed_scaling_factor,
            )

        routed_output, shared_output = maybe_execute_in_parallel(
            routed,
            lambda: self.decode._shared_expert(mlp, hidden_states),
            mlp.event_dict[EventType.Main],
            mlp.event_dict[EventType.MoeShared],
            mlp.aux_stream,
        )
        if self.use_table:
            rows = hidden_states.shape[0]
            expert_weights, expert_rows = self.identity_routing[rows]
            comm_buffer, buffer_flags = hop_workspace(self.decode.mapping, rows, self.hidden_size)
            norm = self.layer.shared_head.norm
            normed, _ = mnnvl_moe_finalize_hop(
                routed_output,
                expert_weights,
                expert_rows,
                shared_output,
                residual,
                norm.weight,
                norm.variance_epsilon,
                comm_buffer,
                buffer_flags,
            )
            return normed
        # Match the plain TP4 path's BF16 local sum, then FP32 rank reduction.
        # Fake identity-routing finalize would introduce different rank rounding.
        partial = routed_output + shared_output
        normed, _ = self.decode._hop(self.layer.shared_head.norm, partial, residual)
        return normed

    def head(self, hidden_states, *, next_step=False):
        # The layer already applied shared_head.norm; do not normalize again.
        if not (self.use_table and next_step):
            return self.decode._lm_head_shard(self.lm_head, hidden_states), None
        rows = hidden_states.shape[0]
        hidden_half = torch.empty(
            (rows, self.hidden_rows.shape[0]), dtype=torch.float32, device=hidden_states.device
        )
        self.half_start.record()
        # B2 launches its LM head first; B1 folds hnorm into the hidden-half GEMM.
        logits = self.decode._lm_head_shard(self.lm_head, hidden_states) if rows > 1 else None
        aux_stream = self.layer.mlp.aux_stream
        with torch.cuda.stream(aux_stream):
            aux_stream.wait_event(self.half_start)
            normed = hidden_states if rows == 1 else self.layer.hnorm(hidden_states)
            draft_hidden_half(
                normed,
                self.hidden_rows,
                hidden_half,
                self.hnorm_weight if rows == 1 else None,
                self.layer.hnorm.variance_epsilon if rows == 1 else None,
            )
            self.half_done.record()
        if logits is None:
            logits = self.decode._lm_head_shard(self.lm_head, hidden_states)
        return logits, hidden_half

    def sample_step(
        self,
        logits,
        hidden_states,
        draft_tokens,
        step,
        positions,
        gather_ids,
        seq_lens,
        kv_lens,
        num_accepted,
        accepted_tokens,
        batch_indices,
        *,
        hidden_half=None,
    ):
        """Write the token and lengths once, with optional next-step glue/output rows."""
        rows, draft_len = draft_tokens.shape
        last = step + 1 == draft_len
        first = step == 0
        if self.use_table and not last:
            norm = self.layer.input_layernorm
            normed = hidden_states.new_empty((rows, self.hidden_size))
            residual = torch.empty_like(normed)
            next_positions = positions.new_empty((rows,))
            torch.cuda.current_stream().wait_event(self.half_done)
            glm_draft_ehproj_table(
                logits,
                self.table_exchange.buffer,
                self.table_exchange.flags,
                self.decode.mapping.tp_size,
                draft_tokens,
                step,
                hidden_half,
                self.token_table,
                norm.weight,
                norm.variance_epsilon,
                normed,
                residual,
                positions,
                next_positions,
                gather_ids,
                seq_lens,
                first,
                kv_lens,
                2 if first else 1,
                num_accepted if first else None,
                draft_len,
                None,
                [],
                None,
                None,
                None,
                None,
                None,
            )
            return (normed, residual), next_positions, None
        glue = None if last else hidden_states.new_empty((rows, self.window[1] - self.window[0]))
        next_positions = None if last else positions.new_empty((rows,))
        next_new_tokens = draft_tokens.new_empty((rows, draft_len + 1)) if last else None
        glm_draft_sample_glue(
            logits,
            self.sample_exchange.buffer,
            self.sample_exchange.flags,
            self.decode.mapping.tp_size,
            draft_tokens,
            step,
            None if last else self.embed_table,
            None if last else hidden_states,
            None if last else self.enorm_weight,
            None if last else self.hnorm_weight,
            0.0 if last else self.layer.enorm.variance_epsilon,
            0 if last else self.window[0],
            0 if last else self.window[1],
            glue,
            None if last else positions,
            next_positions,
            None if last else gather_ids,
            seq_lens,
            first,
            kv_lens,
            2 if first else 1,
            num_accepted if first else None,
            draft_len,
            # Current DSA metadata owns these derived outputs and refreshes them below.
            None,
            [],
            None,
            None,
            None,
            None,
            None,
            accepted_tokens if last else None,
            batch_indices if last else None,
            num_accepted if last else None,
            next_new_tokens,
        )
        return glue, next_positions, next_new_tokens


@torch.no_grad()
def build_small_batch_draft(
    decode: DecodeTarget, layer, lm_head, weights: DraftWeights
) -> SmallBatchDraftKernels | None:
    """Construct the helper before KV profiling, with group-uniform setup failure."""
    if decode.mapping.tp_size not in (4, 8):
        return None
    comm = _get_mnnvl_workspace_comm(decode.mapping)
    eligible = False
    draft_len = None
    try:
        spec = decode.core.model_config.spec_config
        draft_len = None if spec is None else spec.max_draft_len
        eligible = bool(DRAFT_SHAPES.get((decode.mapping.tp_size, draft_len), 0)) and (
            _static_layout_supported(decode, layer, lm_head, weights)
        )
    except Exception:
        # A rank-local layout/inspection failure must still reach the vote.
        pass
    if not _all_succeeded(comm, eligible):
        return None
    # The collective workspace completes on all ranks before local allocations.
    table_exchange = (
        DraftExchangeWorkspace.for_ehproj_table(decode.mapping)
        if decode.mapping.tp_size == 8
        else None
    )
    sample_exchange = DraftExchangeWorkspace.for_sample_glue(decode.mapping)
    error, state = None, None
    try:
        state = SmallBatchDraftKernels(
            decode, layer, lm_head, weights, sample_exchange, table_exchange, draft_len=draft_len
        )
        torch.cuda.synchronize()
    except Exception as exc:
        error = exc
    if not _all_succeeded(comm, error is None):
        raise RuntimeError("small-batch MTP setup failed on at least one rank") from error
    return state
