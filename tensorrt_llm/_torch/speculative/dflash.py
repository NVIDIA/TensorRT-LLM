# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
from torch import nn

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from ..pyexecutor.mamba_cache_manager import MambaHybridCacheManager
from ..pyexecutor.resource_manager import BaseResourceManager
from .interface import SpecMetadata, SpecWorkerBase

if TYPE_CHECKING:
    from ...llmapi.llm_args import DFlashDecodingConfig


@dataclass
class DFlashSpecMetadata(SpecMetadata):
    """Metadata for DFlash speculative decoding.

    Captures hidden states from specific target model layers during the target
    forward pass, which are then projected through fc + hidden_norm and fed to
    the DFlash draft model as cross-attention context.
    """

    batch_indices_cuda: Optional[torch.Tensor] = None
    spec_resource_manager: Optional[BaseResourceManager] = None

    # Hidden state capture fields
    layers_to_capture: Optional[List[int]] = None
    hidden_size: int = 0
    max_num_tokens: int = 0
    dtype: torch.dtype = torch.bfloat16
    captured_hidden_states: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.batch_indices_cuda = torch.empty(
            [self.max_num_requests],
            dtype=torch.int,
            device="cuda",
        )

        self.is_spec_dec_tree = False
        self.is_spec_dec_dynamic_tree = False

        # Set up hidden state capture buffer
        if self.layers_to_capture is not None and len(self.layers_to_capture) > 0:
            self.layers_to_capture = sorted(list(self.layers_to_capture))
            self.num_capture_layers = len(self.layers_to_capture)
            # O(1) lookups for is_layer_capture() and maybe_capture_hidden_states()
            self._capture_layer_set = frozenset(self.layers_to_capture)
            self._layer_to_idx = {lid: i for i, lid in enumerate(self.layers_to_capture)}
            self.captured_hidden_states = torch.empty(
                (self.max_num_tokens, self.hidden_size * self.num_capture_layers),
                dtype=self.dtype,
                device="cuda",
            )
            logger.info(
                f"DFlash: capturing hidden states from layers {self.layers_to_capture}, "
                f"buffer shape {self.captured_hidden_states.shape}"
            )
        else:
            self.num_capture_layers = 0
            self._capture_layer_set = frozenset()
            self._layer_to_idx = {}

    def prepare(self):
        assert self.request_ids is not None

        num_seqs = len(self.request_ids)
        batch_indices = torch.arange(
            num_seqs, dtype=torch.int, device="cpu", pin_memory=prefer_pinned()
        )
        self.batch_indices_cuda[:num_seqs].copy_(batch_indices, non_blocking=True)

        # Update slot mapping for DFlash context buffers
        worker = getattr(self, "_dflash_worker", None)
        if worker is not None and worker._ctx_buf_inited:
            current = set(self.request_ids)
            for rid in list(worker._req_to_slot.keys()):
                if rid not in current:
                    slot = worker._req_to_slot.pop(rid)
                    worker._ctx_len[slot] = 0
                    worker._free_slots.append(slot)

            # Default to slot 0 for unknown request IDs (e.g. during warmup
            # where synthetic requests may not have assigned slots).
            mapping = torch.tensor(
                [worker._req_to_slot.get(rid, 0) for rid in self.request_ids],
                dtype=torch.long,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            worker._batch_to_slot[:num_seqs].copy_(mapping, non_blocking=True)

    def is_layer_capture(self, layer_id: int) -> bool:
        return layer_id in self._capture_layer_set

    def maybe_capture_hidden_states(
        self, layer_id: int, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> None:
        """Capture hidden states from a target model layer into the buffer."""
        if self.captured_hidden_states is None:
            return
        i = self._layer_to_idx.get(layer_id)
        if i is not None:
            num_tokens = hidden_states.shape[0]
            to_save = hidden_states + residual if residual is not None else hidden_states
            self.captured_hidden_states[
                :num_tokens, i * self.hidden_size : (i + 1) * self.hidden_size
            ].copy_(to_save, non_blocking=True)

    def get_hidden_states(self, num_tokens: int) -> Optional[torch.Tensor]:
        """Get captured hidden states (all layers concatenated)."""
        if self.captured_hidden_states is None:
            return None
        return self.captured_hidden_states[
            :num_tokens, : self.hidden_size * self.num_capture_layers
        ]


class DFlashWorker(SpecWorkerBase):
    """
    Worker for DFlash speculative decoding.

    DFlash uses the draft model with mask tokens to predict multiple
    draft tokens in parallel. The DFlash draft model uses cross-attention
    to target model hidden states captured from specific layers.

    The target features are projected through fc + hidden_norm and fed
    to the draft model as K/V context in cross-attention. The context
    accumulates across steps - each step adds newly accepted tokens'
    projected hidden states to a per-request buffer, giving the draft
    model the full history of target features.

    Reference: https://arxiv.org/pdf/2602.06036
    """

    def __init__(
        self,
        spec_config: "DFlashDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(use_separate_draft_kv_cache)
        self.spec_config = spec_config
        self.mapping = mapping
        self._mask_embed = None  # Cached mask token embedding (CUDA-graph safe)
        self._resolved_mask_token_id = None
        self._resolved_block_size = None

        # Pre-allocated context buffers (lazy-init on first forward).
        # Replaces per-request Python dicts with fixed-size CUDA buffers
        # indexed by slot, enabling CUDA graph compatibility.
        self._ctx_buf_inited = False
        self._ctx_buf = None  # [max_batch, max_ctx, proj_dim]
        self._ctx_pos_buf = None  # [max_batch, max_ctx]
        self._ctx_len = None  # [max_batch] actual context length per slot
        self._batch_to_slot = None  # [max_batch] maps batch pos -> buffer slot
        self._max_ctx = 0
        self._proj_dim = 0

        # Slot management (Python, updated in prepare() and eager mode)
        self._req_to_slot = {}  # request_id -> slot index
        self._free_slots = deque()  # available slot indices

        logger.info(
            f"DFlashWorker initialized with use_separate_draft_kv_cache={use_separate_draft_kv_cache}"
        )

    @property
    def max_draft_len(self) -> int:
        return self.spec_config.max_draft_len

    @property
    def _draft_tokens_per_req(self) -> int:
        """Total tokens per gen request in the draft forward.

        Uses 2K to fit all accepted tokens (up to K+1) plus K-1 mask tokens,
        ensuring K unique predictions regardless of how many tokens were accepted.
        """
        return 2 * self.max_draft_len

    def _lazy_init_ctx_buffers(self, draft_model, spec_metadata):
        """Initialize pre-allocated context buffers on first forward."""
        if self._ctx_buf_inited:
            return

        max_batch = spec_metadata.max_num_requests

        if hasattr(draft_model, "fc"):
            self._proj_dim = draft_model.fc.weight.shape[0]
        else:
            self._proj_dim = spec_metadata.hidden_size

        config_max_ctx = getattr(self.spec_config, "max_ctx_len", None)
        if config_max_ctx is not None:
            self._max_ctx = config_max_ctx
        else:
            config = getattr(draft_model, "config", None)
            max_pos = getattr(config, "max_position_embeddings", 8192) if config else 8192
            self._max_ctx = min(max_pos, 8192)

        dtype = draft_model.fc.weight.dtype if hasattr(draft_model, "fc") else torch.bfloat16

        self._ctx_buf = torch.zeros(
            (max_batch, self._max_ctx, self._proj_dim), dtype=dtype, device="cuda"
        )
        self._ctx_pos_buf = torch.zeros((max_batch, self._max_ctx), dtype=torch.long, device="cuda")
        self._ctx_len = torch.zeros(max_batch, dtype=torch.long, device="cuda")
        self._batch_to_slot = torch.zeros(max_batch, dtype=torch.long, device="cuda")

        self._free_slots = deque(range(max_batch))
        self._req_to_slot = {}
        self._ctx_buf_inited = True

        logger.info(
            f"DFlash: allocated ctx buffers: max_batch={max_batch}, "
            f"max_ctx={self._max_ctx}, proj_dim={self._proj_dim}, dtype={dtype}"
        )

    def _prepare_attn_metadata_for_dflash(self, attn_metadata, spec_metadata):
        """Save attn_metadata fields that DFlash modifies during forward."""
        is_capturing = torch.cuda.is_current_stream_capturing()

        if spec_metadata.is_cuda_graph and not is_capturing:
            attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda", "kv_lens_cuda")
        else:
            attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")

    def _prepare_kv_for_draft_forward(
        self,
        attn_metadata,
        num_accepted_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
    ):
        """Adjust kv_lens_cuda so the draft model sees correct RoPE positions."""
        if hasattr(attn_metadata, "kv_lens_cuda"):
            self._kv_rewind_amount = 1 - num_accepted_tokens[num_contexts:batch_size]
            self._kv_rewind_nc = num_contexts
            self._kv_rewind_bs = batch_size

            if batch_size > num_contexts:
                attn_metadata.kv_lens_cuda[num_contexts:batch_size] += 1

            attn_metadata.update_for_spec_dec()

    def _apply_kv_rewind_after_draft(self, attn_metadata, spec_metadata):
        """Apply the deferred kv_lens rewind after the draft forward."""
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if is_warmup:
            return

        if hasattr(self, "_kv_rewind_amount") and hasattr(attn_metadata, "kv_lens_cuda"):
            nc = self._kv_rewind_nc
            bs = self._kv_rewind_bs
            attn_metadata.kv_lens_cuda[nc:bs] -= self._kv_rewind_amount
            attn_metadata.kv_lens_cuda[nc:bs].clamp_(min=0)

    def _store_prefill_context(
        self,
        draft_model,
        spec_metadata: "DFlashSpecMetadata",
        attn_metadata,
        position_ids: torch.Tensor,
        total_target_tokens: int,
    ):
        """Capture prefill hidden states and store as initial accumulated context.

        During prefill (context requests), the target model processes all prompt
        tokens.  We project their captured hidden states through fc + hidden_norm
        and store them per-request so the draft model can use the full prompt
        as cross-attention context on subsequent gen steps.
        """
        if not hasattr(draft_model, "fc") or not hasattr(draft_model, "hidden_norm"):
            return

        num_ctx_tokens = attn_metadata.num_ctx_tokens
        if num_ctx_tokens == 0:
            return

        captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
        if captured_hs is None:
            return

        # Project context tokens through fc + hidden_norm
        ctx_hs = captured_hs[:num_ctx_tokens]
        ctx_proj = draft_model.fc(ctx_hs.to(draft_model.fc.weight.dtype))
        ctx_proj = draft_model.hidden_norm(ctx_proj)

        # Split by request and store/append accumulated context.
        # Context requests may arrive in chunks (chunked prefill), so we
        # must APPEND successive chunks for the same request rather than
        # overwriting.  If a previously-finished request id is reused for a
        # brand-new request, the new chunk's first position will be 0, which
        # signals a fresh start → replace instead of append.
        offset = 0
        num_contexts = attn_metadata.num_contexts
        for i in range(num_contexts):
            req_id = spec_metadata.request_ids[i]
            slen = int(attn_metadata._seq_lens[i])
            chunk_proj = ctx_proj[offset : offset + slen].detach()
            chunk_pos = position_ids[offset : offset + slen].long().detach()

            first_pos = chunk_pos[0].item() if slen > 0 else 0

            # Assign slot for new requests or reset for reused IDs
            if req_id not in self._req_to_slot or first_pos == 0:
                if req_id in self._req_to_slot:
                    old_slot = self._req_to_slot[req_id]
                    self._ctx_len[old_slot] = 0
                    self._free_slots.append(old_slot)
                if not self._free_slots:
                    logger.warning("DFlash: no free slots, skipping context store")
                    offset += slen
                    continue
                slot = self._free_slots.popleft()
                self._req_to_slot[req_id] = slot
                self._ctx_len[slot] = 0

            slot = self._req_to_slot[req_id]
            cur = int(self._ctx_len[slot].item())
            end = min(cur + slen, self._max_ctx)
            actual = end - cur
            if actual > 0:
                self._ctx_buf[slot, cur:end] = chunk_proj[:actual].to(self._ctx_buf.dtype)
                self._ctx_pos_buf[slot, cur:end] = chunk_pos[:actual]
                self._ctx_len[slot] = end
            offset += slen

    def forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
    ):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        raw_logits = logits
        K = self.max_draft_len

        # Lazy init buffers and attach worker reference for prepare()
        self._lazy_init_ctx_buffers(draft_model, spec_metadata)
        spec_metadata._dflash_worker = self

        # Save context lengths before warmup to prevent accumulation
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if is_warmup:
            saved_ctx_len = self._ctx_len.clone()

        self._execute_guided_decoder_if_present(logits)

        # draft_tokens buffer has (2K-1) entries per gen request; extract the K real drafts
        if num_gens > 0:
            draft_tokens_raw = spec_metadata.draft_tokens
            draft_tokens = draft_tokens_raw.reshape(num_gens, 2 * K - 1)[:, :K]
        else:
            draft_tokens = spec_metadata.draft_tokens.reshape(0, K)

        # logits have 2K entries per gen request; extract K+1 for acceptance
        if num_gens > 0:
            ctx_logits = logits[:num_contexts]
            vocab_size = logits.shape[-1]
            gen_logits_2k = logits[num_contexts:].reshape(num_gens, 2 * K, vocab_size)
            gen_logits_kp1 = gen_logits_2k[:, : K + 1, :].reshape(-1, vocab_size)
            logits_for_accept = torch.cat([ctx_logits, gen_logits_kp1], dim=0)
        else:
            logits_for_accept = logits

        accepted_tokens, num_accepted_tokens = self._sample_and_accept_draft_tokens_base(
            logits_for_accept, draft_tokens, num_contexts, batch_size, spec_metadata
        )

        # Update GDN/Mamba recurrent states to the accepted token's state.
        if num_gens > 0 and isinstance(attn_metadata.kv_cache_manager, MambaHybridCacheManager):
            attn_metadata.kv_cache_manager.update_mamba_states(
                attn_metadata=attn_metadata,
                num_accepted_tokens=num_accepted_tokens,
                state_indices=attn_metadata.mamba_metadata.state_indices,
            )

        # Pad accepted_tokens from (batch, K+1) to (batch, 2K) to match sampler buffer
        if K > 1:
            acc_padding = torch.zeros(
                (batch_size, K - 1), dtype=accepted_tokens.dtype, device=accepted_tokens.device
            )
            accepted_tokens = torch.cat([accepted_tokens, acc_padding], dim=1)

        self._prepare_attn_metadata_for_dflash(attn_metadata, spec_metadata)
        self._prepare_kv_for_draft_forward(
            attn_metadata, num_accepted_tokens, num_contexts, batch_size
        )

        # Collapse mrope [3, 1, N] to 1D by taking the first (temporal) dimension.
        # The draft model uses standard 1D RoPE, so only scalar positions are needed.
        if position_ids.ndim == 3:
            position_ids = position_ids[0, 0]
        else:
            position_ids = position_ids.squeeze(0)

        # Get total tokens processed by target model (for hidden state extraction)
        total_target_tokens = input_ids.shape[0]

        # Capture prefill (context) hidden states for future gen steps.
        # This gives the draft model the full prompt context, not just gen tokens.
        if num_contexts > 0:
            self._store_prefill_context(
                draft_model, spec_metadata, attn_metadata, position_ids, total_target_tokens
            )
            # Rebuild batch_to_slot after prefill assigns new slots
            if self._ctx_buf_inited and spec_metadata.request_ids:
                num_seqs = len(spec_metadata.request_ids)
                mapping = [self._req_to_slot.get(rid, 0) for rid in spec_metadata.request_ids]
                self._batch_to_slot[:num_seqs].copy_(
                    torch.tensor(mapping, dtype=torch.long, device="cuda")
                )

        inputs = self.prepare_1st_drafter_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            accepted_tokens=accepted_tokens,
            num_accepted_tokens=num_accepted_tokens,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=draft_model,
            total_target_tokens=total_target_tokens,
        )

        draft_kv_cache_manager = self.get_draft_kv_cache_manager(resource_manager)

        if num_gens > 0:
            with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
                # Use custom dflash_forward with cross-attention
                hidden_states_out = draft_model.dflash_forward(
                    noise_embedding=inputs["noise_embedding"],
                    target_hidden=inputs["target_hidden"],
                    query_positions=inputs["query_positions"],
                    context_positions=inputs["context_positions"],
                    num_ctx_per_req=inputs["num_ctx_per_req"],
                )

                # Gather K logits per gen request from mask positions (1..K).
                # hidden_states_out is flat: [num_gens * block_size, hidden_dim]
                block_size = self._resolved_block_size
                request_bases = torch.arange(num_gens, dtype=torch.long, device="cuda") * block_size
                offsets = torch.arange(K, dtype=torch.long, device="cuda")
                # Masks are at positions 1..K in each request's block_size output
                gen_gather_ids = (request_bases.unsqueeze(1) + 1 + offsets.unsqueeze(0)).flatten()
                gen_gather_ids = gen_gather_ids.clamp(max=hidden_states_out.shape[0] - 1)

                gen_logits = draft_model.logits_processor(
                    hidden_states_out[gen_gather_ids], draft_model.lm_head, attn_metadata, True
                )

                vocab_size = gen_logits.shape[-1]
                gen_logits = gen_logits.reshape(num_gens, self.max_draft_len, vocab_size)

                d2t = getattr(draft_model.model, "d2t", None)
                gen_draft_tokens = torch.argmax(gen_logits, dim=-1, keepdim=False).long()

                if d2t is not None:
                    gen_draft_tokens = d2t[gen_draft_tokens] + gen_draft_tokens

                gen_draft_tokens = gen_draft_tokens.type(torch.int32)

                # Pad from (num_gens, K) to (num_gens, 2K-1).
                if K > 1:
                    pad = torch.zeros((num_gens, K - 1), dtype=torch.int32, device="cuda")
                    gen_draft_tokens = torch.cat([gen_draft_tokens, pad], dim=1)

        else:
            gen_draft_tokens = torch.empty((0, 2 * K - 1), dtype=torch.int32, device="cuda")

        if num_contexts > 0 and num_gens > 0:
            ctx_draft_tokens = torch.zeros(
                (num_contexts, 2 * K - 1), dtype=torch.int32, device="cuda"
            )
            next_draft_tokens = torch.cat([ctx_draft_tokens, gen_draft_tokens], dim=0)
        elif num_contexts > 0:
            next_draft_tokens = torch.zeros(
                (num_contexts, 2 * K - 1), dtype=torch.int32, device="cuda"
            )
        else:
            next_draft_tokens = gen_draft_tokens

        self._restore_attn_metadata_from_spec_dec(attn_metadata)
        self._apply_kv_rewind_after_draft(attn_metadata, spec_metadata)

        next_new_tokens = self._prepare_next_new_tokens(
            accepted_tokens,
            next_draft_tokens,
            spec_metadata.batch_indices_cuda,
            batch_size,
            num_accepted_tokens,
        )

        # Restore context lengths after warmup
        if is_warmup:
            self._ctx_len.copy_(saved_ctx_len)

        return {
            "logits": raw_logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }

    def prepare_1st_drafter_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: DFlashSpecMetadata,
        draft_model: nn.Module,
        total_target_tokens: int = 0,
    ):
        """
        Prepare inputs for DFlash draft model with proper cross-attention.

        For gen requests, builds:
        - noise_embedding: token embeddings for [accepted_tokens + mask_tokens] (2K per req)
        - target_hidden: projected target features at accepted positions (context for cross-attn)
        - query_positions: position IDs for all 2K query tokens
        - context_positions: position IDs for the accepted (context) tokens
        - num_ctx_per_req: per-request context token counts

        The target_hidden stays CONSTANT across layers and provides K/V context.
        """
        num_contexts = attn_metadata.num_contexts
        batch_size = attn_metadata.num_seqs
        num_gens = batch_size - num_contexts

        # Resolve mask_token_id and block_size once, cache for subsequent calls
        if self._resolved_mask_token_id is None:
            if (
                hasattr(self.spec_config, "mask_token_id")
                and self.spec_config.mask_token_id is not None
            ):
                self._resolved_mask_token_id = self.spec_config.mask_token_id
            elif hasattr(draft_model, "mask_token_id"):
                self._resolved_mask_token_id = draft_model.mask_token_id
            elif hasattr(draft_model.model, "mask_token_id"):
                self._resolved_mask_token_id = draft_model.model.mask_token_id
            else:
                raise ValueError(
                    "DFlash requires mask_token_id to be set. Please set it in DFlashDecodingConfig "
                    "or ensure the draft model config has 'dflash_config.mask_token_id' or 'mask_token_id'."
                )
        mask_token_id = self._resolved_mask_token_id

        if self._resolved_block_size is None:
            self._resolved_block_size = getattr(draft_model, "block_size", None) or (
                self.max_draft_len + 1
            )

        # Get the embed_tokens layer from the draft model
        embed_tokens = draft_model.draft_model_full.model.embed_tokens
        hidden_dim = (
            spec_metadata.hidden_size if spec_metadata.hidden_size > 0 else hidden_states.shape[-1]
        )

        if num_gens > 0:
            gen_num_accepted = num_accepted_tokens[num_contexts : num_contexts + num_gens]
            gen_accepted_tokens = accepted_tokens[num_contexts : num_contexts + num_gens, :]

            total_tokens_per_req = self._draft_tokens_per_req  # 2K
            K = self.max_draft_len

            # Get captured multi-layer hidden states from spec_metadata
            captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
            has_target_features = (
                captured_hs is not None
                and hasattr(draft_model, "fc")
                and hasattr(draft_model, "hidden_norm")
            )

            # Use cached block_size (resolved once on first call)
            block_size = self._resolved_block_size
            query_tokens_per_req = block_size

            if self._mask_embed is None:
                mask_token = torch.tensor([mask_token_id], dtype=torch.long, device="cuda")
                self._mask_embed = embed_tokens(mask_token).detach()

            # Start with all mask embeddings for block_size tokens
            noise_embed_2d = self._mask_embed.expand(
                num_gens * query_tokens_per_req, hidden_dim
            ).clone()
            noise_embed_2d = noise_embed_2d.reshape(num_gens, query_tokens_per_req, hidden_dim)

            # Fill position 0 (bonus) with embedding of last accepted token
            bonus_indices = (gen_num_accepted - 1).long().clamp(min=0).unsqueeze(1)
            bonus_token_ids = gen_accepted_tokens.gather(1, bonus_indices).squeeze(1)
            bonus_embeds = embed_tokens(bonus_token_ids.long())
            noise_embed_2d[:, 0, :] = bonus_embeds.to(noise_embed_2d.dtype)

            # Get slots for gen requests from pre-computed mapping
            slots = self._batch_to_slot[num_contexts : num_contexts + num_gens]

            # Position starts from accumulated context length (BEFORE
            # adding current step's features).
            gen_pos_starts = self._ctx_len[slots].int()

            # Context positions for storing new accepted tokens (reused below)
            offsets_kp1 = torch.arange(K + 1, dtype=torch.long, device="cuda")
            ctx_position_ids = (
                gen_pos_starts.unsqueeze(1) + offsets_kp1.unsqueeze(0).int()
            )  # [num_gens, K+1]

            # Query positions start AFTER the full context including current
            # step's accepted features
            gen_query_start = gen_pos_starts + gen_num_accepted.to(torch.int32)
            query_offsets = torch.arange(query_tokens_per_req, dtype=torch.int32, device="cuda")
            query_position_ids = gen_query_start.unsqueeze(1) + query_offsets.unsqueeze(0)

            # Accumulate new accepted features into context buffers
            if has_target_features:
                gen_start = attn_metadata.num_ctx_tokens
                gen_hs = captured_hs[gen_start : gen_start + num_gens * total_tokens_per_req]
                gen_hs = gen_hs.reshape(num_gens, total_tokens_per_req, -1)

                # Only project K+1 tokens (the accepted ones), not all 2K
                gen_hs_to_project = gen_hs[:, : K + 1, :].reshape(-1, gen_hs.shape[-1])
                projected_to_store = draft_model.fc(
                    gen_hs_to_project.to(draft_model.fc.weight.dtype)
                )
                projected_to_store = draft_model.hidden_norm(projected_to_store)
                projected_to_store = projected_to_store.reshape(num_gens, K + 1, -1)
                gen_num_accepted_long = gen_num_accepted.long()
                col_idx = self._ctx_len[slots].unsqueeze(1) + offsets_kp1.unsqueeze(0)
                write_mask = offsets_kp1.unsqueeze(0) < gen_num_accepted_long.unsqueeze(1)
                col_idx = col_idx.clamp(max=self._max_ctx - 1)

                # Fixed-size writes for CUDA graph compatibility:
                # Write ALL entries but zero out invalid ones. Invalid
                # writes land at clamped column indices beyond valid
                # _ctx_len range, so they're harmless.
                slot_flat = slots.unsqueeze(1).expand(-1, K + 1).reshape(-1)
                col_flat = col_idx.reshape(-1)
                proj_flat = projected_to_store.reshape(-1, self._proj_dim)
                pos_flat = ctx_position_ids.long().reshape(-1)

                mask_1d = write_mask.reshape(-1)
                proj_masked = proj_flat * mask_1d.unsqueeze(1).to(proj_flat.dtype)
                pos_masked = pos_flat * mask_1d.long()

                self._ctx_buf[slot_flat, col_flat] = proj_masked.to(self._ctx_buf.dtype)
                self._ctx_pos_buf[slot_flat, col_flat] = pos_masked

                self._ctx_len[slots] += gen_num_accepted_long
                self._ctx_len.clamp_(max=self._max_ctx)

            # Read padded context from buffers (fixed shape for CUDA graph)
            target_hidden = self._ctx_buf[slots]
            context_positions = self._ctx_pos_buf[slots].long()
            num_ctx_per_req_t = self._ctx_len[slots]

            # 3D padded tensors for CUDA graph compatible dflash_forward
            noise_embedding = noise_embed_2d
            query_positions = query_position_ids.long()

            # Update seq_lens for gen requests to 2K
            attn_metadata._seq_lens_cuda[num_contexts : num_contexts + num_gens] = (
                total_tokens_per_req
            )
            attn_metadata._seq_lens[num_contexts : num_contexts + num_gens] = total_tokens_per_req
        else:
            noise_embedding = hidden_states.new_empty(0, 0, hidden_dim)
            target_hidden = hidden_states.new_empty(0, 0, hidden_dim)
            query_positions = torch.empty(0, 0, dtype=torch.long, device="cuda")
            context_positions = torch.empty(0, 0, dtype=torch.long, device="cuda")
            num_ctx_per_req_t = torch.empty(0, dtype=torch.long, device="cuda")

        return {
            "noise_embedding": noise_embedding,
            "target_hidden": target_hidden,
            "query_positions": query_positions,
            "context_positions": context_positions,
            "num_ctx_per_req": num_ctx_per_req_t,
        }
