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
"""
DraftTarget One-Model Speculative Decoding Implementation.

This module implements a one-model approach for DraftTarget speculative decoding,
where the draft and target models share the same model engine. The draft model
layers are integrated into the target model's KV cache and run in a single forward pass.
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from ..pyexecutor.sampler import TorchSampler
from .interface import SpecMetadata, SpecWorkerBase
from .mtp import MTPSampler
from .pearl_trace import log as _pearl_log
from .pearl_trace import tensor_rows as _pearl_tensor_rows
from .pearl_trace import to_int_list as _pearl_to_int_list

if TYPE_CHECKING:
    from ...llmapi.llm_args import DraftTargetDecodingConfig


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class DraftTargetOneModelSpecMetadata(SpecMetadata):
    """
    Metadata for DraftTarget one-model speculative decoding.

    This class manages the batch information needed for the one-model DraftTarget
    approach where draft and target models share the same model engine.
    Unlike Eagle3/MTP, DraftTarget does not require capturing hidden states
    from the target model to pass to the draft model.
    """

    # The max number of tokens
    max_num_tokens: int = 0
    # The index of the batch inputs
    batch_indices_cuda: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.batch_indices_cuda = torch.empty(
            [self.max_num_requests],
            dtype=torch.int,
            device="cuda",
        )

    def prepare(self):
        """Prepare the metadata before model forward."""
        assert self.request_ids is not None
        # Update batch indices
        num_seqs = len(self.request_ids)
        batch_indices = torch.arange(
            num_seqs, dtype=torch.int, device="cpu", pin_memory=prefer_pinned()
        )
        self.batch_indices_cuda[:num_seqs].copy_(batch_indices, non_blocking=True)
        self.num_tokens -= self.num_generations * self.max_draft_len
        self.is_spec_dec_tree = False
        self.is_spec_dec_dynamic_tree = False


class DraftTargetOneModelSampler(MTPSampler):
    """
    Sampler for DraftTarget one-model speculative decoding.

    Inherits from MTPSampler to reuse the speculative decoding sampling logic.
    """

    def __init__(self, args: TorchSampler.Args):
        super().__init__(args, nextn=args.max_draft_len)


class DraftTargetOneModelWorker(SpecWorkerBase):
    def __init__(
        self,
        spec_config: "DraftTargetDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(use_separate_draft_kv_cache)
        self.spec_config = spec_config
        self.mapping = mapping
        self._rdma_offload_enabled = bool(
            getattr(spec_config, "draft_offload_enabled", False)
            or _env_enabled("TLLM_DRAFT_RDMA_OFFLOAD")
        )
        self._rdma_offload_v2 = bool(
            getattr(spec_config, "draft_offload_v2", False)
            or _env_enabled("TLLM_DRAFT_RDMA_OFFLOAD_V2")
        )
        self._rdma_v2_offload_layer = None
        if self._rdma_offload_enabled:
            if getattr(mapping, "tp_size", 1) != 1 or getattr(mapping, "pp_size", 1) != 1:
                raise RuntimeError(
                    "RDMA draft offload target path currently supports only "
                    "single-rank TP/PP. Disable draft_offload_enabled for "
                    "multi-rank runs."
                )
            if not self._rdma_offload_v2:
                raise RuntimeError("RDMA draft offload requires draft_offload_v2=True")
            from .ibverbs_draft_offload import IbverbsDraftOffloadConfig, IbverbsDraftOffloadLayer

            self._rdma_v2_offload_layer = IbverbsDraftOffloadLayer(
                IbverbsDraftOffloadConfig(
                    nic_name=getattr(spec_config, "draft_offload_nic_name", "mlx5_0"),
                    server_host=getattr(spec_config, "draft_offload_server_host", "127.0.0.1"),
                    server_port=int(getattr(spec_config, "draft_offload_server_port", 47000)),
                    remote_peer_name=getattr(
                        spec_config, "draft_offload_v2_remote_peer_name", "draft_lpu"
                    ),
                    max_num_requests=int(
                        getattr(spec_config, "draft_offload_v2_max_num_requests", 256)
                    ),
                    max_draft_len=int(spec_config.max_draft_len),
                    transport=str(getattr(spec_config, "draft_offload_v2_transport", "ibverbs")),
                    draft_model_path=getattr(spec_config, "draft_offload_v2_model_path", None),
                    draft_model_dtype=str(
                        getattr(spec_config, "draft_offload_v2_model_dtype", "bfloat16")
                    ),
                    draft_kv_cache_free_fraction=float(
                        getattr(spec_config, "draft_offload_v2_kv_cache_free_fraction", 0.4)
                    ),
                    tcp_prompt_port=int(
                        getattr(spec_config, "draft_offload_v2_tcp_prompt_port", 0)
                    ),
                )
            )
            logger.info(
                "DraftTarget RDMA-v2 (izzy compatible) enabled: host=%s port=%s nic=%s "
                "max_num_requests=%s max_draft_len=%s",
                spec_config.draft_offload_server_host,
                spec_config.draft_offload_server_port,
                spec_config.draft_offload_nic_name,
                spec_config.draft_offload_v2_max_num_requests,
                spec_config.max_draft_len,
            )

    @property
    def max_draft_len(self) -> int:
        return self.spec_config.max_draft_len

    def _prepare_attn_metadata_for_draft_target(
        self,
        attn_metadata: AttentionMetadata,
        spec_metadata: DraftTargetOneModelSpecMetadata,
    ):
        """
        Save the attention metadata fields modified by DraftTarget.

        During CUDA-graph warmup, kv_lens_cuda is also saved/restored to avoid
        cross-warmup accumulation. During capture and normal inference we keep
        kv_lens_cuda live so the updates persist.
        """
        is_capturing = torch.cuda.is_current_stream_capturing()

        if (
            spec_metadata.is_cuda_graph
            and not is_capturing
            and hasattr(attn_metadata, "kv_lens_cuda")
            and isinstance(attn_metadata.kv_lens_cuda, torch.Tensor)
        ):
            attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda", "kv_lens_cuda")
        else:
            attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")

    def _update_kv_after_first_draft_step(
        self,
        attn_metadata: AttentionMetadata,
        num_accepted_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
    ):
        if hasattr(attn_metadata, "kv_lens_cuda"):
            attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= (
                self.max_draft_len - num_accepted_tokens[num_contexts:batch_size]
            )
            attn_metadata.kv_lens_cuda[:num_contexts] += 1

        # Some attention backends keep extra indexing state derived from
        # seq_lens / kv_lens that must be refreshed for chained drafting.
        attn_metadata.update_for_spec_dec()

    def _update_kv_for_chained_draft_step(
        self,
        attn_metadata: AttentionMetadata,
        batch_size: int,
    ):
        if hasattr(attn_metadata, "kv_lens_cuda"):
            attn_metadata.kv_lens_cuda[:batch_size] += 1

        attn_metadata.update_for_spec_dec()

    def forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata: AttentionMetadata,
        spec_metadata: DraftTargetOneModelSpecMetadata,
        draft_model: nn.Module,
        resource_manager=None,
        is_warmup: bool = False,
    ):
        """
        Technically incorrect at the moment.
        Leverages Eagle3/MTP setup that does this for the context
        input_ids_ctx[:-1].copy_(input_prompt_ids[1:])
        In DraftTarget, we do not want to shift, which necessitates increasing the final chunk of each request by 1
        for the final accepted token.  This creates a big headache since then the kv lens, seq_lens, token counts all
        have to be updated and then reverted when heading back to the target.  TODO: non trivially fix this issue.
        """

        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        raw_logits = logits

        self._execute_guided_decoder_if_present(logits)

        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata
        )
        if self._rdma_offload_enabled and not bool(is_warmup):
            draft_tokens_for_log = None
            if spec_metadata.draft_tokens is not None and num_gens > 0:
                try:
                    draft_tokens_for_log = spec_metadata.draft_tokens.reshape(
                        num_gens, self.max_draft_len
                    )
                except Exception:
                    draft_tokens_for_log = spec_metadata.draft_tokens
            _pearl_log(
                "target",
                "verify_result",
                request_ids=[int(r) for r in getattr(spec_metadata, "request_ids", [])],
                batch_size=int(batch_size),
                num_contexts=int(num_contexts),
                num_generations=int(num_gens),
                max_draft_len=int(self.max_draft_len),
                input_draft_tokens=_pearl_tensor_rows(
                    draft_tokens_for_log,
                    int(num_gens),
                    width=int(self.max_draft_len),
                ),
                accepted_token_counts=_pearl_to_int_list(
                    num_accepted_tokens, limit=int(batch_size)
                ),
                accepted_tokens=_pearl_tensor_rows(
                    accepted_tokens,
                    int(batch_size),
                    width=int(self.max_draft_len) + 1,
                ),
            )

        if self._rdma_offload_enabled:
            if bool(is_warmup):
                # Warmup: initialize RDMA connection so the first real decode
                # step does not pay connection-setup latency. The channel
                # starts lazily on first real forward, so warmup only returns
                # a correctly-shaped zero tensor.
                next_draft_tokens = torch.zeros(
                    (batch_size, self.max_draft_len), dtype=torch.int32, device=logits.device
                )
            else:
                next_draft_tokens = self._rdma_offload_draft_tokens(
                    accepted_tokens=accepted_tokens,
                    num_accepted_tokens=num_accepted_tokens,
                    position_ids=position_ids,
                    logits=logits,
                    batch_size=batch_size,
                    request_ids=getattr(spec_metadata, "request_ids", None),
                    input_ids=input_ids,
                    attn_metadata=attn_metadata,
                )
            next_new_tokens = self._prepare_next_new_tokens(
                accepted_tokens,
                next_draft_tokens,
                spec_metadata.batch_indices_cuda,
                batch_size,
                num_accepted_tokens,
            )
            attn_metadata.use_spec_decoding = True
            return {
                "logits": raw_logits,
                "new_tokens": accepted_tokens,
                "new_tokens_lens": num_accepted_tokens,
                "next_draft_tokens": next_draft_tokens,
                "next_new_tokens": next_new_tokens,
            }

        # Prepare attention metadata for speculative decoding and save state for restore
        self._prepare_attn_metadata_for_draft_target(attn_metadata, spec_metadata)

        # Prepare inputs for the first draft forward
        position_ids = position_ids.squeeze(0)
        inputs = self.prepare_1st_drafter_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            accepted_tokens=accepted_tokens,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
        )

        next_draft_tokens = []
        original_all_rank_num_tokens = attn_metadata.all_rank_num_tokens

        # Get the draft KV cache manager if using separate layouts
        draft_kv_cache_manager = self.get_draft_kv_cache_manager(resource_manager)

        with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
            for i in range(self.max_draft_len):
                if i == 0:
                    start_ids_gen = (
                        spec_metadata.batch_indices_cuda[:num_gens] * (self.max_draft_len + 1)
                    ).long()
                    gather_ids_gen = (
                        start_ids_gen
                        + num_accepted_tokens[num_contexts:]
                        - 1
                        + attn_metadata.num_ctx_tokens
                    )
                    gather_ids = torch.concat(
                        [spec_metadata.gather_ids[:num_contexts], gather_ids_gen], dim=0
                    )
                else:
                    gather_ids = spec_metadata.batch_indices_cuda[:batch_size]

                if self.guided_decoder is not None:
                    new_tokens = inputs["input_ids"][gather_ids]
                    self.guided_decoder.add_draft_batch(
                        new_tokens, num_accepted_tokens, draft_step=i
                    )

                if original_all_rank_num_tokens is not None:
                    if i == 0:
                        attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens
                    elif spec_metadata.all_rank_num_seqs is not None:
                        attn_metadata.all_rank_num_tokens = spec_metadata.all_rank_num_seqs

                hidden_states = draft_model.model(**inputs)
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]

                # Disable spec-dec mode for chained draft steps
                attn_metadata.use_spec_decoding = False

                logits = draft_model.logits_processor(
                    hidden_states[gather_ids], draft_model.lm_head, attn_metadata, True
                )
                if self.guided_decoder is not None:
                    d2t = getattr(draft_model.model, "d2t", None)
                    self.guided_decoder.execute_draft_batch(logits, d2t, draft_step=i)

                new_draft_token = self.draft_decoder(logits, draft_model)
                next_draft_tokens.append(new_draft_token)

                # Update inputs and metadata for next draft step
                position_ids = inputs["position_ids"][gather_ids] + 1
                if i == 0:
                    attn_metadata._seq_lens[:batch_size].fill_(1)
                    attn_metadata._seq_lens_cuda[:batch_size].fill_(1)
                    attn_metadata.on_update()
                    if inputs["attn_metadata"].kv_cache_manager is not None:
                        attn_metadata.host_request_types[: attn_metadata.num_contexts].fill_(1)
                        attn_metadata.num_contexts = 0
                    self._update_kv_after_first_draft_step(
                        attn_metadata, num_accepted_tokens, num_contexts, batch_size
                    )
                else:
                    self._update_kv_for_chained_draft_step(attn_metadata, batch_size)

                inputs = {
                    "input_ids": new_draft_token,
                    "position_ids": position_ids,
                    "attn_metadata": attn_metadata,
                    "spec_metadata": spec_metadata,
                }

        next_draft_tokens = torch.stack(next_draft_tokens, dim=1)

        # Restore attention metadata to original state
        self._restore_attn_metadata_from_spec_dec(attn_metadata)
        if original_all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens

        next_new_tokens = self._prepare_next_new_tokens(
            accepted_tokens,
            next_draft_tokens,
            spec_metadata.batch_indices_cuda,
            batch_size,
            num_accepted_tokens,
        )

        attn_metadata.use_spec_decoding = True

        return {
            "logits": raw_logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }

    def _rdma_offload_draft_tokens(
        self,
        *,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        logits: torch.Tensor,
        batch_size: int,
        request_ids: Optional[list] = None,
        input_ids: Optional[torch.Tensor] = None,
        attn_metadata: Optional["AttentionMetadata"] = None,
    ) -> torch.Tensor:
        # V2 path: izzy-compatible 96-byte WRITE_WITH_IMM via
        # ``IbverbsDraftOffloadLayer``.  All bookkeeping (round_seq,
        # route binding, response dispatch) happens inside the layer.
        if self._rdma_v2_offload_layer is not None:
            if request_ids is None or len(request_ids) != int(batch_size):
                # Fall back to a synthetic request id; v2 needs one per row.
                request_ids = list(range(int(batch_size)))
            # ``IbverbsDraftOffloadLayer.forward`` expects exactly one
            # start-position per request (length == batch_size), but
            # TRT-LLM passes the *full* per-token position_ids
            # (length == total prompt+gen tokens in batch).  Build a
            # per-request position vector — last position is what the
            # draft side needs to know where to resume from.  Only
            # batch_size=1 is exercised today (Phase 4 limitation); the
            # multi-request path lands in Phase 9.
            if position_ids is None or int(position_ids.numel()) == 0:
                per_request_positions = torch.zeros(
                    (int(batch_size),), dtype=torch.int64, device=logits.device
                )
            elif int(position_ids.numel()) == int(batch_size):
                per_request_positions = position_ids.reshape(-1)
            else:
                last_pos = int(position_ids.reshape(-1)[-1].detach().cpu().item())
                per_request_positions = torch.tensor(
                    [last_pos] * int(batch_size), dtype=torch.int64, device=logits.device
                )

            # Auto prompt push on context phase: when ``num_contexts > 0``
            # the request is being prefilled on the target side this step,
            # so the draft side must also prefill before the first round
            # trip lands. We push the full prompt token sequence for each
            # context row. Batch_size==1 today (see the multi-request
            # caveat above); the multi-row generalization slices
            # ``input_ids`` by ``attn_metadata.seq_lens[:num_contexts]``.
            #
            # PEARL pre-verify timeline:
            #   draft prompt_init precomputes d_f from the prompt;
            #   target context prefill computes t_f from target logits;
            #   the first data-plane round sends t_f to draft;
            #   draft keeps its d_f branch if d_f == t_f, otherwise it
            #   rolls back to the prompt and regenerates from t_f.
            if attn_metadata is not None and input_ids is not None:
                num_contexts = int(getattr(attn_metadata, "num_contexts", 0))
                if num_contexts > 0:
                    flat = input_ids.reshape(-1)
                    seq_lens = getattr(attn_metadata, "_seq_lens", None)
                    if seq_lens is None:
                        seq_lens = getattr(attn_metadata, "seq_lens", None)
                    cursor = 0
                    for row in range(num_contexts):
                        if seq_lens is not None:
                            length = int(seq_lens[row].item())
                        else:
                            # Fall back to the single-request shortcut.
                            length = int(getattr(attn_metadata, "num_ctx_tokens", flat.numel()))
                        prompt_tokens = flat[cursor : cursor + length].detach().cpu().tolist()
                        cursor += length
                        try:
                            self._rdma_v2_offload_layer.push_prompt(
                                int(request_ids[row]),
                                prompt_tokens,
                            )
                        except RuntimeError as exc:
                            logger.warning(
                                "draft_offload push_prompt failed for request %s: %s",
                                request_ids[row],
                                exc,
                            )

            return self._rdma_v2_offload_layer(
                input_ids=accepted_tokens,
                position_ids=per_request_positions,
                accepted_tokens=accepted_tokens,
                num_accepted_tokens=num_accepted_tokens,
                batch_size=int(batch_size),
                num_contexts=0,
                request_ids=request_ids,
            )

        raise RuntimeError("RDMA draft offload layer was not initialized")

    def sample_and_accept_draft_tokens(
        self,
        logits: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: DraftTargetOneModelSpecMetadata,
    ):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        if spec_metadata.draft_tokens is None:
            draft_tokens = torch.zeros(
                (num_gens, self.max_draft_len), dtype=torch.int, device=logits.device
            )
        else:
            draft_tokens = spec_metadata.draft_tokens.reshape(num_gens, self.max_draft_len)

        return self._sample_and_accept_draft_tokens_base(
            logits, draft_tokens, num_contexts, batch_size, spec_metadata
        )

    def draft_decoder(
        self,
        logits: torch.Tensor,
        draft_model: nn.Module,
    ):
        d2t = getattr(draft_model.model, "d2t", None)
        return self._draft_sampler_greedy(logits, d2t)

    def prepare_1st_drafter_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: DraftTargetOneModelSpecMetadata,
    ):
        num_contexts = attn_metadata.num_contexts
        batch_size = attn_metadata.num_seqs
        num_gens = batch_size - num_contexts

        if num_contexts > 0:
            input_ids_ctx = self._prepare_context_input_ids(
                input_ids,
                attn_metadata.num_ctx_tokens,
                spec_metadata.gather_ids,
                accepted_tokens,
                num_contexts,
            ).to(torch.int32)
        else:
            input_ids_ctx = torch.empty(0, dtype=torch.int32, device="cuda")

        if num_gens > 0:
            input_ids_gen = accepted_tokens[num_contexts:, :].flatten().to(torch.int32)
        else:
            input_ids_gen = torch.empty(0, dtype=torch.int32, device="cuda")

        input_ids = torch.cat([input_ids_ctx, input_ids_gen], dim=0)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }
