# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Draft-token verification for two-model speculative decoding.

Only reached when the drafter runs as a separate engine (NGram, two-model
Eagle3/draft-target). One-model speculation (MTP / Eagle3 one-model / SA)
verifies via ``SpecSampler`` and never enters this path -- see the note in
``speculative/drafter.py`` on which drafters use ``TorchSampler``.

``TwoModelSpecDecHandler`` owns the three verification strategies and the
dispatch between them; ``TorchSampler`` owns one instance and delegates to
:meth:`TwoModelSpecDecHandler.process_draft_tokens`.
"""

from typing import TYPE_CHECKING, Optional, cast

import torch

from tensorrt_llm._utils import prefer_pinned

from ..llm_request import LlmRequest, get_draft_token_length
from ..resource_manager import ResourceManager
from .sampler_common import DEFAULT_BEAM_IDX, FinishReasonsList, add_token
from .sampler_features import handle_stop_criteria
from .sampler_strategy import GREEDY, Strategy, _request_strategy, sample

if TYPE_CHECKING:
    from ...speculative.spec_tree_manager import SpecTreeManager
    from .sampler import TorchSampler

__all__ = ["TwoModelSpecDecHandler"]


def get_rejected_indices(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    generator: torch.Generator,
    draft_tokens: list[int],
) -> torch.Tensor:
    num_draft_tokens = draft_probs.size(0)
    draft_tokens = draft_tokens[:num_draft_tokens]
    token_idx = torch.arange(num_draft_tokens, dtype=torch.int32, device=generator.device)
    draft_tokens_cuda = torch.tensor(
        draft_tokens, dtype=torch.int32, pin_memory=prefer_pinned()
    ).to(device=generator.device, non_blocking=True)
    p = draft_probs[token_idx, draft_tokens_cuda]
    q = target_probs.squeeze(0)[token_idx, draft_tokens_cuda]
    accept_probs = torch.minimum(torch.ones((), device=generator.device, dtype=q.dtype), q / p)
    rejected_indices = (
        torch.rand(accept_probs.shape, generator=generator, device=accept_probs.device)
        > accept_probs
    ).nonzero()
    return rejected_indices


def sample_rejected(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    generator: torch.Generator,
    num_accepted: int,
) -> int:
    last_draft = draft_probs[num_accepted]
    last_target = target_probs[num_accepted]
    new = last_target - last_draft
    new = torch.where(new > 0, new, 0.0)
    new_token = torch.multinomial(new, num_samples=1, generator=generator).squeeze(-1)
    return cast(int, new_token.item())


class TwoModelSpecDecHandler:
    """Verifies draft tokens produced by a separate drafter engine.

    Holds a back-reference to the owning ``TorchSampler`` because verification
    is interleaved with sampler-owned concerns: finish-reason bookkeeping
    (``finish_if_reason``), the RNG
    (``get_generator``) and the speculation tree metadata
    (``get_spec_tree_manager``).
    """

    def __init__(self, sampler: "TorchSampler"):
        self._sampler = sampler

    def _process_draft_tokens_greedy(
        self,
        request: LlmRequest,
        new_tokens: list[list[list[int]]],
        finish_reasons: FinishReasonsList,
    ) -> int:
        new_token = add_token(request, new_tokens, beam_idx=DEFAULT_BEAM_IDX)
        stop = self._sampler.finish_if_reason(
            request, finish_reasons, step=0, beam_idx=DEFAULT_BEAM_IDX
        )
        if stop or get_draft_token_length(request) == 0:
            return 0
        num_accepted = 0

        if self._sampler._force_num_accepted_tokens != 0:
            # Force acceptance of up to force_num_accepted_tokens draft tokens
            force_limit = min(
                self._sampler._force_num_accepted_tokens, len(request.py_draft_tokens)
            )
            for _ in request.py_draft_tokens[:force_limit]:
                num_accepted += 1
                new_token = add_token(
                    request, new_tokens, beam_idx=DEFAULT_BEAM_IDX, step=num_accepted
                )
                if self._sampler.finish_if_reason(
                    request, finish_reasons, step=num_accepted, beam_idx=DEFAULT_BEAM_IDX
                ):
                    break
        else:
            for draft_token in request.py_draft_tokens:
                if draft_token != new_token:
                    # Reject.
                    break

                num_accepted += 1
                new_token = add_token(
                    request, new_tokens, beam_idx=DEFAULT_BEAM_IDX, step=num_accepted
                )
                if self._sampler.finish_if_reason(
                    request, finish_reasons, step=num_accepted, beam_idx=DEFAULT_BEAM_IDX
                ):
                    break
        return num_accepted

    def _process_draft_tokens_tree(
        self,
        request: LlmRequest,
        new_tokens_tensor: torch.Tensor,
        new_tokens_list: list[list[list[int]]],
        finish_reasons: FinishReasonsList,
        spec_tree_manager: "SpecTreeManager",
    ) -> int:
        """Tree verification for draft token tree based speculative decoding.

        This function will only be called for the target model.

        Verification logic:
            Find the longest prefix match. Since each node in the tree has a related path,
            we can find the longest match by comparing all the paths.
        Args:
            request: LlmRequest. The request with draft tokens.
            new_tokens: torch.Tensor. [max_total_draft_tokens + 1, max_num_sequences, max_beam_width], host buffer.
                        The tokens generated by the target model
                        The relationship between [max_total_draft_tokens + 1] and the draft token tree:
                        If the current node is accepted, what is the NEXT token_id that the target model will generate?
                        For example, new_tokens[0, req_idx, 1] indicates the NEXT token_id sampled from the root
                        node in the draft token tree if it is accepted.
                        We know that the root node in the draft token tree is always accepted. Therefore,
                        new_tokens[0, req_idx, 1] indicates the token_id following the root node,
                        corresponding to the first layer in the draft token tree (the root node is the 0th layer).
                        Similarly, new_tokens[1, req_idx, 1] represents the NEXT token_id if the first token in the
                        first layer of the draft tokens tree is accepted.
            spec_tree_manager: SpecTreeManager. which contains the tree structure and other meta information
                               of the tree.
        """
        # handle the target model request
        # For the target model, we will do the tree verification logic.
        seq_slot = request.py_seq_slot
        assert seq_slot is not None
        eagle_paths = spec_tree_manager.get_eagle_paths(seq_slot)

        all_draft_tokens = torch.tensor(request.py_draft_tokens)  # [max_total_draft_tokens]
        all_target_tokens = new_tokens_tensor[:, seq_slot, :].squeeze(
            -1
        )  # [max_total_draft_tokens]
        assert all_target_tokens.shape[0] == spec_tree_manager.max_total_draft_tokens + 1

        longest_accepted_len = 0
        longest_match_path_idx = -1

        for path_idx, path in enumerate(eagle_paths):
            path_exclude_root = (
                path[1:] - 1
            )  # [max_draft_len], '[1:]' since the new_tokens does not contain the root node.
            # '-1' is the index shift after exclude the root node.
            draft_tokens_indices = path_exclude_root[path_exclude_root >= 0]  # [max_draft_len]
            target_tokens_indices = path[path >= 0]  # [max_draft_len + 1]

            assert len(draft_tokens_indices) == len(target_tokens_indices) - 1

            cur_draft_tokens = all_draft_tokens[draft_tokens_indices]
            cur_target_tokens = all_target_tokens[target_tokens_indices]

            cur_accepted_len = cast(
                int,
                torch.cumprod((cur_draft_tokens == cur_target_tokens[:-1]).int(), dim=-1)
                .sum()
                .item(),
            )

            # Accepted one more token from the target model.
            cur_accepted_len += 1

            if cur_accepted_len > longest_accepted_len:
                longest_accepted_len = cur_accepted_len
                longest_match_path_idx = path_idx

        assert longest_accepted_len >= 1
        if longest_accepted_len == 1:
            assert longest_match_path_idx == 0

        # Take the longest accepted path as the next new token.
        num_accepted_draft_tokens = 0
        for idx in eagle_paths[longest_match_path_idx][:longest_accepted_len]:
            step = cast(int, idx.item())
            add_token(request, new_tokens_list, beam_idx=DEFAULT_BEAM_IDX, step=step)
            num_accepted_draft_tokens += 1
            if self._sampler.finish_if_reason(
                request,
                finish_reasons,
                step=step,
                beam_idx=DEFAULT_BEAM_IDX,
            ):
                break

        assert num_accepted_draft_tokens <= longest_accepted_len

        tree_node_indices = eagle_paths[longest_match_path_idx][1:num_accepted_draft_tokens]
        request.py_num_accepted_draft_tokens_indices = (tree_node_indices - 1).tolist()

        return num_accepted_draft_tokens - 1

    @torch.inference_mode()
    def _process_draft_tokens_rejection_sampling(
        self,
        request: LlmRequest,
        new_tokens_list: list[list[list[int]]],
        new_tokens_tensor: torch.Tensor,
    ) -> int:
        """We cannot use finish_if_reason in _process_draft_tokens_rejection_sampling because it *writes to new_tokens*,
        rendering the finish reason calculation in sample_async stale (incorrect) for this batch"""
        assert request.py_draft_logits is not None
        # FIXME: Passing a dummy vocab_size could result in unnecessary
        #        filtering of vocab_size logits, out of vocab_size in
        #        total. The 'sample' below should generally be avoided
        #        by retaining the draft_probs during drafting (TRTLLM-7772).
        draft_sampling_strategy = _request_strategy(request, vocab_size=2**31)
        generator = self._sampler.get_generator(request.py_draft_logits.device)
        _, draft_probs, _ = sample(
            draft_sampling_strategy,
            request.py_draft_logits,
            generator=generator,
        )
        assert draft_probs is not None
        target_probs = request.py_target_probs
        assert target_probs is not None
        d2t = getattr(request, "d2t", None)
        if d2t is not None:
            vocab_d = draft_probs.shape[-1]
            vocab_t = target_probs.shape[-1]
            assert d2t.numel() == vocab_d, f"d2t size mismatch: {d2t.numel()} != {vocab_d}"
            assert d2t.device == draft_probs.device, (
                f"d2t device mismatch: {d2t.device} != {draft_probs.device}"
            )
            aligned_draft_probs = torch.zeros(
                (*draft_probs.shape[:-1], vocab_t),
                device=draft_probs.device,
                dtype=draft_probs.dtype,
            )
            source_indices = torch.arange(vocab_d, device=draft_probs.device)
            target_indices = (source_indices + d2t) % vocab_t
            aligned_draft_probs[..., target_indices] = draft_probs
            draft_probs = aligned_draft_probs
        rejected_indices = get_rejected_indices(
            draft_probs,
            target_probs,
            generator,
            request.py_draft_tokens,
        )
        sample_last = True
        if rejected_indices.numel() == 0:
            num_initially_accepted = get_draft_token_length(request)
            sample_last = False
        else:
            num_initially_accepted = cast(int, rejected_indices[0].item())
        num_accepted = num_initially_accepted
        for i in range(num_accepted):
            new_token = request.py_draft_tokens[i]
            new_tokens_tensor[i, request.seq_slot, DEFAULT_BEAM_IDX] = new_token
            request.add_new_token(new_token, DEFAULT_BEAM_IDX)
            if handle_stop_criteria(
                request, new_token, beam_idx=DEFAULT_BEAM_IDX, max_seq_len=self._sampler.max_seq_len
            ):
                num_accepted = i + 1
                return num_accepted
        if sample_last:
            new_token = sample_rejected(draft_probs, target_probs, generator, num_accepted)
            new_tokens_tensor[num_accepted, request.seq_slot, DEFAULT_BEAM_IDX] = new_token
            request.add_new_token(new_token, DEFAULT_BEAM_IDX)
        else:
            new_token = add_token(
                request, new_tokens_list, beam_idx=DEFAULT_BEAM_IDX, step=num_accepted
            )
        handle_stop_criteria(
            request, new_token, beam_idx=DEFAULT_BEAM_IDX, max_seq_len=self._sampler.max_seq_len
        )

        return num_accepted

    @staticmethod
    def _speculation_could_use_rejection_sampling(
        request: LlmRequest, strategy: Optional[Strategy] = None
    ) -> bool:
        if strategy is None:
            strategy = _request_strategy(
                request,
                vocab_size=2**31,  # vocab_size does not affect greediness
            )
        return strategy != GREEDY and get_draft_token_length(request) > 0

    def process_draft_tokens(
        self,
        request: LlmRequest,
        new_tokens_tensor: torch.Tensor,
        new_tokens_list: list[list[list[int]]],
        finish_reasons: FinishReasonsList,
        resource_manager: Optional[ResourceManager] = None,
    ) -> int:
        if not (
            self._speculation_could_use_rejection_sampling(request)
            # NB: '_speculation_could_use_rejection_sampling' is called in sample_async, which precludes
            #     inspection of .py_draft_logits, because it is not set yet when the overlap path
            #     is used.
            #
            #     OTOH, some drafters (e.g. NGram) do not provide draft logits, precluding rejection
            #     sampling. The current solution accepts that .py_target_probs may sometimes be
            #     computed, even though .py_draft_logits may never be set and the target probs
            #     may ultimately not be required.
            and request.py_draft_logits is not None
        ):
            spec_tree_manager = self._sampler.get_spec_tree_manager(resource_manager)
            if spec_tree_manager is not None:
                num_accepted = self._process_draft_tokens_tree(
                    request,
                    new_tokens_tensor=new_tokens_tensor,
                    new_tokens_list=new_tokens_list,
                    finish_reasons=finish_reasons,
                    spec_tree_manager=spec_tree_manager,
                )
            else:
                num_accepted = self._process_draft_tokens_greedy(
                    request, new_tokens=new_tokens_list, finish_reasons=finish_reasons
                )
            return num_accepted
        else:
            return self._process_draft_tokens_rejection_sampling(
                request, new_tokens_list=new_tokens_list, new_tokens_tensor=new_tokens_tensor
            )
