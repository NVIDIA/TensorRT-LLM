import itertools
import math
from typing import List, Optional

import torch
import xgrammar

from ...bindings.executor import GuidedDecodingConfig, GuidedDecodingParams
from .llm_request import LlmRequest
from .resource_manager import BaseResourceManager, SlotManager
from .scheduler import ScheduledRequests


class GuidedDecoderResourceManager(BaseResourceManager):

    def __init__(self, max_num_sequences: int):
        self.slot_manager = SlotManager(max_num_sequences)

    def get_max_resource_count(self) -> int:
        return self.slot_manager.max_num_requests

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        return int(request.guided_decoding_params is not None)

    def prepare_resources(self, scheduled_batch: ScheduledRequests) -> None:
        for llm_req in itertools.chain(scheduled_batch.context_requests,
                                       scheduled_batch.generation_requests):
            if llm_req.guided_decoding_params is None:
                continue
            if llm_req.is_context_init_state and llm_req.context_current_position == llm_req.prepopulated_prompt_len:
                self.slot_manager.add_slot(llm_req.request_id)

    def free_resources(self, request: LlmRequest) -> None:
        if request.guided_decoding_params is not None:
            self.slot_manager.remove_slot(request.request_id)


class GuidedDecoder:
    bitmask_dtype = torch.int32

    def __init__(self, guided_decoding_config: GuidedDecodingConfig,
                 max_num_sequences: int, vocab_size_padded: int):
        self.guided_decoding_backend = guided_decoding_config.backend
        self.max_num_sequences = max_num_sequences
        self.vocab_size_padded = vocab_size_padded

        if self.guided_decoding_backend == GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR:
            if guided_decoding_config.tokenizer_str is not None:
                metadata = xgrammar.TokenizerInfo._detect_metadata_from_hf(
                    guided_decoding_config.tokenizer_str)
                tokenizer_info = xgrammar.TokenizerInfo(
                    guided_decoding_config.encoded_vocab,
                    vocab_type=metadata["vocab_type"],
                    vocab_size=vocab_size_padded,
                    stop_token_ids=guided_decoding_config.stop_token_ids,
                    add_prefix_space=metadata["add_prefix_space"])
            else:
                tokenizer_info = xgrammar.TokenizerInfo(
                    guided_decoding_config.encoded_vocab,
                    xgrammar.VocabType.RAW,
                    vocab_size=vocab_size_padded,
                    stop_token_ids=guided_decoding_config.stop_token_ids)
            self.xgrammar_compiler = xgrammar.GrammarCompiler(tokenizer_info)
            self.xgrammar_matchers: List[Optional[
                xgrammar.GrammarMatcher]] = [None] * self.max_num_sequences
            self.bitmask = torch.empty(self.max_num_sequences,
                                       self.bitmask_size,
                                       dtype=self.bitmask_dtype,
                                       device='cuda')
            self.bitmask_host = torch.empty(self.max_num_sequences,
                                            self.bitmask_size,
                                            dtype=self.bitmask_dtype,
                                            pin_memory=True)

        self._stream = torch.cuda.Stream()

    @property
    def bitmask_size(self) -> int:
        return math.ceil(self.vocab_size_padded / 32)

    def build(self, scheduled_requests: ScheduledRequests,
              resource_manager: GuidedDecoderResourceManager) -> None:
        if self.guided_decoding_backend == GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR:
            for llm_req in itertools.chain(
                    scheduled_requests.context_requests,
                    scheduled_requests.generation_requests):
                if llm_req.guided_decoding_params is None:
                    continue
                slot = resource_manager.slot_manager.get_slot(
                    llm_req.request_id)
                if llm_req.is_context_init_state and llm_req.context_current_position == llm_req.prepopulated_prompt_len:
                    # The request is in the first context forward step (considering kv cache reuse).
                    guide_type = llm_req.guided_decoding_params.guide_type
                    guide = llm_req.guided_decoding_params.guide
                    match guide_type:
                        case GuidedDecodingParams.GuideType.JSON:
                            compiled_grammar = self.xgrammar_compiler.compile_builtin_json_grammar(
                            )
                        case GuidedDecodingParams.GuideType.JSON_SCHEMA:
                            compiled_grammar = self.xgrammar_compiler.compile_json_schema(
                                guide)
                        case GuidedDecodingParams.GuideType.REGEX:
                            grammar = xgrammar.Grammar.from_regex(guide)
                            compiled_grammar = self.xgrammar_compiler.compile_grammar(
                                grammar)
                        case GuidedDecodingParams.GuideType.EBNF_GRAMMAR:
                            grammar = xgrammar.Grammar.from_ebnf(guide)
                            compiled_grammar = self.xgrammar_compiler.compile_grammar(
                                grammar)
                        case _:
                            raise ValueError(
                                f"Unrecognized guide type: {guide_type}.")
                    self.xgrammar_matchers[slot] = xgrammar.GrammarMatcher(
                        compiled_grammar)

                elif llm_req.is_generation_in_progress_state:
                    # The request is in a generation forward step.
                    # Currently, guided decoding does not support with beam search.
                    self.xgrammar_matchers[slot].accept_token(
                        llm_req.get_last_tokens(0))
                else:
                    continue

                # Fill the bitmask on host and asynchorously copy to device.
                self.xgrammar_matchers[slot].fill_next_token_bitmask(
                    self.bitmask_host, slot)
                with torch.cuda.stream(self._stream):
                    self.bitmask[slot].copy_(self.bitmask_host[slot],
                                             non_blocking=True)

    def execute(self, scheduled_requests: ScheduledRequests,
                logits: torch.Tensor,
                resource_manager: GuidedDecoderResourceManager) -> None:
        assert logits.size(0) == len(scheduled_requests.context_requests) + len(
            scheduled_requests.generation_requests)
        torch.cuda.current_stream().wait_stream(self._stream)

        if self.guided_decoding_backend == GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR:
            batched_logits, batched_bitmask = [], []
            for i, llm_req in enumerate(
                    itertools.chain(scheduled_requests.context_requests,
                                    scheduled_requests.generation_requests)):
                if llm_req.guided_decoding_params is None:
                    continue
                if llm_req.is_context_init_state and not llm_req.is_last_context_chunk(
                ):
                    continue
                batched_logits.append(logits[i])
                slot = resource_manager.slot_manager.get_slot(
                    llm_req.request_id)
                batched_bitmask.append(self.bitmask[slot])

            if len(batched_logits) > 0:
                torch.ops.trtllm.logits_bitmask(batched_logits, batched_bitmask)
