from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from tensorrt_llm.bindings import ModelConfig, WorldConfig, SamplingConfig
from tensorrt_llm.bindings.executor import DecodingConfig
from tensorrt_llm.bindings.internal.batch_manager import DecoderInputBuffers, LlmRequest
from tensorrt_llm.bindings.internal.runtime import Request


@dataclass
class GenerateRequestOptions:
    """
    Python implementation of GenerateRequestOptions.
    Implements the logic in `generateRequestOptions.cpp`.
    """
    speculative_decoding_fast_logits: bool
    is_leader_in_orch_mode: bool
    is_normalize_log_probs: bool

    def __call__(
        self,
        model_config: ModelConfig,
        world_config: WorldConfig,
        decoding_config: DecodingConfig,
        context_requests: List[LlmRequest],
        logits_type: torch.dtype,
        input_buffers: DecoderInputBuffers,
    ) -> Tuple[torch.Tensor, List[Request], List[SamplingConfig]]:
        """Process context requests and prepare decoder requests.

        Args:
            model_config: Model configuration
            world_config: World configuration
            decoding_config: Decoding configuration
            context_requests: List of context requests to process
            logits_type: Data type for logits
            input_buffers: Input buffers for decoder

        Returns:
            Tuple containing:
            - Batch slots view tensor
            - List of decoder requests
            - List of sampling configs
        """
        batch_size = 0
        decoder_input_size = 0

        # Calculate batch size and decoder input size
        for llm_req in context_requests:
            req_tokens = llm_req.get_tokens(0)
            if llm_req.is_last_context_chunk():
                decoder_input_size += len(req_tokens)
                batch_size += 1

        # Resize input buffer
        input_buffers.inputs_ids = torch.empty(decoder_input_size, dtype=torch.int32, device='cuda')

        # Create batch slots view
        batch_slots_view = input_buffers.setup_batch_slots[:batch_size]

        decoder_requests = []
        sampling_configs = []

        batch_idx = 0
        input_offset = 0

        # Process each context request
        for llm_req in context_requests:
            if not llm_req.is_last_context_chunk():
                continue

            prompt_len = llm_req.prompt_len
            req_tokens = torch.tensor(llm_req.get_tokens(0))
            assert len(req_tokens) == prompt_len

            # Create input view and copy tokens
            input_view = input_buffers.inputs_ids[input_offset:input_offset + prompt_len]
            input_view.copy_(req_tokens, non_blocking=True)

            # Create decoder request
            decoder_request = Request(
                ids=input_view,
                input_len=prompt_len,
                max_new_tokens=llm_req.max_new_tokens,
                end_id=llm_req.end_id
            )

            # Set sampling config
            llm_req.sampling_config.normalize_log_probs = self.is_normalize_log_probs

            # Handle speculative decoding
            if model_config.speculative_decoding_mode.is_draft_tokens_external:
                if llm_req.has_draft_tokens():
                    draft_tokens = llm_req.draft_tokens
                    decoder_request.draft_tokens = draft_tokens.clone(memory_format=torch.contiguous_format).pin_memory()
                    
                    draft_logits = llm_req.draft_logits
                    if draft_logits is not None:
                        decoder_request.draft_logits = self._retrieve_draft_logits(
                            model_config, world_config, draft_logits)
                    
                    decoder_request.generated_tokens_per_engine_step = len(draft_tokens) + 1
                else:
                    decoder_request.generated_tokens_per_engine_step = 1
            elif not model_config.speculative_decoding_mode.is_none:
                decoder_request.generated_tokens_per_engine_step = model_config.get_max_decoding_tokens()

            # Handle Medusa speculative decoding
            if model_config.speculative_decoding_mode.is_medusa:
                # NOTE: The logic of Medusa requires a RuntimeBuffers object.
                raise NotImplementedError("Medusa speculative decoding is not currently supported in TRTLLMDecoder.")

            # Handle lookahead decoding
            elif model_config.speculative_decoding_mode.is_lookahead_decoding:
                decoder_request.lookahead_runtime_config = (
                    llm_req.get_lookahead_config() or decoding_config.get_lookahead_decoding_config()
                )

            # Handle explicit draft tokens
            elif model_config.speculative_decoding_mode.is_explicit_draft_tokens:
                decoder_request.dtype = model_config.get_data_type()

            # Handle Eagle speculative decoding
            elif model_config.speculative_decoding_mode.is_eagle:
                decoder_request.eagle_config = (
                    llm_req.get_eagle_config() or decoding_config.get_eagle_config()
                )

            # Handle embedding bias
            if llm_req.embedding_bias is not None:
                decoder_request.embedding_bias = self._get_embedding_bias(
                    logits_type, llm_req.embedding_bias)

            # Handle bad words list
            if llm_req.bad_words_list is not None:
                decoder_request.bad_words_list = llm_req.bad_words_list.clone().squeeze_(0)

            # Handle stop words list
            if llm_req.stop_words_list is not None:
                decoder_request.stop_words_list = llm_req.stop_words_list.clone().squeeze_(0)

            # Set batch slot and add to lists
            batch_slots_view[batch_idx] = llm_req.seq_slot
            decoder_requests.append(decoder_request)
            sampling_configs.append(llm_req.sampling_config)

            input_offset += prompt_len
            batch_idx += 1

        return batch_slots_view, decoder_requests, sampling_configs

    def _retrieve_draft_logits(
        self,
        model_config: ModelConfig,
        world_config: WorldConfig,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve draft logits for speculative decoding.

        Args:
            model_config: Model configuration
            world_config: World configuration
            tensor: Input tensor containing draft logits

        Returns:
            Processed draft logits tensor
        """
        raise NotImplementedError("Draft logits are not currently supported in TRTLLMDecoder.")
    
    def _get_embedding_bias(self, logits_type: torch.dtype, tensor: torch.Tensor) -> torch.Tensor:
        """Get embedding bias tensor with correct data type.

        Args:
            logits_type: Target data type for logits
            tensor: Input embedding bias tensor

        Returns:
            Processed embedding bias tensor
        """
        # Return tensor if types match
        if tensor.dtype == logits_type:
            return tensor

        # Handle FP32 to FP16 conversion
        if tensor.dtype == torch.float32 and logits_type == torch.float16:
            return tensor.to(logits_type, device='cuda')

        raise RuntimeError("Embedding bias data type must be same as model logits type.")
