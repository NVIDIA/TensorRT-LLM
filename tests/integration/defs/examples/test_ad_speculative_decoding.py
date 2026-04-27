# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

import pytest
import torch
import torch.nn as nn
from defs.conftest import llm_models_root
from test_common.llm_data import hf_id_to_local_model_dir
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.utils.generic import ModelOutput

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.auto_deploy.llm import LLM
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import (
    EagleDrafterForCausalLM,
    EagleWrapper,
    EagleWrapperConfig,
)
from tensorrt_llm._torch.auto_deploy.models.eagle import EagleDrafterFactory
from tensorrt_llm.llmapi import Eagle3DecodingConfig

prompts = [
    "What is the capital of France?",
    "Please explain the concept of gravity in simple words and a single sentence.",
    "What are the main differences between Python and C++?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
]

EAGLE_MODEL_SUBPATH = "EAGLE3-LLaMA3.1-Instruct-8B"
LLAMA_BASE_SUBPATH = "llama-3.1-model/Llama-3.1-8B-Instruct"
EAGLE_MAX_DRAFT_LEN = 3


def get_model_paths():
    """Get model paths using llm_models_root()."""
    models_root = llm_models_root()
    base_model = os.path.join(models_root, LLAMA_BASE_SUBPATH)
    eagle_model = os.path.join(models_root, EAGLE_MODEL_SUBPATH)

    print(f"Base model path: {base_model}")
    print(f"EAGLE model path: {eagle_model}")
    return base_model, eagle_model


@pytest.mark.parametrize(
    ("attn_backend", "compile_backend"),
    [
        ("trtllm", "torch-cudagraph"),
        ("flashinfer", "torch-simple"),
    ],
)
def test_autodeploy_eagle3_one_model_acceptance_rate(attn_backend: str, compile_backend: str):
    """Test Eagle3 one-model acceptance rate with AutoDeploy engine.

    Runs Eagle3 one-model speculative decoding with streaming and verifies
    that the acceptance rate is above a minimum threshold.
    Parameterized over attention backend and compile backend.
    """
    print("\n" + "=" * 80)
    print(
        f"Testing AutoDeploy Eagle3 One-Model Acceptance Rate "
        f"(attn_backend={attn_backend}, compile_backend={compile_backend})"
    )
    print("=" * 80)

    base_model, eagle_model = get_model_paths()

    print(f"\nBase Model: {base_model}")
    print(f"Eagle3 Model: {eagle_model}")

    max_draft_len = EAGLE_MAX_DRAFT_LEN

    speculative_config = Eagle3DecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model=eagle_model,
        eagle3_one_model=True,
        eagle3_layers_to_capture={1, 15, 28},
    )

    with LLM(
        model=base_model,
        skip_loading_weights=False,
        runtime="trtllm",
        world_size=1,
        speculative_config=speculative_config,
        # Force the Eagle3 draft to match the target (Llama 3.1 8B is bfloat16).
        # Shared KV cache requires matching dtypes between target and draft.
        speculative_model_kwargs={"torch_dtype": "bfloat16"},
        compile_backend=compile_backend,
        attn_backend=attn_backend,
        max_num_tokens=512,
        # max_batch_size must leave room for an extend-only sample batch during
        # resize_kv_cache, i.e. max_num_tokens // max_batch_size >= 1 + max_draft_len.
        # Otherwise the sample batch is classified as decode-only and the Eagle
        # wrapper rejects it ("decode without drafting is not supported").
        # TODO: remove once resize_kv_cache is spec-aware.
        # See: https://github.com/NVIDIA/TensorRT-LLM/issues/13348
        max_batch_size=128,
    ) as llm:
        _run_acceptance_rate_check(llm, max_draft_len)


def _run_acceptance_rate_check(llm, max_draft_len: int, min_acceptance_rate: float = 0.10):
    """Common helper for acceptance rate tests.

    Submits all requests simultaneously so the executor processes them concurrently
    (batch size > 1), then consumes streaming results to compute acceptance rates.
    """
    batch_tok_ids = [llm.tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(max_tokens=128, temperature=0, seed=42)

    print("\nRunning Eagle3 speculative decoding with streaming...")
    print(f"Submitting all {len(batch_tok_ids)} requests simultaneously...")

    # Submit all requests before consuming any results so they are in-flight concurrently.
    generators = [
        llm.generate_async(tok_ids, sampling_params, streaming=True) for tok_ids in batch_tok_ids
    ]

    for i, gen in enumerate(generators):
        num_tokens = 0
        num_drafted = 0
        num_accepted = 0

        for output in gen:
            new_tokens = output.outputs[0].token_ids
            num_drafted += max_draft_len
            num_accepted += len(new_tokens) - num_tokens - 1
            num_tokens = len(new_tokens)

        accept_rate = num_accepted / num_drafted

        generated_text = output.outputs[0].text
        if not generated_text:
            generated_text = llm.tokenizer.decode(output.outputs[0].token_ids)
        print(f"\n[PROMPT {i}] {prompts[i]}")
        print(f"[OUTPUT {i}] {generated_text}")

        print(f"\nRequest {i + 1} Acceptance Rate Statistics:")
        print(f"  Total tokens drafted: {num_drafted}")
        print(f"  Total tokens accepted: {num_accepted}")
        print(f"  Acceptance rate: {accept_rate:.2%}")

        assert accept_rate > min_acceptance_rate, (
            f"Request {i + 1}: Acceptance rate {accept_rate:.2%} is below minimum threshold "
            f"{min_acceptance_rate:.0%}"
        )

    print("\n" + "=" * 80)
    print("SUCCESS! All requests passed acceptance rate threshold")
    print("=" * 80)


def load_weights(model_path: Path, model: torch.nn.Module):
    """Load weights from checkpoint while applying the same _checkpoint_conversion_mapping that the factory uses.

    Returns: tuple of (loaded_keys, missing_keys, unexpected_keys)
    """
    # 1. Load checkpoint keys
    bin_path = model_path / "pytorch_model.bin"
    safetensors_path = model_path / "model.safetensors"

    if safetensors_path.exists():
        from safetensors import safe_open

        with safe_open(safetensors_path, framework="pt") as f:
            checkpoint_keys_original = list(f.keys())
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        checkpoint_keys_original = list(state_dict.keys())
        del state_dict
    else:
        raise FileNotFoundError(f"No checkpoint found at {model_path}")

    # 2. Apply _checkpoint_conversion_mapping (same logic as hf.py _remap_param_names_load_hook)
    # This is the key part - the factory does this exact same thing in lines 496-512 of hf.py
    conversion_mapping = getattr(model, "_checkpoint_conversion_mapping", None)
    checkpoint_keys_remapped = []

    for key in checkpoint_keys_original:
        new_key = key
        if conversion_mapping:
            for pattern, replacement in conversion_mapping.items():
                new_key = re.sub(pattern, replacement, new_key)
        checkpoint_keys_remapped.append(new_key)

    # 3. Get model's expected keys
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(checkpoint_keys_remapped)

    # 4. Calculate differences
    loaded_keys = checkpoint_keys & model_keys
    missing_in_checkpoint = model_keys - checkpoint_keys
    unexpected_in_checkpoint = checkpoint_keys - model_keys

    return loaded_keys, missing_in_checkpoint, unexpected_in_checkpoint


def test_eagle_model_with_weights():
    """Test EagleModel forward pass with loaded weights using the EagleDrafterFactory.

    This test uses EagleDrafterFactory to initialize the model, which directly
    builds the Eagle drafter model based on the checkpoint's model_type:

    1. Factory creates config via AutoConfig.from_pretrained
    2. Factory selects EagleDrafterForCausalLM based on model_type="llama"
    3. Factory creates model via _from_config
    4. Factory loads weights via load_or_random_init -> _load_checkpoint

    This ensures the test validates the exact initialization path used in production.
    """
    print("\n" + "=" * 80)
    print("Test: EagleModel forward pass with loaded weights (via EagleDrafterFactory)")
    print("=" * 80)

    _, eagle_model_path = get_model_paths()
    eagle_path = Path(eagle_model_path)

    # 1. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Create factory
    # EagleDrafterFactory directly builds the correct drafter model based on model_type
    print("Creating EagleDrafterFactory...")
    factory = EagleDrafterFactory(
        model=eagle_model_path,
        skip_loading_weights=False,  # We want to test weight loading
    )

    # 3. Build model using factory
    # Factory flow:
    #   build_model() -> prefetch_checkpoint() -> _build_model()
    #   _build_model() -> _get_model_config() (gets base LlamaConfig)
    #   _build_model() -> selects EagleDrafterForCausalLM for model_type="llama"
    #   _build_model() -> EagleDrafterForCausalLM._from_config(config)
    print("Building model via factory.build_model('meta')...")
    model = factory.build_model("meta")
    print(f"Model type: {type(model).__name__}")
    print(f"Model config type: {type(model.config).__name__}")

    # 4. Load weights from checkpoint and compare to model's expected keys
    print("\n--- Weight Loading Analysis ---")
    loaded_keys, missing_keys, unexpected_keys = load_weights(eagle_path, model)

    print(f"Total model parameters: {len(loaded_keys) + len(missing_keys)}")
    print(f"Total checkpoint keys: {len(loaded_keys) + len(unexpected_keys)}")
    print(f"✅ Weights to be loaded: {len(loaded_keys)}")
    print(f"⚠️  Missing in checkpoint (will be random init): {len(missing_keys)}")
    print(f"⚠️  Unexpected in checkpoint (will be ignored): {len(unexpected_keys)}")

    if unexpected_keys:
        print("\nUnexpected keys (in checkpoint but model doesn't expect):")
        for key in sorted(unexpected_keys):
            if "t2d" in key:
                print(f"  - {key} (expected: not used in Eagle3 for Llama3.1-8B-Instruct)")
            else:
                print(f"  - {key}")

    if loaded_keys:
        print(f"\nLoaded keys ({len(loaded_keys)} total):")
        for key in sorted(loaded_keys)[:20]:
            print(f"  - {key}")
        if len(loaded_keys) > 20:
            print(f"  ... and {len(loaded_keys) - 20} more")

    print("--- End Weight Analysis ---\n")

    # Verify expected missing and unexpected keys
    # These are the keys we expect based on Eagle3 architecture:
    # - embed_tokens: shared from target model (not in Eagle checkpoint)
    # - t2d: target-to-draft mapping, not used in Eagle3 (uses d2t instead)
    expected_unexpected_keys = {"model.t2d"}

    assert len(missing_keys) == 0, (
        f"Expect all keys to be loaded.\nKeys that are missing: {missing_keys}\n"
    )

    assert unexpected_keys == expected_unexpected_keys, (
        f"Unexpected keys in checkpoint.\n"
        f"Expected: {expected_unexpected_keys}\n"
        f"Got: {unexpected_keys}\n"
        f"Extra unexpected: {unexpected_keys - expected_unexpected_keys}\n"
        f"Not unexpected (but expected): {expected_unexpected_keys - unexpected_keys}"
    )

    print("✅ Weight loading analysis matches expected missing/unexpected keys!")

    # 5. Load weights using factory (mimics actual pipeline)
    # If tensor shapes do not match with how they are used in the forward() function, we will
    # get an error.
    print("Loading weights via factory.load_or_random_init()...")
    factory.load_or_random_init(model, device)
    print("Weights loaded successfully via factory interface!")

    model.eval()


###############################################################################
# Set up to test the prefill-only version of the EagleWrapper model in test_eagle_wrapper_forward().
# This helps us guarantee that the EagleWrapper model, before it enters AutoDeploy, is working correctly,
# The test does not rely on any TRTLLM logic.
###############################################################################
class PrefillOnlyEagleResourceManager:
    """Simple resource manager for Eagle speculative decoding (prefill-only variant).

    Stores hidden states for use by draft loop in EagleWrapper.forward().
    """

    def __init__(
        self,
        hidden_size: int,
        num_capture_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        max_draft_len: int,
        target_dtype: torch.dtype,
    ):
        # Buffer for hidden states from target model: [max_tokens, hidden_size * num_capture_layers]
        # Uses flattened 2D format to match ADHiddenStateManager
        self.hidden_states = torch.empty(
            max_batch_size * (max_seq_len + max_draft_len),
            hidden_size * num_capture_layers,
            device="cuda",
            dtype=target_dtype,
        )


class LlamaModelWithCapture(LlamaModel):
    """LlamaModel that captures un-normalized hidden states from specified layers.

    Overwrites the base model's forward method to capture hidden states from specified layers.
    Base model's forward method is otherwise copied from LlamaModel in HuggingFace.
    Takes PrefillOnlyEagleResourceManager as an argument to store captured hidden states.
    """

    def __init__(
        self,
        config,
        layers_to_capture: Optional[Set[int]] = None,
        resource_manager: Optional[PrefillOnlyEagleResourceManager] = None,
    ):
        super().__init__(config)
        # layers_to_capture: set of layer indices (0-indexed) to capture
        # If None, capture all layers
        if layers_to_capture is None:
            self.layers_to_capture = set(range(config.num_hidden_layers))
        else:
            self.layers_to_capture = set(layers_to_capture)

        self.resource_manager = resource_manager

        # Validate layer indices
        for idx in self.layers_to_capture:
            if idx < 0 or idx >= config.num_hidden_layers:
                raise ValueError(
                    f"Layer index {idx} out of range. "
                    f"Model has {config.num_hidden_layers} layers (0 to {config.num_hidden_layers - 1})"
                )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            # prefill only - no past key values.
            cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Buffer to collect captured hidden states
        captured_hidden_states = []

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            # Capture this layer's output if it's in our list
            if layer_idx in self.layers_to_capture:
                captured_hidden_states.append(hidden_states)

        # Apply final normalization for last_hidden_state
        last_hidden_state = self.norm(hidden_states)

        # Store captured hidden states in resource manager if available
        # Resource manager uses 2D flattened format: [max_tokens, hidden_size * num_capture_layers]
        if self.resource_manager is not None and captured_hidden_states:
            concatenated = torch.cat(captured_hidden_states, dim=-1)
            batch_size, seq_len, total_hidden_size = concatenated.shape
            assert self.resource_manager.hidden_states.shape[-1] == total_hidden_size, (
                f"Resource manager buffer last dim {self.resource_manager.hidden_states.shape[-1]} "
                f"!= concatenated hidden states last dim {total_hidden_size}"
            )
            # Flatten to [batch_size * seq_len, total_hidden_size] for 2D format
            flattened = concatenated.view(batch_size * seq_len, total_hidden_size)
            self.resource_manager.hidden_states[: (batch_size * seq_len), :].copy_(flattened)

        return BaseModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            hidden_states=tuple(captured_hidden_states) if captured_hidden_states else None,
        )


@dataclass
class LlamaForCausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None


class LlamaForCausalLMWithCapture(nn.Module):
    """Wrapper combining LlamaModelWithCapture with lm_head for EagleWrapper testing.

    EagleWrapper.forward() expects target_model(input_ids, position_ids) to return logits.
    This class wraps LlamaModelWithCapture (which captures hidden states to resource manager)
    and adds the lm_head to produce logits.
    """

    def __init__(self, base_model, capture_model):
        super().__init__()
        self.model = capture_model  # LlamaModelWithCapture with resource_manager
        self.lm_head = base_model.lm_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids, inputs_embeds=inputs_embeds, position_ids=position_ids, **kwargs
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return LlamaForCausalLMOutput(
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.model.lm_head

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        resource_manager,
        capture_layers,
        dtype=torch.bfloat16,
    ):
        """Load a base model and create a LlamaForCausalLMWithCapture with shared weights."""
        print(f"Loading {model_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map={"": 0},
        )
        base_model.eval()

        # Create LlamaModelWithCapture that shares weights with the base model
        original_llama_model = base_model.model

        capture_model = LlamaModelWithCapture.__new__(LlamaModelWithCapture)
        nn.Module.__init__(capture_model)

        capture_model.config = original_llama_model.config
        capture_model.layers_to_capture = capture_layers
        capture_model.resource_manager = resource_manager

        # Share all modules (no weight copying)
        capture_model.embed_tokens = original_llama_model.embed_tokens
        capture_model.layers = original_llama_model.layers
        capture_model.norm = original_llama_model.norm
        capture_model.rotary_emb = original_llama_model.rotary_emb
        capture_model.gradient_checkpointing = original_llama_model.gradient_checkpointing

        return cls(base_model, capture_model)


def build_eagle_wrapper(
    base_model_path: str,
    eagle_model_path: str,
    resource_manager: PrefillOnlyEagleResourceManager,
    capture_layers: Set[int],
    max_seq_len: int,
    max_draft_len: int,
    target_dtype: torch.dtype,
    device: torch.device,
) -> tuple[EagleWrapper, nn.Module]:
    """Build an EagleWrapper model for testing.

    This function encapsulates the model building logic using manual model building.

    Returns:
        A tuple of (eagle_wrapper, target_model) where:
            - eagle_wrapper: The EagleWrapper model ready for inference.
            - target_model: The target model (for verification steps).
    """
    # Build EagleWrapper manually.
    print("\n" + "-" * 40)
    print("Building EagleWrapper")
    print("-" * 40)

    # Create target model with capture
    target_model = LlamaForCausalLMWithCapture.from_pretrained(
        base_model_path, resource_manager, capture_layers, target_dtype
    )
    print("✓ Created target model with capture")

    # Create draft model using EagleDrafterFactory (mimics production pipeline)
    # This ensures weights are loaded correctly via the same path as AutoDeploy
    print("\nCreating draft model via EagleDrafterFactory...")
    draft_factory = EagleDrafterFactory(
        model=eagle_model_path,
        skip_loading_weights=False,
    )

    # Build model on meta device first, then load weights
    draft_model = draft_factory.build_model("meta")
    print(f"  Model type: {type(draft_model).__name__}")

    # Load weights via factory
    print("  Loading weights via factory.load_or_random_init()...")
    draft_factory.load_or_random_init(draft_model, device)
    draft_model.eval()

    # Create EagleWrapper config
    wrapper_config = EagleWrapperConfig(
        max_draft_len=max_draft_len,
        load_embedding_from_target=draft_model.load_embedding_from_target,
        load_lm_head_from_target=draft_model.load_lm_head_from_target,
    )

    # Build EagleWrapper (this also loads weights from target into draft model where necessary)
    eagle_wrapper = EagleWrapper(
        config=wrapper_config,
        target_model=target_model,
        draft_model=draft_model,
        resource_manager=resource_manager,
    )
    eagle_wrapper.eval()
    print("✓ Built EagleWrapper")

    return eagle_wrapper, target_model


def generate_target_outputs(
    target_model: nn.Module,
    input_ids: torch.Tensor,
    num_iterations: int,
) -> torch.Tensor:
    """Generate tokens from target model using greedy sampling.

    Runs target_model.forward() in a loop, taking the last logit from each output,
    applying greedy sampling with torch.argmax, and appending to input_ids.

    Args:
        target_model: Model that returns logits from forward(input_ids, position_ids).
        input_ids: Initial input token ids of shape [batch_size, seq_len].
        num_iterations: Number of tokens to generate.

    Returns:
        output_ids: Tensor of shape [batch_size, seq_len + num_iterations] containing
            the original input_ids plus the generated tokens.
    """
    device = input_ids.device
    init_seq_len = input_ids.shape[1]
    print(f"Initial sequence length: {init_seq_len}")
    current_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(num_iterations):
            # Generate position_ids from current sequence length
            seq_len = current_ids.shape[1]
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
            position_ids = position_ids.expand(current_ids.shape[0], -1)

            # Forward pass
            logits = target_model(current_ids, position_ids=position_ids).logits

            # Take the last logit and apply greedy sampling
            last_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)  # [batch_size, 1]

            # Append to current_ids
            current_ids = torch.cat([current_ids, next_token], dim=1)

        return current_ids


def print_token_analysis(
    input_ids: torch.Tensor,
    num_previously_accepted: torch.Tensor,
    target_output_ids: torch.Tensor,
    tokenizer,
) -> None:
    """Print debug analysis of accepted vs speculative tokens for each batch.

    Args:
        input_ids: Current input token ids of shape [batch_size, seq_len].
        num_previously_accepted: Number of accepted tokens per batch [batch_size].
        target_output_ids: Reference output from target model [batch_size, total_seq_len].
        tokenizer: Tokenizer for decoding tokens to text.
    """
    batch_size = input_ids.shape[0]
    print("\n  --- Token Analysis (per batch) ---")

    for i in range(batch_size):
        prev_accepted_i = num_previously_accepted[i].item()

        # Accepted tokens (before speculation): input_ids[i, :num_previously_accepted[i]]
        accepted_tokens = input_ids[i, :prev_accepted_i]
        # Speculative tokens: input_ids[i, num_previously_accepted[i]:]
        speculative_tokens = input_ids[i, prev_accepted_i:]

        # Target model's expected token at this position
        if prev_accepted_i < target_output_ids.shape[1]:
            target_token_at_pos = target_output_ids[i, prev_accepted_i]
        else:
            target_token_at_pos = None

        print(f"\n  Batch {i}:")
        print(f"    num_previously_accepted: {prev_accepted_i}")
        print(
            f"    Accepted tokens ({accepted_tokens.shape[0]} tokens): {accepted_tokens.tolist()}"
        )
        accepted_text = tokenizer.decode(accepted_tokens, skip_special_tokens=True)
        print(f'    Accepted text: "{accepted_text}"')
        print(
            f"    Speculative tokens ({speculative_tokens.shape[0]} tokens): {speculative_tokens.tolist()}"
        )
        if speculative_tokens.shape[0] > 0:
            spec_text = tokenizer.decode(speculative_tokens, skip_special_tokens=False)
            print(f'    Speculative text: "{spec_text}"')
        if target_token_at_pos is not None:
            target_tok_id = target_token_at_pos.item()
            target_tok_str = tokenizer.decode([target_tok_id])
            print(
                f'    Target model\'s next token at pos {prev_accepted_i}: {target_tok_id} ("{target_tok_str}")'
            )


def manual_sample_and_verify(
    next_target_inputs: list,
    num_accepted_tokens: torch.Tensor,
    target_model: nn.Module,
    eagle_wrapper: nn.Module,
    max_draft_len: int,
    device: torch.device,
) -> list:
    """Manually verify speculative tokens using sample_and_verify.

    This is used for batch_size > 1 where truncation prevents speculative tokens
    from being fed back, so we verify them manually before truncation.

    Args:
        next_target_inputs: List of tensors, one per batch element.
        num_accepted_tokens: Number of tokens accepted so far per batch [batch_size].
        target_model: The target model for running forward pass.
        eagle_wrapper: The EagleWrapper containing sample_and_verify.
        max_draft_len: Maximum draft length (for capping counts).
        device: Device to run on.

    Returns:
        List of (num_accepted, num_speculative) tuples for each batch element.
    """
    batch_size = len(next_target_inputs)

    # Due to our truncation trick, all sequences should have the same length
    seq_lens = [seq.shape[0] for seq in next_target_inputs]
    assert all(slen == seq_lens[0] for slen in seq_lens), (
        f"All sequences should have same length due to truncation, got {seq_lens}"
    )
    verify_seq_len = seq_lens[0]

    # Stack into batched tensor
    stacked_inputs = torch.stack(next_target_inputs, dim=0)  # [batch_size, seq_len]

    # Run target model forward to get logits
    verify_position_ids = (
        torch.arange(verify_seq_len, device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    with torch.no_grad():
        verify_target_logits = target_model(stacked_inputs, position_ids=verify_position_ids).logits

    # new_num_previously_accepted = num_accepted_tokens + 1
    # This represents the tokens accepted after target model's output from this iteration
    new_num_previously_accepted = num_accepted_tokens + 1

    # Call sample_and_verify to get acceptance counts
    _, verify_newly_accepted, _, _ = eagle_wrapper.sample_and_verify(
        stacked_inputs, verify_target_logits, new_num_previously_accepted
    )

    # Build results list
    results = []
    for i in range(batch_size):
        num_accepted_i = min(verify_newly_accepted[i].item(), max_draft_len)
        num_speculative = next_target_inputs[i].shape[0] - new_num_previously_accepted[i].item()
        results.append((num_accepted_i, num_speculative))

    return results


def verify_eagle_wrapper_output(output, tokenizer, batch_size, num_previously_accepted):
    """Verify the output structure and values from EagleWrapper forward pass.

    Args:
        output: The output from EagleWrapper forward pass.
        tokenizer: The tokenizer for decoding tokens.
        batch_size: The batch size.
        num_previously_accepted: Tensor of previously accepted token counts.
    """
    # Verify output structure
    print("\nOutput verification:")
    assert output is not None, "Output should not be None"
    assert hasattr(output, "new_tokens"), "Output should have new_tokens"
    assert hasattr(output, "new_tokens_lens"), "Output should have new_tokens_lens"

    print(f"  new_tokens: {type(output.new_tokens)} with {len(output.new_tokens)} items")
    for i, tokens in enumerate(output.new_tokens):
        new_tokens_text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"    batch {i}: shape {tokens.shape}, tokens: {tokens.tolist()}")
        print(f'    batch {i}: decoded: "{new_tokens_text}"')

    # Compute num_accepted_tokens from new_tokens_lens + num_previously_accepted
    num_accepted_tokens = num_previously_accepted + output.new_tokens_lens

    print(f"  new_tokens_lens: {output.new_tokens_lens}")
    print(f"  num_accepted_tokens (computed): {num_accepted_tokens}")

    # Verify new_tokens_lens is within expected bounds
    assert output.new_tokens_lens.shape == (batch_size,), (
        f"new_tokens_lens shape should be ({batch_size},), got {output.new_tokens_lens.shape}"
    )


@pytest.mark.skip(
    reason="EagleWrapper interface was refactored (resource_manager removed from __init__, "
    "sample_and_verify removed); test needs to be updated to match the new interface. "
    "This test is valuable for validating Eagle3 correctness (acceptance ratio) directly "
    "on the EagleWrapper model *before* the full export + transforms + KV-cache pipeline, "
    "making it much easier to debug Eagle3 model issues in isolation. TODO: rewrite to "
    "match the current EagleWrapper prefill-only and KV-cache forward interfaces."
)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_eagle_wrapper_forward(batch_size: int):
    """Test EagleWrapper forward pass with target and draft models.

    This test validates the full speculative decoding loop:
    1. Target model processes input and captures hidden states
    2. Draft model generates speculative tokens
    3. EagleWrapper orchestrates verification and drafting

    For batch size 1, we call EagleWrapper forward in the expected way. Each iteration generates a "golden token"
    (target output) and draft tokens. We input all of them to the wrapper model,
    which verifies the draft tokens against the target output. It then outputs the accepted tokens
    and newly generated draft tokens, along with numbers of accepted tokens, and the process repeats.

    For batch size > 1, we need to work around the fact that as we run the loop described above, the sequences lengths
    in the batch will get out of sync. So instead, we do not provide validated draft tokens as input in each iteration -
    we just input the first accepted token from the previous iteration
    (which we know was generated by the target model), which keeps the batches in sync.

    To verify that the output draft tokens are reasonable, we run a manual target model verification step
    after each iteration. We record how many of the output draft tokens were accepted.

    In the end, we test that the acceptance ratio of the draft tokens generated by the EagleWrapper is reasonable.

    Args:
        batch_size: Number of prompts to process in parallel.
    """
    print("\n" + "=" * 80)
    print("Test: EagleWrapper forward pass")
    print("=" * 80)

    # Set random seeds for reproducibility
    torch.manual_seed(42)

    # Get model paths using integration test conventions
    base_model_path, eagle_model_path = get_model_paths()
    eagle_path = Path(eagle_model_path)

    # Configuration
    capture_layers = {1, 15, 28}  # Layers to capture for Eagle3
    num_capture_layers = len(capture_layers)
    hidden_size = 4096  # Llama 3.1-8B hidden size
    dtype = torch.bfloat16
    device = torch.device("cuda")

    # Test dimensions
    max_batch_size = 4
    max_seq_len = 1024
    max_draft_len = 3

    # Tokenize the test prompts
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    # Llama uses left padding for batch inference
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if batch_size == 1:
        input_ids = tokenizer.encode(prompts[0], return_tensors="pt").to(device)
    else:
        tokenized = tokenizer(
            prompts[:batch_size],
            return_tensors="pt",
            padding=True,
        )
        input_ids = tokenized.input_ids.to(device)

    print(f"input_ids: {input_ids}")
    seq_len = input_ids.shape[1]
    init_seq_len = seq_len  # Store initial sequence length for final comparison

    print("\nTest configuration:")
    print(f"  target_model: {base_model_path}")
    print(f"  eagle_model: {eagle_path}")
    print(f"  batch_size: {batch_size}, seq_len: {seq_len}")
    print(f"  max_draft_len: {max_draft_len}")
    print(f"  capture_layers: {capture_layers}")
    print(f"  prompts: {prompts[:batch_size]}")
    print(f"  input_ids: {input_ids}")

    # Create resource manager
    resource_manager = PrefillOnlyEagleResourceManager(
        hidden_size=hidden_size,
        num_capture_layers=num_capture_layers,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        max_draft_len=max_draft_len,
        target_dtype=dtype,
    )
    print("\n✓ Created resource manager")
    print(f"  target_hidden_states shape: {resource_manager.hidden_states.shape}")

    # Build eagle_wrapper and target_model using the refactored function
    eagle_wrapper, target_model = build_eagle_wrapper(
        base_model_path=base_model_path,
        eagle_model_path=str(eagle_path),
        resource_manager=resource_manager,
        capture_layers=capture_layers,
        max_seq_len=max_seq_len,
        max_draft_len=max_draft_len,
        target_dtype=dtype,
        device=device,
    )

    # Create test inputs (input_ids already created from tokenizer above)
    position_ids = (
        torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    )
    # Set previously_accepted_tokens to the length of input_ids (all context tokens are accepted)
    # Shape should be [batch_size] - a 1D tensor with one value per batch
    num_previously_accepted = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)

    print("\nTest inputs:")
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  input_ids: {input_ids}")
    print(f"  position_ids shape: {position_ids.shape}")
    print(f"  num_previously_accepted: {num_previously_accepted}")

    # Generate target model outputs with greedy sampling
    print("\nGenerating target model outputs with greedy sampling (for verification)...")
    target_output_ids = generate_target_outputs(target_model, input_ids, num_iterations=100)
    print(f"  target_output_ids shape: {target_output_ids.shape}")
    print(f"  target_output_ids: {target_output_ids}")

    # Decode to text as sanity check
    generated_text = tokenizer.decode(target_output_ids[0], skip_special_tokens=True)
    print(f"\n  Target model greedy generation decoded text:\n  {generated_text}")

    print("\n✓ EagleWrapper forward pass completed successfully!")
    print("✓ Output structure verified")
    print("✓ new_tokens_lens within expected bounds")
    print("✓ Target model greedy generation completed")

    print("\n================================================")

    num_iterations = 70

    # Dictionary to track distribution of new_tokens_lens
    # keys: 0 to max_draft_len
    # newly_accepted_counts[i]: number of times the number of accepted draft tokens was i
    newly_accepted_counts = {i: 0 for i in range(max_draft_len + 1)}

    for iteration in range(num_iterations):
        print(f"\n{'=' * 40}")
        print(f"EagleWrapper forward pass - Iteration {iteration + 1}/{num_iterations}")
        print(f"{'=' * 40}")

        seq_len = input_ids.shape[1]

        # Debug: Print speculative tokens, accepted tokens, and target comparison
        print_token_analysis(input_ids, num_previously_accepted, target_output_ids, tokenizer)

        kwargs = {
            "num_previously_accepted": num_previously_accepted,
        }
        with torch.no_grad():
            output = eagle_wrapper(
                input_ids=input_ids,
                position_ids=position_ids,
                **kwargs,
            )

        verify_eagle_wrapper_output(output, tokenizer, batch_size, num_previously_accepted)

        # Prepare next_target_inputs
        # output.new_tokens[i] contains the full draft_input_ids tensor, but the valid prefix
        # has length num_accepted_tokens[i] + max_draft_len. We slice to get only valid tokens.
        # We then prepend the first token from the previous iteration's input_ids.
        # This prepending is only needed for prefill-only mode, since in the cached case, the first token
        # will always be in the KV cache.
        # Compute num_accepted_tokens from num_previously_accepted + new_tokens_lens
        num_accepted_tokens = num_previously_accepted + output.new_tokens_lens
        valid_prefix_len = num_accepted_tokens + max_draft_len
        next_target_inputs = [
            torch.cat(
                (input_ids[i, 0].unsqueeze(0), output.new_tokens[i][: valid_prefix_len[i]]),
                dim=0,
            )
            for i in range(batch_size)
        ]

        # Track distribution of newly accepted tokens by reading new_tokens_lens from the output.
        # For batch size = 1, we are inputting draft tokens to the wrapper model, so new_tokens_lens
        # gives the number of accepted tokens from drafts in the previous iteration.
        if batch_size == 1:
            for val in output.new_tokens_lens.tolist():
                newly_accepted_counts[val] += 1
            print(f"  newly_accepted_counts so far: {newly_accepted_counts}")

        # For batch_size > 1, we use manual target model verification below instead to check which of the draft tokens
        # generated in *this* iteration would be accepted by the target model.
        else:
            # For batch_size > 1, verify acceptance using sample_and_verify()
            # before truncation (since truncation prevents speculative tokens from being fed back)
            verify_results = manual_sample_and_verify(
                next_target_inputs,
                num_accepted_tokens,
                target_model,
                eagle_wrapper,
                max_draft_len,
                device,
            )

            # Update newly_accepted_counts map
            for i, (num_accepted_i, num_speculative) in enumerate(verify_results):
                newly_accepted_counts[num_accepted_i] += 1
                print(
                    f"  [Batch {i}] sample_and_verify: {num_accepted_i}/{num_speculative} speculative accepted"
                )

            # Truncate to keep shapes consistent across batches in each iteration.
            # We know that the first token that is generated in this iteration is accepted, so it is "safe".
            # All speculative tokens are truncated regardless of whether they are accepted or not.
            # This is a hack to prevent the sequence lengths from getting out of sync across batches in each iteration
            # without needing to change the padding every iteration.
            truncate_len = input_ids.shape[1] + 1
            next_target_inputs = [seq[:truncate_len] for seq in next_target_inputs]

        next_target_inputs = torch.stack(next_target_inputs, dim=0)

        print(f"  next_target_inputs: {next_target_inputs}")
        print(f"  next_target_inputs.shape: {next_target_inputs.shape}")

        # Update for next iteration
        input_ids = next_target_inputs
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)

        if batch_size > 1:
            # For multi-batch: increment by 1 (we truncated, so just advance by one token)
            num_previously_accepted = num_previously_accepted + 1
        else:
            # For single batch: accept the tokens accepted in the previous iteration, plus one
            # for the output token that was generated by the target.
            num_previously_accepted = num_accepted_tokens + 1

    print(f"\n{'=' * 40}")
    print(f"Loop completed: {num_iterations} iterations")
    print("Newly accepted tokens distribution:")
    for k, v in newly_accepted_counts.items():
        print(f"  {k}: {v}")

    # Calculate acceptance ratio
    # For batch_size == 1: uses new_tokens_lens from eagle wrapper
    # For batch_size > 1: uses manual verification against target model (since truncation
    #                     prevents speculative tokens from being fed back)
    total_accepted = sum(k * v for k, v in newly_accepted_counts.items())
    # First iteration has no tokens to newly accept, subsequent iterations have max_draft_len potential

    num_iterations_with_drafts = num_iterations - 1 if batch_size == 1 else num_iterations
    total_potential = max_draft_len * (num_iterations_with_drafts) * batch_size
    acceptance_ratio = total_accepted / total_potential if total_potential > 0 else 0.0
    print(f"\nAcceptance ratio: {total_accepted}/{total_potential} = {acceptance_ratio:.3f}")
    if batch_size > 1:
        print("  (batch_size > 1: measured via manual target model verification)")
    assert acceptance_ratio > 0.1, (
        f"Acceptance ratio {acceptance_ratio:.3f} is too low (expected > 0.1)"
    )

    print("\n" + "=" * 80)
    print("FINAL OUTPUT COMPARISON")
    print("=" * 80)
    for i in range(batch_size):
        print(f"\n{'─' * 40}")
        print(f"BATCH {i}")
        print(f"{'─' * 40}")
        print(f"\n[Target Model Output] ({target_output_ids[i].shape[0]} tokens):")
        print(f"  Tokens: {target_output_ids[i].tolist()}")
        print(f'  Text: "{tokenizer.decode(target_output_ids[i], skip_special_tokens=True)}"')
        print(f"\n[Eagle Wrapper Output] ({input_ids[i].shape[0]} tokens):")
        print(f"  Tokens: {input_ids[i].tolist()}")
        print(f'  Text: "{tokenizer.decode(input_ids[i], skip_special_tokens=True)}"')
    print("\n" + "=" * 80)

    # Verify that the first 10 generated tokens match between target model and eagle wrapper
    # They seem to diverge after awhile but are semantically the same.
    # Note that even running the target model in decode vs prefill mode, the outputs seem to diverge similarly,
    # so this is not worrisome. This test provides a check that they are "similar enough" to each other.
    num_tokens_to_check = 10
    print(f"\nVerifying first {num_tokens_to_check} generated tokens match...")
    for i in range(batch_size):
        target_generated = target_output_ids[i, init_seq_len : init_seq_len + num_tokens_to_check]
        eagle_generated = input_ids[i, init_seq_len : init_seq_len + num_tokens_to_check]

        print(f"  Batch {i}:")
        print(f"    Target: {target_generated.tolist()}")
        print(f"    Eagle:  {eagle_generated.tolist()}")

        assert torch.equal(target_generated, eagle_generated), (
            f"Batch {i}: First {num_tokens_to_check} generated tokens do not match!\n"
            f"  Target: {target_generated.tolist()}\n"
            f"  Eagle:  {eagle_generated.tolist()}"
        )
    print(f"✓ First {num_tokens_to_check} generated tokens match for all batches!")


def _load_valid_safetensors_index(index_path: Path):
    """Load a safetensors index JSON, skipping invalid and Git-LFS pointer files."""
    if not index_path.exists():
        return None

    try:
        index_text = index_path.read_text(encoding="utf-8")
    except OSError:
        return None

    if index_text.lstrip().startswith("version https://git-lfs.github.com/spec/v1"):
        return None

    try:
        index = json.loads(index_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(index, dict):
        return None

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        return None

    return index


def _analyze_mtp_weight_loading(model_path: Path, model):
    """Analyze weight loading for MTP models with safetensors index.

    MTP checkpoints use multiple safetensors files with an index. This function
    loads the checkpoint keys from the index and applies the model's
    _checkpoint_conversion_mapping to determine which keys will be loaded.

    Args:
        model_path: Path to the MTP model directory
        model: The instantiated model

    Returns:
        Tuple of (loaded_keys, missing_keys, unexpected_keys)
    """
    # Load checkpoint keys from safetensors index
    index_path = model_path / "model.safetensors.index.json"
    index = _load_valid_safetensors_index(index_path)
    if index is None:
        raise ValueError(
            "Expected a valid safetensors index JSON. "
            f"Path was missing, malformed, or a Git-LFS pointer: {index_path}"
        )

    # Get MTP-specific checkpoint keys (those starting with "mtp.")
    checkpoint_keys_original = [k for k in index["weight_map"].keys() if k.startswith("mtp.")]
    if not checkpoint_keys_original:
        raise ValueError(f"No mtp.* keys found in safetensors index: {index_path}")

    # Apply _checkpoint_conversion_mapping (same logic as hf.py _remap_param_names_load_hook)
    conversion_mapping = getattr(model, "_checkpoint_conversion_mapping", None)
    checkpoint_keys_remapped = []

    for key in checkpoint_keys_original:
        new_key = key
        if conversion_mapping:
            for pattern, replacement in conversion_mapping.items():
                new_key = re.sub(pattern, replacement, new_key)
        checkpoint_keys_remapped.append(new_key)

    # Get model's expected keys
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(checkpoint_keys_remapped)

    # Calculate differences
    loaded_keys = checkpoint_keys & model_keys
    missing_in_checkpoint = model_keys - checkpoint_keys
    unexpected_in_checkpoint = checkpoint_keys - model_keys

    return loaded_keys, missing_in_checkpoint, unexpected_in_checkpoint


def test_nemotron_mtp_model_with_weights():
    """Test NemotronH MTP model weight loading using EagleDrafterFactory.

    This test verifies that:
    1. EagleDrafterFactory can create EagleDrafterForCausalLM with NemotronH layers
    2. Weights are correctly loaded with the mtp.* -> model.* key mapping
    3. All expected model parameters are loaded from the MTP checkpoint

    The MTP model uses a checkpoint that contains both backbone.* and mtp.* keys.
    Only mtp.* keys are loaded. Shared parameters (embed_tokens, lm_head) are NOT
    created in the model (load_embedding_from_target=True, load_lm_head_from_target=True),
    so they don't appear as missing keys. They are shared from the target model at runtime.
    """
    print("\n" + "=" * 80)
    print("Test: NemotronH MTP model weight loading (via EagleDrafterFactory)")
    print("=" * 80)

    mtp_model_name = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"

    mtp_model_path = hf_id_to_local_model_dir(mtp_model_name)
    mtp_path = Path(mtp_model_path)
    index_path = mtp_path / "model.safetensors.index.json"

    # Check for a valid index JSON and verify it has mtp.* keys.
    index = _load_valid_safetensors_index(index_path)
    assert index is not None, (
        "Expected a valid safetensors index JSON. "
        f"Path was missing, malformed, or a Git-LFS pointer: {index_path}"
    )
    mtp_source_keys = {k for k in index["weight_map"].keys() if k.startswith("mtp.")}
    assert mtp_source_keys, f"Expected at least one mtp.* key in {index_path}"

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create factory - use EagleDrafterFactory for NemotronH MTP
    print("Creating EagleDrafterFactory...")
    factory = EagleDrafterFactory(
        model=mtp_model_path,
        skip_loading_weights=False,
    )

    # Build model using factory
    print("Building model via factory.build_model('meta')...")
    model = factory.build_model("meta")
    print(f"Model type: {type(model).__name__}")
    print(f"Model config type: {type(model.config).__name__}")

    # Verify model type is EagleDrafterForCausalLM
    assert isinstance(model, EagleDrafterForCausalLM), (
        f"Expected EagleDrafterForCausalLM, got {type(model).__name__}"
    )

    # Analyze weight loading
    print("\n--- Weight Loading Analysis ---")
    loaded_keys, missing_keys, unexpected_keys = _analyze_mtp_weight_loading(mtp_path, model)

    print(f"Total model parameters: {len(loaded_keys) + len(missing_keys)}")
    print(f"Total MTP checkpoint keys: {len(loaded_keys) + len(unexpected_keys)}")
    print(f"✅ Weights to be loaded: {len(loaded_keys)}")
    print(f"⚠️  Missing in checkpoint (should be 0): {len(missing_keys)}")
    print(f"⚠️  Unexpected in checkpoint (should be 0): {len(unexpected_keys)}")

    if missing_keys:
        print("\nMissing keys (expected - shared from target model):")
        for key in sorted(missing_keys):
            if "embed_tokens" in key:
                print(f"  - {key} (shared embedding from target)")
            elif "lm_head" in key:
                print(f"  - {key} (shared lm_head from target)")
            else:
                print(f"  - {key}")

    if unexpected_keys:
        print("\nUnexpected keys (should not happen):")
        for key in sorted(unexpected_keys):
            print(f"  - {key}")

    print("--- End Weight Analysis ---\n")

    # Verify expected missing and unexpected keys
    # MTP checkpoint does NOT contain embed_tokens or lm_head, but that's OK because:
    # - embed_tokens: shared from target model (load_embedding_from_target=True → model doesn't create it)
    # - lm_head: shared from target model (load_lm_head_from_target=True → model doesn't create it)
    # Since neither parameter is created in the model, they don't appear as missing keys.
    # Note: For NemotronH, layers_handle_final_norm=True, so the wrapper doesn't create self.norm.
    # The final norm is inside the layers (final_layernorm), which IS in the checkpoint.
    expected_missing_keys = set()  # All model params are loaded; shared params aren't created
    expected_unexpected_keys = set()  # All checkpoint keys should be used

    assert missing_keys == expected_missing_keys, (
        f"Unexpected missing keys.\n"
        f"Expected: {expected_missing_keys}\n"
        f"Got: {missing_keys}\n"
        f"Extra missing: {missing_keys - expected_missing_keys}\n"
        f"Not missing (but expected): {expected_missing_keys - missing_keys}"
    )

    assert unexpected_keys == expected_unexpected_keys, (
        f"Unexpected keys in checkpoint.\n"
        f"Expected: {expected_unexpected_keys}\n"
        f"Got: {unexpected_keys}\n"
        f"Extra unexpected: {unexpected_keys - expected_unexpected_keys}"
    )

    print("✅ Weight loading analysis matches expected missing/unexpected keys!")

    # Load weights using factory
    print("Loading weights via factory.load_or_random_init()...")
    factory.load_or_random_init(model, device, disable_preload=True)
    print("Weights loaded successfully via factory interface!")

    model.eval()
    print("✅ NemotronH MTP model created and weights loaded successfully!")
