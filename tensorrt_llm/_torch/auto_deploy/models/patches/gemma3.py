"""A patch for Gemma3Model to pass token_type_ids to Gemma3TextModel and make mask functions export-compatible."""

from typing import List, Optional, Union

import torch
import torch.nn as nn

from transformers import masking_utils
from transformers.cache_utils import Cache
from transformers.models.gemma3 import modeling_gemma3
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Model,
    Gemma3ModelOutputWithPast,
    create_causal_mask,
    create_sliding_window_causal_mask,
    token_type_ids_mask_function,
)

from ...export.interface import BaseExportPatch, ExportPatchRegistry


def _noop_create_causal_mask(**kwargs):
    """Return None to skip vmap-based mask creation during export."""
    return None


def _noop_create_sliding_window_causal_mask(**kwargs):
    """Return None to skip vmap-based mask creation during export."""
    return None


def _gemma3_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **lm_kwargs,
) -> Union[tuple, Gemma3ModelOutputWithPast]:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Replace image id with PAD if the image token if OOV, to avoid index-errors
    if input_ids is not None and self.config.image_token_id >= self.vocab_size:
        special_image_mask = input_ids == self.config.image_token_id
        llm_input_ids = input_ids.clone()
        llm_input_ids[special_image_mask] = 0
    else:
        llm_input_ids = input_ids

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(llm_input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # Merge text and images
    image_features = None
    if pixel_values is not None:
        image_features = self.get_image_features(pixel_values)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        special_image_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_features
        )
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config.get_text_config(),
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        if token_type_ids is not None and inputs_embeds.shape[1] != 1:
            # We need to pass an additional mask function to account for token type ids, and it needs to be an `or`

            # First find where a new image block starts: 1 if image and previous not image
            # The images cannot attend to future images, but can attend to all prev images and to itself bidirectionally
            is_image = (token_type_ids == 1).to(cache_position.device)
            new_image_start = is_image & ~nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
            image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
            image_group_ids = torch.where(
                is_image,
                image_group_ids,
                torch.full_like(token_type_ids, -1, device=is_image.device),
            )
            mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                token_type_ids.to(cache_position.device),
                image_group_ids,
                self.config.mm_tokens_per_image,
            )

        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

    # Pass token_type_ids to language_model
    outputs = self.language_model(
        attention_mask=causal_mask_mapping,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        token_type_ids=token_type_ids,
        **lm_kwargs,
    )

    return Gemma3ModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values if use_cache else None,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
    )


@ExportPatchRegistry.register("hf_gemma3")
class Gemma3ModelPatch(BaseExportPatch):
    """Patch for Gemma3Model to pass token_type_ids and make mask functions export-compatible."""

    def _apply_patch(self):
        """Apply the Gemma3Model patch."""
        # Patch forward to pass token_type_ids to language_model
        self.original_values["Gemma3Model.forward"] = Gemma3Model.forward
        Gemma3Model.forward = _gemma3_model_forward

        # Patch mask functions to return None (avoids vmap incompatibility with torch.export)
        # We need to patch both masking_utils AND the module-level references in modeling_gemma3
        self.original_values["mu.create_causal_mask"] = masking_utils.create_causal_mask
        self.original_values["mu.create_sliding_window"] = (
            masking_utils.create_sliding_window_causal_mask
        )
        self.original_values["mg.create_causal_mask"] = modeling_gemma3.create_causal_mask
        self.original_values["mg.create_sliding_window"] = (
            modeling_gemma3.create_sliding_window_causal_mask
        )

        masking_utils.create_causal_mask = _noop_create_causal_mask
        masking_utils.create_sliding_window_causal_mask = _noop_create_sliding_window_causal_mask
        modeling_gemma3.create_causal_mask = _noop_create_causal_mask
        modeling_gemma3.create_sliding_window_causal_mask = _noop_create_sliding_window_causal_mask

    def _revert_patch(self):
        """Revert the Gemma3Model patch."""
        Gemma3Model.forward = self.original_values["Gemma3Model.forward"]
        masking_utils.create_causal_mask = self.original_values["mu.create_causal_mask"]
        masking_utils.create_sliding_window_causal_mask = self.original_values[
            "mu.create_sliding_window"
        ]
        modeling_gemma3.create_causal_mask = self.original_values["mg.create_causal_mask"]
        modeling_gemma3.create_sliding_window_causal_mask = self.original_values[
            "mg.create_sliding_window"
        ]
