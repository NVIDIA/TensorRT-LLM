"""A patch to handle vision branch in Llama4ForConditionalGeneration.

NOTE: most patches are not used at the moment since only text submodule is exported. Keeping it here
for future reference in case we decide to also export the image model.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Llama4ForConditionalGeneration
from transformers.models.llama4.modeling_llama4 import Llama4CausalLMOutputWithPast, Llama4TextMoe

from ...export.interface import BaseExportPatch, DisabledBaseExportPatch, ExportPatchRegistry


# Copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py#L1651
# With some modifications that won't affect current execution logic:
# 1. Vison branch managed by torch.cond to enable both text-only and text+image input during runtime.
# 2. Input arg `image_sizes` are set to none
#    as the input to torch.cond true/false branch needs fixed argument type during export
# 3. Do not return `image_hidden_states` as it is calculated inside the vision branch
#    and invisible to the function outside.
def _forward_with_cond(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[Union[int, List[int]]] = None,
    vision_feature_select_strategy: Optional[str] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    image_sizes: torch.Tensor = None,  # image_sizes set as None
    **lm_kwargs,
) -> Union[Tuple, Llama4CausalLMOutputWithPast]:
    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    vision_feature_layer = (
        vision_feature_layer
        if vision_feature_layer is not None
        else self.config.vision_config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_config.vision_feature_select_strategy
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if pixel_values is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
        )

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    def _vision_branch(inputs_embeds, pixel_values, input_ids):
        image_features = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_sizes=None,
        )

        vision_flat = image_features.view(-1, image_features.size(-1))
        projected_vision_flat = self.multi_modal_projector(vision_flat).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        # NOTE: get_placeholder_mask is not supported by torch.export due to numel check ###########
        # special_image_mask = self.get_placeholder_mask(
        #     input_ids, inputs_embeds=inputs_embeds, image_features=projected_vision_flat
        # )
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(
                    self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device
                )
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        # n_image_tokens = special_image_mask.sum()
        special_image_mask = (
            special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        )
        ### END OF get_placeholder_mask ############################################################

        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, projected_vision_flat)

        return inputs_embeds

    def _no_vision_branch(inputs_embeds, pixel_values, input_ids):
        # https://github.com/pytorch/pytorch/issues/158375
        return inputs_embeds.clone()

    # decide by whether there is any non-zero pixel_values
    has_image: torch.Tensor = torch.any(pixel_values != 0)

    inputs_embeds = torch.cond(
        has_image,
        _vision_branch,
        _no_vision_branch,
        (inputs_embeds, pixel_values, input_ids),
    )

    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        **lm_kwargs,
    )

    logits = outputs[0]

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        if attention_mask is not None:
            # we use the input attention mask to shift the logits and labels, because it is 2D.
            # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
            shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
            shift_logits = logits[..., :-1, :][
                shift_attention_mask.to(logits.device) != 0
            ].contiguous()
            shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1).to(shift_logits.device),
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Llama4CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=None,  # skip outputting this for simplicity
    )


# NOTE: registered as patch that is disabled by default since it is not used at the moment
@ExportPatchRegistry.register("hf_llama4_vision")
class Llama4VisionPatch(DisabledBaseExportPatch):
    """Patch for Llama4ForConditionalGeneration to make it compatible with torch.export.

    This patch replaces the forward method of Llama4ForConditionalGeneration with
    a version that uses the torch.cond to handle the optional vision branch.
    """

    def _apply_patch(self):
        """Apply the Llama4 vision patch."""
        # Store original forward method
        self.original_values["Llama4ForConditionalGeneration.forward"] = (
            Llama4ForConditionalGeneration.forward
        )

        # Apply patch by replacing the forward method
        Llama4ForConditionalGeneration.forward = _forward_with_cond

    def _revert_patch(self):
        """Revert the Llama4 vision patch."""
        # Restore original forward method
        Llama4ForConditionalGeneration.forward = self.original_values[
            "Llama4ForConditionalGeneration.forward"
        ]


def _moe_forward_with_transpose(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_scores, router_logits = self.router(hidden_states)
    routed_in = hidden_states.repeat(router_scores.shape[1], 1)

    # BUG IN ORIGINAL CODE
    # routed_in = routed_in * router_scores.reshape(-1, 1)
    # END OF BUG IN ORIGINAL CODE

    # PATCH STARTED
    routed_in = routed_in * router_scores.transpose(0, 1).reshape(-1, 1)
    # PATCH ENDED

    routed_out = self.experts(routed_in)
    out = self.shared_expert(hidden_states)
    out.add_(routed_out.reshape(router_scores.shape[1], -1, routed_out.shape[-1]).sum(dim=0))
    return out, router_logits


# TODO: remove this patch once https://github.com/huggingface/transformers/pull/40609 is merged,
# gets released, and TRT-LLM updates to the relevant transformers version --> this is part of
# 4.56.1 onwards.
@ExportPatchRegistry.register("hf_llama4_moe")
class Llama4MoEPatch(BaseExportPatch):
    """Patch for Llama4 MoE routing to fix its current accuracy issue."""

    def _apply_patch(self):
        """Apply the Llama4 MoE routing patch."""
        # Store original forward method
        self.original_values["Llama4TextMoe.forward"] = Llama4TextMoe.forward

        # Apply patch by replacing the forward method
        Llama4TextMoe.forward = _moe_forward_with_transpose

    def _revert_patch(self):
        """Revert the Llama4 MoE routing patch."""
        Llama4TextMoe.forward = self.original_values["Llama4TextMoe.forward"]
