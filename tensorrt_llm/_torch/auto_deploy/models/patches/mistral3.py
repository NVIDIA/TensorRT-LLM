"""A patch for the Mistral3Model to make it compatible with torch.export.

NOTE: most patches are not used at the moment since only text submodule is exported. Keeping it here
for future reference in case we decide to also export the image model.
"""

from typing import List, Optional, Union

import torch
from transformers.models.mistral3.modeling_mistral3 import (
    Mistral3Model,
    Mistral3ModelOutputWithPast,
)

from ...export.interface import DisabledBaseExportPatch, ExportPatchRegistry


def _get_image_features_flat(
    self,
    pixel_values: torch.FloatTensor,
    image_sizes: torch.Tensor,
    vision_feature_layer: Optional[Union[int, List[int]]] = None,
    **kwargs,
):
    vision_feature_layer = (
        vision_feature_layer
        if vision_feature_layer is not None
        else self.config.vision_feature_layer
    )

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    image_outputs = self.vision_tower(
        pixel_values, image_sizes=image_sizes, output_hidden_states=True, **kwargs
    )

    if isinstance(vision_feature_layer, int):
        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
    else:
        hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
        selected_image_feature = torch.cat(hs_pool, dim=-1)

    image_features = self.multi_modal_projector(selected_image_feature.squeeze(0), image_sizes)
    image_features = image_features.squeeze(0)
    return image_features


# NOTE: the main reason for this patch's existence is the `torch.cond` branching logic to handle the
# presence / absence of image features in a `torch.export`-compatible way.
def _mistral_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[Union[int, List[int]]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[tuple, Mistral3ModelOutputWithPast]:
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
        else self.config.vision_feature_layer
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if pixel_values is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
        )

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    def _no_vision_branch(
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        pixel_values: torch.Tensor,
        image_sizes: Optional[torch.Tensor],
    ):
        return inputs_embeds.clone()

    def _vision_branch(
        # ! The type annotations in the original transformers code are all wrong.
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        pixel_values: torch.Tensor,
        image_sizes: Optional[torch.Tensor],
    ):
        pixel_values = pixel_values.to(torch.bfloat16)
        image_features = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            image_sizes=image_sizes,
        )
        # HF returns a list of tensors; our patch may already return a single tensor.
        # Only concatenate when a list/tuple is returned.
        if isinstance(image_features, (list, tuple)):
            image_features = torch.cat(image_features, dim=0)

        special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds

    # Decide by whether there is any non-zero pixel_values.
    has_image: torch.Tensor = (pixel_values is not None) and torch.any(pixel_values != 0)

    # `torch.cond` serves 2 purposes here:
    # 1. It lets the export stage know that there could be both image and no-image branches.
    #    Without this, the export stage would just assume that whatever the example input contains
    #    is representative of _all_ inputs at runtime. This means that, if we export it with images
    #    in the inputs, it would crash when called without images (i.e. in text-only mode).
    # 2. It introduces a subgraph, which the pattern matcher will ignore. This is important as we
    #    do not want the vision model's attention ops to be converted by the pattern matcher to have
    #    KV cache enabled on them, as it would be both unnecessary to do so and potentially bad for
    #    performance.
    inputs_embeds = torch.cond(
        has_image,
        _vision_branch,
        _no_vision_branch,
        (input_ids, inputs_embeds, pixel_values, image_sizes),
    )

    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    return Mistral3ModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        # NOTE: this is hardcoded since we make no use of this.
        image_hidden_states=None,
    )


# NOTE: registered as patch that is disabled by default since it is not used at the moment
@ExportPatchRegistry.register("hf_mistral3")
class Mistral3ModelPatch(DisabledBaseExportPatch):
    """Patch for `Mistral3Model`."""

    def _apply_patch(self):
        """Apply the Mistral3Model patch."""
        self.original_values["Mistral3Model.forward"] = Mistral3Model.forward
        self.original_values["Mistral3Model.get_image_features"] = Mistral3Model.get_image_features

        Mistral3Model.forward = _mistral_forward
        Mistral3Model.get_image_features = _get_image_features_flat

    def _revert_patch(self):
        """Revert the Mistral3Model patch."""
        # Restore original forward method.
        Mistral3Model.forward = self.original_values["Mistral3Model.forward"]
        Mistral3Model.get_image_features = self.original_values["Mistral3Model.get_image_features"]
