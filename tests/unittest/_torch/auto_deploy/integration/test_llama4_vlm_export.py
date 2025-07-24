from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from _model_test_utils import _hf_model_dir_or_hub_id
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Llama4ForConditionalGeneration
from transformers.models.llama4.modeling_llama4 import Llama4CausalLMOutputWithPast
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transformations._graph import move_to_device


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
        original_inputs_embeds_shape = inputs_embeds.shape

        vision_flat = image_features.view(-1, image_features.size(-1))
        projected_vision_flat = self.multi_modal_projector(vision_flat)

        special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
        final_mask = special_image_mask.to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-1))

        final_mask_1d = final_mask[..., 0].reshape(-1)
        # num_tokens_to_fill = final_mask_1d.sum()

        # This condition statement breaks torch.export:
        # TODO: sanity check on the inputs for this
        # if num_tokens_to_fill != projected_vision_flat.size(0):
        #     raise ValueError(
        #         f"Mismatch: final_mask wants {num_tokens_to_fill} embeddings, "
        #         f"but multi_modal_projector returned {projected_vision_flat.size(0)}"
        #     )

        expanded_mask = final_mask_1d.unsqueeze(-1).expand(-1, inputs_embeds.size(-1))
        inputs_embeds.masked_scatter_(expanded_mask, projected_vision_flat)

        return inputs_embeds.view(original_inputs_embeds_shape)

    def _no_vision_branch(inputs_embeds, pixel_values, input_ids):
        return inputs_embeds

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
        if attention_mask is not None:
            shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
            shift_logits = logits[..., :-1, :][
                shift_attention_mask.to(logits.device) != 0
            ].contiguous()
            shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
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


def test_build_run_llama4_vlm():
    atol = 1e-3
    rtol = 1e-3

    model_id = _hf_model_dir_or_hub_id(
        f"{llm_models_root()}/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    config = AutoConfig.from_pretrained(model_id)
    config.text_config.num_hidden_layers = 2
    config.text_config.intermediate_size = 64
    config.text_config.intermediate_size_mlp = 128
    config.vision_config.num_hidden_layers = 2

    # The returned cache <class 'transformers.cache_utils.HybridChunkedCache'> breaks torch.export
    config.text_config.use_cache = False

    model = Llama4ForConditionalGeneration(config).eval().to("cuda").bfloat16()

    img1 = Image.new("RGB", (16, 16), color=(128, 128, 128))
    img2 = Image.new("RGB", (16, 16), color=(64, 64, 64))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
                {"type": "text", "text": "What's the difference?"},
            ],
        },
    ]

    inputs = (
        processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        .to(model.device)
        .to(torch.bfloat16)
    )

    with torch.inference_mode():
        # the original model queried with text-only
        out_text_only = model(inputs["input_ids"], None, inputs["attention_mask"])

    Llama4ForConditionalGeneration.forward = _forward_with_cond

    with torch.inference_mode():
        out_real = model(inputs["input_ids"], inputs["pixel_values"], inputs["attention_mask"])
        out_dummy = model(
            inputs["input_ids"], torch.zeros_like(inputs["pixel_values"]), inputs["attention_mask"]
        )
        torch.testing.assert_close(out_dummy.logits, out_text_only.logits, rtol=rtol, atol=atol)

    gm = torch_export_to_gm(
        model,
        (inputs["input_ids"], inputs["pixel_values"], inputs["attention_mask"]),
        kwargs={},
    )
    move_to_device(gm, model.device)

    with torch.inference_mode():
        out_real_gm = gm(inputs["input_ids"], inputs["pixel_values"], inputs["attention_mask"])
        torch.testing.assert_close(out_real.logits, out_real_gm.logits, rtol=rtol, atol=atol)
        out_dummy_gm = gm(
            inputs["input_ids"], torch.zeros_like(inputs["pixel_values"]), inputs["attention_mask"]
        )
        torch.testing.assert_close(out_dummy.logits, out_dummy_gm.logits, rtol=rtol, atol=atol)
        torch.testing.assert_close(out_dummy_gm.logits, out_text_only.logits, rtol=rtol, atol=atol)

        assert not torch.allclose(out_real.logits, out_dummy.logits, rtol=rtol, atol=atol), (
            "Expected outputs to differ between text only input and text+image input"
        )
