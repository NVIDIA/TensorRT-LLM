import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (AutoProcessor, AutoTokenizer, PretrainedConfig,
                          PreTrainedModel, Qwen2_5_VLForConditionalGeneration,
                          Qwen2VLForConditionalGeneration)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from tensorrt_llm.inputs.multimodal import MultimodalParams

from ..._utils import nvtx_range_debug
from ...functional import RopeEmbeddingUtils, RotaryScalingType
from ...inputs import (ExtraProcessedInputs, InputProcessor,
                       MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       register_input_processor)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import (find_uncached_mm_embeds,
                                        fuse_input_embeds)
from .modeling_utils import register_auto_model, register_vision_encoder

DISAGG = os.getenv('TLLM_MULTIMODAL_DISAGGREGATED', '0') == '1'
ENABLE_FP8_BLOCK_SCALE = os.getenv('TLLM_ENABLE_FP8_BLOCK_SCALE', '0') == '1'


class Qwen2VLInputProcessorBase(InputProcessor):

    def __init__(self,
                 model_path: str,
                 model_config: PretrainedConfig,
                 tokenizer: AutoTokenizer,
                 trust_remote_code: bool = True):
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.use_fast = True
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=self.use_fast,
            trust_remote_code=trust_remote_code)

        self.tllm_multimodal_token_id = self.model_config.vocab_size + 1

    @classmethod
    def get_rope_index(
        cls,
        model_config: PretrainedConfig,
        input_ids: Optional[torch.IntTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        This is a generalized implementation that can be used by both Qwen2VL and Qwen2_5_VL models.
        The main difference between the two implementations is how temporal position IDs are calculated.

        Args:
            model_config: The model configuration
            input_ids: Indices of input sequence tokens in the vocabulary
            image_grid_thw: The temporal, height and width of feature shape of each image in LLM
            video_grid_thw: The temporal, height and width of feature shape of each video in LLM
            attention_mask: Mask to avoid performing attention on padding token indices
            second_per_grid_ts: The time interval (in seconds) for each grid along the temporal dimension

        Returns:
            position_ids: A tensor of shape (3, batch_size, sequence_length)
            mrope_position_deltas: A tensor of shape (batch_size)
        """
        spatial_merge_size = model_config.vision_config.spatial_merge_size
        image_token_id = model_config.image_token_id
        video_token_id = model_config.video_token_id
        vision_start_token_id = model_config.vision_start_token_id
        mrope_position_deltas = []

        # Handle case with no vision inputs
        if image_grid_thw is None and video_grid_thw is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(
                    input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[
                    -1]
            else:
                position_ids = (torch.arange(input_ids.shape[1],
                                             device=input_ids.device).view(
                                                 1, 1, -1).expand(
                                                     3, input_ids.shape[0], -1))
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

        # Handle case with vision inputs
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)

        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                    llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # Calculate temporal position IDs based on model type
                if hasattr(model_config.vision_config, 'tokens_per_second'):
                    # Qwen2_5_VL style temporal position calculation
                    if isinstance(second_per_grid_t, torch.Tensor):
                        second_per_grid_t = second_per_grid_t.item()
                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(
                        -1, llm_grid_h * llm_grid_w)
                    time_tensor = expanded_range * second_per_grid_t * model_config.vision_config.tokens_per_second
                    t_index = time_tensor.long().flatten()
                else:
                    # Qwen2VL style temporal position calculation
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
                        -1, llm_grid_h * llm_grid_w).flatten()

                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
                    llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
                    llm_grid_t, llm_grid_h, -1).flatten()

                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len +
                    st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                    llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 -
                                         len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    def get_num_tokens_per_image(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 1,
        do_resize: bool = True,
    ):
        patch_size = self.model_config.vision_config.patch_size
        merge_size = self.model_config.vision_config.spatial_merge_size
        temporal_patch_size = self.model_config.vision_config.temporal_patch_size
        if do_resize:
            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                min_pixels=self.processor.image_processor.min_pixels,
                max_pixels=self.processor.image_processor.max_pixels,
            )
            image_width, image_height = resized_width, resized_height

        padded_num_frames = num_frames + num_frames % temporal_patch_size

        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = image_height // patch_size
        grid_w = image_width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size**2)

        return num_vision_tokens

    def _preprocess(self, text: dict[str, any], mm_data: dict[str, any],
                    mm_processor_kwargs: Dict[str, Any]):
        images = mm_data.get("image")
        videos = mm_data.get("video")
        do_rescale = True
        if images and isinstance(images[0], torch.Tensor):
            do_rescale = False
        if videos and isinstance(videos[0][0], torch.Tensor):
            do_rescale = False
            # transformers=4.53.1 does not support GPU video tensors in Qwen2VL processor.
            videos = [[frame.to("cpu") for frame in video] for video in videos]
        return self.processor(text=[text],
                              images=images,
                              videos=videos,
                              padding=True,
                              do_rescale=do_rescale,
                              return_tensors='pt',
                              **mm_processor_kwargs)

    def _postprocess(self, input_ids: torch.IntTensor) -> torch.IntTensor:
        # NOTE: Qwen2-VL's input processor is doing all the work for fusing input_ids with mm_tokens.
        # So, we just replace mm_tokens with expanded out-of-vocab ids
        masks = (input_ids == self.model_config.image_token_id) | (
            input_ids == self.model_config.vision_token_id) | (
                input_ids == self.model_config.video_token_id)
        input_ids[masks] = self.tllm_multimodal_token_id
        return input_ids

    def get_mrope_config(
            self,
            input_ids: torch.IntTensor,
            image_grid_thw: torch.LongTensor,
            video_grid_thw: torch.LongTensor,
            attention_mask: torch.Tensor,
            second_per_grid_ts: torch.Tensor = None) -> dict[str, torch.Tensor]:
        mrope_position_ids, mrope_position_deltas = self.__class__.get_rope_index(
            self.model_config, input_ids, image_grid_thw, video_grid_thw,
            attention_mask, second_per_grid_ts)

        mrope_config = {}
        mrope_config['mrope_position_ids'] = mrope_position_ids.to('cpu')
        mrope_config['mrope_position_deltas'] = mrope_position_deltas.to(
            'cpu').to(torch.int32)
        return mrope_config

    @torch.inference_mode()
    def __call__(
        self,
        inputs: TextPrompt,
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data, mm_processor_kwargs = inputs.get("prompt"), \
                        inputs.get("multi_modal_data", {}), inputs.get("mm_processor_kwargs", {})
        with nvtx_range_debug("transformers input preprocess"):
            processed_inputs = self._preprocess(text_prompt, mm_data,
                                                mm_processor_kwargs)
        if not mm_data:
            fused_input_ids = processed_inputs['input_ids']
            # Flatten the tensor to get a simple list of integers
            return fused_input_ids.flatten().to(torch.int32).tolist(), {}

        pixel_values = processed_inputs.get('pixel_values', None)
        pixel_values_videos = processed_inputs.get('pixel_values_videos', None)
        assert pixel_values is not None or pixel_values_videos is not None, "No multimodal data found"

        multimodal_data = {}
        if pixel_values is not None:
            multimodal_data["image"] = {
                "pixel_values": pixel_values,
                "image_grid_thw": processed_inputs.get('image_grid_thw')
            }
        if pixel_values_videos is not None:
            multimodal_data["video"] = {
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": processed_inputs.get('video_grid_thw')
            }

        input_ids = processed_inputs['input_ids']
        # TODO: We can move this to the LLM-side.
        mrope_config = self.get_mrope_config(
            input_ids, processed_inputs.get('image_grid_thw', None),
            processed_inputs.get('video_grid_thw', None),
            processed_inputs.get('attention_mask', None),
            processed_inputs.get('second_per_grid_ts', None))
        multimodal_data["mrope_config"] = mrope_config

        fused_input_ids = self._postprocess(input_ids[0])

        return fused_input_ids.to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }


class Qwen2VisionModelBase(nn.Module):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 model_class: type[PreTrainedModel]):
        super().__init__()
        pretrained_config = model_config.pretrained_config
        self.model_config = model_config
        self.device = f"cuda:{model_config.mapping.rank}"

        model_path = pretrained_config._name_or_path
        # TODO: Change the model class to TRT-LLM's Qwen2VisionModel
        # Currently, copying vision encoder on all devices.
        # NOTE: Using attn_implementation='flash_attention_2' to avoid the issue of vision model's GPU OOM.
        model = model_class.from_pretrained(
            model_path,
            torch_dtype=pretrained_config.torch_dtype,
            attn_implementation='flash_attention_2').eval()
        # TODO: Make vision model compatible with meta init mode and load_weights at the same place
        self.visual = model.visual.to(self.device)
        
        # Check if FP8 Block Scale mode is enabled
        # Priority: Environment variable > Config file > Default value
        config_enable = getattr(pretrained_config, 'enable_fp8_block_scale', False)
        self.enable_fp8_block_scale = ENABLE_FP8_BLOCK_SCALE or config_enable
        print(f"FP8 Block Scale mode: {'ENABLED' if self.enable_fp8_block_scale else 'DISABLED'}")
        if ENABLE_FP8_BLOCK_SCALE:
            print("  - Enabled via environment variable TLLM_ENABLE_FP8_BLOCK_SCALE=1")
        elif config_enable:
            print("  - Enabled via config file")
        else:
            print("  - Disabled (use TLLM_ENABLE_FP8_BLOCK_SCALE=1 or set enable_fp8_block_scale=True in config)")
        
        if self.enable_fp8_block_scale:
            # Define layer name patterns to be replaced with FP8 Block Scale
            # Now supports MLP layers, handling dimension mismatch through padding
            self.fp8_block_scale_patterns = [
                "blocks.*.attn.qkv",  # All block attention qkv
                "blocks.*.attn.proj",  # Re-enable attention projection, fix reshape logic
                "blocks.*.mlp.gate_proj",  # All block mlp gate_proj
                "blocks.*.mlp.down_proj",  # All block mlp down_proj
                "blocks.*.mlp.up_proj",  # All block mlp up_proj
            ]
            
            # Allow custom replacement patterns through configuration
            if hasattr(pretrained_config, 'fp8_block_scale_patterns'):
                self.fp8_block_scale_patterns = pretrained_config.fp8_block_scale_patterns
            
            # Print model structure for debugging
            print("Visual model structure:")
            for name, module in self.visual.named_modules():
                if isinstance(module, torch.nn.Linear):
                    print(f"  Linear layer: {name}")
            
            # Enable replacement functionality - now with pre-quantized weights
            self._replace_linear_layers_with_pre_quantization()
        else:
            print("Skipping FP8 Block Scale layer replacement, using original implementation")
        
        self.post_config()

    def post_config(self):
        self.config = self.visual.config

    def _to_device(
        self, input_tensor: Union[torch.Tensor, List, None]
    ) -> Union[torch.Tensor, List, None]:
        if input_tensor is None:
            return None
        elif isinstance(input_tensor, list):
            return [self._to_device(item) for item in input_tensor]
        elif isinstance(input_tensor, torch.Tensor):
            return input_tensor.to(self.device)

    def _parse_and_batch_multimodal_data(
        self, multimodal_params: List[MultimodalParams]
    ) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:

        pixel_values_list = []
        pixel_values_videos_list = []
        image_grid_thw_list = []
        video_grid_thw_list = []

        for multimodal_param in multimodal_params:
            # Process images if present
            if multimodal_param.multimodal_data.get("image") is not None:
                pixel_values_list.append(
                    multimodal_param.multimodal_data["image"]["pixel_values"])
                image_grid_thw_list.append(
                    multimodal_param.multimodal_data["image"]["image_grid_thw"])

            # Process videos if present
            if multimodal_param.multimodal_data.get("video") is not None:
                pixel_values_videos_list.append(
                    multimodal_param.multimodal_data["video"]
                    ["pixel_values_videos"])
                video_grid_thw_list.append(
                    multimodal_param.multimodal_data["video"]["video_grid_thw"])

        # Concatenate tensors
        mm_content_dict = {}
        if pixel_values_list:
            mm_content_dict["pixel_values"] = torch.cat(
                pixel_values_list,
                dim=0) if len(pixel_values_list) > 1 else pixel_values_list[0]
        if pixel_values_videos_list:
            mm_content_dict["pixel_values_videos"] = torch.cat(
                pixel_values_videos_list,
                dim=0) if len(pixel_values_videos_list
                              ) > 1 else pixel_values_videos_list[0]

        # Prepare extra data
        mm_extra_data = {}
        if image_grid_thw_list:
            mm_extra_data["image_grid_thw"] = torch.cat(
                image_grid_thw_list, dim=0) if len(
                    image_grid_thw_list) > 1 else image_grid_thw_list[0]
        if video_grid_thw_list:
            mm_extra_data["video_grid_thw"] = torch.cat(
                video_grid_thw_list, dim=0) if len(
                    video_grid_thw_list) > 1 else video_grid_thw_list[0]

        return mm_content_dict, mm_extra_data

    def _replace_linear_layers_with_pre_quantization(self):
        """
        Replace linear layers and pre-quantize weights to avoid repeated quantization during forward pass
        """
        import re
        import torch.nn as nn
        
        # Directly iterate through all submodules of the visual module
        for name, module in self.visual.named_modules():
            # Check if it's a linear layer
            if isinstance(module, nn.Linear):
                # Check if it matches any pattern
                should_replace = False
                for pattern in self.fp8_block_scale_patterns:
                    # Convert pattern to regex
                    regex_pattern = pattern.replace("*", r"\d+")
                    if re.match(regex_pattern, name):
                        should_replace = True
                        break
                
                if should_replace:
                    # Check if weight dimensions meet TensorRT-LLM requirements
                    # For matrix multiplication input @ weight.T, N dimension is in_features
                    weight = module.weight
                    in_features = weight.shape[0]  # Input feature dimension
                    out_features = weight.shape[1]  # Output feature dimension
                    print(f"DEBUG: Checking {name}, weight.shape={weight.shape}, in_features={in_features}, out_features={out_features}, in_features%16={in_features % 16}")
                    
                    if in_features % 16 != 0:
                        print(f"Skipping {name}: in_features ({in_features}) not divisible by 16")
                        continue
                    
                    try:
                        # Create pre-quantized FP8 Block Scale replacement
                        fp8_linear = self._create_pre_quantized_fp8_block_linear(module)
                        
                        # Find parent module and child module names
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        
                        if parent_name:
                            # Get parent module
                            parent_module = self.visual
                            for part in parent_name.split('.'):
                                parent_module = getattr(parent_module, part)
                            
                            # Replace child module
                            setattr(parent_module, child_name, fp8_linear)
                        else:
                            # Direct replacement
                            setattr(self.visual, child_name, fp8_linear)
                            
                        print(f"Replaced Linear layer with Pre-quantized FP8 Block Scale: {name}")
                    except Exception as e:
                        print(f"Failed to replace {name}: {e}")

    def _create_pre_quantized_fp8_block_linear(self, original_linear):
        """
        Create pre-quantized FP8 Block Linear replacement layer
        
        Args:
            original_linear: Original nn.Linear layer
            
        Returns:
            Pre-quantized FP8 Block Linear layer
        """
        import torch.nn as nn
        
        class PreQuantizedTrtllmFp8BlockLinear(nn.Module):
            def __init__(self, original_linear):
                super().__init__()
                self.original_linear = original_linear
                
                # Pre-quantize weights and scaling factors
                print(f"Pre-quantizing weights for layer with shape {original_linear.weight.shape}")
                self.weight_fp8, self.weight_scale = self._pre_quantize_weight(original_linear.weight)
                
                # Move quantized weights and scaling factors to CPU to save GPU memory
                self.weight_fp8 = self.weight_fp8.cpu()
                self.weight_scale = self.weight_scale.cpu()
                
                print(f"Pre-quantization completed. Weight FP8 shape: {self.weight_fp8.shape}, Scale shape: {self.weight_scale.shape}")
                
                try:
                    import tensorrt_llm
                    pass
                except ImportError:
                    raise ImportError("TensorRT-LLM is not installed.")

            def _pre_quantize_weight(self, weight: torch.Tensor):
                """
                Pre-quantize weights, executed once during initialization
                """
                print(f"Starting pre-quantization for weight with shape {weight.shape}")
                
                # Check if N dimension is divisible by 16
                if weight.shape[1] % 16 != 0:
                    print(f"Warning: Matrix N dimension ({weight.shape[1]}) not divisible by 16, skipping FP8 quantization")
                    return weight, torch.ones(1, device=weight.device, dtype=torch.float32)
                
                # Execute block-wise quantization
                quantized_weight, scale = self._create_blockwise_quantized_weight(weight)
                
                if quantized_weight.dtype != torch.float8_e4m3fn:
                    print(f"Warning: Failed to quantize weight, using original")
                    return weight, torch.ones(1, device=weight.device, dtype=torch.float32)
                
                print(f"Pre-quantization successful. Quantized weight shape: {quantized_weight.shape}")
                return quantized_weight, scale

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                """Forward method using pre-quantized weights"""
                # Get parameters from original linear layer
                bias = getattr(self.original_linear, 'bias', None)
                
                # Check if input dimensions meet requirements
                input_features = input.shape[-1]
                if input_features % 16 != 0:
                    print(f"Using original linear layer: input_features ({input_features}) not divisible by 16")
                    return self.original_linear(input)
                
                # Save original shape and data type
                origin_shape = input.shape
                origin_dtype = input.dtype
                input = input.to(torch.bfloat16)

                if input.dim() > 2:
                    input = input.reshape(-1, input.shape[-1])

                # Execute input FP8 quantization
                act_input_fp8, input_scale = torch.ops.trtllm.fp8_quantize_1x128(input)
                
                # Move pre-quantized weights and scaling factors to current device
                weight_fp8 = self.weight_fp8.to(input.device)
                weight_scale = self.weight_scale.to(input.device)
                
                # Execute FP8 GEMM
                output = torch.ops.trtllm.fp8_block_scaling_gemm(act_input_fp8, weight_fp8, input_scale, weight_scale)
                output = output.to(origin_dtype)

                if bias is not None:
                    output = output + bias
                    
                # Handle output shape
                if output.dim() == 2:
                    if len(origin_shape) == 3:
                        batch_size, seq_len, hidden_size = origin_shape
                        output = output.reshape(batch_size, seq_len, hidden_size)
                    elif len(origin_shape) == 2:
                        pass  # No reshape needed
                    else:
                        return self.original_linear(input)
                    
                return output
            
            def _create_blockwise_quantized_weight(
                self,
                param_value: torch.Tensor,
                block_size: int = 128,
            ):
                """
                Create block-wise quantized weights
                Reference: transformers fp8 128*128 block quantization
                Supports padding non-128-multiple matrices to 128 multiples
                """
                param_value = param_value.to(torch.float32)

                # Get FP8 min/max values
                fp8_min = torch.finfo(torch.float8_e4m3fn).min
                fp8_max = torch.finfo(torch.float8_e4m3fn).max

                rows, cols = param_value.shape[-2:]
                original_shape = param_value.shape
                
                # Check if N dimension is divisible by 16 (TensorRT-LLM FP8 GEMM requirement)
                # For matrix multiplication input @ weight.T, N dimension is cols (in_features)
                if cols % 16 != 0:
                    print(f"Warning: Matrix N dimension ({cols}) not divisible by 16, skipping FP8 quantization")
                    return param_value, torch.ones(1, device=param_value.device, dtype=torch.float32)
                
                # Calculate padding needed for rows and columns
                # Round up to block_size multiples
                target_rows = ((rows + block_size - 1) // block_size) * block_size
                target_cols = ((cols + block_size - 1) // block_size) * block_size
                pad_rows = target_rows - rows
                pad_cols = target_cols - cols
                
                # Perform padding if needed
                if pad_rows > 0 or pad_cols > 0:
                    print(f"Padding matrix from ({rows}, {cols}) to ({rows + pad_rows}, {cols + pad_cols})")
                    
                    # Create padded weight matrix
                    padded_weight = torch.zeros(
                        rows + pad_rows, cols + pad_cols, 
                        device=param_value.device, dtype=param_value.dtype
                    )
                    
                    # Copy original weights to top-left corner of padded matrix
                    padded_weight[:rows, :cols] = param_value
                    
                    # Use padded weights for quantization
                    param_value = padded_weight
                    rows, cols = rows + pad_rows, cols + pad_cols
                
                # Now matrix dimensions are multiples of 128, can perform block-wise quantization
                block_size_m, block_size_n = block_size, block_size
                param_value_orig_shape = param_value.shape
                param_value = param_value.reshape(
                    -1, rows // block_size_m, block_size_m, cols // block_size_n, block_size_n
                ).permute(0, 1, 3, 2, 4)

                # Calculate scaling factor for each block
                max_abs = torch.amax(torch.abs(param_value), dim=(-1, -2))
                scale = fp8_max / max_abs
                scale_orig_shape = scale.shape
                scale = scale.unsqueeze(-1).unsqueeze(-1)

                @torch.compiler.disable()
                def _quantize(param_value, scale, fp8_min, fp8_max):
                    # Quantize the weights
                    quantized_param = torch.clamp(param_value * scale, min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

                    quantized_param = quantized_param.permute(0, 1, 3, 2, 4)
                    # Reshape back to matrix shape
                    quantized_param = quantized_param.reshape(param_value_orig_shape)

                    # Reshape scale to match the number of blocks
                    scale = scale.reshape(scale_orig_shape).squeeze().reciprocal()

                    return quantized_param, scale

                quantized_param, scale = _quantize(param_value, scale, fp8_min, fp8_max)
                
                # If original matrix was padded, crop back to original size
                if pad_rows > 0 or pad_cols > 0:
                    quantized_param = quantized_param[:original_shape[0], :original_shape[1]]
                
                return quantized_param, scale
            
            @property
            def weight(self):
                return self.original_linear.weight
            
            @property
            def bias(self):
                return getattr(self.original_linear, 'bias', None)
            
            @property
            def in_features(self):
                return self.original_linear.in_features
            
            @property
            def out_features(self):
                return self.original_linear.out_features
        
        return PreQuantizedTrtllmFp8BlockLinear(original_linear)

    def is_fp8_blockscale_enabled(self) -> bool:
        """
        Check if FP8 Block Scale mode is enabled
        
        Returns:
            bool: True if FP8 mode is enabled, False otherwise
        """
        return getattr(self, 'enable_fp8_block_scale', False)

    @torch.inference_mode()
    def forward(self, multimodal_params: List[MultimodalParams]):

        mm_content_data, mm_extra_data = self._parse_and_batch_multimodal_data(
            multimodal_params)
        pixel_values = mm_content_data.get("pixel_values", None)
        pixel_values_videos = mm_content_data.get("pixel_values_videos", None)

        image_grid_thw = mm_extra_data.get("image_grid_thw", None)
        video_grid_thw = mm_extra_data.get("video_grid_thw", None)

        embeds = []
        if pixel_values is not None:
            pixel_values = self._to_device(
                pixel_values
            )  # TODO: remove this once we have the shared tensor
            image_grid_thw = self._to_device(image_grid_thw)
            pixel_values = pixel_values.to(self.visual.dtype)
            embeds.append(self.visual(pixel_values, grid_thw=image_grid_thw))

        if pixel_values_videos is not None:
            pixel_values_videos = self._to_device(pixel_values_videos)
            video_grid_thw = self._to_device(video_grid_thw)
            pixel_values_videos = pixel_values_videos.to(self.visual.dtype)
            embeds.append(
                self.visual(pixel_values_videos, grid_thw=video_grid_thw))
        return embeds


class Qwen2VLModelBase(PreTrainedModel):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        model_config.pretrained_config.rope_scaling['type'] = 'mrope'
        config = model_config.pretrained_config

        assert model_config.attn_backend == 'TRTLLM', "Qwen2/2.5-VL only supports TRTLLM backend now"
        super().__init__(config)

        self.model_config = model_config
        if hasattr(self, "llm"):
            return

        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config.architectures = ["Qwen2ForCausalLM"]
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)
        self.vocab_size = config.vocab_size
        self.model_dtype = getattr(config, "torch_dtype", torch.float16)
        logger.info(f"{self.dtype=} {self.model_dtype=}")
        self.post_config()
        self.is_loaded = True

    def init_rotary_cos_sin_ori(self):
        _, rotary_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
            num_pos=self.model_config.pretrained_config.max_position_embeddings,
            dim=int(self.model_config.pretrained_config.hidden_size /
                    self.model_config.pretrained_config.num_attention_heads),
            theta=float(self.model_config.pretrained_config.rope_theta),
            scale_type=RotaryScalingType.mrope)
        self.rotary_cos_sin = torch.from_numpy(rotary_cos_sin).to(self.device)
        self.rotary_cos_sin = self.rotary_cos_sin.reshape(
            self.model_config.pretrained_config.max_position_embeddings,
            int(self.model_config.pretrained_config.hidden_size /
                self.model_config.pretrained_config.num_attention_heads / 2), 2)

        self.cos_ori = self.rotary_cos_sin[:, :, 0]
        self.sin_ori = self.rotary_cos_sin[:, :, 1]

    def load_weights(self, weights):
        self.llm.load_weights(weights)
        self.init_rotary_cos_sin_ori()

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    def post_config(self):
        # use llm.config as config for pytorch model engine
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    def _parse_and_concat_mrope_config(
            self, multimodal_params: List[MultimodalParams],
            num_context_requests: int,
            num_generation_requests: int) -> dict[str, torch.Tensor]:
        """
        Parse and concatenate mrope configuration from multimodal parameters.
        """

        mrope_configs = [
            param.multimodal_data.get('mrope_config')
            for param in multimodal_params if param.multimodal_data
            and param.multimodal_data.get('mrope_config')
        ]
        if not mrope_configs:
            return {}

        batched_mrope_config = {}
        if num_context_requests > 0:
            cos_sin_tensors = [
                config['mrope_rotary_cos_sin']
                for config in mrope_configs[:num_context_requests]
                if config.get('mrope_rotary_cos_sin') is not None
            ]
            if cos_sin_tensors:
                batched_mrope_config['mrope_rotary_cos_sin'] = torch.cat(
                    cos_sin_tensors, dim=0)

        if num_generation_requests > 0:
            generation_mrope_configs = mrope_configs[
                -num_generation_requests:] if len(
                    mrope_configs) >= num_generation_requests else mrope_configs
            position_delta_tensors = [
                config['mrope_position_deltas']
                for config in generation_mrope_configs
                if config.get('mrope_position_deltas') is not None
            ]
            if position_delta_tensors:
                batched_mrope_config['mrope_position_deltas'] = torch.cat(
                    position_delta_tensors, dim=0)

        return batched_mrope_config

    def add_rotary_cos_sin(self, multimodal_params: List[MultimodalParams]):
        for param in multimodal_params:
            mrope_config = param.multimodal_data.get('mrope_config')
            if mrope_config:
                mrope_position_ids = mrope_config.get('mrope_position_ids',
                                                      None)
                if mrope_position_ids is None:
                    continue
                mrope_position_ids = mrope_position_ids.transpose(1, 0)
                mrope_position_ids_padding = torch.zeros(
                    mrope_position_ids.shape[:-1] +
                    (self.model_config.pretrained_config.
                     max_position_embeddings, ),
                    dtype=torch.int32,
                    device=mrope_position_ids.device)
                mrope_position_ids_padding[:, :, :mrope_position_ids.
                                           shape[-1]] = mrope_position_ids

                mrope_position_ids_padding = mrope_position_ids_padding.to(
                    self.cos_ori.device)
                cos = self.cos_ori[mrope_position_ids_padding]
                sin = self.sin_ori[mrope_position_ids_padding]

                mrope_section = [16, 24, 24]
                cos = torch.cat([
                    m[:, i % 3]
                    for i, m in enumerate(cos.split(mrope_section, dim=-1))
                ],
                                dim=-1).unsqueeze(-1)
                sin = torch.cat([
                    m[:, i % 3]
                    for i, m in enumerate(sin.split(mrope_section, dim=-1))
                ],
                                dim=-1).unsqueeze(-1)
                concat_cos_sin = torch.concatenate((cos, sin), axis=-1)
                concat_cos_sin = concat_cos_sin.reshape(concat_cos_sin.shape[0],
                                                        -1)

                mrope_config['mrope_rotary_cos_sin'] = concat_cos_sin.to(
                    self.device)

        return multimodal_params

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        VLM forward logic with inflight batching support.
        """
        num_context_requests, num_generation_requests = attn_metadata.num_contexts, attn_metadata.num_generations
        logger.debug(
            f"num_context_requests: {num_context_requests}, num_generation_requests: {num_generation_requests}"
        )

        multimodal_params = kwargs.get("multimodal_params", [])
        multimodal_params = self.add_rotary_cos_sin(multimodal_params)

        mm_embeds = []
        mrope_config = {}

        if len(multimodal_params) > 0:
            if not DISAGG:
                mm_embeds = self.mm_encoder.forward(
                    multimodal_params[:num_context_requests])
            else:
                mm_embeds = [
                    multimodal_param.multimodal_data["multimodal_embedding"]
                    for multimodal_param in multimodal_params
                ]
            mrope_config = self._parse_and_concat_mrope_config(
                multimodal_params, num_context_requests,
                num_generation_requests)
            mm_embeds = find_uncached_mm_embeds(
                mm_embeds, multimodal_params[:num_context_requests])

        if 'mrope_position_deltas' in kwargs:
            mrope_config['mrope_position_deltas'] = kwargs[
                'mrope_position_deltas']

        input_ids, input_embeds = fuse_input_embeds(self.llm.model.embed_tokens,
                                                    input_ids, mm_embeds)
        output_prob = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            return_context_logits=return_context_logits,
            mrope_config=mrope_config)

        logger.debug(f'output shape: {output_prob.shape}')
        return output_prob


@register_vision_encoder(Qwen2VisionModelBase,
                         vlm_base_model=Qwen2VLForConditionalGeneration)
@register_auto_model("Qwen2VLForConditionalGeneration")
@register_input_processor(
    Qwen2VLInputProcessorBase,
    model_type="qwen2_vl",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|vision_start|><|image_pad|><|vision_end|>",
            "video": "<|vision_start|><|video_pad|><|vision_end|>"
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    ))
class Qwen2VLModel(Qwen2VLModelBase):

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args,
                 **kwargs):
        super().__init__(model_config, *args, **kwargs)
        if not DISAGG:
            self.mm_encoder = Qwen2VisionModelBase(
                model_config, Qwen2VLForConditionalGeneration)


@register_vision_encoder(Qwen2VisionModelBase,
                         vlm_base_model=Qwen2_5_VLForConditionalGeneration)
@register_auto_model("Qwen2_5_VLForConditionalGeneration")
@register_input_processor(
    Qwen2VLInputProcessorBase,
    model_type="qwen2_5_vl",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|vision_start|><|image_pad|><|vision_end|>",
            "video": "<|vision_start|><|video_pad|><|vision_end|>"
        }))
class Qwen2_5_VLModel(Qwen2VLModelBase):

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args,
                 **kwargs):
        super().__init__(model_config, *args, **kwargs)
        if not DISAGG:
            self.mm_encoder = Qwen2VisionModelBase(
                model_config, Qwen2_5_VLForConditionalGeneration)
