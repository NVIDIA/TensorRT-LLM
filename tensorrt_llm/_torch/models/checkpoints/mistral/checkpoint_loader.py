from typing import Optional

from tensorrt_llm._torch.models.checkpoints.base_config_loader import BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import BaseWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader
from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_loader
from tensorrt_llm.quantization.mode import QuantAlgo


@register_checkpoint_loader("mistral")
class MistralCheckpointLoader(HfCheckpointLoader):
    def __init__(
        self,
        *,
        weight_loader: Optional[BaseWeightLoader] = None,
        weight_mapper: Optional[BaseWeightMapper] = None,
        config_loader: Optional[BaseConfigLoader] = None,
    ):
        super().__init__(
            weight_loader=weight_loader, weight_mapper=weight_mapper, config_loader=config_loader
        )
        self._checkpoint_format = "mistral"
        self.mm_module_mapping = {
            "vision_encoder": "vision_tower",
            "pre_mm_projector_norm": "multi_modal_projector.norm",
            "vision_language_adapter": "multi_modal_projector",
            "patch_merger": "multi_modal_projector.patch_merger",
        }

    def preprocess_weights(self, weights: dict) -> dict:
        """
        Aggregate weights by module
        """
        hf_weights = {}

        for key, value in weights.items():
            modules = key.split(".")

            if modules[0] not in self.mm_module_mapping.keys():
                hf_weights["language_model." + key] = value

            else:
                modules[0] = self.mm_module_mapping[modules[0]]
                hf_weights[".".join(modules)] = value

        return hf_weights

    def broadcast_per_tensor_scales(self, weights):
        import math

        scales = [k for k in weights.keys() if k.endswith("qscale_weight")]
        for scale in scales:
            name = ".".join(scale.split(".")[:-1])
            weight_shape = weights[f"{name}.weight"].shape
            broadcast = weights[scale].expand(
                math.ceil(weight_shape[0] / 128),
                math.ceil(weight_shape[1] / 128),
            )
            weights[scale] = broadcast[:]

    def reverse_nvfp4_global_scales(self, weights):
        for key in weights.keys():
            if "global_scale" in key:
                weights[key] = 1.0 / weights[key]

    def load_weights(self, checkpoint_dir: str, **kwargs):
        model_config = kwargs.pop("model_config", None)
        assert model_config is not None, "model_config is required"
        weights = super().weight_loader.load_weights(checkpoint_dir, mapping=None, **kwargs)
        params_map = self.weight_mapper.mistral_llm_mapping.copy()
        if model_config is not None:
            if model_config.quant_config.quant_algo == QuantAlgo.NVFP4:
                quantization_weights_map = {
                    "weight_packed": "weight",
                    "input_global_scale": "input_scale",
                    "weight_global_scale": "weight_scale_2",
                }
            elif model_config.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
                quantization_weights_map = {
                    "weight_scale": "weight_scale_inv",
                }
            params_map.update(quantization_weights_map)
        weights = self.preprocess_weights(weights)
        weights = self.weight_mapper.rename_by_params_map(weights=weights, params_map=params_map)

        # FIXME mimic DS fp8 till per tensor supported
        self.broadcast_per_tensor_scales(weights)
        self.reverse_nvfp4_global_scales(weights)
        return weights

    def get_default_config_loader(self) -> MistralConfigLoader:
        return MistralConfigLoader()


@register_checkpoint_loader("mistral_large_3")
class MistralLarge3CheckpointLoader(MistralCheckpointLoader):
    def __init__(
        self,
        *,
        weight_loader: Optional[BaseWeightLoader] = None,
        weight_mapper: Optional[BaseWeightMapper] = None,
        config_loader: Optional[BaseConfigLoader] = None,
    ):
        super().__init__(
            weight_loader=weight_loader, weight_mapper=weight_mapper, config_loader=config_loader
        )
        self._checkpoint_format = "mistral_large_3"
