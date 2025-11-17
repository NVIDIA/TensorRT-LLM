from typing import Optional

from tensorrt_llm._torch.models.checkpoints.base_config_loader import BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import BaseWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader
from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_loader


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

    def preprocess_weights(self, weights: dict) -> dict:
        """
        Aggregate weights by module
        """
        hf_weights = {}

        for key, value in weights.items():
            modules = key.split(".")

            hf_weights["language_model." + key] = value

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

    def load_weights(self, checkpoint_dir: str, **kwargs):
        weights = super().weight_loader.load_weights(checkpoint_dir, **kwargs)
        weights = self.preprocess_weights(weights)
        # FIXME @okozlova mimic DS fp8 till per tensor supported
        self.broadcast_per_tensor_scales(weights)
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
