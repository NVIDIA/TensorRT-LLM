from dataclasses import dataclass
from typing import Optional, Union

from torch import nn

from tensorrt_llm._torch.model_config import TConfig
from tensorrt_llm._torch.models.checkpoints.config_loader_interface import \
    ConfigLoaderInterface
from tensorrt_llm._torch.models.checkpoints.mapper_auto import \
    CheckpointMapperAuto
from tensorrt_llm._torch.models.checkpoints.weight_loader_interface import \
    WeightLoaderInterface
from tensorrt_llm._torch.models.checkpoints.weight_mapper_interface import \
    WeightMapperInterface
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM


@dataclass(kw_only=True)
class CheckpointLoader:
    checkpoint_format: str
    weight_loader: Optional[WeightLoaderInterface] = None
    weight_mapper: Optional[WeightMapperInterface] = None
    config_loader: Optional[ConfigLoaderInterface] = None

    def get_weight_mapper(self, model: Union[nn.Module,
                                             DecoderModelForCausalLM],
                          config: TConfig) -> "WeightMapperInterface":
        weight_mapper = None

        if self.weight_mapper is not None:
            self.weight_mapper.init(model, config)
            return self.weight_mapper
        else:
            # The name of the registered mapper should be the model architecture
            model_arch = config.pretrained_config.architectures[0]
            weight_mapper = CheckpointMapperAuto.get(self.checkpoint_format,
                                                     model_arch)
            weight_mapper.init(model, config)
            return weight_mapper
