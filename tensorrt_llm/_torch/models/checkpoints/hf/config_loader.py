from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.base_config_loader import \
    BaseConfigLoader
from tensorrt_llm._torch.models.modeling_utils import register_config_loader


@register_config_loader("HF")
class HfConfigLoader(BaseConfigLoader):

    def load(self, checkpoint_dir: str, **kwargs) -> ModelConfig:
        return ModelConfig.from_pretrained(checkpoint_dir, **kwargs)
