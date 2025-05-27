from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.config_loader_interface import \
    ConfigLoaderInterface
from tensorrt_llm._torch.models.modeling_utils import \
    register_auto_config_loader


@register_auto_config_loader("HF")
class HfConfigLoader(ConfigLoaderInterface):

    def load(self, checkpoint_dir: str, **kwargs) -> ModelConfig:
        return ModelConfig.from_pretrained(checkpoint_dir, **kwargs)
