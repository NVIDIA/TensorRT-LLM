# --------------------------------------------------------
# Adapted from https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B under MIT License
#     LICENSE is in incl_licenses directory.
# --------------------------------------------------------

import copy

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class NVLM_D_Config(PretrainedConfig):
    model_type = 'NVLM_D'
    is_composition = True

    def __init__(
        self,
        use_backbone_lora=0,
        use_llm_lora=0,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        dynamic_image_size=False,
        use_thumbnail=False,
        ps_version='v1',
        image_tag_type="nvlm",
        vision_projection_config=None,
        **kwargs
    ):
        super().__init__(**kwargs)


        self.vision_config = AutoConfig.from_pretrained("nvidia/C-RADIOv2-H", trust_remote_code=True)
        self.llm_config = AutoConfig.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
        self.llm_config.vocab_size = 128512

        if vision_projection_config is None:
            vision_projection_config = {
                "hidden_size": self.llm_config.hidden_size,
                "normalization": "LayerNorm",
            }
        self.vision_projection_config = VisionProjectionConfig(**vision_projection_config)

        # Assign configuration values
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # Pixel shuffle version
        self.image_tag_type = image_tag_type

        # Log important parameters
        logger.info(f'ps_version: {self.ps_version}')

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Overrides the default `PretrainedConfig.to_dict`.

        Returns:
            Dict[str, Any]: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['use_thumbnail'] = self.use_thumbnail
        output['ps_version'] = self.ps_version
        output['image_tag_type'] = self.image_tag_type
        output['vision_projection_config'] = self.vision_projection_config.to_dict()

        return output


class VisionProjectionConfig(PretrainedConfig):
    """Vision projection configuration."""
    def __init__(self, hidden_size, vit_hidden_size, llm_hidden_size, **kwargs):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.vit_hidden_size = vit_hidden_size
        self.llm_hidden_size = llm_hidden_size

