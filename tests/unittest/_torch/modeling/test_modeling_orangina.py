import unittest

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_orangina import (OranginaForCausalLM,
                                                          OranginaModelConfig)


class TestOrangina(unittest.TestCase):

    def test_sanity(self):
        orangina_config = OranginaModelConfig()
        orangina_config.num_hidden_layers = 8

        quant_config = None
        model_config = ModelConfig(pretrained_config=orangina_config,
                                   quant_config=quant_config)
        model = OranginaForCausalLM(model_config)
        print(model)
