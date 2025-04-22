"""Experimental Llama-4 support."""

from transformers.models.llama4.modeling_llama4 import Llama4ForConditionalGeneration

from .factory import ModelFactoryRegistry
from .hf import HFFactory


@ModelFactoryRegistry.register("hf-llama4")
class Llama4Factory(HFFactory):
    """The Llama 4 model is currently not supported under any of the Auto classes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_kwargs["text_config"] = self.model_kwargs.get("text_config", {})
        self.model_kwargs["text_config"]["use_cache"] = False
        self.model_kwargs["text_config"]["max_position_embeddings"] = self.model_kwargs[
            "max_position_embeddings"
        ]

    def _from_config(self, *args, **kwargs):
        kwargs.pop("trust_remote_code", None)
        return Llama4ForConditionalGeneration._from_config(*args, **kwargs)

    @property
    def automodel_from_config(self):
        return self._from_config
