from .factory import ModelFactoryRegistry
from .hf import AutoModelForCausalLMFactory


@ModelFactoryRegistry.register("NemotronFlashForCausalLM")
class NemotronFlashForCausalLMFactory(AutoModelForCausalLMFactory):
    # TODO: custom tokenizer initialization system
    def init_tokenizer(self):
        if self.tokenizer is None:
            return None

        from .custom import NemotronFlashPreTrainedTokenizerFast

        model_config, _ = self._get_model_config()
        return NemotronFlashPreTrainedTokenizerFast.from_pretrained(
            self.tokenizer,
            **self.tokenizer_kwargs,
            num_memory_tokens=model_config.num_memory_tokens,
            vocab_size_model=model_config.vocab_size,
        )
