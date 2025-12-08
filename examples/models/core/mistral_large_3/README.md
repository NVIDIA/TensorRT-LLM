## How to use the modules

The following explains how to use the different modules of Mistral Large V3.

```python
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3ForCausalLM
from tensorrt_llm._torch.models.modeling_mistral import Mistral3VLM
# from tensorrt_llm.llmapi.tokenizer import MistralTokenizer
from tensorrt_llm._torch.models.checkpoints.mistral.checkpoint_loader import MistralCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.mistral.weight_mapper import MistralLarge3WeightMapper
from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader
from transformers import AutoTokenizer
```

### Tokenizer
```python
mtok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
```

### Config and model instance
```python
config_loader = MistralConfigLoader()
config = config_loader.load(MODEL_DIR)

model = Mistral3VLM(model_config=config)
assert isinstance(model.llm, DeepseekV3ForCausalLM)
```

### Checkpoint loading
```python
weight_mapper=MistralLarge3WeightMapper()
loader = MistralCheckpointLoader(weight_mapper=weight_mapper)

weights_dict = loader.load_weights(MODEL_DIR)
```

### Weight loading
#### E2E
```python
model.load_weights(weights_dict, weight_mapper=weight_mapper) # target usage
```
#### By module
```python
def _filter_weights(weights, prefix):
    return {
        name[len(prefix):]: weight
        for name, weight in weights.items() if name.startswith(prefix)
    }

llm_weights = weight_mapper.rename_by_params_map(
    params_map=weight_mapper.mistral_llm_mapping,
    weights=_filter_weights(weights_dict, "language_model."))
model.llm.load_weights(llm_weights, weight_mapper=weight_mapper)
```
