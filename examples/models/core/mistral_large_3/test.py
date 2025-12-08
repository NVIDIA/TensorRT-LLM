TEST_TOKENIZER = False
TEST_CONFIG_LOADER = False
TEST_CHECKPOINT_LOADER = True

MODEL_DIR = (
    "/home/scratch.trt_llm_data/llm-models/Mistral-Large-3-675B/Mistral-Large-3-675B-Instruct-2512"
)
if TEST_TOKENIZER:
    ## Test Tokenizer
    from transformers import AutoTokenizer

    mtok = AutoTokenizer.from_pretrained(MODEL_DIR)
    print(f"mtok: {mtok}")

    print(mtok.encode("Hello, world!"))
    print(mtok.decode([1, 22177, 1044, 4304, 1033]))

if TEST_CONFIG_LOADER or True:
    ## Test config loader
    from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader

    config_loader = MistralConfigLoader()
    config = config_loader.load(MODEL_DIR)
    print(f"config: {config}")

if TEST_CHECKPOINT_LOADER:
    from tensorrt_llm._torch.models.checkpoints.mistral.checkpoint_loader import (
        MistralCheckpointLoader,
    )
    from tensorrt_llm._torch.models.checkpoints.mistral.weight_mapper import (
        MistralLarge3WeightMapper,
    )

    weight_mapper = MistralLarge3WeightMapper()
    loader = MistralCheckpointLoader(weight_mapper=weight_mapper)
    weights_dict = loader.load_weights(MODEL_DIR, model_config=config)
    print(f"weights_dict.keys(): {weights_dict.keys()}")
