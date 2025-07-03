import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import \
    HfCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.hf.config_loader import \
    HfConfigLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import \
    HfWeightLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper


@pytest.fixture
def llm_kwargs(request):
    variant = request.param
    if variant == "vanilla":
        return {}
    if variant == "format_only":
        return {"checkpoint_format": "HF"}
    if variant == "partial_loader":
        return {"checkpoint_loader": HfCheckpointLoader()}
    if variant == "full_loader":
        return {
            "checkpoint_loader":
            HfCheckpointLoader(
                weight_loader=HfWeightLoader(),
                weight_mapper=HfWeightMapper(),
                config_loader=HfConfigLoader(),
            )
        }
    raise ValueError(f"Unknown variant: {variant}")


@pytest.mark.parametrize(
    "llm_kwargs",
    ["vanilla", "format_only", "partial_loader", "full_loader"],
    indirect=True,
)
def test_successful_checkpoint_loader_initialization(llm_kwargs):
    model_path = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"

    llm = LLM(model=model_path, **llm_kwargs)

    prompts = ["Hello, how are you?"]
    outputs = llm.generate(prompts)
    print(f"Model Output is: {outputs[0].outputs[0].text}")
    assert 1 == 1


# def test_mock_deepL():
