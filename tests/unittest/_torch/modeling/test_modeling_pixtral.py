import pytest
import torch
import transformers
from transformers.models.pixtral import modeling_pixtral as hf_modeling_pixtral

from tensorrt_llm import mapping as mapping_lib
from tensorrt_llm._torch import model_config as model_config_lib
from tensorrt_llm._torch.models import modeling_pixtral


@pytest.fixture
def pixtral_vision_config():
    # Values taken from:
    # https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/config.json
    return model_config_lib.ModelConfig(
        pretrained_config=transformers.PixtralVisionConfig(
            hidden_size=1024,
            num_attention_heads=16,
            torch_dtype=torch.bfloat16,
            hidden_act="silu",
        ),
    )


@pytest.fixture
def set_seed():
    torch.manual_seed(322)


def init_hf_model(cls, config, dtype, device):
    """Helper function for initializing a model from `transformers`.

    The reason this function exists is: by default, instantiating a `transformers` model also
    eagerly initializes the model's weights on the CPU, which takes an absurdly long time to
    complete.

    Instead, we lazily instantiate the model, and initialize the weights only after moving it to
    the requested `device`.
    """
    from transformers import modeling_utils as t_modeling_utils

    with t_modeling_utils.no_init_weights():
        model = cls(config).eval()

    model.to(device=device)
    model.init_weights()
    model.to(dtype=dtype)

    return model


@pytest.mark.parametrize(
    "mapping",
    [
        mapping_lib.Mapping(world_size=2, tp_size=2),
        mapping_lib.Mapping(world_size=3, tp_size=3),
        mapping_lib.Mapping(world_size=4, tp_size=2, pp_size=2),
        mapping_lib.Mapping(world_size=8, tp_size=2, pp_size=2, cp_size=2),
    ],
)
def test_pixtral_vision_model_rejects_tp_size_greater_than_one(pixtral_vision_config, mapping):
    pixtral_vision_config.mapping = mapping
    with pytest.raises(NotImplementedError, match="tp_size > 1"):
        modeling_pixtral.PixtralVisionModel(model_config=pixtral_vision_config)


@torch.no_grad()
@pytest.mark.usefixtures("set_seed")
def test_pixtral_vision_model_vs_hf(pixtral_vision_config):
    dtype = torch.bfloat16
    device = torch.device("cuda")
    pretrained_config = pixtral_vision_config.pretrained_config

    pixtral_model = (
        modeling_pixtral.PixtralVisionModel(model_config=pixtral_vision_config).eval().to(device)
    )
    hf_pixtral_model = init_hf_model(
        cls=hf_modeling_pixtral.PixtralVisionModel,
        config=pretrained_config,
        dtype=dtype,
        device=device,
    )
    # Make sure both models have the same weights.
    pixtral_model.load_weights(hf_pixtral_model.state_dict())

    batch_size = 1
    height, width, channels = 123, 456, 3
    pixel_values = torch.randn(batch_size, channels, height, width, device=device, dtype=dtype)
    image_sizes = torch.tensor([[height, width]])
    out = pixtral_model(
        pixel_values=pixel_values,
        image_sizes=image_sizes,
    )
    with torch.autocast(device_type="cuda", dtype=dtype):
        hf_out = (
            hf_pixtral_model(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
            )
            .last_hidden_state.squeeze(0)
            .to(dtype=dtype)
        )

    torch.testing.assert_close(out, hf_out, atol=0.2, rtol=0.2)
