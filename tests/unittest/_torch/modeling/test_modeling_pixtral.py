import gc
import os
import pathlib
import pickle
import sys

import cloudpickle
import mpi4py
import pytest
import torch
import transformers
from transformers.models.pixtral import modeling_pixtral as hf_modeling_pixtral

import tensorrt_llm
from tensorrt_llm import mapping as mapping_lib
from tensorrt_llm._torch import model_config as model_config_lib
from tensorrt_llm._torch.models import modeling_pixtral

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
cloudpickle.register_pickle_by_value(sys.modules[__name__])
mpi4py.MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)


def make_pixtral_vision_config():
    # Values taken from:
    # https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/config.json
    return model_config_lib.ModelConfig(
        pretrained_config=transformers.PixtralVisionConfig(
            hidden_size=1024,
            num_attention_heads=16,
            dtype=torch.bfloat16,
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


@torch.no_grad()
@pytest.mark.usefixtures("set_seed")
def test_pixtral_vision_model_vs_hf():
    dtype = torch.bfloat16
    device = torch.device("cuda")
    pixtral_vision_config = make_pixtral_vision_config()
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

    batch_size = 2
    height, width, channels = 123, 456, 3
    pixel_values = torch.randn(batch_size, channels, height, width, device=device, dtype=dtype)
    image_sizes = torch.tensor([[height, width], [height - 7, width - 11]])
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


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@torch.no_grad()
def test_tensor_parallelism(mpi_pool_executor, tmp_path):
    mapping = mapping_lib.Mapping(world_size=2, tp_size=2)
    if (num_available_devices := torch.cuda.device_count()) < mapping.world_size:
        pytest.skip(f"{num_available_devices=} is less than the requested {mapping.world_size}.")

    dtype = torch.bfloat16
    device = torch.device("cuda")
    pixtral_vision_config = make_pixtral_vision_config()
    pretrained_config = pixtral_vision_config.pretrained_config

    hf_pixtral_model = init_hf_model(
        cls=hf_modeling_pixtral.PixtralVisionModel,
        config=pretrained_config,
        dtype=dtype,
        device=device,
    )
    # Save HF weights to disk so they can be used by worker processes.
    state_dict = hf_pixtral_model.state_dict()
    hf_weights_path = tmp_path / "hf_weights.pt"
    torch.save(state_dict, hf_weights_path)

    pixtral_model = (
        modeling_pixtral.PixtralVisionModel(model_config=pixtral_vision_config).eval().to("cuda")
    )
    pixtral_model.load_weights(state_dict)
    # Save the number of params to check that the model gets shared in the workers.
    num_params = sum(p.numel() for p in pixtral_model.parameters())

    batch_size = 2
    height, width, channels = 123, 456, 3
    pixel_values = torch.randn(batch_size, channels, height, width, device=device, dtype=dtype)
    image_sizes = torch.tensor([[height, width], [height - 7, width - 11]])

    ref_out = pixtral_model(pixel_values=pixel_values, image_sizes=image_sizes)

    # Move to CPU before sending across process barrier.
    ref_out = ref_out.to("cpu")
    pixel_values = pixel_values.to("cpu")
    image_sizes = image_sizes.to("cpu")

    # Free up GPU memory on rank 0.
    del state_dict
    del hf_pixtral_model
    del pixtral_model
    gc.collect()
    torch.cuda.empty_cache()

    # NOTE: we cannot send `pixtral_vision_config` across the process barrier, as it contains
    # `weakref` objects, which cannot be pickled. Instead, each worker will recreate it by
    # calling the `make_pixtral_vision_config` function.
    world_size = mapping.world_size
    results = mpi_pool_executor.starmap(
        _run_pixtral_and_compare_against_ref,
        [
            (
                mapping_lib.Mapping(tp_size=world_size, world_size=world_size, rank=rank),
                hf_weights_path,
                pixel_values,
                image_sizes,
                ref_out,
                num_params,
            )
            for rank in range(world_size)
        ],
    )

    for r in results:
        assert r


def _run_pixtral_and_compare_against_ref(
    mapping: mapping_lib.Mapping,
    hf_weights_path: pathlib.Path,
    pixel_values: torch.Tensor,
    image_sizes: torch.Tensor,
    expected_output: torch.Tensor,
    total_num_params: int,
) -> bool:
    rank = tensorrt_llm.mpi_rank()
    # Smoke check.
    world_size = tensorrt_llm.mpi_world_size()
    assert world_size > 1

    torch.cuda.set_device(rank)

    pixel_values = pixel_values.to("cuda")
    image_sizes = image_sizes.to("cuda")
    expected_output = expected_output.to("cuda")

    pixtral_vision_config = make_pixtral_vision_config()
    pixtral_vision_config.mapping = mapping
    pixtral_model = (
        modeling_pixtral.PixtralVisionModel(model_config=pixtral_vision_config).eval().to("cuda")
    )
    state_dict = torch.load(hf_weights_path, map_location="cuda")
    pixtral_model.load_weights(state_dict)

    # Smoke check to see that we are indeed sharding the model.
    rank_num_params = sum(p.numel() for p in pixtral_model.parameters())
    params_fraction = rank_num_params / total_num_params
    assert params_fraction < 1.0
    assert params_fraction == pytest.approx(1.0 / world_size, rel=1e-2)

    out = pixtral_model(
        pixel_values=pixel_values,
        image_sizes=image_sizes,
    )
    torch.testing.assert_close(out, expected_output, atol=0.2, rtol=0.2)
    return True
