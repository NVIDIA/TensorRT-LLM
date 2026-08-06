import os

try:
    pass
except ImportError:
    pass

import pytest
from defs.common import venv_check_call
from defs.conftest import get_device_count, llm_models_root


@pytest.fixture(scope="module")
def ray_example_root(llm_root):
    example_root = os.path.join(llm_root, "examples", "ray_orchestrator")
    return example_root


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("tp_size,pp_size,ep_size", [
    (2, 1, 2),
], ids=["tep2"])
def test_llm_inference_distributed_ray(ray_example_root, llm_venv, tp_size,
                                       pp_size, ep_size):
    world_size = tp_size * pp_size

    if get_device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs.")

    script_path = os.path.join(ray_example_root,
                               "llm_inference_distributed_ray.py")

    cmd = [
        script_path, "--tp_size",
        str(tp_size), "--pp_size",
        str(pp_size), "--moe_ep_size",
        str(ep_size)
    ]

    model_dir = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"
    cmd.extend(["--model_dir", model_dir])

    venv_check_call(llm_venv, cmd)
