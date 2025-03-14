import os

import pytest
from defs.common import venv_check_call, venv_mpi_check_call
from defs.conftest import get_device_count, get_sm_version

LPIPS_THRESHOLD = 0.05


def _read_image(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    import numpy as np
    import torch
    from PIL import Image

    img = torch.from_numpy(np.asarray(Image.open(path)).copy()).permute(
        2, 0, 1).float() / 255.0
    img = img.unsqueeze(0).to("cuda")

    return img


@pytest.fixture(scope="module")
def reference_image(sdxl_example_root, llm_venv):
    print("Creating reference image...")
    build_cmd = [f"{sdxl_example_root}/build_sdxl_unet.py", "--size=1024"]
    venv_check_call(llm_venv, build_cmd)

    ref_img = os.path.join(llm_venv.get_working_directory(), "sdxl_ref.png")
    inference_cmd = [
        f"{sdxl_example_root}/run_sdxl.py",
        f"--size={1024}",
        f"--num-warmup-runs={0}",
        f"--avg-runs={1}",
        f"--output={ref_img}",
    ]
    venv_check_call(llm_venv, inference_cmd)

    return _read_image(ref_img)


@pytest.mark.parametrize("num_gpu", [2, 4],
                         ids=lambda num_gpu: f'num_gpu:{num_gpu}')
def test_sdxl_1node_multi_gpus(sdxl_example_root, sdxl_model_root, llm_venv,
                               reference_image, num_gpu):
    if get_device_count() < num_gpu:
        pytest.skip(f"devices are less than {num_gpu}.")

    if get_sm_version() >= 100:
        pytest.skip(f"This test is not supported in Blackwell architecture")

    print("Building engines...")
    build_cmd = [
        f"{sdxl_example_root}/build_sdxl_unet.py",
        f"--model_dir={sdxl_model_root}", f"--size={1024}"
    ]
    venv_mpi_check_call(
        llm_venv,
        ["mpirun", "-n", str(num_gpu), "--allow-run-as-root"], build_cmd)

    print("Run SDXL...")
    output_img = os.path.join(llm_venv.get_working_directory(),
                              f"sdxl_output_{num_gpu}_gpu.png")
    inference_cmd = [
        f"{sdxl_example_root}/run_sdxl.py", f"--model_dir={sdxl_model_root}",
        f"--size={1024}", f"--num-warmup-runs={0}", f"--avg-runs={1}",
        f"--output={output_img}"
    ]
    venv_mpi_check_call(
        llm_venv,
        ["mpirun", "-n", str(num_gpu), "--allow-run-as-root"], inference_cmd)

    img = _read_image(output_img)

    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to("cuda")
    lpips.update(reference_image, img)

    lpips_score = lpips.compute().item()
    print("LPIPS score: %f" % lpips_score)

    assert lpips_score < LPIPS_THRESHOLD
