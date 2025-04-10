from pathlib import Path

import pytest
import torch
from defs.common import venv_check_call
from defs.conftest import get_sm_version, llm_models_root


@pytest.mark.parametrize("model_name", ["DeepSeek-R1"], ids=["deepseek_r1"])
@pytest.mark.parametrize("quant", ["fp4", "fp8"])
@pytest.mark.parametrize("tp_size", [8], ids=["tp8"])
@pytest.mark.parametrize("pp_size", [1], ids=["pp1"])
@pytest.mark.parametrize("ep_size", [1, 4, 8], ids=["ep1", "ep4", "ep8"])
@pytest.mark.parametrize("mtp_nextn", [0, 1, 2],
                         ids=["nextn0", "nextn1", "nextn2"])
@pytest.mark.parametrize("enable_dp", [True, False],
                         ids=["enable_dp", "disable_dp"])
@pytest.mark.parametrize("enable_cuda_graph", [True, False],
                         ids=["enable_cuda_graph", "disable_cuda_graph"])
@pytest.mark.parametrize(
    "enable_overlap_scheduler", [True, False],
    ids=["enable_overlap_scheduler", "disable_overlap_scheduler"])
def test_deepseek_gpqa_llmapi(llmapi_example_root, llm_datasets_root, llm_venv,
                              model_name, quant, tp_size, pp_size, ep_size,
                              mtp_nextn, enable_dp, enable_cuda_graph,
                              enable_overlap_scheduler):
    model_path = {
        "fp8": "DeepSeek-R1",
        "fp4": "DeepSeek-R1-FP4",
    }
    assert quant in model_path.keys()

    is_fp8 = quant == "fp8"
    is_fp4 = quant == "fp4"

    if ep_size > tp_size:
        pytest.skip(
            f"Expert parallel size {ep_size} must be less than or equal to tensor parallel size {tp_size}"
        )

    if torch.cuda.device_count() < tp_size * pp_size:
        pytest.skip(f"Not enough GPUs available, need {tp_size * pp_size} "
                    f"but only have {torch.cuda.device_count()}")

    if is_fp8:
        pytest.skip(
            f"FP8 is not supported for gpqa test, and it will be added in the near future"
        )

    if is_fp4 and get_sm_version() < 100:
        pytest.skip(
            f"FP4 is not supported in this SM version {get_sm_version()}")

    if pp_size > 1:
        pytest.skip(
            "PP is not supported for gpqa test, and it will be added in the near future"
        )

    model_dir = str(Path(llm_models_root()) / model_name / model_path[quant])
    gpqa_data_path = str(Path(llm_datasets_root) / "gpqa/gpqa_diamond.csv")

    assert Path(model_dir).exists()

    print("Run GPQA test")
    gpqa_cmd = [
        f"{llmapi_example_root}/../gpqa_llmapi.py",
        f"--hf_model_dir={model_dir}", f"--data_dir={gpqa_data_path}",
        f"--tp_size={tp_size}", f"--ep_size={ep_size}", "--concurrency=8",
        f"--mtp_nextn={mtp_nextn}", "--print_iter_log", "--batch_size=32",
        "--max_num_tokens=4096", "--check_accuracy",
        "--accuracy_threshold=0.65", "--num_runs=3"
    ]
    if enable_cuda_graph:
        gpqa_cmd.append("--use_cuda_graph")
    if enable_overlap_scheduler:
        gpqa_cmd.append("--enable_overlap_scheduler")
    if enable_dp:
        gpqa_cmd.append("--enable_attention_dp")

    venv_check_call(llm_venv, gpqa_cmd)
