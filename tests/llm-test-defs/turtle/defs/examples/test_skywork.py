import os

import pytest
from defs.common import (convert_weights, generate_summary_cmd, venv_check_call,
                         venv_mpi_check_call)
from defs.trt_test_alternative import check_call


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize(
    "use_attention_plugin",
    [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"],
)
@pytest.mark.parametrize(
    "use_gemm_plugin",
    [True, False],
    ids=["enable_gemm_plugin", "disable_gemm_plugin"],
)
@pytest.mark.parametrize(
    "skywork_model_root",
    ["Skywork-13B-base", "Skywork-13B-Math"],
    indirect=True,
)
@pytest.mark.parametrize(
    "context_fmha_type",
    ['enabled', 'enabled_with_fp32_acc', 'disabled'],
)
@pytest.mark.parametrize(
    "dtype",
    ["bfloat16"],
)
def test_llm_skywork_single_gpu_summary(skywork_example_root, cmodel_dir,
                                        skywork_model_root, llm_datasets_root,
                                        llm_rouge_root, llm_venv, engine_dir,
                                        use_attention_plugin, use_gemm_plugin,
                                        context_fmha_type, dtype):
    model_name = os.path.basename(skywork_model_root)
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=f"{skywork_example_root}/../llama",
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=skywork_model_root,
                                data_type=dtype)
    print("Building engines...")
    max_input_len = max_output_len = 512
    max_batch_size = 32
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--max_batch_size={max_batch_size}",
        f"--max_input_len={max_input_len}",
        f"--max_seq_len={max_output_len + max_input_len}",
        f"--output_dir={engine_dir}",
    ]
    if use_attention_plugin:
        build_cmd.append(f"--gpt_attention_plugin={dtype}")
    if use_gemm_plugin:
        build_cmd.append(f"--gemm_plugin={dtype}")
    if context_fmha_type == 'enabled':
        build_cmd.append("--context_fmha=enable")
    elif context_fmha_type == 'disabled':
        build_cmd.append("--context_fmha=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run engines...")
    summary_cmd = generate_summary_cmd(skywork_example_root,
                                       hf_model_dir=skywork_model_root,
                                       engine_dir=engine_dir,
                                       data_type=dtype,
                                       max_input_length=max_input_len,
                                       output_len=max_input_len,
                                       batch_size=max_batch_size,
                                       tensorrt_llm_rouge1_threshold=19,
                                       eval_task="summarize",
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)
    if context_fmha_type == 'enabled_with_fp32_acc':
        summary_cmd.append("--enable_context_fmha_fp32_acc")
    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize(
    "use_attention_plugin",
    [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"],
)
@pytest.mark.parametrize(
    "use_gemm_plugin",
    [True, False],
    ids=["enable_gemm_plugin", "disable_gemm_plugin"],
)
@pytest.mark.parametrize(
    "skywork_model_root",
    ["Skywork-13B-base", "Skywork-13B-Math"],
    indirect=True,
)
@pytest.mark.parametrize(
    "context_fmha_type",
    ['enabled', 'enabled_with_fp32_acc', 'disabled'],
)
def test_llm_skywork_1node_2gpus_summary(skywork_example_root, cmodel_dir,
                                         skywork_model_root, llm_datasets_root,
                                         llm_rouge_root, llm_venv, engine_dir,
                                         use_attention_plugin, use_gemm_plugin,
                                         context_fmha_type):
    dtype = "bfloat16"
    model_name = os.path.basename(skywork_model_root)
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=f"{skywork_example_root}/../llama",
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=skywork_model_root,
                                data_type=dtype,
                                gpus=2)

    print("Building engines...")
    max_input_len = max_output_len = 512
    max_batch_size = 32
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--max_batch_size={max_batch_size}",
        f"--max_input_len={max_input_len}",
        f"--max_seq_len={max_output_len + max_input_len}",
        f"--output_dir={engine_dir}",
        f"--workers={2}",
    ]
    if use_attention_plugin:
        build_cmd.append(f"--gpt_attention_plugin={dtype}")
    if use_gemm_plugin:
        build_cmd.append(f"--gemm_plugin={dtype}")
    if context_fmha_type == 'enabled':
        build_cmd.append("--context_fmha=enable")
    elif context_fmha_type == 'disabled':
        build_cmd.append("--context_fmha=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run engines...")
    summary_cmd = generate_summary_cmd(skywork_example_root,
                                       hf_model_dir=skywork_model_root,
                                       engine_dir=engine_dir,
                                       data_type=dtype,
                                       max_input_length=max_input_len,
                                       output_len=max_input_len,
                                       batch_size=max_batch_size,
                                       tensorrt_llm_rouge1_threshold=19,
                                       eval_task="summarize",
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)
    if context_fmha_type == 'enabled_with_fp32_acc':
        summary_cmd.append("--enable_context_fmha_fp32_acc")
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)
