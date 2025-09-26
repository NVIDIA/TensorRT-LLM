"""
MLPerf target tests
"""
import os
import subprocess
from argparse import Namespace
from copy import deepcopy

import pytest
from defs.common import get_cpp_benchmark, get_trt_llm_lib_dir, venv_check_call
from defs.conftest import get_device_count, get_gpu_device_list, llm_models_root
from defs.trt_test_alternative import check_call

### End of utility functions
"""
Test: Runs the gptManagerBenchmark on LLama TRTLLM engine and checks accuracy of predictions
Steps:
    1. Quantize the model:          step_quantize
    2. Build the engine:            step_engine_build
    3. Run engine and get outputs:  step_run_llm
    4. Check prediction accuracy:   step_check_accuracy
"""


# Test step 1: Quantize the model
# MLPerf step: python examples/quantization/quantize.py --dtype=float16  --output_dir=<> --model_dir=<> --qformat=fp8 --kv_cache_dtype=fp8 --tp_size 2
def step_quantize(tp_size, llm_venv, llm_root, model_root, model,
                  calib_dataset):
    quantized_model_path = "{}/test_mlperf_quantized_models/{}-tp{}-pp1/".format(
        llm_venv.get_working_directory(), model, tp_size)
    tekit_example_dir = os.path.join(llm_root, "examples/")

    # Set MLPerf params explicitly
    quantize_cmd = [
        f"{tekit_example_dir}/quantization/quantize.py", "--dtype=float16",
        "--qformat=fp8", "--kv_cache_dtype=fp8", f"--tp_size={tp_size}",
        f"--output_dir={quantized_model_path}", f"--model_dir={model_root}",
        "--calib_size=1024", f"--calib_dataset={calib_dataset}"
    ]

    venv_check_call(llm_venv, quantize_cmd)

    return quantized_model_path


# Test step 2: Build the TRTLLM engine
# MLPerf step:
#   python3 -m tensorrt_llm.commands.build --gpt_attention_plugin=float16 --max_batch_size=896 --max_input_len=1024 --max_seq_len=2048 --max_beam_width=1 \
#         --max_num_tokens=4096 --output_dir=<> --checkpoint_dir=<> --context_fmha=enable --remove_input_padding=enable \
#         --paged_kv_cache=enable --workers=2


def step_engine_build(quantized_model_path, system_config, engine_dir,
                      llm_venv):

    batch_size = system_config.batch_size
    beam_width = system_config.beam_width
    max_input_len = system_config.max_input_len
    max_seq_len = system_config.max_seq_len
    max_num_tokens = system_config.max_num_tokens
    num_workers = system_config.num_workers
    use_fp8_context_fmha = "enable" if system_config.fp8_fmha else "disable"

    build_cmd = [
        "trtllm-build",
        "--gpt_attention_plugin=float16",
        f"--max_batch_size={batch_size}",
        f"--max_input_len={max_input_len}",
        f"--max_seq_len={max_seq_len}",
        f"--max_beam_width={beam_width}",
        f"--max_num_tokens={max_num_tokens}",
        f"--output_dir={engine_dir}",
        f"--checkpoint_dir={quantized_model_path}",
        "--context_fmha=enable",
        f"--use_fp8_context_fmha={use_fp8_context_fmha}",
        "--remove_input_padding=enable",
        "--paged_kv_cache=enable",
        f"--workers={num_workers}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)
    return engine_dir


DEFAULT_RPARAMS = Namespace(
    engine_dir=None,
    api="executor",
    # type="IFB",
    dataset=None,
    output_csv="gptmanager_bench_results.csv",
    max_num_samples=24576,
    beam_width=1,
    warm_up=2,
    eos_id=-1,
    pad_id=-1,
    max_tokens_in_paged_kvcache=None,
    kv_cache_free_gpu_mem_fraction=None,
    streaming=False,
    enable_kv_cache_reuse=False,
    enable_chunked_context=False,
    return_context_logits=False,
    return_generation_logits=False,
    scheduler_policy="guaranteed_no_evict",
    static_emulated_batch_size=None,
    log_level="verbose",
    log_iteration_data=False,
    wait_sleep="25",
    lora_dir=None,
    lora_host_cache_bytes=None,
    lora_num_device_mod_layers=None,
    responses_json=None)
"""
./benchmarks/gptManagerBenchmark \
    --engine_dir <> \
    --dataset <> \
    --max_num_samples 24576 \
    --beam_width 1 \
    --eos_id 2 \
    --pad_id 2 \
    --kv_cache_free_gpu_mem_fraction 0.95 \
    --scheduler_policy max_utilization \
    --output_csv <>
"""


# Test step 3: Run the gptManagerBenchmark and get outputs
def step_run_llm(system_config,
                 engine_path,
                 dataset_path,
                 llm_venv,
                 llm_root,
                 kv_cache_free_gpu_mem_fraction=0.95):
    tp, pp = system_config.tp_size, system_config.pp_size
    eos_id, pad_id = system_config.eos_id, system_config.pad_id
    max_num_samples = system_config.num_samples
    beam_width = system_config.beam_width

    benchmark_exe = get_cpp_benchmark('gptManagerBenchmark', llm_root)
    workspace_path = llm_venv.get_working_directory()
    run_params = deepcopy(DEFAULT_RPARAMS)
    run_params.beam_width = beam_width
    run_params.engine_dir = engine_path
    run_params.dataset = dataset_path
    run_params.max_num_samples = max_num_samples
    run_params.eos_id = eos_id
    run_params.pad_id = pad_id
    run_params.kv_cache_free_gpu_mem_fraction = kv_cache_free_gpu_mem_fraction
    run_params.scheduler_policy = "max_utilization"
    run_params.responses_json = os.path.join(
        workspace_path, f"responses_test_mlperf_tp{tp}_pp{pp}.json")
    run_params.output_csv = os.path.join(
        workspace_path, f"perf_stats_test_mlperf_tp{tp}_pp{pp}.csv")

    run_params_dict = vars(run_params)
    run_params_dict['type'] = "IFB"

    bench_cmd = [benchmark_exe]
    for key, val in run_params_dict.items():
        if val is None or val is False:
            continue
        if val is True:
            val = ""
        bench_cmd.append("--" + str(key))
        bench_cmd.append(str(val))

    envs = deepcopy(os.environ)
    _ = envs.pop("CUDA_VISIBLE_DEVICES", "")
    envs[
        "LD_LIBRARY_PATH"] = f'{get_trt_llm_lib_dir(llm_venv)}:{os.path.dirname(benchmark_exe)}:{envs.get("LD_LIBRARY_PATH", "")}'

    print(
        f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", None)}')

    num_ranks = tp * pp
    if num_ranks > 1:
        mpi_cmd = ["mpirun", "-n", f"{num_ranks}", "--allow-run-as-root"]
        bench_cmd = mpi_cmd + bench_cmd

    print(f"Running gptManagerBenchmark using cmd: {' '.join(bench_cmd)}")
    subprocess.check_output(bench_cmd, env=envs)
    return run_params.responses_json


def step_check_accuracy(responses_file, dataset_path, model_root, llm_venv,
                        llm_root):
    """
        python3 /code/tensorrt_llm/benchmarks/python/check_accuracy_mlperf.py
                --dataset <>
                --responses <>
                --base_model <>
    """
    accuracy_script = os.path.join(
        llm_root, "benchmarks/python/check_accuracy_mlperf.py")
    accuracy_check_cmd = [
        f"{accuracy_script}", "--dataset", f"{dataset_path}", "--responses",
        f"{responses_file}", "--base_model", f"{model_root}"
    ]
    venv_check_call(llm_venv, accuracy_check_cmd)


LlamaBaseSystem = Namespace(tp_size=None,
                            pp_size=1,
                            batch_size=None,
                            max_input_len=1024,
                            max_seq_len=2048,
                            max_num_tokens=4096,
                            beam_width=1,
                            num_workers=None,
                            num_samples=24576,
                            eos_id=2,
                            pad_id=2,
                            fp8_fmha=False)

GptjBaseSystem = Namespace(tp_size=1,
                           pp_size=1,
                           batch_size=None,
                           max_input_len=1919,
                           max_seq_len=2047,
                           max_num_tokens=4096,
                           beam_width=4,
                           num_workers=1,
                           num_samples=13368,
                           eos_id=50256,
                           pad_id=50256,
                           fp8_fmha=False)


def get_mlperf_system_config(model: str, system: str, fp8_fmha: bool):
    if model == "llama_v2_70b_chat":
        return get_mlperf_llama_system_config(system)
    elif model == "gpt_j":
        return get_mlperf_gptj_system_config(system, fp8_fmha)
    raise RuntimeError(f"Unexpected model: {system}")


def get_mlperf_llama_system_config(system: str):
    system_config = deepcopy(LlamaBaseSystem)
    if system == "H100x2":
        system_config.tp_size = 2
        system_config.batch_size = 896
        system_config.num_workers = 2
    elif system == "H200x1":
        system_config.tp_size = 1
        system_config.batch_size = 806
        system_config.num_workers = 1
    else:
        raise RuntimeError(f"No Llama config found for system: {system}")

    return system_config


def get_mlperf_gptj_system_config(system: str, fp8_fmha: bool):
    system_config = deepcopy(GptjBaseSystem)
    system_config.fp8_fmha = fp8_fmha
    if system == "H100x1":
        system_config.batch_size = 192
    elif system == "H200x1":
        system_config.batch_size = 396
    else:
        raise RuntimeError(f"No GPT-J config found for system: {system}")

    return system_config


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("fp8_fmha", [True, False], ids=["fp8_fmha_enable", ""])
@pytest.mark.parametrize("system", ["H100x2", "H200x1", "H100x1"])
@pytest.mark.parametrize("model", ["llama_v2_70b_chat", "gpt_j"])
def test_mlperf_results(system, model, fp8_fmha, llm_venv, llm_root,
                        engine_dir):
    "Run mlperf tests on H100/H200."

    if f"NVIDIA {system[:-2]}" not in get_gpu_device_list()[0]:
        pytest.skip(f"{system} test is not supported.")

    if "gpt_j" in model and "x2" in system:
        pytest.skip("This test is invalid.")
    if "v2_70b" in model and "H100x1" in system:
        pytest.skip("This test is invalid.")
    if "v2_70b" in model and "x2" in system and get_device_count() < 2:
        pytest.skip("This test is invalid.")

    system_config = get_mlperf_system_config(model, system, fp8_fmha)
    models_root = llm_models_root()

    if model == "llama_v2_70b_chat":
        model_root = os.path.join(models_root, "llama-models-v2",
                                  "llama-v2-70b-chat-hf")
        input_dataset = os.path.join(
            models_root, "datasets", "common",
            "open_orca_inputs_24576.trtllm.gptManagerBenchmark.json")
        reference_dataset = os.path.join(
            models_root, "datasets", "common",
            "open_orca_gpt4_tokenized_llama.sampled_24576.pkl")
        calib_dataset = os.path.join(models_root, "datasets", "common",
                                     "mlperf_gptj_openorca_calibration_1k")
    elif model == "gpt_j":
        model_root = os.path.join(models_root, "gptj-6b-mlperf-inf")
        input_dataset = os.path.join(
            models_root, "datasets", "common",
            "cnn_dailymail_eval.gptManagerBenchmark.json")
        reference_dataset = os.path.join(models_root, "datasets", "common",
                                         "cnn_dailymail_eval.json")
        calib_dataset = os.path.join(models_root, "datasets", "common",
                                     "mlperf_llama2_openorca_calibration_1k")

    assert os.path.exists(model_root)
    assert os.path.exists(input_dataset)
    assert os.path.exists(reference_dataset)

    quantized_model_path = step_quantize(system_config.tp_size, llm_venv,
                                         llm_root, model_root, model,
                                         calib_dataset)
    step_engine_build(quantized_model_path, system_config, engine_dir, llm_venv)

    responses_file = step_run_llm(system_config, engine_dir, input_dataset,
                                  llm_venv, llm_root)
    step_check_accuracy(responses_file, reference_dataset, model_root, llm_venv,
                        llm_root)
