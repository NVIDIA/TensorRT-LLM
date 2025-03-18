# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import os
import platform
import re
from difflib import SequenceMatcher
from pathlib import Path

from packaging import version

from .trt_test_alternative import check_call, check_output, exists, is_windows


def venv_check_call(venv, cmd, env=None):

    def _war_check_call(*args, **kwargs):
        kwargs["cwd"] = venv.get_working_directory()
        return check_call(*args, **kwargs)

    venv.run_cmd(cmd, caller=_war_check_call, env=env)


def venv_check_output(venv, cmd):

    def _war_check_output(*args, **kwargs):
        kwargs["cwd"] = venv.get_working_directory()
        output = check_output(*args, **kwargs)
        return output

    return venv.run_cmd(cmd, caller=_war_check_output)


def venv_mpi_check_call(venv, mpi_cmd, python_cmd):
    """
    This function WAR check_call() to run python_cmd with mpi.
    If mpi_cmd = ["mpirun", "-n", "2"] and python_cmd = ["run.py"], the command will be:

    "mpirun -n 2 <venv python> run.py"

    """

    def _war_check_call(*args, **kwargs):
        assert len(args) == 1, "bad args"
        arg_list, = args
        merged_cmd = copy.deepcopy(mpi_cmd)
        merged_cmd.extend(arg_list)
        kwargs["cwd"] = venv.get_working_directory()
        return check_call(merged_cmd, **kwargs)

    venv.run_cmd(python_cmd, caller=_war_check_call)


def venv_mpi_check_output(venv, mpi_cmd, python_cmd, env=None):
    """
    This function WAR check_output() to run python_cmd with mpi.
    If mpi_cmd = ["mpirun", "-n", "2"] and python_cmd = ["run.py"], the command will be:

    "mpirun -n 2 <venv python> run.py"

    """

    def _war_check_output(*args, **kwargs):
        assert len(args) == 1, "bad args"
        arg_list, = args
        merged_cmd = copy.deepcopy(mpi_cmd)
        merged_cmd.extend(arg_list)
        kwargs["cwd"] = venv.get_working_directory()
        return check_output(merged_cmd, **kwargs)

    return venv.run_cmd(python_cmd, caller=_war_check_output, env=env)


def parse_mpi_cmd(cmd):
    if platform.system() == "Windows":
        # Simply fetch necessary args from Linux cmd then fill Windows cmd because:
        # 1. We use Microsoft MPI on Windows, while Open-MPI on Linux. Args are not compatible.
        # 2. Multi-GPU is actually not supported on Windows for now.
        flags = ("-n", "-np")
        # append None if not found
        indices = [idx for idx in range(len(cmd)) if cmd[idx] in flags] + [
            None,
        ]
        index = indices[0]
        return ["mpiexec", cmd[index], cmd[index + 1]] if index else cmd
    else:
        return cmd


class PluginOptions:

    def __init__(self,
                 gpt_attention: str = None,
                 bert_attention: str = None,
                 gemm: str = None,
                 layernorm: str = None):
        self.gpt_attention = gpt_attention
        self.bert_attention = bert_attention
        self.gemm = gemm

    def to_legacy_args(self):
        args = []
        if self.gpt_attention is not None:
            args.extend(["--use_gpt_attention_plugin", self.gpt_attention])
        if self.bert_attention is not None:
            args.extend(["--use_bert_attention_plugin", self.bert_attention])
        if self.gemm is not None:
            args.extend(["--use_gemm_plugin", self.gemm])
        return args

    def to_args(self):
        args = []
        if self.gpt_attention is not None:
            args.extend(["--gpt_attention_plugin", self.gpt_attention])
        else:
            args.extend(["--gpt_attention_plugin", "disable"])
        if self.bert_attention is not None:
            args.extend(["--bert_attention_plugin", self.bert_attention])
        else:
            args.extend(["--bert_attention_plugin", "disable"])
        if self.gemm is not None:
            args.extend(["--gemm_plugin", self.gemm])
        else:
            args.extend(["--gemm_plugin", "disable"])
        return args


def prune_checkpoint(llm_venv, checkpoint_dir):
    pruned_checkpoint_dir = checkpoint_dir + ".pruned"
    prune_cmd = [
        "trtllm-prune", f"--checkpoint_dir={checkpoint_dir}",
        f"--out_dir={pruned_checkpoint_dir}"
    ]

    check_call(" ".join(prune_cmd), shell=True, env=llm_venv._new_env)
    return pruned_checkpoint_dir


def refit_model(llm_venv, engine_dir, unpruned_model_dir):
    refit_engine_dir = f"{engine_dir}_refit_full"
    refit_cmd = [
        "trtllm-refit", f"--checkpoint_dir={unpruned_model_dir}",
        f"--engine_dir {engine_dir}", f"--output_dir {refit_engine_dir}"
    ]

    check_call(" ".join(refit_cmd), shell=True, env=llm_venv._new_env)
    return refit_engine_dir


def convert_weights(llm_venv,
                    example_root,
                    cmodel_dir,
                    model,
                    model_path,
                    quant_ckpt_path=None,
                    data_type="float16",
                    gpus=1,
                    tp_size=None,
                    pp_size=None,
                    model_type=None,
                    use_parallel_embedding=False,
                    embedding_sharding_dim=0,
                    load_by_shard=False,
                    int8_kv_cache=False,
                    use_weight_only=False,
                    workers=1,
                    processes=None,
                    smoothquant=0,
                    per_channel=False,
                    per_token=False,
                    fp8_kv_cache=False,
                    enable_fp8=False,
                    weight_only_precision=None,
                    per_group=False,
                    batch_size=8,
                    multimodal=False,
                    ckpt_type='hf',
                    load_model_on_cpu=False,
                    **kwargs):
    "Convert weights from HF transformers format to FT format"
    converted_model_path = os.path.join(cmodel_dir, model, data_type)
    script = "convert_checkpoint.py"

    tp_size = gpus if tp_size is None else tp_size
    pp_size = gpus // tp_size if pp_size is None else pp_size
    gpus = tp_size * pp_size
    model_dir = f'{converted_model_path}/{gpus}-gpu'

    # TODO: add other models command
    if "gpt2" in model:
        script = "convert_checkpoint.py"
        convert_cmd = [
            f"{example_root}/{script}", f"--output_dir={model_dir}",
            f"--dtype={data_type}", f"--tp_size={tp_size}",
            f"--pp_size={pp_size}"
        ]
        if "next" in model:
            convert_cmd.extend(["--nemo_ckpt_path", model_path])
        else:
            convert_cmd.extend(["--model_dir", model_path])
        if "smooth" in model:
            convert_cmd.extend(["--smoothquant", "0.5"])
        if "kv" in model and "int8" in model:
            convert_cmd.append("--int8_kv_cache")

    elif "t5" in model or "bart" in model or "ul2" in model or "wmt" in model or "nougat" in model or 'pix2struct' in model:
        assert model_type, "Encoder-Decoder models must specify model architecture type"
        script = "convert_checkpoint.py"
        convert_cmd = [
            f"{example_root}/{script}", "--model_dir", model_path,
            "--output_dir", converted_model_path, f"--model_type={model_type}",
            f"--tp_size={tp_size}", f"--pp_size={pp_size}",
            f"--dtype={data_type}"
        ]
        if "nougat" in model:
            convert_cmd.append("--nougat")

        model_dir = converted_model_path

    elif "opt" in model and model_type == "blip2":
        convert_cmd = [
            f"{example_root}/{script}",
            f"--model_dir={model_path}",
            f"--output_dir={model_dir}",
            f"--model_type={model_type}",
            f"--dtype={data_type}",
            f"--tp_size={tp_size}",
            f"--pp_size={pp_size}",
        ]

    elif "whisper" in model_path:
        script = "convert_checkpoint.py"
        convert_cmd = [
            f"{example_root}/{script}", "--model_dir", model_path,
            "--output_dir", converted_model_path
        ]
        model_dir = converted_model_path

    elif "mamba" in model:
        convert_cmd = [
            f"{example_root}/{script}", "--model_dir", model_path,
            "--output_dir", model_dir, f"--dtype={data_type}",
            f"--tp_size={tp_size}"
        ]

    elif "llama" in model or "llava" in model or "vila" in model:
        convert_cmd = [
            f"{example_root}/{script}", "--output_dir", model_dir,
            f"--dtype={data_type}", f"--tp_size={tp_size}",
            f"--pp_size={pp_size}"
        ]

        if 'meta-ckpt' in model:
            convert_cmd.extend(['--meta_ckpt_dir', model_path])
        else:
            convert_cmd.extend(['--model_dir', model_path])

        if 'code_llama_1gpu' in model:
            convert_cmd.extend(['--rotary_base=1000000'])
            convert_cmd.extend(['--vocab_size=32016'])
        elif 'code_llama' in model:
            convert_cmd.extend(['--rotary_base=1000000'])
            convert_cmd.extend(['--vocab_size=32000'])
        if 'int4_gptq' in model:
            convert_cmd.extend([
                "--use_weight_only", "--weight_only_precision=int4_gptq",
                f"--quant_ckpt_path={quant_ckpt_path}", "--per_group"
            ])
        if 'int8_gptq' in model:
            convert_cmd.extend([
                "--use_weight_only", "--weight_only_precision=int8_gptq",
                f"--quant_ckpt_path={quant_ckpt_path}", "--per_group",
                "--group_size=64"
            ])

        if 'awq' in model:
            convert_cmd.extend([
                "--use_weight_only", "--weight_only_precision=int4_awq",
                "--group_size=128"
            ])
        if 'hf_fp8' in model:
            convert_cmd.extend(["--use_fp8"])

    elif "draft_target_model" in model:
        if "gpt" in model_path:
            example_name = "gpt"
        elif "llama" in model_path:
            example_name = "llama"
        script = f"{example_root}/../{example_name}/convert_checkpoint.py"
        convert_cmd = [
            f"{script}",
            "--model_dir",
            model_path,
            "--output_dir",
            model_dir,
            f"--dtype={data_type}",
        ]

    elif "prompt_lookup" in model:
        if "gpt" in model_path:
            example_name = "gpt"
        elif "llama" in model_path:
            example_name = "llama"
        script = f"{example_root}/../{example_name}/convert_checkpoint.py"
        convert_cmd = [
            f"{script}",
            "--model_dir",
            model_path,
            "--output_dir",
            model_dir,
            f"--dtype={data_type}",
        ]

    elif "medusa" in model:
        convert_cmd = [
            f"{example_root}/{script}", "--model_dir", model_path[0],
            "--medusa_model_dir", model_path[1], "--output_dir", model_dir,
            f"--dtype={data_type}", f"--tp_size={tp_size}",
            f"--pp_size={pp_size}", "--num_medusa_heads=4"
        ]
    elif "redrafter" in model:
        redrafter_num_beams = kwargs.pop("redrafter_num_beams")
        redrafter_draft_len_per_beam = kwargs.pop(
            "redrafter_draft_len_per_beam")
        convert_cmd = [
            f"{example_root}/{script}", "--model_dir", model_path[0],
            "--drafter_model_dir", model_path[1], "--output_dir", model_dir,
            f"--dtype={data_type}", f"--tp_size={tp_size}",
            f"--redrafter_num_beams={redrafter_num_beams}",
            f"--redrafter_draft_len_per_beam={redrafter_draft_len_per_beam}"
        ]
    elif "eagle" in model:
        if len(model_path) == 2:
            # Test the checkpoint released from HF, which requires two separate weights,
            # one for the base model and one for the EagleNets.
            convert_cmd = [
                f"{example_root}/{script}", "--model_dir", model_path[0],
                "--eagle_model_dir", model_path[1], "--output_dir", model_dir,
                f"--dtype={data_type}", f"--tp_size={tp_size}",
                f"--pp_size={pp_size}", "--num_eagle_layers=4",
                "--max_draft_len=63", "--max_non_leaves_per_layer=10"
            ]
        else:
            # Test the checkpoint released from ModelOpt, which only requires one weight,
            # which includes both the base model and EagleNets, and is an FP8 datatype.
            convert_cmd = [
                f"{example_root}/{script}", "--model_dir", model_path,
                "--output_dir", model_dir, f"--dtype={data_type}",
                f"--tp_size={tp_size}", f"--pp_size={pp_size}",
                "--num_eagle_layers=4", "--max_draft_len=63",
                "--max_non_leaves_per_layer=10"
            ]
    elif "recurrentgemma" in model:
        convert_cmd = [
            f"{example_root}/{script}", "--model_dir", model_path,
            "--output_dir", model_dir, f"--dtype={data_type}",
            f"--world_size={tp_size}", f"--ckpt_type={ckpt_type}"
        ]
    elif "cogvlm" in model:
        convert_cmd = [
            f"{example_root}/{script}", "--model_dir", model_path,
            "--output_dir", model_dir, f"--dtype={data_type}",
            f"--tp_size={tp_size}", f"--pp_size={pp_size}",
            "--use_prompt_tuning"
        ]
    elif "fuyu" in model or "kosmos" in model:
        gpt_variant = "kosmos-2" if "kosmos" in model else "persimmon"
        convert_cmd = [
            f"{example_root}/{script}", "--model_dir", model_path,
            "--output_dir", model_dir, "--dtype", data_type, "--gpt_variant",
            gpt_variant
        ]
    elif "neva-22b" in model:
        convert_cmd = [
            f"{example_root}/{script}", "--nemo_ckpt_path", model_path,
            "--output_dir", model_dir, "--dtype", data_type,
            "--nemo_rename_key", "model:model.language_model",
            "attention.linear_qkv.layer_norm_bias:input_layernorm.bias",
            "attention.linear_qkv.layer_norm_weight:input_layernorm.weight",
            "mlp.linear_fc1.layer_norm_bias:post_attention_layernorm.bias",
            "mlp.linear_fc1.layer_norm_weight:post_attention_layernorm.weight",
            "linear_qkv:query_key_value", "linear_fc1:dense_h_to_4h",
            "linear_fc2:dense_4h_to_h", "linear_proj:dense", "decoder:encoder"
        ]
    elif "video-neva" in model:

        nemotron_root = os.path.join(example_root, "../", "nemotron")

        if llm_venv:
            # Install Python requirements for nemotron
            llm_venv.run_cmd([
                "-m", "pip", "install", "-r",
                os.path.join(nemotron_root, "requirements.txt")
            ])

        qformat = 'full_prec'
        model_name = 'nemotron-video-neva'
        converted_model_path = os.path.join(cmodel_dir, model_name, qformat)
        model_dir = f'{converted_model_path}/{gpus}-gpu'
        # Overwrite the model_path with the nemotron model path
        model_path = os.path.join(os.path.dirname(os.path.dirname(model_path)),
                                  'nemotron', 'Nemotron-4-15B-SteerLM.nemo')
        convert_cmd = [
            f"{example_root}/../quantization/quantize.py",
            f"--nemo_ckpt_path={model_path}",
            "--batch_size=64",
            f"--dtype={data_type}",
            f"--qformat={qformat}",
            f"--output_dir={model_dir}",
        ]
    elif "dit-xl" in model.lower():
        convert_cmd = [
            f"{example_root}/{script}",
            f"--timm_ckpt={model_path}",
            f"--output_dir={model_dir}",
            f"--dtype={data_type}",
            f"--tp_size={tp_size}",
            f"--pp_size={pp_size}",
        ]
        if kwargs.get("enable_fp8_linear") is not None:
            convert_cmd.append("--fp8_linear")
    elif "stdit" in model.lower():
        convert_cmd = [
            f"{example_root}/{script}",
            f"--timm_ckpt={model_path}/model.safetensors",
            f"--output_dir={model_dir}",
            f"--dtype={data_type}",
            f"--tp_size={tp_size}",
            f"--pp_size={pp_size}",
        ]
    elif "bert" in model.lower():
        convert_cmd = [
            f"{example_root}/{script}",
            f"--model={model}",
            f"--model_dir={model_path}",
            f"--output_dir={model_dir}",
            f"--dtype={data_type}",
            f"--tp_size={tp_size}",
        ]
    elif "granite" in model.lower():
        convert_cmd = [
            f"{example_root}/{script}",
            f"--model_dir={model_path}",
            f"--output_dir={model_dir}",
            f"--dtype={data_type}",
            f"--tp_size={tp_size}",
        ]
    elif "stable-diffusion-3.5" in model.lower():
        convert_cmd = [
            f"{example_root}/{script}",
            f"--model_path={model_path}",
            f"--output_dir={model_dir}",
            f"--tp_size={tp_size}",
        ]
    else:
        convert_cmd = [
            f"{example_root}/{script}",
            f"--model_dir={model_path}",
            f"--output_dir={model_dir}",
            f"--dtype={data_type}",
            f"--tp_size={tp_size}",
            f"--pp_size={pp_size}",
        ]

    if use_parallel_embedding:
        convert_cmd.append("--use_parallel_embedding")
        convert_cmd.append(f"--embedding_sharding_dim={embedding_sharding_dim}")
    if load_by_shard:
        convert_cmd.extend(["--load_by_shard"])
    if load_model_on_cpu:
        convert_cmd.extend(["--load_model_on_cpu"])
    if workers > 1:
        convert_cmd.extend([f"--workers={workers}"])
    if int8_kv_cache:
        convert_cmd.append("--int8_kv_cache")
    if use_weight_only:
        convert_cmd.append("--use_weight_only")
    if weight_only_precision:
        convert_cmd.append(f"--weight_only_precision={weight_only_precision}")
    if processes is not None:
        convert_cmd.append(f"--processes={processes}")
    if smoothquant > 0:
        convert_cmd.append(f"--smoothquant={smoothquant}")
    if per_channel:
        convert_cmd.append("--per_channel")
    if per_token:
        convert_cmd.append("--per_token")
    if enable_fp8:
        convert_cmd.append('--enable_fp8')
    if fp8_kv_cache:
        convert_cmd.append('--fp8_kv_cache')
    if quant_ckpt_path:
        convert_cmd.append(f"--quant_ckpt_path={quant_ckpt_path}")
    if per_group:
        convert_cmd.append("--per_group")

    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                convert_cmd.append(f"--{key}")
        else:
            convert_cmd.extend([f"--{key}={value}"])

    if llm_venv:
        venv_check_call(llm_venv, convert_cmd)
        return model_dir
    else:
        return convert_cmd, model_dir


def similarity_score(a, b):
    "similar compare a and b "
    return SequenceMatcher(None, a, b).ratio()


def similar(a, b, threshold=0.8):
    "similar compare a and b "
    return similarity_score(a, b) >= threshold


def generate_build_cmd(example_root, *args, **kwargs):
    "generate build command"
    build_cmd = [f"{example_root}/build.py"]
    dtype = kwargs.get("dtype")

    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                if ('plugin' in key) and (not "lookup" in key):
                    build_cmd.extend([f"--{key}", dtype])
                else:
                    build_cmd.append(f"--{key}")
        else:
            build_cmd.extend([f"--{key}", f"{value}"])

    for arg in args:
        build_cmd.append(f"--{arg}")

    return build_cmd


def generate_summary_cmd(example_root, *args, **kwargs):
    "generate summary command"
    summary_cmd = [
        f"{example_root}/../summarize.py", "--test_trt_llm", "--check_accuracy"
    ]

    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                summary_cmd.append(f"--{key}")
        else:
            summary_cmd.extend([f"--{key}", f"{value}"])

    for arg in args:
        summary_cmd.append(f"--{arg}")

    return summary_cmd


def generate_mmlu_cmd(example_root, *args, **kwargs):
    "generate mmlu command"
    mmlu_cmd = [
        f"{example_root}/../mmlu.py", "--test_trt_llm", "--check_accuracy"
    ]

    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                mmlu_cmd.append(f"--{key}")
        else:
            mmlu_cmd.extend([f"--{key}", f"{value}"])

    for arg in args:
        mmlu_cmd.append(f"--{arg}")

    return mmlu_cmd


def generate_deterministic_cmd(example_root, *args, **kwargs):
    "generate deterministic command"
    deterministic_cmd = [
        f"{example_root}/mixtral_deterministic.py",
        "--check_deterministic_accuracy"
    ]

    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                deterministic_cmd.extend(f"--{key}")
        else:
            deterministic_cmd.extend([f"--{key}", f"{value}"])

    for arg in args:
        deterministic_cmd.append(f"--{arg}")

    return deterministic_cmd


def quantize_data(llm_venv,
                  example_root,
                  model_dir,
                  dtype,
                  quantize_dir,
                  qformat="full_prec",
                  tp_size=1,
                  pp_size=1,
                  cp_size=1,
                  calib_size=512,
                  kv_cache_dtype=None,
                  **kwargs):
    "quanize data and return data dir"
    model_name = os.path.basename(model_dir)
    output_dir = os.path.join(quantize_dir, model_name, dtype, qformat,
                              f"tp{tp_size}pp{pp_size}")
    if kv_cache_dtype:
        output_dir = os.path.join(output_dir, kv_cache_dtype)
    else:
        output_dir = os.path.join(output_dir, "no_kv_cache")

    quantize_cmd = [
        f"{example_root}/../quantization/quantize.py",
        f"--model_dir={model_dir}",
        f"--dtype={dtype}",
        f"--qformat={qformat}",
        f"--output_dir={output_dir}",
        f"--tp_size={tp_size}",
        f"--pp_size={pp_size}",
        f"--cp_size={cp_size}",
        f"--calib_size={calib_size}",
    ]

    if kv_cache_dtype:
        quantize_cmd.append(f"--kv_cache_dtype={kv_cache_dtype}")

    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                quantize_cmd.append(f"--{key}")
        else:
            quantize_cmd.extend([f"--{key}", f"{value}"])

    if llm_venv:
        if not exists(output_dir):
            venv_check_call(llm_venv, quantize_cmd)
        return output_dir
    else:
        return quantize_cmd, output_dir


def find_tensorrt(ld_library_path):
    MAX_SEARCH_HEIGHT = 10
    ld_library_path = ld_library_path.split(os.pathsep)
    for trt_lib_dir in ld_library_path:
        trt_lib_dir = Path(trt_lib_dir)
        trt_nvinfer_lib = trt_lib_dir / "libnvinfer.so"
        if trt_nvinfer_lib.exists():
            trt_root_dir = trt_lib_dir
            for i in range(MAX_SEARCH_HEIGHT):
                trt_root_dir = trt_root_dir.parent
                trt_include_dir = trt_root_dir / "include"
                trt_nvinfer_header = trt_include_dir / "NvInfer.h"
                if trt_nvinfer_header.exists():
                    return str(trt_include_dir), str(trt_lib_dir)
    return None, None


def get_trt_llm_lib_dir(venv):
    output = venv.run_raw(
        "import tensorrt_llm; print(f'{tensorrt_llm.__path__[0]}/libs')",
        caller=check_output).strip()

    if "TensorRT-LLM version: " in output:
        output = output.split('\n')[-1]

    return output.strip()


def trt_gte(venv, major: int, minor: int = 0):
    """
    Check if TRT version is greater than or equal to major.minor
    """
    ver = venv.run_output("import tensorrt;print(tensorrt.__version__)")
    trt_ver = version.parse(ver)
    return trt_ver.major >= major and trt_ver.minor >= minor


def parse_output(text):
    "parse output"
    results = []
    text_lists = re.split(r"Input \[Text \d\]:", text)
    for item in text_lists:
        item = item.replace(os.linesep, "")
        while True:
            match = re.search(
                r"(Output \[Text \d+ Beam \d+\]: \"(.*?)\")(Output|Input|$)",
                item, re.MULTILINE)
            if match is None:
                break
            _, end = match.span(1)
            results.append(match.group(2))
            item = item[end:]

    return results


def run_and_check(llm_venv, run_cmd, valid_outputs, streaming=False):
    print("Running inference...")
    output = venv_check_output(llm_venv, run_cmd)

    if not streaming:
        output = parse_output(output)[0]
        assert any([
            similar(output, expect, threshold=0.95) for expect in valid_outputs
        ]), f"output is: {output}"
    else:
        # Fetch all outputs and expect a monotonically increasing similarity
        similarities = []
        for suboutput in parse_output(output):
            similarities.append(
                max([
                    similarity_score(suboutput, expect)
                    for expect in valid_outputs
                ]))
        assert (
            all(x <= y for x, y in zip(similarities, similarities[1:]))
        ), f"streaming outputs must have a monotonically increasing similarity score. similarities: {similarities}"
        output = parse_output(output)[-1]
        assert any([
            similar(output, expect, threshold=0.95) for expect in valid_outputs
        ]), f"output is: {output}"


def get_cpp_benchmark(cpp_benchmark_name, llm_root):
    suffix = ".exe" if is_windows() else ""
    cpp_benchmark_name += suffix
    # In CI/CD, we copy the cpp binary into the same folder as cpp to avoid package sanity
    ci_path = os.path.join(os.path.dirname(os.path.realpath(llm_root)),
                           "benchmarks", "cpp", cpp_benchmark_name)
    if os.path.exists(ci_path):
        return ci_path
    # In QA, we keep the benchmark build at its original location
    qa_path = os.path.join(llm_root, "cpp", "build", "benchmarks",
                           cpp_benchmark_name)
    if os.path.exists(qa_path):
        return qa_path
    raise Exception(
        f"Cannot find cpp benchmark binary in either {ci_path} or {qa_path}. Did you forget --benchmark in building TRT-LLM?"
    )


def generate_dummy_loras(
        hf_model_dir,
        lora_output_dir,
        num_loras=1,
        lora_rank=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        zero_weights=False):

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    print("Creating pseudo LoRAs...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    lora_config = LoraConfig(r=lora_rank,
                             target_modules=target_modules,
                             bias="none",
                             task_type="CAUSAL_LM")
    lora_output_paths = []
    for lora_idx in range(num_loras):
        lora_model = get_peft_model(model, lora_config)
        if zero_weights:
            for param in lora_model.parameters():
                param.data.zero_()
        pseudo_lora_dir = f"{lora_output_dir}/pseudo_lora_{lora_idx}"
        lora_model.save_pretrained(pseudo_lora_dir)
        lora_output_paths.append(pseudo_lora_dir)
    return lora_output_paths


def test_multi_lora_support(
    hf_model_dir,
    tllm_ckpt_dir,
    engine_dir,
    llm_venv,
    example_root,
    num_loras=2,
    lora_rank=8,
    target_hf_modules=["q_proj", "k_proj", "v_proj"],
    target_trtllm_modules=["attn_q", "attn_k", "attn_v"],
    zero_lora_weights=True,
    use_code_prompts=False,
):
    print("Creating dummy LoRAs...")
    lora_paths = generate_dummy_loras(
        hf_model_dir=hf_model_dir,
        lora_output_dir=llm_venv.get_working_directory(),
        num_loras=num_loras,
        lora_rank=lora_rank,
        target_modules=target_hf_modules,
        zero_weights=zero_lora_weights)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={tllm_ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
        "--gemm_plugin=auto",
        "--lora_plugin=auto",
        "--max_batch_size=8",
        "--max_input_len=512",
        "--max_seq_len=562",
        "--lora_dir",
        f"{lora_paths[0]}",
        f"{lora_paths[1]}",
        "--max_lora_rank=8",
        "--lora_target_modules",
        *target_trtllm_modules,
        "--max_beam_width=1",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    if use_code_prompts:
        input_prompts = [
            "Write a function that outputs the fibonacci sequence.",
            "Convert the following C++ code to Python:  x = 0;x++;",
            "Find the largest prime factor of 42.",
            "write a unit test for this function: $(cat fib.py)",
            "# A simple python function to remove whitespace from a string:",
            "How to load CodeLlama from HuggingFace?",
        ]
    else:
        input_prompts = [
            "Hey how are you doing today?",
            "How is the weather in Seattle, WA?",
            "Is it ok to fill diesel in a petrol car?",
            "Can you check the top 5 trending songs on spotify?",
            "What is the capital of France?",
            "How to load CodeLlama from HuggingFace?",
        ]

    print("Run inference with C++ runtime with pybind...")
    run_cmd = [
        f"{example_root}/../run.py",
        f"--tokenizer_dir={hf_model_dir}",
        f"--engine_dir={engine_dir}",
        "--input_text",
        *input_prompts,
        "--lora_task_uids",
        "-1",
        "0",
        "1",
        "-1",
        "0",
        "1",
        "--top_p=0.5",
        "--top_k=0",
        "--random_seed=0",
        "--max_output_len=30",
    ]
    venv_check_call(llm_venv, run_cmd)


def generate_dummy_medusa(hf_model_dir,
                          save_dir,
                          mode='medusa',
                          num_heads=4,
                          num_layers=1):

    import os

    import modelopt.torch.opt as mto
    import modelopt.torch.speculative as mtsp
    import transformers
    from modelopt.torch.export import export_hf_checkpoint

    # Create the base model.
    model = transformers.AutoModelForCausalLM.from_pretrained(
        hf_model_dir, trust_remote_code=True)

    if mode == "medusa":
        config = {
            "medusa_num_heads": num_heads,
            "medusa_num_layers": num_layers,
        }
    elif mode == "eagle":
        config = {"eagle_num_layers": num_layers}
    else:
        raise NotImplementedError(f"Unknown mode {mode}.")
    mtsp.convert(model, [(mode, config)])

    tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create a dummy trainer.
    trainer = transformers.Trainer(model=model, tokenizer=tokenizer)
    trainer._move_model_to_device(model, 'cuda')

    # Enable HF checkpointing so that the saved model will contain the speculative decoding module.
    mto.enable_huggingface_checkpointing()
    trainer.save_model(os.path.join(save_dir, 'native'))
    tokenizer.save_pretrained(os.path.join(save_dir, 'native'))

    import modelopt.torch.quantization as mtq
    import modelopt.torch.utils.dataset_utils as dataset_utils

    mto.enable_huggingface_checkpointing()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        os.path.join(save_dir, 'native'))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join(save_dir, 'native'))

    calib_dataloader = dataset_utils.get_dataset_dataloader(
        dataset_name="cnn_dailymail",
        tokenizer=tokenizer,
        batch_size=1,
        num_samples=1,
        device=model.device,
        include_labels=False,
    )

    quant_cfg = getattr(mtq, "FP8_DEFAULT_CFG")
    quant_cfg["quant_cfg"]["*output_quantizer"] = {
        "num_bits": (4, 3),
        "axis": None,
        "enable": True,
    }

    calibrate_loop = dataset_utils.create_forward_loop(
        calib_dataloader, dataloader=calib_dataloader)
    model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    mtq.print_quant_summary(model)

    export_hf_checkpoint(model,
                         dtype=model.config.torch_dtype,
                         export_dir=os.path.join(save_dir, 'fp8'))
