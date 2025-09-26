# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

import pytest
import torch
from defs.common import convert_weights, venv_check_call, venv_mpi_check_call
from defs.conftest import (get_device_memory, get_sm_version,
                           skip_post_blackwell, skip_pre_ada)
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


@pytest.fixture(scope="module")
def multimodal_example_root(llm_root):
    "Get multimodal example root"
    example_root = os.path.join(llm_root, "examples", "models", "core",
                                "multimodal")

    return example_root


@pytest.fixture(scope="function")
def recover_transformers(llm_venv, llm_root):
    "Recover transformers"

    yield

    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(llm_root, "requirements.txt")
    ])


def _call_run_cmd(llm_venv, llm_root, cmd, world_size):
    if world_size == 1:
        venv_check_call(llm_venv, cmd)
    else:
        venv_mpi_check_call(
            llm_venv, ["mpirun", "-n",
                       str(world_size), "--allow-run-as-root"], cmd)


dataset_path_mapping = {
    'cnn_dailymail': 'cnn_dailymail',
    'scienceqa': 'derek-thomas___science_qa',
}


def _test_llm_multimodal_general(llm_venv,
                                 llm_root,
                                 llm_datasets_root,
                                 cmodel_dir,
                                 engine_dir,
                                 batch_size,
                                 data_type,
                                 tp_size,
                                 pp_size,
                                 multimodal_example_root,
                                 multimodal_model_root,
                                 recover_transformers,
                                 calibration_dataset=None,
                                 qformat=None,
                                 kv_cache_dtype=None,
                                 cpp_e2e=False,
                                 num_beams=1):

    # Empty the torch CUDA cache before each multimodal test to reduce risk of OOM errors.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    world_size = tp_size * pp_size
    print("Locate model checkpoints in test storage...")
    tllm_model_name, model_ckpt_path = multimodal_model_root

    if "neva-22b" in tllm_model_name and get_device_memory() < 80000:
        pytest.skip("GPU memory is insufficient.")
    if "Mistral-Small" in tllm_model_name and get_device_memory() < 80000:
        pytest.skip("GPU memory is insufficient.")

    print("Converting huggingface model into binary format...")
    # ckpt from llm_models/<model_name> --> cmodels/<model_name>/<dtype>
    model_name = tllm_model_name
    model_name = "pix2struct" if model_name == "deplot" else model_name
    opt_example_root = multimodal_example_root + "/../models/contrib/opt"
    enc_dec_example_root = multimodal_example_root + "/../enc_dec"
    llama_example_root = multimodal_example_root + "/../llama"
    cogvlm_example_root = multimodal_example_root + "/../cogvlm"
    gpt_example_root = multimodal_example_root + "/../gpt"
    nemotron_example_root = multimodal_example_root + "/../nemotron"
    phi_example_root = multimodal_example_root + "/../phi"
    mllama_example_root = multimodal_example_root + "/../mllama"
    qwen_example_root = multimodal_example_root + "/../qwen"
    internlm_example_root = multimodal_example_root + "/../internlm2"

    opt_model = "opt" in model_name
    nougat_model = "nougat" in model_name
    gpt_model = "fuyu" in model_name or "neva-22b" in model_name or "kosmos" in model_name
    pix2struct_model = "pix2struct" in model_name
    enc_dec_model = "t5" in model_name or nougat_model or pix2struct_model
    llava_model = "llava" in model_name
    llava_next_model = "llava-v1.6" in model_name
    llava_next_vision_trtllm_engine_model = "vision-trtllm" in model_name and llava_next_model
    llava_onevision_model = "llava-onevision" in model_name
    llava_onevision_video_model = "video" in model_name and llava_onevision_model
    vila_model = "VILA" in model_name
    cogvlm_model = "cogvlm" in model_name
    nemotron_model = "video-neva" in model_name
    phi3_model = "phi-3" in model_name.lower()
    phi4_model = "phi-4" in model_name.lower()
    mllama_model = 'Llama-3.2' in model_name
    qwen2_vl_model = 'Qwen2-VL' in model_name
    internlm_model = 'internlm-xcomposer2' in model_name
    mistral_model = 'Mistral-Small' in model_name
    if enc_dec_model:
        builder_root = enc_dec_example_root
        if nougat_model:
            model_type = "bart"
        if pix2struct_model:
            model_type = "pix2struct"
        if "t5" in model_name:
            model_type = "blip2"
    elif gpt_model:
        builder_root, model_type = gpt_example_root, "gpt"
    elif llava_onevision_model:
        builder_root, model_type = qwen_example_root, "qwen"
    elif qwen2_vl_model:
        builder_root, model_type = qwen_example_root, "qwen"
    elif internlm_model:
        builder_root, model_type = internlm_example_root, "internlm"
    elif llava_model or vila_model:
        builder_root, model_type = llama_example_root, "llama"
    elif mistral_model:
        builder_root, model_type = llama_example_root, "llama"
    elif cogvlm_model:
        builder_root, model_type = cogvlm_example_root, "cogvlm"
    elif nemotron_model:
        builder_root, model_type = nemotron_example_root, "nemotron"
    elif phi3_model:
        model_name = model_name.split('/')[-1]  # Remove HF directory name
        builder_root, model_type = phi_example_root, "phi-3-vision"
    elif phi4_model:
        builder_root, model_type = phi_example_root, "phi-4-multimodal"
    elif opt_model:
        builder_root, model_type = opt_example_root, "blip2"
    elif mllama_model:
        builder_root, model_type = mllama_example_root, "mllama"

    use_weight_only = (not enc_dec_model) and (data_type in [
        'int4_weight_only', 'int8_weight_only'
    ])
    weight_only_precision = data_type.split('_')[0] if use_weight_only else None
    if use_weight_only: data_type = 'float16'

    if vila_model:
        print(
            "VILA model has dependencies on certain HuggingFace version. Need to pip install until this limitation is removed."
        )
        check_call(
            f"pip install -r {multimodal_example_root}/requirements-vila.txt",
            shell=True,
            env=llm_venv._new_env)
    elif llava_onevision_model:
        check_call(
            f"pip install -r {multimodal_example_root}/requirements-llava_onevision.txt",
            shell=True,
            env=llm_venv._new_env)
    elif qwen2_vl_model:
        check_call(
            f"pip install -r {multimodal_example_root}/requirements-qwen2vl.txt",
            shell=True,
            env=llm_venv._new_env)
    elif internlm_model:
        check_call(
            f"pip install -r {multimodal_example_root}/requirements-internlm-xcomposer2.txt",
            shell=True,
            env=llm_venv._new_env)
    elif mllama_model:
        check_call(f"pip install -r {mllama_example_root}/requirements.txt",
                   shell=True,
                   env=llm_venv._new_env)
    if qformat == 'fp8':
        convert_cmd = [
            f"{multimodal_example_root}/../../../quantization/quantize.py",
            f"--model_dir={model_ckpt_path}",
            f"--calib_dataset={llm_datasets_root}/{dataset_path_mapping[calibration_dataset]}",
            f"--dtype={data_type}",
            f"--qformat={qformat}",
            f"--kv_cache_dtype={kv_cache_dtype}",
            f"--output_dir={cmodel_dir}",
            f"--calib_size=16",
        ]
        venv_check_call(llm_venv, convert_cmd)
        converted_weight_dir = cmodel_dir
    else:
        converted_weight_dir = convert_weights(
            llm_venv,
            builder_root,
            cmodel_dir,
            model_name,
            model_ckpt_path,
            data_type=data_type,
            gpus=tp_size,
            model_type=model_type,
            use_weight_only=use_weight_only,
            weight_only_precision=weight_only_precision,
            tp_size=tp_size,
            pp_size=pp_size,
            batch_size=batch_size,
            multimodal=True)

    print("Build LLM engines...")
    model_name = model_name.split('/')[-1]  # Remove HF directory name
    llm_engine_dir = f"{engine_dir}/{model_name}/{world_size}-gpu"
    if "opt" in model_name or llava_model or vila_model or gpt_model or nemotron_model or phi3_model or phi4_model or qwen2_vl_model or mistral_model:
        max_input_len_text = 1024
        max_output_len = 200
        if llava_next_model:
            multimodal_len = 4096
        elif llava_onevision_model:
            multimodal_len = 7300
        elif llava_model:
            multimodal_len = 576
        elif vila_model:
            multimodal_len = 196
        elif phi3_model:
            multimodal_len = 5120
        elif phi4_model:
            multimodal_len = 5120
        elif mistral_model:
            multimodal_len = 5120
        elif "fuyu" in model_name:
            multimodal_len = 2640
        elif "neva-22b" in model_name:
            multimodal_len = 729
        elif "video-neva" in model_name:
            multimodal_len = 3072
        elif "kosmos" in model_name:
            multimodal_len = 64
        elif "Qwen2-VL" in model_name:
            multimodal_len = 3552
        else:
            multimodal_len = 32
        max_input_len = max_input_len_text + batch_size * multimodal_len
        max_seq_len = max_input_len + max_output_len
        build_cmd = [
            "trtllm-build",
            f"--checkpoint_dir={converted_weight_dir}",
            f"--output_dir={llm_engine_dir}/llm",
            f"--gpt_attention_plugin {data_type}",
            f"--gemm_plugin={data_type}",
            f"--max_batch_size={batch_size}",
            f"--max_multimodal_len={batch_size * multimodal_len}",
            f"--max_input_len={max_input_len}",
            f"--max_seq_len={max_seq_len}",
            f"--max_num_tokens={max_input_len}",
            f"--max_beam_width={num_beams}",
        ]

        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)
    elif internlm_model:
        max_input_len_text = 1536
        max_output_len = 200
        multimodal_len = 1225
        max_input_len = max_input_len_text + batch_size * multimodal_len
        max_seq_len = max_input_len + max_output_len

        max_lora_rank = 256
        lora_dir = "."
        build_cmd = [
            "trtllm-build",
            f"--checkpoint_dir={converted_weight_dir}",
            f"--output_dir={llm_engine_dir}",
            f"--gpt_attention_plugin {data_type}",
            f"--gemm_plugin={data_type}",
            f"--lora_plugin={data_type}",
            f"--lora_dir={lora_dir}",
            f"--max_lora_rank={max_lora_rank}",
            f"--max_batch_size={batch_size}",
            f"--max_multimodal_len={batch_size * multimodal_len}",
            f"--max_input_len={max_input_len}",
            f"--max_seq_len={max_seq_len}",
            f"--max_num_tokens={max_input_len}",
            f"--max_beam_width={num_beams}",
        ]
        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)
    elif enc_dec_model:
        components = ["decoder"] if nougat_model or pix2struct_model else [
            "encoder", "decoder"
        ]
        for component in components:
            build_cmd = [
                "trtllm-build",
                f"--checkpoint_dir={converted_weight_dir}/{component}",
                f"--output_dir={llm_engine_dir}/{data_type}/llm/{component}",
                "--paged_kv_cache=enable",
                "--moe_plugin=disable",
                f"--max_batch_size={batch_size}",
                "--max_seq_len=412",
                f"--gemm_plugin={data_type}",
                f"--bert_attention_plugin={data_type}",
                f"--gpt_attention_plugin={data_type}",
                "--remove_input_padding=enable",
                f"--max_beam_width={num_beams}",
            ]

            # for non-T5 models, FP16/BF16
            if model_type == "t5" or data_type == "float32":
                build_cmd.append("--context_fmha=disable")
            if "t5" in model_name:
                if component == "encoder":
                    build_cmd.append(f"--max_multimodal_len={32 * batch_size}")
                    build_cmd.append("--max_input_len=412")
                else:
                    build_cmd.append("--max_encoder_input_len=412")
                    build_cmd.append(f"--max_input_len=1")
            else:  # Nougat
                assert nougat_model or pix2struct_model
                if component == "encoder":
                    build_cmd.append(f"--max_multimodal_len={588 * batch_size}")

                # only decoder for nougat
                if nougat_model:
                    build_cmd.append(
                        f"--max_encoder_input_len={588 * batch_size}")
                else:
                    build_cmd.append(
                        f"--max_encoder_input_len={2048 * batch_size}")

                build_cmd.append(f"--max_input_len=1")

            check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)
    elif cogvlm_model:
        build_cmd = [
            "trtllm-build",
            f"--checkpoint_dir={converted_weight_dir}",
            f"--output_dir={llm_engine_dir}/llm",
            f"--gemm_plugin={data_type}",
            f"--gpt_attention_plugin={data_type}",
            f"--remove_input_padding=enable",
            f"--max_batch_size={batch_size}",
            f"--max_input_len=2048",
            f"--max_seq_len=2048",
            f"--paged_kv_cache=enable",
            f"--bert_attention_plugin=disable",
            f"--moe_plugin=disable",
            f"--max_multimodal_len=61440",
            f"--max_beam_width={num_beams}",
        ]
        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)
    elif mllama_model:
        # set max_encoder_input_len = 6404 for running both non-instruct model and instruct model
        build_cmd = [
            "trtllm-build",
            f"--checkpoint_dir={converted_weight_dir}",
            f"--output_dir={llm_engine_dir}/llm",
            f"--gemm_plugin={data_type}",
            f"--max_num_tokens=4096",
            f"--max_seq_len=2048",
            f"--max_batch_size={batch_size}",
            f"--max_encoder_input_len=6404",
            f"--max_beam_width={num_beams}",
        ]
        if kv_cache_dtype == 'fp8':
            build_cmd.extend([
                "--use_fp8_context_fmha=enable",
                "--use_paged_context_fmha=enable",
            ])
        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Build visual engines...")
    vision_model_type = model_name
    if 'llava' in model_name: vision_model_type = 'llava'
    if 'llava-v1.6' in model_name: vision_model_type = 'llava_next'
    elif llava_onevision_model: vision_model_type = 'llava_onevision'
    elif 'VILA' in model_name: vision_model_type = 'vila'
    elif nougat_model: vision_model_type = 'nougat'
    elif pix2struct_model: vision_model_type = 'pix2struct'
    elif 'cogvlm' in model_name: vision_model_type = 'cogvlm'
    elif 'fuyu' in model_name: vision_model_type = 'fuyu'
    elif 'neva-22b' in model_name: vision_model_type = 'neva'
    elif 'video-neva' in model_name: vision_model_type = 'video-neva'
    elif phi3_model: vision_model_type = "phi-3-vision"
    elif phi4_model: vision_model_type = "phi-4-multimodal"
    elif 'blip2' in model_name: vision_model_type = 'blip2'
    elif 'Llama-3.2' in model_name: vision_model_type = 'mllama'
    elif "Qwen2-VL" in model_name: vision_model_type = 'qwen2_vl'
    elif 'internlm' in model_name: vision_model_type = 'internlm-xcomposer2'
    elif 'Mistral-Small' in model_name: vision_model_type = 'pixtral'

    vit_batch_size = batch_size
    if vision_model_type == "llava_next":
        vit_batch_size = vit_batch_size * 5
    elif vision_model_type == 'llava_onevision':
        vit_batch_size = vit_batch_size * 32

    llm_engine_subdir = f"{data_type}" if enc_dec_model else ""
    # Phi4MM has both vision and audio. Engine build dumps to vision and audio dirs automatically by builder.
    component_dir = "vision" if vision_model_type != "phi-4-multimodal" else ""
    build_cmd = [
        f"{multimodal_example_root}/build_multimodal_engine.py",
        f"--output_dir={os.path.join(llm_engine_dir, llm_engine_subdir, component_dir)}",
        f"--model_type={vision_model_type}",
        f"--model_path={model_ckpt_path}",
        f"--max_batch_size={vit_batch_size}",
    ]
    if vision_model_type == "vila":
        vila_path = model_ckpt_path + "/../VILA"
        build_cmd.extend([f"--vila_path={vila_path}"])
    if llava_next_vision_trtllm_engine_model:
        script_root = f"{multimodal_example_root}/../vit"
        convert_cmd = [
            f"{script_root}/convert_checkpoint.py",
            f"--model_dir={model_ckpt_path}",
            f"--output_dir={os.path.join(cmodel_dir, model_name, data_type, component_dir)}",
            f"--dtype={data_type}",
            f"--vision_tp_size={tp_size}",
        ]
        venv_check_call(llm_venv, convert_cmd)

        build_cmd = [
            "trtllm-build",
            f"--checkpoint_dir={os.path.join(cmodel_dir, model_name, data_type, component_dir)}",
            f"--output_dir={os.path.join(llm_engine_dir, llm_engine_subdir, component_dir)}",
            f"--max_batch_size={vit_batch_size}",
            f"--remove_input_padding disable",
            f"--bert_attention_plugin disable",
        ]
        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)
    else:
        venv_check_call(llm_venv, build_cmd)

    if llava_next_vision_trtllm_engine_model:
        cp_cmd = [
            "cp",
            f"{os.path.join(cmodel_dir, model_name, data_type, 'vision', 'image_newlines.safetensors')}",
            f"{os.path.join(llm_engine_dir, llm_engine_subdir, 'vision')}",
        ]
        check_call(" ".join(cp_cmd), shell=True, env=llm_venv._new_env)

    print("Run inference...")
    hf_model_dir = model_ckpt_path + "/../vicuna-7b-v1.5" if cogvlm_model else model_ckpt_path
    hf_model_dir = converted_weight_dir if "neva" in model_name else hf_model_dir
    video_path = os.path.join(
        os.path.dirname(model_ckpt_path), "test_video",
        "video_test.mp4") if "video-neva" in model_name else ""
    run_cmd = [
        f"{multimodal_example_root}/run.py",
        f"--engine_dir={llm_engine_dir}/{llm_engine_subdir}",
        f"--hf_model_dir={hf_model_dir}", "--max_new_tokens=30",
        f"--batch_size={batch_size}", "--check_accuracy",
        "--enable_context_fmha_fp32_acc"
    ]
    if vision_model_type == 'phi-4-multimodal':
        audio_path = f"{model_ckpt_path}/examples/what_is_shown_in_this_image.wav"
        run_cmd.extend(["--audio_path", f"{audio_path}"])
    if vision_model_type in ['llava', 'vila'] and batch_size > 1:
        # batch inference test
        if vision_model_type == 'vila':
            input_text = [
                '"<image>\n Please elaborate what you see in the images?"'
            ] * batch_size
        else:
            input_text = ['"\\n Which city is this? Answer:"'] * batch_size
        run_cmd.append("--input_text")
        run_cmd.extend(input_text)
    if enc_dec_model:
        run_cmd.extend(["--cross_kv_cache_fraction", "0.5"])
    if vision_model_type == "neva" and not cpp_e2e:
        # randomly pick one to test the python runtime
        run_cmd.extend(["--session", "python"])
    if vision_model_type == "video-neva":
        run_cmd.extend(["--video_path", video_path])
    if llava_onevision_video_model:
        run_cmd.extend(["--video_path", 'llava-onevision-accuracy'])
    if phi3_model or phi4_model:
        run_cmd.extend(["--kv_cache_free_gpu_memory_fraction", "0.4"])
    if cpp_e2e:
        run_cmd.extend(["--session", "cpp"])
    if num_beams > 1:
        run_cmd.extend(["--num_beams", str(num_beams)])

    if mllama_model:
        if qformat is None:
            run_cmd_vision = run_cmd.copy()
            run_cmd_vision.extend([
                "--cross_kv_cache_fraction=0.5",  # mllama uses cross attention
                "--image_path",
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
                "--input_text",
                "If I had to write a haiku for this one"
            ])

            print("Run mllama vision test in with example image ...")
            _call_run_cmd(llm_venv, llm_root, run_cmd_vision, world_size)

            print("multimodal_example_root: ", multimodal_example_root)
            print("llm_root: ", llm_root)
            run_cmd_vision = run_cmd.copy()
            run_cmd_vision.extend([
                "--cross_kv_cache_fraction=0.5",  # mllama uses cross attention
                "--image_path",
                os.path.join(
                    llm_root,
                    "tests/integration/test_input_files/excel_table_test.jpg"),
                "--input_text",
                "What is the total income? Answer:"
            ])

            print("Run mllama vision test with random image ...")

            run_cmd_text = run_cmd.copy()
            run_cmd_text.extend([
                "--cross_kv_cache_fraction=0.5",  # mllama uses cross attention
                "--input_text",
                "The key to life is",
            ])
            print("Run mllama text test...")
            _call_run_cmd(llm_venv, llm_root, run_cmd_text, world_size)
    else:
        _call_run_cmd(llm_venv, llm_root, run_cmd, world_size)

    # Run evaluation test
    if batch_size == 1 and (data_type == "float16" or qformat == 'fp8'):
        print(f"prepare to run eval test")

        # for blip2-t5, ref: https://github.com/huggingface/transformers/issues/25491
        if "t5" in model_name:
            check_call("pip uninstall -y apex",
                       shell=True,
                       env=llm_venv._new_env)

        # Threshold are set based on the HF correctness for 20 iterations
        threshold_map = {
            'blip2-opt-2.7b': 35,
            'blip2-flan-t5-xl': 55,
            'llava-1.5-7b-hf': 65,
            'llava-v1.6-mistral-7b-hf': 65,
            'llava-onevision-qwen2-7b-ov-hf': 80,
            'VILA1.5-3b': 75,  # from local TRT-LLM run
            'fuyu-8b': 70,
            'kosmos-2': 60,
            'Phi-3-vision-128k-instruct': 75,
            'Phi-3.5-vision-instruct': 85,
            'Llama-3.2-11B-Vision': 60,  # The expected score is 62
            'Llama-3.2-11B-Vision-Instruct': 75,  # The expected score is 77
            'Qwen2-VL-7B-Instruct': 80,
        }

        if model_name not in threshold_map:
            print(f"Skip {model_name} evaluation test.")
            return

        # TODO: Delete these lines after resolving the issues
        # For llava - input tokens are not parsed correctly with '<image>\n'
        # For llava_next - correctness lower than HF, and needs lower transformer version built
        # For Phi-3 - correctness lower than HF
        # For qwen_vl - runtime issue with eval.py -- need to unify prompt generation logics
        # For internvl - not added to the test
        if llava_model or llava_next_model or phi3_model or qwen2_vl_model:
            return

        eval_task = "lmms-lab/ai2d" if mllama_model else "lmms-lab/VQAv2"

        eval_cmd = [
            f"{multimodal_example_root}/eval.py",
            f"--model_type={vision_model_type}",
            f"--engine_dir={llm_engine_dir}/{llm_engine_subdir}",
            f"--hf_model_dir={hf_model_dir}", "--enable_context_fmha_fp32_acc",
            "--test_trtllm",
            f"--accuracy_threshold={threshold_map[model_name]}",
            f"--eval_task={eval_task}"
        ]

        if mllama_model:
            eval_cmd.extend([
                f"--dataset_dir={llm_datasets_root}/lmms-lab___ai2d/",
                "--cross_kv_cache_fraction=0.5", "--max_ite=100"
            ])
        else:
            eval_cmd.extend([
                f"--dataset_dir={llm_datasets_root}/lmms-lab__VQAv2_valid_2000samples/"
            ])

        if phi3_model:
            eval_cmd.extend(["--kv_cache_free_gpu_memory_fraction", "0.4"])
        elif enc_dec_model:
            eval_cmd.extend(["--cross_kv_cache_fraction", "0.5"])

        print(f"Run {model_name} evaluation test...")
        _call_run_cmd(llm_venv, llm_root, eval_cmd, world_size)


@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize('cpp_e2e', [False, True],
                         ids=lambda cpp_e2e: f'cpp_e2e:{cpp_e2e}')
@pytest.mark.parametrize("batch_size", [1, 8],
                         ids=lambda batch_size: f'bs:{batch_size}')
@pytest.mark.parametrize(
    "data_type",
    ['float16', 'bfloat16', 'int4_weight_only', 'int8_weight_only'])
@pytest.mark.parametrize("tp_size", [1, 2], ids=lambda tp_size: f'tp:{tp_size}')
@pytest.mark.parametrize("pp_size", [1, 2], ids=lambda pp_size: f'pp:{pp_size}')
@pytest.mark.parametrize("multimodal_model_root", [
    'blip2-opt-2.7b',
    'blip2-flan-t5-xl',
    'llava-1.5-7b-hf',
    'llava-v1.6-mistral-7b-hf',
    pytest.param('llava-v1.6-mistral-7b-hf-vision-trtllm',
                 marks=pytest.mark.skipif(get_device_memory() < 50000,
                                          reason="Skip due to low memory")),
    'llava-onevision-qwen2-7b-ov-hf',
    'llava-onevision-qwen2-7b-ov-hf-video',
    pytest.param('nougat-base', marks=skip_post_blackwell),
    'VILA1.5-3b',
    'cogvlm-chat',
    'fuyu-8b',
    pytest.param('deplot', marks=skip_post_blackwell),
    pytest.param('neva-22b',
                 marks=pytest.mark.skip(reason="RCCA https://nvbugs/5220761")),
    'kosmos-2',
    pytest.param('video-neva', marks=skip_post_blackwell),
    pytest.param('Phi-3-vision-128k-instruct', marks=skip_post_blackwell),
    pytest.param('Phi-3.5-vision-instruct', marks=skip_post_blackwell),
    pytest.param('Phi-4-multimodal-instruct', marks=skip_post_blackwell),
    pytest.param('Llama-3.2-11B-Vision', marks=skip_post_blackwell),
    'Qwen2-VL-7B-Instruct',
    'internlm-xcomposer2-vl-7b',
    'Mistral-Small-3.1-24B-Instruct-2503',
],
                         indirect=True)
def test_llm_multimodal_general(llm_venv, llm_root, llm_datasets_root,
                                cmodel_dir, engine_dir, batch_size, data_type,
                                tp_size, pp_size, multimodal_example_root,
                                multimodal_model_root, recover_transformers,
                                cpp_e2e, num_beams):
    _test_llm_multimodal_general(llm_venv,
                                 llm_root,
                                 llm_datasets_root,
                                 cmodel_dir,
                                 engine_dir,
                                 batch_size,
                                 data_type,
                                 tp_size,
                                 pp_size,
                                 multimodal_example_root,
                                 multimodal_model_root,
                                 recover_transformers,
                                 cpp_e2e=cpp_e2e,
                                 num_beams=num_beams)


@skip_pre_ada
@pytest.mark.parametrize('cpp_e2e', [False, True],
                         ids=lambda cpp_e2e: f'cpp_e2e:{cpp_e2e}')
@pytest.mark.parametrize("batch_size", [1, 8],
                         ids=lambda batch_size: f'bs:{batch_size}')
@pytest.mark.parametrize("data_type", ['float16', 'bfloat16'])
@pytest.mark.parametrize("tp_size", [1, 2], ids=lambda tp_size: f'tp:{tp_size}')
@pytest.mark.parametrize("pp_size", [1, 2], ids=lambda pp_size: f'pp:{pp_size}')
@pytest.mark.parametrize("multimodal_model_root", [
    'blip2-opt-2.7b',
    'blip2-flan-t5-xl',
    'llava-1.5-7b-hf',
    'llava-v1.6-mistral-7b-hf',
    'llava-onevision-qwen2-7b-ov-hf',
    'llava-onevision-qwen2-7b-ov-hf-video',
    'nougat-base',
    'VILA1.5-3b',
    'cogvlm-chat',
    'fuyu-8b',
    'deplot',
    'neva-22b',
    'kosmos-2',
    'video-neva',
    'Phi-3-vision-128k-instruct',
    'Phi-3.5-vision-instruct',
    'Phi-4-multimodal-instruct',
    pytest.param('Llama-3.2-11B-Vision-Instruct', marks=skip_post_blackwell),
    pytest.param('Llama-3.2-11B-Vision', marks=skip_post_blackwell),
    'Qwen2-VL-7B-Instruct',
],
                         indirect=True)
@pytest.mark.parametrize('calibration_dataset', ['scienceqa', 'cnn_dailymail'])
@pytest.mark.parametrize('qformat', ['fp8'])
@pytest.mark.parametrize('kv_cache_dtype', ['fp8'])
def test_llm_fp8_multimodal_general(
        llm_venv, llm_root, llm_datasets_root, cmodel_dir, engine_dir,
        batch_size, data_type, tp_size, pp_size, multimodal_example_root,
        multimodal_model_root, recover_transformers, calibration_dataset,
        qformat, kv_cache_dtype, cpp_e2e):
    _test_llm_multimodal_general(llm_venv,
                                 llm_root,
                                 llm_datasets_root,
                                 cmodel_dir,
                                 engine_dir,
                                 batch_size,
                                 data_type,
                                 tp_size,
                                 pp_size,
                                 multimodal_example_root,
                                 multimodal_model_root,
                                 recover_transformers,
                                 calibration_dataset,
                                 qformat,
                                 kv_cache_dtype,
                                 cpp_e2e=cpp_e2e)
