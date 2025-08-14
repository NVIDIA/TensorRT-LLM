import os

from .trt_test_alternative import check_call, print_info

install_requirement_cmd = "pip3 install -r requirements.txt; pip install sentencepiece --upgrade"


def append_timing_cache_args(build_cmd):
    TIMING_CACHE_DIR = os.environ.get("TIMING_CACHE_DIR", "")
    if TIMING_CACHE_DIR:
        timing_cache = os.path.join(TIMING_CACHE_DIR, "model.cache")
        build_cmd.append(f"--input_timing_cache={timing_cache}")
        build_cmd.append(f"--output_timing_cache={timing_cache}")


def prepare_medusa_vicuna_7b_engine(tensorrt_llm_medusa_example_root,
                                    vicuna_7b_model_root,
                                    medusa_vicuna_7b_model_root):
    # Convert Medusa from HF
    ckpt_dir = os.path.join(tensorrt_llm_medusa_example_root, "model_dir",
                            "medusa_vicuna_7b")
    convert_cmd = [
        "python3", f"{tensorrt_llm_medusa_example_root}/convert_checkpoint.py",
        f"--model_dir={vicuna_7b_model_root}",
        f"--medusa_model_dir={medusa_vicuna_7b_model_root}",
        f"--output_dir={ckpt_dir}", "--dtype=float16", "--num_medusa_heads=4"
    ]

    # Build Medusa: float16
    engine_dir = os.path.join(tensorrt_llm_medusa_example_root, "engine_dir",
                              "medusa_vicuna_7b")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gemm_plugin=float16",
        "--max_batch_size=8",
        "--max_seq_len=600",
        "--speculative_decoding_mode=medusa",
    ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_medusa_example_root)
        check_call(convert_cmd, shell=True)
        check_call(build_cmd, shell=True)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_eagle_vicuna_7b_engine(tensorrt_llm_eagle_example_root,
                                   vicuna_7b_model_root,
                                   eagle_vicuna_7b_model_root):
    # Convert Eagle from HF
    ckpt_dir = os.path.join(tensorrt_llm_eagle_example_root, "model_dir",
                            "eagle_vicuna_7b")
    convert_cmd = [
        "python3", f"{tensorrt_llm_eagle_example_root}/convert_checkpoint.py",
        f"--model_dir={vicuna_7b_model_root}",
        f"--eagle_model_dir={eagle_vicuna_7b_model_root}",
        f"--output_dir={ckpt_dir}", "--dtype=float16", "--num_eagle_layers=4"
    ]

    # Build Eagle: float16
    engine_dir = os.path.join(tensorrt_llm_eagle_example_root, "engine_dir",
                              "eagle_vicuna_7b")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gemm_plugin=float16",
        "--max_batch_size=8",
        "--max_seq_len=600",
        "--speculative_decoding_mode=eagle",
    ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_eagle_example_root)
        check_call(convert_cmd, shell=True)
        check_call(build_cmd, shell=True)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_t5_small_engine(tensorrt_llm_enc_dec_example_root,
                            t5_small_model_root):
    # Convert T5 from HF
    ckpt_dir = os.path.join(tensorrt_llm_enc_dec_example_root, "model_dir",
                            "t5_small")
    convert_cmd = [
        "python3",
        f"{tensorrt_llm_enc_dec_example_root}/convert_checkpoint.py",
        "--model_type=t5",
        f"--model_dir={t5_small_model_root}",
        f"--output_dir={ckpt_dir}",
        "--dtype=float16",
    ]

    # Build encoder and decoder
    encoder_engine_dir = os.path.join(tensorrt_llm_enc_dec_example_root,
                                      "engine_dir", "t5_small_encoder")
    decoder_engine_dir = os.path.join(tensorrt_llm_enc_dec_example_root,
                                      "engine_dir", "t5_small_decoder")

    encoder_build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}/encoder",
        f"--output_dir={encoder_engine_dir}",
        "--kv_cache_type=disabled",
        "--moe_plugin=disable",
        "--max_beam_width=1",
        "--max_batch_size=8",
        "--max_input_len=512",
        "--max_seq_len=512",
        "--gemm_plugin=float16",
        "--bert_attention_plugin=float16",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--context_fmha=disable",
    ]
    decoder_build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}/decoder",
        f"--output_dir={decoder_engine_dir}",
        "--moe_plugin=disable",
        "--max_beam_width=1",
        "--max_batch_size=8",
        "--max_input_len=1",
        "--max_seq_len=512",
        "--max_encoder_input_len=512",
        "--gemm_plugin=float16",
        "--bert_attention_plugin=float16",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--context_fmha=disable",
    ]
    append_timing_cache_args(encoder_build_cmd)
    append_timing_cache_args(decoder_build_cmd)

    convert_cmd = " ".join(convert_cmd)
    encoder_build_cmd = " ".join(encoder_build_cmd)
    decoder_build_cmd = " ".join(decoder_build_cmd)
    if not os.path.exists(encoder_build_cmd):
        check_call(convert_cmd, shell=True)
        check_call(encoder_build_cmd, shell=True)
        check_call(decoder_build_cmd, shell=True)

    else:
        print_info(f"Reusing engine: {encoder_engine_dir}")
        print_info(f"Reusing engine: {decoder_engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {encoder_build_cmd}")
        print_info(f"Skipped: {decoder_build_cmd}")

    assert os.path.exists(
        encoder_engine_dir), f"{encoder_engine_dir} does not exists."
    assert os.path.exists(
        decoder_engine_dir), f"{decoder_engine_dir} does not exists."
    return encoder_engine_dir, decoder_engine_dir


def prepare_whisper_large_engine(tensorrt_llm_whisper_example_root,
                                 whisper_large_model_root):
    # Convert OpenAI Whisper Checkpoint
    ckpt_dir = os.path.join(tensorrt_llm_whisper_example_root, "model_dir",
                            "whisper_large")
    convert_cmd = [
        "python3",
        f"{tensorrt_llm_whisper_example_root}/convert_checkpoint.py",
        f"--model_dir={whisper_large_model_root}",
        f"--output_dir={ckpt_dir}",
    ]

    # Build encoder and decoder
    encoder_engine_dir = os.path.join(tensorrt_llm_whisper_example_root,
                                      "engine_dir", "whisper_large_encoder")
    decoder_engine_dir = os.path.join(tensorrt_llm_whisper_example_root,
                                      "engine_dir", "whisper_large_decoder")

    encoder_build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}/encoder",
        f"--output_dir={encoder_engine_dir}",
        "--moe_plugin=disable",
        "--max_batch_size=8",
        "--gemm_plugin=disable",
        "--bert_attention_plugin=float16",
        "--max_input_len=3000",
        "--max_seq_len=3000",
    ]
    decoder_build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}/decoder",
        f"--output_dir={decoder_engine_dir}",
        "--moe_plugin=disable",
        "--max_beam_width=4",
        "--max_batch_size=8",
        "--max_seq_len=114",
        "--max_input_len=14",
        "--max_encoder_input_len=3000",
        "--gemm_plugin=float16",
        "--bert_attention_plugin=float16",
        "--gpt_attention_plugin=float16",
    ]
    append_timing_cache_args(encoder_build_cmd)
    append_timing_cache_args(decoder_build_cmd)

    convert_cmd = " ".join(convert_cmd)
    encoder_build_cmd = " ".join(encoder_build_cmd)
    decoder_build_cmd = " ".join(decoder_build_cmd)
    if not os.path.exists(encoder_build_cmd):
        check_call(convert_cmd, shell=True)
        check_call(encoder_build_cmd, shell=True)
        check_call(decoder_build_cmd, shell=True)

    else:
        print_info(f"Reusing engine: {encoder_engine_dir}")
        print_info(f"Reusing engine: {decoder_engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {encoder_build_cmd}")
        print_info(f"Skipped: {decoder_build_cmd}")

    assert os.path.exists(
        encoder_engine_dir), f"{encoder_engine_dir} does not exists."
    assert os.path.exists(
        decoder_engine_dir), f"{decoder_engine_dir} does not exists."
    return encoder_engine_dir, decoder_engine_dir


def prepare_gpt_350m_engine(type, tensorrt_llm_gpt_example_root,
                            gpt_tokenizer_model_root):
    # Convert GPT weights from HF
    if type in ["medium_target_ifb", "medium_control_ifb"]:
        ckpt_dir = os.path.join(tensorrt_llm_gpt_example_root, "model_dir",
                                "gpt_350m_medium")
    else:
        ckpt_dir = os.path.join(tensorrt_llm_gpt_example_root, "model_dir",
                                "gpt_350m")
    convert_cmd = [
        "python3", f"{tensorrt_llm_gpt_example_root}/convert_checkpoint.py",
        f"--model_dir={gpt_tokenizer_model_root}", "--dtype=float16",
        f"--output_dir={ckpt_dir}"
    ]

    # Build GPT
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "gpt350m_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "gpt350m_ifb")
    elif type == "medium_target_ifb":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "gpt350m_medium_target_ifb")
    elif type == "medium_control_ifb":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "gpt350m_medium_control_ifb")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--context_fmha=enable",
        "--use_paged_context_fmha=enable",
        "--remove_input_padding=enable",
        "--max_batch_size=8",
        "--max_num_tokens=7392",
        "--gather_generation_logits",
        f"--output_dir={engine_dir}",
    ]

    if type == "medium_target_ifb":
        build_cmd += [
            "--max_draft_len=5",
            "--speculative_decoding_mode=draft_tokens_external",
        ]

    if type in ["ifb", "medium_target_ifb", "medium_control_ifb"]:
        build_cmd += [
            "--kv_cache_type=paged",
        ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(convert_cmd, shell=True)
        check_call(build_cmd, shell=True)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_gptj_6b_engine(type, tensorrt_llm_gptj_example_root,
                           gptj_tokenizer_model_root):
    # Convert GPT weights from HF
    ckpt_dir = os.path.join(tensorrt_llm_gptj_example_root, "model_dir",
                            "gptj_6b")
    convert_cmd = [
        "python3",
        f"{tensorrt_llm_gptj_example_root}/convert_checkpoint.py",
        f"--model_dir={gptj_tokenizer_model_root}",
        "--dtype=float16",
        f"--output_dir={ckpt_dir}",
    ]

    # Build GPT
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_gptj_example_root, "engine_dir",
                                  "gptj_6b_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_gptj_example_root, "engine_dir",
                                  "gptj_6b_ifb")

    build_cmd = [
        "trtllm-build",
        f"--model_config={ckpt_dir}/config.json",
        "--context_fmha=enable",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        f"--output_dir={engine_dir}",
    ]

    if type == "ifb":
        build_cmd += [
            "--kv_cache_type=paged",
        ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gptj_example_root)
        check_call(convert_cmd, shell=True)
        check_call(build_cmd, shell=True)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_blip2_opt_engine(tensorrt_llm_multimodal_example_root,
                             tensorrt_llm_opt_example_root,
                             blip2_opt_model_root):
    # Convert OPT from HF
    ckpt_dir = os.path.join(tensorrt_llm_multimodal_example_root, "model_dir",
                            "blip2_opt")
    convert_cmd = [
        "python3",
        f"{tensorrt_llm_opt_example_root}/convert_checkpoint.py",
        "--model_type=blip2",
        f"--model_dir={blip2_opt_model_root}",
        "--dtype=float16",
        f"--output_dir={ckpt_dir}",
    ]

    # Build OPT
    engine_dir = os.path.join(tensorrt_llm_multimodal_example_root,
                              "engine_dir", "blip2_opt")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        "--gemm_plugin=float16",
        "--max_beam_width=1",
        "--max_batch_size=8",
        "--max_multimodal_len=256",
        "--max_input_len=924",
        "--max_seq_len=1024",
        "--use_paged_context_fmha=enable",
        f"--output_dir={engine_dir}",
    ]

    multimodal_engine_dir = os.path.join(tensorrt_llm_multimodal_example_root,
                                         "tmp", "trt_engines", "blip2-opt-2.7b",
                                         "multimodal_encoder")
    build_visual_engine_cmd = [
        "python3",
        "build_multimodal_engine.py",
        "--model_type=blip2",
        f"--model_path={blip2_opt_model_root}",
        f"--output_dir={multimodal_engine_dir}",
        "--max_batch_size=8",
    ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    build_visual_engine_cmd = " ".join(build_visual_engine_cmd)
    if not os.path.exists(engine_dir):
        check_call(convert_cmd, shell=True)
        check_call(build_cmd, shell=True)
        check_call(build_visual_engine_cmd,
                   shell=True,
                   cwd=tensorrt_llm_multimodal_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")
        print_info(f"Skipped: {build_visual_engine_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    assert os.path.exists(
        multimodal_engine_dir), f"{multimodal_engine_dir} does not exists."
    return engine_dir, multimodal_engine_dir


def prepare_llava_engine(tensorrt_llm_multimodal_example_root,
                         tensorrt_llm_llama_example_root, llava_model_root):
    # Convert LLAMA from HF
    ckpt_dir = os.path.join(tensorrt_llm_multimodal_example_root, "model_dir",
                            "llava")
    convert_cmd = [
        "python3",
        f"{tensorrt_llm_llama_example_root}/convert_checkpoint.py",
        f"--model_dir={llava_model_root}",
        f"--output_dir={ckpt_dir}",
        "--dtype=float16",
    ]

    # Build LLAVA
    engine_dir = os.path.join(tensorrt_llm_multimodal_example_root,
                              "engine_dir", "llava")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        "--gemm_plugin=float16",
        "--max_batch_size=8",
        "--max_multimodal_len=4608",
        "--max_input_len=2048",
        "--max_seq_len=2560",
        f"--output_dir={engine_dir}",
    ]

    multimodal_engine_dir = os.path.join(tensorrt_llm_multimodal_example_root,
                                         "tmp", "trt_engines",
                                         "llava-1.5-7b-hf",
                                         "multimodal_encoder")
    build_visual_engine_cmd = [
        "python3",
        "build_multimodal_engine.py",
        "--model_type=llava",
        f"--model_path={llava_model_root}",
        f"--output_dir={multimodal_engine_dir}",
        "--max_batch_size=8",
    ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    build_visual_engine_cmd = " ".join(build_visual_engine_cmd)
    if not os.path.exists(engine_dir):
        check_call(convert_cmd, shell=True)
        check_call(build_cmd, shell=True)
        check_call(build_visual_engine_cmd,
                   shell=True,
                   cwd=tensorrt_llm_multimodal_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")
        print_info(f"Skipped: {build_visual_engine_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    assert os.path.exists(
        multimodal_engine_dir), f"{multimodal_engine_dir} does not exists."
    return engine_dir, multimodal_engine_dir


def prepare_llava_onevision_engine(tensorrt_llm_multimodal_example_root,
                                   tensorrt_llm_qwen_example_root,
                                   llava_onevision_model_root,
                                   llm_backend_all_models_root):
    # Convert Qwen from HF
    ckpt_dir = os.path.join(tensorrt_llm_multimodal_example_root, "model_dir",
                            "llava_onevision_7b")
    convert_cmd = [
        "python3",
        f"{tensorrt_llm_qwen_example_root}/convert_checkpoint.py",
        f"--model_dir={llava_onevision_model_root}",
        f"--output_dir={ckpt_dir}",
        "--dtype=float16",
    ]

    # Build Qwen
    engine_dir = os.path.join(tensorrt_llm_multimodal_example_root,
                              "engine_dir", "llava_onevision_7b")
    multimodal_engine_dir = os.path.join(tensorrt_llm_multimodal_example_root,
                                         "multimodal_engine_dir",
                                         "llava_onevision_7b")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        "--gemm_plugin=float16",
        "--max_batch_size=1",
        "--max_input_len=7500",
        "--max_seq_len=7600",
        "--max_multimodal_len=7300",
        f"--output_dir={engine_dir}",
    ]

    build_visual_engine_cmd = [
        "python3",
        "build_multimodal_engine.py",
        "--model_type=llava_onevision",
        f"--model_path={llava_onevision_model_root}",
        f"--output_dir={multimodal_engine_dir}",
        "--max_batch_size=16",
    ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    build_visual_engine_cmd = " ".join(build_visual_engine_cmd)
    if not os.path.exists(engine_dir):
        install_requirement_cmd = "pip3 install -r multimodal/requirements-llava-onevision.txt"
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=llm_backend_all_models_root)
        check_call(convert_cmd, shell=True)
        check_call(build_cmd, shell=True)
        check_call(build_visual_engine_cmd,
                   shell=True,
                   cwd=tensorrt_llm_multimodal_example_root)
    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")
        print_info(f"Skipped: {build_visual_engine_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    assert os.path.exists(
        multimodal_engine_dir), f"{multimodal_engine_dir} does not exists."
    return engine_dir, multimodal_engine_dir


def prepare_mllama_engine(tensorrt_llm_multimodal_example_root,
                          tensorrt_llm_mllama_example_root, mllama_model_root,
                          llm_backend_root):
    # Convert MLLAMA from HF
    model_name = "Llama-3.2-11B-Vision"
    ckpt_dir = os.path.join(tensorrt_llm_multimodal_example_root, "model_dir",
                            model_name)
    convert_cmd = [
        "python3",
        f"{tensorrt_llm_mllama_example_root}/convert_checkpoint.py",
        f"--model_dir={mllama_model_root}",
        "--dtype=bfloat16",
        f"--output_dir={ckpt_dir}",
    ]

    # Build MLLAMA
    engine_dir = os.path.join(tensorrt_llm_multimodal_example_root,
                              "engine_dir", model_name)
    multimodal_engine_dir = os.path.join(tensorrt_llm_multimodal_example_root,
                                         "multimodal_engine_dir", model_name)

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        "--gemm_plugin auto",
        "--max_beam_width=1",
        "--max_batch_size 8",
        "--max_seq_len 2048",
        "--max_num_tokens 4096",
        "--max_encoder_input_len 8200",
        "--use_paged_context_fmha=enable",
        f"--output_dir={engine_dir}",
    ]

    build_visual_engine_cmd = [
        "python3", "build_multimodal_engine.py", "--model_type=mllama",
        f"--model_path={mllama_model_root}", "--max_batch_size=8",
        f"--output_dir={multimodal_engine_dir}"
    ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    build_visual_engine_cmd = " ".join(build_visual_engine_cmd)
    if not os.path.exists(engine_dir) or not os.path.exists(
            multimodal_engine_dir):
        check_call(install_requirement_cmd, shell=True, cwd=llm_backend_root)
        check_call(convert_cmd, shell=True)
        check_call(build_cmd, shell=True)
        check_call(build_visual_engine_cmd,
                   shell=True,
                   cwd=tensorrt_llm_multimodal_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")
        print_info(f"Skipped: {build_visual_engine_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    assert os.path.exists(
        multimodal_engine_dir), f"{multimodal_engine_dir} does not exists."
    return engine_dir, multimodal_engine_dir


def prepare_llama3_v1_8b_engine(tensorrt_llm_llama_example_root,
                                llama3_v1_8b_model_root):
    engine_dir = os.path.join(tensorrt_llm_llama_example_root, "engine_dir",
                              "llama3_v1_8b")
    ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                            "llama3_v1_8b")

    convert_cmd = [
        "python3",
        "convert_checkpoint.py",
        f"--model_dir={llama3_v1_8b_model_root}",
        f"--output_dir={ckpt_dir}",
        "--dtype=bfloat16",
        f"--tp_size=1",
        f"--workers=1",
    ]

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gemm_plugin=bfloat16",
        "--workers=1",
        "--max_batch_size=2048",
        "--max_input_len=2048",
        "--max_seq_len=4096",
        "--max_beam_width=1",
        "--gpt_attention_plugin=bfloat16",
        "--reduce_fusion=disable",
        "--max_num_tokens=16384",
        "--use_paged_context_fmha=disable",
        "--multiple_profiles=disable",
    ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(convert_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_gpt_gather_logits_engine(type, tensorrt_llm_gpt_example_root,
                                     gpt_tokenizer_model_root):
    # Convert GPT weights from HF
    ckpt_dir = os.path.join(tensorrt_llm_gpt_example_root, "model_dir",
                            "gpt_gather_logits")
    convert_cmd = [
        "python3", f"{tensorrt_llm_gpt_example_root}/convert_checkpoint.py",
        f"--model_dir={gpt_tokenizer_model_root}", "--dtype=float16",
        f"--output_dir={ckpt_dir}"
    ]

    # Build GPT
    engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                              "gpt_gather_logits")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--context_fmha=enable",
        "--remove_input_padding=enable",
        "--max_batch_size=128",
        "--max_seq_len=600",
        "--gather_all_token_logits",
        "--max_num_tokens=38400",
        f"--output_dir={engine_dir}",
    ]

    if type == "ifb":
        build_cmd += [
            "--kv_cache_type=paged",
        ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(convert_cmd, shell=True)
        check_call(build_cmd, shell=True)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_gpt_return_logits_engine(type, tensorrt_llm_gpt_example_root,
                                     gpt_tokenizer_model_root):
    # Convert GPT weights from HF
    ckpt_dir = os.path.join(tensorrt_llm_gpt_example_root, "model_dir",
                            "gpt_return_logits")
    convert_cmd = [
        "python3", f"{tensorrt_llm_gpt_example_root}/convert_checkpoint.py",
        f"--model_dir={gpt_tokenizer_model_root}", "--dtype=float16",
        f"--output_dir={ckpt_dir}"
    ]

    # Build GPT
    engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                              "gpt_return_logits")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        "--max_batch_size=4",
        "--max_seq_len=540",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
        "--max_num_tokens=38400",
        "--use_paged_context_fmha=enable",
        "--gather_generation_logits",
        f"--output_dir={engine_dir}",
    ]

    if type == "ifb":
        build_cmd += [
            "--kv_cache_type=paged",
        ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(convert_cmd, shell=True)
        check_call(build_cmd, shell=True)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_gpt_2b_lora_engine(type, tensorrt_llm_gpt_example_root,
                               gpt_2b_lora_model_root, models_root,
                               weight_streaming):
    # Convert GPT from NeMo
    ckpt_dir = os.path.join(tensorrt_llm_gpt_example_root, "model_dir",
                            "gpt_2b_lora")
    gpt_2b_nemo_model = os.path.join(models_root, "GPT-2B-001_bf16_tp1.nemo")

    convert_ckpt_cmd = [
        "python3", f"{tensorrt_llm_gpt_example_root}/convert_checkpoint.py",
        f"--nemo_ckpt_path={gpt_2b_nemo_model}", "--dtype=float16",
        f"--output_dir={ckpt_dir}"
    ]

    # prepare more test metrials
    gpt_2b_lora_900_nemo_model = os.path.join(gpt_2b_lora_model_root,
                                              "gpt2b_lora-900.nemo")
    convert_lora_train_cmd = [
        "python3", f"{tensorrt_llm_gpt_example_root}/nemo_lora_convert.py",
        f"-i={gpt_2b_lora_900_nemo_model}", "--storage-type=float16",
        "--write-cpp-runtime-tensors", f"-o=gpt-2b-lora-train-900"
    ]
    convert_lora_train_tllm_cmd = [
        "python3", f"{tensorrt_llm_gpt_example_root}/nemo_lora_convert.py",
        f"-i={gpt_2b_lora_900_nemo_model}", "--storage-type=float16",
        f"-o=gpt-2b-lora-train-900-tllm"
    ]

    check_call(f"cp {gpt_2b_lora_model_root}/gpt2b_lora-900.nemo ./",
               shell=True,
               cwd=tensorrt_llm_gpt_example_root)
    check_call(f"cp {gpt_2b_lora_model_root}/input.csv ./",
               shell=True,
               cwd=tensorrt_llm_gpt_example_root)

    # Build GPT
    engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                              "gpt_2b_lora_ib")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--lora_plugin=float16",
        f"--lora_dir={gpt_2b_lora_900_nemo_model}",
        "--lora_ckpt_source=nemo",
        "--lora_target_modules=attn_qkv",
        "--remove_input_padding=enable",
        "--max_batch_size=8",
        "--max_seq_len=1052",
        f"--output_dir={engine_dir}",
    ]

    if weight_streaming:
        build_cmd += ["--gemm_plugin=disable", "--weight_streaming"]
    else:
        build_cmd += [
            "--gemm_plugin=float16",
        ]

    if type == "ifb":
        build_cmd += [
            "--kv_cache_type=paged",
        ]

    append_timing_cache_args(build_cmd)
    convert_ckpt_cmd = " ".join(convert_ckpt_cmd)
    build_cmd = " ".join(build_cmd)
    convert_lora_train_cmd = " ".join(convert_lora_train_cmd)
    convert_lora_train_tllm_cmd = " ".join(convert_lora_train_tllm_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(convert_ckpt_cmd, shell=True)
        check_call(convert_lora_train_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(convert_lora_train_tllm_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(build_cmd, shell=True)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_ckpt_cmd}")
        print_info(f"Skipped: {build_cmd}")
        print_info(f"Skipped: {convert_lora_train_cmd}")
        print_info(f"Skipped: {convert_lora_train_tllm_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_gpt_175b_engine(type, tensorrt_llm_gpt_example_root,
                            tensorrt_llm_example_root):
    # Build GPT
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "gpt_175b_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "gpt_175b_ifb")

    convert_cmd = [
        "python3", f"{tensorrt_llm_example_root}/generate_checkpoint_config.py",
        f"--output_path={engine_dir}/ckpt_config.json",
        "--architecture=GPTForCausalLM", "--dtype=float16",
        "--num_hidden_layers=96", "--num_attention_heads=96",
        "--hidden_size=12288", "--vocab_size=51200", "--hidden_act=gelu",
        "--tp_size=8"
    ]

    build_cmd = [
        "trtllm-build",
        f"--model_config={engine_dir}/ckpt_config.json",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--gemm_plugin=float16",
        "--max_batch_size=32",
        "--max_seq_len=544",
        f"--output_dir={engine_dir}",
    ]

    if type == "ifb":
        build_cmd += [
            "--kv_cache_type=paged",
        ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(convert_cmd, shell=True, cwd=tensorrt_llm_gpt_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_gpt_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_gpt_multi_node_engine(type, tensorrt_llm_gpt_example_root,
                                  tensorrt_llm_example_root):
    # Build GPT
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "gpt_multi_node_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "gpt_multi_node_ifb")

    convert_cmd = [
        "python3", f"{tensorrt_llm_example_root}/generate_checkpoint_config.py",
        f"--output_path={engine_dir}/ckpt_config.json",
        "--architecture=GPTForCausalLM", "--dtype=float16",
        "--num_hidden_layers=96", "--num_attention_heads=96",
        "--hidden_size=12288", "--vocab_size=51200", "--hidden_act=gelu",
        "--tp_size=16"
    ]

    build_cmd = [
        "trtllm-build",
        f"--model_config={engine_dir}/ckpt_config.json",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--gemm_plugin=float16",
        "--max_batch_size=32",
        "--max_seq_len=544",
        f"--output_dir={engine_dir}",
    ]

    if type == "ifb":
        build_cmd += [
            "--kv_cache_type=paged",
        ]

    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(convert_cmd, shell=True, cwd=tensorrt_llm_gpt_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_gpt_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_llama_v2_7b_engine(type, tensorrt_llm_llama_example_root,
                               llama_v2_tokenizer_model_root):
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root, "engine_dir",
                                  "llama_v2_7b_python_backend")
        ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                                "llama_v2_7b_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root, "engine_dir",
                                  "llama_v2_7b_ifb")
        ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                                "llama_v2_7b_ifb")
    # The path of weights in data server
    meta_ckpt_dir = os.path.join(llama_v2_tokenizer_model_root, "7B")

    convert_cmd = [
        "python3",
        "convert_checkpoint.py",
        f"--meta_ckpt_dir={meta_ckpt_dir}",
        f"--output_dir={ckpt_dir}",
        "--dtype=bfloat16",
    ]

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=bfloat16",
        "--gemm_plugin=bfloat16",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
    ]

    if type == "ifb":
        build_cmd += [
            "--kv_cache_type=paged",
        ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(convert_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_llama_v2_13b_engine(tensorrt_llm_llama_example_root,
                                llama_v2_tokenizer_model_root):
    engine_dir = os.path.join(tensorrt_llm_llama_example_root, "engine_dir",
                              "llama_v2_13b_ifb")
    ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                            "llama_v2_13b_ifb")
    # The path of weights in data server
    meta_ckpt_dir = os.path.join(llama_v2_tokenizer_model_root, "13B")

    convert_cmd = [
        "python3",
        "convert_checkpoint.py",
        f"--meta_ckpt_dir={meta_ckpt_dir}",
        f"--output_dir={ckpt_dir}",
        "--dtype=bfloat16",
        "--tp_size=2",
        "--workers=2",
    ]

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--workers=2",
        "--max_batch_size=64",
        "--tokens_per_block=64",
        "--use_paged_context_fmha=enable",
        "--context_fmha=enable",
        "--paged_kv_cache=enable",
        "--max_num_tokens=8192",
        "--max_input_len=4096",
        "--max_seq_len=4096",
    ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(convert_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_llama_v3_8b_engine(tensorrt_llm_example_root,
                               tensorrt_llm_llama_example_root,
                               llama_v3_8b_model_root,
                               workers=8,
                               data_type="bfloat16"):
    engine_dir = os.path.join(tensorrt_llm_llama_example_root, "engine_dir",
                              f"llama_v3_8b_{data_type}_ifb")
    ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                            f"llama_v3_8b_{data_type}_ifb")

    if data_type == "bfloat16":
        convert_cmd = [
            "python3",
            "convert_checkpoint.py",
            f"--model_dir={llama_v3_8b_model_root}",
            f"--output_dir={ckpt_dir}",
            "--dtype=bfloat16",
            f"--tp_size={workers}",
            f"--workers={workers}",
        ]
    elif data_type == "fp8":
        convert_cmd = [
            "python3",
            f"{tensorrt_llm_example_root}/quantization/quantize.py",
            f"--model_dir={llama_v3_8b_model_root}",
            "--dtype=float16",
            "--qformat=fp8",
            "--kv_cache_dtype=fp8",
            f"--output_dir={ckpt_dir}",
            "--tp_size=8",
        ]

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gemm_plugin=auto",
        "--moe_plugin=auto",
        "--nccl_plugin=auto",
        "--gpt_attention_plugin=auto",
        "--use_paged_context_fmha=enable",
        "--remove_input_padding=enable",
        "--use_fused_mlp=enable",
        "--multiple_profiles=enable",
        "--kv_cache_type=paged",
        "--max_seq_len=4096",
        "--max_batch_size=96",
        f"--workers={workers}",
        "--gather_generation_logits",
    ]

    if data_type == "fp8":
        build_cmd += [
            "--use_fp8_context_fmha=enable",
        ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(convert_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_llama_v3_70b_engine(type,
                                tensorrt_llm_example_root,
                                tensorrt_llm_llama_example_root,
                                llama_v3_70b_model_root,
                                data_type="bfloat16"):
    if type == "control_ifb":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root, "engine_dir",
                                  f"llama_v3_70b_{data_type}_control_ifb")
    elif type == "target_ifb":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root, "engine_dir",
                                  f"llama_v3_70b_{data_type}_target_ifb")
    ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                            "llama_v3_70b_ifb")

    if data_type == "bfloat16":
        convert_cmd = [
            "python3",
            "convert_checkpoint.py",
            f"--model_dir={llama_v3_70b_model_root}",
            f"--output_dir={ckpt_dir}",
            "--dtype=bfloat16",
            "--tp_size=8",
            "--workers=8",
        ]
    elif data_type == "fp8":
        convert_cmd = [
            "python3",
            f"{tensorrt_llm_example_root}/quantization/quantize.py",
            f"--model_dir={llama_v3_70b_model_root}",
            "--dtype=float16",
            "--qformat=fp8",
            "--kv_cache_dtype=fp8",
            f"--output_dir={ckpt_dir}",
            "--tp_size=8",
        ]

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gemm_plugin=auto",
        "--moe_plugin=auto",
        "--nccl_plugin=auto",
        "--gpt_attention_plugin=auto",
        "--use_paged_context_fmha=enable",
        "--remove_input_padding=enable",
        "--use_fused_mlp=enable",
        "--multiple_profiles=enable",
        "--kv_cache_type=paged",
        "--max_seq_len=4096",
        "--max_batch_size=96",
        "--workers=8",
        "--gather_generation_logits",
    ]
    if type == "target_ifb":
        build_cmd += [
            "--max_draft_len=10",
            "--speculative_decoding_mode=draft_tokens_external",
        ]

    if data_type == "fp8":
        build_cmd += [
            "--use_fp8_context_fmha=enable",
        ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(convert_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_llama_v2_70b_engine(type,
                                tensorrt_llm_llama_example_root,
                                llama_v2_tokenizer_model_root,
                                use_lad=False):
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root, "engine_dir",
                                  "llama_v2_70b_python_backend")
        ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                                "llama_v2_70b_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root, "engine_dir",
                                  "llama_v2_70b_ifb")
        ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                                "llama_v2_70b_ifb")
    if use_lad:
        engine_dir = os.path.join(tensorrt_llm_llama_example_root, "engine_dir",
                                  "llama_v2_70b_ifb_lad")
        ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                                "llama_v2_70b_ifb_lad")
    # The path of weights in data server
    meta_ckpt_dir = os.path.join(llama_v2_tokenizer_model_root, "70B")
    convert_cmd = [
        "python3",
        "convert_checkpoint.py",
        f"--meta_ckpt_dir={meta_ckpt_dir}",
        f"--output_dir={ckpt_dir}",
        "--dtype=bfloat16",
        "--tp_size=8",
    ]

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=bfloat16",
        "--gemm_plugin=bfloat16",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
    ]

    if type == "lad":
        build_cmd += [
            "--max_draft_len=83",
            "--speculative_decoding_mode=lookahead_decoding"
        ]

    if type == "ifb":
        build_cmd += [
            "--kv_cache_type=paged",
        ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(convert_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_gpt_next_ptuning_engine(type, tensorrt_llm_gpt_example_root,
                                    gpt_next_ptuning_model_root):
    if type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "gpt_next_ptuning_ifb")

    # Convert weights from HF
    nemo_model_path = os.path.join(gpt_next_ptuning_model_root,
                                   "megatron_converted_8b_tp4_pp1.nemo")
    ckpt_dir = os.path.join(tensorrt_llm_gpt_example_root, "model_dir",
                            "gpt_next_ptuning")
    convert_weights_cmd = [
        "python3",
        "convert_checkpoint.py",
        f"--nemo_ckpt_path={nemo_model_path}",
        f"--output_dir={ckpt_dir}",
        "--dtype=float16",
    ]

    # Convert ptuning table
    nemo_model_path = os.path.join(gpt_next_ptuning_model_root,
                                   "email_composition.nemo")
    convert_table_cmd = [
        "python3",
        "nemo_prompt_convert.py",
        f"-i={nemo_model_path}",
        "-o=email_composition.npy",
    ]

    # Copy input.csv
    check_call(f"cp {gpt_next_ptuning_model_root}/input.csv ./",
               shell=True,
               cwd=tensorrt_llm_gpt_example_root)

    # Build engine
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--kv_cache_type=paged",
        "--gemm_plugin=float16",
        "--context_fmha=enable",
        "--max_batch_size=8",
        "--max_seq_len=1052",
        "--max_beam_width=1",
        f"--output_dir={engine_dir}",
        "--max_prompt_embedding_table_size=800",
    ]

    append_timing_cache_args(build_cmd)
    convert_weights_cmd = " ".join(convert_weights_cmd)
    convert_table_cmd = " ".join(convert_table_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(convert_weights_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(convert_table_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_gpt_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_weights_cmd}")
        print_info(f"Skipped: {convert_table_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    assert os.path.exists(ckpt_dir), f"{ckpt_dir} does not exists."
    return engine_dir, ckpt_dir


def prepare_mistral_v1_7b_engine(type, tensorrt_llm_llama_example_root,
                                 mistral_v1_tokenizer_model_root):
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root, "engine_dir",
                                  "mistral_v1_7b_python_backend")
        ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                                "mistral_v1_7b_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root, "engine_dir",
                                  "mistral_v1_7b_ifb")
        ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                                "mistral_v1_7b_ifb")
    elif type == "beam_search":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root, "engine_dir",
                                  "mistral_v1_7b_beam_search")
        ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                                "mistral_v1_7b_beam_search")

    convert_cmd = [
        "python3",
        "convert_checkpoint.py",
        f"--model_dir={mistral_v1_tokenizer_model_root}",
        f"--output_dir={ckpt_dir}",
        "--dtype=float16",
    ]

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
        "--max_input_len=8192",
    ]

    if type == "ifb":
        build_cmd += [
            "--kv_cache_type=paged",
        ]
    elif type == "python_backend":
        build_cmd += [
            "--max_batch_size=8",
        ]
    elif type == "beam_search":
        build_cmd += [
            "--kv_cache_type=paged",
            "--max_batch_size=32",
            "--max_beam_width=10",
        ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(convert_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_rcca_nvbug_4323566_engine(type, tensorrt_llm_gpt_example_root,
                                      gpt_tokenizer_model_root):
    # Convert GPT weights from HF
    ckpt_dir = os.path.join(tensorrt_llm_gpt_example_root, "model_dir",
                            "rcca_nvbug_4323566")
    convert_cmd = [
        "python3", f"{tensorrt_llm_gpt_example_root}/convert_checkpoint.py",
        f"--model_dir={gpt_tokenizer_model_root}", "--dtype=float16",
        f"--output_dir={ckpt_dir}"
    ]

    # Build GPT
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "rcca_nvbug_4323566_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "rcca_nvbug_4323566_ifb")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--context_fmha=enable",
        "--remove_input_padding=enable",
        "--max_batch_size=64",
        "--max_seq_len=1024",
        f"--output_dir={engine_dir}",
    ]

    if type == "ifb":
        build_cmd += [
            "--kv_cache_type=paged",
        ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(convert_cmd, shell=True)
        check_call(build_cmd, shell=True)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_rcca_nvbug_4342666_engine(type, tensorrt_llm_llama_example_root,
                                      llama_v2_tokenizer_model_root):
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root, "engine_dir",
                                  "rcca_nvbug_4342666_python_backend")
        ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                                "rcca_nvbug_4342666_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root, "engine_dir",
                                  "rcca_nvbug_4342666_ifb")
        ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                                "rcca_nvbug_4342666_ifb")
    # Weights of Llama-v2-7b-chat model
    meta_ckpt_dir = os.path.join(llama_v2_tokenizer_model_root, "7BF")
    convert_cmd = [
        "python3",
        "convert_checkpoint.py",
        f"--meta_ckpt_dir={meta_ckpt_dir}",
        f"--output_dir={ckpt_dir}",
        "--dtype=bfloat16",
    ]

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=bfloat16",
        "--gemm_plugin=bfloat16",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
        "--max_beam_width=4",
    ]

    if type == "ifb":
        build_cmd += [
            "--kv_cache_type=paged",
        ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(convert_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_rcca_nvbug_4895566_engine(tensorrt_llm_llama_example_root,
                                      mistral_v1_tokenizer_model_root):

    engine_dir = os.path.join(tensorrt_llm_llama_example_root, "engine_dir",
                              "rcca_nvbug_4895566")
    ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                            "rcca_nvbug_4895566")

    convert_cmd = [
        "python3",
        "convert_checkpoint.py",
        f"--model_dir={mistral_v1_tokenizer_model_root}",
        f"--output_dir={ckpt_dir}",
        "--dtype=float16",
    ]

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
        "--kv_cache_type=paged",
        "--max_input_len=5",  # mModel->getMaxInputLen() = max_seq_len - 1 = 4
        "--max_seq_len=5",
        "--max_num_tokens=5",
    ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(convert_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_tiny_llama_1b_engine(type, tensorrt_llm_llama_example_root,
                                 tiny_llama_model_root,
                                 tensorrt_llm_example_root):

    engine_dir = os.path.join(tensorrt_llm_llama_example_root, "engine_dir",
                              "tiny_llama_1b")
    ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                            "tiny_llama_1b")
    xgrammar_tokenizer_info_dir = os.path.join(tensorrt_llm_llama_example_root,
                                               "tokenizer_info",
                                               "tiny_llama_1b")
    xgrammar_tokenizer_info_path = os.path.join(xgrammar_tokenizer_info_dir,
                                                'xgrammar_tokenizer_info.json')

    convert_cmd = [
        "python3",
        "convert_checkpoint.py",
        f"--model_dir={tiny_llama_model_root}",
        f"--output_dir={ckpt_dir}",
        "--dtype=float16",
    ]

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--use_paged_context_fmha=enable",
    ]

    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)

    if not os.path.exists(engine_dir):
        check_call(convert_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)
    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    if type == "tensorrtllm" and not os.path.exists(
            xgrammar_tokenizer_info_path):
        convert_xgrammar_info_cmd = [
            "python3", "generate_xgrammar_tokenizer_info.py",
            f"--model_dir={tiny_llama_model_root}",
            f"--output_dir={xgrammar_tokenizer_info_dir}"
        ]
        convert_xgrammar_info_cmd = " ".join(convert_xgrammar_info_cmd)
        check_call(convert_xgrammar_info_cmd,
                   shell=True,
                   cwd=tensorrt_llm_example_root)
    else:
        print_info(
            f"Reusing xgrammar's tokenizer info: {xgrammar_tokenizer_info_path}"
        )

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."

    return engine_dir, xgrammar_tokenizer_info_path


def prepare_rcca_nvbug_4714193_engine(tensorrt_llm_example_root,
                                      tensorrt_llm_mixtral_example_root,
                                      mixtral_8x7b_v0_1_model_root,
                                      llm_backend_root):
    engine_dir = os.path.join(tensorrt_llm_mixtral_example_root, "engine_dir",
                              "rcca_nvbug_4714193")
    ckpt_dir = os.path.join(tensorrt_llm_mixtral_example_root, "ckpt_dir",
                            "rcca_nvbug_4714193")

    # Quantize model
    quantize_cmd = [
        "python3",
        f"{tensorrt_llm_example_root}/quantization/quantize.py",
        f"--model_dir={mixtral_8x7b_v0_1_model_root}",
        "--dtype=float16",
        "--qformat=fp8",
        "--kv_cache_dtype=fp8",
        f"--output_dir={ckpt_dir}",
        "--calib_size=512",
        "--tp_size=2",
    ]

    # Build engine
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--max_batch_size=32",
        "--max_input_len=8196",
        "--max_seq_len=10244",
        "--workers=2",
        "--max_num_tokens=13000",
        "--use_paged_context_fmha=enable",
        "--use_fp8_context_fmha=enable",
        "--remove_input_padding=enable",
    ]

    append_timing_cache_args(build_cmd)
    quantize_cmd = " ".join(quantize_cmd)
    build_cmd = " ".join(build_cmd)

    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd, shell=True, cwd=llm_backend_root)
        check_call(quantize_cmd,
                   shell=True,
                   cwd=tensorrt_llm_mixtral_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_mixtral_example_root)
    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {quantize_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_mistral3_pixtral_engine(tensorrt_llm_multimodal_example_root,
                                    tensorrt_llm_llama_example_root,
                                    mistral_small_model_root):
    # Convert Mistral3 from HF
    model_base_name = os.path.basename(mistral_small_model_root.rstrip("/"))
    ckpt_dir = os.path.join(tensorrt_llm_multimodal_example_root, "model_dir",
                            model_base_name)
    convert_cmd = [
        "python3",
        f"{tensorrt_llm_llama_example_root}/convert_checkpoint.py",
        "--dtype=bfloat16",
        f"--model_dir={mistral_small_model_root}",
        f"--output_dir={ckpt_dir}",
    ]

    # Build Mistral3 LLM engine
    engine_dir = os.path.join(tensorrt_llm_multimodal_example_root,
                              "engine_dir", model_base_name)

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        "--max_batch_size=4",
        "--max_input_len=8192",
        "--max_seq_len=8192",
        # Allow an arbitrary number of image tokens by setting:
        # max_multimodal_len = max_batch_size * max_input_len
        "--max_multimodal_len=32768",
        "--use_paged_context_fmha=enable",
        f"--output_dir={engine_dir}",
    ]

    # Build Pixtral visual encoder engine
    multimodal_engine_dir = os.path.join(tensorrt_llm_multimodal_example_root,
                                         "tmp", "trt_engines", model_base_name,
                                         "multimodal_encoder")
    build_visual_engine_cmd = [
        "python3",
        "build_multimodal_engine.py",
        "--model_type=pixtral",
        f"--model_path={mistral_small_model_root}",
        f"--output_dir={multimodal_engine_dir}",
        "--max_batch_size=2",
    ]

    append_timing_cache_args(build_cmd)
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    build_visual_engine_cmd = " ".join(build_visual_engine_cmd)
    if not os.path.exists(engine_dir) or not os.path.exists(
            multimodal_engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(convert_cmd, shell=True)
        check_call(build_cmd, shell=True)
        check_call(build_visual_engine_cmd,
                   shell=True,
                   cwd=tensorrt_llm_multimodal_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")
        print_info(f"Skipped: {build_visual_engine_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    assert os.path.exists(
        multimodal_engine_dir), f"{multimodal_engine_dir} does not exists."
    return engine_dir, multimodal_engine_dir
