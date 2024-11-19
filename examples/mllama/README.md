# MLLaMA
===

## Latest text model workflow on Huggingface

* install latest transformers

```bash
pip install -U transformers
```

* build vision encoder model via onnx

```bash
python examples/multimodal/build_visual_engine.py --model_type mllama \
                                                  --model_path Llama-3.2-11B-Vision/ \
                                                  --output_dir /tmp/mllama/trt_engines/encoder/
```
* build and run decoder model by TRT LLM

```bash
python examples/mllama/convert_checkpoint.py --model_dir Llama-3.2-11B-Vision/ \
                              --output_dir /tmp/mllama/trt_ckpts \
                              --dtype bfloat16

python3 -m tensorrt_llm.commands.build \
            --checkpoint_dir /tmp/mllama/trt_ckpts \
            --output_dir /tmp/mllama/trt_engines/decoder/ \
            --max_num_tokens 4096 \
            --max_seq_len 2048 \
            --workers 1 \
            --gemm_plugin auto \
            --max_batch_size 4 \
            --max_encoder_input_len 4100 \
            --input_timing_cache model.cache

# Run image+text test on multimodal/run.py with C++ runtime
python3 examples/multimodal/run.py --visual_engine_dir /tmp/mllama/trt_engines/encoder/ \
                                   --visual_engine_name visual_encoder.engine \
                                   --llm_engine_dir /tmp/mllama/trt_engines/decoder/ \
                                   --hf_model_dir Llama-3.2-11B-Vision/ \
                                   --image_path https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg \
                                   --input_text "<|image|><|begin_of_text|>If I had to write a haiku for this one" \
                                   --max_new_tokens 50 \
                                   --batch_size 2

# Run text only test on multimodal/run.py with C++ runtime
python3 examples/multimodal/run.py --visual_engine_dir /tmp/mllama/trt_engines/encoder/ \
                                   --visual_engine_name visual_encoder.engine \
                                   --llm_engine_dir /tmp/mllama/trt_engines/decoder/ \
                                   --hf_model_dir Llama-3.2-11B-Vision/ \
                                   --input_text "The key to life is" \
                                   --max_new_tokens 50 \
                                   --batch_size 2

Use model_runner_cpp by default. To switch to model_runner, set `--use_py_session` in the command mentioned above.

python3 examples/multimodal/eval.py --visual_engine_dir /tmp/mllama/trt_engines/encoder/ \
                                   --visual_engine_name visual_encoder.engine \
                                   --llm_engine_dir /tmp/mllama/trt_engines/decoder/ \
                                   --hf_model_dir Llama-3.2-11B-Vision/ \
                                   --test_trtllm \
                                   --accuracy_threshold 65 \
                                   --eval_task lmms-lab/ai2d
```
