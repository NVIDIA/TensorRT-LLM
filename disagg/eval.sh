python /home/scratch.timothyg_gpu/TensorRT-LLM/examples/llm-eval/lm-eval-harness/lm_eval_tensorrt_llm.py \
    --model local-completions \
    --model_args "base_url=http://localhost:8400/v1/completions,model=FP8-FP8,tokenizer=/home/scratch.trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct" \
    --tasks mmlu_generative \
    --output_path /home/scratch.timothyg_gpu/TensorRT-LLM/disagg/output/FP8-FP8 \
    --batch_size 1 \
    --log_samples &