<!-- omit from toc -->
# Multi-Modal

The engine-build multimodal workflow that used to live here
(`build_multimodal_engine.py` / `run.py` / `eval.py` on top of
`trtllm-build`) was removed together with the legacy TensorRT backend.

Multimodal models are supported on the PyTorch backend. See:

- [Supported models](https://nvidia.github.io/TensorRT-LLM/models/supported-models.html)
- [`trtllm-serve`](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html) and the
  [LLM Python API](https://nvidia.github.io/TensorRT-LLM/llm-api/index.html)
- Multimodal serving examples under `examples/llm-api/` and
  `examples/serve/`

## Qwen-Image-Bench Evaluator

[`qwen_image_bench_eval.py`](qwen_image_bench_eval.py) runs the
Qwen-Image-Bench VLM judge through TensorRT-LLM to evaluate an image generated
by another image or vision-language model. Provide the original generation
prompt and the generated image path:

```bash
python examples/models/core/multimodal/qwen_image_bench_eval.py \
    --model_path /path/to/Qwen-Image-Bench \
    --prompt "A cute cat playing piano" \
    --image_path examples/visual_gen/cat_piano.png \
    --output_path qwen_image_bench_result.json \
    --include_raw_outputs
```

The script evaluates all five Qwen-Image-Bench level-1 dimensions by default:
Quality, Aesthetics, Alignment, Real-world Fidelity, and Creative Generation.
